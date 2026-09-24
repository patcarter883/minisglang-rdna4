"""DFlash / DSpark drafter attention: torch (old) vs attn_prefill_paged HIP kernel (new).

One process, one leased card, REAL drafter weights. Three things are measured:

  1. PARITY. How close the HIP path is to the torch path it replaced, and to an fp32 attention
     reference. A drafted token only changes ACCEPTANCE (the target verifies every draft), so the
     contract is closeness, not bit-identity: max|delta| on the attention output and the block
     logits, and the argmax agreement rate of the drafted tokens. The OLD bf16 torch path is also
     scored against the fp32 reference so the new path's flip rate has a floor to be read against.
  2. DETERMINISM. The new captured propose must be eager == replayed, bit for bit, and replay ==
     replay.
  3. TIMING. Graph replay + CUDA events, arms INTERLEAVED in one process with a CONTROL arm (a second,
     independently captured graph of the NEW path — its spread against NEW is the noise floor).
     Reported for the attention alone (all layers, including each arm's KV staging: the old
     gather+cat vs the new in-place scratch write) and for the whole captured propose step.

Inputs are realistic where they can be: real drafter weights, the real target lm_head / embed table,
real text tokens (the anchor and the embedded context). The target AUX hidden states are synthetic
(embed-derived + noise) because producing real ones needs the target forward; fc + hidden_norm
renormalise them before any attention sees them.

TP=2 is modelled per rank: head counts halved exactly as `_DFlashLayer` shards them, row-parallel
all_reduce replaced by identity (a collective is the same cost in both arms, and is not attention).
Parity runs at TP=1 (whole heads, no collective): attention heads are independent, so per-head
numerics are the same at either TP.

  gpu-lease -n 1 -- bash -c 'docker run --rm ... minisgl-rdna4:specfix-20260924 \
      python /engine/tools/dflash_drafter_attn_bench.py --out /engine/docs/journal/measurements/...'
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import struct
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("ENGINE_PY", "/engine/python"))

import minisgl.distributed.info as _tpi  # noqa: E402


def set_tp_info(rank: int, size: int) -> None:
    """The engine sets TP once per process; this harness models TP=1 and a TP=2 rank in turn."""
    _tpi._TP_INFO = _tpi.DistributedInfo(rank, size)


set_tp_info(0, 1)

HUB = os.path.expanduser("~/.cache/huggingface/hub")
_NO_POS = -(1 << 40)
_KV_SLACK = 128

PAIRS = {
    # serve.sh qwen35b-awq: z-lab drafter, k_dflash=15 (block 16), 5x sliding@4096 + 1x full.
    # full_attention ring cap = min(max_seq_len, 8192) + block (no FULL_CAP override on this arm).
    "qwen36-35b-dflash": dict(draft="z-lab/Qwen3.6-35B-A3B-DFlash",
                              target="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", full_cap=8192, N=(1, 4)),
    # serve.sh qwen38-27b-*: RadixArk DSpark, k=6 (block 7), 5x full_attention, FULL_CAP=2048,
    # MINISGL_SPEC_MAX_BS=1.
    "qwen38-27b-dspark": dict(draft="RadixArk/Qwen3.8-27B-DSpark",
                              target="RedHatAI/Qwen3.8-27B-MXFP4", full_cap=2048, N=(1,)),
}
PREFIXES = (512, 2048, 8192, 32768)


def snap(repo: str) -> str:
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"))
    ref = open(os.path.join(d, "refs", "main")).read().strip()
    return os.path.join(d, "snapshots", ref)


def read_tensor(folder: str, name: str) -> torch.Tensor:
    """One tensor from a (possibly sharded) safetensors checkpoint, via the header only."""
    from safetensors import safe_open
    for f in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        if name in hdr:
            with safe_open(f, "pt") as so:
                return so.get_tensor(name)
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------
# The OLD torch attention, verbatim from models/dflash.py @ rdna4 cb91ee0b (the parent of this
# change). `fp32=True` is the reference: the same formulation with q/K/V upcast to fp32.
# ------------------------------------------------------------------------------------------------
def _attn_torch_eager(layer, q, K, V, attn_mask, fp32):
    group = layer.num_heads // layer.num_kv_heads
    K = K.repeat_interleave(group, dim=1)
    V = V.repeat_interleave(group, dim=1)
    if fp32:
        q, K, V0 = q.float(), K.float(), V
        V = V.float()
    scores = torch.einsum("bhd,shd->bhs", q, K) * layer.scale
    if attn_mask is not None:
        scores = scores + attn_mask.unsqueeze(1)
    probs = scores.softmax(dim=-1).to(V.dtype)
    out = torch.einsum("bhs,shd->bhd", probs, V)
    return out.to(V0.dtype) if fp32 else out


def _attn_torch_batched(layer, q_flat, K, V, attn_mask, N, Q, fp32):
    H, Hkv, hd = layer.num_heads, layer.num_kv_heads, layer.head_dim
    group = H // Hkv
    qg = q_flat.view(N, Q, Hkv, group, hd)
    dt = V.dtype
    if fp32:
        qg, K, V = qg.float(), K.float(), V.float()
    scores = torch.einsum("nqgrd,nsgd->nqgrs", qg, K) * layer.scale
    scores = scores + attn_mask.view(N, Q, 1, 1, -1)
    probs = scores.softmax(dim=-1).to(V.dtype)
    return torch.einsum("nqgrs,nsgd->nqgrd", probs, V).reshape(N * Q, H, hd).to(dt)


def _qkv(layer, x, pos, T):
    H, Hkv, hd = layer.num_heads, layer.num_kv_heads, layer.head_dim
    q = layer.q_proj.forward(x).view(T, H, hd)
    k_noise = layer.k_proj.forward(x).view(T, Hkv, hd)
    v_noise = layer.v_proj.forward(x).view(T, Hkv, hd)
    layer.q_norm.forward_inplace(q)
    layer.k_norm.forward_inplace(k_noise)
    q_flat, kn_flat = layer._rotary.forward(
        pos.reshape(T), q.reshape(T, H * hd).contiguous(), k_noise.reshape(T, Hkv * hd).contiguous())
    return q_flat, kn_flat.view(T, Hkv, hd), v_noise


def _post(layer, x, attn, residual, T):
    H, hd = layer.num_heads, layer.head_dim
    if layer.gated:
        gate = F.softplus(layer.g_proj.forward(x).float()).to(attn.dtype)
        attn = attn * gate.unsqueeze(-1)
    h = residual + layer.o_proj.forward(attn.reshape(T, H * hd))
    return h + layer._mlp(layer.post_attention_layernorm.forward(h))


def old_attend_block(layer, hidden, block_pos, k_ctx, v_ctx, attn_mask, fp32=False):
    B = hidden.shape[0]
    x = layer.input_layernorm.forward(hidden)
    q_flat, k_noise, v_noise = _qkv(layer, x, block_pos, B)
    K = torch.cat([k_ctx, k_noise], dim=0)
    V = torch.cat([v_ctx, v_noise], dim=0)
    attn = _attn_torch_eager(layer, q_flat.view(B, layer.num_heads, layer.head_dim), K, V,
                             attn_mask, fp32)
    return _post(layer, x, attn, hidden, B)


def old_attend_block_batched(layer, hidden, block_pos, k_ctx, v_ctx, attn_mask, fp32=False):
    N, Q = hidden.shape[0], hidden.shape[1]
    T = N * Q
    flat = hidden.reshape(T, -1)
    x = layer.input_layernorm.forward(flat)
    q_flat, k_noise, v_noise = _qkv(layer, x, block_pos, T)
    K = torch.cat([k_ctx, k_noise.view(N, Q, *k_noise.shape[1:])], dim=1)
    V = torch.cat([v_ctx, v_noise.view(N, Q, *v_noise.shape[1:])], dim=1)
    attn = _attn_torch_batched(layer, q_flat, K, V, attn_mask, N, Q, fp32)
    return _post(layer, x, attn, flat, T).view(N, Q, -1)


def old_denoise_batched(model, noise, block_pos, k_pool, v_pool, slots, masks, fp32=False):
    """The pre-change denoise_batched: per-layer gather of the ring row, then cat + torch attention.
    The pools now carry Q scratch columns; the old code never had them, so it gathers [:, :C] only."""
    Q = noise.shape[1]
    hidden = noise
    for l, layer in enumerate(model.layers):
        C = k_pool[l].shape[1] - Q
        hidden = old_attend_block_batched(
            layer, hidden, block_pos, k_pool[l][:, :C][slots], v_pool[l][:, :C][slots], masks[l],
            fp32)
    return model.norm.forward(hidden.reshape(-1, hidden.shape[-1])).view_as(hidden)


def old_denoise_cached(model, noise, prefix_kv, block_pos, fp32=False):
    hidden = noise
    P_full = prefix_kv[0][0].shape[0]
    P = model.window_prefix(P_full)
    if P < P_full:
        d = P_full - P
        prefix_kv = [(k[d:], v[d:]) for (k, v) in prefix_kv]
    masks = model.layer_masks(P, noise.shape[0], noise.device)
    for layer, (k_ctx, v_ctx), mask in zip(model.layers, prefix_kv, masks):
        hidden = old_attend_block(layer, hidden, block_pos, k_ctx, v_ctx, mask, fp32)
    return model.norm.forward(hidden)


# ------------------------------------------------------------------------------------------------
# Drafter construction with REAL weights (the proposer's own loader, on a stub proposer).
# ------------------------------------------------------------------------------------------------
class _Embed:
    def __init__(self, w):
        self.weight = w

    def forward(self, ids):
        return F.embedding(ids, self.weight)


class _Head:
    """Stand-in for the borrowed target lm_head: the REAL weight, plain F.linear to fp32 logits.
    At TP=2 each rank owns vocab/2 rows; the timing arm uses that shard (the all_gather is a
    collective, identical in both arms, and excluded)."""

    def __init__(self, w):
        self.w = w

    def logits_all_rows(self, x):
        return F.linear(x, self.w).float()


class _NoComm:
    def all_reduce(self, y):
        return y


def build(pair: dict, tp: int, device, dtype, head_rows: int | None = None):
    from minisgl.models.dflash import DFlashDraftModel
    from minisgl.spec.dflash import DFlashProposer, dflash_block_size, dflash_layer_masks

    set_tp_info(0, tp)
    folder = snap(pair["draft"])
    raw = json.load(open(os.path.join(folder, "config.json")))
    hf = SimpleNamespace(**raw)
    dfc = raw.get("dflash_config") or {}
    L = raw["num_hidden_layers"]
    causal, window = dflash_layer_masks(hf, L)
    rp = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    theta = float(rp.get("rope_theta") or raw.get("rope_theta") or 1e6)
    rs = None
    if rp and str(rp.get("rope_type", "default")) not in ("default", "None"):
        rs = tuple(sorted((k, v) for k, v in rp.items() if isinstance(v, (str, int, float, bool))))
    block, _ = dflash_block_size(hf, 15)
    prev_dt = torch.get_default_dtype()
    torch.set_default_dtype(dtype)   # as the engine does: no fp32 staging of the full drafter
    with torch.device(device):
        model = DFlashDraftModel(
            hidden_size=raw["hidden_size"], intermediate_size=raw["intermediate_size"],
            num_layers=L, num_heads=raw["num_attention_heads"],
            num_kv_heads=raw["num_key_value_heads"], head_dim=raw["head_dim"],
            num_aux_layers=len(dfc["target_layer_ids"]), rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=theta, rope_scaling=rs, max_position=raw.get("max_position_embeddings", 262144),
            layer_causal=causal, layer_window=window)
    torch.set_default_dtype(prev_dt)
    stub = DFlashProposer.__new__(DFlashProposer)
    stub._draft, stub._dtype, stub._device, stub._compressed = model, dtype, device, False
    os.environ.pop("MINISGL_DFLASH_QUANT", None)   # bf16 drafter: attention is what is measured
    stub._load_draft_weights(folder)
    if tp > 1:
        for layer in model.layers:
            for lin in (layer.o_proj, layer.down_proj):
                if lin._comm is not None:
                    lin._comm = _NoComm()
    tfolder = snap(pair["target"])
    emb = read_tensor(tfolder, "model.language_model.embed_tokens.weight").to(device, dtype)
    head = read_tensor(tfolder, "lm_head.weight")
    if head_rows is not None:
        head = head[:head_rows]
    model.bind_embed(_Embed(emb))
    model.bind_lm_head(_Head(head.to(device, dtype)))
    set_tp_info(0, 1)
    return model, dict(block=block, mask_id=int(dfc.get("mask_token_id") or raw.get("mask_token_id")),
                       causal=causal, window=window, raw=raw)


def text_tokens(pair: dict, n: int) -> torch.Tensor:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(snap(pair["target"]), "tokenizer.json"))
    src = []
    for pat in ("/engine/docs/journal/*.md", "/engine/python/minisgl/spec/*.py",
                "/engine/python/minisgl/models/*.py"):
        for f in sorted(glob.glob(pat)):
            src.append(open(f, errors="ignore").read())
    ids = tok.encode("\n".join(src)).ids
    while len(ids) < n:
        ids = ids + ids
    return torch.tensor(ids[:n], dtype=torch.int64)


def make_aux(model, toks, gen, device, dtype):
    """Synthetic target aux [P, n_aux, hidden]: the context tokens' embedding rows at a per-layer
    scale, plus noise — structured by real text, renormalised by fc + hidden_norm downstream.
    Built in row chunks straight into the activation dtype (a 32k-row fp32 aux is 2 GiB)."""
    n_aux = model.num_aux_layers
    P = toks.shape[0]
    H = model.hidden_size
    out = torch.empty((P, n_aux, H), device=device, dtype=dtype)
    sc = torch.linspace(4.0, 40.0, n_aux, device=device).view(1, n_aux, 1)
    for lo in range(0, P, 2048):
        hi = min(P, lo + 2048)
        e = model._embed.forward(toks[lo:hi].to(device)).float()
        out[lo:hi] = (e.unsqueeze(1) * sc + torch.randn(
            (hi - lo, n_aux, H), generator=gen, device=device) * 0.5).to(dtype)
    return out


# ------------------------------------------------------------------------------------------------
# The captured propose body, mirrored from spec/dflash.py propose_body (ring projection -> per-layer
# absolute-position mask -> batched denoise -> head -> argmax / DSpark Markov walk).
# ------------------------------------------------------------------------------------------------
class Ring:
    def __init__(self, model, meta, pair, N, device, dtype):
        L = model.layers
        self.Q = meta["block"]
        self.A = self.Q
        self.win = meta["window"]
        self.cau = meta["causal"]
        self.cap = [(w + _KV_SLACK) if w > 0 else (pair["full_cap"] + self.A) for w in self.win]
        Hkv, hd = L[0].num_kv_heads, L[0].head_dim
        S = N + 1                                     # N slots + NULL
        self.pk = [torch.zeros(S, c + self.Q, Hkv, hd, device=device, dtype=dtype) for c in self.cap]
        self.pv = [torch.zeros(S, c + self.Q, Hkv, hd, device=device, dtype=dtype) for c in self.cap]
        self.ppos = {c: torch.full((S, c), _NO_POS, dtype=torch.int64, device=device)
                     for c in sorted(set(self.cap))}
        self.N = N
        self.slots = torch.arange(N, device=device, dtype=torch.int64)
        self.blk_tri = torch.where(
            torch.arange(self.Q, device=device).view(-1, 1) >= torch.arange(self.Q, device=device),
            0.0, float("-inf")).float()
        self.blk_open = torch.zeros(self.Q, self.Q, device=device)
        self.model = model
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def fill(self, slot, aux, end):
        """Cold rebuild: newest min(P, cap) committed rows into each ring (chunked like the proposer)."""
        P = aux.shape[0]
        for c in sorted(set(self.cap)):
            ids = [l for l, cc in enumerate(self.cap) if cc == c]
            mc = min(P, c)
            self.ppos[c][slot].fill_(_NO_POS)
            for lo in range(0, mc, 2048):
                hi = min(lo + 2048, mc)
                rows = aux[P - mc + lo: P - mc + hi]
                pos = torch.arange(end - mc + lo, end - mc + hi, dtype=torch.int64, device=self.device)
                col = pos % c
                ws = torch.full((hi - lo,), slot, dtype=torch.int64, device=self.device)
                self.model.project_prefix_into(rows, pos.to(torch.int32), self.pk, self.pv, ws, col,
                                               layer_ids=ids)
                self.ppos[c][slot, col] = pos

    def stage(self, aux_tails, ends, anchors, mask_id):
        """Static propose inputs (what stage_propose H2Ds): the newest A aux rows per request."""
        N, A, Q = self.N, self.A, self.Q
        self.g_aux = torch.stack(aux_tails)                              # [N, A, n_aux, hidden]
        e = torch.tensor(ends, device=self.device).view(N, 1)
        self.g_pos = e - A + torch.arange(A, device=self.device).view(1, A)   # [N, A] abs pos
        blk = torch.full((N, Q), mask_id, dtype=torch.int64, device=self.device)
        blk[:, 0] = torch.tensor(anchors, device=self.device)
        self.g_blk_ids = blk
        self.g_blk_pos = e + torch.arange(Q, device=self.device).view(1, Q)   # [N, Q]

    def masks(self):
        N, Q = self.N, self.Q
        qa = self.g_blk_pos.unsqueeze(2)
        cache, out = {}, []
        for l in range(len(self.model.layers)):
            key = (self.cau[l], self.win[l], self.cap[l])
            if key not in cache:
                c_, w_, cap = key
                pa = self.ppos[cap][self.slots].unsqueeze(1)
                keep = (pa != _NO_POS).expand(N, Q, cap)
                if c_:
                    keep = keep & (pa <= qa)
                if w_ > 0:
                    keep = keep & ((qa - pa) < w_)
                blk = self.blk_tri if c_ else self.blk_open
                cache[key] = torch.cat([torch.where(keep, 0.0, float("-inf")).float(),
                                        blk.expand(N, Q, Q)], dim=2)
            out.append(cache[key])
        return out

    def body(self, arm: str):
        """One whole propose step. arm: 'new' (shipped denoise_batched) | 'old' | 'ref' (old, fp32)."""
        m = self.model
        N, A, Q = self.N, self.A, self.Q
        ws = self.slots.view(N, 1).expand(N, A).reshape(-1)
        wp = self.g_pos.reshape(-1)
        cols = [torch.remainder(wp, c) for c in self.cap]
        m.project_prefix_into(self.g_aux.reshape(N * A, *self.g_aux.shape[2:]),
                              wp.to(torch.int32), self.pk, self.pv, ws, cols)
        for c, pp in self.ppos.items():
            pp[ws, torch.remainder(wp, c)] = wp
        masks = self.masks()
        noise = m.embed(self.g_blk_ids.reshape(-1)).to(self.dtype).view(N, Q, -1)
        bp = self.g_blk_pos.to(torch.int32)
        if arm == "new":
            hidden = m.denoise_batched(noise, bp, self.pk, self.pv, self.slots, masks)
        else:
            hidden = old_denoise_batched(m, noise, bp, self.pk, self.pv, self.slots, masks,
                                         fp32=(arm == "ref"))
        rows = hidden[:, :-1] if m.has_markov else hidden[:, 1:]
        logits = m.head(rows.reshape(N * (Q - 1), -1))
        if m.has_markov:
            ids = m.markov_block_argmax(logits.view(N, Q - 1, -1), self.g_blk_ids[:, 0]).reshape(-1)
        else:
            ids = logits.argmax(-1)
        conf = None
        if m.has_confidence:
            draft = ids.view(N, Q - 1)
            prev = torch.cat([self.g_blk_ids[:, :1], draft[:, :-1]], dim=1)
            conf = m.confidence(rows, prev)
        return hidden, logits, ids.view(N, Q - 1), conf

    def attn_only(self, arm: str, qkv, masks):
        """Attention alone, all layers, INCLUDING each arm's KV staging: old = gather ring row + cat
        block + grouped einsum/softmax/einsum; new = block K/V into the scratch cols + the kernel
        (and its once-per-forward mask fold + page metadata, which the shipped forward also pays)."""
        from minisgl.models.dflash import drafter_attend, drafter_attn_meta
        N, Q = self.N, self.Q
        outs = []
        metas = {}
        if arm == "new":
            masks = self.model._fold_masks(masks)
        for l, layer in enumerate(self.model.layers):
            q_flat, kn, vn = qkv[l]
            C = self.cap[l]
            if arm == "new":
                pl = C + Q
                if pl not in metas:
                    metas[pl] = drafter_attn_meta(N, Q, pl, self.device, self.model._group,
                                                  pages=self.slots)
                self.pk[l][self.slots, C:] = kn.view(N, Q, *kn.shape[1:])
                self.pv[l][self.slots, C:] = vn.view(N, Q, *vn.shape[1:])
                outs.append(drafter_attend(q_flat.view(N * Q, layer.num_heads, layer.head_dim),
                                           self.pk[l], self.pv[l], metas[pl], layer.scale,
                                           masks[l]))
            else:
                K = torch.cat([self.pk[l][:, :C][self.slots], kn.view(N, Q, *kn.shape[1:])], dim=1)
                V = torch.cat([self.pv[l][:, :C][self.slots], vn.view(N, Q, *vn.shape[1:])], dim=1)
                outs.append(_attn_torch_batched(layer, q_flat, K, V, masks[l], N, Q, arm == "ref"))
        return outs


# ------------------------------------------------------------------------------------------------
# Timing: graph replay, CUDA events, interleaved arms, a control arm.
# ------------------------------------------------------------------------------------------------
@torch.inference_mode()
def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g, out


def time_arms(graphs: dict, reps=15, iters=40):
    """Median ms per replay per arm, arms interleaved in a shuffled order every rep."""
    res = {k: [] for k in graphs}
    order = list(graphs)
    rng = random.Random(0)
    for g in graphs.values():
        for _ in range(5):
            g.replay()
    torch.cuda.synchronize()
    for _ in range(reps):
        rng.shuffle(order)
        for k in order:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(iters):
                graphs[k].replay()
            b.record()
            torch.cuda.synchronize()
            res[k].append(a.elapsed_time(b) / iters)
    return {k: statistics.median(v) for k, v in res.items()}


# ------------------------------------------------------------------------------------------------
def parity_and_determinism(name, pair, device, dtype, log):
    """TP=1, whole heads, real weights: captured-path parity + determinism, eager-path parity."""
    model, meta = build(pair, 1, device, dtype)
    Q = meta["block"]
    gen = torch.Generator(device=device).manual_seed(1234)
    toks = text_tokens(pair, max(PREFIXES) + 8192)
    tot = {"new": [0, 0], "old": [0, 0]}
    log(f"\n## {name} — PARITY (TP=1, real weights, block {Q}; vs fp32-attention reference)")
    log(f"{'path':>8} {'P':>6} {'N':>2} {'attn|d| new':>12} {'attn|d| old':>12} "
        f"{'logit|d| new':>13} {'logit|d| old':>13} {'agree new':>10} {'agree old':>10} "
        f"{'new==old':>9}")
    with torch.inference_mode():
        for P in PREFIXES:
            for N in (1, 4):
                ring = Ring(model, meta, pair, N, device, dtype)
                tails, ends, anchors = [], [], []
                for i in range(N):
                    off = (i * 977 + P * 3) % 4096
                    t = toks[off: off + P + 1]
                    aux = make_aux(model, t[:P], gen, device, dtype)
                    ring.fill(i, aux, P)
                    tails.append(aux[P - ring.A:].clone())
                    ends.append(P)
                    anchors.append(int(t[P - 1]))
                ring.stage(tails, ends, anchors, meta["mask_id"])
                # attention-only on layer inputs taken from the new path's own q/k/v (same inputs
                # for every arm), all layers
                masks = ring.masks()
                x0 = model.embed(ring.g_blk_ids.reshape(-1)).to(dtype)
                qkv = []
                for layer in model.layers:
                    x = layer.input_layernorm.forward(x0)
                    qkv.append(_qkv(layer, x, ring.g_blk_pos.to(torch.int32), N * Q))
                a_new = ring.attn_only("new", qkv, masks)
                a_old = ring.attn_only("old", qkv, masks)
                a_ref = ring.attn_only("ref", qkv, masks)
                da_new = max((a - r).float().abs().max().item() for a, r in zip(a_new, a_ref))
                da_old = max((a - r).float().abs().max().item() for a, r in zip(a_old, a_ref))
                _, lg_new, id_new, _ = ring.body("new")
                _, lg_old, id_old, _ = ring.body("old")
                _, lg_ref, id_ref, _ = ring.body("ref")
                n = id_ref.numel()
                ag_new = int((id_new == id_ref).sum())
                ag_old = int((id_old == id_ref).sum())
                tot["new"][0] += ag_new; tot["new"][1] += n
                tot["old"][0] += ag_old; tot["old"][1] += n
                log(f"{'ring':>8} {P:>6} {N:>2} {da_new:>12.3e} {da_old:>12.3e} "
                    f"{(lg_new - lg_ref).abs().max().item():>13.3e} "
                    f"{(lg_old - lg_ref).abs().max().item():>13.3e} "
                    f"{ag_new:>5}/{n:<4} {ag_old:>5}/{n:<4} {int((id_new == id_old).sum()):>4}/{n}")
                del ring, aux, a_new, a_old, a_ref, qkv, masks
                torch.cuda.empty_cache()
            # EAGER path (denoise_cached) at the same prefix, one request, several blocks.
            ag = [0, 0, 0]
            dl = [0.0, 0.0]
            for b in range(4):
                off = (b * 1531) % 4096
                t = toks[off: off + P + 1]
                aux = make_aux(model, t[:P], gen, device, dtype)
                ctx_pos = torch.arange(P, dtype=torch.int32, device=device)
                pkv = model.project_prefix(aux, ctx_pos)
                ids = torch.full((Q,), meta["mask_id"], dtype=torch.int64, device=device)
                ids[0] = int(t[P - 1])
                noise = model.embed(ids).to(dtype)
                bp = torch.arange(P, P + Q, dtype=torch.int32, device=device)
                outs = {}
                for arm in ("new", "old", "ref"):
                    h = (model.denoise_cached(noise, pkv, bp) if arm == "new"
                         else old_denoise_cached(model, noise, pkv, bp, fp32=(arm == "ref")))
                    rows = h[:-1] if model.has_markov else h[1:]
                    lg = model.head(rows)
                    am = (model.markov_block_argmax(lg, ids[0]) if model.has_markov
                          else lg.argmax(-1))
                    outs[arm] = (lg, am)
                n = outs["ref"][1].numel()
                ag[0] += int((outs["new"][1] == outs["ref"][1]).sum())
                ag[1] += int((outs["old"][1] == outs["ref"][1]).sum())
                ag[2] += n
                dl[0] = max(dl[0], (outs["new"][0] - outs["ref"][0]).abs().max().item())
                dl[1] = max(dl[1], (outs["old"][0] - outs["ref"][0]).abs().max().item())
                del pkv
            tot["new"][0] += ag[0]; tot["new"][1] += ag[2]
            tot["old"][0] += ag[1]; tot["old"][1] += ag[2]
            log(f"{'eager':>8} {P:>6} {1:>2} {'':>12} {'':>12} {dl[0]:>13.3e} {dl[1]:>13.3e} "
                f"{ag[0]:>5}/{ag[2]:<4} {ag[1]:>5}/{ag[2]:<4}")
            torch.cuda.empty_cache()
    log(f"drafted-token argmax agreement with the fp32-attention reference: "
        f"NEW {tot['new'][0]}/{tot['new'][1]} ({100 * tot['new'][0] / tot['new'][1]:.2f}%)  "
        f"OLD torch bf16 {tot['old'][0]}/{tot['old'][1]} "
        f"({100 * tot['old'][0] / tot['old'][1]:.2f}%)")

    # DETERMINISM: new captured propose, eager vs replay vs replay.
    ok = True
    for N in (1, 4):
        with torch.inference_mode():
            ring = Ring(model, meta, pair, N, device, dtype)
            tails, ends, anchors = [], [], []
            for i in range(N):
                P = 3000 + 1000 * i
                t = toks[i * 100: i * 100 + P + 1]
                aux = make_aux(model, t[:P], gen, device, dtype)
                ring.fill(i, aux, P)
                tails.append(aux[P - ring.A:].clone()); ends.append(P); anchors.append(int(t[P - 1]))
            ring.stage(tails, ends, anchors, meta["mask_id"])
            h_e, lg_e, id_e, _ = ring.body("new")
            h_e, lg_e, id_e = h_e.clone(), lg_e.clone(), id_e.clone()
        g, (h_g, lg_g, id_g, _) = capture(lambda: ring.body("new"))
        g.replay(); torch.cuda.synchronize()
        r1 = (h_g.clone(), lg_g.clone(), id_g.clone())
        g.replay(); torch.cuda.synchronize()
        eq_eg = torch.equal(h_e, r1[0]) and torch.equal(lg_e, r1[1]) and torch.equal(id_e, r1[2])
        eq_gg = torch.equal(r1[0], h_g) and torch.equal(r1[1], lg_g) and torch.equal(r1[2], id_g)
        log(f"DETERMINISM N={N}: eager==replay {eq_eg}  replay==replay {eq_gg}")
        ok = ok and eq_eg and eq_gg
        del g, ring
    del model
    torch.cuda.empty_cache()
    return ok


def timing(name, pair, device, dtype, log):
    """TP=2 per-rank shapes: attention-only and whole-propose, old vs new vs control."""
    raw = json.load(open(os.path.join(snap(pair["draft"]), "config.json")))
    model, meta = build(pair, 2, device, dtype, head_rows=raw["vocab_size"] // 2)
    L0 = model.layers[0]
    Q = meta["block"]
    gen = torch.Generator(device=device).manual_seed(99)
    toks = text_tokens(pair, max(PREFIXES) + 8)
    caps = sorted(set(zip(meta["causal"], meta["window"],
                          [(w + _KV_SLACK) if w > 0 else pair["full_cap"] + Q for w in meta["window"]])))
    log(f"\n## {name} — TIMING (TP=2 per rank: H={L0.num_heads} Hkv={L0.num_kv_heads} "
        f"hd={L0.head_dim}, block {Q}, {len(model.layers)} layers, rings (causal,window,cap)={caps})")
    log("graph replay, CUDA events, median of 15 interleaved reps x 40 replays; ms per step")
    log(f"{'N':>2} {'P':>6} | {'attn old':>9} {'attn new':>9} {'attn ctl':>9} {'speedup':>8} | "
        f"{'prop old':>9} {'prop new':>9} {'prop ctl':>9} {'speedup':>8} | {'ctl spread':>10}")
    rows = []
    for N in pair["N"]:
        for P in PREFIXES:
            with torch.inference_mode():
                ring = Ring(model, meta, pair, N, device, dtype)
                tails, ends, anchors = [], [], []
                for i in range(N):
                    aux = make_aux(model, toks[:P], gen, device, dtype)
                    ring.fill(i, aux, P)
                    tails.append(aux[P - ring.A:].clone()); ends.append(P); anchors.append(int(toks[P - 1]))
                ring.stage(tails, ends, anchors, meta["mask_id"])
                masks = ring.masks()
                x0 = model.embed(ring.g_blk_ids.reshape(-1)).to(dtype)
                qkv = [_qkv(l, l.input_layernorm.forward(x0), ring.g_blk_pos.to(torch.int32), N * Q)
                       for l in model.layers]
            graphs = {}
            for arm, key in (("old", "a_old"), ("new", "a_new"), ("new", "a_ctl")):
                graphs[key], _ = capture(lambda arm=arm: ring.attn_only(arm, qkv, masks))
            for arm, key in (("old", "p_old"), ("new", "p_new"), ("new", "p_ctl")):
                graphs[key], _ = capture(lambda arm=arm: ring.body(arm))
            t = time_arms(graphs)
            spread = max(abs(t["a_ctl"] - t["a_new"]) / t["a_new"],
                         abs(t["p_ctl"] - t["p_new"]) / t["p_new"]) * 100
            log(f"{N:>2} {P:>6} | {t['a_old']:>9.4f} {t['a_new']:>9.4f} {t['a_ctl']:>9.4f} "
                f"{t['a_old'] / t['a_new']:>7.2f}x | {t['p_old']:>9.4f} {t['p_new']:>9.4f} "
                f"{t['p_ctl']:>9.4f} {t['p_old'] / t['p_new']:>7.2f}x | {spread:>9.2f}%")
            rows.append(dict(pair=name, N=N, P=P, **{k: round(v, 5) for k, v in t.items()}))
            del graphs, ring
            torch.cuda.empty_cache()
    # EAGER fallback path (denoise_cached) at the same prefixes, graph-captured for timing only.
    log(f"eager-fallback denoise_cached (N=1; S = window_prefix(P)+block keys per layer):")
    log(f"{'P':>6} | {'old':>9} {'new':>9} {'ctl':>9} {'speedup':>8}")
    for P in PREFIXES:
        with torch.inference_mode():
            aux = make_aux(model, toks[:P], gen, device, dtype)
            pkv = model.project_prefix(aux, torch.arange(P, dtype=torch.int32, device=device))
            ids = torch.full((Q,), meta["mask_id"], dtype=torch.int64, device=device)
            noise = model.embed(ids).to(dtype)
            bp = torch.arange(P, P + Q, dtype=torch.int32, device=device)
        graphs = {
            "old": capture(lambda: old_denoise_cached(model, noise, pkv, bp))[0],
            "new": capture(lambda: model.denoise_cached(noise, pkv, bp))[0],
            "ctl": capture(lambda: model.denoise_cached(noise, pkv, bp))[0],
        }
        t = time_arms(graphs, reps=9, iters=20)
        log(f"{P:>6} | {t['old']:>9.4f} {t['new']:>9.4f} {t['ctl']:>9.4f} "
            f"{t['old'] / t['new']:>7.2f}x")
        rows.append(dict(pair=name, path="eager", P=P, **{k: round(v, 5) for k, v in t.items()}))
        del graphs, pkv
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------------------------------------------------
# CCA drafter (ZAYA DFlashCCADraftModel). No trained checkpoint is on this box, so random weights at
# the ZAYA geometry (hidden 2048, head_dim 128, 8 q / 2 k heads); the attention it runs is the block's
# own [seed | block] bidirectional softmax, which the old code did in fp32 torch.
# ------------------------------------------------------------------------------------------------
def _old_cca_forward(layer, x, seed):
    """models/dflash_cca.py _CCADrafterLayer.forward @ cb91ee0b, verbatim (fp32 torch attention)."""
    from minisgl.models.dflash_cca import _rmsnorm_heads
    S, T, H = x.shape
    residual = x
    h = layer.input_layernorm.forward(x)
    if seed is not None:
        s_ = layer.input_layernorm.forward(seed).unsqueeze(1)
        hin = torch.cat([s_, h], dim=1)
        off = 1
    else:
        hin, off = h, 0
    L = hin.shape[1]
    q = layer.linear_q.forward(hin)
    k = layer.linear_k.forward(hin)
    v = layer.val_proj.forward(hin)
    qk = torch.cat([q, k], dim=-1).transpose(1, 2)
    qk = F.conv1d(qk, layer.conv_qk_weight, layer.conv_qk_bias, padding=layer._conv_pad,
                  groups=layer.latent_q + layer.latent_k)[..., :L].transpose(1, 2)
    q = qk[..., :layer.latent_q]
    k = qk[..., layer.latent_q:]
    q = _rmsnorm_heads(q, layer.num_q_heads, layer.head_dim, layer.sqrt_head_dim)
    k = _rmsnorm_heads(k, layer.num_k_heads, layer.head_dim, layer.sqrt_head_dim)
    temp = layer._temp_eff().view(1, 1, layer.num_k_heads, 1)
    qh = q.view(S, L, layer.num_q_heads, layer.head_dim).float()
    kh = k.view(S, L, layer.num_k_heads, layer.head_dim).float() * temp
    vh = v.view(S, L, layer.num_k_heads, layer.head_dim).float()
    kh = kh.repeat_interleave(layer.gqa_groups, dim=2)
    vh = vh.repeat_interleave(layer.gqa_groups, dim=2)
    attn = torch.einsum("slhd,smhd->shlm", qh, kh) / layer.sqrt_head_dim
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum("shlm,smhd->slhd", attn, vh).reshape(S, L, layer.latent_q).to(x.dtype)
    out = layer.o_proj.forward(out)[:, off:, :]
    h = residual + out
    h2 = layer.post_attention_layernorm.forward(h)
    gated = F.silu(layer.gate_proj.forward(h2)) * layer.up_proj.forward(h2)
    return h + layer.down_proj.forward(gated)


def cca_section(device, dtype, log):
    from minisgl.models.dflash_cca import DFlashCCADraftModel
    set_tp_info(0, 1)
    Hd, I, NL, QH, KH, HD, V = 2048, 5632, 2, 8, 2, 128, 32768
    gen = torch.Generator(device=device).manual_seed(7)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    with torch.device(device):
        m = DFlashCCADraftModel(Hd, I, NL, QH, KH, HD, 1, Hd, 3, 1e-6, True, "attn")
    torch.set_default_dtype(prev)

    def r(shape, sc):
        return (torch.randn(shape, generator=gen, device=device) * sc).to(dtype)
    for lay in m.layers:
        lay.input_layernorm.weight = r((Hd,), 0.1) + 1
        lay.post_attention_layernorm.weight = r((Hd,), 0.1) + 1
        for lin, (o, i) in ((lay.linear_q, (QH * HD, Hd)), (lay.linear_k, (KH * HD, Hd)),
                            (lay.val_proj, (KH * HD, Hd)), (lay.o_proj, (Hd, QH * HD)),
                            (lay.gate_proj, (I, Hd)), (lay.up_proj, (I, Hd)),
                            (lay.down_proj, (Hd, I))):
            lin.weight = r((o, i), i ** -0.5)
        lay.conv_qk_weight = r(lay.conv_qk_weight.shape, 0.5)
        lay.conv_qk_bias = r(lay.conv_qk_bias.shape, 0.02)
        lay.temp = r((KH,), 0.3)
    m.norm.weight = r((Hd,), 0.1) + 1
    head = r((V, Hd), Hd ** -0.5)
    log(f"\n## ZAYA CCA drafter (random weights; hidden {Hd}, {QH}q/{KH}k heads, hd {HD}, "
        f"{NL} layers) — attention fp32 torch (old) vs HIP (new)")
    log(f"{'S':>3} {'L':>3} | {'hidden|d| new-vs-old':>20} {'agree':>9} | "
        f"{'old ms':>8} {'new ms':>8} {'ctl ms':>8} {'speedup':>8}")
    for S_, L_ in ((1, 16), (4, 16), (1, 7)):
        with torch.inference_mode():
            x = r((S_, L_ - 1, Hd), 1.0)
            seed = r((S_, Hd), 1.0)
            h_new = m.denoise(x, seed)

            def old():
                y = x
                for lay in m.layers:
                    y = _old_cca_forward(lay, y, seed)
                return m.norm.forward(y)
            h_old = old()
            a_new = (h_new.float() @ head.float().T).argmax(-1)
            a_old = (h_old.float() @ head.float().T).argmax(-1)
            d = (h_new - h_old).abs().max().item()
        gs = {"old": capture(old)[0], "new": capture(lambda: m.denoise(x, seed))[0],
              "ctl": capture(lambda: m.denoise(x, seed))[0]}
        t = time_arms(gs)
        n = a_new.numel()
        log(f"{S_:>3} {L_:>3} | {d:>20.3e} {int((a_new == a_old).sum()):>4}/{n:<4} | "
            f"{t['old']:>8.4f} {t['new']:>8.4f} {t['ctl']:>8.4f} {t['old'] / t['new']:>7.2f}x")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--skip-timing", action="store_true")
    a = ap.parse_args()
    device, dtype = torch.device("cuda"), torch.bfloat16
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    props = torch.cuda.get_device_properties(0)
    log(f"# dflash drafter attention: torch vs attn_prefill_paged — {props.name}, "
        f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')} "
        f"(lease card {os.environ.get('LEASE_ROCR_DEVICES', '?')})")
    ok, rows = True, []
    for name in a.pairs.split(","):
        if name == "cca":
            continue
        pair = PAIRS[name]
        if not a.skip_parity:
            ok = parity_and_determinism(name, pair, device, dtype, log) and ok
        if not a.skip_timing:
            rows += timing(name, pair, device, dtype, log)
    if "cca" in a.pairs.split(",") or a.pairs == ",".join(PAIRS):
        cca_section(device, dtype, log)
    log(f"\nDETERMINISM GATE: {'PASS' if ok else 'FAIL'}")
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
            json.dump(rows, f, indent=1)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
