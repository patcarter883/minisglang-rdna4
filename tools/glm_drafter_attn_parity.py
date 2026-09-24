#!/usr/bin/env python
"""GLM drafters on the HIP attention kernels: parity vs the old torch attention, eager == graph
replay, and old-vs-new propose timing — ONE process, real checkpoint weights, served shapes.

Covers the two GLM-4.7-Flash drafters:
  * the MTP head (QuantTrio/GLM-4.7-Flash-AWQ layer 47): GLMMTPAttention now stores the LATENT
    [c_KV | k_rope] per token in the draft ring and runs mla_hip.mla_decode (absorbed MLA) over it;
    the old path materialized per-head K/V, gathered k_buf[slot_rows] (the WHOLE ring) and ran torch
    einsum + softmax with an additive -inf mask over all R columns.
  * the EAGLE3 drafter (thoughtworks/GLM-4.7-Flash-Eagle3): GQA 16/4 x 128 now on
    attn_decode.flash_decode_paged over the ring in place; the old path was the same gather + masked
    torch einsum.

The OLD arm is the deleted code, transcribed verbatim below (`old_*`), running on the SAME module
weights. The CONTRACT (the draft only affects acceptance — the target verifies every token):
  * new vs old numerically close: max|delta| of the attention output and the logits, and drafted-token
    (argmax) agreement, TEACHER-FORCED (both arms fed identical inputs every step) and FREE-RUNNING
    (each arm chains its own drafts from the same state);
  * new path eager == CUDA-graph replay, bit for bit.

Shapes: MTP head at TP=2 PER-RANK attention shapes (10 of 20 heads — rank 0's slice of q_b/kv_b/o_proj)
run at TP=1; its MoE is the full-width TP=1 expert stack and its lm-head the full vocab (so the
whole-propose absolute includes ~2x a rank's MoE/lm-head; the old->new DELTA is exact). K and the ring
come from tools/serve.sh's glm profile: MTP K=2 (K+1 = 3 head steps/propose), EAGLE3 K=6, ring 512.

Inputs are REALISTIC, not randn: prompt tokens are real text through the GLM tokenizer, and the
per-position hidden states are the drafter's OWN output hiddens, produced by teacher-forcing it over
that text (the MTP/EAGLE3 later draft steps consume exactly such hiddens). The rows cover the ring
cases that matter: a cold prompt (position 0 unseedable), a radix prefix hit (a HOLE [0, origin]
never written), a wrapped long prompt, and stale rejected-draft columns after partial accepts.

Run (GPU, under the lease):
  gpu-lease -n 1 -- bash -c 'docker run --rm ... -v <worktree>:/engine minisgl-rdna4:specfix-20260924 \
      -lc "PYTHONPATH=/engine/python:/opt/kernels python /engine/tools/glm_drafter_attn_parity.py"'
Env: PARTS=mtp,eagle3  TIMING=1  ITERS=300  ROUNDS=5
"""
from __future__ import annotations

import dataclasses
import glob
import os
import sys
import time

import torch

sys.path.insert(0, "/engine/python")

DEV = torch.device("cuda")
DT = torch.float16            # engine dtype for GLM-4.7-Flash-AWQ (config torch_dtype, --dtype auto)
MP = "QuantTrio/GLM-4.7-Flash-AWQ"
EP = "thoughtworks/GLM-4.7-Flash-Eagle3"
ITERS = int(os.environ.get("ITERS", "300"))
ROUNDS = int(os.environ.get("ROUNDS", "5"))
PARTS = os.environ.get("PARTS", "mtp,eagle3").split(",")
TIMING = os.environ.get("TIMING", "1") == "1"
PARITY = os.environ.get("PARITY", "1") == "1"


def init_dist():
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29561")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from minisgl.distributed import set_tp_info
    set_tp_info(rank=0, size=1)
    from minisgl.layers import set_rope_device
    set_rope_device(DEV)


def real_tokens(n: int) -> torch.Tensor:
    """n real token ids: repo docs + source text through the GLM tokenizer."""
    from transformers import AutoTokenizer
    from minisgl.utils import download_hf_weight
    tok = AutoTokenizer.from_pretrained(download_hf_weight(MP))
    text, ids = "", []
    for f in sorted(glob.glob("/engine/docs/*.md")) + sorted(glob.glob("/engine/python/minisgl/spec/*.py")):
        text += open(f, errors="ignore").read() + "\n"
        if len(text) > 12 * n:
            break
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert len(ids) >= n, f"only {len(ids)} tokens of text"
    return torch.tensor(ids[:n], dtype=torch.int64, device=DEV)


# =============================================================================== ring bookkeeping
class Ring:
    """Mirrors the proposer's ring state (spec/mtp.py, spec/draft_model.py): k/v buffers, the absolute
    position per column (-1 empty) and the committed cursor per slot."""

    def __init__(self, slots: int, R: int, dims):
        nk, kd, nv, vd = dims
        self.R = R
        self.k = torch.zeros(slots, R, nk, kd, device=DEV, dtype=DT)
        self.v = torch.zeros(slots, R, nv, vd, device=DEV, dtype=DT)
        self.pos = torch.full((slots, R), -1, dtype=torch.int64, device=DEV)
        self.cur = torch.zeros(slots, dtype=torch.int64, device=DEV)

    def keep(self, slots, q_abs):
        pa = self.pos[slots]
        qa = q_abs.unsqueeze(1)
        return (pa >= 0) & (pa <= qa) & ((qa - pa) < self.R)

    def seed_runs(self, origin: int, end: int):
        """seed_prefill's position range + wrap split: [(p0, p1, col0), ...]."""
        p_lo = max(origin + 1, end - self.R)
        n = end - p_lo
        if n <= 0:
            return []
        c0 = p_lo % self.R
        first = min(n, self.R - c0)
        runs = [(p_lo, p_lo + first, c0)]
        if first < n:
            runs.append((p_lo + first, end, 0))
        return runs

    def mark_seeded(self, slot, origin, end):
        for p0, p1, _ in self.seed_runs(origin, end):
            ap = torch.arange(p0, p1, device=DEV)
            self.pos[slot, torch.remainder(ap, self.R)] = ap
        self.cur[slot] = end


def stats(a: torch.Tensor, b: torch.Tensor):
    d = (a.float() - b.float()).abs()
    rel = d.max().item() / max(b.float().abs().max().item(), 1e-30)
    return d.max().item(), rel


# ============================================================================== timing machinery
def phase(msg: str):
    """Breadcrumb BEFORE every capture/replay window, so a hang names its (case, arm, stage)."""
    print(f"[phase] {msg}", flush=True)


def capture(fn, tag: str = ""):
    phase(f"capture begin {tag}")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # Keep the captured outputs referenced for the graph's whole life: replays write into them,
        # so they must never go back to the pool while the graph can still be replayed.
        g.keep_outputs = fn()
    torch.cuda.synchronize()
    phase(f"capture done  {tag}")
    return g


def time_arms(graphs: dict) -> dict:
    """Interleaved arms, rotated order per round; each window = ITERS replays between two events.
    Returns arm -> list of per-round us/replay."""
    res = {k: [] for k in graphs}
    names = list(graphs)
    for r in range(ROUNDS):
        order = names[r % len(names):] + names[: r % len(names)]
        for k in order:
            g = graphs[k]
            phase(f"replay round {r} arm {k}")
            for _ in range(20):
                g.replay()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            e0.record()
            for _ in range(ITERS):
                g.replay()
            e1.record()
            e1.synchronize()
            res[k].append(e0.elapsed_time(e1) * 1000.0 / ITERS)
    return res


_KEEP: list = []


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def emit(out: list, line: str):
    """Print AND append to $OUT immediately: a process that dies later must not lose its results."""
    print(line, flush=True)
    out.append(line)
    if os.environ.get("OUT"):
        with open(os.environ["OUT"], "a") as f:
            f.write(line + "\n")


def report_timing(tag: str, res: dict, out: list):
    if set(res) != {"old", "new", "new_ctl"}:
        emit(out, f"[time] {tag:<44} " + "  ".join(f"{k} {med(v):9.1f} us" for k, v in res.items()))
        return
    o, n, c = med(res["old"]), med(res["new"]), med(res["new_ctl"])
    ctl = (c - n) / n * 100.0
    line = (f"[time] {tag:<44} old {o:9.1f} us  new {n:8.1f} us  ctl {c:8.1f} us  "
            f"speedup {o / n:6.2f}x  ctl-spread {ctl:+.2f}%  "
            f"(old rounds {min(res['old']):.1f}-{max(res['old']):.1f}, "
            f"new {min(res['new']):.1f}-{max(res['new']):.1f})")
    emit(out, line)


# ================================================================================== GLM MTP head
def old_mtp_attn(a, x, positions, k_buf, v_buf, slot_rows, write_col, mask_bias):
    """The DELETED GLMMTPAttention.forward_draft_masked, verbatim (materialized per-head K/V ring,
    whole-ring gather, torch einsum/softmax with an additive -inf mask)."""
    B = x.shape[0]
    H, nope, rope, vhd = a.num_heads, a.qk_nope, a.qk_rope, a.v_head_dim
    q = a.q_b_proj.forward(a.q_a_layernorm.forward(a.q_a_proj.forward(x)))
    q = q.view(B, H, a.qk_head_dim)
    q_nope, q_rope = q[..., :nope], q[..., nope:]
    kv = a.kv_a_proj_with_mqa.forward(x)
    c_kv = a.kv_a_layernorm.forward(kv[:, : a.kv_lora_rank].contiguous())
    k_rope = kv[:, a.kv_lora_rank :]
    q_rope, k_rope = a.rotary.forward(
        positions, q_rope.reshape(B, H * rope).contiguous(), k_rope.contiguous())
    q_rope = q_rope.view(B, H, rope)
    kvb = a.kv_b_proj.forward(c_kv).view(B, H, nope + vhd)
    k_nope, v = kvb[..., :nope], kvb[..., nope:]
    k_full = torch.cat([k_nope, k_rope.unsqueeze(1).expand(B, H, rope)], dim=-1)
    q_full = torch.cat([q_nope, q_rope], dim=-1)
    k_buf[slot_rows, write_col] = k_full
    v_buf[slot_rows, write_col] = v
    Ks = k_buf[slot_rows]
    Vs = v_buf[slot_rows]
    scores = torch.einsum("bhd,bshd->bhs", q_full, Ks) * a.scale_attn
    scores = scores + mask_bias.view(B, 1, -1)
    probs = scores.softmax(dim=-1).to(Vs.dtype)
    o = torch.einsum("bhs,bshd->bhd", probs, Vs)
    return a.o_proj.forward(o.reshape(B, H * vhd))


def old_mtp_seed(a, x, positions, k_buf, v_buf, slot, start_col):
    """The DELETED GLMMTPAttention.seed_kv_masked, verbatim."""
    S = x.shape[0]
    H, nope, rope, vhd = a.num_heads, a.qk_nope, a.qk_rope, a.v_head_dim
    q = a.q_b_proj.forward(a.q_a_layernorm.forward(a.q_a_proj.forward(x)))
    q = q.view(S, H, a.qk_head_dim)
    q_rope = q[..., nope:]
    kv = a.kv_a_proj_with_mqa.forward(x)
    c_kv = a.kv_a_layernorm.forward(kv[:, : a.kv_lora_rank].contiguous())
    k_rope = kv[:, a.kv_lora_rank :]
    _, k_rope = a.rotary.forward(positions, q_rope.reshape(S, H * rope).contiguous(), k_rope.contiguous())
    kvb = a.kv_b_proj.forward(c_kv).view(S, H, nope + vhd)
    k_nope, v = kvb[..., :nope], kvb[..., nope:]
    k_full = torch.cat([k_nope, k_rope.unsqueeze(1).expand(S, H, rope)], dim=-1)
    k_buf[slot, start_col : start_col + S] = k_full
    v_buf[slot, start_col : start_col + S] = v


def old_mtp_dims(a):
    return a.num_heads, a.qk_head_dim, a.num_heads, a.v_head_dim


def load_mtp_head():
    import safetensors
    from minisgl.models import weight as W
    from minisgl.models.config import ModelConfig
    from minisgl.models.glm4_moe_lite import GLMMTPHead
    from minisgl.utils import cached_load_hf_config
    from minisgl.utils.torch_utils import torch_dtype

    hf = cached_load_hf_config(MP)
    names = W.checkpoint_tensor_names(MP)
    cfg = ModelConfig.from_hf(hf, spec_algorithm="mtp", ckpt_tensor_names=names)
    N = cfg.num_layers
    Hfull = cfg.num_qo_heads
    Hr = Hfull // 2                                   # TP=2 per-rank heads (rank 0's slice)
    expert_quant = cfg.quant
    backbone = dataclasses.replace(cfg, quant=None, num_qo_heads=Hr)
    mtp_quant = expert_quant
    if expert_quant is not None and not expert_quant.is_module_quantized(
            f"model.layers.{N}.mlp.experts.0.gate_proj"):
        mtp_quant = None
    with torch.device("meta"), torch_dtype(DT):
        head = GLMMTPHead(backbone, layer_id=N, expert_quant=mtp_quant)

    real_open = safetensors.safe_open

    class _OnlyMTP:                                   # the loader reads layer N's tensors only
        def __init__(self, *a, **k):
            self._cm = real_open(*a, **k)

        def __enter__(self):
            self._f = self._cm.__enter__()
            return self

        def __exit__(self, *e):
            return self._cm.__exit__(*e)

        def keys(self):
            return [k for k in self._f.keys() if f"layers.{N}." in k]

        def get_tensor(self, k):
            return self._f.get_tensor(k)

    W.safetensors.safe_open = _OnlyMTP
    try:
        sd = {k: W.cast_checkpoint_tensor(k, v, DT) for k, v in W.load_weight(MP, DEV, "mtp")}
    finally:
        W.safetensors.safe_open = real_open
    qk = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    sd = {k.removeprefix("mtp."): v for k, v in sd.items()}
    sd["self_attn.q_b_proj.weight"] = sd["self_attn.q_b_proj.weight"][: Hr * qk].contiguous()
    sd["self_attn.kv_b_proj.weight"] = sd["self_attn.kv_b_proj.weight"][
        : Hr * (cfg.qk_nope_head_dim + cfg.v_head_dim)].contiguous()
    sd["self_attn.o_proj.weight"] = sd["self_attn.o_proj.weight"][:, : Hr * cfg.v_head_dim].contiguous()
    head.load_state_dict(sd)
    head.post_load()
    print(f"[load] GLM MTP head layer {N}: {Hr}/{Hfull} heads (rank-0 slice), "
          f"expert quant={type(mtp_quant).__name__ if mtp_quant else None}", flush=True)
    return head


class MTPArms:
    """Both arms over the same head. 'new' = the shipped head.step_masked on the latent ring + meta;
    'old' = the deleted torch attention on a materialized ring. Step bodies mirror propose_body."""

    def __init__(self, head, R, slots):
        from minisgl.spec.draft_attn import DraftAttnBuilder
        from minisgl.models.utils import norm_then_mlp
        self.h, self.R = head, R
        self._ntm = norm_then_mlp
        self.rings = {"new": Ring(slots, R, head.self_attn.draft_buffer_dims()),
                      "old": Ring(slots, R, old_mtp_dims(head.self_attn))}
        self.builder = DraftAttnBuilder(R, DEV)

    def seed(self, arm, slot, tokens, hiddens, origin, end):
        rg = self.rings[arm]
        for p0, p1, c0 in rg.seed_runs(origin, end):
            pos = torch.arange(p0, p1, device=DEV, dtype=torch.int32)
            if arm == "new":
                self.h.seed_buffered(tokens[p0:p1], hiddens[p0 - 1 : p1 - 1], pos, rg.k, rg.v, slot, c0)
            else:
                with torch.inference_mode():
                    fused = self.h.fuse(self.h.embed(tokens[p0:p1]), hiddens[p0 - 1 : p1 - 1])
                    x = self.h.input_layernorm.forward(fused, None)[0]
                    old_mtp_seed(self.h.self_attn, x, pos, rg.k, rg.v, slot, c0)
        rg.mark_seeded(slot, origin, end)

    def step(self, arm, tok, hid, slots, q_abs, positions, want_attn=False):
        """One head step (== head.step_masked for 'new'); returns (logits, hidden, attn_out|None)."""
        h, rg = self.h, self.rings[arm]
        wc = torch.remainder(q_abs, self.R)
        rg.pos[slots, wc] = q_abs
        keep = rg.keep(slots, q_abs)
        fused = h.fuse(h.embed(tok), hid)
        if arm == "new" and not want_attn:
            lg, hd = h.step_masked(fused, positions, rg.k, rg.v, slots, wc,
                                   self.builder.meta(slots, q_abs, keep))
            return lg, hd, None
        x, res = h.input_layernorm.forward(fused, None)
        if arm == "new":
            a = h.self_attn.forward_draft_masked(x, positions, rg.k, rg.v, slots, wc,
                                                 self.builder.meta(slots, q_abs, keep))
        else:
            mask = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
            a = old_mtp_attn(h.self_attn, x, positions, rg.k, rg.v, slots, wc, mask)
        x, res = self._ntm(h.post_attention_layernorm, h.mlp, a, res, fuse_actquant=h._fuse_actquant)
        hid = x + res
        return h.shared_head.forward(hid), hid, a

    def attn_core(self, arm, q_nope, q_rope, slots, q_abs, keep):
        """The attention CORE only — what the change replaced: old = mask + whole-ring gather + einsum
        + softmax + einsum over the materialized ring; new = DraftAttnMeta + W_UK absorb + mla_decode
        + W_UV absorb over the latent ring. Projections/RoPE/ring write/o_proj excluded (attn-module
        timing has those)."""
        a, rg = self.h.self_attn, self.rings[arm]
        B = q_nope.shape[0]
        if arm == "new":
            meta = self.builder.meta(slots, q_abs, keep)
            qa = torch.einsum("thn,hnl->thl", q_nope, a._w_uk)
            qf = torch.cat([qa, q_rope], dim=-1).contiguous()
            ms, R, _, D = rg.k.shape
            ol = a._mla_decode(qf, rg.k.view(ms * R, 1, D), meta.block_table, meta.ctx_lens,
                               a.scale_attn, 0, 0, a.qk_rope)
            return torch.einsum("thl,hdl->thd", ol, a._w_uv)
        mask = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        Ks, Vs = rg.k[slots], rg.v[slots]
        sc = torch.einsum("bhd,bshd->bhs", q_full, Ks) * a.scale_attn + mask.view(B, 1, -1)
        pr = sc.softmax(dim=-1).to(Vs.dtype)
        return torch.einsum("bhs,bshd->bhd", pr, Vs)

    def attn_only(self, arm, x, slots, q_abs, positions):
        """The attention op alone (ring write + attention + o_proj), for attention-only timing."""
        a, rg = self.h.self_attn, self.rings[arm]
        wc = torch.remainder(q_abs, self.R)
        if arm == "new":
            keep = rg.keep(slots, q_abs)
            return a.forward_draft_masked(x, positions, rg.k, rg.v, slots, wc,
                                          self.builder.meta(slots, q_abs, keep))
        keep = rg.keep(slots, q_abs)
        mask = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
        return old_mtp_attn(a, x, positions, rg.k, rg.v, slots, wc, mask)


# ================================================================================ EAGLE3 drafter
def old_eagle3_step(d, embed_e, hidden, positions, k_buf, v_buf, slot_rows, write_col, mask_bias,
                    attn_only_q=None):
    """The DELETED GLMEagle3DraftModel.step_masked attention core, verbatim (grouped einsum over the
    gathered whole ring + additive -inf mask). Returns (logits, out_hidden, attn_out)."""
    B = embed_e.shape[0]
    H, Hkv, hd = d.num_heads, d.num_kv_heads, d.head_dim
    widened = d._widened(embed_e, hidden)
    q = d.q_proj.forward(widened).view(B, H, hd)
    k = d.k_proj.forward(widened).view(B, Hkv, hd)
    v = d.v_proj.forward(widened).view(B, Hkv, hd)
    q_flat, k_flat = d.rotary.forward(positions, q.reshape(B, H * hd).contiguous(),
                                      k.reshape(B, Hkv * hd).contiguous())
    q = q_flat.view(B, H, hd)
    k = k_flat.view(B, Hkv, hd)
    k_buf[slot_rows, write_col] = k
    v_buf[slot_rows, write_col] = v
    rep = H // Hkv
    qg = q.view(B, Hkv, rep, hd)
    Ks = k_buf[slot_rows]
    Vs = v_buf[slot_rows]
    scores = torch.einsum("bgrd,bsgd->bgrs", qg, Ks) * d.scale
    scores = scores + mask_bias.view(B, 1, 1, -1)
    probs = scores.softmax(dim=-1).to(Vs.dtype)
    attn = torch.einsum("bgrs,bsgd->bgrd", probs, Vs).reshape(B, H * hd)
    attn_out = d.o_proj.forward(attn)
    residual = hidden + attn_out
    normed = d.post_attention_layernorm.forward(residual)
    out_hidden = residual + d._mlp(normed)
    logits = d.lm_head.forward(d.norm.forward(out_hidden))
    return logits, out_hidden, attn_out


def new_eagle3_step_instrumented(d, embed_e, hidden, positions, k_buf, v_buf, slot_rows, write_col, meta):
    """GLMEagle3DraftModel.step_masked with the attention output exposed (same ops, same order)."""
    from minisgl.spec.draft_attn import paged_draft_attention
    B = embed_e.shape[0]
    H, Hkv, hd = d.num_heads, d.num_kv_heads, d.head_dim
    widened = d._widened(embed_e, hidden)
    q = d.q_proj.forward(widened).view(B, H, hd)
    k = d.k_proj.forward(widened).view(B, Hkv, hd)
    v = d.v_proj.forward(widened).view(B, Hkv, hd)
    q_flat, k_flat = d.rotary.forward(positions, q.reshape(B, H * hd).contiguous(),
                                      k.reshape(B, Hkv * hd).contiguous())
    q = q_flat.view(B, H, hd)
    k = k_flat.view(B, Hkv, hd)
    k_buf[slot_rows, write_col] = k
    v_buf[slot_rows, write_col] = v
    attn = paged_draft_attention(q, k_buf, v_buf, meta, d.scale).reshape(B, H * hd)
    attn_out = d.o_proj.forward(attn)
    residual = hidden + attn_out
    normed = d.post_attention_layernorm.forward(residual)
    out_hidden = residual + d._mlp(normed)
    logits = d.lm_head.forward(d.norm.forward(out_hidden))
    return logits, out_hidden, attn_out


def load_eagle3():
    import safetensors.torch as st
    from minisgl.layers import VocabParallelEmbedding
    from minisgl.models.glm_eagle3 import GLMEagle3DraftModel
    from minisgl.utils import cached_load_hf_config, download_hf_weight
    hf = cached_load_hf_config(EP)
    rp = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {}
    with torch.device(DEV):
        d = GLMEagle3DraftModel(
            hidden_size=int(hf.hidden_size), intermediate_size=int(hf.intermediate_size),
            num_heads=int(hf.num_attention_heads), num_kv_heads=int(hf.num_key_value_heads),
            head_dim=int(getattr(hf, "head_dim", 128)), num_aux_layers=3,
            draft_vocab_size=int(hf.draft_vocab_size), target_vocab_size=154880,
            rms_norm_eps=float(hf.rms_norm_eps),
            rope_theta=float(rp.get("rope_theta", getattr(hf, "rope_theta", 1e6))),
            # as the (fixed) proposer sizes it: the table must cover TARGET positions, not the
            # checkpoint's 4096 (see spec/draft_model.py) — timing runs at position 100000.
            max_position=max(int(hf.max_position_embeddings), 131072))
    sd = st.load_file(glob.glob(os.path.join(download_hf_weight(EP), "*.safetensors"))[0], device="cuda")
    for ck, v in sd.items():
        if ck in ("d2t", "t2d"):
            continue
        obj, leaf = ck.removeprefix("midlayer.").removeprefix("self_attn.").removeprefix("mlp.").split(".")
        setattr(getattr(d, obj), leaf, v.to(DT).contiguous())
    d.d2t = sd["d2t"].to(torch.int64)
    # the borrowed target embedding (real weights, full vocab at TP=1)
    import json
    from safetensors import safe_open
    mf = download_hf_weight(MP)
    wm = json.load(open(os.path.join(mf, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(mf, wm["model.embed_tokens.weight"]), "pt", device="cuda") as f:
        ew = f.get_tensor("model.embed_tokens.weight").to(DT)
    with torch.device(DEV):
        emb = VocabParallelEmbedding(num_embeddings=ew.shape[0], embedding_dim=ew.shape[1])
    emb.weight = ew
    d.bind_embed(emb)
    print(f"[load] EAGLE3 drafter {d.num_heads}q/{d.num_kv_heads}kv x {d.head_dim}", flush=True)
    return d


class E3Arms:
    def __init__(self, d, R, slots):
        from minisgl.spec.draft_attn import DraftAttnBuilder
        self.d, self.R = d, R
        dims = d.draft_buffer_dims()
        self.rings = {"new": Ring(slots, R, dims), "old": Ring(slots, R, dims)}
        self.builder = DraftAttnBuilder(R, DEV)
        self.d2t = d.d2t

    def seed(self, arm, slot, tokens, hiddens, origin, end):
        rg = self.rings[arm]
        for p0, p1, c0 in rg.seed_runs(origin, end):
            pos = torch.arange(p0, p1, device=DEV, dtype=torch.int32)
            self.d.seed_buffered(self.d.embed(tokens[p0:p1]), hiddens[p0 - 1 : p1 - 1], pos,
                                 rg.k, rg.v, slot, c0)
        rg.mark_seeded(slot, origin, end)

    def step(self, arm, tok, hid, slots, q_abs, positions, want_attn=False):
        d, rg = self.d, self.rings[arm]
        wc = torch.remainder(q_abs, self.R)
        rg.pos[slots, wc] = q_abs
        keep = rg.keep(slots, q_abs)
        e = d.embed(tok)
        if arm == "new":
            meta = self.builder.meta(slots, q_abs, keep)
            if not want_attn:
                lg, hd = d.step_masked(e, hid, positions, rg.k, rg.v, slots, wc, meta)
                return lg, hd, None
            return new_eagle3_step_instrumented(d, e, hid, positions, rg.k, rg.v, slots, wc, meta)
        mask = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
        return old_eagle3_step(d, e, hid, positions, rg.k, rg.v, slots, wc, mask)

    def attn_core(self, arm, q, slots, q_abs, keep):
        """Attention CORE only: old = mask + whole-ring gather + grouped einsum/softmax/einsum; new =
        DraftAttnMeta + attn_decode.flash_decode_paged over the ring in place."""
        from minisgl.spec.draft_attn import paged_draft_attention
        d, rg = self.d, self.rings[arm]
        B = q.shape[0]
        H, Hkv, hd = d.num_heads, d.num_kv_heads, d.head_dim
        if arm == "new":
            return paged_draft_attention(q, rg.k, rg.v, self.builder.meta(slots, q_abs, keep), d.scale)
        mask = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
        Ks, Vs = rg.k[slots], rg.v[slots]
        sc = torch.einsum("bgrd,bsgd->bgrs", q.view(B, Hkv, H // Hkv, hd), Ks) * d.scale + mask.view(B, 1, 1, -1)
        pr = sc.softmax(dim=-1).to(Vs.dtype)
        return torch.einsum("bgrs,bsgd->bgrd", pr, Vs)

    def attn_only(self, arm, x_e, x_h, slots, q_abs, positions):
        """q/k/v proj + ring write + attention + o_proj (the step minus MLP/head)."""
        from minisgl.spec.draft_attn import paged_draft_attention
        d, rg = self.d, self.rings[arm]
        B = x_e.shape[0]
        H, Hkv, hd = d.num_heads, d.num_kv_heads, d.head_dim
        wc = torch.remainder(q_abs, self.R)
        keep = rg.keep(slots, q_abs)
        widened = d._widened(x_e, x_h)
        q = d.q_proj.forward(widened).view(B, H, hd)
        k = d.k_proj.forward(widened).view(B, Hkv, hd)
        v = d.v_proj.forward(widened).view(B, Hkv, hd)
        qf, kf = d.rotary.forward(positions, q.reshape(B, H * hd).contiguous(), k.reshape(B, Hkv * hd).contiguous())
        q, k = qf.view(B, H, hd), kf.view(B, Hkv, hd)
        rg.k[slots, wc] = k
        rg.v[slots, wc] = v
        if arm == "new":
            attn = paged_draft_attention(q, rg.k, rg.v, self.builder.meta(slots, q_abs, keep), d.scale)
            return d.o_proj.forward(attn.reshape(B, H * hd))
        rep = H // Hkv
        mask = torch.where(keep, 0.0, float("-inf")).to(torch.float32)
        Ks, Vs = rg.k[slots], rg.v[slots]
        sc = torch.einsum("bgrd,bsgd->bgrs", q.view(B, Hkv, rep, hd), Ks) * d.scale + mask.view(B, 1, 1, -1)
        pr = sc.softmax(dim=-1).to(Vs.dtype)
        return d.o_proj.forward(torch.einsum("bgrs,bsgd->bgrd", pr, Vs).reshape(B, H * hd))


# ======================================================================== shared propose drivers
def next_tok(arms, logits):
    t = logits.argmax(dim=-1)
    if isinstance(arms, E3Arms):
        t = t + arms.d2t[t]                    # draft vocab -> target vocab, as the proposer does
    return t


def chain_hiddens(arms, tokens, h0, reset=False):
    """Teacher-force the drafter over the real text on a scratch slot of the NEW arm; returns the
    per-position output hiddens (the realistic `prev_hidden` stream for seeding).

    reset=True feeds the SAME seed feature h0 at every position instead of the previous output: the
    EAGLE3 midlayer's residual IS its hidden input, so a 1600-step self-chain grows the residual
    stream without bound and overflows fp16 — in serving it only ever chains K=6 steps from fc(aux).
    One step from a fixed seed over real tokens + real context is the in-distribution feature."""
    L = tokens.shape[0]
    hs = torch.empty(L, h0.shape[-1], device=DEV, dtype=DT)
    slot = torch.tensor([0], device=DEV)
    rg = arms.rings["new"]
    rg.pos[0].fill_(-1)
    hid = h0.view(1, -1)
    with torch.inference_mode():
        for p in range(L):
            q = torch.tensor([p], device=DEV)
            _, out, _ = arms.step("new", tokens[p : p + 1], hid, slot, q, q.to(torch.int32))
            hs[p] = out[0]
            if not reset:
                hid = out
    rg.pos[0].fill_(-1)
    return hs


def propose(arms, arm, slots, cur, tok, hid, n_iter, forced=None, want_attn=False):
    """n_iter head steps (MTP: K+1, EAGLE3: K). Returns (drafts[list of [B]], logits, hiddens, attns)."""
    out, lgs, hids, ats = [], [], [], []
    base = cur.clone()
    for j in range(n_iter):
        q_abs = cur + j
        lg, hid, a = arms.step(arm, tok, hid, slots, q_abs, (base + j).to(torch.int32), want_attn)
        nt = next_tok(arms, lg)
        out.append(nt)
        lgs.append(lg)
        hids.append(hid)
        ats.append(a)
        tok = forced[j] if forced is not None else nt
    return out, lgs, hids, ats


def parity(arms, tokens, hiddens, n_iter, K, name, log):
    """Mixed-row parity at the served ring: 4 rows = cold short / radix hole / near-full / wrapped,
    3 propose rounds with partial accepts (so stale rejected-draft columns exist)."""
    R = arms.R
    rows = [(1, 0, 40), (2, 200, 300), (3, 0, R - 1), (4, 0, 3 * R + 17)]  # (slot, origin, end)
    for arm in ("new", "old"):
        for slot, origin, end in rows:
            arms.seed(arm, slot, tokens, hiddens, origin, end)
    slots = torch.tensor([r[0] for r in rows], device=DEV)
    worst_a = worst_l = worst_ar = worst_lr = 0.0
    agree_tf = total_tf = 0
    free_agree = free_total = 0
    accepts = [[0, 1, K, 0], [K, 0, 1, 1], [1, K, 0, K]]
    with torch.inference_mode():
        for rnd, acc in enumerate(accepts):
            cur = arms.rings["new"].cur[slots].clone()
            assert torch.equal(cur, arms.rings["old"].cur[slots])
            tok = tokens[cur]                            # the confirmed token at each row's cursor
            hid = hiddens[cur - 1]
            # free-running first (each arm its own chain) on a SNAPSHOT so it does not disturb state
            snap = {a: (arms.rings[a].k.clone(), arms.rings[a].v.clone(), arms.rings[a].pos.clone())
                    for a in ("new", "old")}
            dn, _, _, _ = propose(arms, "new", slots, cur, tok, hid, n_iter)
            do, _, _, _ = propose(arms, "old", slots, cur, tok, hid, n_iter)
            steps = min(K, n_iter)
            for j in range(steps):
                eq = (dn[j] == do[j])
                free_agree += int(eq.sum())
                free_total += eq.numel()
            for a in ("new", "old"):
                arms.rings[a].k.copy_(snap[a][0]); arms.rings[a].v.copy_(snap[a][1])
                arms.rings[a].pos.copy_(snap[a][2])
            # teacher-forced: both arms fed the NEW arm's drafts
            dn, ln, hn, an = propose(arms, "new", slots, cur, tok, hid, n_iter, want_attn=True)
            do, lo, ho, ao = propose(arms, "old", slots, cur, tok, hid, n_iter, forced=dn, want_attn=True)
            for j in range(n_iter):
                da, ra = stats(an[j], ao[j])
                dl, rl = stats(ln[j], lo[j])
                worst_a, worst_ar = max(worst_a, da), max(worst_ar, ra)
                worst_l, worst_lr = max(worst_l, dl), max(worst_lr, rl)
                if j < K:
                    eq = ln[j].argmax(-1) == lo[j].argmax(-1)
                    agree_tf += int(eq.sum())
                    total_tf += eq.numel()
                line = (f"[parity] {name} round {rnd} step {j}: attn max|d| {da:.3e} (rel {ra:.2e})  "
                        f"logits max|d| {dl:.3e} (rel {rl:.2e})  argmax agree "
                        f"{(ln[j].argmax(-1) == lo[j].argmax(-1)).tolist()}  "
                        f"live keys {arms.rings['new'].keep(slots, cur + j).sum(-1).tolist()}")
                emit(log, line)
            for a in ("new", "old"):
                rg = arms.rings[a]
                for i, n in enumerate(acc):
                    s = rows[i][0]
                    rg.cur[s] += (1 + n) if isinstance(arms, MTPArms) else min(1 + n, K)
    line = (f"[parity] {name} SUMMARY: worst attn max|d| {worst_a:.3e} (rel {worst_ar:.2e}), worst logits "
            f"max|d| {worst_l:.3e} (rel {worst_lr:.2e}); drafted argmax agreement teacher-forced "
            f"{agree_tf}/{total_tf}, free-running {free_agree}/{free_total}")
    emit(log, line)
    return worst_ar, agree_tf, total_tf


def eager_vs_graph(arms, tokens, hiddens, n_iter, name, log):
    """New path only: the propose body captured in a CUDA graph must replay bit-identically to eager."""
    slots = torch.tensor([1, 2, 3, 4], device=DEV)
    cur = arms.rings["new"].cur[slots].clone()
    tok = tokens[cur].clone()
    hid = hiddens[cur - 1].clone()
    snap = (arms.rings["new"].k.clone(), arms.rings["new"].v.clone(), arms.rings["new"].pos.clone())
    res = {}

    def body():
        d, lg, hs, _ = propose(arms, "new", slots, cur, tok, hid, n_iter)
        res["d"] = torch.stack(d)
        res["l"] = lg[-1]
        res["h"] = hs[-1]

    def restore():
        arms.rings["new"].k.copy_(snap[0]); arms.rings["new"].v.copy_(snap[1])
        arms.rings["new"].pos.copy_(snap[2])

    with torch.inference_mode():
        body()
        eager = {k: v.clone() for k, v in res.items()}
        restore()
        g = capture(body)
        restore()
        g.replay()
        torch.cuda.synchronize()
        _KEEP.append(g)
        ok = all(torch.equal(eager[k], res[k]) for k in eager)
    restore()
    line = (f"[graph] {name}: eager == graph replay (drafts, last logits, last hidden) bit-identical: "
            f"{'YES' if ok else 'NO'}; drafts {eager['d'].t().tolist()}")
    emit(log, line)
    return ok


def timing(arms_cls, model, n_iter, name, log, hdim):
    """Graph replay, interleaved old/new/new_ctl, at ring R in {512, 2k, 8k} with a FULL live ring and
    at the served R=512 with a short live context (64) — the capacity-vs-live case."""
    cases = [(512, 512), (2048, 2048), (8192, 8192), (512, 64)]
    if os.environ.get("CASES"):
        cases = [tuple(int(v) for v in c.split("/")) for c in os.environ["CASES"].split(",")]
    for bs in [int(b) for b in os.environ.get("BS", "1,4").split(",")]:
        for R, live in cases:
            arms = arms_cls(model, R, bs + 1)
            slots = torch.arange(bs, device=DEV)
            cur0 = 100_000
            for a in ("new", "old"):
                rg = arms.rings[a]
                rg.k.normal_(0, 0.5)
                rg.v.normal_(0, 0.5)
                ap = torch.arange(cur0 - live, cur0, device=DEV)
                for s in range(bs):
                    rg.pos[s, torch.remainder(ap, R)] = ap
                rg.cur[:bs] = cur0
            cur = torch.full((bs,), cur0, device=DEV, dtype=torch.int64)
            tok = torch.randint(0, 150000, (bs,), device=DEV)
            hid = torch.randn(bs, hdim, device=DEV, dtype=DT)
            q_abs = cur.clone()
            posi = cur.to(torch.int32)
            if isinstance(arms, MTPArms):
                x = arms.h.input_layernorm.forward(arms.h.fuse(arms.h.embed(tok), hid), None)[0].clone()
            else:
                xe = arms.d.embed(tok).clone()
            keep = arms.rings["new"].keep(slots, q_abs)
            assert torch.equal(keep, arms.rings["old"].keep(slots, q_abs))
            if isinstance(arms, MTPArms):
                Hh, nope, rope = arms.h.self_attn.num_heads, arms.h.self_attn.qk_nope, arms.h.self_attn.qk_rope
                qn = torch.randn(bs, Hh, nope + rope, device=DEV, dtype=DT)
                core_args = (qn[..., :nope], qn[..., nope:], slots, q_abs, keep)
            else:
                core_args = (torch.randn(bs, arms.d.num_heads, arms.d.head_dim, device=DEV, dtype=DT),
                             slots, q_abs, keep)
            if os.environ.get("PROFILE") == "1":
                from torch.profiler import ProfilerActivity, profile
                with torch.inference_mode():
                    for arm in ("new", "old"):
                        for _ in range(5):
                            arms.attn_core(arm, *core_args)
                        torch.cuda.synchronize()
                        with profile(activities=[ProfilerActivity.CUDA]) as prof:
                            for _ in range(20):
                                arms.attn_core(arm, *core_args)
                            torch.cuda.synchronize()
                        print(f"[profile] {name} attn-core arm={arm} bs={bs} R={R} live={live} (20 iters)")
                        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
            with torch.inference_mode():
                gw, ga, gc = {}, {}, {}
                arm_list = [a for a in (("old", "old"), ("new", "new"), ("new_ctl", "new"))
                            if a[0] in os.environ.get("ARMS", "old,new,new_ctl").split(",")]
                # The new arm's kernel inputs, checked on the HOST before anything is captured: every
                # block-table entry inside the ring view, every length in [1, R].
                meta = arms.builder.meta(slots, q_abs, keep)
                bt, cl = meta.block_table, meta.ctx_lens
                ms = arms.rings["new"].k.shape[0]
                assert int(bt.min()) >= 0 and int(bt.max()) < ms * R, (int(bt.min()), int(bt.max()), ms * R)
                assert 1 <= int(cl.min()) and int(cl.max()) <= R, cl.tolist()
                phase(f"meta ok: bt in [{int(bt.min())}, {int(bt.max())}] < {ms * R}, ctx {cl.tolist()}")
                for key, arm in arm_list:
                    tg = f"{name} bs={bs} R={R} live={live} arm={key}"
                    gw[key] = capture(lambda arm=arm: propose(arms, arm, slots, cur, tok, hid, n_iter),
                                      f"{tg} propose")
                    gc[key] = capture(lambda arm=arm: arms.attn_core(arm, *core_args), f"{tg} core")
                    if isinstance(arms, MTPArms):
                        ga[key] = capture(lambda arm=arm: arms.attn_only(arm, x, slots, q_abs, posi),
                                          f"{tg} module")
                    else:
                        ga[key] = capture(lambda arm=arm: arms.attn_only(arm, xe, hid, slots, q_abs, posi),
                                          f"{tg} module")
                report_timing(f"{name} attn-core   bs={bs} R={R} live={live}", time_arms(gc), log)
                report_timing(f"{name} attn-module bs={bs} R={R} live={live}", time_arms(ga), log)
                report_timing(f"{name} propose({n_iter} steps) bs={bs} R={R} live={live}", time_arms(gw), log)
            # Graphs are NEVER destroyed in-process: tearing down these captured graphs (both arms,
            # MoE + lm-head bodies) corrupted the host heap on this torch/ROCm build ("double free
            # or corruption" in the NEXT replay or the teardown itself). Run one (bs, case) per
            # process (tools/glm_drafter_attn_bench.sh) and leave via os._exit.
            _KEEP.append((gw, ga, gc, arms))


def main():
    init_dist()
    log = []
    card = os.environ.get("ROCR_VISIBLE_DEVICES", "?")
    emit(log, f"[env] card ROCR_VISIBLE_DEVICES={card} torch={torch.__version__} "
              f"device={torch.cuda.get_device_name(0)} ITERS={ITERS} ROUNDS={ROUNDS} dtype={DT} "
              f"PARTS={PARTS} PARITY={PARITY} TIMING={TIMING} BS={os.environ.get('BS', '1,4')} "
              f"CASES={os.environ.get('CASES', 'all')}")
    toks = real_tokens(3 * 512 + 64)
    ok_all = True
    if "mtp" in PARTS:
        head = load_mtp_head()
        K = 2
        arms = MTPArms(head, 512, 6)
        if PARITY:
            hs = chain_hiddens(arms, toks, torch.randn(head.hidden_size, device=DEV, dtype=DT))
            rel, a, t = parity(arms, toks, hs, K + 1, K, "MTP", log)
            ok_all &= rel < 5e-2
            ok_all &= eager_vs_graph(arms, toks, hs, K + 1, "MTP", log)
        del arms
        if TIMING:
            timing(MTPArms, head, K + 1, "MTP", log, head.hidden_size)
        del head
        torch.cuda.empty_cache()
    if "eagle3" in PARTS:
        d = load_eagle3()
        K = 6
        arms = E3Arms(d, 512, 6)
        if PARITY:
            with torch.inference_mode():
                h0 = d.fuse_aux(torch.randn(1, 3, d.hidden_size, device=DEV, dtype=DT) * 0.5)
            hs = chain_hiddens(arms, toks, h0, reset=True)
            assert torch.isfinite(hs).all(), "non-finite EAGLE3 seed features"
            rel, a, t = parity(arms, toks, hs, K, K, "EAGLE3", log)
            ok_all &= rel < 5e-2
            ok_all &= eager_vs_graph(arms, toks, hs, K, "EAGLE3", log)
        del arms
        if TIMING:
            timing(E3Arms, d, K, "EAGLE3", log, d.hidden_size)
    emit(log, "[result] " + ("PASS" if ok_all else "FAIL"))
    sys.stdout.flush()
    os._exit(0 if ok_all else 1)   # skip graph/static-buffer destructors (see timing())


if __name__ == "__main__":
    main()
