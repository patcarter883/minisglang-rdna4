"""CAMMemory — the minisgl serve-side CAM editable-memory object.

Ports the minimal serving surface of the memory-organ CAM path, self-contained and CPU-importable:

  * `_ProductKeyStore`  — top-k product-key addressed value bank (from cam/pk_store.py:ProductKeyStore),
    read/write/init_state only (no training-only return_ctx / addr-sup / query-BN paths).
  * `_PKAdapter`        — the read/write front-end (from cam/pk_store_adapter.py:PKStoreAdapter):
    `_e`, `_pool_subject`, `_maxsim_reduce`, `persistent_write`, `persistent_bank`. The frozen base
    embed table is NOT reloaded — a reference to the serve model's embedding weight is used.
  * `_GatedMemoryTap`   — the zero-init gated cross-attention tap (from cam/gated_tap.py:GatedMemoryTap),
    fp32 params, additive cast back to the base dtype (byte-exact no-op at gamma=0).
  * `_GateRouter` + `signal_features` + `_inj_pertoken` — the per-token logit gate (from cam/gate_router.py).

CONTRACT (docs/zaya-port/CAM_SERVE_CONTRACT.md):
    CAMMemory(checkpoint_dir, base_embed, lm_head_weight)
    remember(subject_ids, prompt_last_logits[, object_ids]) -> bool   # base-uncertainty write gate
    read(subject_ids) -> (bank [1,K,mem], conf [1])
    apply_tap(h, bank, conf) -> h'                                    # residual-stream tap at tap_layer
    router_delta(base_last_logits, bank, conf) -> logit delta        # per-token router-gated injection

Env-driven CAM knobs that change addressing/injection are BAKED INTO meta.json by the exporter (WS-B)
and read from there, NOT from os.environ (a shared server must not depend on process env for numeric
correctness — integration_design.md §4 risk 2). See README.md for the exact expected checkpoint keys.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time


def _canon_subject(text: str) -> str:
    """#10 OPTIONAL subject canonicalization (MINISGL_CAM_CANON=1): lowercase + strip punctuation +
    collapse whitespace, so case/punctuation paraphrases of a subject key IDENTICALLY (the tau sweep
    showed a lowercased restatement drops to cos~0.58 under the pooled-embedding key; canonicalizing at
    both write and query time makes it 1.0). No-op unless enabled. Applied at the subject-string
    tokenization sites (scheduler); a trained semantic key is the memory-organ CAM_GTE_KEYS path."""
    if os.environ.get("MINISGL_CAM_CANON") != "1":
        return text
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

N_SIG = 8  # gate-router label-free signal count (fixed; see signal_features)


# --------------------------------------------------------------------------------------------------
# Gate router (ported from cam/gate_router.py) — per-token logit-space injection
# --------------------------------------------------------------------------------------------------
def signal_features(off: torch.Tensor, raw: torch.Tensor, conf: Optional[torch.Tensor]) -> torch.Tensor:
    """off [B,V] base last-token logits; raw [B,V] store push logits; conf [B] or None retrieval
    strength. Returns [B,N_SIG] label-free signal matrix (no true label used). Verbatim port."""
    B, V = off.shape
    p_off = torch.softmax(off, -1)
    logV = torch.log(torch.tensor(float(V), device=off.device))
    ent = -(p_off * torch.log(p_off.clamp_min(1e-12))).sum(-1) / logV
    store_tok = raw.argmax(-1)
    base_top = off.argmax(-1)
    idx = torch.arange(B, device=off.device)
    p_tgt = p_off[idx, store_tok]
    p_top = p_off[idx, base_top]
    store_prob = torch.softmax(raw, -1)
    store_peak = store_prob.max(-1).values
    store_ent = -(store_prob * torch.log(store_prob.clamp_min(1e-12))).sum(-1) / logV
    top2 = p_off.topk(2, -1).values
    base_margin = top2[:, 0] - top2[:, 1]
    c = torch.zeros(B, device=off.device) if conf is None else torch.log1p(conf.float().to(off.device)) / 10.0
    return torch.stack([c, ent, 1.0 - p_tgt, store_peak, p_tgt - p_top, p_top, base_margin, store_ent], dim=-1)


def _inj_pertoken(raw: torch.Tensor, g2: torch.Tensor, alpha_ref: float, topk: int) -> torch.Tensor:
    """Per-token injection: g2 [B,2] = (g_top, g_rest). g_top scales the store's argmax token, g_rest the
    rest of the top-k. Returns the additive logit delta [B,V]. Verbatim port."""
    B, V = raw.shape
    topk = min(topk, V)
    _, keep = raw.topk(topk, -1)                         # sorted desc -> keep[:,0] is the target
    w = torch.zeros_like(raw)
    w.scatter_(1, keep, g2[:, 1:2].expand(-1, topk))     # all top-k get g_rest
    w.scatter_(1, keep[:, :1], g2[:, 0:1])               # target overridden to g_top
    return alpha_ref * w * raw


class _GateRouter(nn.Module):
    def __init__(self, hidden: int = 32, n_out: int = 2):
        super().__init__()
        self.n_out = n_out
        self.net = nn.Sequential(
            nn.Linear(N_SIG, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_out),
        )

    def gain(self, sig: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.net(sig))                 # [B,n_out] in (0,1)
        return g.squeeze(-1) if self.n_out == 1 else g


# --------------------------------------------------------------------------------------------------
# Product-key store (ported from cam/pk_store.py) — serving read/write only
# --------------------------------------------------------------------------------------------------
class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class _ProductKeyStore(nn.Module):
    """Serving subset of ProductKeyStore: init_state / _address / head_query / write / read(return_conf)."""

    def __init__(self, d_hub: int, n_sub: int = 32, topk: int = 8, sub_topk: int = 4, n_heads: int = 3):
        super().__init__()
        assert d_hub % 2 == 0
        self.d_hub = d_hub
        self.d_half = d_hub // 2
        self.n_sub = n_sub
        self.N = n_sub * n_sub
        self.topk = topk
        self.sub_topk = sub_topk
        self.n_heads = n_heads
        self.write_beta = 1.0
        self.codebook1 = nn.Parameter(F.normalize(torch.randn(n_sub, self.d_half), dim=1))
        self.codebook2 = nn.Parameter(F.normalize(torch.randn(n_sub, self.d_half), dim=1))
        self.to_wkey = nn.Linear(d_hub, d_hub, bias=False)
        self.to_wval = nn.Linear(d_hub, d_hub, bias=False)
        self.read_q = nn.ModuleList([nn.Linear(d_hub, d_hub, bias=False) for _ in range(n_heads)])
        self.read_o = nn.ModuleList([nn.Linear(d_hub, d_hub, bias=False) for _ in range(n_heads)])
        self.read_norm = nn.ModuleList([_RMSNorm(d_hub) for _ in range(n_heads)])
        self.read_out_norm = _RMSNorm(d_hub)
        self.head_bias = nn.Parameter(torch.zeros(n_heads, d_hub))

    def init_state(self, batch: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch, self.N, self.d_hub, device=device, dtype=dtype)

    def _address(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Q, _ = q.shape
        st = self.sub_topk
        q1, q2 = q[..., :self.d_half], q[..., self.d_half:]
        s1 = q1 @ self.codebook1.t()
        s2 = q2 @ self.codebook2.t()
        v1, i1 = s1.topk(st, dim=-1)
        v2, i2 = s2.topk(st, dim=-1)
        cand = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).reshape(B, Q, -1)
        slot = (i1.unsqueeze(-1) * self.n_sub + i2.unsqueeze(-2)).reshape(B, Q, -1)
        w, sel = cand.topk(self.topk, dim=-1)
        slot_idx = torch.gather(slot, -1, sel)
        slot_w = torch.softmax(w, dim=-1)
        return slot_idx, slot_w

    def head_query(self, query: torch.Tensor, h: int = 0) -> torch.Tensor:
        return self.read_q[h](query) + self.head_bias[h]

    def write(self, V: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
              addr: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, A, _ = keys.shape
        bank_dtype = V.dtype
        wk = self.to_wkey(keys) if addr is None else addr
        wv = self.to_wval(values)
        slot_idx, slot_w = self._address(wk)
        gidx = slot_idx.reshape(B, A * self.topk, 1).expand(-1, -1, self.d_hub)
        cur = torch.gather(V, 1, gidx).reshape(B, A, self.topk, self.d_hub).float()
        delta = self.write_beta * slot_w.unsqueeze(-1) * (wv.unsqueeze(2).float() - cur)
        Vnew = V if not V.requires_grad else V.clone()
        Vnew.scatter_add_(1, gidx, delta.reshape(B, A * self.topk, self.d_hub).to(bank_dtype))
        return Vnew

    def read(self, V: torch.Tensor, query: torch.Tensor,
             return_conf: bool = False) -> Tuple[torch.Tensor, list, Optional[torch.Tensor]]:
        B, Q, _ = query.shape
        out = query.new_zeros(B, Q, self.d_hub)
        head_norms: list = []
        conf: Optional[torch.Tensor] = None
        for h in range(self.n_heads):
            qh = self.read_q[h](query) + self.head_bias[h]
            slot_idx, slot_w = self._address(qh)
            gidx = slot_idx.reshape(B, Q * self.topk, 1).expand(-1, -1, self.d_hub)
            vals = torch.gather(V, 1, gidx).reshape(B, Q, self.topk, self.d_hub).float()
            ctx = (slot_w.unsqueeze(-1) * vals).sum(dim=2)
            if return_conf and h == 0:
                conf = ctx.norm(dim=-1).mean(dim=1)          # [B] factual-head pre-norm retrieval strength
            oh = self.read_o[h](self.read_norm[h](ctx))
            out = out + oh
            head_norms.append(float(oh.detach().norm(dim=-1).mean()))
        out = self.read_out_norm(out)
        return out, head_norms, conf

    # ---- POINTER id-bank (#100): exact token id at the addressed slot, no value reconstruction ----
    def init_ids(self, batch: int, device) -> torch.Tensor:
        """A parallel per-slot token-ID bank [B,N] (init -1 = empty). Records WHICH token each slot owns
        so delivery can look up the exact id at the addressed slot instead of reconstructing a lossy value
        — decoupling the store's (reliable) ADDRESSING from its (lossy) value reconstruction (#100)."""
        return torch.full((batch, self.N), -1, dtype=torch.long, device=device)

    def write_ids(self, Vid: torch.Tensor, keys: torch.Tensor, ids: torch.Tensor,
                  addr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Record token `ids` [B,A] at each association's TOP-1 addressed slot (the argmax-weight slot the
        read selects). Same head-query addressing as the K1 value write ⇒ write-slot == read-slot."""
        wk = self.head_query(keys) if addr is None else addr
        slot_idx, slot_w = self._address(wk)                 # [B,A,topk]
        top = slot_w.argmax(dim=-1, keepdim=True)            # [B,A,1]
        top_slot = torch.gather(slot_idx, -1, top).squeeze(-1)  # [B,A]
        return Vid.scatter(1, top_slot, ids)

    def read_ids(self, Vid: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Look up the exact token id at the TOP-1 addressed slot (head-0 read addressing). [B,Q]->[B,Q]."""
        slot_idx, slot_w = self._address(self.head_query(query))
        top = slot_w.argmax(dim=-1, keepdim=True)
        top_slot = torch.gather(slot_idx, -1, top).squeeze(-1)
        return torch.gather(Vid, 1, top_slot)                # -1 where the slot is empty


# --------------------------------------------------------------------------------------------------
# PK adapter read/write front-end (ported from cam/pk_store_adapter.py) — persistent path only
# --------------------------------------------------------------------------------------------------
class _PKAdapter(nn.Module):
    """Read/write front-end for the persistent (Track-4) store. Holds in_proj/norm/subj_pool_q/store/
    readout_q/out_proj. The frozen base embed table is a REFERENCE to the serve model's embedding weight
    (never a copy) — `_e` embeds via F.embedding against it, matching `save_ckpt`'s dropped-embed rebuild."""

    def __init__(self, embed_weight: torch.Tensor, base_hidden: int, mem_dim: int, k: int,
                 n_sub: int, store_topk: int, store_sub_topk: int, read_heads: int, n_key_heads: int,
                 *, learned_key_pool: bool, key_maxsim: bool, key_maxsim_temp: float, write_at_read: bool):
        super().__init__()
        self.mem_dim = mem_dim
        self.k = k
        self.n_key_heads = max(1, n_key_heads)
        self.learned_key_pool = learned_key_pool
        self.key_maxsim = key_maxsim
        self.key_maxsim_temp = key_maxsim_temp
        self.write_at_read = write_at_read
        self._embed_weight = embed_weight                    # reference, NOT an nn.Parameter (no copy)
        self.in_proj = nn.Linear(base_hidden, mem_dim, bias=False)
        self.norm = nn.LayerNorm(mem_dim)
        self.subj_pool_q = nn.Parameter(torch.randn(self.n_key_heads, mem_dim) * 0.02)
        self.store = _ProductKeyStore(mem_dim, n_sub=n_sub, topk=store_topk, sub_topk=store_sub_topk,
                                      n_heads=read_heads)
        self.readout_q = nn.Parameter(torch.randn(k, mem_dim) * 0.02)
        self.out_proj = nn.Linear(mem_dim, base_hidden, bias=False)

    @property
    def device(self):
        return self.in_proj.weight.device

    def _e(self, ids: torch.Tensor) -> torch.Tensor:
        """base ids [B,L] -> [B,L,mem_dim] normalized mem embeds (frozen embed -> in_proj -> LayerNorm)."""
        e = F.embedding(ids, self._embed_weight).float()
        return self.norm(self.in_proj(e))

    def _pool_subject(self, span: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Pool a subject-span embed [B,L,mem] -> subject key/query. learned attention pool when
        learned_key_pool (multi-vector [B,H,mem] when n_key_heads>1), else uniform mean [B,mem]."""
        if self.learned_key_pool:
            scores = torch.einsum("blm,hm->bhl", span, self.subj_pool_q) / (self.mem_dim ** 0.5)
            w = torch.softmax(scores, dim=-1)
            pooled = torch.einsum("bhl,blm->bhm", w, span)   # [B,H,mem]
            if self.n_key_heads > 1:
                return pooled
            pooled = pooled[:, 0]
        else:
            pooled = span.mean(dim=1)
        return pooled.unsqueeze(1) if keepdim else pooled

    def _maxsim_reduce(self, read: torch.Tensor) -> torch.Tensor:
        """Soft-MaxSim over multi-vector reads [B,H,mem] -> [B,1,mem]. No-op unless key_maxsim and H>1."""
        if read.shape[1] <= 1 or not self.key_maxsim:
            return read
        w = torch.softmax(read.norm(dim=-1) / self.key_maxsim_temp, dim=1)
        return (w.unsqueeze(-1) * read).sum(dim=1, keepdim=True)

    def persistent_write(self, V: torch.Tensor, keys: torch.Tensor, vals: torch.Tensor) -> torch.Tensor:
        """Error-correcting delta write of (keys,vals) into standing bank V. write_at_read (K1 default)
        addresses the write with the read query head_query(key,0) so the value lands at the read slot."""
        addr = self.store.head_query(keys, 0) if self.write_at_read else None
        return self.store.write(V, keys, vals, addr=addr)

    def persistent_bank(self, V: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Read standing bank V with subject query q [B,Lq,mem] -> pooled [B,K,mem]; sets self._last_conf."""
        read, _hn, self._last_conf = self.store.read(V, q, return_conf=True)
        read = self._maxsim_reduce(read)
        B = q.shape[0]
        pq = self.readout_q.unsqueeze(0).expand(B, -1, -1)
        attn = torch.softmax(pq @ read.transpose(1, 2) / (self.mem_dim ** 0.5), dim=-1)
        return attn @ read

    def persistent_write_ids(self, Vid: torch.Tensor, keys: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """POINTER write (#100): record token `ids` [B,A] at the addressed slots of the id-bank Vid,
        using the SAME K1 head-query addressing as persistent_write so write-slot == read-slot."""
        addr = self.store.head_query(keys, 0) if self.write_at_read else None
        return self.store.write_ids(Vid, keys, ids, addr=addr)

    def persistent_read_ids(self, Vid: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """POINTER read: exact token id at the addressed slot -> [B,Q]. Uses the store's reliable
        ADDRESSING, skipping the lossy value reconstruction that floored multi-token delivery at ~0.5/tok."""
        return self.store.read_ids(Vid, q)


# --------------------------------------------------------------------------------------------------
# Gated memory tap (ported from cam/gated_tap.py) — residual-stream injection at tap_layer
# --------------------------------------------------------------------------------------------------
class _GatedMemoryTap(nn.Module):
    """Zero-init gated cross-attention from the residual stream into the K-slot memory bank. fp32 params;
    additive update cast back to the base dtype (byte-exact no-op at gamma=0)."""

    def __init__(self, base_hidden: int, mem_dim: int, n_heads: int = 8, conf_gate: bool = False,
                 n_rel: int = 1, norm_gate: bool = False, twosided: bool = False):
        super().__init__()
        assert base_hidden % n_heads == 0
        self.H, self.n_heads, self.d_head = base_hidden, n_heads, base_hidden // n_heads
        self.to_q = nn.Linear(base_hidden, base_hidden, bias=False)
        self.to_k = nn.Linear(mem_dim, base_hidden, bias=False)
        self.to_v = nn.Linear(mem_dim, base_hidden, bias=False)
        self.to_o = nn.Linear(base_hidden, base_hidden, bias=False)
        self.gamma = nn.Parameter(torch.zeros(base_hidden))
        self.gate_alpha = nn.Parameter(torch.tensor(-6.0))
        self.supp = nn.Parameter(torch.tensor(-4.0))
        self.null_key = nn.Parameter(torch.zeros(1, n_heads, 1, self.d_head))
        self.conf_gate = conf_gate
        self.n_rel = n_rel
        self.norm_gate = norm_gate
        self.twosided = twosided
        self.conf_scale = nn.Parameter(torch.tensor(4.0))
        self.conf_bias = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("conf_ema", torch.full((n_rel,), -1.0))

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, h: torch.Tensor, bank: Optional[torch.Tensor],
                conf: Optional[torch.Tensor] = None, relidx: int = 0) -> torch.Tensor:
        """h [B,T,H] residual hidden -> injected [B,T,H]. No-op when bank is None. Verbatim math port
        (norm_gate/twosided read from baked flags, not os.environ). Eval-only: conf_ema is a pure read."""
        if bank is None:
            return h
        wdt = self.to_q.weight.dtype                         # tap compute dtype (fp32)
        h32 = h.to(wdt)
        bank = bank.to(device=h.device, dtype=wdt)
        B = h32.shape[0]
        q = self._split(self.to_q(h32))                      # [B,nh,T,dh]
        k = self._split(self.to_k(bank))                     # [B,nh,K,dh]
        v = self._split(self.to_v(bank))
        nk = self.null_key.to(wdt).expand(B, self.n_heads, 1, self.d_head)
        k = torch.cat([k, nk], dim=2)
        v = torch.cat([v, torch.zeros_like(nk)], dim=2)
        a = torch.softmax(q @ k.transpose(-1, -2) / (self.d_head ** 0.5), dim=-1)
        ctx = (a @ v).transpose(1, 2).reshape(h32.shape)
        y = self.to_o(ctx)
        if self.norm_gate:
            ydir = y / (y.norm(dim=-1, keepdim=True) + 1e-6)
            alpha = torch.sigmoid(self.gate_alpha)
            upd = alpha * h32.norm(dim=-1, keepdim=True) * ydir
        else:
            g = torch.tanh(self.gamma)
            upd = g * y
        c = None
        if self.conf_gate and conf is not None:
            cf = conf.to(device=h.device, dtype=wdt).detach()
            ri = max(0, min(int(relidx), self.n_rel - 1))
            scale = self.conf_ema[ri].clamp_min(1e-4)
            c = torch.sigmoid(self.conf_scale * (cf / scale - self.conf_bias))   # [B] in (0,1)
            upd = c.view(B, 1, 1) * upd
        if self.twosided:
            s = torch.sigmoid(self.supp)
            cs = c.view(B, 1, 1) if c is not None else 1.0
            upd = upd - s * cs * h32
        return h + upd.to(h.dtype)


# --------------------------------------------------------------------------------------------------
# CAMMemory — the public serve object
# --------------------------------------------------------------------------------------------------
def _subject_bank(subject_ids: List[int], B: int) -> int:
    """Stable subject-identity hash -> bank index in [0,B) on the discrete token-ids (write==read route)."""
    if B <= 1:
        return 0
    h = hashlib.md5(",".join(map(str, subject_ids)).encode()).hexdigest()
    return int(h, 16) % B


def _strip_modctdict_prefix(sd: dict) -> dict:
    """taps saved as a ModuleDict keyed by str(layer) have keys like '24.to_q.weight'. Strip a leading
    numeric component so a single-tap state_dict ('to_q.weight') loads uniformly."""
    out = {}
    for kk, vv in sd.items():
        parts = kk.split(".", 1)
        out[parts[1] if len(parts) == 2 and parts[0].isdigit() else kk] = vv
    return out


def _load_filtered(module: nn.Module, sd: dict, name: str) -> None:
    """Load `sd` into `module`, keeping only keys the module defines (tolerates the dropped embed/unembed
    and any extra training-only tensors in the donor state_dict)."""
    own = module.state_dict()
    keep = {kk: vv for kk, vv in sd.items() if kk in own}
    missing, unexpected = module.load_state_dict(keep, strict=False)
    dropped = [kk for kk in sd if kk not in own]
    if dropped:
        logger.info("CAM %s: ignored %d non-matching ckpt keys (e.g. %s)", name, len(dropped), dropped[:4])
    real_missing = [m for m in missing if not (m.startswith("_") or "conf_ema" in m)]
    if real_missing:
        logger.warning("CAM %s: MISSING params after load: %s", name, real_missing)


class _NsState:
    """Per-namespace EDITABLE store state (#6 multi-tenant isolation). Trained adapter/tap/router/embed
    weights are SHARED across namespaces; only this — the value banks, the cosine-NN subject index, the
    fact table, and the freeze flag — is partitioned per tenant/session so one conversation cannot read
    or overwrite another's memory."""
    # seq: monotonic LRU clock (#9); evicted: count dropped for capacity; last_key: most-recent write (#12 undo)
    __slots__ = ("banks", "subj_keys", "subj_objs", "subj_tuple", "facts", "frozen",
                 "seq", "evicted", "last_key")

    def __init__(self, banks, frozen=False, subj_keys=None, subj_objs=None, subj_tuple=None, facts=None):
        self.banks = banks
        self.subj_keys = [] if subj_keys is None else subj_keys
        self.subj_objs = [] if subj_objs is None else subj_objs
        self.subj_tuple = [] if subj_tuple is None else subj_tuple
        self.facts = {} if facts is None else facts
        self.frozen = frozen
        self.seq = 0
        self.evicted = 0
        self.last_key = None


class CAMMemory:
    """Serve-side editable memory: a standing product-key store (B disjoint banks) + trained tap + router.

    Construct once per engine. Then per request:
      * write:  remember(subject_ids, prompt_last_logits[, object_ids])  (base-uncertainty gate)
      * read:   bank, conf = read(subject_ids)                            (once at prefill)
      * tap:    h = apply_tap(h, bank, conf)                              (after tap_layer, staged on model)
      * router: logits = base_last_logits + router_delta(base_last_logits, bank, conf)  (at lm_head)

    If `checkpoint_dir` is missing/None, the object constructs DISABLED (enabled=False): remember/read are
    no-ops and read returns (None, None), so a server can boot without a checkpoint and simply not offer
    memory. `import minisgl.cam.memory` never touches a checkpoint.
    """

    def __init__(self, checkpoint_dir: Optional[str], base_embed, lm_head_weight: torch.Tensor,
                 pointer_only: bool = False, decode=None):
        self.enabled = False
        # POINTER/RETRIEVE-ONLY mode (set when the served model lacks the tap seam, e.g. MoE 35B): the
        # trained tap/adapter/router are dimensioned to the CHECKPOINT's base and would matmul-mismatch a
        # different served hidden width, so their CALL sites (_write value-store, read/apply_tap, router)
        # are skipped. Exact-object delivery + ambient retrieve run from the cosine subject index, which
        # uses only the served model's own (full-vocab) embedding and is dimension-independent.
        self.pointer_only = pointer_only
        self.banks: Optional[List[torch.Tensor]] = None
        self._facts: dict = {}                               # tuple(subject_ids) -> {"object_ids", ...}
        self._pending_object: Optional[List[int]] = None
        # #6 safe defaults so _state() never AttributeErrors on a DISABLED store (populated when enabled).
        self._ns_states: dict = {}
        self._init_banks: list = []
        self._default_frozen = False

        if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
            logger.warning("CAMMemory: no checkpoint dir (%s) — constructing DISABLED (memory off).",
                           checkpoint_dir)
            return

        embed_weight = base_embed.weight if hasattr(base_embed, "weight") else base_embed
        device = embed_weight.device
        self.device = device
        self.lm_head_weight = lm_head_weight

        meta_path = os.path.join(checkpoint_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        self.meta = meta

        # --- scalar/knob config (baked into meta by the exporter; see README) ---
        self.tap_layer = int(meta["tap_layer"])
        # n_banks is a pure SERVING knob (subject-bucket count): the store codebooks/projections are
        # bucket-agnostic (banks are just parallel value/id states), so more banks = fewer subjects per
        # bank = far less pointer-delivery collision at scale — with NO re-export/re-training. The
        # robustness sweep measured n_banks=32 -> 0.37 vs 512 -> 0.91 span-exact @ N=400. Override via
        # MINISGL_CAM_NBANKS to scale to the deployment's fact count (~4x expected N is a good rule).
        self.n_banks = int(os.environ.get("MINISGL_CAM_NBANKS") or meta.get("n_banks", 1))
        self.remember_tau = float(meta.get("remember_tau", 0.5))
        self.router_alpha = float(meta.get("router_alpha", 1.5))
        self.router_topk = int(meta.get("topk", 16))         # router multigate top-k (NOT the store topk)
        self.obj_latent = bool(meta.get("obj_latent", False))
        self.value_suppress = float(meta.get("value_suppress", 0.0))
        pooled_subj_key = bool(meta.get("pooled_subj_key", True))
        self._pooled_subj_key = pooled_subj_key
        # POINTER id-bank (#100): the max object length the per-position id-banks retain. Position
        # separation comes from SEPARATE id-banks per position (subject-keyed), so no learned pos_tag is
        # needed — the pointer stores exact ids, so its only requirement is write-slot == read-slot.
        self.mt_cap = int(meta.get("mt_positions", 0)) or 16

        # --- load raw tensors ---
        tap_sd = _strip_modctdict_prefix(torch.load(os.path.join(checkpoint_dir, "tap.pt"),
                                                     map_location=device, weights_only=False))
        adapter_sd = torch.load(os.path.join(checkpoint_dir, "adapter.pt"),
                                map_location=device, weights_only=False)
        router_sd = torch.load(os.path.join(checkpoint_dir, "router.pt"),
                               map_location=device, weights_only=False)

        # --- build the tap (shapes inferred from tap.pt where possible) ---
        base_hidden = int(tap_sd["to_q.weight"].shape[0])
        mem_dim = int(tap_sd["to_k.weight"].shape[1])
        # Robust pointer_only gate: even when the tap SEAM is present, the loaded tap/adapter/router are
        # dimensioned to the CHECKPOINT's base_hidden. If the SERVED model's hidden differs (e.g. a 4B
        # checkpoint on the 35B-A3B: 2560 vs 2048) every tap/adapter/router matmul mismatches, so fall
        # back to pointer/retrieve-only (dimension-independent) instead of crashing in _write/read.
        served_hidden = int(embed_weight.shape[1])
        if base_hidden != served_hidden and not self.pointer_only:
            logger.warning("CAMMemory: checkpoint base_hidden=%d != served hidden=%d — tap/adapter/router "
                           "DISABLED, pointer/retrieve-only (train a checkpoint on this base for the tap).",
                           base_hidden, served_hidden)
            self.pointer_only = True
        tap_heads = int(tap_sd["null_key"].shape[1]) if "null_key" in tap_sd else int(meta.get("tap_heads", 8))
        n_rel = int(tap_sd["conf_ema"].shape[0]) if "conf_ema" in tap_sd else int(meta.get("n_rel", 1))
        self.tap = _GatedMemoryTap(
            base_hidden, mem_dim, n_heads=tap_heads,
            conf_gate=bool(meta.get("conf_gate", False)), n_rel=n_rel,
            norm_gate=bool(meta.get("norm_gate", False)), twosided=bool(meta.get("twosided", False)),
        ).to(device).float()
        _load_filtered(self.tap, tap_sd, "tap")
        self.tap.eval()

        # --- build the adapter (shapes inferred from adapter.pt where possible) ---
        k_slots = int(adapter_sd["readout_q"].shape[0]) if "readout_q" in adapter_sd else int(meta.get("k", 8))
        a_mem = int(adapter_sd["readout_q"].shape[1]) if "readout_q" in adapter_sd else mem_dim
        n_sub = int(adapter_sd["store.codebook1"].shape[0]) if "store.codebook1" in adapter_sd \
            else int(meta.get("n_sub", 32))
        n_key_heads = int(adapter_sd["subj_pool_q"].shape[0]) if "subj_pool_q" in adapter_sd \
            else int(meta.get("n_key_heads", 1))
        read_heads = sum(1 for kk in adapter_sd if kk.startswith("store.read_q.")
                         and kk.endswith(".weight")) or int(meta.get("read_heads", 3))
        self.adapter = _PKAdapter(
            embed_weight, base_hidden, a_mem, k_slots,
            n_sub=n_sub, store_topk=int(meta.get("store_topk", 8)),
            store_sub_topk=int(meta.get("store_sub_topk", 4)),
            read_heads=read_heads, n_key_heads=n_key_heads,
            learned_key_pool=bool(meta.get("learned_key_pool", False)),
            key_maxsim=bool(meta.get("key_maxsim", False)),
            key_maxsim_temp=float(meta.get("key_maxsim_temp", 0.1)),
            write_at_read=bool(meta.get("write_at_read", True)),
        ).to(device).float()
        _load_filtered(self.adapter, adapter_sd, "adapter")
        self.adapter.eval()

        # --- build the router (shapes inferred from router.pt) ---
        hidden = int(router_sd["net.0.weight"].shape[0])
        last_w = [kk for kk in router_sd if kk.startswith("net.") and kk.endswith(".weight")][-1]
        n_out = int(router_sd[last_w].shape[0])
        self.router = _GateRouter(hidden=hidden, n_out=n_out).to(device).float()
        _load_filtered(self.router, router_sd, "router")
        self.router.eval()

        # --- freeze everything + empty banks ---
        for m in (self.tap, self.adapter, self.router):
            for p in m.parameters():
                p.requires_grad_(False)
        self.banks = [self.adapter.store.init_state(1, device, dtype=torch.float32)
                      for _ in range(self.n_banks)]
        # Bank tensor dims (read()/persistent_bank return [1, K, mem_dim]) — the graph-capture static
        # buffer [max_bs, K, mem_dim] needs these at engine build (Phase 2).
        self.k_slots = k_slots
        self.mem_dim = a_mem
        # POINTER delivery via a COSINE-NN SUBJECT INDEX (#100): a stored subject key = L2-normalised mean
        # of the base model's INPUT embeddings over the subject tokens; delivery returns the object of the
        # nearest stored subject by cosine (above deliver_tau). This is EXACT retrieval (no product-key
        # slot collision → span-exact stays 1.0 to N=500) and order/title/case robust (reordered / "Ms. X"
        # deliver 1.0; unknown subjects max-cos ≤0.51 so tau=0.7 rejects them). Supersedes the per-position
        # product-key id-bank for delivery — no new model, uses the base embeds we already hold.
        self._embed_w = embed_weight
        self._subj_keys: List[torch.Tensor] = []             # [base_hidden] normalised pooled subject keys
        self._subj_objs: List[List[int]] = []                # parallel object-id sequences
        self._subj_tuple: List[tuple] = []                   # parallel tuple(subject_ids) (update/forget)
        self.deliver_tau = float(os.environ.get("MINISGL_CAM_DELIVER_TAU", "0.7"))
        self.enabled = True
        # ---- whitened-GTE semantic subject key (opt-in: MINISGL_CAM_GTE_KEY=1) --------------------------
        # The base-embed key above is LEXICAL: it addresses a paraphrased subject poorly (bake-off
        # addr_para 0.06). A whitened GTE-ModernColBERT key lifts paraphrase addressing to ~0.43 — semantic
        # similarity plus soft-ZCA whitening (which kills the anisotropy that otherwise makes semantic keys
        # collide: NN-cos 0.99 -> 0.58). `_decode` (subject_ids -> text) lets _subj_key encode from text with
        # no threading through every call site; store keys rebuilt from facts on load become GTE keys too.
        # Loads lazily on CPU (short subjects encode in ms; keeps GPU for the base) and falls back to the
        # base-embed key if the encoder or artifact is missing, so a misconfig degrades, never crashes.
        self._gte = None            # (encoder, mu, W) when active; None -> base-embed key
        self._decode = decode       # callable(subject_ids)->str, set by the caller (scheduler tokenizer)
        if os.environ.get("MINISGL_CAM_GTE_KEY") == "1":
            self._load_gte_key()
        # ---- write gating (protect a curated/ingested store from ambient auto-write) -------------------
        # frozen: read-only — refuse AMBIENT auto-write (explicit force ingest still writes). Flip at
        #   runtime via freeze()/unfreeze() (POST /cam/freeze) or start frozen with MINISGL_CAM_FROZEN=1.
        # write_policy 'no-clobber' (DEFAULT): auto-write may ADD a genuinely-new subject but never
        #   overwrite/shadow an existing one (exact id match, or cosine >= protect_tau to a stored key).
        #   Set MINISGL_CAM_WRITE_POLICY=overwrite for the old always-write behaviour.
        # protect_tau DEFAULTS TO deliver_tau (0.70): "refuse to re-write what the store would already
        #   deliver for this subject." Measured on real Qwen3.5-4B subject keys (tau_sweep): 0.70 sits above
        #   distinct subjects (<=0.51) and different-people-sharing-a-name (<=0.62), and catches confident
        #   same-subject paraphrases (reorder/trailing/"the"/titles). A missed rewording only appends a
        #   redundant entry (_write overwrites only on EXACT subject-id match); a false refuse would silently
        #   drop a new fact — so the default errs toward learning. Override with MINISGL_CAM_PROTECT_TAU.
        self.frozen = os.environ.get("MINISGL_CAM_FROZEN") == "1"
        self.write_policy = os.environ.get("MINISGL_CAM_WRITE_POLICY", "no-clobber")
        self.protect_tau = float(os.environ.get("MINISGL_CAM_PROTECT_TAU", str(self.deliver_tau)))
        # --- per-namespace editable state (#6) --- the DEFAULT namespace WRAPS the objects built above,
        # so single-store callers are byte-unchanged; new namespaces clone the checkpoint's initial banks
        # and start with an empty index (shared trained weights are never duplicated).
        self._default_frozen = self.frozen
        self._init_banks = [b.detach().clone() for b in self.banks]
        self._ns_states = {"default": _NsState(self.banks, self.frozen, self._subj_keys,
                                               self._subj_objs, self._subj_tuple, self._facts)}
        # #9 capacity: per-namespace fact cap (0 = unlimited); LRU eviction on overflow.
        self.max_facts = int(os.environ.get("MINISGL_CAM_MAX_FACTS", "0"))
        # Write-side semantic dedup: on a NON-exact-id re-remember whose subject key is a near-duplicate
        # of a stored one (cosine >= write_dedup_tau), MERGE onto that entry (latest phrasing+object wins)
        # instead of appending a paraphrase duplicate. Default 1.0 = OFF (base-embed key space is not
        # calibrated for it); _load_gte_key drops it to a measured 0.82 when the semantic GTE key is active
        # (0.82 sits above the distinct-subject cosine ceiling ~0.79 — e.g. "Mozart" vs "Leopold Mozart" —
        # so genuinely-different subjects never silently merge). Explicit env override always wins.
        _dedup = os.environ.get("MINISGL_CAM_WRITE_DEDUP_TAU", "").strip()
        self.write_dedup_tau = float(_dedup) if _dedup else 1.0
        # #12 audit: append-only ring buffer of write/forget/evict events (subject/object/source/ns/ts).
        self._audit: list = []
        self._audit_max = int(os.environ.get("MINISGL_CAM_AUDIT_MAX", "2000"))
        # #7 persistence: load-on-boot + debounced autosave to MINISGL_CAM_STORE_PATH.
        self.store_path = os.environ.get("MINISGL_CAM_STORE_PATH")
        self._save_interval = float(os.environ.get("MINISGL_CAM_SAVE_INTERVAL", "5"))
        self._dirty = False
        self._last_save = 0.0
        logger.info("CAMMemory loaded: tap_layer=%d n_banks=%d mem_dim=%d K=%d tap_heads=%d read_heads=%d "
                    "router n_out=%d tau=%.3f", self.tap_layer, self.n_banks, a_mem, k_slots, tap_heads,
                    read_heads, n_out, self.remember_tau)
        if self.store_path and os.path.isfile(self.store_path):   # #7 load-on-boot
            try:
                n = self.restore(self.store_path)
                logger.info("CAMMemory: restored %d edits across %d namespace(s) from %s",
                            n, len(self._ns_states), self.store_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("CAMMemory: store restore failed (%s) — starting empty.", e)

    # ---- object supply for the write gate --------------------------------------------------------
    def set_pending_object(self, object_ids: List[int]) -> None:
        """Stage the object token-ids for the NEXT remember() call whose object_ids arg is omitted (keeps
        the contract's 2-arg remember(subject_ids, prompt_last_logits) callable)."""
        self._pending_object = list(object_ids)

    # ---- write gate ------------------------------------------------------------------------------
    @torch.no_grad()
    def _state(self, ns: str = None) -> "_NsState":
        """Editable state for a namespace (#6), lazily created (cloned initial banks + empty index) on first
        use. ns=None/"default" -> the shared default store, so single-store callers are unchanged."""
        ns = ns or "default"
        st = self._ns_states.get(ns)
        if st is None:
            st = _NsState([b.clone() for b in self._init_banks], self._default_frozen)
            self._ns_states[ns] = st
        return st

    def namespaces(self) -> list:
        return list(self._ns_states.keys())

    def remember(self, subject_ids: List[int], prompt_last_logits: torch.Tensor,
                 object_ids: Optional[List[int]] = None, ns: str = None) -> bool:
        """Base-uncertainty WRITE GATE: store subject->object iff the base can't already recall the object
        (softmax(prompt_last_logits)[object first token] < remember_tau). Returns whether it was stored.

        The object token-ids are supplied EITHER via the explicit `object_ids` arg OR (if omitted) via a
        prior set_pending_object(...) — WS-A's resolution of the contract's "object ids passed via
        set_pending_object or a param" clause. `prompt_last_logits` is the base's last-position logits for
        the fact-probe prompt (e.g. "The capital of France is"); the caller computes it with memory OFF."""
        if not self.enabled:
            return False
        obj = object_ids if object_ids is not None else self._pending_object
        if not obj:
            raise ValueError("remember() needs object_ids (arg) or a prior set_pending_object(...)")
        self._pending_object = None
        obj_first = int(obj[0])
        p = float(torch.softmax(prompt_last_logits.float().reshape(-1), -1)[obj_first])
        if p >= self.remember_tau:
            return False                                     # base already knows it -> store the unknowable only
        self._write(subject_ids, obj, ns=ns, base_p=p)
        return True

    @torch.no_grad()
    def _write(self, subject_ids: List[int], object_ids: List[int], ns: str = None,
               base_p: float = 0.0) -> None:
        st = self._state(ns)
        # Value-store / product-key write (feeds the tap+router fallback). Skipped in pointer-only mode:
        # the adapter is sized to the CHECKPOINT's base hidden, so `_e`/`persistent_write` would matmul-
        # mismatch a different served base (e.g. a 4B checkpoint on a 35B serve). The pointer index below
        # is dimension-independent and is all the exact-object delivery path needs.
        if not self.pointer_only:
            dev = self.adapter.device
            tids = torch.tensor([list(subject_ids)], dtype=torch.long, device=dev)
            subj_emb = self.adapter._e(tids)                     # [1,S,mem]
            key = self.adapter._pool_subject(subj_emb, keepdim=True) if self._pooled_subj_key \
                else subj_emb[:, -1:]                             # [1,1,mem] or [1,H,mem]
            if self.obj_latent and len(object_ids) > 1:
                oi = torch.tensor([list(object_ids)], dtype=torch.long, device=dev)
                val = self.adapter._e(oi).mean(1, keepdim=True)  # [1,1,mem] object phrase latent
            else:
                val = self.adapter._e(torch.tensor([[int(object_ids[0])]], dtype=torch.long, device=dev))
            if key.shape[1] > 1:                                 # multi-vector keys: same value to H slots
                val = val.expand(-1, key.shape[1], -1)
            b = _subject_bank(list(subject_ids), self.n_banks)
            st.banks[b] = self.adapter.persistent_write(st.banks[b], key, val)
        # POINTER delivery index: store the subject's cosine key + its EXACT object token sequence, so
        # /cam/ask delivers the whole multi-token object losslessly and paraphrase-robustly (the value
        # bank above only carries the first-token seed for the router/tap fallback). Update-in-place on
        # re-remember of the same subject.
        k = tuple(int(s) for s in subject_ids)
        obj = [int(o) for o in object_ids]
        key_vec = self._subj_key(subject_ids)
        st.seq += 1
        if k in st.subj_tuple:
            i = st.subj_tuple.index(k)                            # exact re-remember: update object + key
            st.subj_keys[i], st.subj_objs[i] = key_vec, obj
            fact_key = k
        else:
            i = self._dedup_match(st, key_vec)                    # paraphrase near-duplicate of a stored subject?
            if i is not None:                                     # MERGE: newest OBJECT wins, but keep the
                fact_key = st.subj_tuple[i]                       # FIRST-SEEN subject key ANCHORED — updating it
                st.subj_objs[i] = obj                             # to each new phrasing would drift the anchor and
                self._audit_add("merge", ns, fact_key, obj)       # cascade-merge a later distinct subject (data loss)
            else:                                                 # genuinely new subject: append
                st.subj_tuple.append(k); st.subj_keys.append(key_vec); st.subj_objs.append(obj)
                fact_key = k
        st.facts[fact_key] = {"object_ids": obj, "base_p": float(base_p), "used": st.seq}  # #6 index + #9 LRU clock
        st.last_key = fact_key                                   # #12 undo target
        self._audit_add("write", ns, fact_key, obj)             # #12 audit
        self._maybe_evict(st, ns)                               # #9 capacity
        self._dirty = True                                      # #7 persistence

    @torch.no_grad()
    def _dedup_match(self, st, key_vec: torch.Tensor) -> Optional[int]:
        """Index of a stored subject whose key is a near-duplicate of key_vec (cosine >= write_dedup_tau),
        else None. Collapses paraphrase re-remembers ("Mozart" vs "the composer Mozart") onto one entry
        instead of appending a duplicate. tau is set ABOVE the measured distinct-subject cosine ceiling
        (~0.79) so genuinely-different-but-related subjects (e.g. Mozart vs Leopold Mozart) never merge —
        a missed merge is a recoverable duplicate; a wrong merge is silent data loss."""
        if self.write_dedup_tau >= 1.0 or not st.subj_keys:
            return None
        sims = torch.stack(st.subj_keys).to(key_vec.device) @ key_vec
        j = int(sims.argmax())
        return j if float(sims[j]) >= self.write_dedup_tau else None

    # ---- #9 capacity / eviction --------------------------------------------------------------------
    @torch.no_grad()
    def _maybe_evict(self, st, ns: str) -> None:
        """Evict least-recently-used facts from a namespace once it exceeds max_facts (0 = unlimited).
        Delivery-correct: drops from the cosine-NN index + fact table (bank residue is the router/tap
        fallback only; a full rebuild is the #12 true-erase path)."""
        if self.max_facts <= 0:
            return
        while len(st.facts) > self.max_facts:
            victim = min(st.facts, key=lambda kk: st.facts[kk].get("used", 0))   # LRU
            obj = st.facts[victim]["object_ids"]
            del st.facts[victim]
            if victim in st.subj_tuple:
                i = st.subj_tuple.index(victim)
                del st.subj_tuple[i]; del st.subj_keys[i]; del st.subj_objs[i]
            st.evicted += 1
            self._audit_add("evict", ns, victim, obj)

    # ---- #12 audit ---------------------------------------------------------------------------------
    def _audit_add(self, op: str, ns: str, subject_ids, object_ids) -> None:
        self._audit.append({"ts": time.time(), "op": op, "ns": ns or "default",
                            "subject_ids": list(subject_ids), "object_ids": list(object_ids)})
        if len(self._audit) > self._audit_max:
            del self._audit[:len(self._audit) - self._audit_max]

    def audit_log(self, ns: str = None, limit: int = 100) -> list:
        """Recent write/forget/evict events (most-recent last), optionally filtered to a namespace (#12)."""
        rows = self._audit if ns is None else [r for r in self._audit if r["ns"] == (ns or "default")]
        return rows[-limit:]

    # ---- #7 persistence: debounced autosave --------------------------------------------------------
    def autosave(self, force: bool = False) -> bool:
        """Snapshot to store_path if dirty and the debounce interval elapsed (or force). Returns whether it
        saved. Cheap no-op when no store_path / not dirty."""
        if not self.store_path or (not self._dirty and not force):
            return False
        now = time.time()
        if not force and (now - self._last_save) < self._save_interval:
            return False
        try:
            self.snapshot(self.store_path)
            self._dirty = False
            self._last_save = now
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("CAMMemory autosave failed: %s", e)
            return False

    @torch.no_grad()
    def _load_gte_key(self) -> None:
        """Load the whitened-GTE key: a GTE-ModernColBERT encoder (CPU) + the soft-ZCA (mu, W) artifact
        (MINISGL_CAM_GTE_WHITEN). On ANY failure (no pylate / model / artifact) leave self._gte = None so
        _subj_key falls back to the base-embed key — the feature degrades, it never breaks the serve."""
        import pickle
        try:
            import numpy as np
            from .gte_encoder import GTEEncoder
            path = os.environ.get("MINISGL_CAM_GTE_WHITEN", "/cam_gte/gte_whiten.pkl")
            art = pickle.load(open(path, "rb"))
            model = os.environ.get("MINISGL_CAM_GTE_MODEL", art.get("model", "lightonai/GTE-ModernColBERT-v1"))
            enc = GTEEncoder(model)
            mu = torch.tensor(np.asarray(art["mu"], dtype=np.float32))
            W = torch.tensor(np.asarray(art["W"], dtype=np.float32))
            self._gte = (enc, mu, W)
            # GTE cosine scale differs from base-embed's: paraphrases land ~0.7-0.9, unrelated ~0.2, so
            # the base-embed-calibrated 0.7 default is a touch high. Drop to a GTE default unless the
            # operator set MINISGL_CAM_DELIVER_TAU explicitly.
            if "MINISGL_CAM_DELIVER_TAU" not in os.environ:
                self.deliver_tau = float(os.environ.get("MINISGL_CAM_GTE_DELIVER_TAU", "0.55"))
                self.protect_tau = self.deliver_tau if "MINISGL_CAM_PROTECT_TAU" not in os.environ else self.protect_tau
            # Semantic key is now live, so paraphrase re-remembers collide — enable write-side dedup at a
            # measured, silent-data-loss-safe threshold (above the ~0.79 distinct-subject ceiling).
            if not os.environ.get("MINISGL_CAM_WRITE_DEDUP_TAU", "").strip():
                self.write_dedup_tau = float(os.environ.get("MINISGL_CAM_GTE_WRITE_DEDUP_TAU", "0.82"))
            print(f"CAM: whitened-GTE subject key ACTIVE (model={model}, dim={int(W.shape[0])}, tau={self.deliver_tau}, "
                  f"whiten-fit n={art.get('n_fit')}, nn {art.get('nn_raw')}->{art.get('nn_whitened')})", flush=True)
        except Exception as e:  # noqa: BLE001 — degrade to base-embed key, don't crash
            self._gte = None
            print(f"CAM: whitened-GTE key requested but unavailable ({e}); using the base-embed key", flush=True)

    @torch.no_grad()
    def _gte_key(self, text: str) -> torch.Tensor:
        """Whitened-GTE subject key from TEXT: masked-mean-pooled GTE key -> (g-mu)@W -> L2-norm."""
        enc, mu, W = self._gte
        g = enc.encode([text or ""])[0]                      # [dim]
        return F.normalize((g - mu) @ W, dim=-1)

    def _subj_key(self, subject_ids: List[int]) -> torch.Tensor:
        """Subject key for the cosine index. Default: L2-normalised MEAN of the base input embeddings over
        the subject tokens (lexical, order/title/case robust). With MINISGL_CAM_GTE_KEY=1 (and a decoder
        wired): a whitened-GTE semantic key instead — decode the ids back to text and encode, so both stored
        keys and query keys live in the same whitened-GTE space and paraphrased subjects address correctly."""
        if self._gte is not None and self._decode is not None:
            return self._gte_key(self._decode(subject_ids))
        ids = torch.tensor([list(subject_ids)], dtype=torch.long, device=self._embed_w.device)
        e = F.embedding(ids, self._embed_w).float()          # [1,S,base_hidden] raw base input embeds
        return F.normalize(e.mean(1), dim=-1)[0]             # [base_hidden]

    def reindex(self) -> None:
        """Rebuild every namespace's cosine subject keys from its stored facts under the CURRENT _subj_key.
        Used after the GTE decoder is wired so a store loaded with base-embed keys migrates into the GTE
        key space (the object/fact tables are untouched — only the derived keys are recomputed)."""
        for st in self._ns_states.values():
            st.subj_keys[:] = [self._subj_key(list(t)) for t in st.subj_tuple]

    # ---- write gating (freeze / no-clobber) ------------------------------------------------------
    @torch.no_grad()
    def has_subject(self, subject_ids: List[int], tau: float = None, ns: str = None) -> bool:
        """True if a sufficiently-similar subject is already stored IN THIS NAMESPACE — exact id match, or
        cosine >= tau to an existing subject key. Used by the no-clobber policy to protect curated entries."""
        tau = self.protect_tau if tau is None else tau
        st = self._state(ns)
        if tuple(int(s) for s in subject_ids) in st.subj_tuple:
            return True
        if not st.subj_keys:
            return False
        q = self._subj_key(subject_ids)
        sims = torch.stack(st.subj_keys).to(q.device) @ q
        return bool(float(sims.max().item()) >= tau)

    def write_allowed(self, subject_ids: List[int], *, source: str = "auto", ns: str = None) -> bool:
        """Gate an incoming write. Explicit ingest (source='force') always writes — freeze/no-clobber only
        constrain AMBIENT auto-write (source='auto'): refused when the namespace is frozen, or (no-clobber
        policy) when the subject is already curated so the existing value is preserved."""
        if source == "force":
            return True
        if self._state(ns).frozen:
            return False
        if self.write_policy == "no-clobber" and self.has_subject(subject_ids, ns=ns):
            return False
        return True

    def freeze(self, ns: str = None) -> bool:
        self._state(ns).frozen = True
        return True

    def unfreeze(self, ns: str = None) -> bool:
        self._state(ns).frozen = False
        return False

    # ---- POINTER delivery (#100): exact object via cosine-NN over the subject index ------------------
    @torch.no_grad()
    def deliver_object_ids(self, subject_ids: List[int], ns: str = None) -> List[int]:
        """The exact object token sequence for a subject IN THIS NAMESPACE, via nearest stored subject by
        cosine (>= deliver_tau). Exact retrieval (no slot collision at scale) + paraphrase-robust; returns
        [] for an unknown subject (max-cos < tau) so /cam/ask falls back cleanly. The #100 serving unlock:
        memory supplies the unknowable object tokens, the base then continues the sentence."""
        st = self._state(ns)
        if not self.enabled or not st.subj_objs:
            return []
        q = self._subj_key(subject_ids)                      # [d]
        sims = torch.stack(st.subj_keys).to(q.device) @ q     # [M] cosine (keys + q are unit-norm)
        j = int(sims.argmax().item())
        if float(sims[j].item()) < self.deliver_tau:
            return []                                        # unknown subject -> no confident delivery
        rec = st.facts.get(st.subj_tuple[j])                 # #9 LRU: mark this fact recently used
        if rec is not None:
            st.seq += 1; rec["used"] = st.seq
        return list(st.subj_objs[j])

    @torch.no_grad()
    def deliver_object_ids_batch(self, subjects_ids: List[List[int]],
                                 ns: str = None) -> List[Optional[List[int]]]:
        """Batched deliver_object_ids for the transparent-read path (which queries MANY n-gram-span
        candidates per prompt). SEMANTICALLY IDENTICAL to calling deliver_object_ids on each candidate
        in the given order — same per-row argmax, same deliver_tau gate, same LRU bump order — but stacks
        the stored [M,d] key matrix ONCE and does a single [C,d]@[d,M] matmul instead of rebuilding the
        stack + matmul per candidate (the O(candidates×facts) cost that made big-prompt retrieves heavy
        under concurrency). Returns a list aligned to `subjects_ids`: object ids on a confident match
        (>= deliver_tau), else None."""
        st = self._state(ns)
        if not self.enabled or not st.subj_objs or not subjects_ids:
            return [None] * len(subjects_ids)
        K = torch.stack(st.subj_keys)                              # [M,d] (unit-norm keys), stacked ONCE
        Q = torch.stack([self._subj_key(s) for s in subjects_ids]).to(K.device)  # [C,d] (unit-norm)
        sims = Q @ K.t()                                           # [C,M] cosine
        out: List[Optional[List[int]]] = []
        for i in range(len(subjects_ids)):
            j = int(sims[i].argmax().item())
            if float(sims[i, j].item()) < self.deliver_tau:
                out.append(None)                                   # unknown subject -> no confident match
                continue
            rec = st.facts.get(st.subj_tuple[j])                   # #9 LRU: mark recently used (same as singular)
            if rec is not None:
                st.seq += 1; rec["used"] = st.seq
            out.append(list(st.subj_objs[j]))
        return out

    # ---- read (once per request, at prefill) -----------------------------------------------------
    @torch.no_grad()
    def read(self, subject_ids: List[int], ns: str = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """subject_ids -> (bank [1,K,mem], conf [1]). Returns (None, None) when disabled. Cheap B=1
        product-key read from THIS NAMESPACE's banks; run on the scheduler at prefill and reuse across the
        request's decode steps."""
        if not self.enabled:
            return None, None
        st = self._state(ns)
        dev = self.adapter.device
        tids = torch.tensor([list(subject_ids)], dtype=torch.long, device=dev)
        q = self.adapter._e(tids)
        if self.adapter.learned_key_pool:
            q = self.adapter._pool_subject(q, keepdim=True)  # symmetric with the pooled write key
        b = _subject_bank(list(subject_ids), self.n_banks)
        bank = self.adapter.persistent_bank(st.banks[b], q)         # [1,K,mem]
        conf = self.adapter._last_conf                               # [1] or None
        if conf is None:
            conf = torch.zeros(1, device=dev)
        return bank, conf

    # ---- tap (staged on the model, applied after tap_layer) --------------------------------------
    @torch.no_grad()
    def apply_tap(self, h: torch.Tensor, bank: Optional[torch.Tensor],
                  conf: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply the gated tap to the flat residual hidden h [num_tokens, H]. bank [1,K,mem] is broadcast
        across all token rows (MVP single-request semantics); conf [1] gates the whole injection. No-op
        (returns h unchanged) when bank is None. fp32 compute, additive cast back to h.dtype."""
        if not self.enabled or bank is None:
            return h
        h3 = h.unsqueeze(0)                                  # [1, N, H]
        out = self.tap.forward(h3, bank, conf)               # [1, N, H]
        return out.squeeze(0)

    @torch.no_grad()
    def apply_tap_rows(self, h: torch.Tensor, bank: torch.Tensor,
                       conf: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-ROW tap: h [N,H] with bank [N,K,mem] (one bank per token row) -> injected [N,H]. Used by
        the graph-capture decode path where a static [max_bs,K,mem] buffer carries a distinct bank per
        request row. The tap's forward is already batched over its leading dim, so treat each row as its
        own batch element (T=1): h[N,1,H] x bank[N,K,mem]. A zero bank row => tap no-op for that row
        (padding / non-memory / seed-once-placed). For N==1 this is byte-identical to apply_tap."""
        if not self.enabled:
            return h
        out = self.tap.forward(h.unsqueeze(1), bank, conf)   # [N,1,H]
        return out.squeeze(1)

    # ---- router (per-token logit-space injection, at the lm_head) ---------------------------------
    @torch.no_grad()
    def router_delta(self, base_last_logits: torch.Tensor, bank: Optional[torch.Tensor],
                     conf: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-token router-gated logit delta for the current decode step. Add it to `base_last_logits`
        (memory-OFF logits) at the lm_head. Returns zeros when disabled / no bank. The caller owns the
        seed-once policy (stop calling once the store's object token has landed)."""
        if not self.enabled or bank is None:
            return torch.zeros_like(base_last_logits)
        squeezed = base_last_logits.dim() == 1
        off = base_last_logits.float().reshape(1, -1) if squeezed else base_last_logits.float()
        lm = self.lm_head_weight
        raw = (self.adapter.out_proj(bank).mean(1).to(lm.device, lm.dtype) @ lm.t()).float()  # [1,V]
        if raw.shape[0] != off.shape[0]:
            raw = raw.expand(off.shape[0], -1)
        g2 = self.router.gain(signal_features(off, raw, conf))       # [B,2]
        delta = _inj_pertoken(raw, g2, self.router_alpha, self.router_topk)
        delta = delta.to(base_last_logits.dtype)
        return delta.reshape(-1) if squeezed else delta

    # ---- control-plane conveniences (WS-C edit API builds on these) ------------------------------
    def store_token_of(self, bank: torch.Tensor) -> int:
        """The store's argmax vocab token for a read bank (the 'seed' token for seed-once generation)."""
        lm = self.lm_head_weight
        raw = (self.adapter.out_proj(bank).mean(1).to(lm.device, lm.dtype) @ lm.t()).float()
        return int(raw.argmax(-1).reshape(-1)[0].item())

    def facts(self, ns: str = None) -> list:
        """List stored (subject_ids, object_ids) associations for a namespace (for /cam/facts)."""
        return [{"subject_ids": list(k), **v} for k, v in self._state(ns).facts.items()]

    def stats(self, ns: str = None) -> dict:
        """Per-bank occupancy + crowding health for a namespace's product-key VALUE banks. The value-bank
        read degrades past ~9 edits/bank — but that only affects the router/tap FALLBACK now; the primary
        /cam/ask delivery is the cosine-NN subject index (exact, collision-free)."""
        st = self._state(ns)
        loads = [0] * self.n_banks
        for sids in st.facts:
            loads[_subject_bank(list(sids), self.n_banks)] += 1
        total = sum(loads)
        mx = max(loads) if loads else 0
        mean = (total / self.n_banks) if self.n_banks else 0.0
        return {
            "B": self.n_banks, "total_edits": total, "max_bank_load": mx,
            "imbalance": (mx / mean) if mean else 0.0,
            "crowded_banks": [b for b, ln in enumerate(loads) if ln > 9],
            "banks": [{"index": b, "n_edits": ln} for b, ln in enumerate(loads) if ln > 0],
            "frozen": st.frozen, "write_policy": self.write_policy,
            "namespace": ns or "default", "namespaces": len(self._ns_states),
            "max_facts": self.max_facts, "evicted": st.evicted,            # #9 capacity
            "persistent": bool(self.store_path), "dirty": self._dirty,     # #7 persistence
        }

    def list_namespaces(self) -> list:
        """Enumerate every live namespace store with its fact count + freeze state (ops / test hygiene;
        spine feedback #4). 'default' always exists."""
        return [{"namespace": ns, "facts": self.stats(ns)["total_edits"],
                 "frozen": bool(self._state(ns).frozen)} for ns in list(self._ns_states)]

    def drop_namespace(self, ns: str) -> bool:
        """Delete a namespace's editable store (scratch/test cleanup; spine feedback #4). Refuses to drop
        'default'/None. Returns True if a store was removed. Marks dirty for the next autosave."""
        if not ns or ns == "default" or ns not in self._ns_states:
            return False
        del self._ns_states[ns]
        self._dirty = True
        self.autosave(force=True)   # durable: persist the drop so a restart can't resurrect the namespace
        return True

    def metrics_summary(self) -> dict:
        """Store totals across ALL namespaces for the Prometheus /metrics snapshot (cheap; iterates
        the per-namespace fact dicts). facts/evicted/crowded are summed; max_bank_load is the max."""
        facts = evicted = crowded = 0
        max_load = 0
        for ns in list(self._ns_states):
            s = self.stats(ns)
            facts += s["total_edits"]
            evicted += int(s.get("evicted", 0))
            crowded += len(s.get("crowded_banks", []))
            max_load = max(max_load, int(s["max_bank_load"]))
        return {"facts": facts, "namespaces": len(self._ns_states), "evicted": evicted,
                "max_bank_load": max_load, "crowded_banks": crowded}

    @torch.no_grad()
    def snapshot(self, path: str) -> int:
        """Persist the editable state of ALL namespaces to `path` (the trained adapter/tap/router live in
        the checkpoint, not here). Returns total #edits saved. (Cosine-NN index is rebuilt from facts on
        restore.) See #7 for lifecycle wiring."""
        # In pointer-only mode the value banks are unused (delivery is the cosine index, rebuilt from
        # facts on restore) — skip them: they are ~all of the snapshot's hundreds of MB, so this makes the
        # force-save-on-delete + debounced autosave cheap (store.pt -> KB). restore() re-inits empty banks.
        ns_blob = {ns: {"banks": (None if self.pointer_only else [b.detach().cpu() for b in st.banks]),
                        "facts": st.facts, "frozen": st.frozen} for ns, st in self._ns_states.items()}
        torch.save({"ns_states": ns_blob,
                    "meta": {"n_banks": self.n_banks, "k_slots": self.k_slots, "mem_dim": self.mem_dim,
                             "base_model": self.meta.get("base_model"),
                             "pointer_only": self.pointer_only}}, path)
        return sum(len(st.facts) for st in self._ns_states.values())

    @torch.no_grad()
    def restore(self, path: str) -> int:
        """Load a snapshot (all namespaces). Hard-fails on a bank-count mismatch. Rebuilds each namespace's
        cosine-NN delivery index from its facts. Returns total #edits."""
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = d.get("meta", {})
        if int(m.get("n_banks", self.n_banks)) != self.n_banks:
            raise ValueError(f"snapshot n_banks={m.get('n_banks')} != store n_banks={self.n_banks}")
        blob = d.get("ns_states")
        if blob is None:                                      # legacy single-store snapshot (banks+facts)
            blob = {"default": {"banks": d["banks"], "facts": d.get("facts", {}), "frozen": False}}
        self._ns_states = {}
        for ns, s in blob.items():
            _b = s.get("banks")                               # None -> pointer-mode slim snapshot: re-init
            banks = ([b.to(self.device, dtype=torch.float32) for b in _b] if _b is not None
                     else [self.adapter.store.init_state(1, self.device, dtype=torch.float32)
                           for _ in range(self.n_banks)])
            st = _NsState(banks, bool(s.get("frozen", False)))
            self._ns_states[ns] = st
            for k, rec in s.get("facts", {}).items():         # rebuild the cosine-NN index from facts
                key_vec = self._subj_key(list(k))
                st.subj_tuple.append(tuple(k)); st.subj_keys.append(key_vec)
                st.subj_objs.append(list(rec["object_ids"])); st.facts[tuple(k)] = rec
            st.seq = max([r.get("used", 0) for r in st.facts.values()] or [0])   # #9 continue the LRU clock
        return sum(len(st.facts) for st in self._ns_states.values())

    def save(self) -> int:
        """Force a persistence snapshot now (explicit flush, e.g. POST /cam/save). Returns #edits, or -1
        when no store_path is configured."""
        if not self.store_path:
            return -1
        n = self.snapshot(self.store_path)
        self._dirty = False; self._last_save = time.time()
        return n

    def reload(self) -> int:
        """#11 DP-scale: re-read the store from store_path to pick up writes made by ANOTHER replica that
        shares the same backing file — cheap eventual-consistency for a shared-backing DP deployment.
        Returns total #edits after reload, or -1 without a store file. (Strong consistency + concurrent
        writes need an external KV backend — see docs/zaya-port/CAM_DP_SCALE.md.)"""
        if not self.store_path or not os.path.isfile(self.store_path):
            return -1
        return self.restore(self.store_path)

    # --- WS-C API aliases (the edit-plane calls these exact names) ---
    def list_facts(self, ns: str = None) -> list:
        return self.facts(ns)

    def seed_token(self, bank: torch.Tensor, conf=None) -> int:
        return self.store_token_of(bank)

    def delete(self, subject_ids: List[int], ns: str = None) -> bool:
        return self.forget(subject_ids, ns=ns)

    @torch.no_grad()
    def reset(self, ns: str = None) -> None:
        """Re-init empty banks (drop all edits) for a namespace."""
        if not self.enabled:
            return
        st = self._state(ns)
        st.banks = [self.adapter.store.init_state(1, self.device, dtype=torch.float32)
                    for _ in range(self.n_banks)]
        st.subj_keys, st.subj_objs, st.subj_tuple = [], [], []
        st.facts = {}

    @torch.no_grad()
    def forget(self, subject_ids: List[int], ns: str = None) -> bool:
        """Remove one subject from a namespace: drop it from the delivery index (exact), and re-init its
        value bank + replay the OTHER facts routed to that bank (the store's delta write has no per-slot
        erase, so we rebuild the affected bank from the surviving edits — the router/tap fallback)."""
        if not self.enabled:
            return False
        st = self._state(ns)
        key = tuple(int(s) for s in subject_ids)
        if key not in st.facts:
            return False
        del st.facts[key]
        if key in st.subj_tuple:                              # drop from the cosine-NN delivery index
            i = st.subj_tuple.index(key)
            del st.subj_tuple[i]; del st.subj_keys[i]; del st.subj_objs[i]
        b = _subject_bank(list(subject_ids), self.n_banks)
        st.banks[b] = self.adapter.store.init_state(1, self.device, dtype=torch.float32)
        for subj, rec in list(st.facts.items()):
            if _subject_bank(list(subj), self.n_banks) == b:
                self._write(list(subj), rec["object_ids"], ns=ns)   # rebuild value bank; index in-place
        self._audit_add("forget", ns, key, [])                # #12 audit
        self._dirty = True                                    # #7 persistence
        self.autosave(force=True)                             # DURABLE DELETE (spine delete-resurrect): the
        # debounced autosave can miss a delete before a restart -> the fact resurrects from the last
        # snapshot on load-on-boot. Force an immediate persist so the deletion survives a restart with no
        # explicit /cam/save. No-op when persistence is off (store_path unset).
        return True

    @torch.no_grad()
    def undo(self, ns: str = None) -> dict:
        """#12: undo the most-recent WRITE in a namespace (forget that subject). Returns the undone
        {subject_ids, object_ids} or {} if there is nothing to undo."""
        st = self._state(ns)
        k = st.last_key
        if not k or k not in st.facts:
            return {}
        obj = list(st.facts[k]["object_ids"])
        self.forget(list(k), ns=ns)
        st.last_key = None
        return {"subject_ids": list(k), "object_ids": obj}

    @torch.no_grad()
    def rebuild(self, ns: str = None) -> int:
        """#12 TRUE ERASE / compaction: fully re-init a namespace's banks and replay only the surviving
        facts, discarding all delta residue from forgotten/overwritten edits. Returns #facts replayed."""
        if not self.enabled:
            return 0
        st = self._state(ns)
        survivors = list(st.facts.items())
        st.banks = [self.adapter.store.init_state(1, self.device, dtype=torch.float32)
                    for _ in range(self.n_banks)]
        st.subj_keys, st.subj_objs, st.subj_tuple, st.facts = [], [], [], {}
        for k, rec in survivors:
            self._write(list(k), rec["object_ids"], ns=ns, base_p=rec.get("base_p", 0.0))
        self._dirty = True
        return len(survivors)
