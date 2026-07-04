"""GPU parity for the NEW mask_bias arg on attn_prefill_paged (C.1 — the TiDAR fused path).

Same paged setup as attn_prefill_paged_parity.py, but drives the kernel with causal=0 + an explicit
[total_q, max_kv] additive mask (0 allowed / -inf denied) and checks vs an fp32 SDPA reference using
the SAME allow matrix. Three masks:
  (1) causal-via-mask   : mask ENCODES causal -> must match the causal kernel/SDPA (proves the
                          additive path reproduces causal; the single most important gate).
  (2) bidirectional     : allow everything (no causal) -> tests denial-free non-causal.
  (3) tidar-block       : prefix causal + new tokens see all prefix + bidir within the new block ->
                          the shape the fused TiDAR mask uses.
Backward-compat: the existing parity harness calls the op WITHOUT mask_bias (default None) — run it
too to confirm the causal path is unchanged.

Run inside vllm22-w4a8:combined under a 1-card lease.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import op  # noqa: F401  loads the .so + registers torch.ops.attn_prefill_paged.*

DEV = "cuda"
torch.manual_seed(0)
COS_MIN = 0.9995
ULP_RTOL = 0.016
ULP_ATOL = 0.004
NEG = float("-inf")


def _build_paged(specs, Hq, Hk, D, block_size=16):
    S = len(specs)
    ctx = [p + q for p, q in specs]
    qlens = [q for _, q in specs]
    total_q = sum(qlens)
    maxctx = max(ctx)
    kd = torch.randn(S, maxctx, Hk, D, device=DEV, dtype=torch.bfloat16)
    vd = torch.randn(S, maxctx, Hk, D, device=DEV, dtype=torch.bfloat16)
    q = torch.randn(total_q, Hq, D, device=DEV, dtype=torch.bfloat16)
    bps = (maxctx + block_size - 1) // block_size
    num_blocks = S * bps + 3
    perm = torch.randperm(num_blocks, device=DEV).int()
    k_cache = torch.zeros(num_blocks, block_size, Hk, D, device=DEV, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros(S, bps, device=DEV, dtype=torch.int32)
    for b in range(S):
        for lb in range(bps):
            phys = int(perm[b * bps + lb].item())
            block_table[b, lb] = phys
            for off in range(block_size):
                j = lb * block_size + off
                if j < ctx[b]:
                    k_cache[phys, off] = kd[b, j]
                    v_cache[phys, off] = vd[b, j]
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0).tolist()), device=DEV, dtype=torch.int32)
    ctxt = torch.tensor(ctx, device=DEV, dtype=torch.int32)
    return dict(kd=kd, vd=vd, q=q, k_cache=k_cache, v_cache=v_cache, block_table=block_table,
                cu=cu, ctxt=ctxt, ctx=ctx, qlens=qlens, total_q=total_q, maxctx=maxctx, rep=Hq // Hk)


def _allow(kind, p, ql, cl):
    """[ql, cl] bool allow for one seq: query local i (global pos p+i) attends key j (0..cl-1)."""
    qpos = p + torch.arange(ql, device=DEV)
    kpos = torch.arange(cl, device=DEV)
    if kind == "causal":
        return kpos[None, :] <= qpos[:, None]
    if kind == "bidir":
        return torch.ones(ql, cl, dtype=torch.bool, device=DEV)
    if kind == "tidar":
        # prefix keys [0,p): always allowed. new keys [p,cl): bidirectional among the new block.
        a = torch.zeros(ql, cl, dtype=torch.bool, device=DEV)
        a[:, :p] = True
        a[:, p:] = True  # new block bidirectional (i attends all new j)
        return a
    raise ValueError(kind)


def check_mask(name, specs, Hq, Hk, D, kind, block_size=16) -> bool:
    scale = D ** -0.5
    g = _build_paged(specs, Hq, Hk, D, block_size)
    max_kv = g["maxctx"]
    mask_bias = torch.zeros(g["total_q"], max_kv, device=DEV, dtype=torch.float32)
    for b, (p, ql) in enumerate(specs):
        cl = g["ctx"][b]
        off = int(g["cu"][b].item())
        allow = _allow(kind, p, ql, cl)
        mask_bias[off:off + ql, :cl] = torch.where(allow, 0.0, NEG)

    got = torch.ops.attn_prefill_paged.flash_prefill_paged(
        g["q"], g["k_cache"], g["v_cache"], g["block_table"], g["cu"], g["ctxt"],
        scale, 0, 0, max(g["qlens"]), 0, mask_bias).float()  # causal=0, mask carries everything

    ref = torch.empty(g["total_q"], Hq, D, device=DEV)
    for b, (p, ql) in enumerate(specs):
        cl = g["ctx"][b]
        off = int(g["cu"][b].item())
        kbe = g["kd"][b, :cl].float().repeat_interleave(g["rep"], dim=1)
        vbe = g["vd"][b, :cl].float().repeat_interleave(g["rep"], dim=1)
        qi = g["q"][off:off + ql].float()
        scores = torch.einsum("qhd,khd->qhk", qi, kbe) * scale
        allow = _allow(kind, p, ql, cl)
        scores = scores.masked_fill(~allow[:, None, :], NEG)
        ref[off:off + ql] = torch.einsum("qhk,khd->qhd", F.softmax(scores, dim=-1), vbe)

    ref_b = ref.bfloat16().float()
    d = (got - ref_b).abs().max().item()
    viol = ((got - ref_b).abs() - (ULP_ATOL + ULP_RTOL * ref_b.abs())).clamp(min=0).max().item()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    ok = (cos >= COS_MIN) and (viol <= 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} cos={cos:.6f}  max|Δ|={d:.3e}  ulp_viol={viol:.2e}")
    return ok


def main() -> None:
    print("=== attn_prefill_paged mask_bias parity (causal=0 + additive mask vs SDPA) ===")
    ok = True
    print("--- (1) causal-via-mask : additive mask must reproduce causal ---")
    ok &= check_mask("causal-mask cold prefix0 q64", [(0, 64)], 16, 2, 128, "causal")
    ok &= check_mask("causal-mask extend p100 q32", [(100, 32)], 16, 2, 128, "causal")
    ok &= check_mask("causal-mask batch mixed", [(0, 16), (50, 40), (128, 8)], 16, 2, 128, "causal")
    ok &= check_mask("causal-mask D256 p100 q32", [(100, 32)], 16, 2, 256, "causal")
    print("--- (2) bidirectional : full attention (no causal) ---")
    ok &= check_mask("bidir cold prefix0 q48", [(0, 48)], 16, 2, 128, "bidir")
    ok &= check_mask("bidir extend p64 q32", [(64, 32)], 16, 2, 128, "bidir")
    print("--- (3) tidar-block : prefix causal-free + new block bidir ---")
    ok &= check_mask("tidar-block p100 q20 (B=4)", [(100, 20)], 16, 2, 128, "tidar")
    ok &= check_mask("tidar-block batch", [(50, 20), (128, 20)], 16, 2, 128, "tidar")
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
