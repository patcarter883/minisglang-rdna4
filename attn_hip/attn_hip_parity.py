"""Numeric parity for the native attn_hip flash-prefill kernel (GPU).

Checks torch.ops.attn_hip.flash_prefill against a pure-torch reference of the EXACT math it
implements (scaled dot-product attention with GQA + causal + optional sliding window). The kernel
accumulates QK^T and P@V in fp32, so a faithful kernel matches the fp32 reference to a few e-3 in
bf16; a real indexing/fragment-layout/mask bug shows as a large max|Δ| (the classic gfx11->gfx12
WMMA accumulator-layout trap = whole rows or columns wrong).

Run inside the combined ROCm image UNDER a 1-card lease (executes HIP/rocwmma kernels):
    scripts/gpu-lease.sh -n 1 -- bash -c 'docker run --rm \
      -v <repo>:/engine -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
      -v <repo>/.triton-cache-combined:/root/.triton \
      --entrypoint bash vllm22-w4a8:combined -lc \
      "source /app/.venv/bin/activate && cd /engine/attn_hip && \
       GPU_ARCHS=gfx1201 python setup.py build_ext --inplace && python attn_hip_parity.py"'
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import attn_hip  # loads the .so + registers torch.ops.attn_hip.*

DEV = "cuda"
torch.manual_seed(0)


def ref_attention(q, k, v, scale, causal, sliding_window):
    """q:[S,Hq,D] k/v:[S,Hk,D] -> [S,Hq,D]. fp32 reference; GQA via head repeat."""
    S, Hq, D = q.shape
    Hk = k.shape[1]
    rep = Hq // Hk
    qf = q.float().permute(1, 0, 2)                       # [Hq,S,D]
    kf = k.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # [Hq,S,S]
    i = torch.arange(S, device=q.device)
    if causal:
        mask = i[:, None] < i[None, :]                     # key > query -> -inf
        if sliding_window > 0:
            mask = mask | ((i[:, None] - i[None, :]) >= sliding_window)
        attn = attn.masked_fill(mask[None], float("-inf"))
    out = torch.matmul(F.softmax(attn, dim=-1), vf)        # [Hq,S,D]
    return out.permute(1, 0, 2).contiguous()               # [S,Hq,D]


# Pass criteria for a bf16 kernel judged against an fp32 reference.
#   The kernel is fp32-INTERNAL (QK^T/softmax/P@V all accumulate in fp32 — verified by substituting
#   scalar-fp32 QK^T, fp32 P, and scalar-fp32 P@V one at a time: each leaves the result unchanged),
#   but it RETURNS bf16. bf16's rounding is RELATIVE, so the raw max|Δ| vs an fp32 SDPA reference is
#   dominated by the unavoidable bf16 rounding of the OUTPUT — ~4e-3 on |out|~1 rows, larger on the
#   peaked early causal rows where attention mass concentrates on 1-3 keys and |out| can reach ~2.
#   So we DON'T compare to raw fp32 (too strict for any bf16 kernel — even a perfect one fails it).
#   Two correct gates instead:
#     (1) cosine-sim vs fp32 — the engine's standing oracle; a real layout/mask bug tanks it. ~0.99999.
#     (2) max|Δ| vs the fp32 reference ROUNDED TO bf16 — the kernel's error WITHIN bf16 representability
#         (residual = bf16 WMMA score/P rounding), the fair bar for a bf16-output kernel.
#
# The (2) bound MUST be bf16-ULP-relative, not a flat absolute: bf16 has an 8-bit significand, so its
# ULP scales with magnitude — at |out|~2 (peaked early causal rows where attention mass concentrates
# on 1-3 keys) one ULP is already 1.56e-2, while at |out|~0.3 it is ~2e-3. A flat 5e-3 is therefore
# sub-ULP-strict above |out|~0.6 and FALSELY fails a correct kernel on causal rows (verified: the
# unchanged head_dim 64/128 path fails the flat bar identically). The fair bar is a few bf16 ULP at
# the batch's peak output magnitude, with a small absolute floor for low-magnitude (non-causal) cases.
COS_MIN = 0.9995
ULP_TOL = 2.0       # allow 2 bf16 ULP at the peak |out| (intermediate P->bf16 + WMMA + output rounding)
DELTA_FLOOR = 5e-3  # absolute floor for small-magnitude outputs


def _bf16_ulp(mag: float) -> float:
    """One bf16 ULP at magnitude `mag` (8-bit significand -> 2^(exp-7))."""
    if mag <= 0.0:
        return 2.0 ** -7
    return 2.0 ** (math.floor(math.log2(mag)) - 7)


def check(name, S, Hq, Hk, D, causal=1, sw=0) -> bool:
    scale = D ** -0.5
    q = torch.randn(S, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(S, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(S, Hk, D, device=DEV, dtype=torch.bfloat16)
    got = torch.ops.attn_hip.flash_prefill(q, k, v, scale, causal, sw).float()
    ref = ref_attention(q, k, v, scale, causal, sw)
    ref_b = ref.bfloat16().float()                       # the best a bf16 output could represent
    d_fp32 = (got - ref).abs().max().item()              # info: dominated by bf16 output rounding
    d_b = (got - ref_b).abs().max().item()               # fair: error within bf16 representability
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    delta_bar = max(DELTA_FLOOR, ULP_TOL * _bf16_ulp(ref_b.abs().max().item()))
    ok = (cos >= COS_MIN) and (d_b <= delta_bar)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} cos={cos:.6f}  "
          f"max|Δ|bf16={d_b:.3e}  (bar={delta_bar:.3e}, vs-fp32={d_fp32:.3e})")
    return ok


def main() -> None:
    print("=== attn_hip flash_prefill parity (cos vs fp32 + max|Δ| vs bf16-rounded ref) ===")
    ok = True
    # Geometry close to Qwen3.5/3.6 attention layers (head_dim 128, GQA).
    ok &= check("causal D128 S64  Hq16/Hk2", 64, 16, 2, 128)
    ok &= check("causal D128 S128 Hq16/Hk2", 128, 16, 2, 128)
    ok &= check("causal D128 S100 (ragged) ", 100, 16, 2, 128)   # non-multiple of BR/BC
    ok &= check("noncausal D128 S96 Hq8/Hk8", 96, 8, 8, 128, causal=0)
    ok &= check("SWA=64 D128 S160 Hq16/Hk2 ", 160, 16, 2, 128, sw=64)
    ok &= check("causal D64  S128 Hq8/Hk1  ", 128, 8, 1, 64)
    # head_dim 256 (Qwen3.6 full-attn) — BR=BC=16 tiling so the fp32 smem fits gfx1201 LDS.
    ok &= check("causal D256 S64  Hq16/Hk2", 64, 16, 2, 256)
    ok &= check("causal D256 S128 Hq16/Hk2", 128, 16, 2, 256)
    ok &= check("causal D256 S100 (ragged) ", 100, 16, 2, 256)   # non-multiple of BR/BC
    ok &= check("noncausal D256 S96 Hq8/Hk8", 96, 8, 8, 256, causal=0)
    ok &= check("SWA=64 D256 S160 Hq16/Hk2 ", 160, 16, 2, 256, sw=64)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
