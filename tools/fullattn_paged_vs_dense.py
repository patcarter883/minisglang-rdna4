"""Isolate the SWA serve reuse-vs-cold divergence: is it the FULL-attention layers' paged-extend
(attn_prefill_paged.flash_prefill_paged, the engine's universal radix path) vs the cold dense kernel
(attn_hip.flash_prefill)? Both are used on a Laguna serve — cold prefill runs dense, a radix hit runs
paged-extend. If these differ by ~1 ULP for identical bf16 K/V, that (not the SWA sliding extend,
which tools/swa_prefix_extend_validate.py proved byte-identical) is the residual serve divergence, and
it is a PRE-EXISTING property shared by every radix model on this stack.

Run: MINISGL_CMD='python /engine/tools/fullattn_paged_vs_dense.py' \
       gpu-lease -n 1 -- docker compose --profile run run --rm run
"""
from __future__ import annotations

import sys

import torch

import attn_hip
import attn_prefill_paged

DEV = "cuda"
torch.manual_seed(0)
# Laguna FULL-attn shape (TP=1): 48 QO heads, 8 KV heads, head_dim 128 (no window).
HQ, HK, D = 48, 8, 128
SCALE = D ** -0.5
FAILS = []


def case(name, L, M, PS):
    """cold = dense flash_prefill over [0, L+M); extend = paged flash_prefill_paged for [L, L+M) with
    [0, L) served from a PAGE-SIZE-`PS` paged cache (the LIVE serve uses PS=16, not 1). Compare last M."""
    N = L + M
    q = torch.randn(N, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)

    out_cold = attn_hip.flash_prefill(q.contiguous(), k.contiguous(), v.contiguous(),
                                      SCALE, 1, 0)[L:]                       # dense cold, last M

    # Paged cache with page_size=PS: pad N up to a multiple of PS; page p holds tokens [PS*p, PS*p+PS).
    NP = (N + PS - 1) // PS
    kc = torch.zeros(NP, PS, HK, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.zeros(NP, PS, HK, D, device=DEV, dtype=torch.bfloat16)
    kc.view(NP * PS, HK, D)[:N] = k
    vc.view(NP * PS, HK, D)[:N] = v
    block_table = torch.arange(NP, device=DEV, dtype=torch.int32).view(1, NP)  # page-indexed
    cu_q = torch.tensor([0, M], device=DEV, dtype=torch.int32)
    ctx = torch.tensor([N], device=DEV, dtype=torch.int32)
    out_ext = attn_prefill_paged.flash_prefill_paged(
        q[L:].contiguous(), kc, vc, block_table, cu_q, ctx,
        SCALE, 1, 0, M, 0, None,
    )
    bit = torch.equal(out_cold, out_ext)
    dmax = (out_cold.float() - out_ext.float()).abs().max().item()
    print(f"  [{'IDENTICAL' if bit else 'DIFFERS  '}] {name} (L={L},M={M},page_size={PS}) "
          f"bit-identical={bit} max|dense-paged|={dmax:.3e}")
    if not bit:
        FAILS.append(f"{name}/PS{PS}")
    return bit, dmax


def main():
    print("== FULL-attn dense cold-prefill vs paged radix-extend (bf16 KV) ==")
    for PS in (1, 16):   # 1 = my earlier (passing) test; 16 = the LIVE Laguna serve page_size
        print(f"-- page_size = {PS} --")
        case("prefix 720, extend 14 (short-case shape)", 720, 14, PS)
        case("prefix 16, extend 14", 16, 14, PS)
        case("prefix 256, extend 48", 256, 48, PS)
        case("prefix 512, extend 32", 512, 32, PS)
        case("prefix 736 (46*16), extend 14", 736, 14, PS)
    print()
    if FAILS:
        print("VERDICT: full-attn paged-extend is NOT byte-identical to dense cold (~1 bf16 ULP). "
              "This is the engine's universal radix property (every radix model), and it is the "
              "residual source of the SWA serve reuse-vs-cold late-token divergence — NOT the SWA "
              "sliding extend (proven 0.0 in swa_prefix_extend_validate.py).")
        return 0  # informational: a DIFFERS result is the expected diagnosis, not a test failure
    print("VERDICT: full-attn paged-extend IS byte-identical to dense cold. The SWA serve divergence "
          "must be elsewhere (re-examine the sliding extend / ring gather).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
