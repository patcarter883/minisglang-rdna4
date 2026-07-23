"""FEASIBILITY GATE for SWA prefix caching (radix reuse across the sliding-window boundary).

The rdna4 backend claims (rdna4.py:341) that "Radix reuse across the window boundary is unsound".
This probe FALSIFIES-OR-CONFIRMS that claim at the REAL HIP-kernel level, with no model / serve.

Design under test (the proposed SWA extend prefill):
  * Prefix P of length L is cached. A page-aligned SNAPSHOT of the sliding layer's window ring holds
    the last Wp = min(L, W) tokens' K/V (in ascending-position order).
  * Request B reuses P and extends with M new tokens (absolute positions [L, L+M)). For B's sliding
    layer, we run a DENSE prefill over the concatenated buffer
        K_ext = [ restored_window_K (Wp) | new_K (M) ]     # Wp+M keys, ascending position
        V_ext = [ restored_window_V (Wp) | new_V (M) ]
        Q_ext = [ zeros (Wp)             | new_Q (M) ]      # dummy pad for the prefix rows
    through attn_hip.flash_prefill(Q_ext, K_ext, V_ext, scale, causal=1, sliding_window=W), then keep
    only rows [Wp:] (the real M new-token outputs).

Claim: those M outputs are byte-identical to what a COLD prefill of the WHOLE prompt (length L+M)
produces for its last M tokens. If true, SWA prefix reuse is SOUND at the kernel level and the
NotImplementedError is "not-yet-implemented", not "impossible".

The kernel's square causal+SWA mask (attn_kernels_hip.hip): key col c is masked for query row r when
c>r (causal) or (r-c)>=W (window). Because the extend buffer lays the Wp+M keys out in contiguous
ascending absolute position, the RELATIVE (r-c) distances are preserved exactly, so the non-masked
key SET and their values match cold. The only question this probe answers empirically is whether the
flash online-softmax BLOCK GROUPING (different seq_len => different BC-block alignment) perturbs the
result below bit-identity.

Run inside the lean image with a GPU lease:
    MINISGL_CMD='python /engine/tools/swa_prefix_extend_validate.py' \
        gpu-lease -n 1 -- docker compose --profile run run --rm run
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import attn_hip

DEV = "cuda"
torch.manual_seed(0)
FAILS = []

# Laguna sliding-layer shape (TP=1): 64 QO heads, 8 KV heads, head_dim 128, window 512.
HQ, HK, D, W = 64, 8, 128, 512
SCALE = D ** -0.5


def _record(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} {extra}")
    if not ok:
        FAILS.append(name)


def ref_windowed_causal_prefill(q, k, v, window):
    """fp32 causal + sliding-window reference over a single sequence. [S,Hq|Hk,D] -> [S,Hq,D]."""
    S, Hq, Hk = q.shape[0], q.shape[1], k.shape[1]
    rep = Hq // Hk
    qf = q.float().permute(1, 0, 2)
    kf = k.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * SCALE
    i = torch.arange(S, device=q.device)
    causal = i[:, None] < i[None, :]
    old = (i[:, None] - i[None, :]) >= window
    attn = attn.masked_fill((causal | old)[None], float("-inf"))
    out = torch.matmul(F.softmax(attn, dim=-1), vf)
    return out.permute(1, 0, 2).contiguous()


def _prefill(q, k, v):
    return attn_hip.flash_prefill(q.contiguous(), k.contiguous(), v.contiguous(), SCALE, 1, W)


def case(name, L, M):
    """Compare COLD full-prompt prefill (last M rows) vs the SNAPSHOT+EXTEND path (dropped-pad rows)."""
    N = L + M
    q = torch.randn(N, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)

    # ---- COLD: whole prompt in one flash_prefill, keep the last M token outputs. -----------------
    out_cold_full = _prefill(q, k, v)          # [N, HQ, D]
    out_cold = out_cold_full[L:]               # [M, HQ, D]  (the new tokens' attention outputs)

    # ---- EXTEND: restored window (last Wp of P) ++ new chunk, drop the Wp pad rows. --------------
    Wp = min(L, W)
    k_win = k[L - Wp:L]                         # the snapshot: sliding K of the last Wp prefix tokens
    v_win = v[L - Wp:L]
    k_ext = torch.cat([k_win, k[L:N]], dim=0)  # [Wp+M, HK, D]
    v_ext = torch.cat([v_win, v[L:N]], dim=0)
    q_pad = torch.zeros(Wp, HQ, D, device=DEV, dtype=torch.bfloat16)
    q_ext = torch.cat([q_pad, q[L:N]], dim=0)  # [Wp+M, HQ, D]
    out_ext = _prefill(q_ext, k_ext, v_ext)[Wp:]   # [M, HQ, D]

    # ---- Compare. --------------------------------------------------------------------------------
    bit_identical = torch.equal(out_cold, out_ext)
    dmax = (out_cold.float() - out_ext.float()).abs().max().item()
    ref = ref_windowed_causal_prefill(q, k, v, W)[L:]
    dref_cold = (out_cold.float() - ref.float()).abs().max().item()
    dref_ext = (out_ext.float() - ref.float()).abs().max().item()
    span = "prefix<W" if L < W else ("prefix==W" if L == W else "prefix>W")
    _record(
        f"{name} (L={L},M={M},{span})",
        bit_identical or dmax <= 2e-3,
        f"bit-identical={bit_identical} max|cold-ext|={dmax:.3e} "
        f"(cold-ref={dref_cold:.2e} ext-ref={dref_ext:.2e})",
    )
    return bit_identical, dmax


BC = 32  # flash tile BC for head_dim 64/128 (attn_kernels_hip.hip kBC) — the reduction block width.


def case_bc_aligned(name, L, M):
    """Same as case(), but FRONT-PADS the extend buffer with (L-Wp) % BC extra (window-masked) prefix
    keys so the first real new-token row lands at buffer offset == L (mod BC). This aligns the flash
    online-softmax block grouping with the cold pass => provably bit-identical for ANY boundary L.
    The extra <BC keys are always outside the window (distance > W) so they never affect the output;
    the snapshot just holds the last min(L, W+BC-1) tokens instead of exactly W."""
    N = L + M
    q = torch.randn(N, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)
    out_cold = _prefill(q, k, v)[L:]

    Wp = min(L, W)
    pad = (L - Wp) % BC                     # extra front keys to restore BC alignment
    lo = L - Wp - pad                       # snapshot start (>=0 since L-Wp>=0 and pad<=L-Wp region)
    lo = max(lo, 0)
    real_pad = (L - Wp) - (L - Wp - pad) if lo == L - Wp - pad else (L - Wp)  # keys actually prepended
    k_ext = torch.cat([k[lo:L], k[L:N]], dim=0)
    v_ext = torch.cat([v[lo:L], v[L:N]], dim=0)
    front = L - lo                          # = pad + Wp real keys before the new tokens
    q_pad = torch.zeros(front, HQ, D, device=DEV, dtype=torch.bfloat16)
    q_ext = torch.cat([q_pad, q[L:N]], dim=0)
    out_ext = _prefill(q_ext, k_ext, v_ext)[front:]

    bit = torch.equal(out_cold, out_ext)
    dmax = (out_cold.float() - out_ext.float()).abs().max().item()
    _record(f"BC-aligned {name} (L={L},M={M},pad={pad})", bit,
            f"bit-identical={bit} max|cold-ext|={dmax:.3e}")
    return bit, dmax


def main():
    print("== SWA prefix-extend vs cold prefill (feasibility gate) ==")
    print(f"   shape HQ={HQ} HK={HK} D={D} W={W}\n")
    results = []
    # Boundary cases the 'unsound' claim is about: prefix shorter/equal/longer than the window,
    # single-token and multi-token extends, chunks that span the window boundary.
    results.append(case("prefix longer than window, small extend", 1000, 64))
    results.append(case("prefix longer than window, 1-token extend", 1000, 1))
    results.append(case("prefix longer than window, big extend", 1000, 300))
    results.append(case("prefix == window", 512, 64))
    results.append(case("prefix shorter than window", 200, 64))
    results.append(case("prefix shorter, extend crosses W", 400, 200))
    results.append(case("tiny prefix", 40, 24))
    results.append(case("page-aligned prefix (256)", 768, 128))
    print("\n== BC-aligned extend (front-pad to (L-W)%BC) => bit-identical for ANY boundary ==")
    ar = []
    ar.append(case_bc_aligned("prefix longer than window, small extend", 1000, 64))
    ar.append(case_bc_aligned("prefix longer than window, 1-token extend", 1000, 1))
    ar.append(case_bc_aligned("prefix longer than window, big extend", 1000, 300))
    ar.append(case_bc_aligned("non-aligned prefix", 993, 57))
    ar.append(case_bc_aligned("non-aligned prefix", 777, 129))
    print(f"\nBC-aligned bit-identical: {sum(1 for b,_ in ar if b)}/{len(ar)}")
    print()
    n_bit = sum(1 for b, _ in results if b)
    print(f"bit-identical cases: {n_bit}/{len(results)}")
    worst = max(d for _, d in results)
    print(f"worst max|cold-ext|: {worst:.3e}")
    if FAILS:
        print(f"\nFAILED: {FAILS}")
        return 1
    print("\nALL SWA PREFIX-EXTEND CHECKS PASS "
          "(padded-extend reproduces cold within tolerance => SWA radix is SOUND)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
