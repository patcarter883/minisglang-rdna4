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

# GEOMETRY IS A PARAMETER, not a constant. The defaults below are Laguna's sliding layer (TP=1) —
# the shape this gate was originally written against, so an unparameterised run is unchanged. But the
# whole point of the gate is the flash BLOCK GROUPING, which depends on head_dim and window, and
# Gemma4's sliding layers are NOT Laguna's: 256/8 at W=1024 (per rank at TP=2: 8 QO, 4 KV), against
# a 512/2 main pool. Hardcoding one model's numbers would have "proved" losslessness for a shape the
# serve never runs. Override with SWA_HQ / SWA_HK / SWA_D / SWA_W.
import os as _os
HQ = int(_os.environ.get("SWA_HQ", 64))
HK = int(_os.environ.get("SWA_HK", 8))
D = int(_os.environ.get("SWA_D", 128))
W = int(_os.environ.get("SWA_W", 512))
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


def case_production(name, L, M, expect_bit=True):
    """The SHAPE THE SERVE ACTUALLY RUNS: `rdna4.py::_swa_prefill_extend`, reproduced line for line.

    `case_bc_aligned` above is NOT this. It front-pads with REAL prefix keys taken from further back;
    production front-pads with ZEROS, and skips the pad entirely when `pad == 0`. Those are different
    buffers, and only one of them ships. The gap mattered: every BC-aligned case above lands on
    pad in {2, 16, 18} — the residue is a property of the (L, W) pair the case happens to pick — so
    `pad == 0` was never exercised, and `pad == 0` is exactly what a page-aligned radix boundary
    produces (L=3168, W=1024 -> (3168-1024) % 32 == 0). A serve hit it on the first partial prefix
    reuse and the gate had nothing to say."""
    N = L + M
    q = torch.randn(N, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(N, HK, D, device=DEV, dtype=torch.bfloat16)
    out_cold = _prefill(q, k, v)[L:]

    Wp = min(L, W)
    pad = (L - Wp) % BC
    k_win, v_win = k[L - Wp:L], v[L - Wp:L]
    Hk = k_win.shape[1]
    front = pad + Wp
    # EXACTLY rdna4.py: zero KEYS for the pad, zero QUERIES for the whole front, drop `front` rows.
    parts_k = ([k_win.new_zeros((pad, Hk, D))] if pad else []) + [k_win, k[L:N]]
    parts_v = ([v_win.new_zeros((pad, Hk, D))] if pad else []) + [v_win, v[L:N]]
    k_ext = torch.cat(parts_k, dim=0)
    v_ext = torch.cat(parts_v, dim=0)
    q_ext = torch.cat([q.new_zeros((front, HQ, D)), q[L:N]], dim=0)
    out_ext = _prefill(q_ext, k_ext, v_ext)[front:]

    bit = torch.equal(out_cold, out_ext)
    dmax = (out_cold.float() - out_ext.float()).abs().max().item()
    _record(f"PROD {name} (L={L},M={M},pad={pad})", bit == expect_bit,
            f"bit-identical={bit} max|cold-ext|={dmax:.3e}"
            + ("" if bit == expect_bit else f"  <-- EXPECTED bit-identical={expect_bit}"))
    return bit, dmax, pad


def sweep_pad(M):
    """Every BC residue at a fixed chunk length. The failure this gate exists to catch is a property
    of `pad`, so sweeping it is the only way to find out WHICH residues are sound rather than
    inferring soundness from whichever residues the round numbers happened to produce."""
    print(f"\n-- pad sweep at M={M} (L chosen so (L-W) % BC walks 0..{BC - 1}) --")
    bad = []
    for r in range(BC):
        L = W + 2 * BC * 8 + r          # L-Wp = 512+r on any W, so the residue is exactly r
        bit, dmax, pad = case_production(f"pad={r:2d}", L, M, expect_bit=True)
        if not bit:
            bad.append((pad, dmax))
    print(f"   pads that are NOT bit-identical: {[p for p, _ in bad] or 'none'}")
    return bad


def sweep_chunk(pad_target=0):
    """Chunk lengths at a fixed pad. A partial prefix hit produces a SHORT first chunk (the page
    remainder), which the round-number cases above never produced either."""
    print(f"\n-- chunk sweep at pad={pad_target} --")
    bad = []
    for M in (1, 2, 5, 7, 16, 17, 31, 32, 33, 64, 128):
        L = W + 2 * BC * 8 + pad_target
        bit, dmax, _ = case_production(f"M={M:3d}", L, M, expect_bit=True)
        if not bit:
            bad.append((M, dmax))
    print(f"   chunk lengths that are NOT bit-identical: {[m for m, _ in bad] or 'none'}")
    return bad


def main():
    print("== SWA prefix-extend vs cold prefill (feasibility gate) ==")
    print(f"   shape HQ={HQ} HK={HK} D={D} W={W}\n")
    results = []
    # The cases are stated RELATIVE TO THE WINDOW, not in absolute tokens: "prefix longer than the
    # window" is the thing under test, and at Laguna's W=512 a fixed 1000 means that while at Gemma4's
    # W=1024 it would silently mean the opposite. S keeps the W=512 numbers bit-for-bit what they were.
    S = max(1, W // 512)
    results.append(case("prefix longer than window, small extend", 1000 * S, 64))
    results.append(case("prefix longer than window, 1-token extend", 1000 * S, 1))
    results.append(case("prefix longer than window, big extend", 1000 * S, 300))
    results.append(case("prefix == window", W, 64))
    results.append(case("prefix shorter than window", 200 * S, 64))
    results.append(case("prefix shorter, extend crosses W", 400 * S, 200 * S))
    results.append(case("tiny prefix", 40, 24))
    results.append(case("page-aligned prefix", 768 * S, 128))
    print("\n== BC-aligned extend (front-pad to (L-W)%BC) => bit-identical for ANY boundary ==")
    ar = []
    ar.append(case_bc_aligned("prefix longer than window, small extend", 1000 * S, 64))
    ar.append(case_bc_aligned("prefix longer than window, 1-token extend", 1000 * S, 1))
    ar.append(case_bc_aligned("prefix longer than window, big extend", 1000 * S, 300))
    ar.append(case_bc_aligned("non-aligned prefix", 993 * S, 57))
    ar.append(case_bc_aligned("non-aligned prefix", 777 * S, 129))
    print(f"\nBC-aligned bit-identical: {sum(1 for b,_ in ar if b)}/{len(ar)}")

    # ---- the shapes the SERVE runs, which is what this gate missed the first time ----------------
    print("\n== PRODUCTION extend (rdna4.py::_swa_prefill_extend, zero-key pad) ==")
    # The three shapes a block-diffusion partial prefix hit actually produces on Gemma4 (W=1024,
    # page_size=16): a FULL hit extends a few tokens from a page-aligned boundary with pad 16, and a
    # PARTIAL hit page-splits into a short chunk at pad 0 followed by the page remainder.
    case_production("full-hit tail", 3184, 5)
    case_production("partial-hit chunk1 (page-split)", 3168, 16)
    case_production("partial-hit chunk2", 3184, 7)
    sweep_pad(16)
    sweep_chunk(0)
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
