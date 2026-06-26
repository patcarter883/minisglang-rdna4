"""Engagement + correctness test for the gdn_hip recurrent-state Frobenius-norm clamp
(SSM_STATE_MAX_NORM, the Stuffed-Mamba guardrail in gdn_kernels.hip).

Strategy: drive the per-head state ||H||_F PAST the cap, then compare the kernel against TWO torch
references — one WITH the same Frobenius clamp, one WITHOUT. A correct clamp must
  (a) match the CLAMPED reference (it computes the right thing),
  (b) hold every per-head ||H||_F <= cap (it actually fires), and
  (c) differ from the UNCLAMPED reference, which must itself exceed the cap (the scenario is real).
The recurrent gdn_hip_parity.py covers the no-op (in-range) path; this covers the engaged path.

Run inside the combined ROCm image UNDER a 1-card lease (executes HIP kernels):
    .../gpu-lease.sh -n 1 -- bash -c 'docker run ... PYTHONPATH=/engine python /engine/tools/gdn_hip_clamp_test.py'
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import gdn_hip  # noqa: F401  (loads the .so + registers torch.ops.gdn_hip.*)

DEV = "cuda"
torch.manual_seed(0)
H, HV, K, V = 16, 32, 128, 128
SCALE = K ** -0.5
MAXN = 1000.0  # must match SSM_STATE_MAX_NORM in gdn_kernels.hip


def _softplus(x):
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


def ref_step(S, q, k, v, g, beta, clamp):
    S = S * torch.exp(g)
    v = (v - S @ k) * beta
    S = S + torch.outer(v, k)
    o = S @ q
    if clamp:
        n = S.norm()  # Frobenius over [V,K]
        if n > MAXN:
            S = S * (MAXN / n)
    return o, S


def _run_ref(state0, q, k, v, a, b, A_log, dt_bias, lens, clamp):
    """Per-(seq,head) recurrence over a packed varlen batch; returns final per-slot state."""
    st = state0.clone()
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    for n, L in enumerate(lens):
        slot = n + 1
        for hv in range(HV):
            hq = hv // (HV // H)
            S = st[slot, hv]
            for t in range(cu[n], cu[n + 1]):
                qn = F.normalize(q[t, hq], dim=-1, eps=1e-6) * SCALE
                kn = F.normalize(k[t, hq], dim=-1, eps=1e-6)
                g = -torch.exp(A_log[hv]) * _softplus(a[t, hv] + dt_bias[hv])
                beta = torch.sigmoid(b[t, hv])
                _, S = ref_step(S, qn, kn, v[t, hv].clone(), g, beta, clamp)
            st[slot, hv] = S
    return st


def test_prefill_long() -> bool:
    # The gated delta rule self-stabilizes (the v - S@k correction drives S toward a fixed point), so
    # fresh accumulation plateaus well below the cap. To exercise the PREFILL token-loop clamp we seed
    # a large INITIAL state (has_initial_state=1, ||H||_F ~ 2560 >> cap) and run tokens with near-zero
    # decay + tiny updates, so each token's clamp must re-bound the carried state at the cap.
    T = 16
    lens = [T]
    num_slots = 3
    q = torch.randn(T, H, K, device=DEV)
    k = torch.randn(T, H, K, device=DEV)
    v = torch.randn(T, HV, V, device=DEV) * 0.25    # small updates: state magnitude set by the seed
    a = torch.full((T, HV), 4.0, device=DEV)        # large dt
    b = torch.zeros(T, HV, device=DEV)              # beta ~ 0.5
    A_log = torch.full((HV,), -6.0, device=DEV)      # |A| tiny -> exp(g) ~ 1, ~no forgetting
    dt_bias = torch.zeros(HV, device=DEV)
    state0 = torch.randn(num_slots, HV, V, K, device=DEV) * 20.0  # seed ||H||_F ~ 2560 >> cap
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    idx = torch.tensor([1], dtype=torch.long, device=DEV)
    has_init = torch.ones(1, dtype=torch.uint8, device=DEV)       # carry the seeded state in

    ref_c = _run_ref(state0, q, k, v, a, b, A_log, dt_bias, lens, clamp=True)
    ref_u = _run_ref(state0, q, k, v, a, b, A_log, dt_bias, lens, clamp=False)

    got = state0.clone()
    torch.ops.gdn_hip.gdn_prefill(q, k, v, a, b, A_log, dt_bias, cu, idx, has_init, got, SCALE, 1)

    n_unclamped = ref_u[1].norm(dim=(-2, -1)).max().item()   # would-be max per-head ||H||_F, no clamp
    n_kernel = got[1].norm(dim=(-2, -1)).max().item()        # kernel's max per-head ||H||_F
    d_clamped = (got[1] - ref_c[1]).abs().max().item()       # kernel vs clamped reference

    engaged = n_unclamped > MAXN
    bounded = n_kernel <= MAXN * 1.02
    correct = d_clamped <= 5e-3 * max(1.0, n_kernel)
    print(f"  prefill_long: unclamped||H||F(max)={n_unclamped:.1f}  kernel||H||F(max)={n_kernel:.1f}  "
          f"vs-clamped-ref max|Δ|={d_clamped:.3e}")
    print(f"    engaged(unclamped>cap)={engaged}  bounded(kernel<=cap)={bounded}  matches-clamped-ref={correct}")
    return engaged and bounded and correct


def test_decode_big_state() -> bool:
    # Initial state already over the cap -> a single decode step must re-bound it.
    B, num_slots = 2, 4
    q = torch.randn(B, H, K, device=DEV)
    k = torch.randn(B, H, K, device=DEV)
    v = torch.randn(B, HV, V, device=DEV)
    a = torch.randn(B, HV, device=DEV)
    b = torch.randn(B, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    state0 = torch.randn(num_slots, HV, V, K, device=DEV) * 20.0  # ||H||_F ~ 2560 >> 1000
    idx = torch.tensor([1, 3], dtype=torch.long, device=DEV)

    # reference (clamped): mirror gdn_decode for the two live slots
    ref_c = state0.clone()
    for bi in range(B):
        slot = int(idx[bi])
        for hv in range(HV):
            hq = hv // (HV // H)
            qn = F.normalize(q[bi, hq], dim=-1, eps=1e-6) * SCALE
            kn = F.normalize(k[bi, hq], dim=-1, eps=1e-6)
            g = -torch.exp(A_log[hv]) * _softplus(a[bi, hv] + dt_bias[hv])
            beta = torch.sigmoid(b[bi, hv])
            _, S = ref_step(ref_c[slot, hv], qn, kn, v[bi, hv].clone(), g, beta, clamp=True)
            ref_c[slot, hv] = S

    got = state0.clone()
    torch.ops.gdn_hip.gdn_decode(q, k, v, a, b, A_log, dt_bias, got, idx, SCALE, 1)
    n_kernel = got[idx].norm(dim=(-2, -1)).max().item()
    d_clamped = (got[idx] - ref_c[idx]).abs().max().item()
    bounded = n_kernel <= MAXN * 1.02
    correct = d_clamped <= 5e-3 * max(1.0, n_kernel)
    print(f"  decode_big_state: kernel||H||F(max)={n_kernel:.1f}  vs-clamped-ref max|Δ|={d_clamped:.3e}")
    print(f"    bounded(kernel<=cap)={bounded}  matches-clamped-ref={correct}")
    return bounded and correct


def main() -> None:
    print("=== gdn_hip recurrent-state Frobenius-clamp engagement test (cap=%.0f) ===" % MAXN)
    ok = True
    ok &= test_prefill_long()
    ok &= test_decode_big_state()
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
