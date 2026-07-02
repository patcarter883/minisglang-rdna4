"""CPU float64 gradcheck for the pure-torch GDN backward references (gdn_hip/autograd.py).

This runs ENTIRELY on CPU with NO GPU and NO gdn_hip_C*.so: it gradchecks the pure-torch
DIFFERENTIABLE references (`ref_gdn_prefill_core`, `ref_causal_conv1d_fwd`, `ref_rmsnorm_gated`)
that the autograd backward recomputes. If a reference's analytic gradient (torch autograd) matches
its finite-difference gradient in float64, the recompute-backward is self-consistent — that is the
CPU-provable half of correctness. The GPU half (native forward + this backward vs fla) is
tools/gdn_backward_validate.py, run separately under a lease.

We ALSO independently re-verify each reference's FORWARD against a from-scratch oracle of the op's
documented math (a separate implementation than the reference — the recurrent `ref_step` scan for
prefill, an explicit sliding-window for conv, the closed form for rmsnorm), BEFORE trusting the
gradient. A gradient of a wrong forward is a wrong gradient that still gradchecks.

Run (CPU, no GPU, no .so):
    /tmp/gdnbwd-venv/bin/python tools/gdn_backward_gradcheck.py
or inside any environment with a CPU torch:
    python tools/gdn_backward_gradcheck.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

# import the references directly (no .so load — gdn_hip/op.py would try to load the extension).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gdn_hip"))
import autograd as gdn_bwd  # noqa: E402  (gdn_hip/autograd.py, imported as a bare module)

torch.manual_seed(0)
DEV = "cpu"
DT = torch.float64  # gradcheck needs double precision


# ----------------------------------------------------------------------------------------------
# Independent forward oracles (a SECOND implementation of the op math, to catch a bug in the
# reference before we trust its gradient).
# ----------------------------------------------------------------------------------------------
def _softplus(x):
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


def _l2norm(x):  # kernel convention: rsqrt(sumsq + 1e-6)
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)


def oracle_gdn_prefill(q, k, v, a, b, A_log, dt_bias, scale):
    """Independent recurrent scan (ref_step form), matching tools/gdn_hip_parity.py's oracle:
      S*=exp(g); v -= S@k; v*=beta; S += outer(v,k); o = S@q."""
    T, H, K = q.shape
    HV, Vv = v.shape[1], v.shape[2]
    rep = HV // H
    out = torch.zeros(T, HV, Vv, dtype=q.dtype, device=q.device)
    for hv in range(HV):
        hq = hv // rep
        S = torch.zeros(Vv, K, dtype=q.dtype, device=q.device)
        for t in range(T):
            qn = _l2norm(q[t, hq]) * scale
            kn = _l2norm(k[t, hq])
            g = -torch.exp(A_log[hv]) * _softplus(a[t, hv] + dt_bias[hv])
            beta = torch.sigmoid(b[t, hv])
            S = S * torch.exp(g)
            vt = (v[t, hv] - S @ kn) * beta
            S = S + torch.outer(vt, kn)
            out[t, hv] = S @ qn
    return out


def oracle_conv(x, weight, bias, activation):
    """Independent sliding-window depthwise causal conv (zero left pad) + SiLU."""
    T, C = x.shape
    W = weight.shape[1]
    out = torch.zeros(T, C, dtype=x.dtype, device=x.device)
    hist = torch.zeros(C, W - 1, dtype=x.dtype, device=x.device)
    for t in range(T):
        win = torch.cat([hist, x[t].unsqueeze(-1)], dim=-1)  # [C, W]
        acc = (win * weight).sum(-1) + (bias if bias is not None else 0.0)
        out[t] = F.silu(acc) if activation else acc
        hist = win[:, 1:]
    return out


def oracle_rmsnorm(x, z, weight, eps):
    inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * inv * weight * F.silu(z)


# ----------------------------------------------------------------------------------------------
def _fwd_match(name, ref, oracle, tol=1e-9):
    d = (ref.double() - oracle.double()).abs().max().item()
    ok = d <= tol and torch.isfinite(ref).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] forward-vs-oracle {name:22s} max|Δ|={d:.3e} (tol={tol:.1e})")
    return ok


def check_prefill():
    print("--- gdn_prefill core ---")
    # small: T=4 tokens, H=2 key heads, HV=4 value heads (rep=2), K=V=3
    T, H, HV, K, Vv = 4, 2, 4, 3, 3
    scale = K ** -0.5
    q = torch.randn(T, H, K, dtype=DT, device=DEV)
    k = torch.randn(T, H, K, dtype=DT, device=DEV)
    v = torch.randn(T, HV, Vv, dtype=DT, device=DEV)
    a = torch.randn(T, HV, dtype=DT, device=DEV)
    b = torch.randn(T, HV, dtype=DT, device=DEV)
    A_log = torch.randn(HV, dtype=DT, device=DEV)
    dt_bias = torch.randn(HV, dtype=DT, device=DEV)

    ref = gdn_bwd.ref_gdn_prefill_core(q, k, v, a, b, A_log, dt_bias, scale, True)
    orc = oracle_gdn_prefill(q, k, v, a, b, A_log, dt_bias, scale)
    ok = _fwd_match("gdn_prefill", ref, orc)

    for t in (q, k, v, a, b, A_log, dt_bias):
        t.requires_grad_(True)
    f = lambda *ins: gdn_bwd.ref_gdn_prefill_core(*ins, scale, True)  # noqa: E731
    g = torch.autograd.gradcheck(f, (q, k, v, a, b, A_log, dt_bias), eps=1e-6, atol=1e-5, rtol=1e-3,
                                 raise_exception=False)
    print(f"  [{'PASS' if g else 'FAIL'}] gradcheck gdn_prefill (q,k,v,a,b,A_log,dt_bias)")
    # per-input isolation: gradcheck each input alone (others detached) to localize a bad grad
    base = (q, k, v, a, b, A_log, dt_bias)
    names = ("q", "k", "v", "a", "b", "A_log", "dt_bias")
    for i, nm in enumerate(names):
        ins = [t.detach().clone() for t in base]
        ins[i].requires_grad_(True)
        gi = torch.autograd.gradcheck(
            lambda xi, i=i, ins=ins: gdn_bwd.ref_gdn_prefill_core(
                *[xi if j == i else ins[j] for j in range(len(ins))], scale, True),
            (ins[i],), eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=False)
        print(f"      [{'PASS' if gi else 'FAIL'}] grad wrt {nm}")
        ok &= gi
    return ok and g


def check_conv():
    print("--- causal_conv1d_fwd ---")
    T, C, W = 6, 5, 4  # GDN W=4
    x = torch.randn(T, C, dtype=DT, device=DEV)
    weight = torch.randn(C, W, dtype=DT, device=DEV)
    bias = torch.randn(C, dtype=DT, device=DEV)

    ref = gdn_bwd.ref_causal_conv1d_fwd(x, weight, bias, True)
    orc = oracle_conv(x, weight, bias, True)
    ok = _fwd_match("conv (bias, silu)", ref, orc)
    ref_nb = gdn_bwd.ref_causal_conv1d_fwd(x, weight, None, True)
    orc_nb = oracle_conv(x, weight, None, True)
    ok &= _fwd_match("conv (no bias)", ref_nb, orc_nb)
    ref_na = gdn_bwd.ref_causal_conv1d_fwd(x, weight, bias, False)
    orc_na = oracle_conv(x, weight, bias, False)
    ok &= _fwd_match("conv (no activation)", ref_na, orc_na)

    x.requires_grad_(True); weight.requires_grad_(True); bias.requires_grad_(True)
    g = torch.autograd.gradcheck(
        lambda x, w, bb: gdn_bwd.ref_causal_conv1d_fwd(x, w, bb, True), (x, weight, bias),
        eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=False)
    print(f"  [{'PASS' if g else 'FAIL'}] gradcheck conv (x, weight, bias) + SiLU")
    return ok and g


def check_rmsnorm():
    print("--- rmsnorm_gated ---")
    M, D = 8, 6
    eps = 1e-5
    x = torch.randn(M, D, dtype=DT, device=DEV)
    z = torch.randn(M, D, dtype=DT, device=DEV)
    weight = torch.randn(D, dtype=DT, device=DEV)

    ref = gdn_bwd.ref_rmsnorm_gated(x, z, weight, eps)
    orc = oracle_rmsnorm(x, z, weight, eps)
    ok = _fwd_match("rmsnorm_gated", ref, orc)

    x.requires_grad_(True); z.requires_grad_(True); weight.requires_grad_(True)
    g = torch.autograd.gradcheck(
        lambda x, z, w: gdn_bwd.ref_rmsnorm_gated(x, z, w, eps), (x, z, weight),
        eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=False)
    print(f"  [{'PASS' if g else 'FAIL'}] gradcheck rmsnorm_gated (x, z, weight)")
    return ok and g


def main():
    print(f"=== CPU float64 gradcheck of GDN backward references (device={DEV}, dtype=float64) ===")
    r1 = check_prefill()
    r2 = check_conv()
    r3 = check_rmsnorm()
    allok = r1 and r2 and r3
    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS — pure-torch GDN backward references are gradient self-consistent"
          if allok else "FAIL (see above)")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
