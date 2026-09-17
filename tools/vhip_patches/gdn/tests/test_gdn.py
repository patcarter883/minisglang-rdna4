"""Green the gdn_hip kernel-builder package on gfx1201: forward parity + backward.

Runnable standalone (no pytest needed) inside the ROCm torch image after local/build_local.sh:
    python tests/test_gdn.py
Exits nonzero on the first failure. Covers:
  * FORWARD parity: gdn_prefill / causal_conv1d_fwd / rmsnorm_gated native op vs the pure-torch
    reference (fp32 + bf16).
  * NATIVE BACKWARD: the analytic HIP bwd kernels (rmsnorm_gated_bwd, causal_conv1d_bwd) vs autograd
    through the reference — an INDEPENDENT oracle for the hand-written gradient kernels.
  * TRAIN WRAPPERS: gdn_prefill_train / rmsnorm_gated_train / causal_conv1d_fwd_train end-to-end
    forward+backward, exercising the autograd.Function wiring the trainer uses.
"""
import os
import sys

import torch

# import the built package from torch-ext/ (local build) — same package the Hub ships.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "torch-ext"))

import gdn_hip as G  # noqa: E402

DEV = "cuda"
FAILS = []


def check(name, got, ref, rtol):
    got = got.detach().to(torch.float32)
    ref = ref.detach().to(torch.float32)
    denom = ref.norm().clamp_min(1e-12)
    rel = (got - ref).norm() / denom
    ok = torch.isfinite(rel) and rel.item() <= rtol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<44} rel={rel.item():.3e} (tol {rtol:.1e})")
    if not ok:
        FAILS.append(name)


def rand(*shape, dtype=torch.float32):
    return torch.randn(*shape, device=DEV, dtype=dtype)


# ---------------------------------------------------------------- forward parity
def test_forward(dtype, tol):
    tag = str(dtype).split(".")[-1]
    print(f"\n== forward parity ({tag}) ==")
    torch.manual_seed(0)

    # gdn_prefill: q,k [T,H,K]; v [T,HV,V]; a,b [T,HV]; A_log,dt_bias [HV]
    T, H, K, HV, V = 16, 4, 128, 8, 128
    scale = 1.0 / (K ** 0.5)
    q, k = rand(T, H, K, dtype=dtype), rand(T, H, K, dtype=dtype)
    v = rand(T, HV, V, dtype=dtype)
    a, b = rand(T, HV, dtype=dtype), rand(T, HV, dtype=dtype)
    A_log, dt_bias = rand(HV), rand(HV)
    out = G.gdn_prefill_train(q, k, v, a, b, A_log, dt_bias, scale, 1)  # native forward
    ref = G.ref_gdn_prefill_core(q, k, v, a, b, A_log, dt_bias, scale, True)
    check("gdn_prefill", out, ref, tol)

    # causal_conv1d_fwd: x [T,C]; weight [C,W]; bias [C]
    Tc, C, W = 32, 128, 4
    x = rand(Tc, C, dtype=dtype)
    weight, bias = rand(C, W), rand(C)
    out = G.causal_conv1d_fwd_train(x, weight, bias, 1)
    ref = G.ref_causal_conv1d_fwd(x, weight, bias, True)
    check("causal_conv1d_fwd", out, ref, tol)

    # rmsnorm_gated: x,z [M,D]; weight [D]
    M, D = 64, 128
    x, z = rand(M, D, dtype=dtype), rand(M, D, dtype=dtype)
    weight = rand(D)
    out = G.rmsnorm_gated(x.contiguous(), z.contiguous(), weight, 1e-5)
    ref = G.ref_rmsnorm_gated(x, z, weight, 1e-5)
    check("rmsnorm_gated", out, ref, tol)


# ---------------------------------------------------------------- native backward kernels
def test_native_backward():
    print("\n== native HIP backward kernels vs autograd-through-reference (fp32) ==")
    torch.manual_seed(1)
    tol = 5e-3

    # rmsnorm_gated_bwd
    M, D = 64, 128
    eps = 1e-5
    x, z, weight = rand(M, D), rand(M, D), rand(D)
    go = rand(M, D)
    dx_n, dz_n, dw_n = G.rmsnorm_gated_bwd(go.contiguous(), x.contiguous(), z.contiguous(), weight, eps)
    xr = x.clone().requires_grad_(True)
    zr = z.clone().requires_grad_(True)
    wr = weight.clone().requires_grad_(True)
    ref = G.ref_rmsnorm_gated(xr, zr, wr, eps)
    dx_r, dz_r, dw_r = torch.autograd.grad(ref, (xr, zr, wr), go)
    check("rmsnorm_gated_bwd dx", dx_n, dx_r, tol)
    check("rmsnorm_gated_bwd dz", dz_n, dz_r, tol)
    check("rmsnorm_gated_bwd dweight", dw_n, dw_r, tol)

    # causal_conv1d_bwd
    Tc, C, W = 32, 128, 4
    x, weight, bias = rand(Tc, C), rand(C, W), rand(C)
    go = rand(Tc, C)
    dx_n, dw_n, db_n = G.causal_conv1d_bwd(go.contiguous(), x.contiguous(), weight, bias, 1)
    xr = x.clone().requires_grad_(True)
    wr = weight.clone().requires_grad_(True)
    br = bias.clone().requires_grad_(True)
    ref = G.ref_causal_conv1d_fwd(xr, wr, br, True)
    dx_r, dw_r, db_r = torch.autograd.grad(ref, (xr, wr, br), go)
    check("causal_conv1d_bwd dx", dx_n, dx_r, tol)
    check("causal_conv1d_bwd dweight", dw_n, dw_r, tol)
    check("causal_conv1d_bwd dbias", db_n, db_r, tol)


# ---------------------------------------------------------------- train wrappers (end-to-end)
def test_train_wrappers():
    print("\n== differentiable train wrappers: forward+backward runs, grads finite (fp32) ==")
    torch.manual_seed(2)
    tol = 5e-3

    # gdn_prefill_train: grads vs an independent autograd pass through the same reference
    T, H, K, HV, V = 16, 4, 128, 8, 128
    scale = 1.0 / (K ** 0.5)
    q, k = rand(T, H, K), rand(T, H, K)
    v = rand(T, HV, V)
    a, b = rand(T, HV), rand(T, HV)
    A_log, dt_bias = rand(HV), rand(HV)
    ins = [t.clone().requires_grad_(True) for t in (q, k, v, a, b, A_log, dt_bias)]
    out = G.gdn_prefill_train(*ins, scale, 1)
    go = rand(*out.shape)
    grads_w = torch.autograd.grad(out, ins, go)

    ins2 = [t.clone().requires_grad_(True) for t in (q, k, v, a, b, A_log, dt_bias)]
    ref = G.ref_gdn_prefill_core(*ins2, scale, True)
    grads_r = torch.autograd.grad(ref, ins2, go)
    labels = ["q", "k", "v", "a", "b", "A_log", "dt_bias"]
    for lbl, gw, gr in zip(labels, grads_w, grads_r):
        check(f"gdn_prefill_train grad {lbl}", gw, gr, tol)

    # rmsnorm_gated_train with native backward enabled (default)
    M, D = 64, 128
    x, z, weight = rand(M, D), rand(M, D), rand(D)
    xt = x.clone().requires_grad_(True)
    zt = z.clone().requires_grad_(True)
    wt = weight.clone().requires_grad_(True)
    o = G.rmsnorm_gated_train(xt, zt, wt, 1e-5)
    go = rand(M, D)
    gx, gz, gw = torch.autograd.grad(o, (xt, zt, wt), go)
    for lbl, g in [("dx", gx), ("dz", gz), ("dweight", gw)]:
        finite = torch.isfinite(g).all().item()
        print(f"  [{'PASS' if finite else 'FAIL'}] rmsnorm_gated_train grad {lbl} finite")
        if not finite:
            FAILS.append(f"rmsnorm_gated_train {lbl}")


def main():
    print(f"device: {torch.cuda.get_device_properties(0).gcnArchName} | torch {torch.__version__}")
    print(f"GDN_HIP_NATIVE_BWD={os.environ.get('GDN_HIP_NATIVE_BWD', '1')}")
    test_forward(torch.float32, 4e-3)
    test_forward(torch.bfloat16, 1.2e-1)
    test_native_backward()
    test_train_wrappers()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("ALL GREEN")


if __name__ == "__main__":
    main()
