"""GPU validation of the native GDN forward + recompute backward (gdn_hip/autograd.py) vs fla.

Run under a 1-card lease in the combined ROCm image (executes the HIP kernels + the pure-torch
recompute backward on-device). This is the GPU half of the correctness proof; the CPU half is
tools/gdn_backward_gradcheck.py (float64 gradcheck of the pure-torch references, no GPU).

Two levels, each for the RECURRENT (gdn_prefill, exact) and WMMA (gdn_prefill_wmma, ~1e-3) ops:

  LEVEL 1 — op wiring: for each of the 3 ops, run the NATIVE forward + the REGISTERED backward
    (gdn_hip.autograd.enable() → the raw op is differentiable) and compare the input-grads against
    torch.autograd.grad over the pure-torch reference forward on the SAME inputs. This confirms the
    autograd registration hands back exactly the reference's gradient (cosine ~1, max|Δ| at the
    dtype floor), i.e. the recompute plumbing is correct on-device.

  LEVEL 2 — end-to-end vs fla: build ONE GDN layer both ways — minisgl QwenGatedDeltaNet (native
    HIP fwd + registered bwd) and HF Qwen3_5GatedDeltaNet (pure-torch, DIFFERENTIABLE fla) — copy HF
    weights into minisgl, feed identical hidden_states, and compare the GRAD of the input
    hidden_states (dL/d hidden, L = sum of a fixed random projection of the mixer output). The input
    grad traces back through q,k,v,a,b — exactly the perturbed-residual path the downstream trainer
    backprops. cosine-sim + max|Δ| + rel-L2. Recurrent should be tight; wmma looser (~1e-3 fwd).

  gdn_prefill (recurrent): exact forward, so BOTH levels should be tight (cos>0.999).
  gdn_prefill_wmma:        ~1e-3 forward vs the reference the backward recomputes → the grad is the
                           reference's exact grad, ~1e-3 off the true wmma jacobian; looser tol.

Invocation (MANDATORY env-forwarding form — the raw `-e VAR=$VAR` in `gpu-lease -- docker run`
expands in the OUTER shell to empty; wrap the whole docker run in `bash -c` INSIDE the lease so the
lease's HIP/ROCR vars are in scope):

  gpu-lease -n 1 -- bash -c 'docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    -e HF_HUB_OFFLINE=1 \
    -v "$PWD":/engine \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint bash vllm22-w4a8:combined -lc \
    "source /app/.venv/bin/activate && PYTHONPATH=/engine/python:/engine \
     python /engine/tools/gdn_backward_validate.py"'

(Mount the WORKTREE as $PWD — run this from /home/pat/code/minisgl-rdna4-gdnbwd. The gdn_hip_C*.so
must already be built in that checkout. --level {1,2,both} selects; default both.)
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import gdn_hip  # loads the .so + registers torch.ops.gdn_hip.*
from gdn_hip import autograd as gdn_bwd

DEV = "cuda"

# Real GDN geometry (Qwen3.5/3.6): num_k_heads=16, num_v_heads=32, head_k/v_dim=128.
H, HV, K, V = 16, 32, 128, 128
SCALE = K ** -0.5


def _metrics(got: torch.Tensor, ref: torch.Tensor):
    g, r = got.float().flatten(), ref.float().flatten()
    cos = F.cosine_similarity(g, r, dim=0).item()
    dmax = (g - r).abs().max().item()
    rel = ((g - r).norm() / (r.norm() + 1e-9)).item()
    return cos, dmax, rel


def _report(name, got, ref, cos_thr, rel_thr):
    cos, dmax, rel = _metrics(got, ref)
    ok = cos >= cos_thr and rel <= rel_thr and torch.isfinite(got).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:46s} cos={cos:.6f} max|Δ|={dmax:.3e} relL2={rel:.3e}"
          f"  (cos>={cos_thr}, rel<={rel_thr:.0e})")
    return ok


# ==============================================================================================
# LEVEL 1 — op wiring: native fwd + registered bwd vs torch.autograd.grad over the reference.
# ==============================================================================================
def level1_prefill(wmma: bool, dtype: torch.dtype) -> bool:
    torch.manual_seed(0)
    T = 24
    q = torch.randn(T, H, K, device=DEV, dtype=dtype)
    k = torch.randn(T, H, K, device=DEV, dtype=dtype)
    v = torch.randn(T, HV, V, device=DEV, dtype=dtype)
    a = torch.randn(T, HV, device=DEV, dtype=dtype)
    b = torch.randn(T, HV, device=DEV, dtype=dtype)
    A_log = torch.randn(HV, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=DEV, dtype=torch.float32)
    gout = torch.randn(T, HV, V, device=DEV, dtype=torch.float32)

    ins = [t.clone().requires_grad_(True) for t in (q, k, v, a, b, A_log, dt_bias)]
    train_fn = gdn_bwd.gdn_prefill_wmma_train if wmma else gdn_bwd.gdn_prefill_train
    out = train_fn(*ins, SCALE, 1)
    got_grads = torch.autograd.grad(out.float(), ins, gout)

    # reference grads: pure-torch reference forward on identical inputs
    rins = [t.detach().clone().float().requires_grad_(True) for t in (q, k, v, a, b, A_log, dt_bias)]
    ref = gdn_bwd.ref_gdn_prefill_core(*rins, SCALE, 1)
    ref_grads = torch.autograd.grad(ref, rins, gout)

    tag = "wmma" if wmma else "recur"
    # context: native forward vs the reference forward (recurrent should be exact-ish, wmma ~1e-3)
    fcos, fdmax, frel = _metrics(out, ref)
    print(f"  [ctx ] L1 prefill[{tag},{dtype}] forward vs ref  cos={fcos:.6f} max|Δ|={fdmax:.3e} relL2={frel:.3e}")
    # exact op → tight; both should match the reference grad closely (this checks the WIRING, so even
    # wmma is tight here — the forward output isn't in this comparison, only that bwd==reference grad).
    ct, rt = (0.999, 5e-2) if dtype == torch.float32 else (0.995, 1.5e-1)
    ok = True
    for nm, g, r in zip("q k v a b A_log dt_bias".split(), got_grads, ref_grads):
        ok &= _report(f"L1 prefill[{tag},{dtype}] grad_{nm}", g, r, ct, rt)
    return ok


def level1_conv(dtype: torch.dtype) -> bool:
    torch.manual_seed(1)
    T, C, W = 24, H * K * 2 + HV * V, 4  # conv_dim = 2*key_dim + value_dim
    x = torch.randn(T, C, device=DEV, dtype=dtype)
    weight = torch.randn(C, W, device=DEV, dtype=torch.float32)
    bias = torch.randn(C, device=DEV, dtype=torch.float32)
    gout = torch.randn(T, C, device=DEV, dtype=torch.float32)

    xin = x.clone().requires_grad_(True)
    win = weight.clone().requires_grad_(True)
    bin_ = bias.clone().requires_grad_(True)
    out = gdn_bwd.causal_conv1d_fwd_train(xin, win, bin_, 1)
    got = torch.autograd.grad(out.float(), (xin, win, bin_), gout)

    rx = x.detach().clone().float().requires_grad_(True)
    rw = weight.detach().clone().requires_grad_(True)
    rb = bias.detach().clone().requires_grad_(True)
    ref = gdn_bwd.ref_causal_conv1d_fwd(rx, rw, rb, 1)
    refg = torch.autograd.grad(ref, (rx, rw, rb), gout)

    ct, rt = (0.999, 5e-2) if dtype == torch.float32 else (0.995, 1.5e-1)
    ok = True
    for nm, g, r in zip("x weight bias".split(), got, refg):
        ok &= _report(f"L1 conv[{dtype}] grad_{nm}", g, r, ct, rt)
    return ok


def level1_rmsnorm(dtype: torch.dtype) -> bool:
    torch.manual_seed(2)
    M, D = 64, V
    x = torch.randn(M, D, device=DEV, dtype=dtype)
    z = torch.randn(M, D, device=DEV, dtype=dtype)
    weight = torch.randn(D, device=DEV, dtype=torch.float32)
    eps = 1e-5
    gout = torch.randn(M, D, device=DEV, dtype=torch.float32)

    xin = x.clone().requires_grad_(True)
    zin = z.clone().requires_grad_(True)
    win = weight.clone().requires_grad_(True)
    out = gdn_bwd.rmsnorm_gated_train(xin, zin, win, eps)
    got = torch.autograd.grad(out.float(), (xin, zin, win), gout)

    rx = x.detach().clone().float().requires_grad_(True)
    rz = z.detach().clone().float().requires_grad_(True)
    rw = weight.detach().clone().requires_grad_(True)
    ref = gdn_bwd.ref_rmsnorm_gated(rx, rz, rw, eps)
    refg = torch.autograd.grad(ref, (rx, rz, rw), gout)

    ct, rt = (0.999, 5e-2) if dtype == torch.float32 else (0.995, 1.5e-1)
    ok = True
    for nm, g, r in zip("x z weight".split(), got, refg):
        ok &= _report(f"L1 rmsnorm[{dtype}] grad_{nm}", g, r, ct, rt)
    return ok


# ==============================================================================================
# LEVEL 2 — end-to-end: minisgl QwenGatedDeltaNet (native fwd + registered bwd) vs HF fla layer,
#           comparing dL/d hidden_states.
# ==============================================================================================
def level2(wmma: bool) -> bool:
    import os
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    from minisgl.gdn.layer import QwenGatedDeltaNet

    os.environ["GDN_HIP_WMMA_PREFILL"] = "0" if not wmma else "1"
    MODEL = "Qwen/Qwen3.5-4B"
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    dtype = torch.bfloat16

    hf = Qwen3_5GatedDeltaNet(tc, layer_idx=0).to(DEV).to(dtype).eval()
    with torch.no_grad():
        hf.A_log.copy_(torch.log(torch.empty(tc.linear_num_value_heads, device=DEV).uniform_(1, 16)))
        hf.dt_bias.uniform_(-2, 2)

    ms = QwenGatedDeltaNet(
        hidden_size=tc.hidden_size, num_k_heads=tc.linear_num_key_heads,
        num_v_heads=tc.linear_num_value_heads, head_k_dim=tc.linear_key_head_dim,
        head_v_dim=tc.linear_value_head_dim, conv_kernel_size=tc.linear_conv_kernel_dim,
        eps=tc.rms_norm_eps, dtype=dtype, device=DEV,
    )
    with torch.no_grad():
        ms.in_proj_qkvz.weight.copy_(torch.cat([hf.in_proj_qkv.weight, hf.in_proj_z.weight], dim=0))
        ms.in_proj_ba.weight.copy_(torch.cat([hf.in_proj_b.weight, hf.in_proj_a.weight], dim=0))
        ms.conv1d_weight.copy_(hf.conv1d.weight)
        ms.A_log.copy_(hf.A_log.float())
        ms.dt_bias.copy_(hf.dt_bias.float())
        ms.norm.weight.copy_(hf.norm.weight)
        ms.out_proj.weight.copy_(hf.out_proj.weight)

    T = 24
    x0 = torch.randn(1, T, tc.hidden_size, device=DEV, dtype=dtype)
    proj = torch.randn(tc.hidden_size, device=DEV, dtype=torch.float32)  # fixed scalarizer for L

    # HF grad (pure-torch fla, differentiable)
    xhf = x0.clone().requires_grad_(True)
    hf_out = hf(xhf)[0]
    if hf_out.dim() == 3:
        hf_out = hf_out[0]
    Lhf = (hf_out.float().reshape(T, -1) @ proj).sum()
    (ghf,) = torch.autograd.grad(Lhf, xhf)

    # minisgl grad (native HIP fwd + registered recompute bwd)
    conv_dim = ms.conv_dim
    Kc = tc.linear_conv_kernel_dim
    conv_state = torch.zeros(2, conv_dim, Kc - 1, device=DEV, dtype=torch.float32)
    ssm_state = torch.zeros(2, tc.linear_num_value_heads, tc.linear_value_head_dim,
                            tc.linear_key_head_dim, device=DEV, dtype=torch.float32)
    qsl = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    sidx = torch.tensor([1], dtype=torch.int32, device=DEV)
    hinit = torch.tensor([False], device=DEV)
    xms = x0.clone().requires_grad_(True)
    ms_out = ms.forward_prefill(xms[0], conv_state, ssm_state, qsl, sidx, hinit)
    Lms = (ms_out.float().reshape(T, -1) @ proj).sum()
    (gms,) = torch.autograd.grad(Lms, xms)

    tag = "wmma" if wmma else "recurrent"
    # forward agreement (context) then the grad comparison (the deliverable)
    fcos, fdmax, frel = _metrics(ms_out, hf_out)
    print(f"  [ctx ] L2[{tag}] forward   cos={fcos:.6f} max|Δ|={fdmax:.3e} relL2={frel:.3e}")
    ct, rt = (0.999, 3e-2) if not wmma else (0.99, 8e-2)
    return _report(f"L2[{tag}] dL/d_hidden (vs fla)", gms, ghf, ct, rt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU (run under a lease)"
    print(f"=== GDN backward GPU validation (device={torch.cuda.get_device_name()}) ===")
    gdn_bwd.enable()  # register recompute-backward on the raw ops (so forward_prefill is differentiable)

    allok = True
    if args.level in ("1", "both"):
        print("\n--- LEVEL 1: native fwd + registered bwd vs reference autograd ---")
        for dt in (torch.float32, torch.bfloat16):
            allok &= level1_prefill(False, dt)   # recurrent
            allok &= level1_prefill(True, dt)    # wmma
            allok &= level1_conv(dt)
            allok &= level1_rmsnorm(dt)
    if args.level in ("2", "both"):
        print("\n--- LEVEL 2: end-to-end minisgl layer grad vs HF fla layer grad ---")
        allok &= level2(False)  # recurrent
        allok &= level2(True)   # wmma

    print("\n" + "=" * 70)
    print("RESULT:", "ALL PASS — native GDN forward + recompute backward matches fla autograd"
          if allok else "FAIL (see above)")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
