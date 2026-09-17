"""Standalone repro of the vhip spec-verify call, with vLLM's REAL state shapes. No serve boot.

vLLM (mamba_utils.gated_delta_net_state_shape) allocates, per TP rank:
    conv_state : _orient_conv_shape(conv_dim/tp, (W-1) + num_spec)   -> width 5, NOT W-1=3
    ssm_state  : (num_v_heads/tp, head_v_dim, head_k_dim)            -> matches minisgl's layout
and the glue may hand the conv state over as a TRANSPOSED view. minisgl instead owns a plain
(slots, C, W-1) contiguous conv state — that difference is the prime suspect for the `inf` the
in-serve A/B measured at layer 0 on the first spec step.

Sweeps the conv-state variants against the same inputs and reports where non-finite values appear.
"""

import torch
import gdn_hip

torch.manual_seed(0)
DEV = "cuda"
# Qwen3.6-35B-A3B at TP=2, per rank
NK, NV, HK, HV = 8, 16, 128, 128
C = HK * NK * 2 + HV * NV          # 4096, matches the C seen in-serve
W, NUM_SPEC = 4, 2
WM1 = W - 1
SLOTS = 8
QLEN = 3                            # 1 verified + 2 drafts
N = 1                               # one spec sequence

cu = torch.tensor([0, QLEN], dtype=torch.int32, device=DEV)
x = torch.randn(QLEN, C, device=DEV).float()
wt = torch.randn(C, W, device=DEV).float() * 0.1
a = torch.randn(QLEN, NV, device=DEV).float()
b = torch.randn(QLEN, NV, device=DEV).float()
A_log = torch.randn(NV, device=DEV).float()
dt_bias = torch.randn(NV, device=DEV).float()
load = torch.tensor([1], dtype=torch.long, device=DEV)
hi = torch.ones(N, dtype=torch.uint8, device=DEV)
scale = HK ** -0.5


def run(label, conv_state):
    ssm = torch.zeros(SLOTS, NV, HV, HK, device=DEV, dtype=torch.bfloat16)
    try:
        conv_out, conv_scr = gdn_hip.causal_conv1d_fwd_verify(
            x.contiguous(), wt, None, cu, load, hi, conv_state, QLEN, 1
        )
        q, k, v = conv_out.split([HK * NK, HK * NK, HV * NV], dim=-1)
        core, ssm_scr = gdn_hip.gdn_prefill_verify(
            q.reshape(-1, NK, HK).contiguous(),
            k.reshape(-1, NK, HK).contiguous(),
            v.reshape(-1, NV, HV).contiguous(),
            a.contiguous(), b.contiguous(), A_log, dt_bias,
            cu, load, hi, ssm, QLEN, scale, 1,
        )
        print(f"[{label}]")
        print(f"    conv_state: shape={tuple(conv_state.shape)} contig={conv_state.is_contiguous()} "
              f"stride={conv_state.stride()}")
        for nm, t in (("conv_out", conv_out), ("conv_scratch", conv_scr),
                      ("core", core), ("ssm_scratch", ssm_scr), ("ssm_state", ssm)):
            f = torch.isfinite(t.float())
            print(f"    {nm:13s} finite={bool(f.all())} "
                  f"nonfinite={int((~f).sum())}/{t.numel()} absmax="
                  f"{t.float()[f].abs().max().item() if bool(f.any()) else float('nan'):.4f}")
    except Exception as e:
        print(f"[{label}] RAISED {type(e).__name__}: {e}")


# A) what minisgl owns: plain (slots, C, W-1) contiguous
run("A minisgl layout  (slots,C,W-1) contiguous",
    torch.zeros(SLOTS, C, WM1, device=DEV).float())

# B) vLLM width, dim-first: (slots, C, W-1+num_spec) contiguous
run("B vLLM width, contiguous (slots,C,5)",
    torch.zeros(SLOTS, C, WM1 + NUM_SPEC, device=DEV).float())

# C) vLLM width, TRANSPOSED view: (slots, 5, C).transpose(-1,-2) -> (slots, C, 5) non-contiguous
run("C vLLM width, TRANSPOSED view (slots,C,5)",
    torch.zeros(SLOTS, WM1 + NUM_SPEC, C, device=DEV).float().transpose(-1, -2))

# D) load slot holds UNINITIALIZED memory (vLLM allocates the mamba cache with torch.empty; if the
#    load slot was never written, the kernel convolves garbage -> inf). This is what an in-serve
#    wrong-slot / missing-initial-state would look like.
cs_garbage = torch.empty(SLOTS, C, WM1 + NUM_SPEC, device=DEV, dtype=torch.float32)
cs_garbage.fill_(float("nan"))
run("D load slot = uninitialized (NaN) conv state", cs_garbage)

# E) realistic-magnitude gating: A_log positive would make -exp(A_log)*softplus() explode.
print(f"\n[E] gating sanity: A_log range=({A_log.min():.3f},{A_log.max():.3f}) "
      f"-exp(A_log) range=({(-A_log.exp()).min():.3f},{(-A_log.exp()).max():.3f})")
