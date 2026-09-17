"""Reproduce the serve's actual sequence standalone: PREFILL writes the state, then a SPEC VERIFY
step loads it from the same slot.

This is the handoff that fails in-serve (conv_out -> NaN on the first spec step at layer 0). The
prefill uses gdn_prefill_wmma (chunked WMMA) while verify uses gdn_prefill_verify (the recurrent
oracle) — if those two disagree about the recurrent-state layout, the verify reads garbage even
though the slot index is correct, which looks exactly like an uninitialized slot.

Also checks the has_init flag, which the vhip glue currently passes as unconditional ones.
"""

import torch
import gdn_hip

torch.manual_seed(0)
DEV = "cuda"
NK, NV, HK, HV = 8, 16, 128, 128
C = HK * NK * 2 + HV * NV
W, NUM_SPEC = 4, 2
WM1 = W - 1
SLOTS = 8
SLOT = 1                       # prefill writes here; verify must resume from here
scale = HK ** -0.5

wt = torch.randn(C, W, device=DEV).float() * 0.1
A_log = torch.randn(NV, device=DEV).float()
dt_bias = torch.randn(NV, device=DEV).float()

# vLLM's real allocation: conv width (W-1)+num_spec, ssm (HV, V, K)
conv_state = torch.zeros(SLOTS, C, WM1 + NUM_SPEC, device=DEV).float()
ssm_state = torch.zeros(SLOTS, NV, HV, HK, device=DEV, dtype=torch.bfloat16)
idx = torch.tensor([SLOT], dtype=torch.long, device=DEV)


def split(conv_out):
    q, k, v = conv_out.split([HK * NK, HK * NK, HV * NV], dim=-1)
    return (q.reshape(-1, NK, HK).contiguous(),
            k.reshape(-1, NK, HK).contiguous(),
            v.reshape(-1, NV, HV).contiguous())


def report(tag, **ts):
    out = []
    for n, t in ts.items():
        f = torch.isfinite(t.float())
        out.append(f"{n} finite={bool(f.all())} absmax="
                   f"{t.float()[f].abs().max().item() if bool(f.any()) else float('nan'):.4f}")
    print(f"  {tag}: " + " | ".join(out))


# ---- STEP 1: PREFILL (what _forward_core_gdn_hip does on a non-spec step) --------------------
PLEN = 16
cu_p = torch.tensor([0, PLEN], dtype=torch.int32, device=DEV)
xp = torch.randn(PLEN, C, device=DEV).float()
ap = torch.randn(PLEN, NV, device=DEV).float()
bp = torch.randn(PLEN, NV, device=DEV).float()
has_init_p = torch.zeros(1, dtype=torch.uint8, device=DEV)     # cold prefill: NO initial state

conv_out_p = gdn_hip.causal_conv1d_fwd(xp, wt, None, cu_p, idx, has_init_p, conv_state, 1)
qp, kp, vp = split(conv_out_p)
core_p = gdn_hip.gdn_prefill_wmma(qp, kp, vp, ap, bp, A_log, dt_bias,
                                  cu_p, idx, has_init_p, ssm_state, scale, 1)
print("STEP 1 prefill (gdn_prefill_wmma writes slot", SLOT, ")")
report("after prefill", core=core_p,
       conv_slot=conv_state[SLOT], ssm_slot=ssm_state[SLOT])

# ---- STEP 2: SPEC VERIFY resuming from that slot ---------------------------------------------
QLEN = 3
cu_v = torch.tensor([0, QLEN], dtype=torch.int32, device=DEV)
xv = torch.randn(QLEN, C, device=DEV).float()
av = torch.randn(QLEN, NV, device=DEV).float()
bv = torch.randn(QLEN, NV, device=DEV).float()

for label, has_init_v in (("has_init=1 (what vhip passes)", torch.ones(1, dtype=torch.uint8, device=DEV)),
                          ("has_init=0", torch.zeros(1, dtype=torch.uint8, device=DEV))):
    cs, ss = conv_state.clone(), ssm_state.clone()
    conv_out_v, conv_scr = gdn_hip.causal_conv1d_fwd_verify(
        xv, wt, None, cu_v, idx, has_init_v, cs, QLEN, 1)
    qv, kv, vv = split(conv_out_v)
    core_v, ssm_scr = gdn_hip.gdn_prefill_verify(
        qv, kv, vv, av, bv, A_log, dt_bias, cu_v, idx, has_init_v, ss, QLEN, scale, 1)
    print(f"STEP 2 verify  [{label}]")
    report("  result", conv_out=conv_out_v, core=core_v,
           ssm_scratch=ssm_scr, conv_scratch=conv_scr)

# ---- STEP 3: NON-CONTIGUOUS state views ------------------------------------------------------
# The existing non-spec glue makes a contiguous shadow of conv/ssm before calling these kernels
# (`ssm_state_k = ssm_state if ssm_state.is_contiguous() else ssm_state.contiguous()`), then copies
# back. The spec path passes vLLM's tensors RAW. If the prefill/verify kernels are not stride-aware
# for the SSM state, a strided view is read as garbage — which is what the serve sees.
print("STEP 3 non-contiguous ssm_state view")
ssm_nc = torch.zeros(SLOTS, NV, HV, HK + 8, device=DEV, dtype=torch.bfloat16)[..., :HK]
conv_nc = torch.zeros(SLOTS, WM1 + NUM_SPEC, C, device=DEV).float().transpose(-1, -2)
print(f"  ssm contig={ssm_nc.is_contiguous()} stride={ssm_nc.stride()} | "
      f"conv contig={conv_nc.is_contiguous()} stride={conv_nc.stride()}")
co = gdn_hip.causal_conv1d_fwd(xp, wt, None, cu_p, idx, has_init_p, conv_nc, 1)
q2, k2, v2 = split(co)
cp2 = gdn_hip.gdn_prefill_wmma(q2, k2, v2, ap, bp, A_log, dt_bias,
                               cu_p, idx, has_init_p, ssm_nc, scale, 1)
report("  prefill w/ strided state", core=cp2, ssm_slot=ssm_nc[SLOT], conv_slot=conv_nc[SLOT])
cov, _cs2 = gdn_hip.causal_conv1d_fwd_verify(
    xv, wt, None, cu_v, idx, torch.ones(1, dtype=torch.uint8, device=DEV), conv_nc, QLEN, 1)
q3, k3, v3 = split(cov)
cv2, ss2 = gdn_hip.gdn_prefill_verify(
    q3, k3, v3, av, bv, A_log, dt_bias, cu_v, idx,
    torch.ones(1, dtype=torch.uint8, device=DEV), ssm_nc, QLEN, scale, 1)
report("  verify  w/ strided state", conv_out=cov, core=cv2, ssm_scratch=ss2)
