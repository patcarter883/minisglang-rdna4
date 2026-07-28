"""Is gdn_prefill_wmma(use_l2norm=0) on PRE-normalized q/k equivalent to use_l2norm=1 on raw q/k?

The vhip spec+prefill shim passes use_l2norm=0 because upstream already l2-normalized q/k in
fused_post_conv_prep (and correspondingly calls the chunk op with use_qk_l2norm_in_kernel=False).
Every other gdn_hip call site passes 1. If the flag gates anything beyond the norm itself — e.g. the
q scaling — the shim silently computes the wrong thing, which is the remaining suspect for the
"one correct token then !!!!" corruption.
"""

import torch
import gdn_hip

torch.manual_seed(0)
DEV = "cuda"
NV, NK, HK, HV = 4, 4, 128, 128
SEQS = [7, 5]
T = sum(SEQS)
cu = torch.tensor([0, *torch.cumsum(torch.tensor(SEQS), 0).tolist()], dtype=torch.int32, device=DEV)

q = torch.randn(T, NK, HK, device=DEV).float()
k = torch.randn(T, NK, HK, device=DEV).float()
v = torch.randn(T, NV, HV, device=DEV).float()
a = torch.randn(T, NV, device=DEV).float()
b = torch.randn(T, NV, device=DEV).float()
A_log = torch.randn(NV, device=DEV).float()
dt_bias = torch.randn(NV, device=DEV).float()
si = torch.arange(len(SEQS), device=DEV, dtype=torch.long)
hi = torch.ones(len(SEQS), device=DEV, dtype=torch.uint8)
scale = HK ** -0.5


def l2(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)


# A: kernel does the norm itself, from RAW q/k  (the working call sites' convention)
sA = torch.zeros(len(SEQS), NV, HV, HK, device=DEV).float()
outA = gdn_hip.gdn_prefill_wmma(q, k, v, a, b, A_log, dt_bias, cu, si, hi, sA, scale, 1)

# B: caller pre-normalizes, kernel told not to  (what the shim does)
sB = torch.zeros(len(SEQS), NV, HV, HK, device=DEV).float()
outB = gdn_hip.gdn_prefill_wmma(
    l2(q).contiguous(), l2(k).contiguous(), v, a, b, A_log, dt_bias, cu, si, hi, sB, scale, 0
)

o_err = (outA - outB).abs().max().item()
s_err = (sA - sB).abs().max().item()
print(f"out  max|A-B| = {o_err:.6f}   (rel to |out| {outA.abs().max().item():.4f})")
print(f"state max|A-B| = {s_err:.6f}   (rel to |state| {sA.abs().max().item():.4f})")
if o_err > 1e-2 or s_err > 1e-2:
    print("\n=> use_l2norm=0 is NOT equivalent to pre-normalizing: the flag gates more than the")
    print("   norm (q scaling), so the shim must pass RAW q/k with use_l2norm=1.")
else:
    print("\n=> equivalent; use_l2norm=0 on pre-normalized input is fine. Look elsewhere.")
