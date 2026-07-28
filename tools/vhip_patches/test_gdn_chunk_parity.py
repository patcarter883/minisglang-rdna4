"""Direct parity: gdn_hip.gdn_prefill_wmma vs vLLM's Triton chunk_gated_delta_rule.

This is the exact substitution the vhip spec+prefill shim performs. Upstream hands the chunk op
post-gating (g, beta) + a dense initial_state; gdn_hip wants raw (a, b, A_log, dt_bias) + a
slot-indexed state. If the two disagree here, the shim's corruption ("one correct token then
!!!!") is a math/contract mismatch rather than anything to do with vLLM's spec plumbing.

g and beta are derived exactly as vLLM's fused_post_conv_prep does:
    g    = -exp(A_log) * softplus(a + dt_bias)
    beta = sigmoid(b)
"""

import torch
import torch.nn.functional as F
import gdn_hip
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule as fla_chunk

torch.manual_seed(0)
DEV = "cuda"
NK, NV, HK, HV = 4, 8, 128, 128       # v heads = 2x k heads, as in Qwen GDN
SEQS = [7, 5]
T = sum(SEQS)
cu = torch.tensor([0, *torch.cumsum(torch.tensor(SEQS), 0).tolist()], dtype=torch.int32, device=DEV)

# gdn_hip is dtype-generic (fp32/fp16/bf16) — run it in the SERVE's dtype, not a forced fp32.
DT = torch.bfloat16
q = torch.randn(T, NK, HK, device=DEV).to(DT)
k = torch.randn(T, NK, HK, device=DEV).to(DT)
v = torch.randn(T, NV, HV, device=DEV).to(DT)
a = torch.randn(T, NV, device=DEV).to(DT)
b = torch.randn(T, NV, device=DEV).to(DT)
A_log = torch.randn(NV, device=DEV).float()
dt_bias = torch.randn(NV, device=DEV).float()
scale = HK ** -0.5


def l2(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)


# ---- gdn_hip: raw q/k, kernel folds l2norm + gating ------------------------------------------
s_hip = torch.zeros(len(SEQS), NV, HV, HK, device=DEV).to(DT)
si = torch.arange(len(SEQS), device=DEV, dtype=torch.long)
hi = torch.ones(len(SEQS), device=DEV, dtype=torch.uint8)
out_hip = gdn_hip.gdn_prefill_wmma(
    q, k, v, a, b, A_log, dt_bias, cu, si, hi, s_hip, scale, 1
)

# ---- vLLM Triton chunk: pre-normalized q/k, precomputed g/beta --------------------------------
g = -torch.exp(A_log) * F.softplus(a + dt_bias)
beta = torch.sigmoid(b)
init = torch.zeros(len(SEQS), NV, HV, HK, device=DEV).to(DT)
out_tri, state_tri = fla_chunk(
    q=l2(q).unsqueeze(0).contiguous().to(DT),
    k=l2(k).unsqueeze(0).contiguous().to(DT),
    v=v.unsqueeze(0).contiguous().to(DT),
    g=g.unsqueeze(0).contiguous().float(),
    beta=beta.unsqueeze(0).contiguous().float(),
    initial_state=init,
    output_final_state=True,
    cu_seqlens=cu,
    use_qk_l2norm_in_kernel=False,
)
out_tri = out_tri.squeeze(0).float()
state_tri = state_tri.float()

out_hip = out_hip.float()
s_hip = s_hip.float()
o_err = (out_hip - out_tri).abs().max().item()
s_err = (s_hip - state_tri).abs().max().item()
print(f"OUT   max|hip-triton| = {o_err:.6f}   |out|={out_tri.abs().max().item():.4f}")
print(f"STATE max|hip-triton| = {s_err:.6f}   |state|={state_tri.abs().max().item():.4f}")
print()
if o_err < 5e-2 and s_err < 5e-2:
    print("=> substitution is sound; corruption is elsewhere (slicing / cu_seqlens / plumbing).")
else:
    print("=> MISMATCH: gdn_prefill_wmma is not a drop-in for the Triton chunk op as called.")
    print(f"   out rel err   = {o_err / max(out_tri.abs().max().item(), 1e-9):.3f}")
    print(f"   state rel err = {s_err / max(state_tri.abs().max().item(), 1e-9):.3f}")
