"""gdn_prefill_wmma vs a pure-PyTorch gated-delta-rule reference, in the EXACT call convention the
vhip spec+prefill shim uses.

No Triton, so this runs in seconds instead of paying the 19-minute chunk_scaled_dot_kkt autotune.
Recurrence transcribed from vLLM's own fused_sigmoid_gating Triton kernel:

    g    = -exp(A_log) * softplus(a + dt_bias)      beta = sigmoid(b)
    S   *= exp(g)
    v   -= (S * k).sum(-1)                          # S:[V,K], k:[K] -> v:[V]
    v   *= beta
    S   += v[:,None] * k[None,:]
    o    = (S * q).sum(-1)

with q/k l2-normalized and q scaled by head_k_dim**-0.5. Sweeps the dtype combinations the shim
straddles (fp32 acts + bf16 state is what the serve actually produces).
"""

import torch
import torch.nn.functional as F
import gdn_hip

torch.manual_seed(0)
DEV = "cuda"
NK, NV, HK, HV = 4, 8, 128, 128        # v heads = 2x k heads (GQA), as in Qwen GDN
SEQS = [7, 5]
T = sum(SEQS)
cu = torch.tensor([0, *torch.cumsum(torch.tensor(SEQS), 0).tolist()], dtype=torch.int32, device=DEV)
scale = HK ** -0.5


def l2(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + 1e-6)


def reference(q, k, v, a, b, A_log, dt_bias, prenormed):
    """fp64 reference. Returns (out[T,NV,HV], final_state[S,NV,HV,HK])."""
    q, k, v = q.double(), k.double(), v.double()
    a, b = a.double(), b.double()
    g = -torch.exp(A_log.double()) * F.softplus(a + dt_bias.double())
    beta = torch.sigmoid(b)
    if not prenormed:
        q, k = l2(q), l2(k)
    q = q * scale
    out = torch.zeros(T, NV, HV, dtype=torch.float64, device=DEV)
    final = torch.zeros(len(SEQS), NV, HV, HK, dtype=torch.float64, device=DEV)
    rep = NV // NK
    for s in range(len(SEQS)):
        lo, hi = int(cu[s]), int(cu[s + 1])
        S = torch.zeros(NV, HV, HK, dtype=torch.float64, device=DEV)
        for t in range(lo, hi):
            for h in range(NV):
                kh = h // rep
                S[h] *= torch.exp(g[t, h])
                vv = v[t, h] - (S[h] * k[t, kh][None, :]).sum(-1)
                vv = vv * beta[t, h]
                S[h] += vv[:, None] * k[t, kh][None, :]
                out[t, h] = (S[h] * q[t, kh][None, :]).sum(-1)
        final[s] = S
    return out, final


base_q = torch.randn(T, NK, HK, device=DEV)
base_k = torch.randn(T, NK, HK, device=DEV)
base_v = torch.randn(T, NV, HV, device=DEV)
base_a = torch.randn(T, NV, device=DEV)
base_b = torch.randn(T, NV, device=DEV)
A_log = torch.randn(NV, device=DEV)
dt_bias = torch.randn(NV, device=DEV)
si = torch.arange(len(SEQS), device=DEV, dtype=torch.long)
hi_ = torch.ones(len(SEQS), device=DEV, dtype=torch.uint8)

# (label, activation dtype, state dtype, pre-normalize q/k + use_l2norm flag)
CASES = [
    ("fp32 acts / fp32 state / kernel-l2norm", torch.float32, torch.float32, False),
    ("fp32 acts / bf16 state / kernel-l2norm  <- SERVE combo", torch.float32, torch.bfloat16, False),
    ("fp32 acts / bf16 state / PRE-normed     <- SHIM combo", torch.float32, torch.bfloat16, True),
    ("bf16 acts / bf16 state / PRE-normed", torch.bfloat16, torch.bfloat16, True),
]

for label, adt, sdt, prenorm in CASES:
    q = (l2(base_q) if prenorm else base_q).to(adt).contiguous()
    k = (l2(base_k) if prenorm else base_k).to(adt).contiguous()
    v = base_v.to(adt).contiguous()
    a = base_a.to(adt).contiguous()
    b = base_b.to(adt).contiguous()
    st = torch.zeros(len(SEQS), NV, HV, HK, device=DEV, dtype=sdt)
    out = gdn_hip.gdn_prefill_wmma(
        q, k, v, a, b, A_log.float(), dt_bias.float(), cu, si, hi_, st,
        scale, 0 if prenorm else 1,
    )
    ref_o, ref_s = reference(
        (l2(base_q) if prenorm else base_q), (l2(base_k) if prenorm else base_k),
        base_v, base_a, base_b, A_log, dt_bias, prenorm,
    )
    o_err = (out.float() - ref_o.float()).abs().max().item()
    s_err = (st.float() - ref_s.float()).abs().max().item()
    # head_v_dim == head_k_dim == 128, so a transposed state is SHAPE-invisible: compare both.
    s_err_T = (st.float() - ref_s.float().transpose(-1, -2)).abs().max().item()
    verdict = "OK " if (o_err < 5e-2 and min(s_err, s_err_T) < 5e-2) else "BAD"
    lay = "as-is" if s_err <= s_err_T else "TRANSPOSED"
    print(f"[{verdict}] {label:52s} out_err={o_err:.5f}  state_err={s_err:.5f}  "
          f"state_err_T={s_err_T:.5f}  -> best={lay}")
