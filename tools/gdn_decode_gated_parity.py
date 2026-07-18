"""Parity: gdn_decode_gated (fused) == gdn_decode + rmsnorm_gated (reference). BIT-EXACT expected.
Checks BOTH the normed output AND the in-place ssm_state recurrent mutation (must match exactly).
Run one card:  gpu-lease -n 1 --timeout 600 -- docker run ... gdn_decode_gated_parity.py
"""
import sys, torch
import gdn_hip as gdn

dev = torch.device("cuda:0")
torch.manual_seed(0)
B, H, HV, K, V = 1, 2, 4, 128, 128
scale = K ** -0.5
eps = 1e-6
dt = torch.bfloat16

def mk():
    q = torch.randn(B, H, K, device=dev, dtype=dt) * 0.3
    k = torch.randn(B, H, K, device=dev, dtype=dt) * 0.3
    v = torch.randn(B, HV, V, device=dev, dtype=dt) * 0.3
    a = torch.randn(B, HV, device=dev, dtype=dt) * 0.3
    b = torch.randn(B, HV, device=dev, dtype=dt) * 0.3
    A_log = torch.randn(HV, device=dev, dtype=torch.float32) * 0.3
    dt_bias = torch.randn(HV, device=dev, dtype=torch.float32) * 0.3
    z = torch.randn(B, HV, V, device=dev, dtype=dt) * 0.5
    nw = torch.randn(V, device=dev, dtype=torch.float32) * 0.3 + 1.0
    ssm = torch.randn(2, HV, V, K, device=dev, dtype=torch.float32) * 0.1  # slot 1 used
    si = torch.tensor([1], device=dev, dtype=torch.long)
    return q, k, v, a, b, A_log, dt_bias, z, nw, ssm, si

q, k, v, a, b, A_log, dt_bias, z, nw, ssm, si = mk()

# ---- reference: gdn_decode then rmsnorm_gated (separate) ----
ssm_ref = ssm.clone()
core = gdn.gdn_decode(q, k, v, a, b, A_log, dt_bias, ssm_ref, si, scale, 1)   # [B,HV,V], mutates ssm_ref
normed_ref = gdn.rmsnorm_gated(core.reshape(-1, V), z.reshape(-1, V), nw, eps).reshape(B, HV, V)

# ---- fused ----
ssm_fus = ssm.clone()
normed_fus = gdn.gdn_decode_gated(q, k, v, a, b, A_log, dt_bias, ssm_fus, si, z, nw, eps, scale, 1)

out_eq = torch.equal(normed_fus, normed_ref)
state_eq = torch.equal(ssm_fus, ssm_ref)
out_maxerr = (normed_fus.float() - normed_ref.float()).abs().max().item()
state_maxerr = (ssm_fus - ssm_ref).abs().max().item()
print(f"OUTPUT   bit-exact={out_eq}  max|Δ|={out_maxerr:.3e}")
print(f"SSMSTATE bit-exact={state_eq}  max|Δ|={state_maxerr:.3e}")
ok = out_eq and state_eq
print("PARITY:", "PASS (bit-exact)" if ok else "FAIL")
sys.exit(0 if ok else 1)
