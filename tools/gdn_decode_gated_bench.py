"""Micro-bench: gdn_decode_gated (1 launch) vs gdn_decode + rmsnorm_gated (2 launches). bs=1 decode."""
import time, torch
import gdn_hip as gdn

dev = torch.device("cuda:0"); torch.manual_seed(0)
B, H, HV, K, V = 1, 2, 4, 128, 128
scale, eps, dt = K ** -0.5, 1e-6, torch.bfloat16
q = torch.randn(B, H, K, device=dev, dtype=dt) * 0.3
k = torch.randn(B, H, K, device=dev, dtype=dt) * 0.3
v = torch.randn(B, HV, V, device=dev, dtype=dt) * 0.3
a = torch.randn(B, HV, device=dev, dtype=dt) * 0.3
bb = torch.randn(B, HV, device=dev, dtype=dt) * 0.3
A_log = torch.randn(HV, device=dev, dtype=torch.float32) * 0.3
dt_bias = torch.randn(HV, device=dev, dtype=torch.float32) * 0.3
z = torch.randn(B, HV, V, device=dev, dtype=dt) * 0.5
nw = torch.randn(V, device=dev, dtype=torch.float32) * 0.3 + 1.0
ssm = torch.randn(2, HV, V, K, device=dev, dtype=torch.float32) * 0.1
si = torch.tensor([1], device=dev, dtype=torch.long)

def ref():
    core = gdn.gdn_decode(q, k, v, a, bb, A_log, dt_bias, ssm, si, scale, 1)
    return gdn.rmsnorm_gated(core.reshape(-1, V), z.reshape(-1, V), nw, eps)

def fused():
    return gdn.gdn_decode_gated(q, k, v, a, bb, A_log, dt_bias, ssm, si, z, nw, eps, scale, 1)

def bench(fn, n=2000):
    for _ in range(50): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6  # us/call

r = bench(ref); f = bench(fused)
print(f"ref (gdn_decode+rmsnorm_gated, 2 launches): {r:.2f} us/call")
print(f"fused (gdn_decode_gated, 1 launch):         {f:.2f} us/call")
print(f"SPEEDUP: {r/f:.2f}x  ({r-f:.2f} us saved/GDN-layer/token)")
