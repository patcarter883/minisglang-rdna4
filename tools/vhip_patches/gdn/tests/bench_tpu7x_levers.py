"""Size the three TPU7x GDN/attn levers BEFORE building anything (prize-first discipline):
  (A) conv1d fusion  — how much does the standalone causal_conv1d cost vs gdn_decode? That (plus the
                       conv_out HBM round-trip) is the ceiling on any fusion win.
  (B) bf16 SSM state — fp32-vs-bf16 state: forward parity at long context + decode/prefill latency.
  (C) attn page size — flash_decode_paged latency across page_size 1/16/32/64/128 (PROD default is 1).
                       Tested with BOTH contiguous and SHUFFLED block tables: page_size=1 in prod
                       scatters every token as its own page, so the shuffled number is the realistic
                       one; contiguous isolates pure index-amortization from de-scatter/coalescing.

Qwen3.5/3.6 GDN shape: H=4 k-heads, HV=8 v-heads, K=V=128 -> conv_dim = 2*H*K + HV*V = 2048.
"""
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "torch-ext"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "attn_decode", "torch-ext"))
import gdn_hip as G  # noqa: E402
try:
    import attn_decode as A  # noqa: E402
    HAVE_ATTN = True
except Exception as e:  # noqa: BLE001
    print(f"(attn_decode not importable: {e})")
    HAVE_ATTN = False

DEV = "cuda"
G_ = torch.Generator(device=DEV).manual_seed(0)
torch.manual_seed(0)
H, HV, K, V = 4, 8, 128, 128
CONV_DIM = 2 * H * K + HV * V  # 2048
W = 4


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # µs


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


print(f"device: {torch.cuda.get_device_properties(0).gcnArchName} | torch {torch.__version__}")
print(f"conv_dim={CONV_DIM}  (H={H} HV={HV} K={K} V={V} W={W})\n")

# ---------------------------------------------------------------- (A) conv fusion prize
print("== (A) conv1d fusion prize: standalone conv vs gdn_decode (µs/call) ==")
for B in (1, 8, 32, 64):
    x = torch.randn(B, CONV_DIM, device=DEV)
    wt = torch.randn(CONV_DIM, W, device=DEV)
    cstate = torch.randn(B + 1, CONV_DIM, W - 1, device=DEV)
    cidx = torch.arange(1, B + 1, device=DEV, dtype=torch.long)
    t_conv = bench(lambda: G.causal_conv1d_update(x, wt, None, cstate, cidx, 1))
    scale = 1.0 / (K ** 0.5)
    q = torch.randn(B, H, K, device=DEV); k = torch.randn(B, H, K, device=DEV)
    v = torch.randn(B, HV, V, device=DEV)
    a = torch.randn(B, HV, device=DEV); b = torch.randn(B, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV); dt_bias = torch.randn(HV, device=DEV)
    ssm = torch.randn(B + 1, HV, V, K, device=DEV)
    idx = torch.arange(1, B + 1, device=DEV, dtype=torch.long)
    t_dec = bench(lambda: G.gdn_decode(q, k, v, a, b, A_log, dt_bias, ssm, idx, scale, 1))
    frac = 100.0 * t_conv / (t_conv + t_dec)
    print(f"  B={B:<3} conv={t_conv:7.2f}  gdn_decode={t_dec:7.2f}  conv share={frac:4.1f}%  (fusion ceiling)")

# ---------------------------------------------------------------- (B) bf16 SSM state
print("\n== (B) bf16 SSM state: forward parity (fp32 state vs bf16 state) at length T ==")
for T in (128, 512, 2048):
    scale = 1.0 / (K ** 0.5)
    q = torch.randn(T, H, K, device=DEV); k = torch.randn(T, H, K, device=DEV)
    v = torch.randn(T, HV, V, device=DEV)
    a = torch.randn(T, HV, device=DEV); b = torch.randn(T, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV); dt_bias = torch.randn(HV, device=DEV)
    cu = torch.tensor([0, T], device=DEV, dtype=torch.int32)
    idx = torch.tensor([1], device=DEV, dtype=torch.long)
    has0 = torch.zeros(1, device=DEV, dtype=torch.uint8)
    ssm32 = torch.zeros(2, HV, V, K, device=DEV, dtype=torch.float32)
    ssm16 = torch.zeros(2, HV, V, K, device=DEV, dtype=torch.bfloat16)
    o32 = G.gdn_prefill(q, k, v, a, b, A_log, dt_bias, cu, idx, has0, ssm32, scale, 1)
    o16 = G.gdn_prefill(q, k, v, a, b, A_log, dt_bias, cu, idx, has0, ssm16, scale, 1)
    print(f"  T={T:<5} out rel(bf16-state vs fp32-state)={rel(o16, o32):.3e}   "
          f"final-state rel={rel(ssm16, ssm32):.3e}")
print("  -- decode latency fp32 vs bf16 state --")
for B in (1, 32, 64):
    scale = 1.0 / (K ** 0.5)
    q = torch.randn(B, H, K, device=DEV); k = torch.randn(B, H, K, device=DEV)
    v = torch.randn(B, HV, V, device=DEV)
    a = torch.randn(B, HV, device=DEV); b = torch.randn(B, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV); dt_bias = torch.randn(HV, device=DEV)
    idx = torch.arange(1, B + 1, device=DEV, dtype=torch.long)
    s32 = torch.randn(B + 1, HV, V, K, device=DEV, dtype=torch.float32)
    s16 = torch.randn(B + 1, HV, V, K, device=DEV, dtype=torch.bfloat16)
    t32 = bench(lambda: G.gdn_decode(q, k, v, a, b, A_log, dt_bias, s32, idx, scale, 1))
    t16 = bench(lambda: G.gdn_decode(q, k, v, a, b, A_log, dt_bias, s16, idx, scale, 1))
    print(f"  B={B:<3} fp32-state={t32:7.2f}  bf16-state={t16:7.2f}  speedup={t32 / t16:4.2f}x")

# ---------------------------------------------------------------- (C) attn page size
if HAVE_ATTN:
    print("\n== (C) flash_decode_paged latency vs page_size (µs/call, Bq=64 Hq=32 Hk=8 D=128 ctx=4096) ==")
    print("   contiguous = best case per page size; shuffled = realistic (prod page_size=1 scatters)")
    Bq, Hq, Hk, D = 64, 32, 8, 128
    CTX = 4096
    scale = 1.0 / (D ** 0.5)
    for ps in (1, 16, 32, 64, 128, 256):
        nbp = (CTX + ps - 1) // ps
        total = Bq * nbp + 8
        q = torch.randn(Bq, Hq, D, device=DEV, dtype=torch.bfloat16)
        kc = torch.randn(total, ps, Hk, D, device=DEV, dtype=torch.bfloat16)
        vc = torch.randn(total, ps, Hk, D, device=DEV, dtype=torch.bfloat16)
        clen = torch.full((Bq,), CTX, device=DEV, dtype=torch.int32)
        bt_contig = torch.arange(Bq * nbp, device=DEV, dtype=torch.int32).reshape(Bq, nbp)
        perm = torch.randperm(Bq * nbp, generator=G_, device=DEV).to(torch.int32)
        bt_shuf = perm.reshape(Bq, nbp)
        spec = "specialized" if ps in (1, 16, 32, 64, 128, 256) else "runtime-fallback"
        try:
            tc = bench(lambda: A.flash_decode_paged(q, kc, vc, bt_contig, clen, scale, 0, 0))
            ts = bench(lambda: A.flash_decode_paged(q, kc, vc, bt_shuf, clen, scale, 0, 0))
            print(f"  page_size={ps:<4} contiguous={tc:7.2f}  shuffled={ts:7.2f}   ({spec})")
        except Exception as e:  # noqa: BLE001
            print(f"  page_size={ps:<4} FAILED: {e}")
