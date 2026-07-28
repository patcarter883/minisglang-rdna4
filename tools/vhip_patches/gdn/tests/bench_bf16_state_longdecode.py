"""bf16 SSM-state LONG-DECODE accumulation check — the open question the prefill parity test can't
answer. In prefill the state stays fp32 in-register across the whole chunk (only the carried store
rounds), so output is bit-identical. In real DECODE the state round-trips through HBM every token:
read (bf16) -> compute fp32 in-register -> write (bf16). So per-step rounding is re-injected every
step and could accumulate over a long generation.

Method: two runs fed the IDENTICAL per-step input sequence from a common seed — one carrying fp32
state, one carrying bf16 state — stepped N tokens. fp32 is the reference. Track output rel error vs
step (does it grow?) and final-state rel. Gated decay exp(g), g<0 is contractive, so errors SHOULD
stay bounded; this measures whether that holds in practice at production sequence lengths.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "torch-ext"))
import gdn_hip as G  # noqa: E402

DEV = "cuda"
H, HV, K, V = 4, 8, 128, 128
SCALE = 1.0 / (K ** 0.5)


def run(B, N, state_dtype, gen):
    """Step N decode tokens carrying state in `state_dtype`; return list of per-step outputs (fp32)
    and the final state (as fp32). Inputs drawn from `gen` so two calls with a fresh-but-equal gen
    see the identical sequence."""
    ssm = torch.zeros(B + 1, HV, V, K, device=DEV, dtype=state_dtype)
    idx = torch.arange(1, B + 1, device=DEV, dtype=torch.long)
    outs = []
    for _ in range(N):
        q = torch.randn(B, H, K, device=DEV, generator=gen)
        k = torch.randn(B, H, K, device=DEV, generator=gen)
        v = torch.randn(B, HV, V, device=DEV, generator=gen)
        a = torch.randn(B, HV, device=DEV, generator=gen)
        b = torch.randn(B, HV, device=DEV, generator=gen)
        A_log = torch.randn(HV, device=DEV, generator=gen)
        dt_bias = torch.randn(HV, device=DEV, generator=gen)
        o = G.gdn_decode(q, k, v, a, b, A_log, dt_bias, ssm, idx, SCALE, 1)
        outs.append(o.float().clone())
    return outs, ssm.float().clone()


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


print(f"device: {torch.cuda.get_device_properties(0).gcnArchName} | torch {torch.__version__}")
print("bf16-state vs fp32-state decode trajectory (identical inputs); rel error per step\n")
for B in (1, 32):
    for N in (256, 1024, 4096):
        g1 = torch.Generator(device=DEV).manual_seed(1234)
        g2 = torch.Generator(device=DEV).manual_seed(1234)
        o32, s32 = run(B, N, torch.float32, g1)
        o16, s16 = run(B, N, torch.bfloat16, g2)
        # per-step output rel error at a few checkpoints + max over the run
        checkpts = sorted({0, N // 4, N // 2, N - 1})
        per_step = [rel(o16[t], o32[t]) for t in range(N)]
        mx = max(per_step)
        mx_at = per_step.index(mx)
        cp = "  ".join(f"@{t}={per_step[t]:.2e}" for t in checkpts)
        print(f"  B={B:<3} N={N:<5} {cp}   max={mx:.2e}@{mx_at}   final-state rel={rel(s16, s32):.2e}")
    print()
