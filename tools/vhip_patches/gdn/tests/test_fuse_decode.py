"""Parity + perf for the fused GDN decode op `gdn_decode_conv_gated`.

Validates that the single fused kernel is BIT-EXACT (max|Δ|=0) vs the separate two-kernel decode chain
    causal_conv1d_update  ->  (split q/k/v)  ->  gdn_decode_gated
on the served-GDN decode shapes, for fp16 + bf16, at batch M in {1,4,8}. Also times per-token latency
(fused vs separate) and reports the launch-count reduction.

Runnable standalone inside the ROCm torch image after local/build_local.sh:
    PYTHONPATH=torch-ext python tests/test_fuse_decode.py
Exits nonzero on any parity failure (max|Δ| must be exactly 0 on out + both mutated states).
"""
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "torch-ext"))

import gdn_hip as G  # noqa: E402

DEV = "cuda"
FAILS = []

# served Qwen3.6-35B-A3B GDN dims (single-card / full model). K=V=128, W=4, GQA ratio HV/H = 2.
H, HV, K, V, W = 16, 32, 128, 128, 4
KEY_DIM = H * K
CONV_DIM = 2 * KEY_DIM + HV * V
EPS = 1e-5
SCALE = 1.0 / (K ** 0.5)


def split_qkv(conv_out, B):
    q = conv_out[:, :KEY_DIM].reshape(B, H, K).contiguous()
    k = conv_out[:, KEY_DIM:2 * KEY_DIM].reshape(B, H, K).contiguous()
    v = conv_out[:, 2 * KEY_DIM:].reshape(B, HV, V).contiguous()
    return q, k, v


def make_inputs(B, dtype, ssm_dtype, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s, dt=dtype: torch.randn(*s, generator=g, device=DEV, dtype=dt)
    num_slots = B + 2
    mixed_qkv = r(B, CONV_DIM)
    conv_weight = r(CONV_DIM, W, dt=torch.float32)
    conv_state = r(num_slots, CONV_DIM, W - 1, dt=torch.float32)
    a, b = r(B, HV), r(B, HV)
    A_log = r(HV, dt=torch.float32)
    dt_bias = r(HV, dt=torch.float32)
    ssm_state = torch.randn(num_slots, HV, V, K, generator=g, device=DEV, dtype=torch.float32).to(ssm_dtype)
    z = r(B, HV, V)
    norm_weight = r(V, dt=torch.float32)
    state_idx = torch.arange(1, B + 1, device=DEV, dtype=torch.long)  # slots 1..B (all active)
    return dict(mixed_qkv=mixed_qkv, conv_weight=conv_weight, conv_state=conv_state, a=a, b=b,
                A_log=A_log, dt_bias=dt_bias, ssm_state=ssm_state, z=z, norm_weight=norm_weight,
                state_idx=state_idx)


def run_separate(inp, B):
    cs = inp["conv_state"].clone()
    ss = inp["ssm_state"].clone()
    conv_out = G.causal_conv1d_update(inp["mixed_qkv"].contiguous(), inp["conv_weight"], None, cs,
                                      inp["state_idx"], 1)
    q, k, v = split_qkv(conv_out, B)
    z_flat = inp["z"].reshape(-1, V).contiguous()
    out = G.gdn_decode_gated(q, k, v, inp["a"].contiguous(), inp["b"].contiguous(), inp["A_log"],
                             inp["dt_bias"], ss, inp["state_idx"], z_flat, inp["norm_weight"], EPS,
                             SCALE, 1)
    return out, cs, ss


def run_fused(inp):
    cs = inp["conv_state"].clone()
    ss = inp["ssm_state"].clone()
    out = G.gdn_decode_conv_gated(inp["mixed_qkv"].contiguous(), inp["conv_weight"], None, cs,
                                  inp["a"].contiguous(), inp["b"].contiguous(), inp["A_log"],
                                  inp["dt_bias"], ss, inp["state_idx"], inp["z"].contiguous(),
                                  inp["norm_weight"], EPS, 1, SCALE, 1)
    return out, cs, ss


def maxabs(x, y):
    return (x.to(torch.float32) - y.to(torch.float32)).abs().max().item()


def parity(B, dtype, ssm_dtype):
    tag = f"M={B} {str(dtype).split('.')[-1]}/ssm={str(ssm_dtype).split('.')[-1]}"
    inp = make_inputs(B, dtype, ssm_dtype, seed=1234 + B)
    o_ref, cs_ref, ss_ref = run_separate(inp, B)
    o_fus, cs_fus, ss_fus = run_fused(inp)
    d_out = maxabs(o_ref, o_fus)
    d_cs = maxabs(cs_ref, cs_fus)
    d_ss = maxabs(ss_ref, ss_fus)
    ok = (d_out == 0.0) and (d_cs == 0.0) and (d_ss == 0.0)
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag:<28} max|Δ| out={d_out:.3e} conv_state={d_cs:.3e} ssm_state={d_ss:.3e}")
    if not ok:
        FAILS.append(tag)


def bench(B, dtype, ssm_dtype, iters=300, warmup=50):
    inp = make_inputs(B, dtype, ssm_dtype, seed=7 + B)
    z_flat = inp["z"].reshape(-1, V).contiguous()
    mq = inp["mixed_qkv"].contiguous()
    ac, bc = inp["a"].contiguous(), inp["b"].contiguous()

    def sep_step(cs, ss):
        conv_out = G.causal_conv1d_update(mq, inp["conv_weight"], None, cs, inp["state_idx"], 1)
        q, k, v = split_qkv(conv_out, B)
        return G.gdn_decode_gated(q, k, v, ac, bc, inp["A_log"], inp["dt_bias"], ss, inp["state_idx"],
                                  z_flat, inp["norm_weight"], EPS, SCALE, 1)

    def fus_step(cs, ss):
        return G.gdn_decode_conv_gated(mq, inp["conv_weight"], None, cs, ac, bc, inp["A_log"],
                                       inp["dt_bias"], ss, inp["state_idx"], inp["z"].contiguous(),
                                       inp["norm_weight"], EPS, 1, SCALE, 1)

    def timeit(fn):
        cs = inp["conv_state"].clone(); ss = inp["ssm_state"].clone()
        for _ in range(warmup):
            fn(cs, ss)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(cs, ss)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # us/token

    us_sep = timeit(sep_step)
    us_fus = timeit(fus_step)
    print(f"  M={B:<2} {str(dtype).split('.')[-1]:>4}: separate(2 launches)={us_sep:7.2f} us  "
          f"fused(1 launch)={us_fus:7.2f} us  speedup={us_sep / us_fus:4.2f}x")


def graph_capture(B, dtype, ssm_dtype):
    """Capture the fused op in a CUDA graph, replay, and require the replay output to be BIT-EXACT vs an
    eager fused step from the identical initial state (static shapes, in-place state mutation, no host sync)."""
    tag = f"M={B} {str(dtype).split('.')[-1]}/ssm={str(ssm_dtype).split('.')[-1]}"
    inp = make_inputs(B, dtype, ssm_dtype, seed=99 + B)
    cs0, ss0 = inp["conv_state"].clone(), inp["ssm_state"].clone()

    # eager reference from the initial state
    o_eager, cs_eager, ss_eager = run_fused(inp)

    # capture: persistent state buffers seeded to the initial state
    cs = cs0.clone(); ss = ss0.clone()
    mq = inp["mixed_qkv"].contiguous(); ac = inp["a"].contiguous(); bc = inp["b"].contiguous()
    zc = inp["z"].contiguous()
    def step():
        return G.gdn_decode_conv_gated(mq, inp["conv_weight"], None, cs, ac, bc, inp["A_log"],
                                       inp["dt_bias"], ss, inp["state_idx"], zc, inp["norm_weight"],
                                       EPS, 1, SCALE, 1)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            cs.copy_(cs0); ss.copy_(ss0); step()
    torch.cuda.current_stream().wait_stream(s)
    gph = torch.cuda.CUDAGraph()
    cs.copy_(cs0); ss.copy_(ss0)
    with torch.cuda.graph(gph):
        out_g = step()
    # replay from the initial state -> must reproduce the eager result bit-for-bit
    cs.copy_(cs0); ss.copy_(ss0)
    gph.replay()
    torch.cuda.synchronize()
    d_out = maxabs(o_eager, out_g)
    d_cs = maxabs(cs_eager, cs)
    d_ss = maxabs(ss_eager, ss)
    ok = torch.isfinite(out_g).all().item() and d_out == 0.0 and d_cs == 0.0 and d_ss == 0.0
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag:<28} replay vs eager max|Δ| out={d_out:.3e} conv_state={d_cs:.3e} ssm_state={d_ss:.3e}")
    if not ok:
        FAILS.append("graph " + tag)


def main():
    print(f"device: {torch.cuda.get_device_properties(0).gcnArchName} | torch {torch.__version__}")
    print(f"GDN dims: H(k-heads)={H} HV(v-heads)={HV} K={K} V={V} W={W} conv_dim={CONV_DIM} ratio={HV // H}")
    print("\n== PARITY: fused gdn_decode_conv_gated vs separate [causal_conv1d_update -> gdn_decode_gated] ==")
    for dtype in (torch.float16, torch.bfloat16):
        for ssm_dtype in (torch.float32, torch.bfloat16):
            for B in (1, 4, 8):
                parity(B, dtype, ssm_dtype)
    print("\n== CUDA-GRAPH CAPTURE: replay bit-exact vs eager (static shapes, in-place state) ==")
    for dtype in (torch.float16, torch.bfloat16):
        graph_capture(1, dtype, torch.float32)
        graph_capture(8, dtype, torch.bfloat16)
    print("\n== PERF: per-token decode latency (us), launches/token 2 -> 1 ==")
    for dtype in (torch.float16, torch.bfloat16):
        for B in (1, 8):
            bench(B, dtype, torch.float32)
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("ALL PARITY GREEN (max|Δ| = 0)")


if __name__ == "__main__":
    main()
