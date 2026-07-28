"""GPU parity gate for the W4A16 grouped-MoE decode path (_run_grouped_moe_w4a16).

Runs INSIDE the ROCm image (needs w4a8_fp8_wmma + vLLM's moe_align/activation helpers):

    VHIP_CMD='export PYTHONPATH=/opt/kernels; python3 /engine/tools/vhip_patches/test_w4a16_moe.py' \
    gpu-lease -n 1 -- docker compose --profile vhip run --rm --no-deps vhip

What it proves, on the REAL Qwen3.6-35B-A3B TP=2 expert shapes:
  1. the W4A16 composition matches a pure-torch fp16 dequant reference (the kernel applies only
     the weight scale, so this is a tight tolerance -- unlike the fp8-act path, which quantises x);
  2. it beats that same reference's error against the fp8-activation path, i.e. W4A16 really is the
     more accurate of our two kernels and the swap is an accuracy WIN, not just a speed bet;
  3. the asymmetric (has_zp) and symmetric (implicit uint4b8 zp=8) weight conventions both decode
     correctly -- Qwen's MoeWNA16 hook reports has_zp=True, GLM/CT report symmetric.

A parity failure here means the served MoE is silently wrong, which is worse than slow.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("moe_experts", os.path.join(HERE, "moe_experts.py"))
moe = importlib.util.module_from_spec(spec)
sys.modules["moe_experts"] = moe
spec.loader.exec_module(moe)

DEV = "cuda"
G = 32  # group size (Qwen3.6-35B-A3B AWQ/CT)


def pack_int4(vals: torch.Tensor) -> torch.Tensor:
    """(E, N, K) uint4 values in [0,15] -> (E, N, K//8) int32, nibble j = input k8*8+j."""
    E, N, K = vals.shape
    out = torch.zeros((E, N, K // 8), dtype=torch.int32, device=vals.device)
    for j in range(8):
        out |= (vals[:, :, j::8].to(torch.int32) & 0xF) << (4 * j)
    return out


def pack_zeros(zp: torch.Tensor) -> torch.Tensor:
    """(E, N, K//G) uint4 zeros -> (E, N//8, K//G) int32, nibble j = zp of output 8*n8+j."""
    E, N, Kg = zp.shape
    out = torch.zeros((E, N // 8, Kg), dtype=torch.int32, device=zp.device)
    for j in range(8):
        out |= (zp[:, j::8, :].to(torch.int32) & 0xF) << (4 * j)
    return out


def make_expert_weights(E, N, K, has_zp, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randint(0, 16, (E, N, K), generator=g, device=DEV, dtype=torch.int32)
    scales = (torch.rand((E, N, K // G), generator=g, device=DEV) * 0.02 + 0.005).to(torch.float16)
    if has_zp:
        zp = torch.randint(1, 15, (E, N, K // G), generator=g, device=DEV, dtype=torch.int32)
        zp_packed = pack_zeros(zp)
    else:
        zp = torch.full((E, N, K // G), 8, device=DEV, dtype=torch.int32)  # implicit uint4b8
        zp_packed = None
    # fp16 dequant reference: (q - zp) * scale, zp/scale broadcast over the group.
    deq = ((q - zp.repeat_interleave(G, dim=2)).to(torch.float16)
           * scales.repeat_interleave(G, dim=2))
    return pack_int4(q), scales, zp_packed, deq


def reference_moe(x, w13_deq, w2_deq, topk_weights, topk_ids):
    """Pure-torch gated-SiLU MoE in fp16 activations / fp32 accumulation."""
    M, K = x.shape
    top_k = topk_ids.size(1)
    out = torch.zeros((M, K), dtype=torch.float32, device=x.device)
    for m in range(M):
        for t in range(top_k):
            e = int(topk_ids[m, t])
            h = (x[m].float() @ w13_deq[e].float().t())        # (2*inter,)
            gate, up = h.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            y = act.half().float() @ w2_deq[e].float().t()     # (K,)
            out[m] += y * float(topk_weights[m, t])
    return out


def run_case(name, M, E, K, inter, top_k, has_zp):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    w13, w13_s, w13_z, w13_deq = make_expert_weights(E, 2 * inter, K, has_zp, seed=1)
    w2, w2_s, w2_z, w2_deq = make_expert_weights(E, K, inter, has_zp, seed=2)

    g = torch.Generator(device=DEV).manual_seed(3)
    x = (torch.randn((M, K), generator=g, device=DEV) * 0.5).to(torch.float16)
    logits = torch.randn((M, E), generator=g, device=DEV)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    ref = reference_moe(x, w13_deq, w2_deq, topk_weights, topk_ids)

    ours = moe._run_grouped_moe_w4a16(
        x, w13, w2, w13_s, w2_s, w13_z, w2_z, topk_weights, topk_ids,
        MoEActivation.SILU, E, None, False, out_dtype=torch.float16).float()

    # the fp8-activation path, for the accuracy comparison (kernel="wmma", W4A16 forced off)
    os.environ["VLLM_ROCM_W4A16_MOE"] = "off"
    try:
        fp8 = moe._run_grouped_moe(
            x, w13, w2, w13_s, w2_s, w13_z, w2_z, topk_weights, topk_ids,
            MoEActivation.SILU, E, None, False, "wmma", out_dtype=torch.float16).float()
    finally:
        os.environ["VLLM_ROCM_W4A16_MOE"] = "auto"

    def rel(a):
        return ((a - ref).norm() / ref.norm()).item()

    r_ours, r_fp8 = rel(ours), rel(fp8)
    cos = torch.nn.functional.cosine_similarity(
        ours.flatten(), ref.flatten(), dim=0).item()
    ok = r_ours < 0.02 and r_ours < r_fp8 and cos > 0.999
    print(f"  {name:34s} M={M:<4d} has_zp={int(has_zp)}  "
          f"rel(W4A16)={r_ours:.5f}  rel(fp8-act)={r_fp8:.5f}  cos={cos:.6f}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(0)
    # Qwen3.6-35B-A3B at TP=2: hidden 2048, moe_intermediate 512 -> 256/shard, top_k 8.
    # E is cut to 32 so the pure-torch reference stays tractable; the kernel indexes experts
    # identically at 32 and 256 (expert_ids is per padded block either way).
    cases = [
        ("qwen35b tp2 decode bs=1", 1, 32, 2048, 256, 8, True),
        ("qwen35b tp2 decode bs=8", 8, 32, 2048, 256, 8, True),
        ("qwen35b tp2 symmetric", 4, 32, 2048, 256, 8, False),
        ("qwen35b tp2 batched M=64", 64, 32, 2048, 256, 8, True),
    ]
    print("W4A16 grouped-MoE parity (vs pure-torch fp16 dequant reference)")
    results = [run_case(*c) for c in cases]

    print("\nno-fallback contract:")
    bad = 0
    try:
        x = torch.randn((1, 2048), device=DEV, dtype=torch.bfloat16)
        w13, w13_s, w13_z, _ = make_expert_weights(4, 512, 2048, True, seed=1)
        w2, w2_s, w2_z, _ = make_expert_weights(4, 2048, 256, True, seed=2)
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        moe._run_grouped_moe_w4a16(
            x, w13, w2, w13_s, w2_s, w13_z, w2_z,
            torch.ones((1, 2), device=DEV), torch.zeros((1, 2), device=DEV, dtype=torch.int32),
            MoEActivation.SILU, 4, None, False)
        print("  bf16 activations             FAIL (returned instead of raising)")
        bad = 1
    except RuntimeError as e:
        print(f"  bf16 activations             PASS (raised: {str(e)[:70]}...)")

    print(f"\n{sum(results)}/{len(results)} parity cases passed"
          f"{'' if not bad else '; contract check FAILED'}")
    sys.exit(0 if all(results) and not bad else 1)


if __name__ == "__main__":
    main()
