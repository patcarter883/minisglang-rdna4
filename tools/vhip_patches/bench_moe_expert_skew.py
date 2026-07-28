"""Measure the MoE expert-SKEW tax — the thing a persistent tile scheduler would recover.

WHY THIS EXISTS (decision gate, not a microbenchmark for its own sake)
    Our decode-GEMV launchers size the grid from the problem: `dim3 grid(N/per_block, P/block_m)`.
    The second dim is one block per padded expert-block, and each of those blocks does a DIFFERENT
    amount of work, because routing hands experts unequal token counts. With a static grid, blocks
    holding 1 real row retire while blocks holding 8 are still running, and the kernel finishes at
    the SLOWEST block. A persistent kernel with an atomic work counter recovers exactly that
    imbalance -- and nothing else. So the honest question before building one is: how big is it?

    Note this is orthogonal to BYLANE. BYLANE fixed INTRA-wave lane starvation; a tile scheduler
    fixes INTER-block imbalance. Neither substitutes for the other.

METHOD
    Pin the ACTIVE EXPERT SET and vary only how rows are DISTRIBUTED across it:
      balanced : every active expert gets the same count (the ceiling a scheduler could reach)
      skewed   : ~2.5x hot/mean, typical of a real router
      zipf     : harsh power law, a pessimistic bound
    Pinning the set is the whole control. A first version varied only "total routed rows" and
    found skew to be FASTER (0.66x) -- because concentrating routing touches FEWER distinct
    experts and so reads less weight. That measured weight traffic, not imbalance. Only the
    imbalance is recoverable by a work queue, so only the imbalance may vary.
    Reported `ntp` is the real (unpadded) sorted length; the ALLOCATED P is worst-cased by
    moe_align regardless of distribution, so it is not a useful signal here.

    VHIP_CMD='export PYTHONPATH=/opt/kernels; python3 /engine/tools/vhip_patches/bench_moe_expert_skew.py' \
    gpu-lease -n 1 -- docker compose --profile vhip run --rm --no-deps vhip
"""
from __future__ import annotations

import torch
import w4a8_fp8_wmma
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

DEV = "cuda"
G = 32
E = 256           # Qwen3.6-35B-A3B
HIDDEN = 2048
INTER = 256       # per TP=2 shard
TOP_K = 8
BLOCK_M = 8


def build_w(N, K, n_experts):
    g = torch.Generator(device=DEV).manual_seed(0)
    w = torch.randint(-(2**31), 2**31 - 1, (n_experts, N, K // 8), generator=g, device=DEV,
                      dtype=torch.int32)
    s = (torch.rand((n_experts, N, K // G), generator=g, device=DEV) * 0.02 + 0.005).to(torch.float16)
    z = torch.randint(0, 2**31 - 1, (n_experts, N // 8, K // G), generator=g, device=DEV,
                      dtype=torch.int32)
    return w, s, z


def routing(kind: str, M: int, n_active: int, seed: int = 0) -> torch.Tensor:
    """(M, TOP_K) int32 expert ids over a FIXED SET of `n_active` experts.

    Pinning the active-expert SET is the whole control. A first version of this benchmark varied
    only "total routed rows" and found skew to be FASTER (0.66x) -- because concentrating the
    routing touches fewer distinct experts and therefore reads less weight. That measured
    weight traffic, not imbalance. Holding the expert set fixed and varying only how rows are
    DISTRIBUTED across it isolates the imbalance, which is the only part a work queue recovers.
    """
    rows = M * TOP_K
    if kind == "balanced":
        counts = [rows // n_active] * n_active
        for i in range(rows - sum(counts)):
            counts[i] += 1
    elif kind == "skewed":          # ~2.5x hot/mean, typical of a real router
        w = [1.0 / (i + 1) ** 0.7 for i in range(n_active)]
        tot = sum(w)
        counts = [max(1, int(rows * x / tot)) for x in w]
    elif kind == "zipf":            # harsh power law: pessimistic bound
        w = [1.0 / (i + 1) ** 1.4 for i in range(n_active)]
        tot = sum(w)
        counts = [max(1, int(rows * x / tot)) for x in w]
    else:
        raise ValueError(kind)
    while sum(counts) > rows:       # trim/pad to exactly `rows`
        counts[counts.index(max(counts))] -= 1
    while sum(counts) < rows:
        counts[counts.index(max(counts))] += 1
    ids = torch.cat([torch.full((c,), e, dtype=torch.int32) for e, c in enumerate(counts) if c > 0])
    return ids.to(DEV).reshape(M, TOP_K).contiguous()


def time_call(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000.0  # us


def run(M: int):
    w2, s2, z2 = build_w(HIDDEN, INTER, E)
    # Every distribution touches the SAME number of experts, so weight bytes read are equal and
    # the only difference left is how lumpily the rows sit across blocks.
    n_active = min(E, max(TOP_K, M * TOP_K // 8))
    print(f"\n=== M={M} tokens, top_k={TOP_K}, {n_active} ACTIVE experts (fixed), "
          f"gemm2 (K={INTER}, N={HIDDEN})")
    print(f"{'routing':>9} {'ntp':>7} {'max/mean rows':>14} {'us':>8}  {'skew tax':>10}")
    base = None
    for kind in ("balanced", "skewed", "zipf"):
        ids = routing(kind, M, n_active)
        sorted_ids, expert_ids, ntp = moe_align_block_size(
            ids, BLOCK_M, E, None, pad_sorted_ids=True, ignore_invalid_experts=True)
        P = sorted_ids.size(0)
        counts = torch.bincount(ids.flatten().long(), minlength=E).float()
        hot = counts[counts > 0]
        lumpiness = (hot.max() / hot.mean()).item()
        x = torch.randn((P, INTER), device=DEV, dtype=torch.float16)

        def call():
            return w4a8_fp8_wmma.mmq_regdirect_w4a16_moe_gemv(
                x, w2, s2, sorted_ids, expert_ids, ntp, HIDDEN, 1, BLOCK_M, w_zeros=z2)

        us = time_call(call)
        rel = "" if base is None else f"{us / base:+.2f}x"
        if base is None:
            base = us
        print(f"{kind:>9} {int(ntp.item()):>7} {lumpiness:>14.2f} {us:>8.1f}  {rel:>10}")


def main():
    print("MoE expert-skew tax — how much a persistent tile scheduler could recover")
    print("(balanced = the ceiling a perfect scheduler reaches; gap to skewed = the prize)")
    for M in (1, 8, 16, 32, 64, 96):   # the M range the GEMV actually serves (gemv_max_m=96)
        run(M)
    print("""
MEASURED VERDICT (2026-07-28, Qwen3.6-35B-A3B TP=2 gemm2 shape): DO NOT build the work queue.
  The time increase tracks `ntp` (+48% rows balanced->skewed) and NOT lumpiness (11-14x max/mean).
  That means the tax is PADDING, not inter-block imbalance -- and a persistent tile scheduler
  recovers imbalance only.
  WHY: in a weight-bound GEMV a block's cost is set by its WEIGHT READ, near-independent of how
  many real rows it holds (8 real rows ~= 1: 21.8us here vs 19.2us in-serve at M_real=1). So
  blocks are near-equal cost however lumpy the routing, and there is little variance to balance.
  Skew does not make blocks UNEVEN, it makes MORE of them: an expert whose rows overflow block_m
  spans extra blocks, and EACH of those re-reads that expert's weights.
  THE REAL LEVER is therefore duplicate weight reads, not scheduling -- give hot experts a larger
  block_m (or one block per expert) so their weight row is read once, not once per block.""")


if __name__ == "__main__":
    main()
