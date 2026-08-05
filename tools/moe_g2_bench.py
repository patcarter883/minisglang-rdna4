"""Isolated bench of the MoE decode gemm2 ARM THE ENGINE ACTUALLY DISPATCHES, per M.

`tools/moe_g2_served_probe.py` established (real Qwen3.6-35B-A3B-AWQ TP=2 shapes, through
`minisgl.quant.kernels.w4a8_moe`) that the gemm2 arm is NOT one kernel:

    M <= 2   ->  fp8_wmma.mmq_fp8_moe_gemm_scatter          (split_k = 4 at M==1, 1 at M==2)
    M >= 3   ->  fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce   (grid = (ceil(N/256), M), no split)

so the served points M=5 and M=6 are on the FUSED gather-reduce, and M=1 is on the scatter. This
bench times each arm at its own M with everything else (route, align, gemm1, silu) hoisted out, so a
change to gemm2 is not diluted by the rest of the MoE.

MALL DISCIPLINE. The gemm2 weight slab actually touched is only `distinct_experts x N x K/2` bytes
(~2 MB at M=1), which sits inside the 64 MB last-level cache and would be fully resident after the
first iteration — timing that measures an L2 replay, not the serve. So the loop cycles `--variants`
INDEPENDENT routes (each with its own sorted_ids/expert_ids/buf2), rotating the expert set past the
MALL by byte count the way the served path does.

  PYTHONPATH=/opt/kernels:/engine/python:/engine python tools/moe_g2_bench.py --m 1,2,3,5,6,8,16,32
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HIDDEN = 2048          # Qwen3.6-35B-A3B text_config.hidden_size
INTER_FULL = 512       # text_config.moe_intermediate_size
E = 256                # text_config.num_experts
TOP_K = 8              # text_config.num_experts_per_tok
GROUP = 32             # quantization_config.group_size
BLOCK_M = 16           # _moe_block_m(M<=32, E=256, top_k=8) == 16


def build_weights(dev, inter):
    """w2 (down-proj) only — this bench times gemm2. Real layout, random values."""
    w2 = torch.randint(-(2**31), 2**31 - 1, (E, HIDDEN, inter // 8), dtype=torch.int32, device=dev)
    w2_s = torch.rand((E, HIDDEN, inter // GROUP), device=dev, dtype=torch.float16) * 0.02 + 0.001
    w2_z = torch.full((E, HIDDEN // 8, inter // GROUP), 0x88888888 - (1 << 32),
                      dtype=torch.int32, device=dev)
    return w2, w2_s, w2_z


def build_route(M, dev, dtype, inter, gen):
    """One independent route + its pre-sorted buf2, exactly as w4a8_moe hands them to gemm2."""
    import moe_hip

    gate = torch.randn((M, E), device=dev, dtype=dtype, generator=gen)
    tw, ti, sorted_ids, expert_ids, ntp = moe_hip.moe_route_align(gate, TOP_K, True, E, BLOCK_M)
    P = sorted_ids.shape[0]
    buf2 = (torch.randn((P, inter), device=dev, dtype=dtype, generator=gen) * 0.05).contiguous()
    tw_flat = tw.reshape(-1).float().contiguous()
    return dict(buf2=buf2, sorted_ids=sorted_ids, expert_ids=expert_ids, ntp=ntp,
                tw_flat=tw_flat, P=P)


def nx_blocks(M: int) -> int:
    """Unsplit block count of the fused gather-reduce: BYLANE tiles N by NWARPS*32 = 256."""
    return ((HIDDEN + 255) // 256) * M


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn(0)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for i in range(iters):
        fn(i)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0  # us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="1,2,3,5,6,8,16,32")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--variants", type=int, default=64, help="independent routes cycled to defeat MALL")
    ap.add_argument("--reps", type=int, default=3, help="repeat the whole sweep; report the median")
    ap.add_argument("--json", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--scatter-sk", default="", help="sweep the M<=2 scatter arm over these split_k")
    ap.add_argument("--fused-sk", default="", help="sweep the M>=3 fused arm over these split_k")
    args = ap.parse_args()

    import fp8_wmma

    dev = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    inter = INTER_FULL // args.tp
    gen = torch.Generator(device=dev).manual_seed(1234)

    have_sk = "split_k" in getattr(fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce, "__doc__", "") or ""
    print(f"[bench] fp8_wmma from {fp8_wmma.__file__}")
    print(f"[bench] shape: hidden(N)={HIDDEN} inter(K)={inter} E={E} top_k={TOP_K} group={GROUP} "
          f"block_m={BLOCK_M} dtype={dtype}")

    w2, w2_s, w2_z = build_weights(dev, inter)
    rows = []
    for M in [int(x) for x in args.m.split(",")]:
        variants = [build_route(M, dev, dtype, inter, gen) for _ in range(args.variants)]
        nv = len(variants)
        arm = "scatter" if M <= 2 else "gather_reduce"

        if arm == "scatter":
            accs = [torch.zeros((M, HIDDEN), dtype=torch.float32, device=dev) for _ in range(nv)]
            split_k = 4 if M == 1 else 1

            def mk(sk):
                def call(i, _sk=sk):
                    v = variants[i % nv]
                    fp8_wmma.mmq_fp8_moe_gemm_scatter(
                        v["buf2"], w2, w2_s, v["sorted_ids"], v["expert_ids"], v["ntp"],
                        v["tw_flat"], accs[i % nv], TOP_K, BLOCK_M,
                        kernel="wmma", w_zeros=w2_z, weight_is_e2m1=False, split_k=_sk)
                return call

            if args.scatter_sk:
                # The engine's `_moe_split_k` is the frozen constant `M == 1 ? 4 : 1`. Sweep it, at
                # BOTH M it can reach, so the number is measured rather than inherited.
                for skv in [int(x) for x in args.scatter_sk.split(",")]:
                    u = sorted(timeit(mk(skv), args.iters, args.warmup)
                               for _ in range(args.reps))[args.reps // 2]
                    rows.append(dict(M=M, arm="scatter", split_k=skv, us=u, P=variants[0]["P"]))
                    print(f"M={M:<3} scatter split_k={skv:<3} {u:8.2f} us"
                          + ("   <-- engine default" if skv == split_k else ""))
                del variants
                torch.cuda.empty_cache()
                continue
            call = mk(split_k)
        else:
            def call(i):
                v = variants[i % nv]
                fp8_wmma.mmq_fp8_moe_gemm2_gather_reduce(
                    v["buf2"], w2, w2_s, v["sorted_ids"], v["expert_ids"], v["ntp"],
                    v["tw_flat"], TOP_K, BLOCK_M, w_zeros=w2_z, weight_is_e2m1=False)

            if args.fused_sk:
                # Sweep the top_k split so the launcher's derivation is CHECKED against the surface
                # rather than assumed to sit on its optimum. The launcher reads the env per call.
                best = None
                for skv in [int(x) for x in args.fused_sk.split(",")]:
                    os.environ["MINISGL_MOE_G2_SPLIT_K"] = str(skv)
                    u = sorted(timeit(call, args.iters, args.warmup)
                               for _ in range(args.reps))[args.reps // 2]
                    rows.append(dict(M=M, arm=arm, split_k=skv, us=u, blocks=nx_blocks(M) * skv))
                    best = u if best is None or u < best else best
                    print(f"M={M:<3} fused split_k={skv:<3} {u:8.2f} us   "
                          f"{nx_blocks(M) * skv:>5} blocks, {nx_blocks(M) * skv * 8:>5} waves")
                os.environ.pop("MINISGL_MOE_G2_SPLIT_K", None)
                u = sorted(timeit(call, args.iters, args.warmup) for _ in range(args.reps))[args.reps // 2]
                rows.append(dict(M=M, arm=arm, split_k="derived", us=u))
                print(f"M={M:<3} fused split_k=DERIVED {u:8.2f} us   "
                      f"({u / best:.3f}x the best swept point)")
                del variants
                torch.cuda.empty_cache()
                continue

        us = sorted(timeit(call, args.iters, args.warmup) for _ in range(args.reps))[args.reps // 2]
        # HBM floor: the distinct expert down-proj slabs this call must read (int4 + fp16 scales).
        blocks = nx_blocks(M) if arm == "gather_reduce" else None
        nx = (HIDDEN + 255) // 256
        rows.append(dict(M=M, arm=arm, us=us, blocks=blocks, P=variants[0]["P"]))
        print(f"M={M:<3} {arm:<14} {us:8.2f} us"
              + (f"   grid=({nx},{M}) = {blocks} blocks, {blocks * 8} waves" if blocks else
                 f"   split_k={4 if M == 1 else 1}  P={variants[0]['P']}"))
        del variants
        torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(tag=args.tag, shape=dict(N=HIDDEN, K=inter, E=E, top_k=TOP_K,
                                                    group=GROUP, block_m=BLOCK_M),
                           iters=args.iters, variants=args.variants, rows=rows), f, indent=1)
        print(f"[bench] wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
