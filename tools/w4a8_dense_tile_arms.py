#!/usr/bin/env python
"""The ARM comparison, with the TILE held equal — the confound the three-way dispatch never removed.

All three w4a8 dense WMMA arms carry their own tile knob, their own tile SET, and their own
independently-frozen 256x128 default:

    prefill_wmma       no env knob. BM=256, plus a hardcoded `M<128 -> BM=64` branch. 2 tiles.
    prefill_wmma_ashuffle  VLLM_W4A8_V10_CFG, default 256x128. 8 tiles, and it is the only arm that
                       shipped BM=16 and BM=32 (it stages B ONLY -- shmem has no BM term -- so a
                       narrow tile costs it nothing).
    wmma_tiled_tuned   VLLM_W4A8_V7_CFG, default 256x128. 12 tiles as shipped, floor BM=64.

So "which arm wins at M=17" was never a clean question: the arms were being compared at DIFFERENT
tiles, and the mid-band is exactly where the tile matters most. This sweep removes the confound two
ways and reports both, because they answer different questions:

  (a) TILE-EQUALIZED -- every arm at the SAME tile, over ashuffle's 8-tile set (which
      wmma_tiled_tuned now instantiates in full). Any surviving gap is ALGORITHM: ashuffle's
      B-only/double-buffered staging + A-shuffle vs the shared A+B LDS core.
  (b) EACH ARM AT ITS OWN BEST TILE -- what a perfectly-tuned dispatch could get from each arm.
      This is the number that decides whether an arm earns its maintenance.

`prefill_wmma` cannot be tile-equalized (no knob), so it appears at its own two tiles, with the
small-M branch ON and OFF (VLLM_W4A8_DENSE_SMALLM_OFF=1). OFF is the control: if its mid-band edge
vanishes when the small-M tile is removed, the edge was the tile, not the arm.

    gpu-lease -n 1 --timeout 5400 -- \
      TILE_TOOL=tools/w4a8_dense_tile_arms.py bash tools/w4a8_dense_tile_surface_run.sh \
        --out /engine/_tile_arms.txt --csv /engine/_tile_arms.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4a8_dense_tile_surface import DEV, rotation, time_graph  # noqa: E402

# ashuffle's instantiated set. wmma_tiled_tuned instantiates all eight (16x128/32x128/256x256/
# 384x128/512x128 were added by this work), so the comparison is exact rather than nearest-neighbour.
COMMON_TILES = ["16x128", "32x128", "64x128", "128x128", "256x128", "256x256", "384x128", "512x128"]

SHAPES = [
    ("g4.q_proj    tp2", 2816, 2048, 32, torch.float16, False),
    ("g4.gate_up   tp1", 2816, 4224, 32, torch.float16, False),
    ("q27.q_proj   tp1", 5120, 6144, 32, torch.bfloat16, True),
    ("lag.gate_up  tp2", 2048, 8192, 32, torch.bfloat16, True),
    ("q35b4.gate_up tp2", 2560, 9216, 32, torch.bfloat16, True),
    ("glm.gate_up  tp2", 2048, 10240, 128, torch.bfloat16, True),
    ("lag.gate_up  tp1", 2048, 16384, 32, torch.bfloat16, True),
    ("q27.gate_up  tp1", 5120, 34816, 32, torch.bfloat16, True),
]
MS = [17, 24, 32, 48, 63, 64, 128, 256, 512, 2048]


def set_env(cand: str) -> None:
    for k in ("VLLM_W4A8_V7_CFG", "VLLM_W4A8_V10_CFG", "VLLM_W4A8_DENSE_SMALLM_OFF"):
        os.environ.pop(k, None)
    if cand.startswith("tiled@"):
        os.environ["VLLM_W4A8_V7_CFG"] = cand.split("@")[1]
    elif cand.startswith("ash@"):
        os.environ["VLLM_W4A8_V10_CFG"] = cand.split("@")[1]
    elif cand == "prefill:smallm_off":
        os.environ["VLLM_W4A8_DENSE_SMALLM_OFF"] = "1"


def arm_of(cand: str) -> str:
    if cand.startswith("tiled@"):
        return "wmma_tiled_tuned"
    if cand.startswith("ash@"):
        return "prefill_wmma_ashuffle"
    return "prefill_wmma"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    fh = open(args.out, "w") if args.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    import fp8_wmma as W

    torch.manual_seed(0)
    cands = (
        ["prefill", "prefill:smallm_off"]
        + [f"tiled@{t}" for t in COMMON_TILES]
        + [f"ash@{t}" for t in COMMON_TILES]
    )
    shapes, ms = SHAPES, MS
    if args.smoke:
        shapes, ms = SHAPES[:1], [17, 256]

    out(f"device: {torch.cuda.get_device_name(0)}   fp8_wmma: {W.__file__}")
    out(f"tile-equalized set: {COMMON_TILES}")
    out("graph-replay timed; rotation sized in BYTES past the 64 MB MALL; us per call\n")

    cf = open(args.csv, "w") if args.csv else None
    if cf:
        cf.write("name,K,N,g,dtype,M,cand,arm,us,reps,R\n")

    for name, K, N, g, dt, zeros in shapes:
        out(f"=== {name}  K={K} N={N} g={g} {str(dt).split('.')[-1]} "
            f"zeros={'awq' if zeros else 'sym'} ===")
        ws = None
        for M in ms:
            del ws
            ws = None
            torch.cuda.empty_cache()
            ws, R, wbytes = rotation(N, K, g, zeros, M)
            x = (torch.randn(M, K, device=DEV) * 0.3).to(dt)
            t, rp = {}, {}
            for c in cands:
                set_env(c)
                arm = arm_of(c)
                try:
                    t[c], rp[c] = time_graph(
                        lambda w, a=arm: W.mmq_fp8_gemm(
                            x, w[0], w[1], kernel=a, w_zeros=w[2], weight_is_e2m1=False
                        ),
                        ws,
                    )
                except Exception as e:  # noqa: BLE001
                    out(f"      (M={M} {c}: {type(e).__name__}: {str(e)[:80]})")
                    torch.cuda.synchronize()
                set_env("prefill")
            if not t:
                continue
            # (a) tile-equalized: at each common tile, tiled vs ash
            eq = []
            for tl in COMMON_TILES:
                a, b = t.get(f"tiled@{tl}"), t.get(f"ash@{tl}")
                if a and b:
                    eq.append(f"{tl}:{'T' if a < b else 'A'}{max(a,b)/min(a,b):.2f}")
            # (b) each arm at its own best
            bests = {}
            for armname, pref in (
                ("tiled", "tiled@"), ("ash", "ash@"), ("prefill", "prefill"),
            ):
                sub = {c: v for c, v in t.items() if c.startswith(pref)}
                if sub:
                    bc = min(sub, key=sub.get)
                    bests[armname] = (bc, sub[bc])
            ordr = sorted(bests, key=lambda k: bests[k][1])
            out(f"    M={M:>5}  own-best: " + "  ".join(
                f"{k}={bests[k][1]:.1f}({bests[k][0].split('@')[-1]})" for k in ordr
            ) + f"   winner={ordr[0]}" + (
                f" by {bests[ordr[1]][1]/bests[ordr[0]][1]:.2f}x" if len(ordr) > 1 else ""
            ))
            out("             tile-equalized (T=tiled A=ash, ratio): " + " ".join(eq))
            if cf:
                for c in sorted(t, key=t.get):
                    cf.write(
                        f"{name},{K},{N},{g},{str(dt).split('.')[-1]},{M},{c},{arm_of(c)},"
                        f"{t[c]:.3f},{rp[c]},{R}\n"
                    )
                cf.flush()
            del x
        del ws
        torch.cuda.empty_cache()
        out("")

    out("done.")
    if fh:
        fh.close()
    if cf:
        cf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
