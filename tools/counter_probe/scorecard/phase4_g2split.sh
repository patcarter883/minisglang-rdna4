#!/usr/bin/env bash
# phase4_g2split.sh — IN-CONTAINER. Occupancy and workgroup count of the production fused MoE decode
# gemm2, BEFORE and AFTER the top_k split on grid.z, at the SERVED shape.
#
# Shape is the real Qwen3.6-35B-A3B-AWQ TP=2 gemm2 (from config.json), NOT the E=32/group=128 probe
# defaults phase3 used: E=256, top_k=8, K=inter/TP=256, N=hidden=2048, group_size=32, block_m=16.
#
# M is 5 and 6 — the served points that actually REACH this kernel. M=1 and 2 do not: `w4a8_moe`
# takes the M<=2 atomic-scatter branch first (see docs/MOE_G2_SPLIT.md), which is why phase3's
# "M=1, the served decode case" row is a configuration the engine never runs.
#
# Counters only. TIMES come from a separate auto-perf-level run — profile_standard pins clocks and a
# bandwidth computed from a pinned timestamp understates the kernel by ~4x.
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SELF/pmc_lib.sh"
OUT="${OUT:-/out}/phase4"; mkdir -p "$OUT"

echo "=== phase4: build ==="
hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm \
      -o /tmp/g2 /probe/scorecard/moe_g2fuse_probe.hip 2>&1 | grep -iE " error" | head -20
[ -x /tmp/g2 ] || { echo "BUILD FAILED (g2)"; exit 1; }

# E top_k K N group_size block_m
SHAPE="256 8 256 2048 32 16"
echo "=== phase4: smoke (served shape) ==="
for m in 5 6; do
  for sk in 1 2; do
    G2FUSE_SPLIT_K=$sk /tmp/g2 $m $SHAPE 4 || { echo "SMOKE M=$m sk=$sk FAILED"; exit 1; }
  done
done

G_OCC="OccupancyPercent"; G_BUSY="MemUnitBusy"; G_WAVE="SQ_WAVES SQ_BUSY_CYCLES"
G_MEM2="GL2C_HIT GL2C_MISS"; G_ISS="WAVE_ISSUE_WAIT"

for m in 5 6; do
  for sk in 1 2 4; do
    tag="g2split_M${m}_sk${sk}"
    echo ""; echo "--- $tag ---"
    for grp in "$G_OCC" "$G_BUSY" "$G_WAVE" "$G_MEM2" "$G_ISS"; do
      G2FUSE_SPLIT_K=$sk pmc_run "$OUT/$tag" "g$(echo "$grp" | md5sum | cut -c1-4)" "$grp" \
        -- env G2FUSE_SPLIT_K=$sk /tmp/g2 $m $SHAPE 4
    done
  done
done
echo "=== phase4 done ==="
