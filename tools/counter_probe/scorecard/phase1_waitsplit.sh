#!/usr/bin/env bash
# phase1_waitsplit.sh — IN-CONTAINER. Break SQ_WAIT_ANY down by wait type on the two decode GEMVs.
#
# THE QUESTION. A prior measurement established that both decode GEMVs are stall-dominated:
# SQ_WAIT_ANY is ~57x SQ_BUSY_CYCLES (fp8) and ~49x (int4). But SQ_WAIT_ANY is an AGGREGATE -- it
# sums memory waits, barrier waits, dependency waits and arbitration waits into one number. So
# "these kernels are stalled" is established and "stalled ON WHAT" is not, and the difference
# decides a live design question:
#
#   * stalls are MEMORY LATENCY  => more waves in flight is what covers them; a low-occupancy
#                                   deep-buffered variant should LOSE on both loaders.
#   * stalls are DEPENDENCY/ISSUE => ILP is the lever, and trading occupancy for registers/ILP has
#                                   a real case.
#
# ROCm 7.14 on gfx1201 advertises three counters that split this:
#   SQ_WAIT_INST_ANY  — cycles waiting on INSTRUCTION fetch/issue (front-end)
#   WAVE_DEP_WAIT     — cycles a wave is blocked on a DATA DEPENDENCY (incl. outstanding loads)
#   WAVE_ISSUE_WAIT   — cycles a wave is ready but LOSES ARBITRATION for an issue slot
# plus SQ_WAIT_ANY as the aggregate they must be read against.
#
# Each is taken in its OWN pass. They are all SQ-block counters and would contend for physical
# slots; a dropped counter in a grouped pass reads as a zero and would invert the conclusion.
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SELF/pmc_lib.sh"
OUT="${OUT:-/out}/phase1"; mkdir -p "$OUT"

echo "=== phase1: build the decode-GEMV harness ==="
# gemv_counters.hip instantiates gemv_decode_core<> directly with each path's PRODUCTION tiling
# (fp8 nw=8/MMAX=8/COLS=1, int4 nw=16/MMAX=1/COLS=2) at ONE shared shape. Equalising the tilings
# would be a fairer-LOOKING and wronger experiment: COLS=2 is the int4 path's ILP mitigation for
# exactly the issue-bound behaviour under test.
hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm \
      -o /tmp/gc /probe/gemv_counters.hip 2>&1 | grep -iE "error" | head -20
[ -x /tmp/gc ] || { echo "BUILD FAILED"; exit 1; }

ARGS="1 4096 4096 128 3"

echo "=== phase1: FIXTURE GATE — reproduce the recorded SQ_WAIT_ANY/SQ_BUSY_CYCLES ratio ==="
# Recorded fixture (2026-08-05, profile_standard, same shape/tilings):
#   fp8  SQ_WAVES 4096  SQ_BUSY_CYCLES 3,460,996  SQ_WAIT_ANY 199,068,620   (~57x)
#   int4 SQ_WAVES 2048  SQ_BUSY_CYCLES 2,042,555  SQ_WAIT_ANY 100,700,976   (~49x)
# If this pass does not land near those, the window is not configured the way the fixture was and
# NOTHING downstream should be trusted -- so it runs first, on purpose.
pmc_run "$OUT" fixture "SQ_WAVES SQ_BUSY_CYCLES SQ_WAIT_ANY" -- /tmp/gc $ARGS

echo "=== phase1: the wait-type split, one counter per pass ==="
for c in SQ_WAIT_ANY SQ_WAIT_INST_ANY WAVE_DEP_WAIT WAVE_ISSUE_WAIT SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAVES; do
  pmc_run "$OUT" "w_$c" "$c" -- /tmp/gc $ARGS
done

echo "=== phase1: supporting context (occupancy / issue utilisation / memory) ==="
for c in OccupancyPercent MeanOccupancyPerCU ValuPipeIssueUtil VALUBusy MemUnitBusy L0CacheHit; do
  pmc_run "$OUT" "s_$c" "$c" -- /tmp/gc $ARGS
done

echo "=== phase1 done ==="
