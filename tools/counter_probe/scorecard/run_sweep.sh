#!/usr/bin/env bash
# run_sweep.sh — HOST side. Take ONE profile_standard window and run every counter phase inside it.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/scorecard/run_sweep.sh [phase ...]
#
# Why one window for the whole sweep: `power_dpm_force_performance_level` is a global, per-card,
# persistent setting, so every set/restore cycle is another chance to leave a card pinned at a
# non-boost clock -- which would silently corrupt every LATER timing job on the box with no error
# anywhere. One window, one restore, verified.
#
# CLOCK DISCIPLINE. Counters are collected under profile_standard (pinned, non-boost). Wall-clock
# TIMES are collected in a SEPARATE auto-perf-level run (`phase0_timing`), OUTSIDE the window.
# Bandwidth claims need bytes/time; bytes are a clock-independent property of the algorithm and come
# from the counters, time comes from the auto run. Mixing a profile_standard time into a bandwidth
# percentage would understate the kernel and is exactly the trap the perf-level caveat warns about.
#
# The script re-enters ITSELF inside the guard (GUARD_OPEN=1) rather than passing a composed command
# string through two levels of quoting -- that indirection was the only fragile part of the design.
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$(dirname "$SELF")"
IMG="${IMG:-rocm/dev-ubuntu-24.04:7.14.0-full}"
KERN="${KERN:-/home/pat/code/rdna4-hip-kernels-cscore}"
OUT="${OUT:-$PROBE/results/scorecard}"
export IMG KERN OUT
mkdir -p "$OUT"

PHASES=("$@")
[ "${#PHASES[@]}" -eq 0 ] && PHASES=(phase1_waitsplit phase2_kernels)

# The lease exports ROCR_VISIBLE_DEVICES=<physical card> and HIP_VISIBLE_DEVICES=0; forward that pair
# VERBATIM. Setting both to the physical index double-filters and breaks whenever the lease assigns
# card 1 (ROCR selects card 1 and re-indexes it to 0, then HIP=1 selects nothing -> "No HIP GPUs").
docker_run() {
  docker run --rm \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
    -e OUT=/out \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    --ipc host --shm-size 16gb \
    -v "$PROBE":/probe:ro -v "$KERN":/kern:ro -v "$OUT":/out \
    --entrypoint bash "$IMG" -lc \
    "set -uo pipefail; export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}; $1"
}

# ============================================================================================
# INSIDE the profile_standard window: run the counter phases and exit.
# ============================================================================================
if [ "${GUARD_OPEN:-0}" = "1" ]; then
  for p in "${PHASES[@]}"; do
    echo ""
    echo "############ $p (profile_standard) ############"
    docker_run "bash /probe/scorecard/$p.sh" 2>&1 | tee "$OUT/$p.log"
  done
  exit 0
fi

# ============================================================================================
# OUTSIDE the window.
# ============================================================================================
echo "=== sweep config ==="
echo "  image   = $IMG"
echo "  kernels = $KERN  ($(cd "$KERN" && git log --oneline -1))"
echo "  probe   = $PROBE"
echo "  out     = $OUT"
echo "  phases  = ${PHASES[*]}"
echo "  lease   : HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"

# phase0 is TIMING ONLY and must run at the card's NORMAL (auto) perf level, so it goes first,
# before the window opens.
COUNTER_PHASES=()
for p in "${PHASES[@]}"; do
  if [ "$p" = "phase0_timing" ]; then
    echo ""
    echo "############ phase0: timings at auto perf level (NO counters, NO pinning) ############"
    docker_run "bash /probe/scorecard/phase0_timing.sh" 2>&1 | tee "$OUT/phase0_timing.log"
  else
    COUNTER_PHASES+=("$p")
  fi
done

if [ "${#COUNTER_PHASES[@]}" -gt 0 ]; then
  echo ""
  echo "############ counter phases, inside ONE profile_standard window ############"
  GUARD_OPEN=1 bash "$PROBE/perf_level_guard.sh" bash "$SELF/run_sweep.sh" "${COUNTER_PHASES[@]}"
fi

echo ""
echo "=== sweep complete; results under $OUT ==="
