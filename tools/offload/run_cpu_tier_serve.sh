#!/usr/bin/env bash
# Boot the qwen4_exp ALL-HOST serve with N MoE layers on the CPU expert tier, prove it is really
# serving, measure it, and tear it down. ONE leg. Run it twice — CPU_LAYERS=0 and CPU_LAYERS=12 —
# for the A/B; the legs differ only in that variable.
#
#   CPU_LAYERS=0  REPO=$PWD tools/offload/run_cpu_tier_serve.sh
#   CPU_LAYERS=12 REPO=$PWD tools/offload/run_cpu_tier_serve.sh
#
# THE CONFIGURATION IS FIXED AND IS NOT A TUNING SURFACE. All MoE from system RAM: WOFF_DEVICE_GB is
# deliberately UNSET, so serve.sh's all-host default (device_gb 0.5 = zero layers) stands. Adding a
# device tier makes the numbers non-comparable with the external target this work is measured
# against, which is why it is not a parameter of this script.
#
# TRAPS THIS SCRIPT EXISTS TO NOT RE-PAY:
#   * `gpu-lease` OVERRIDES COMPOSE_PROJECT_NAME and LEASE_NAME. Nothing here derives the container
#     name; `_cpu_tier_leg.sh` asks `docker compose ps -q serve` inside the lease shell. Deriving it
#     wrong reports CRASHED on a healthy serve and orphans a container holding both cards.
#   * `/health` LIES — see `_serve_probe.py`. Readiness is a real generation.
#   * The host arena needs ~64 GiB and its capacity gate runs BEFORE pinning, so it is blind to the
#     ZFS ARC and to tmpfs (a 46 GiB /tmp holding 14 GiB is 14 GiB of RAM). This refuses up front
#     rather than booting into a 20-minute load that dies at the pin.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CPU_LAYERS="${CPU_LAYERS:-12}"
LABEL="${LABEL:-cpu${CPU_LAYERS}}"
OUT="${OUT:-$REPO/tools/offload/e4m3_gate}"
READY_TIMEOUT="${READY_TIMEOUT:-2400}"
# The pre-flight floor is about the PINNED arena, which at high CPU_LAYERS is empty — so it is a
# caller knob, not a constant. Pageable CPU-tier weights are ordinary reclaimable memory.
MIN_AVAIL_GIB="${MIN_AVAIL_GIB:-68}"

export REPO LABEL READY_TIMEOUT
export PORT="${PORT:-1919}"
export REPS="${REPS:-3}"
export DECODE_TOKENS="${DECODE_TOKENS:-128}"
export DECODE_M="${DECODE_M:-1,2}"

mkdir -p "$OUT"
export LOG="$OUT/$LABEL.serve.log"
rm -f "$OUT/$LABEL.serve.log.probe"
export JSON="$OUT/$LABEL.bench.json"

avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
if (( avail < MIN_AVAIL_GIB )); then
  echo "[cpu-tier] REFUSING: MemAvailable ${avail} GiB < ${MIN_AVAIL_GIB} GiB. The host-arena" \
       "capacity gate runs before pinning and is blind to the ZFS ARC and to tmpfs, so booting now" \
       "means dying 20 minutes in. Try: echo 3 | sudo tee /proc/sys/vm/drop_caches" >&2
  exit 2
fi
echo "[cpu-tier] MemAvailable=${avail}GiB repo=$REPO cpu_layers=$CPU_LAYERS label=$LABEL"

# `--weight-offload-cpu-layers 0` and omitting the flag are not the same code path, so the 0-layer
# leg omits it entirely. That leg IS the shipped configuration, which is what makes it the comparand.
extra=""
[[ "$CPU_LAYERS" != "0" ]] && extra="--weight-offload-cpu-layers $CPU_LAYERS"

# TP and MEM_RATIO are caller-overridable (they were hardcoded TP=2/0.75). TP=1 is a REAL
# configuration for this arm, not a degenerate one: with all 48 MoE layers on the CPU tier the
# pinned arena is empty, so the min_tp=2 rule -- which is stated entirely in terms of pinning a
# ~54 GiB host arena in one process -- does not bind. The defaults are unchanged, so every leg
# measured before this edit is reproduced by running it with no TP/MEM_RATIO set.
export MODEL=qwen4exp SPEC=none TP="${TP:-2}" CONC=2 MEM_RATIO="${MEM_RATIO:-0.75}" GRAPH_BS=0
export MINISGL_ALLOW_TP_BELOW_MIN="${MINISGL_ALLOW_TP_BELOW_MIN:-}"
# ARENA KNOBS ARE NOT SET HERE, and that is deliberate at high CPU_LAYERS. A CPU-tier layer is read
# by CPU cores with ordinary loads, so it needs neither VRAM nor PINNED host memory: at
# CPU_LAYERS=48 the pinned host tier should be EMPTY and `WOFF_HOST_GB` /
# `MINISGL_WEIGHT_ARENA_FLOOR_GIB` should stop mattering entirely. Forcing a floor here would tune a
# tier that does not exist and would keep the host-capacity gate (arena + floor <= MemAvailable) on
# the path, which has refused boots on this box for reasons that were about the ZFS ARC rather than
# about the configuration. Export them from the caller ONLY for a mixed config that really has a
# pinned tier.
export EXTRA_ARGS="$extra"
export MINISGL_HOST_PORT="$PORT"
# The loop-stage host partition (recv/sched/fwd_launch/gpu_wait/commit), every N decode steps. This
# is the instrument the comparand is quoted from. STEP_LOG and the decode panel measure the host
# wall of `_forward`, which under capture returns after enqueueing an async replay and reads ~0.5 ms.
export MINISGL_HOSTPROF=50
# The CPU tier's counters, both sides of the seam. Without them "the tier ran" is unfalsifiable:
# `engaged()` is a SET and saturates at one, so it cannot tell a tier that bound 12 layers and
# executed 3 from one that executed all 12.
export MINISGL_CPU_MOE_STATS=200
# THE CORE SWEEP'S ARM VARIABLES. Exported (not just inherited) so `docker compose` interpolates
# them, and echoed below so the leg's own log records which arm it was -- an A/B whose arm is not in
# its artifact is a measurement of nothing. Empty = the engine's shipped default (2 threads/rank,
# derived core list, derived node-wide cap of 5).
export MINISGL_CPU_MOE_THREADS="${MINISGL_CPU_MOE_THREADS:-}"
export MINISGL_CPU_MOE_CORES="${MINISGL_CPU_MOE_CORES:-}"
export MINISGL_CPU_MOE_CORE_BUDGET="${MINISGL_CPU_MOE_CORE_BUDGET:-}"
# AFTER the exports, not before: this line names every one of them and the script runs under
# `set -u`, so printing the arm before it is defined aborts the leg with "unbound variable" before
# a single card is leased.
echo "[cpu-tier] ARM: threads/rank='${MINISGL_CPU_MOE_THREADS:-<default 2>}'" \
     "cores='${MINISGL_CPU_MOE_CORES:-<derived>}'" \
     "core_budget='${MINISGL_CPU_MOE_CORE_BUDGET:-<derived 5>}'" \
     "stats_every=${MINISGL_CPU_MOE_STATS}"

cd "$REPO" || exit 1
# `-n "$TP"`, not a hardcoded 2: `-n` is HOW MANY cards, and leasing both for a single-card TP=1
# run starves every other agent on this box for the whole leg (CLAUDE.md, the booking rules).
timeout "$((READY_TIMEOUT + 2400))" \
  gpu-lease -n "$TP" -- bash "$REPO/tools/offload/_cpu_tier_leg.sh" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[cpu-tier] leg $LABEL exit=$rc  (76 = the GPU wedged, not a bug in the command — re-run once)"
echo "[cpu-tier] log=$LOG container-log=$LOG.container json=$JSON"
exit "$rc"
