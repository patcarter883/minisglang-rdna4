#!/usr/bin/env bash
# perf_level_counters.sh — do the all-zero gfx1201 counters come back if the perfmon clock is ungated?
#
# HYPOTHESIS: RDNA4's default `auto` power/perf level gates the perfmon clock in some hardware blocks,
# so counters in those blocks record nothing. The documented fix is STABLE_STD, whose SYSFS spelling is
# `profile_standard`. This fits the observed split exactly: the counters that DO read (SQ_WAVES,
# SQ_BUSY_CYCLES, GRBM_*, SQC_ICACHE_*, GPUBusy) are always-on blocks, and everything that reads a hard
# zero (SQ_INSTS_*, SQ_INST_CYCLES_*, SQ_WAIT_*, TA_*, TCP_*, GL2C_*) sits behind the gated clock.
#
# WHY THIS IS NOT A RE-TEST OF A RULED-OUT CAUSE. The 2026-07-30 note records "AUTO perf level gating"
# as CAUSE RULED OUT #1 — but that test ungated the clock and found `--pmc` STILL DEADLOCKED on ROCm
# 7.2.1. It disproved perf level as the cause of the HANG. The symptom here is different and newer:
# on ROCm 7.14 there is no hang at all, collection succeeds, and the counters return zeros. Perf-level
# gating has never been tested against THAT. Different stack, different failure, open question.
#
# ---------------------------------------------------------------------------------------------------
# HAZARD — READ BEFORE RUNNING. `power_dpm_force_performance_level` is a GLOBAL, PER-CARD, PERSISTENT
# setting. It is not scoped to this process. Two consequences:
#   1. Hold an EXCLUSIVE `gpu-lease -n 2` for the whole set/profile/restore cycle. Changing it while
#      another agent is timing silently corrupts their numbers with no error anywhere.
#   2. It MUST be restored on every exit path. This script traps EXIT/INT/TERM and verifies the
#      read-back, and shouts if the restore failed. A card left on `profile_standard` runs every later
#      timing job at a fixed non-boost clock — silent, plausible, wrong numbers.
#
# INTERPRETATION: `profile_standard` PINS clocks to a fixed non-boost state, so ABSOLUTE TIMINGS taken
# here are NOT comparable to normal auto-mode numbers and must never be mixed into an existing timing
# surface. Counter RATIOS (VALU% vs VMEM% vs LDS%) are the point and are unaffected. It pins clocks
# LOWER, not higher, so there is no thermal/power risk.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/perf_level_counters.sh
set -uo pipefail

# gfx1201 DRM cards, resolved by PCI slot — NOT by index. rocm-smi's GPU index and /sys/class/drm's
# cardN do NOT correspond (here ROCm GPU0 = card1, GPU1 = card2, and the Ryzen iGPU is card3). A
# sloppy index match silently targets the iGPU, which is never a compute target.
mapfile -t CARDS < <(
  for c in /sys/class/drm/card*/device; do
    [ -f "$c/power_dpm_force_performance_level" ] || continue
    slot=$(sed -n 's/^PCI_SLOT_NAME=//p' "$c/uevent" 2>/dev/null)
    case "$slot" in 0000:03:00.0|0000:07:00.0) echo "$c" ;; esac
  done
)
[ "${#CARDS[@]}" -eq 2 ] || { echo "expected 2 gfx1201 cards, found ${#CARDS[@]}"; exit 1; }

declare -A ORIG
for c in "${CARDS[@]}"; do ORIG[$c]=$(cat "$c/power_dpm_force_performance_level"); done

restore() {
  local rc=$? bad=0
  echo "--- restoring perf level ---"
  for c in "${CARDS[@]}"; do
    echo "${ORIG[$c]}" | sudo -n tee "$c/power_dpm_force_performance_level" >/dev/null 2>&1
    local now; now=$(cat "$c/power_dpm_force_performance_level" 2>/dev/null)
    if [ "$now" = "${ORIG[$c]}" ]; then echo "  OK  $(basename "$(dirname "$c")") -> $now"
    else echo "  *** RESTORE FAILED *** $(basename "$(dirname "$c")") is '$now', wanted '${ORIG[$c]}'"; bad=1; fi
  done
  [ $bad -eq 0 ] || echo "!!! A CARD IS LEFT PINNED. Every later timing job on it is wrong. Fix by hand."
  exit $rc
}
trap restore EXIT INT TERM

echo "=== original perf levels ==="
for c in "${CARDS[@]}"; do echo "  $(basename "$(dirname "$c")") = ${ORIG[$c]}"; done

echo "=== setting profile_standard ==="
for c in "${CARDS[@]}"; do
  echo profile_standard | sudo -n tee "$c/power_dpm_force_performance_level" >/dev/null 2>&1
  now=$(cat "$c/power_dpm_force_performance_level")
  echo "  $(basename "$(dirname "$c")") -> $now"
  [ "$now" = "profile_standard" ] || { echo "SET FAILED (sudo? kernel rejected?) — aborting"; exit 3; }
done

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${IMG:-rocm/dev-ubuntu-24.04:7.14.0-full}"
KERN="${KERN:-/home/pat/code/rdna4-hip-kernels}"
OUT="$SELF/results/profile_standard"; mkdir -p "$OUT"

echo "=== counters under profile_standard (ROCm 7.14) ==="
docker run --rm \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -v "$SELF":/probe:ro -v "$KERN":/kern:ro -v "$OUT":/out \
  --entrypoint bash "$IMG" -lc '
set -uo pipefail
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
hipcc -O3 --offload-arch=gfx1201 -o /tmp/ch /probe/counter_harness.hip 2>&1 | grep -i error | head
hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm -o /tmp/gc /probe/gemv_counters.hip 2>&1 | grep -iE "^.*error" | head

echo "--- A) the previously-ZERO counters, one per run, on saxpy ---"
printf "%-24s %-6s %s\n" COUNTER VERDICT VALUE
for c in SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_SMEM SQ_INST_CYCLES_VALU \
         SQ_INST_CYCLES_VMEM SQ_WAIT_ANY SQ_WAVE_CYCLES TA_TA_BUSY TCP_REQ GL2C_HIT GL2C_MISS \
         VALUBusy MemUnitBusy OccupancyPercent FetchSize; do
  rm -rf /tmp/s
  timeout -s KILL 90 rocprofv3 --pmc $c -f csv -d /tmp/s -- /tmp/ch 3 >/tmp/s.log 2>&1; rc=$?
  f=$(find /tmp/s -name "*counter*.csv" 2>/dev/null | head -1)
  [ -z "$f" ] && { printf "%-24s %-6s rc=%s\n" "$c" FAIL "$rc"; continue; }
  # Counter_Value is $(NF-2): Kernel_Name is a QUOTED field CONTAINING COMMAS, so a fixed -F, column
  # index reads a timestamp and prints 0 for everything, faking the very result under test.
  v=$(awk -F, "/saxpy/{ if (\$(NF-2)+0 > m) m=\$(NF-2)+0 } END{printf \"%.0f\", m+0}" "$f")
  [ "$v" = "0" ] && printf "%-24s %-6s %s\n" "$c" ZERO "$v" || printf "%-24s %-6s %s\n" "$c" OK "$v"
done

echo "--- B) int4 vs fp8 decode GEMV, instruction mix ---"
for tag in q1 q2; do :; done
rm -rf /out/q1 /out/q2
timeout -s KILL 180 rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_WAVES GRBM_GUI_ACTIVE \
  -f csv -d /out/q1 -- /tmp/gc 1 4096 4096 128 >/out/q1.log 2>&1
echo "  q1 rc=$?"; f=$(find /out/q1 -name "*counter*.csv" | head -1); [ -n "$f" ] && cp "$f" /out/q1.csv
timeout -s KILL 180 rocprofv3 --pmc SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM SQ_BUSY_CYCLES SQ_WAIT_ANY \
  -f csv -d /out/q2 -- /tmp/gc 1 4096 4096 128 >/out/q2.log 2>&1
echo "  q2 rc=$?"; f=$(find /out/q2 -name "*counter*.csv" | head -1); [ -n "$f" ] && cp "$f" /out/q2.csv
'
echo "=== done (restore runs next, via trap) ==="
