#!/usr/bin/env bash
# counter_sanity.sh — WHICH of the counters gfx1201 advertises actually return a real number?
#
# THE FAILURE THIS EXISTS TO CATCH: rocprofv3 exits 0, writes a well-formed CSV, and every value is
# 0.000000. That is indistinguishable from a successful collection unless you know the workload MUST
# have produced a non-zero value. This repo has already lost time to it once ("six attempts, ~95
# minutes, rc=0 with every counter 0.0000"). `--list-avail` advertising a counter for gfx1201 is NOT
# evidence the counter reads: the arch list and the per-arch event IDs are separate data, and an event
# ID inherited from gfx9/gfx11 can point at a reserved slot on gfx12 that quietly reads zero.
#
# METHOD: one counter per rocprofv3 run (so nothing can be blamed on exceeding per-block counter slots
# or on multiplexing), against a workload with a KNOWN non-zero answer — the saxpy harness does VALU
# math, vector loads and stores, and zero LDS. Verdicts:
#   OK    non-zero            -> trustworthy
#   ZERO  collected but 0     -> either genuinely zero for this kernel (LDS counters on a no-LDS
#                                kernel) or a dead event ID. The ZERO column is a WARNING, not a
#                                result: check it against a workload that must exercise that unit
#                                before believing any ratio computed from it.
#   FAIL  run errored/hung
#
#   gpu-lease -n 2 -- bash tools/counter_probe/counter_sanity.sh [image]
#
# RESULT, gfx1201 / ROCm 7.14.0 / rocprofiler-sdk 1.3.2, 2026-08-05 (results/counter_sanity.txt):
# Collection RUNS — no hang, no wedge, seconds per counter. But of the 64 counters gfx1201 advertises,
# only these ELEVEN return a real number:
#     SQ_WAVES  SQ_BUSY_CYCLES  SQC_ICACHE_REQ  SQC_ICACHE_MISSES  GRBM_COUNT  GRBM_GUI_ACTIVE
#     L0CacheHit  GPUBusy  GPU_UTIL  Wavefronts        (+ CU_NUM/SIMD_NUM/SE_NUM, which are agent
#                                                        constants, not counters)
# EVERY instruction-mix and memory-traffic counter reads a hard zero: all SQ_INSTS_* (VALU/SALU/SMEM/
# LDS/FLAT/TEX), all SQ_INST_CYCLES_*, SQ_WAIT_*, SQ_WAVE_CYCLES, all TA_*/TCP_*/GL2C_*, and therefore
# every derived metric built on them (VALUBusy, MemUnitBusy, ValuPipeIssueUtil, FetchSize, L2CacheHit,
# OccupancyPercent, LDSBankConflict, WAVE_*_WAIT).
#
# SO: the "counters wedge the card" half of the old rule is STALE, but the conclusion is unchanged —
# the VALU/SALU/VMEM/LDS utilisation breakdown STILL cannot be measured on this hardware, so
# regime calls ("this kernel is VALU-bound") remain inferences. What IS newly measurable per dispatch:
# wave count, SQ busy cycles, GPU active cycles, icache traffic.
# SQ_WAVES is verified exact (saxpy grid 1<<20 / block 256 -> 4096 WGs x 8 wave32 = 32768).
set -uo pipefail
IMG="${1:-rocm/dev-ubuntu-24.04:7.14.0-full}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run --rm \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -v "$SELF":/probe:ro \
  --entrypoint bash "$IMG" -lc '
set -uo pipefail
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
hipcc -O3 --offload-arch=gfx1201 -o /tmp/ch /probe/counter_harness.hip 2>&1 | grep -i error | head
[ -x /tmp/ch ] || { echo BUILD FAILED; exit 1; }

C="SQ_WAVES SQ_BUSY_CYCLES SQ_ACCUM_PREV SQ_WAVE_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_ANY
   SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_SMEM SQ_INSTS_LDS SQ_INSTS_FLAT
   SQ_INSTS_TEX_LOAD SQ_INSTS_TEX_STORE SQ_INSTS_WAVE32 SQ_INSTS_WAVE32_VALU SQ_INSTS_WAVE32_LDS
   SQ_WAVE32_INSTS SQ_WAVE64_INSTS SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM SQ_INST_LEVEL_LDS
   SQ_INSTS_VEC32_LEVEL_LDS SQC_ICACHE_REQ SQC_ICACHE_HITS SQC_ICACHE_MISSES
   SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE
   GRBM_COUNT GRBM_GUI_ACTIVE
   TA_TA_BUSY TA_BUFFER_LOAD_WAVEFRONTS TA_BUFFER_STORE_WAVEFRONTS
   TCP_REQ TCP_REQ_MISS
   GL2C_HIT GL2C_MISS GL2C_EA_RDREQ GL2C_EA_RDREQ_32B GL2C_EA_RDREQ_64B GL2C_EA_RDREQ_128B
   GL2C_EA_WRREQ GL2C_EA_WRREQ_64B GL2C_EA_WRREQ_STALL
   VALUBusy SALUInsts VALUInsts SFetchInsts MemUnitBusy WriteUnitStalled LDSBankConflict LdsUtil
   L0CacheHit L2CacheHit FetchSize GPUBusy GPU_UTIL OccupancyPercent MeanOccupancyPerCU
   MeanOccupancyPerActiveCU ValuPipeIssueUtil WAVE_DEP_WAIT WAVE_ISSUE_WAIT Wavefronts
   CU_NUM SIMD_NUM SE_NUM"

printf "%-28s %-6s %s\n" COUNTER VERDICT VALUE
for c in $C; do
  rm -rf /tmp/s
  # SIGKILL: rocprofiler deadlocks in its own SIGTERM handler when it aborts.
  timeout -s KILL 60 rocprofv3 --pmc $c -f csv -d /tmp/s -- /tmp/ch 3 >/tmp/s.log 2>&1
  rc=$?
  f=$(find /tmp/s -name "*counter*.csv" 2>/dev/null | head -1)
  if [ -z "$f" ]; then
    why=$(grep -oiE "not supported|invalid|mmap failed with errno [0-9]+|Aborted" /tmp/s.log | head -1)
    printf "%-28s %-6s %s\n" "$c" FAIL "rc=$rc ${why:-}"
    continue
  fi
  # Max over the saxpy dispatches only (skip the memset fill dispatches, which are a different kernel).
  # Counter_Value by $(NF-2), NOT by a fixed column index: Kernel_Name is a quoted field that CONTAINS
  # COMMAS ("saxpy(float const*, float*, float, int)"), so -F, shifts every later column and a fixed
  # index reads a timestamp or an empty cell. That misparse prints ZERO for EVERY counter — including
  # ones known to work — i.e. it fakes exactly the "counters are all broken" conclusion this script
  # exists to test. The trailing 4 fields (Counter_Name, Counter_Value, Start, End) are comma-free.
  v=$(awk -F, "/saxpy/{ if (\$(NF-2)+0 > m) m=\$(NF-2)+0 } END{printf \"%.0f\", m+0}" "$f")
  [ "$v" = "0" ] && printf "%-28s %-6s %s\n" "$c" ZERO "$v" || printf "%-28s %-6s %s\n" "$c" OK "$v"
done'
