#!/usr/bin/env bash
# gemv_counters.sh — collect the hardware-counter utilisation breakdown for the int4 vs fp8 decode GEMV.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/gemv_counters.sh [outdir] [M N K GS]
#
# Runs in rocm/dev-ubuntu-24.04:7.14.0-full because that is the ONLY stack on this box where --pmc
# collects on gfx1201 (7.2.1 and host 7.2.4 both abort in rocprofiler's ring_buffer — see
# probe_counters.sh). We serve on 7.2.1; we PROFILE on 7.14. The kernel source is identical, so the
# measurement transfers; the .so would not, which is why gemv_counters.hip is torch-free.
#
# MULTI-PASS ON PURPOSE. Counters live in a fixed number of hardware slots per block, so a big
# --pmc list either fails to schedule or gets silently multiplexed across dispatches. Each pass here
# is a small, self-consistent group collected in ONE run, so every ratio is computed from counters
# that were live SIMULTANEOUSLY. Never merge these into one --pmc line to "save a run".
set -uo pipefail

OUT="${1:-$PWD/tools/counter_probe/results}"; shift || true
SHAPE=("${@:-}"); [ -z "${SHAPE[0]:-}" ] && SHAPE=(1 4096 4096 128)
IMG="${IMG:-rocm/dev-ubuntu-24.04:7.14.0-full}"
KERN="${KERN:-/home/pat/code/rdna4-hip-kernels}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"

docker run --rm \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -v "$SELF":/probe:ro -v "$KERN":/kern:ro -v "$OUT":/out \
  --entrypoint bash "$IMG" -lc '
set -uo pipefail
# The image ships no ld.so.conf entry for ROCm, so a hipcc-built binary run DIRECTLY fails with
# "libamdhip64.so.7: cannot open shared object file" (exit 127) while the same binary under rocprofv3
# works — the rocprofv3 wrapper sets this itself. Without this line the unprofiled baseline leg is the
# only one that fails, which reads exactly like "the profiler is the only thing that works".
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm \
      -o /tmp/gc /probe/gemv_counters.hip 2>&1 | grep -E "error|Error" | head -20
[ -x /tmp/gc ] || { echo "BUILD FAILED"; exit 1; }
/tmp/gc '"${SHAPE[*]}"' || { echo "HARNESS FAILED"; exit 1; }

run() { local tag="$1"; shift
  # SIGKILL, not SIGTERM: rocprofiler deadlocks in its own SIGTERM handler when it aborts.
  timeout -s KILL 180 rocprofv3 --pmc "$@" -f csv -d /out/$tag -- /tmp/gc '"${SHAPE[*]}"' >/out/$tag.log 2>&1
  local rc=$?
  local f=$(find /out/$tag -name "*counter*.csv" 2>/dev/null | head -1)
  if [ -n "$f" ]; then cp "$f" /out/$tag.csv; echo "  $tag rc=$rc rows=$(( $(wc -l < "$f") - 1 ))";
  else echo "  $tag rc=$rc NO CSV"; grep -oE "mmap failed with errno [0-9]+|not supported|Invalid" /out/$tag.log | sort -u | sed "s/^/    ! /"; fi; }

echo "=== pass 1: instruction mix ==="
run p1 SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_SMEM SQ_WAVES GRBM_GUI_ACTIVE
echo "=== pass 2: issue/stall cycles ==="
run p2 SQ_INST_CYCLES_VALU SQ_INST_CYCLES_VMEM SQ_BUSY_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAVE_CYCLES
echo "=== pass 3: derived utilisation ==="
run p3 VALUBusy MemUnitBusy ValuPipeIssueUtil MeanOccupancyPerCU
echo "=== pass 4: memory traffic ==="
run p4 FetchSize L2CacheHit TCP_REQ TCP_REQ_MISS
'
echo "results in $OUT"
