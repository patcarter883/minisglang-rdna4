#!/usr/bin/env bash
# probe_counters.sh — does rocprofv3 HARDWARE COUNTER collection work on gfx1201?
#
# WHY THIS EXISTS
# ---------------
# Every "this kernel is bound by X" conclusion in this repo is an INFERENCE (static register analysis,
# ISA instruction counting, differential experiments) because `rocprofv3 --pmc` was measured to hang on
# gfx1201. That rule is version-stamped, not eternal: rocprofiler-sdk is the thing that breaks, and this
# box now carries FOUR different ROCm stacks. Re-run this after any ROCm bump before assuming counters
# are still broken, and before building yet another inference-based harness.
#
# WHAT IT DOES — an escalation ladder that stops being interesting at the first failure:
#   A  bare run, no profiler          proves the harness and the card are fine
#   B  --kernel-trace                 proves the profiler ATTACHES at all (this is the known-good path)
#   C  --pmc SQ_WAVES                 ONE counter, tiny kernel — the actual question
#   D  --pmc <VALU/VMEM/LDS set>      the utilisation breakdown we currently cannot measure
#   E  --list-avail                   what the driver claims it supports for this arch
#
# RUN IT (never bare — the card is shared; and ALWAYS with a hard SIGKILL timeout, see below):
#   gpu-lease -n 2 -- bash tools/counter_probe/probe_counters.sh <docker-image>
#   ROCM_HOST=1 gpu-lease -n 2 -- bash tools/counter_probe/probe_counters.sh    # host /opt/rocm instead
#
# MEASURED 2026-08-05, gfx1201 (RX 9070 XT), same host kernel for all three (containers share it, so
# the differences below are ROCm USERSPACE, not driver):
#
#   stack                                sdk     --kernel-trace   --pmc
#   ---------------------------------------------------------------------------------------------
#   ROCm 7.2.1  minisgl-rdna4:lean       1.1.0   OK (0s)          HANGS >90s, needs SIGKILL
#   ROCm 7.2.4  host /opt/rocm           1.1.0   ABORTS instantly  (not reached)
#   ROCm 7.14.0 dev-ubuntu-24.04:7.14.0  1.3.2   OK (1s)          COLLECTS in seconds
#
# The 7.2.4 host abort is `ring_buffer.cpp:106 mmap failed with errno 22` -> SIGABRT -> the process
# then DEADLOCKS inside rocprofv3's own signal handler (futex_wait) and ignores SIGTERM; one sat 258s
# under a `timeout 90`. That is the same errno-22 ring_buffer signature recorded in June 2026 for
# --kernel-trace, so the host 7.2.4 packaging is a REGRESSION relative to the 7.2.1 container.
#
# => PROFILE ON 7.14, SERVE ON 7.2.1. The kernel source is shared, so a counter measurement taken in
#    7.14 transfers; a .so does not (build kernel packages in the image that will run them).
#    AND PIN THE PERF LEVEL. At the default `auto`, 7.14 collection SUCCEEDS (rc=0, well-formed CSV)
#    while every SQ_INSTS_*/SQ_INST_CYCLES_*/SQ_WAIT_*/TA_*/TCP_*/GL2C_* reads a hard ZERO, because
#    gfx1201's `auto` power state gates the perfmon clock in those blocks. Under `profile_standard`
#    they all return real values. That silent-zero mode is far more dangerous than the hang: the hang
#    stops you, the zeros hand you a confident wrong answer. See perf_level_counters.sh.
#
# TWO TRAPS THIS SCRIPT ENCODES — both cost a lease to learn:
#   1. `timeout N` IS NOT ENOUGH. When rocprofiler-sdk aborts, it deadlocks inside its own signal
#      handler (futex_wait) and IGNORES SIGTERM; the process sat 258s under a `timeout 90`. Every leg
#      here uses `timeout -s KILL`, which is the only thing that reclaims the lease.
#   2. A HANG HERE IS USUALLY NOT A GPU WEDGE. Check rocm-smi before you blame the card: a real wedge
#      pins ~100% "use"; this failure leaves the GPU at ~3% use / 0% memory activity because the
#      deadlock is host-side, in the profiler. Do not report a profiler bug as a wedged card.
set -uo pipefail

IMG="${1:-}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIM="${LIM:-90}"

# Legs run inside the target stack. Kept as one string so the host and container paths execute the
# IDENTICAL ladder — otherwise a difference in the ladder gets misread as a difference in ROCm.
read -r -d '' LADDER <<'INNER'
set -uo pipefail
# No ld.so.conf entry for ROCm in these images: a hipcc-built binary run DIRECTLY exits 127
# ("libamdhip64.so.7: cannot open shared object file") while the same binary under rocprofv3 runs fine
# (the wrapper sets this itself). Without this, leg A is the ONLY failing leg — which reads exactly
# backwards, as "the unprofiled baseline is broken but the profiler works".
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
# rocprofv3 in the 7.2.1 serve image needs libdw/libelf, which that image keeps in /opt/rocprof-deps
# (and torch/lib) rather than on the default loader path. Without them EVERY profiled leg exits 127
# and looks like "the profiler is missing", when it is only a loader path.
for d in /opt/rocprof-deps /opt/venv/lib/python3.12/site-packages/torch/lib; do
  [ -f "$d/libdw.so.1" ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$d"
done
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-1}
OUT=/tmp/cp; rm -rf $OUT; mkdir -p $OUT
echo "### stack: rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo '?') rocprofv3=$(rocprofv3 --version 2>&1 | awk '/version:/{print $2; exit}')"

hipcc -O3 --offload-arch=gfx1201 -o /tmp/ch /probe/counter_harness.hip 2>&1 | tail -5
[ -x /tmp/ch ] || { echo "BUILD FAILED"; exit 1; }
echo "### harness: $(stat -c %s /tmp/ch) bytes"

# Hard SIGKILL: rocprofiler deadlocks in its own SIGTERM handler (see header trap #2).
t() { local lbl="$1"; shift
      local s=$SECONDS
      timeout -s KILL "$LIM_I" "$@" >$OUT/log 2>&1; local rc=$?
      local e=$(( SECONDS - s ))
      if [ $rc -eq 137 ]; then printf '%-40s >%3ds  HUNG (SIGKILLed)\n' "$lbl" "$LIM_I"
      else printf '%-40s %4ds  rc=%d\n' "$lbl" "$e" "$rc"; fi
      # The abort signature matters more than the exit code: errno 22 is the ring_buffer mmap bug.
      grep -oE 'mmap failed with errno [0-9]+|_Map_base::at|HSA_STATUS[A-Z_]*|Permission denied' $OUT/log | sort -u | sed 's/^/      ! /'
      return $rc; }

echo "=== A: bare (no profiler) ==="
t "bare harness"                    /tmp/ch 3
echo "=== B: TRACE (attach, no counters) ==="
t "--kernel-trace"                  rocprofv3 --kernel-trace -f csv -d $OUT/tr -- /tmp/ch 3
echo "=== C: ONE COUNTER ==="
t "--pmc SQ_WAVES"                  rocprofv3 --pmc SQ_WAVES -f csv -d $OUT/c1 -- /tmp/ch 3
f=$(find $OUT/c1 -name '*counter*.csv' 2>/dev/null | head -1)
[ -n "$f" ] && { echo "   rows=$(( $(wc -l < "$f") - 1 ))"; head -3 "$f" | sed 's/^/   /'; } || echo "   NO counter csv"
echo "=== D: UTILISATION SET ==="
t "--pmc VALU/VMEM/LDS"             rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_WAVES GRBM_GUI_ACTIVE -f csv -d $OUT/c2 -- /tmp/ch 3
f=$(find $OUT/c2 -name '*counter*.csv' 2>/dev/null | head -1)
[ -n "$f" ] && { echo "   rows=$(( $(wc -l < "$f") - 1 ))"; head -3 "$f" | sed 's/^/   /'; } || echo "   NO counter csv"
echo "=== E: what does the driver advertise? ==="
timeout -s KILL 60 rocprofv3 --list-avail 2>&1 | grep -iE 'gfx|SQ_WAVES|SQ_INSTS_VALU|VALUBusy' | head -12
INNER

if [ "${ROCM_HOST:-0}" = "1" ]; then
  echo "==== TARGET: HOST /opt/rocm ===="
  LIM_I="$LIM" bash -c "$(printf 'LIM_I=%s\n%s' "$LIM" "${LADDER//\/probe\//$SELF_DIR/}")"
  exit $?
fi

[ -n "$IMG" ] || { echo "usage: $0 <docker-image>   (or ROCM_HOST=1 $0)"; exit 2; }
echo "==== TARGET: $IMG ===="
# Devices come from the lease — forward the arbiter's already-composed pair verbatim (CLAUDE.md).
docker run --rm \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -e LIM_I="$LIM" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -v "$SELF_DIR":/probe:ro \
  --entrypoint bash "$IMG" -lc "$LADDER"
