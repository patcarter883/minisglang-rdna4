#!/usr/bin/env bash
# radv_control.sh — CONTROL: can a NON-ROCm userspace collect SQ hardware counters on gfx1201?
#
# This does NOT profile our kernels and cannot be made to. RADV is a separate Vulkan userspace over
# the same amdgpu kernel driver and the same silicon; it only ever sees its own submissions. Its whole
# value here is as a control that decides how to read the ROCm result:
#
#   ROCm result (measured, ROCm 7.14): --pmc COLLECTS on gfx1201 (no hang, no wedge), but every
#   SQ_INSTS_*/SQ_INST_CYCLES_*/TA_*/TCP_*/GL2C_* counter returns a hard zero, while SQ_WAVES,
#   SQ_BUSY_CYCLES and GRBM_* return exact values. So the SQ perfmon block is reachable and partly
#   working. The open question is whether the zeros are dead silicon or wrong per-arch event IDs in
#   ROCm userspace (gfx12 renumbered SQ events; the yaml inherits gfx9/gfx11 IDs).
#
#   If RADV's thread trace (SQTT, which reads the SAME SQ block) collects instruction-level data on
#   this card -> the block works, and the ROCm zeros are a ROCm USERSPACE bug: a version to wait out,
#   or fixable by correcting the event IDs. Worth re-testing on ROCm bumps.
#   If RADV also gets nothing -> the limitation is at or below the kernel driver and the rule is
#   permanent. That stops anyone re-testing this every few months.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/radv_control.sh
#
# Needs a WSI: RGP capture is frame-triggered, so a compute-only submission would need
# MESA_VK_TRACE_PER_SUBMIT. vkcube is used because it is installed and it presents frames.
set -uo pipefail
OUT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}"
mkdir -p "$OUT"; cd "$OUT" || exit 1

# Pick the FIRST gfx1201 Vulkan device. Do not hardcode an index: GPU0 here is the Raphael iGPU, so a
# hardcoded 0 silently runs the control on the wrong (and irrelevant) chip.
IDX=$(vulkaninfo --summary 2>/dev/null | awk '/^GPU[0-9]/{n=substr($1,4); sub(":","",n)} /deviceName.*GFX1201/{print n; exit}')
[ -n "$IDX" ] || { echo "no GFX1201 Vulkan device"; exit 1; }
echo "gfx1201 = Vulkan GPU$IDX"

rm -f ./*.rgp
export MESA_VK_TRACE=rgp
export MESA_VK_TRACE_FRAME=${MESA_VK_TRACE_FRAME:-5}
export RADV_THREAD_TRACE_CACHE_COUNTERS=${RADV_THREAD_TRACE_CACHE_COUNTERS:-1}
export RADV_THREAD_TRACE_INSTRUCTION_TIMING=${RADV_THREAD_TRACE_INSTRUCTION_TIMING:-1}
export RADV_THREAD_TRACE_BUFFER_SIZE=${RADV_THREAD_TRACE_BUFFER_SIZE:-33554432}

# SIGKILL for the same reason as the ROCm probes: a wedged capture must not outlive the lease.
timeout -s KILL 120 vkcube --gpu_number "$IDX" --c 40 --width 256 --height 256 >vkcube.log 2>&1
echo "vkcube rc=$?"
grep -iE 'thread trace|sqtt|rgp|error|not supported|fail' vkcube.log | head -10

# RADV writes the capture to the path it PRINTS (/tmp by default), not to cwd — so read the path out
# of the log rather than globbing here, or a successful capture is reported as a failure.
cap=$(grep -oE "/[^']*\.rgp" vkcube.log | tail -1)
if [ -n "$cap" ] && [ -s "$cap" ]; then
  mv -f "$cap" "$OUT/" 2>/dev/null && cap="$OUT/$(basename "$cap")"
  # "B00P" is the RGP magic; without it the file exists but holds no trace.
  magic=$(head -c 4 "$cap")
  echo "RGP CAPTURE OK: $cap ($(stat -c %s "$cap") bytes, magic=$magic)"
  echo "  => the SQ block DOES collect for a non-ROCm userspace on gfx1201:"
  echo "     silicon + amdgpu driver are fine, so ROCm's all-zero SQ_INSTS_* are a ROCm USERSPACE bug."
else
  echo "NO RGP CAPTURE — see vkcube.log"
fi
