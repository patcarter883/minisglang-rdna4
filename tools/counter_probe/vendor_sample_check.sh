#!/usr/bin/env bash
# vendor_sample_check.sh — check the all-zero SQ_INSTS_* result against AMD'S OWN tooling and sample.
#
# The negative this guards: counter_sanity.sh found that on gfx1201 / ROCm 7.14, SQ_WAVES and
# GRBM_* read exactly right while every SQ_INSTS_*/SQ_INST_CYCLES_*/TA_*/TCP_*/GL2C_* reads a hard
# zero. Before that is treated as settled, it has to survive two obvious objections:
#   (a) "your harness is too trivial / the counters need a richer workload" -> use AMD's bundled
#       instmix sample, which exists specifically to generate a known instruction mix.
#   (b) "you used the wrong frontend; rocprof-compute knows the right per-arch event IDs" ->
#       run rocprof-compute (rocprofiler-compute 3.7.0, bundled in the 7.14 images) on the same
#       binary. If ITS numbers are non-zero, our counter list was wrong, not the stack.
# A vendor sample failing under the vendor frontend is as clean as this gets.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/vendor_sample_check.sh
set -uo pipefail
IMG="${IMG:-rocm/dev-ubuntu-24.04:7.14.0-full}"
OUT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}"
mkdir -p "$OUT"

docker run --rm \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb -v "$OUT":/out \
  --entrypoint bash "$IMG" -lc '
set -uo pipefail
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
S=/opt/rocm/share/rocprofiler-compute/sample
hipcc -O3 --offload-arch=gfx1201 -o /tmp/instmix $S/instmix.hip 2>&1 | grep -i error | head -5
[ -x /tmp/instmix ] || { echo "BUILD FAILED"; exit 1; }
echo "### vendor sample instmix built"

echo "=== 1) rocprofv3 --pmc on the VENDOR sample (one counter per run) ==="
printf "%-24s %-6s %s\n" COUNTER VERDICT VALUE
for c in SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_SMEM SQ_INST_CYCLES_VALU GRBM_GUI_ACTIVE; do
  rm -rf /tmp/v
  timeout -s KILL 120 rocprofv3 --pmc $c -f csv -d /tmp/v -- /tmp/instmix >/tmp/v.log 2>&1; rc=$?
  f=$(find /tmp/v -name "*counter*.csv" 2>/dev/null | head -1)
  if [ -z "$f" ]; then printf "%-24s %-6s rc=%s\n" "$c" FAIL "$rc"; continue; fi
  # Counter_Value is $(NF-2): Kernel_Name is a quoted field CONTAINING COMMAS, so a fixed column
  # index silently reads a timestamp and prints 0 for everything.
  v=$(awk -F, "NR>1{ if (\$(NF-2)+0 > m) m=\$(NF-2)+0 } END{printf \"%.0f\", m+0}" "$f")
  [ "$v" = "0" ] && printf "%-24s %-6s %s\n" "$c" ZERO "$v" || printf "%-24s %-6s %s\n" "$c" OK "$v"
done

echo "=== 2) rocprof-compute: is gfx1201 even a supported arch? ==="
# Ask BEFORE profiling. rocprof-compute is the frontend anyone reaching for a roofline will try next,
# and the arch list is the whole answer: no gfx12 entry means it cannot analyse this card at all,
# independently of whether the underlying counters read. Measured 2026-08-05 on rocprof-compute 3.7.0
# (bundled with ROCm 7.14): supported = gfx908 gfx90a gfx940 gfx941 gfx942 gfx950 gfx1150 gfx1151
# gfx1152. NO gfx1200/gfx1201 -> rocprof-compute is NOT an option on RDNA4. Do not spend a window
# on it. (gfx115x is RDNA3.5, not RDNA4 -- easy to misread as "close enough".)
# --list-metrics REQUIRES an arch argument, so calling it bare errors out and greps clean — which
# would report "no gfx12" from a usage error rather than from the arch list. Read the supported-arch
# list out of --help, where it is printed verbatim under --list-metrics.
rocprof-compute --help 2>&1 | sed -n "/--list-metrics/,/--list-blocks/p" | head -15
echo "--- gfx12 present in that list? ---"
if rocprof-compute --help 2>&1 | sed -n "/--list-metrics/,/--list-blocks/p" | grep -q "gfx12"; then
  echo "  YES -> retry a real profile run"
  cd /tmp && timeout -s KILL 600 rocprof-compute profile -n vend -- /tmp/instmix >/out/rocprof_compute.log 2>&1
  echo "  rocprof-compute profile rc=$?"; tail -25 /out/rocprof_compute.log
else
  echo "  NO -> gfx1201 unsupported by rocprof-compute; nothing to run."
fi
'
