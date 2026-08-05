#!/usr/bin/env bash
# cache_gap.sh — is the isolated-vs-in-serve roofline gap CACHE STATE?
#
# THE EXPERIMENT. Same kernel, same shape, three cache states (hot / MALL-rotated / flush-interleaved),
# one statistic. `evict` sweeps the interleaved byte volume; the sweep answers "how many bytes of
# traffic between two launches are needed to move the isolated number onto the in-serve number", and
# that answer is then compared with the elementwise/norm byte volume MEASURED from the serve trace.
#
# TWO LEGS, ON TWO DIFFERENT TOOLCHAINS, AND THE SPLIT IS LOAD-BEARING:
#
#   TIMING leg  — ROCm 7.2.1, the SERVE image, perf level AUTO.
#       The comparison being made is against in-serve timings, so the isolated binary must be built
#       by the SAME compiler that built the kernels the serve ran. This is not pedantry: this repo
#       has already measured ROCm 7.14 / clang-23 changing these kernels' scratch usage and buying
#       1.35-1.83x on some of them. Timing a 7.14-compiled kernel and calling the difference "the
#       serve's fault" would be a compiler A/B wearing a cache-state costume.
#
#   COUNTER leg — ROCm 7.14, profile_standard.
#       gfx1201 counters do not work anywhere else: 7.2.1 HANGS on --pmc, the host 7.2.4 aborts, and
#       RDNA4's default `auto` perf level GATES the perfmon clock so counters read a hard zero with
#       rc=0 and a well-formed CSV. profile_standard also PINS clocks non-boost, so no timestamp from
#       this leg may be compared with the timing leg. Only the RATIOS are read here — GL2C_MISS and
#       L2 hit rate — which is exactly what confirms or refutes the mechanism.
#
#   gpu-lease -n 2 -- bash tools/counter_probe/ksweep/cache_gap.sh
set -uo pipefail
WT=${WT:-/home/pat/code/minisgl-rdna4-ksweep}
KERN=${KERN:-/home/pat/code/rdna4-hip-kernels-ksweep}
RES=${RES:-$WT/tools/counter_probe/results/ksweep}
mkdir -p "$RES"
SERVE_IMG=${SERVE_IMG:-minisgl-rdna4:ksweep-prof}
CNT_IMG=${CNT_IMG:-rocm/dev-ubuntu-24.04:7.14.0-full}
INC=/kern/fp8_wmma/fp8_wmma_rocm

DOCKER_COMMON=(--rm
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
  -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
  --device /dev/kfd --device /dev/dri --group-add video
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE
  --ipc host --shm-size 16gb
  -v "$WT":/engine -v "$KERN":/kern:ro)

# The shapes are the ones the SERVE actually dispatches, not a sizing sweep. `dense GEMV at 16384^2`
# is where the "81% of HBM" came from and the engine never launches it; it is kept here only as the
# control that reproduces the historical number.
#   loader M N K label
SHAPES=${SHAPES:-"
bf16 1 32768 2048 lm_head
bf16 1 6144  2048 in_proj_qkvz
bf16 1 2048  256  shared_down
int4 1 2048  2048 moe_expert_like
int4 1 6144  2048 qkv_int4
fp8  1 4096  4096 fp8_4096
int4 1 16384 16384 synthetic_control
"}
FLUSHES=${FLUSHES:-"0 8 16 32 64 128 256"}

# ==================================================================================================
# TIMING leg — serve toolchain (ROCm 7.2.1), AUTO clocks.
# ==================================================================================================
OUT=$RES/cache_gap_timing.csv
{
  echo "# cache_gap timing leg"
  echo "# date=$(date -Is) toolchain=SERVE_IMAGE($SERVE_IMG, ROCm 7.2.1) perf_level=auto"
  echo "# kernels=8a8bca6 engine=$(cd "$WT" && git rev-parse --short HEAD) roofline_GBs=706.6"
  echo "loader,M,N,K,cond,flush_mb,rot_copies,total_ns,min_ns,flush_ns,gemv_ns,bytes,GB_s,pct_roofline,label"
} > "$OUT"

echo "=== compiling probe with the SERVE toolchain ==="
docker run "${DOCKER_COMMON[@]}" --entrypoint bash "$SERVE_IMG" -lc "
  hipcc -O3 --offload-arch=gfx1201 -I$INC -o /engine/tools/counter_probe/ksweep/_cgp_serve \
    /engine/tools/counter_probe/ksweep/cache_gap_probe.hip 2>&1 | grep -iE 'error' | head -20
  ls -la /engine/tools/counter_probe/ksweep/_cgp_serve" 2>&1 | tail -5

echo "=== timing sweep (auto clocks, serve toolchain) ==="
while read -r ldr M N K label; do
  [ -z "${ldr:-}" ] && continue
  for cond in hot rot; do
    line=$(docker run "${DOCKER_COMMON[@]}" --entrypoint bash "$SERVE_IMG" -lc \
      "/engine/tools/counter_probe/ksweep/_cgp_serve $ldr $M $N $K $cond 0" 2>>"$RES/cache_gap.err")
    echo "$line" | grep "^CSV," | sed "s/^CSV,//;s/$/,$label/" >> "$OUT"
    echo "  $label $cond: $(echo "$line" | grep '^CSV,' | awk -F, '{print $13" GB/s = "$14"%"}')"
  done
  for f in $FLUSHES; do
    [ "$f" = "0" ] && continue
    line=$(docker run "${DOCKER_COMMON[@]}" --entrypoint bash "$SERVE_IMG" -lc \
      "/engine/tools/counter_probe/ksweep/_cgp_serve $ldr $M $N $K evict $f" 2>>"$RES/cache_gap.err")
    echo "$line" | grep "^CSV," | sed "s/^CSV,//;s/$/,$label/" >> "$OUT"
    echo "  $label evict/${f}MB: $(echo "$line" | grep '^CSV,' | awk -F, '{print $13" GB/s = "$14"%"}')"
  done
done <<< "$SHAPES"
echo "wrote $OUT"

# ==================================================================================================
# COUNTER leg — ROCm 7.14 at profile_standard. RATIOS ONLY; every timestamp here is clock-pinned.
# ==================================================================================================
[ "${SKIP_COUNTERS:-0}" = "1" ] && { echo "counter leg skipped"; exit 0; }

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
    else echo "  *** RESTORE FAILED *** $(basename "$(dirname "$c")") is '$now'"; bad=1; fi
  done
  [ $bad -eq 0 ] || echo "!!! A CARD IS LEFT PINNED. Every later timing job on it is wrong."
  exit $rc
}
trap restore EXIT INT TERM
for c in "${CARDS[@]}"; do
  echo profile_standard | sudo -n tee "$c/power_dpm_force_performance_level" >/dev/null 2>&1
  now=$(cat "$c/power_dpm_force_performance_level")
  echo "  $(basename "$(dirname "$c")") -> $now"
  [ "$now" = "profile_standard" ] || { echo "SET FAILED — aborting"; exit 3; }
done

COUT=$RES/cache_gap_counters.csv
{
  echo "# cache_gap counter leg"
  echo "# date=$(date -Is) toolchain=$CNT_IMG (ROCm 7.14) perf_level=profile_standard"
  echo "# TIMES FROM THIS LEG ARE CLOCK-PINNED AND NOT COMPARABLE WITH THE TIMING LEG."
  echo "# HBM bytes = GL2C_MISS * 256 (calibrated on this card, docs/COUNTER_SCORECARD.md)"
  echo "label,loader,M,N,K,cond,flush_mb,counter,value"
} > "$COUT"

echo "=== compiling probe with the 7.14 toolchain (counters only) ==="
docker run "${DOCKER_COMMON[@]}" --entrypoint bash "$CNT_IMG" -lc "
  export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}
  hipcc -O3 --offload-arch=gfx1201 -I$INC -o /engine/tools/counter_probe/ksweep/_cgp_714 \
    /engine/tools/counter_probe/ksweep/cache_gap_probe.hip 2>&1 | grep -iE 'error' | head -20
  ls -la /engine/tools/counter_probe/ksweep/_cgp_714" 2>&1 | tail -3

# hot vs the largest flush: if the mechanism is cache eviction then GL2C_MISS must RISE and the L2
# hit rate must FALL between these two, on an otherwise identical launch. If they do not move, the
# gap is not cache state and the most attractive explanation is eliminated.
for spec in "bf16 1 32768 2048 lm_head" "bf16 1 6144 2048 in_proj_qkvz" "int4 1 2048 2048 moe_expert_like"; do
  read -r ldr M N K label <<< "$spec"
  for cc in "hot 0" "evict 256"; do
    read -r cond f <<< "$cc"
    for ctr in GL2C_MISS GL2C_HIT OccupancyPercent MemUnitBusy; do
      docker run "${DOCKER_COMMON[@]}" -v "$RES":/out --entrypoint bash "$CNT_IMG" -lc "
        export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}
        rm -rf /tmp/c; timeout -s KILL 200 rocprofv3 --pmc $ctr -f csv -d /tmp/c -- \
          /engine/tools/counter_probe/ksweep/_cgp_714 $ldr $M $N $K $cond $f 12 3 >/dev/null 2>&1
        f=\$(find /tmp/c -name '*counter*.csv' 2>/dev/null | head -1)
        [ -z \"\$f\" ] && { echo NA; exit 0; }
        # Counter_Value is \$(NF-2): Kernel_Name is a QUOTED field CONTAINING COMMAS, so a fixed
        # -F, column index reads a timestamp and prints 0 for everything, faking a broken-counter
        # result. This exact mistake once produced a false 'all counters broken' verdict.
        awk -F, '/gemv_decode_core/{ s+=\$(NF-2); n++ } END{ if(n) printf \"%.0f\", s/n; else print \"NA\" }' \"\$f\"
      " 2>/dev/null | tail -1 | sed "s/^/$label,$ldr,$M,$N,$K,$cond,$f,$ctr,/" >> "$COUT"
    done
    echo "  counters done: $label $cond flush=$f"
  done
done
echo "wrote $COUT"
