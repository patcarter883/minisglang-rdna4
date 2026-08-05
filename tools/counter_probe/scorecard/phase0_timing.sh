#!/usr/bin/env bash
# phase0_timing.sh — IN-CONTAINER. Kernel DURATIONS at the card's normal (auto) perf level.
#
# WHY THIS IS A SEPARATE PHASE, OUTSIDE THE profile_standard WINDOW.
# profile_standard pins BOTH core and memory clocks to a fixed non-boost state. Measured here: the
# int4 16384^2 GEMV takes 949,906 ns pinned, which works out to ~141 GB/s -- against a claim of
# 547 GB/s at normal clocks. That is not a refutation of anything, it is the clock. Any bandwidth
# percentage computed from a profile_standard timestamp is wrong by roughly the clock ratio, and
# would look exactly like a kernel that had regressed.
#
# So the two halves are collected separately and combined:
#   BYTES  <- counters, under profile_standard (a clock-independent property of the algorithm)
#   TIME   <- here, under auto (the number a user actually experiences)
#
# --kernel-trace, NOT --pmc: this needs no perfmon counters, so it needs no ungated perfmon clock,
# so it can run at auto. That is the whole trick.
set -uo pipefail
OUT="${OUT:-/out}/phase0"; mkdir -p "$OUT"

echo "=== phase0: build ==="
hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm \
      -o /tmp/kp /probe/scorecard/kernel_probes.hip 2>&1 | grep -iE "error" | head -20
[ -x /tmp/kp ] || { echo "BUILD FAILED"; exit 1; }

trace() {  # trace <tag> <args...>
  local tag="$1"; shift
  local d="$OUT/$tag"; rm -rf "$d"; mkdir -p "$d"
  echo "--- trace $tag :: /tmp/kp $* ---"
  timeout -s KILL 300 rocprofv3 --kernel-trace -f csv -d "$d" -- /tmp/kp "$@" >"$d/run.log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || echo "    rc=$rc (see $d/run.log)"
}

trace dense_4096   dense 1 4096 4096 128
# PRODUCTION decode shapes (Qwen3.6-35B-A3B TP=2, hidden 2048). The synthetic 16384^2 below is the
# regime the "81% of HBM" verdict was taken in; these are the shapes the engine actually launches,
# and the two do not agree.
trace dense_prod_2048 dense 1 2048 2048 128
trace dense_prod_6144 dense 1 6144 2048 128
trace dense_16384  dense 1 16384 16384 128
trace bf16_down    bf16 1 2048 256
trace bf16_gate    bf16 1 1 2048
trace bf16_qkvz    bf16 1 6144 2048
trace bf16_lmhead  bf16 1 32768 2048
for m in 1 5 6 30; do
  trace "moe1_M$m" moe1 $m 32 8 512 2048 128 16
  trace "moe2_M$m" moe2 $m 32 8 512 2048 128 16
done

echo "=== phase0 done (auto perf level) ==="

# Fused MoE decode gemm2 — the production path. Needs its own binary.
hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm \
      -o /tmp/g2 /probe/scorecard/moe_g2fuse_probe.hip 2>&1 | grep -iE " error" | head -10
if [ -x /tmp/g2 ]; then
  trace_g2() { local tag="$1"; shift; local d="$OUT/$tag"; rm -rf "$d"; mkdir -p "$d"
    echo "--- trace $tag :: /tmp/g2 $* ---"
    timeout -s KILL 300 rocprofv3 --kernel-trace -f csv -d "$d" -- /tmp/g2 "$@" >"$d/run.log" 2>&1; }
  for m in 1 5 6 30; do
    trace_g2 "g2f_K512_M$m" $m 32 8 512 2048 128 16
    trace_g2 "g2f_K256_M$m" $m 32 8 256 2048 128 16
  done
fi
echo "=== phase0 (with g2fuse) done ==="
