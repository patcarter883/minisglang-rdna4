#!/usr/bin/env bash
# Served decode A/B for the FUSED MoE gemm2 top_k split (grid.z).
#
#   gpu-lease -n 2 --timeout 7200 -- bash tools/moe_g2_split_serve_ab.sh
#
# WHAT IS HELD CONSTANT — everything, including the binary. Both legs mount the SAME freshly built
# fp8_wmma .so from the SAME worktree; the only difference is `MINISGL_MOE_G2_SPLIT_K=1`, which
# forces the launcher back to the unsplit grid. That is a stronger control than two builds: it rules
# out a compiler/layout difference being read as the effect, and the isolated bench separately shows
# the sk=1 arm of the new build reproduces the old build's time.
#
# PROVENANCE (mandatory). Each leg boots with MINISGL_MOE_G2_SPLIT_DEBUG=1, so the launcher prints
# the split it chose for the real served shape. The BASE leg must show `split_k=1` and the NEW leg
# must show `split_k>1`, or the comparison is new-vs-itself and the run is VOID.
#
# BAND: DECODE. bs = 1, 5, 6.
#   * bs=1 is M=1, which does NOT reach this kernel at all (w4a8_moe takes the M<=2 atomic-scatter
#     branch). It is here as a NEGATIVE CONTROL: the split must move it by nothing but noise.
#   * bs=5 and bs=6 are M=5 / M=6, the served points that DO reach the fused gather-reduce.
#     CONC=6 is the recorded 35B/16GB TP=2 ceiling, so M can never exceed 6 on this model/box.
set -uo pipefail

WT="${WT:-/home/pat/code/minisgl-rdna4-g2sk}"
KERN="${KERN:-/home/pat/code/rdna4-hip-kernels-g2sk}"
IMG="${G2_IMG:-minisgl-rdna4:post-tile}"
MODEL="${MODEL:-qwen35b-awq}"
TOKS="${TOKS:-256}"
REPS="${REPS:-3}"
CONC="${CONC:-6}"
# PIN the pool: an auto-sized pool differs between legs, so the legs would admit different work and
# the comparison would be of admission policy rather than of the GEMM.
NUM_PAGES="${NUM_PAGES:-2048}"
PROJ="lease-g2sk"
OUT="${OUT:-$WT/tools/_fixtures/moe_g2_split_serve_ab.txt}"
mkdir -p "$(dirname "$OUT")"

# The engine imports fp8_wmma from _kern first; make it the freshly built package.
rm -rf "$WT/_kern"; mkdir -p "$WT/_kern"
cp -a "$KERN/fp8_wmma/torch-ext/fp8_wmma" "$WT/_kern/fp8_wmma"

fatal_log() {
  ( cd "$WT" && docker compose -p "$PROJ" --profile serve logs --no-color 2>&1 ) \
    | grep -qE 'Traceback \(most recent call last\)|AssertionError|RuntimeError|CUDA error|HIP error|torch.OutOfMemoryError'
}
wait_ready() {  # "Up" is NOT ready; fail fast on a rank traceback, otherwise wait.
  local i
  for i in $(seq 1 150); do
    curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0
    docker ps --format '{{.Names}}' | grep -q "^g2sk-serve$" || { echo "  container exited"; return 1; }
    if fatal_log; then echo "  FATAL in rank logs after ${i}0s"; return 1; fi
    sleep 5
  done
  echo "  readiness TIMEOUT"; return 1
}

: > "$OUT"
echo "MoE fused-gemm2 top_k split serve A/B  model=$MODEL TP=2  image=$IMG toks=$TOKS reps=$REPS conc=$CONC" | tee -a "$OUT"
echo "engine  $WT   @ $(git -C "$WT" rev-parse --short HEAD)" | tee -a "$OUT"
echo "kernels $KERN @ $(git -C "$KERN" rev-parse --short HEAD)  (ONE build, both legs)" | tee -a "$OUT"

FAIL=0
for leg in base new; do
  case "$leg" in
    base) SKENV="MINISGL_MOE_G2_SPLIT_K=1" ;;   # forced unsplit
    new)  SKENV="MINISGL_MOE_G2_SPLIT_K=" ;;    # derived from occupancy
  esac
  echo "" | tee -a "$OUT"
  echo "########## LEG $leg  ($SKENV) ##########" | tee -a "$OUT"

  # `env` and not an assignment prefix: `VAR=x $SKENV cmd` expands $SKENV AFTER the shell has
  # finished looking for assignments, so it would be run as the COMMAND, not set as a variable.
  ( cd "$WT" && env \
    MINISGL_IMAGE="$IMG" \
    MINISGL_PYTHONPATH="/engine/_kern:/opt/kernels:/engine/python:/engine" \
    MINISGL_MOE_G2_SPLIT_DEBUG=1 "$SKENV" \
    MODEL="$MODEL" TP=2 CONC="$CONC" LEASE_NAME="g2sk" \
    EXTRA_ARGS="--num-pages $NUM_PAGES" \
    docker compose -p "$PROJ" --profile serve up -d ) >/dev/null 2>&1

  if ! wait_ready; then
    echo "  BOOT FAILED:" | tee -a "$OUT"
    ( cd "$WT" && docker compose -p "$PROJ" --profile serve logs --tail 40 ) 2>&1 | sed 's/^/    /' | tee -a "$OUT"
    ( cd "$WT" && docker compose -p "$PROJ" --profile serve down ) >/dev/null 2>&1
    FAIL=1; continue
  fi

  # ---- drive first so the served shape is actually reached, THEN read the ledger ----
  for bs in 1 5 6; do
    echo "  --- bs=$bs ---" | tee -a "$OUT"
    for r in $(seq 1 "$REPS"); do
      python3 "$WT/tools/_moe_actquant_driver.py" "$bs" "$TOKS" 2>&1 | sed "s/^/    rep$r /" | tee -a "$OUT"
    done
  done

  LEDGER=$( ( cd "$WT" && docker compose -p "$PROJ" --profile serve logs 2>&1 ) \
            | grep -oE '\[g2-split\][^\n]*' | sort -u | tr '\n' '|' )
  echo "  g2-split ledger: ${LEDGER:-<none captured>}" | tee -a "$OUT"
  SPLITS=$(printf '%s' "$LEDGER" | grep -oE 'split_k=[0-9]+' | sort -u | tr '\n' ' ')
  echo "  split_k values seen: ${SPLITS:-<none>}" | tee -a "$OUT"
  if [ "$leg" = "base" ]; then
    case "$SPLITS" in
      "split_k=1 ") echo "  PROVENANCE OK: base ran unsplit" | tee -a "$OUT" ;;
      *) echo "  PROVENANCE FAIL: base saw '$SPLITS' — A/B void" | tee -a "$OUT"; FAIL=1 ;;
    esac
  else
    case "$SPLITS" in
      *split_k=[2-9]*) echo "  PROVENANCE OK: new ran split" | tee -a "$OUT" ;;
      *) echo "  PROVENANCE FAIL: new never split ('$SPLITS') — A/B void" | tee -a "$OUT"; FAIL=1 ;;
    esac
  fi

  ( cd "$WT" && docker compose -p "$PROJ" --profile serve down ) >/dev/null 2>&1
  sleep 20   # let the container's KFD allocations return to the driver
done

echo "" | tee -a "$OUT"
[ "$FAIL" = 0 ] && echo "=== done (provenance asserted) ===" | tee -a "$OUT" \
                || echo "=== DONE WITH FAILURES ===" | tee -a "$OUT"
exit "$FAIL"
