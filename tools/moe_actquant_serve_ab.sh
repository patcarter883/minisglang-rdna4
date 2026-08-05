#!/usr/bin/env bash
# Served decode A/B for the MoE PRODUCER-SIDE activation-quant fusion.
#
#   gpu-lease -n 2 --timeout 7200 -- bash tools/moe_actquant_serve_ab.sh
#
# WHAT IS HELD CONSTANT: both legs mount the SAME freshly-built fp8_wmma package (_kern/fp8_wmma,
# which already accepts the pre-quantized pair). The ONLY difference is whether the engine SUPPLIES
# the pair. That is what makes this an A/B of the fusion and not of a kernel rebuild.
#
# PROVENANCE (mandatory, learned the hard way): each leg dumps the engage ledger. `+prequant` MUST
# appear in the NEW leg and MUST NOT appear in the BASE leg. If that assertion fails the numbers are
# new-vs-itself and the run is void — the script says so and exits nonzero.
#
# BAND: DECODE. Reported at bs = 1, 5, 6, 30 (M=1 is bs=1; the others are real served concurrency).
# This is a LAUNCH-COUNT win — it removes one dispatch per MoE layer per step and therefore does not
# scale with M. Do NOT quote a prefill number for it.
set -uo pipefail

NEW_WT="${NEW_WT:-/home/pat/code/minisgl-rdna4-moeactq}"
BASE_WT="${BASE_WT:-/home/pat/code/minisgl-rdna4-moeactq-base}"
IMG="${MOE_IMG:-minisgl-rdna4:post-tile}"
MODEL="${MODEL:-qwen35b-awq}"
TOKS="${TOKS:-256}"
REPS="${REPS:-3}"
# CONC == --max-running-requests, and serve.sh makes GRAPH_BS FOLLOW CONC. A first attempt at
# CONC=32 therefore captured graphs up to bs=32, whose buffers ate the pool and killed both ranks
# with "Not enough memory for KV cache after reserving ... CUDA-graph buffers" -- while the
# container stayed Up, which is why this script now gates on VRAM and not on `docker ps`.
# 6 is the recorded 35B/16GB TP=2 cap, so it is also the real served ceiling: M can never exceed 6
# on this model/box, which is why the bs sweep below stops there.
CONC="${CONC:-6}"
# PIN the pool. Two reasons, one of them correctness-of-measurement: an auto-sized pool differs
# between the legs (the .so size alone shifts it), so the two legs would admit different amounts of
# work and the comparison would be of admission policy, not of GEMM cost.
# 2048 pages x page_size 16 = 32768 tokens, ~18x the 6 x (prompt+256) this drives.
NUM_PAGES="${NUM_PAGES:-2048}"
OUT="${OUT:-$NEW_WT/tools/_fixtures/moe_actquant_serve_ab.txt}"
mkdir -p "$(dirname "$OUT")"

# READINESS GATE. "Up" is not "ready": the scheduler subprocesses can die inside a container whose
# main process persists, and then a naive poll waits out the full timeout on a corpse.
#
# The gate is the LOG, not VRAM. A first version failed a boot whenever VRAM was still ~0 after 60s,
# which is a FALSE POSITIVE: loading a 35B checkpoint takes longer than that before any VRAM climbs,
# and it killed a leg that was merely slow. A rank that has actually died has printed a traceback;
# one that is loading has not. So: fail fast on a fatal log line, and otherwise just wait.
fatal_log() {
  ( cd "$WT" && docker compose -p lease-moeab --profile serve logs --no-color 2>&1 ) \
    | grep -qE 'Traceback \(most recent call last\)|AssertionError|RuntimeError|CUDA error|HIP error|torch.OutOfMemoryError'
}
wait_ready() {
  local i
  for i in $(seq 1 150); do
    curl -sf -m 2 http://127.0.0.1:1919/health >/dev/null 2>&1 && return 0
    if ! docker ps --format '{{.Names}}' | grep -q '^moeab-serve$'; then
      echo "  container exited during boot"; return 1
    fi
    if fatal_log; then echo "  FATAL in rank logs after ${i}0s"; return 1; fi
    sleep 5
  done
  echo "  readiness TIMEOUT"; return 1
}

: > "$OUT"
echo "MoE producer act-quant serve A/B  model=$MODEL TP=2  image=$IMG  toks=$TOKS reps=$REPS" | tee -a "$OUT"
echo "kernel package (CONSTANT across legs): _kern/fp8_wmma" | tee -a "$OUT"

FAIL=0
for leg in base new; do
  case "$leg" in
    new)  WT="$NEW_WT"  ;;
    base) WT="$BASE_WT" ;;
  esac
  echo "" | tee -a "$OUT"
  echo "########## LEG $leg  ($WT @ $(git -C "$WT" rev-parse --short HEAD)) ##########" | tee -a "$OUT"

  ( cd "$WT" && \
    MINISGL_IMAGE="$IMG" \
    MINISGL_PYTHONPATH="/engine/_kern:/opt/kernels:/engine/python:/engine" \
    MODEL="$MODEL" TP=2 CONC="$CONC" LEASE_NAME="moeab" \
    EXTRA_ARGS="--num-pages $NUM_PAGES" \
    docker compose -p "lease-moeab" --profile serve up -d ) >/dev/null 2>&1

  if ! wait_ready; then
    echo "  BOOT FAILED:" | tee -a "$OUT"
    ( cd "$WT" && docker compose -p lease-moeab --profile serve logs --tail 40 ) 2>&1 | sed 's/^/    /' | tee -a "$OUT"
    ( cd "$WT" && docker compose -p lease-moeab --profile serve down ) >/dev/null 2>&1
    FAIL=1; continue
  fi

  # ---- PROVENANCE: which kernel arms actually engaged this boot ----
  LEDGER=$( ( cd "$WT" && docker compose -p lease-moeab --profile serve logs 2>&1 ) \
            | grep -oE 'fp8_wmma\.mmq_fp8_moe_gemm1_silu\([^)]*\)|fp8_wmma\.mmq_fp8_moe_gemm1_silu_flag[^ ]*' \
            | sort -u | tr '\n' ' ')
  echo "  engaged gemm1 arms: ${LEDGER:-<none captured>}" | tee -a "$OUT"
  if [ "$leg" = "new" ]; then
    case "$LEDGER" in *prequant*) echo "  PROVENANCE OK: +prequant present" | tee -a "$OUT" ;;
      *) echo "  PROVENANCE FAIL: new leg did NOT engage +prequant — A/B is void" | tee -a "$OUT"; FAIL=1 ;;
    esac
  else
    case "$LEDGER" in *prequant*) echo "  PROVENANCE FAIL: base leg engaged +prequant — worktrees crossed" | tee -a "$OUT"; FAIL=1 ;;
      *) echo "  PROVENANCE OK: no +prequant on base" | tee -a "$OUT" ;;
    esac
  fi

  # ---- decode throughput at the REAL decode points ----
  # M = concurrent decode rows. --max-running-requests caps it at CONC, so bs > CONC would just
  # QUEUE and still decode at M=CONC -- it would measure the admission queue, not the GEMM. On the
  # 35B at TP=2 on 16 GB cards that ceiling is 6, so M=30 is simply not reachable for this model.
  for bs in 1 5 6; do
    echo "  --- bs=$bs ---" | tee -a "$OUT"
    for r in $(seq 1 "$REPS"); do
      python3 "$NEW_WT/tools/_moe_actquant_driver.py" "$bs" "$TOKS" 2>&1 | sed "s/^/    rep$r /" | tee -a "$OUT"
    done
  done

  ( cd "$WT" && docker compose -p lease-moeab --profile serve down ) >/dev/null 2>&1
  sleep 20   # let the container's KFD allocations actually return to the driver
done

echo "" | tee -a "$OUT"
[ "$FAIL" = 0 ] && echo "=== done (provenance asserted) ===" | tee -a "$OUT" || echo "=== DONE WITH FAILURES ===" | tee -a "$OUT"
exit "$FAIL"
