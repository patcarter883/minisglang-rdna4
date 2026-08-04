#!/usr/bin/env bash
# swa_radix_canvas_test.sh — the losslessness gate for SWA-radix on the BLOCK-DIFFUSION path.
#
# The autoregressive gate (041036ed) proved SWA-radix lossless on gemma-4 and Laguna. This is the
# same gate — same client, same matrix, same hit-counter evidence — pointed at the canvas execution
# mode, which shipped with SWA-radix forced OFF. Two things make it a different job:
#
#   1. THE CANVAS PATH HAS NO GREEDY MODE. `temperature 0 / top_p 1 / top_k 1` pins the AR sampler;
#      it pins NOTHING here, because a block starts as uniform noise over the whole vocabulary and
#      every denoising step draws a multinomial. Two identical requests to one serve return
#      different text, so an unseeded byte-identity gate is vacuous by construction. Every request
#      below carries `seed`, which pins the request's canvas generator (SamplingParams.seed) and
#      makes the whole trajectory a deterministic function of (prompt, seed).
#   2. THE FLOOR IS WHOLE-BLOCK, NOT PER-TOKEN. A block commits all 256 positions at once, so one
#      flipped denoising step rewrites the entire answer — there is no "stable for the first k
#      tokens" regime to hide in. `--mode repro` runs FIRST for that reason: if the seeded canvas is
#      not reproducible against itself, no cold-vs-hit verdict below means anything.
#
# THE COMPARISON MUST BE IN-PROCESS. A SWA_RADIX=1 serve and a SWA_RADIX=0 serve size their KV pools
# differently (the snapshot store is reserved only when the feature is on), so even a hit=0 request
# does not produce the same bytes across the two. Leg A therefore compares cold vs FULL hit inside
# one serve; leg C (SWA_RADIX=0) is a control that must show hit=0 everywhere, i.e. that leg A's
# identity came from the cache and not from the harness.
#
# THE PARTIAL-HIT CELL NEEDS TWO LEGS AT THE SAME SETTING — that is leg A vs leg B, read by
# swa_radix_canvas_verdict.py. Inside one serve the same prompt cannot be measured cold and then
# partially-hit (measuring it cold inserts it, so the second request is a FULL hit), and the AR
# gate's salted cold reference is meaningless on a canvas, so leg B is byte-identically configured to
# leg A and differs only in `--no-warm`: the shared prefix is never warmed, so leg B's `partial` step
# is a genuine cold measurement of that exact prompt. `--ref-repeats 0` drops the salted reference.
#
# Runs on the HOST under ONE lease covering all three legs:
#   gpu-lease -n 2 --timeout 5400 -- bash tools/swa_radix_canvas_test.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"

MODEL="${DG_MODEL:-cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4}"
IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:gemma4}"
SEED="${SEED:-20260804}"
RUN_ID="${RUN_ID:-canvas}"
CASES="${CASES:-short,xlong}"
TOKENS="${TOKENS:-16,32}"
REPEATS="${REPEATS:-5}"
PORT=1919
RES="$REPO/_swa_canvas_results.txt"
: > "$RES"
CNAME=""

start_serve() {  # $1 = MINISGL_SWA_RADIX value; $2 = tag
  CNAME="swacanvas-$2"
  local log="$REPO/_swa_canvas_$2.log"
  docker rm -f "$CNAME" >/dev/null 2>&1
  # The baked image, no kernel bind-mounts: a .so built elsewhere does not load here and the symptom
  # is a serve that never reaches the GPU rather than a build error. MINISGL_KV_FP8=0 because a
  # byte-identity gate cannot run over an fp8 KV cache.
  MINISGL_IMAGE="$IMAGE" MODEL="$MODEL" TP=2 SPEC=none PORT=$PORT CONC=1 \
    docker compose --profile serve run --rm --no-deps --service-ports --name "$CNAME" \
      -e MINISGL_KV_FP8=0 -e MINISGL_SWA_RADIX="$1" serve >"$log" 2>&1 &
  echo "== leg $2 (MINISGL_SWA_RADIX=$1) starting, log=$log"
}

wait_ready() {
  local i
  for i in $(seq 1 400); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "== ready after ~$((i*3))s"; return 0; }
    sleep 3
  done
  echo "!! readiness timeout"; tail -60 "$REPO"/_swa_canvas_*.log; return 1
}

stop_serve() {
  [ -n "$CNAME" ] || return 0
  docker rm -f "$CNAME" >/dev/null 2>&1
  local i; for i in $(seq 1 60); do docker ps -q -f "name=^${CNAME}$" | grep -q . || break; sleep 1; done
  CNAME=""
  sleep 5
}
trap stop_serve EXIT

client() { python3 "$REPO/tools/swa_radix_client.py" --seed "$SEED" --ref-repeats 0 "$@"; }

echo "########## LEG A: SWA-RADIX ON, prefix WARMED (real hits) ##########"
start_serve 1 on
if wait_ready; then
  grep -i "SWA-radix prefix cache ENABLED" "$REPO/_swa_canvas_on.log" || echo "(warn: enable log not found)"
  {
    echo "=== LEG A: MINISGL_SWA_RADIX=1 warm, seed=$SEED, model=$MODEL"
    echo "--- floor (repro): the seeded canvas against itself ---"
    client --mode repro --cases "$CASES" --token-counts "$TOKENS" --run-id "${RUN_ID}A" --repeats "$REPEATS"
    echo "--- matrix: cold vs full hit vs partial hit ---"
    client --mode matrix --cases "$CASES" --token-counts "$TOKENS" --run-id "${RUN_ID}M"
  } 2>&1 | tee -a "$RES"
  echo "--- SWA-radix HIT log line count ---" | tee -a "$RES"
  grep -ic "SWA-radix HIT" "$REPO/_swa_canvas_on.log" | tee -a "$RES"
fi
stop_serve

echo "########## LEG B: SWA-RADIX ON, prefix NOT warmed (the partial cell's cold reference) ##########"
start_serve 1 nowarm
if wait_ready; then
  {
    echo "=== LEG B: MINISGL_SWA_RADIX=1 --no-warm, seed=$SEED, model=$MODEL"
    client --mode matrix --cases "$CASES" --token-counts "$TOKENS" --run-id "${RUN_ID}M" --no-warm
  } 2>&1 | tee -a "$RES"
fi
stop_serve

echo "########## LEG C: SWA-RADIX OFF (control — hit must be 0 everywhere) ##########"
start_serve 0 off
if wait_ready; then
  {
    echo "=== LEG C: MINISGL_SWA_RADIX=0, seed=$SEED, model=$MODEL"
    client --mode matrix --cases "$CASES" --token-counts "$TOKENS" --run-id "${RUN_ID}M"
  } 2>&1 | tee -a "$RES"
fi
stop_serve

echo "########## SUMMARY ##########"
grep -E "^(VERDICT|REPRO)" "$RES"
python3 "$REPO/tools/swa_radix_canvas_verdict.py" "$RES"
echo "results: $RES"
