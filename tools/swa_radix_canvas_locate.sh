#!/usr/bin/env bash
# swa_radix_canvas_locate.sh — LOCATE the block-diffusion partial-hit divergence, and re-run the AR guard.
#
# §D5 measured a PARTIAL prefix hit whose text differs from the same prompt served cold, and offered
# two causes. One of them is dead: this engine's dense path does not use rocBLAS and its kernels are
# M-invariant by construction (`layers/minv.py::minv_linear` exists precisely so a chunked prefill is
# bit-identical to a single pass, verified 0.0 across all 40 CCA layers by tools/cca_chunk_bisect.py,
# and Gemma4's router goes through it). So "prefill-shape numerics" cannot explain it and the
# divergence is a DEFECT. This run locates it instead of arguing about it.
#
# THREE LEGS, ONE LEASE:
#
#   A  SWA_RADIX=1, prefix WARMED   -> B2 takes a PARTIAL hit. Per-layer KV state digested.
#   B  SWA_RADIX=1, --no-warm       -> the SAME B2 served COLD. Per-layer KV state digested.
#      canvas_state_bisect.py diffs A against B. A prefill is correct iff the KV it leaves behind is
#      bit-identical to a cold one's, because every downstream stage is a pure function of it — so
#      this either names the first differing layer and pool, or exonerates the prefill entirely and
#      sends the hunt downstream to the canvas trajectory. Text cannot make that distinction.
#   C  gemma-4 AUTOREGRESSIVE guard -> the 44.1/44.2 tok/s that every change on this branch is
#      measured against. It is here rather than in its own lease because the cards are contended and
#      a guard that is skipped for lack of a window is a guard that does not exist.
#
# Runs on the HOST:  gpu-lease -n 2 --timeout 5400 -- bash tools/swa_radix_canvas_locate.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"

MODEL="${DG_MODEL:-cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4}"
AR_MODEL="${AR_MODEL:-cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4}"
IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:gemma4}"
SEED="${SEED:-20260804}"
RUN_ID="${RUN_ID:-loc}"
CASES="${CASES:-xlong}"          # the cell that DIVERGED; short/* is inadmissible (see §D5)
TOKENS="${TOKENS:-16}"
PORT=1919
RES="$REPO/_swa_locate_results.txt"
: > "$RES"
CNAME=""

start_serve() {  # $1 = model  $2 = tag  $3 = MINISGL_SWA_RADIX  $4 = MINISGL_STATE_DIGEST
  CNAME="swaloc-$2"
  docker rm -f "$CNAME" >/dev/null 2>&1
  MINISGL_IMAGE="$IMAGE" MODEL="$1" TP=2 SPEC=none PORT=$PORT CONC=1 \
    docker compose --profile serve run --rm --no-deps --service-ports --name "$CNAME" \
      -e MINISGL_KV_FP8=0 -e MINISGL_SWA_RADIX="$3" -e MINISGL_STATE_DIGEST="$4" \
      serve >"$REPO/_swaloc_$2.log" 2>&1 &
  echo "== leg $2 (SWA_RADIX=$3 STATE_DIGEST=$4) starting"
}

wait_ready() {
  local i
  for i in $(seq 1 400); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "== ready ~$((i*3))s"; return 0; }
    sleep 3
  done
  echo "!! readiness timeout"; tail -40 "$REPO"/_swaloc_*.log; return 1
}

stop_serve() {
  [ -n "$CNAME" ] || return 0
  docker rm -f "$CNAME" >/dev/null 2>&1
  local i; for i in $(seq 1 60); do docker ps -q -f "name=^${CNAME}$" | grep -q . || break; sleep 1; done
  CNAME=""; sleep 5
}
trap stop_serve EXIT

client() { python3 "$REPO/tools/swa_radix_client.py" --seed "$SEED" --ref-repeats 0 "$@"; }

echo "########## LEG A: partial HIT, state digested ##########"
start_serve "$MODEL" hit 1 1
if wait_ready; then
  { echo "=== LEG A: partial hit"
    client --mode matrix --cases "$CASES" --token-counts "$TOKENS" --run-id "${RUN_ID}M"
  } 2>&1 | tee -a "$RES"
fi
stop_serve

echo "########## LEG B: the SAME prompt COLD, state digested ##########"
start_serve "$MODEL" cold 1 1
if wait_ready; then
  { echo "=== LEG B: cold reference"
    client --mode matrix --cases "$CASES" --token-counts "$TOKENS" --run-id "${RUN_ID}M" --no-warm
  } 2>&1 | tee -a "$RES"
fi
stop_serve

echo "########## BISECT ##########"
python3 "$REPO/tools/canvas_state_bisect.py" "$REPO/_swaloc_hit.log" "$REPO/_swaloc_cold.log" 2>&1 | tee -a "$RES"

# AR=0 skips leg C. Use it when the guard is already green for this tree and the cards are
# contended: re-leasing two cards to re-measure an unchanged number starves other agents.
if [ "${AR:-1}" = "1" ]; then
echo "########## LEG C: the AUTOREGRESSIVE GUARD (gemma-4 tok/s) ##########"
# Digest OFF here: it is a per-request host-side hash of the whole prefix and would corrupt the very
# number this leg exists to produce.
start_serve "$AR_MODEL" ar 1 0
if wait_ready; then
  { echo "=== LEG C: gemma-4 AR guard, SWA-radix ON"
    for p in "Write one paragraph explaining why the sky is blue." \
             "List three differences between a list and a tuple in Python."; do
      python3 - "$p" <<'PY'
import json, sys, time, urllib.request
body = json.dumps({"model": "x", "max_tokens": 256, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
                   "messages": [{"role": "user", "content": sys.argv[1]}]}).encode()
r = urllib.request.Request("http://localhost:1919/v1/chat/completions", body,
                           {"Content-Type": "application/json"})
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(r, timeout=900) as resp:
        d = json.load(resp)
    dt = time.perf_counter() - t0
    n = d.get("usage", {}).get("completion_tokens", 0)
    print(f"AR-GUARD: {n} tokens in {dt:.2f}s = {n/max(dt,1e-9):.1f} tok/s = {dt/max(n,1)*1000:.1f} ms/token")
except Exception as e:
    print(f"AR-GUARD FAILED: {type(e).__name__}: {e}")
PY
    done
  } 2>&1 | tee -a "$RES"
fi
stop_serve
fi

echo "########## SUMMARY ##########"
grep -E "^VERDICT|^AR-GUARD|IDENTICAL —|DIFFER in|first differing|^PASS|^FAIL" "$RES"
echo "results: $RES"
