#!/usr/bin/env bash
# Matched A/B for PROPOSE GRAPH CAPTURE (deliverable 2), per proposer.
#
# ONE worktree, one image, one boot per leg. The two legs differ ONLY in whether the proposer's
# propose body is replayed from a CUDA graph or run eagerly — it is the SAME callable either way
# (spec/capture.py: `propose_body` is the captured region AND the eager fallback), which is what
# makes "replay == eager" a testable claim instead of an assertion.
#
#   leg cap    : default. Boot log must say "PROPOSE graphs CAPTURED buckets=[...]".
#   leg eager  : MINISGL_SPEC_PROPOSE_NOCAPTURE=1 — the TEMPORARY A/B override in spec/capture.py.
#                Boot log must say "propose capture disabled by A/B override".
#
# WHAT IS MEASURED
#   identity : MINISGL_SPEC_DEBUG=2 prints every drafted chain. Greedy, fixed prompt, fixed seed,
#              NREQ=1. The two legs' `draft=[...]` lines are diffed VERBATIM. This is the gate:
#              capture must be bit-identical to eager for the same inputs, so a single differing
#              chain is a FAILURE, not noise. (Contrast the O(window) work, where the reduction
#              length itself changed and ~1 ULP was expected — nothing changes here at all.)
#   timing   : MINISGL_SPEC_TIMING=1 logs a cuda-synchronized [spec-timing] running mean every 50
#              steps, and — added for this deliverable — the propose-graph replay/eager split on the
#              same line. A leg whose `replay=` stays 0 is not testing what it claims to test.
#   tok/s    : usage.completion_tokens on a NON-streaming request. Never SSE chunks (under spec one
#              chunk carries a whole accepted block).
#
# MUST be invoked UNDER the shared arbiter, which this script does NOT acquire:
#     MODEL=laguna SPEC=dflash gpu-lease -n 2 -- bash tools/propose_capture_ab.sh
set -uo pipefail

WT=/home/pat/code/minisgl-rdna4-propose
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
export MODEL="${MODEL:-laguna}" SPEC="${SPEC:-dflash}" TP="${TP:-2}"
export MEM_RATIO="${MEM_RATIO:-0.93}"
NREQ="${NREQ:-1}"
export CONC="${CONC:-$NREQ}" GRAPH_BS="${GRAPH_BS:-8}"
TAG="${TAG:-${SPEC}_n${NREQ}}"
OUT=${OUT:-$WT/tools/propose_capture_ab_${TAG}.txt}
: > "$OUT"

down() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
cleanup() { down; docker rm -f "$(docker ps -q --filter name=-serve)" >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

model_id() {
  ( cd "$WT" && DRY_RUN=1 MODEL="$MODEL" SPEC="$SPEC" bash tools/serve.sh 2>/dev/null \
      | tr ' ' '\n' | grep -A0 -m1 -E '^[^-].*/' ) || true
}

wait_ready() {
  for _ in $(seq 1 400); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

drive() {  # drive <max_tokens>
  MAXTOK="$1" NREQ="$NREQ" python3 - <<'PY'
import hashlib, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE="http://localhost:1919"
MODEL=json.loads(urllib.request.urlopen(f"{BASE}/v1/models",timeout=30).read())["data"][0]["id"]
NREQ=int(os.environ["NREQ"]); MT=int(os.environ["MAXTOK"])
PROMPT=("Write a complete, well-commented Python implementation of a persistent red-black tree, "
        "including insertion, deletion, rebalancing, and an in-order iterator. Explain the "
        "invariants as you go.")
def run(i):
    # Distinct suffix per stream so NREQ>1 really decodes concurrently instead of collapsing onto
    # one radix-cached prefix. At NREQ=1 the prompt is byte-identical across legs (identity gate).
    txt = PROMPT if NREQ == 1 else f"{PROMPT} (variant {i})"
    b={"model":MODEL,"messages":[{"role":"user","content":txt}],"max_tokens":MT,
       "temperature":0.0,"seed":1234,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=3600).read())
    return d["usage"]["completion_tokens"], d["choices"][0]["message"]["content"]
t=time.perf_counter()
with ThreadPoolExecutor(NREQ) as ex:
    res=list(ex.map(run, range(NREQ)))
w=time.perf_counter()-t
n=sum(k for k,_ in res)
print(f"TOKENS {n}")
print(f"WALL {w:.3f}")
print(f"TPS {n/w:.3f}")
print("MD5 " + hashlib.md5(res[0][1].encode()).hexdigest())
print("SAMPLE " + " ".join(res[0][1].split())[:110])
PY
}

# leg <cap|eager> <identity|timing> <max_tokens>
leg() {
  local tag=$1 mode=$2 mt=$3
  echo "=== $tag [$mode] MODEL=$MODEL SPEC=$SPEC NREQ=$NREQ ===" | tee -a "$OUT"
  down
  local envs=()
  [ "$mode" = identity ] && envs+=(MINISGL_SPEC_DEBUG=2) || envs+=(MINISGL_SPEC_TIMING=1)
  [ "$tag" = eager ] && envs+=(MINISGL_SPEC_PROPOSE_NOCAPTURE=1)
  ( cd "$WT" && env MINISGL_IMAGE="$IMAGE" "${envs[@]}" \
      docker compose --profile serve up -d >/dev/null 2>&1 )
  if ! wait_ready; then
    echo "  FAILED to become ready" | tee -a "$OUT"
    ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs --tail 60 2>&1 ) \
      | tail -60 | tee -a "$OUT"
    down; return 1
  fi
  # PROVENANCE. `sh -c` quoting matters: an unquoted redirect from /proc/1/environ is evaluated by
  # the HOST shell and reads the host's pid 1. And compose sets an unset var to the EMPTY STRING,
  # so "present" is not "set" — print the value.
  local c; c=$( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
  echo -n "  env witness: " | tee -a "$OUT"
  docker exec "$c" sh -c \
    "tr '\0' '\n' < /proc/1/environ | grep -E '^MINISGL_SPEC_(DEBUG|TIMING|PROPOSE_NOCAPTURE)=' | tr '\n' ' '" \
    2>/dev/null | tee -a "$OUT"; echo | tee -a "$OUT"
  echo "  capture witness:" | tee -a "$OUT"
  ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -oE 'PROPOSE graphs CAPTURED buckets=\[[^]]*\]|propose capture disabled by A/B override|propose ring \(.*\)|propose buffers \(.*\)' \
    | sort -u | sed 's/^/    /' | tee -a "$OUT"

  drive "$mt" 2>&1 | tee -a "$OUT"

  if [ "$mode" = identity ]; then
    ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
      | grep -o 'draft=\[[^]]*\]' > "$WT/tools/_pcab_${TAG}_${tag}.txt"
    echo "  drafted chains captured: $(wc -l < "$WT/tools/_pcab_${TAG}_${tag}.txt")" | tee -a "$OUT"
  else
    echo "  [spec-timing] lines:" | tee -a "$OUT"
    ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
      | grep -o '\[spec-timing\].*' | tee -a "$OUT"
  fi
  down
  echo | tee -a "$OUT"
}

MODE="${MODE:-both}"
if [ "$MODE" = identity ] || [ "$MODE" = both ]; then
  leg cap   identity "${IDENT_TOK:-512}"
  leg eager identity "${IDENT_TOK:-512}"
  echo "=== REPLAY-vs-EAGER DRAFT IDENTITY ($TAG) ===" | tee -a "$OUT"
  a="$WT/tools/_pcab_${TAG}_cap.txt"; b="$WT/tools/_pcab_${TAG}_eager.txt"
  # At NREQ=1 the log ORDER is deterministic, so a verbatim diff is the gate. At NREQ>1 it is NOT:
  # each leg is an independent boot, so request admission order and the per-step batch composition
  # differ, which permutes the interleaving of the per-request debug lines WITHOUT changing any
  # chain. So compare the MULTISET there and say which test was applied — a permuted-but-equal set
  # is a pass, and calling it a failure would be an artefact of the instrument, not a result.
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then
    echo "  NO DRAFT CHAINS CAPTURED — the identity gate did not run" | tee -a "$OUT"
  elif diff -q "$a" "$b" >/dev/null; then
    echo "  IDENTICAL (verbatim): $(wc -l < "$a") drafted chains" | tee -a "$OUT"
  elif diff -q <(sort "$a") <(sort "$b") >/dev/null; then
    echo "  IDENTICAL (multiset): $(wc -l < "$a") drafted chains; $(diff "$a" "$b" | grep -c '^<')" \
         "lines differ in LOG ORDER only (independent boots at NREQ=$NREQ)" | tee -a "$OUT"
  else
    echo "  DIFFER: $(diff <(sort "$a") <(sort "$b") | grep -c '^<') of $(wc -l < "$a") chains" \
      | tee -a "$OUT"
    diff <(sort "$a") <(sort "$b") | head -20 | tee -a "$OUT"
  fi
  echo | tee -a "$OUT"
fi

if [ "$MODE" = timing ] || [ "$MODE" = both ]; then
  leg cap   timing "${TIME_TOK:-1024}"
  leg eager timing "${TIME_TOK:-1024}"
fi

down
echo "results -> $OUT"
