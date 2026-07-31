#!/usr/bin/env bash
# Matched A/B for the DFlash O(window) propose rewrite (deliverable 1).
#
# TWO LEGS, TWO WORKTREES, SAME IMAGE. The engine is pure Python and hot-mounted at /engine, so the
# only difference between the legs is the tree that gets mounted:
#     base = /home/pat/code/minisgl-rdna4-proposebase  (d276137c, untouched)
#     new  = /home/pat/code/minisgl-rdna4-propose      (this work)
# No env flag selects between them, so there is no "flag didn't apply" failure mode — but that also
# means provenance MUST be asserted from inside the container, which `leg()` does by md5'ing the
# mounted spec/dflash.py and printing the git sha of the mounted tree.
#
# WHAT IS MEASURED
#   identity : MINISGL_SPEC_DEBUG=2 makes the proposer print every drafted chain. Greedy, fixed
#              prompt, fixed seed. The two legs' [dflash-dbg] draft= lines are diffed VERBATIM. This
#              is the deliverable's gate: every removed key is one the drafter's own mask discards,
#              so the drafted ids must not move.
#   timing   : MINISGL_SPEC_TIMING=1 logs a cuda-synchronized [spec-timing] propose=..ms running mean
#              every 50 steps. Consecutive lines are differenced to recover the per-50-step mean, so
#              propose cost can be read as a function of the drafter's prefix length P (which grows
#              by one per ACCEPTED token: P = seed_tail + generated, NOT the prompt length).
#   tok/s    : usage.completion_tokens on a NON-streaming request. Never SSE chunks.
#
# The long leg generates 1600 tokens on purpose: the drafter's prefix is seeded with only the last 64
# prompt aux positions (prefill_aux_tail), so PROMPT length barely moves P — GENERATION does. A short
# generation never leaves the P < window regime and would measure nothing.
#
# MUST be invoked UNDER the shared arbiter, which this script does NOT acquire:
#     gpu-lease -n 2 -- bash tools/dflash_owindow_ab.sh
set -uo pipefail

NEW_WT=/home/pat/code/minisgl-rdna4-propose
BASE_WT=/home/pat/code/minisgl-rdna4-proposebase
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
NREQ="${NREQ:-1}"
OUT=${OUT:-$NEW_WT/tools/dflash_owindow_ab_n$NREQ.txt}
MODEL_ID=poolside/Laguna-XS-2.1-NVFP4
: > "$OUT"

export MODEL=laguna SPEC=dflash TP=2 MEM_RATIO=0.93
export CONC="${CONC:-$NREQ}" GRAPH_BS="${GRAPH_BS:-8}"

down() { ( cd "$1" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }

wait_ready() {
  for _ in $(seq 1 300); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

drive() {  # drive <max_tokens> ; prints "TOKENS <n>" "WALL <s>" "TPS <t>" and the completion md5
  MAXTOK="$1" NREQ="$NREQ" MODEL_ID="$MODEL_ID" python3 - <<'PY'
import hashlib, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE="http://localhost:1919"; MODEL=os.environ["MODEL_ID"]
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

# leg <tag> <worktree> <mode: identity|timing> <max_tokens>
leg() {
  local tag=$1 wt=$2 mode=$3 mt=$4
  echo "=== $tag [$mode] ===" | tee -a "$OUT"
  down "$NEW_WT"; down "$BASE_WT"
  local envs=()
  if [ "$mode" = identity ]; then envs=(MINISGL_SPEC_DEBUG=2); else envs=(MINISGL_SPEC_TIMING=1); fi
  ( cd "$wt" && env MINISGL_IMAGE="$IMAGE" "${envs[@]}" \
      docker compose --profile serve up -d >/dev/null 2>&1 )
  if ! wait_ready; then
    echo "  FAILED to become ready" | tee -a "$OUT"
    ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs --tail 40 2>&1 ) \
      | tail -40 | tee -a "$OUT"
    down "$wt"; return 1
  fi
  # PROVENANCE. The legs differ only by which tree is bind-mounted, so assert WHICH SOURCE is inside
  # the container -- an env var would prove nothing here. `sh -c` quoting matters: an unquoted
  # redirect from /proc/1/environ is evaluated by the HOST shell and reads the host's pid 1.
  local c; c=$( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
  echo -n "  mounted spec/dflash.py md5: " | tee -a "$OUT"
  docker exec "$c" sh -c "md5sum /engine/python/minisgl/spec/dflash.py | cut -d' ' -f1" \
    2>/dev/null | tee -a "$OUT"
  echo -n "  mounted models/dflash.py md5: " | tee -a "$OUT"
  docker exec "$c" sh -c "md5sum /engine/python/minisgl/models/dflash.py | cut -d' ' -f1" \
    2>/dev/null | tee -a "$OUT"
  echo -n "  env witness: " | tee -a "$OUT"
  docker exec "$c" sh -c \
    "tr '\0' '\n' < /proc/1/environ | grep -E '^MINISGL_SPEC_(DEBUG|TIMING)=' | tr '\n' ' '" \
    2>/dev/null | tee -a "$OUT"; echo | tee -a "$OUT"

  drive "$mt" 2>&1 | tee -a "$OUT"

  if [ "$mode" = identity ]; then
    ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
      | grep -o 'draft=\[[^]]*\]' > "$NEW_WT/tools/_owab_${tag}_drafts.txt"
    echo "  drafted chains captured: $(wc -l < "$NEW_WT/tools/_owab_${tag}_drafts.txt")" | tee -a "$OUT"
  else
    echo "  [spec-timing] lines:" | tee -a "$OUT"
    ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
      | grep -o '\[spec-timing\].*' | tee -a "$OUT"
  fi
  down "$wt"
  echo | tee -a "$OUT"
}

MODE="${MODE:-both}"
if [ "$MODE" = identity ] || [ "$MODE" = both ]; then
  # 640 generated tokens takes the drafter prefix P from 64 to ~704, i.e. ACROSS the 512 window, so
  # the identity gate covers both the P<W (no slicing) and P>W (slicing active) regimes.
  leg base "$BASE_WT" identity "${IDENT_TOK:-640}"
  leg new  "$NEW_WT"  identity "${IDENT_TOK:-640}"
  echo "=== DRAFT-CHAIN IDENTITY ===" | tee -a "$OUT"
  if diff -q "$NEW_WT/tools/_owab_base_drafts.txt" "$NEW_WT/tools/_owab_new_drafts.txt" >/dev/null; then
    echo "  IDENTICAL: $(wc -l < "$NEW_WT/tools/_owab_new_drafts.txt") drafted chains match verbatim" \
      | tee -a "$OUT"
  else
    echo "  DIFFER: $(diff "$NEW_WT/tools/_owab_base_drafts.txt" "$NEW_WT/tools/_owab_new_drafts.txt" \
      | grep -c '^<') of $(wc -l < "$NEW_WT/tools/_owab_base_drafts.txt") lines" | tee -a "$OUT"
    diff "$NEW_WT/tools/_owab_base_drafts.txt" "$NEW_WT/tools/_owab_new_drafts.txt" | head -20 \
      | tee -a "$OUT"
  fi
  echo | tee -a "$OUT"
fi

if [ "$MODE" = timing ] || [ "$MODE" = both ]; then
  leg base "$BASE_WT" timing "${TIME_TOK:-1600}"
  leg new  "$NEW_WT"  timing "${TIME_TOK:-1600}"
fi

down "$NEW_WT"; down "$BASE_WT"
echo "results -> $OUT"
