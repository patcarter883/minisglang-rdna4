#!/usr/bin/env bash
# What DELIVERABLE 2 actually bought, end to end: the per-request EAGER propose it replaced vs the
# batched CAPTURED propose it ships.
#
# `propose_capture_ab.sh` isolates the graph REPLAY alone (both legs run the same batched body). That
# is the correctness gate, but it under-states the deliverable, because capture is not an add-on to
# the old body — the old body could not be captured at all, and rewriting it to be capturable
# (batched, fixed-shape) is where most of the launch storm went. This script measures the pair the
# user actually experiences:
#     base = the parent commit, per-request loop, ~150 eager launches PER REQUEST per step
#     new  = this tree, one batched body, replayed from a graph
#
# TWO WORKTREES, ONE IMAGE. The engine is pure Python and hot-mounted at /engine, so the only
# difference is which tree is mounted — hence provenance is asserted by md5'ing the MOUNTED source
# from inside the container, not by an env var.
#
#     BASE_WT=/home/pat/code/minisgl-rdna4-propbase \
#       gpu-lease -n 2 -- bash tools/propose_capture_vs_base.sh
set -uo pipefail

NEW_WT=/home/pat/code/minisgl-rdna4-propose
BASE_WT=${BASE_WT:-/home/pat/code/minisgl-rdna4-propbase}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
export MODEL="${MODEL:-laguna}" SPEC="${SPEC:-dflash}" TP="${TP:-2}" MEM_RATIO="${MEM_RATIO:-0.93}"
NREQ="${NREQ:-1}"
export CONC="${CONC:-$NREQ}" GRAPH_BS="${GRAPH_BS:-8}"
OUT=${OUT:-$NEW_WT/tools/propose_capture_vs_base_${SPEC}_n${NREQ}.txt}
: > "$OUT"

down() { ( cd "$1" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
cleanup() { down "$NEW_WT"; down "$BASE_WT"; docker rm -f $(docker ps -aq --filter name=-serve) >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

wait_ready() { for _ in $(seq 1 400); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

drive() { MAXTOK="$1" NREQ="$NREQ" python3 - <<'PY'
import hashlib, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE="http://localhost:1919"
MODEL=json.loads(urllib.request.urlopen(f"{BASE}/v1/models",timeout=30).read())["data"][0]["id"]
NREQ=int(os.environ["NREQ"]); MT=int(os.environ["MAXTOK"])
PROMPT=("Write a complete, well-commented Python implementation of a persistent red-black tree, "
        "including insertion, deletion, rebalancing, and an in-order iterator. Explain the "
        "invariants as you go.")
def run(i):
    txt = PROMPT if NREQ == 1 else f"{PROMPT} (variant {i})"
    b={"model":MODEL,"messages":[{"role":"user","content":txt}],"max_tokens":MT,
       "temperature":0.0,"seed":1234,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=3600).read())
    return d["usage"]["completion_tokens"], d["choices"][0]["message"]["content"]
t=time.perf_counter()
with ThreadPoolExecutor(NREQ) as ex: res=list(ex.map(run, range(NREQ)))
w=time.perf_counter()-t; n=sum(k for k,_ in res)
print(f"TOKENS {n}"); print(f"WALL {w:.3f}"); print(f"TPS {n/w:.3f}")
print("MD5 " + hashlib.md5(res[0][1].encode()).hexdigest())
PY
}

leg() {  # leg <tag> <worktree> <max_tokens>
  local tag=$1 wt=$2 mt=$3
  echo "=== $tag  MODEL=$MODEL SPEC=$SPEC NREQ=$NREQ ===" | tee -a "$OUT"
  down "$NEW_WT"; down "$BASE_WT"
  ( cd "$wt" && env MINISGL_IMAGE="$IMAGE" MINISGL_SPEC_TIMING=1 \
      docker compose --profile serve up -d >/dev/null 2>&1 )
  if ! wait_ready; then
    echo "  FAILED to become ready" | tee -a "$OUT"
    ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs --tail 60 2>&1 ) \
      | tail -60 | tee -a "$OUT"; down "$wt"; return 1
  fi
  local c; c=$( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
  echo -n "  mounted spec/dflash.py md5: " | tee -a "$OUT"
  docker exec "$c" sh -c "md5sum /engine/python/minisgl/spec/dflash.py | cut -d' ' -f1" | tee -a "$OUT"
  echo -n "  mounted spec/capture.py: " | tee -a "$OUT"
  docker exec "$c" sh -c "md5sum /engine/python/minisgl/spec/capture.py 2>/dev/null | cut -d' ' -f1 || echo ABSENT" | tee -a "$OUT"
  echo "  tree sha: $( cd "$wt" && git rev-parse --short HEAD )" | tee -a "$OUT"
  drive "$mt" 2>&1 | tee -a "$OUT"
  ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -o '\[spec-timing\].*' | tee -a "$OUT"
  down "$wt"; echo | tee -a "$OUT"
}

leg base "$BASE_WT" "${TIME_TOK:-1024}"
leg new  "$NEW_WT"  "${TIME_TOK:-1024}"
echo "results -> $OUT"
