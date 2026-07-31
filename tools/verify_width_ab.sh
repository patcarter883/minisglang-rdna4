#!/usr/bin/env bash
# DELIVERABLE 3 A/B: adaptive verify width vs the two fixed widths it replaces.
#
# THREE LEGS, ONE IMAGE, THREE WORKTREES. The engine is pure Python and hot-mounted at /engine, so
# the only difference between legs is which tree is mounted — provenance is therefore asserted by
# md5'ing the MOUNTED source from inside the container, never by an env var.
#
#   base   parent commit c172f649. Verify width is FIXED at --spec-num-draft = 16. DFlash caps its
#          block at 15 drafts, so the partial-K padding fires every step and pads 15 -> 16, i.e.
#          qlen 17 — one row past the M<=16 decode-kernel boundary.
#   fixed  THIS tree with `_adaptive_width_ok` pinned False (a one-line A/B-only patch that lives
#          ONLY in the propfix worktree, never in the branch). Multi-width graphs are captured, but
#          the step always uses the widest, 15 -> qlen 16. Isolates the M<=16 clamp alone.
#   adapt  THIS tree, shipped. The step sizes the block from the per-request acceptance EMA and
#          lands on one of the captured widths [3, 7, 15].
#
# base -> fixed is the CLIFF win. fixed -> adapt is the ADAPTIVE win. Reporting only base -> adapt
# would let the cliff fix take credit for the controller (and vice versa).
#
# Measured at NREQ=1 AND NREQ=8: a CONC=1-only A/B on this box has already misread a real +20% lever
# as flat, and the M<=16 cliff is by construction a bs=1 effect (M = padded_bs*(width+1)), so the
# two concurrencies are expected to disagree and that disagreement is the finding, not noise.
#
# tok/s is usage.completion_tokens on a NON-streaming request. Never count SSE chunks: under spec
# ONE chunk carries a whole accepted block.
#
#     NREQ=1 gpu-lease -n 2 -- bash tools/verify_width_ab.sh
set -uo pipefail

NEW_WT=/home/pat/code/minisgl-rdna4-propose
BASE_WT=${BASE_WT:-/home/pat/code/minisgl-rdna4-propbase2}
FIX_WT=${FIX_WT:-/home/pat/code/minisgl-rdna4-propfix}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
export MODEL="${MODEL:-laguna}" SPEC="${SPEC:-dflash}" TP="${TP:-2}" MEM_RATIO="${MEM_RATIO:-0.93}"
NREQ="${NREQ:-1}"
export CONC="${CONC:-$NREQ}" GRAPH_BS="${GRAPH_BS:-8}"
MAXTOK="${MAXTOK:-1024}"
OUT=${OUT:-$NEW_WT/tools/verify_width_ab_n${NREQ}.txt}
: > "$OUT"

down() { ( cd "$1" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
alldown() { down "$NEW_WT"; down "$BASE_WT"; down "$FIX_WT"; }
trap alldown EXIT INT TERM

wait_ready() { for _ in $(seq 1 400); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

drive() { MAXTOK="$MAXTOK" NREQ="$NREQ" python3 - <<'PY'
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
print("HEAD " + res[0][1][:110].replace("\n", " "))
PY
}

leg() {  # leg <tag> <worktree>
  local tag=$1 wt=$2
  echo "=== $tag  MODEL=$MODEL SPEC=$SPEC NREQ=$NREQ MAXTOK=$MAXTOK ===" | tee -a "$OUT"
  alldown
  ( cd "$wt" && env MINISGL_IMAGE="$IMAGE" MINISGL_SPEC_TIMING=1 \
      docker compose --profile serve up -d >/dev/null 2>&1 )
  if ! wait_ready; then
    echo "  FAILED to become ready" | tee -a "$OUT"
    ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs --tail 60 2>&1 ) \
      | tail -60 | tee -a "$OUT"; down "$wt"; return 1
  fi
  local c; c=$( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
  # PROVENANCE: which BUILD is this leg actually running? md5 the mounted source in the container.
  echo -n "  mounted spec/width.py:      " | tee -a "$OUT"
  docker exec "$c" sh -c "md5sum /engine/python/minisgl/spec/width.py 2>/dev/null | cut -d' ' -f1 || echo ABSENT" | tee -a "$OUT"
  echo -n "  mounted scheduler.py md5:   " | tee -a "$OUT"
  docker exec "$c" sh -c "md5sum /engine/python/minisgl/scheduler/scheduler.py | cut -d' ' -f1" | tee -a "$OUT"
  echo "  tree sha: $( cd "$wt" && git rev-parse --short HEAD )$( cd "$wt" && git diff --quiet || echo ' +AB-PATCH' )" | tee -a "$OUT"
  drive 2>&1 | tee -a "$OUT"
  ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -aE "ADAPTIVE verify width|Capturing spec-verify|\[spec-timing\]|\[spec\] mean" \
    | sed 's/^[^ ]* *| *//' | tail -14 | tee -a "$OUT"
  down "$wt"
  echo "" | tee -a "$OUT"
}

leg base  "$BASE_WT"
leg fixed "$FIX_WT"
leg adapt "$NEW_WT"
echo "== A/B COMPLETE ==" | tee -a "$OUT"
