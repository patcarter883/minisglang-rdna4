#!/usr/bin/env bash
# DELIVERABLE 3 GATE: is the adaptive verify width LOSSLESS?
#
# The claim under test is that acceptance statistics choose the WIDTH and never the OUTPUT. The
# strongest observable form of that is: for a GREEDY request, the emitted text must not depend on
# the verify width — and, stronger still, must equal PLAIN DECODE (spec off), because verify gates
# every emitted token.
#
# So this drives ONE greedy prompt at several generation lengths against FOUR configurations:
#     plain   SPEC=none                      — the reference. No drafts at all.
#     base    parent c172f649, width 16      — the shipped fixed width (qlen 17)
#     fixed   this tree, width pinned 15     — multi-width capture, always widest (qlen 16)
#     adapt   this tree, shipped             — width chosen per step from [3, 7, 15]
# and md5s the completion per (config, length).
#
# WHY SEVERAL LENGTHS. Two greedy runs that agree for N tokens can still diverge later: once ONE
# token differs, the trajectories decouple and every later comparison is meaningless. So a single
# long-generation md5 mismatch proves nothing about the mechanism. Short lengths localize where (and
# whether) divergence begins. `adapt` is also run TWICE at the end — a self-vs-self CONTROL, because
# a gate is only informative if the noise floor under it is zero.
#
#     gpu-lease -n 2 -- bash tools/verify_width_lossless.sh
set -uo pipefail

NEW_WT=/home/pat/code/minisgl-rdna4-propose
BASE_WT=${BASE_WT:-/home/pat/code/minisgl-rdna4-propbase2}
FIX_WT=${FIX_WT:-/home/pat/code/minisgl-rdna4-propfix}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
export MODEL="${MODEL:-laguna}" TP="${TP:-2}" MEM_RATIO="${MEM_RATIO:-0.93}"
export CONC=1 GRAPH_BS=8
OUT=${OUT:-$NEW_WT/tools/verify_width_lossless.txt}
: > "$OUT"

down() { ( cd "$1" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
alldown() { down "$NEW_WT"; down "$BASE_WT"; down "$FIX_WT"; }
trap alldown EXIT INT TERM

wait_ready() { for _ in $(seq 1 400); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

drive() { python3 - <<'PY'
import hashlib, json, urllib.request
BASE="http://localhost:1919"
MODEL=json.loads(urllib.request.urlopen(f"{BASE}/v1/models",timeout=30).read())["data"][0]["id"]
PROMPT="Write a Python function that reverses a singly linked list in place, then explain how it works."
for MT in (32, 64, 128, 256):
    b={"model":MODEL,"messages":[{"role":"user","content":PROMPT}],"max_tokens":MT,
       "temperature":0.0,"seed":1234,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=1200).read())
    t=d["choices"][0]["message"]["content"] or ""
    print(f"  MT={MT:4d} n={d['usage']['completion_tokens']:4d} "
          f"finish={d['choices'][0]['finish_reason']:9s} md5={hashlib.md5(t.encode()).hexdigest()}")
    print(f"        head: {t[:90]!r}")
PY
}

leg() {  # leg <tag> <worktree> <spec>
  local tag=$1 wt=$2 spec=$3
  echo "=== $tag (SPEC=$spec) ===" | tee -a "$OUT"
  alldown
  ( cd "$wt" && env MINISGL_IMAGE="$IMAGE" SPEC="$spec" docker compose --profile serve up -d >/dev/null 2>&1 )
  if ! wait_ready; then echo "  FAILED to become ready" | tee -a "$OUT"; down "$wt"; return 1; fi
  local c; c=$( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
  echo -n "  scheduler.py md5: " | tee -a "$OUT"
  docker exec "$c" sh -c "md5sum /engine/python/minisgl/scheduler/scheduler.py | cut -d' ' -f1" | tee -a "$OUT"
  drive 2>&1 | tee -a "$OUT"
  ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -aE "ADAPTIVE verify width|\[spec\] mean" | sed 's/^[^ ]* *| *//' | tail -3 | tee -a "$OUT"
  down "$wt"; echo "" | tee -a "$OUT"
}

leg plain  "$NEW_WT"  none
leg base   "$BASE_WT" dflash
leg fixed  "$FIX_WT"  dflash
leg adapt  "$NEW_WT"  dflash
leg adapt2 "$NEW_WT"  dflash   # self-vs-self CONTROL: the noise floor under the gate
echo "== LOSSLESS GATE COMPLETE ==" | tee -a "$OUT"
