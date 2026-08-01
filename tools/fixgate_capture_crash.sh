#!/usr/bin/env bash
# FIX GATE 1 — the phase-introduced crash in the ENGAGEMENT-REPORTING path itself, and the
# "silent eager that reads green" it sat next to.
#
# DEFECT (found by review, reproduced here): `CapturableProposer.propose_capture_stats` dereferenced
# `self._pc_replays`, which only exists if `init_propose_capture_state` ran. DFlash deliberately
# skips that for any drafter with no bounded prefix (non-causal z-lab / CCA) and for
# MINISGL_DFLASH_PERSIST_KV=0. With MINISGL_SPEC_TIMING=1 — both compose-forwarded — the scheduler
# called it every 50 steps and the worker died mid-serve with
#   AttributeError: 'DFlashProposer' object has no attribute '_pc_replays'
#
# SECOND DEFECT, same call: a proposer with NO captured propose reported `replay=0 eager=0`, i.e.
# 100% eager rendered as ZERO eager — the exact green-number-over-a-silent-fallback this capture
# work exists to prevent.
#
# So this drives the SAME config on the PRE-FIX tree (must CRASH) and the FIXED tree (must serve,
# and must SAY it is always-eager and why). A fix gate that cannot show the failure is not a gate.
#
#   gpu-lease -n 2 -- bash tools/fixgate_capture_crash.sh
set -uo pipefail

NEW_WT=${NEW_WT:-/home/pat/code/minisgl-rdna4-propose}
OLD_WT=${OLD_WT:-/home/pat/code/minisgl-rdna4-prefix753}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
OUT=${OUT:-$NEW_WT/tools/fixgate_capture_crash.txt}
export MODEL="${MODEL:-laguna}" TP=2 MEM_RATIO="${MEM_RATIO:-0.93}" CONC=1 GRAPH_BS=8
export MINISGL_SPEC_TIMING=1 MINISGL_KV_FP8=0
: > "$OUT"

down() { ( cd "$1" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
alldown() { down "$NEW_WT"; down "$OLD_WT"; }
trap alldown EXIT INT TERM

wait_ready() { for _ in $(seq 1 300); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0
    sleep 2; done; return 1; }

drive() {  # enough tokens to cross the 50-step timing boundary several times
  python3 - <<'PY'
import json, urllib.request
BASE = "http://localhost:1919"
try:
    M = json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=30).read())["data"][0]["id"]
    b = {"model": M, "messages": [{"role": "user", "content":
         "Write a Python function that reverses a singly linked list in place, then explain it."}],
         "max_tokens": 600, "temperature": 0.0, "seed": 1234, "stream": False}
    r = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=1200).read())
    print(f"  REQUEST OK  tokens={d['usage']['completion_tokens']} "
          f"head={ (d['choices'][0]['message']['content'] or '')[:60]!r}")
except Exception as e:                                                       # noqa: BLE001
    print(f"  REQUEST FAILED: {type(e).__name__}: {e}")
PY
}

leg() {  # leg <tag> <worktree> <extra env assignments...>
  local tag=$1 wt=$2; shift 2
  echo "=== $tag  ($wt) ===" | tee -a "$OUT"
  alldown
  ( cd "$wt" && env MINISGL_IMAGE="$IMAGE" SPEC=dflash "$@" \
      docker compose --profile serve up -d >/dev/null 2>&1 )
  if ! wait_ready; then echo "  NEVER BECAME READY" | tee -a "$OUT"; fi
  local c; c=$( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
  if [ -n "$c" ]; then
    echo -n "  PROVENANCE spec/capture.py md5: " | tee -a "$OUT"
    docker exec "$c" sh -c "md5sum /engine/python/minisgl/spec/capture.py | cut -d' ' -f1" \
      2>/dev/null | tee -a "$OUT"
    echo -n "  PROVENANCE env MINISGL_DFLASH_PERSIST_KV / SPEC_TIMING: " | tee -a "$OUT"
    docker exec "$c" sh -c \
      "tr '\0' '\n' < /proc/1/environ | grep -E 'PERSIST_KV|SPEC_TIMING' | tr '\n' ' '" \
      2>/dev/null | tee -a "$OUT"; echo "" | tee -a "$OUT"
  fi
  drive 2>&1 | tee -a "$OUT"
  echo "  --- scheduler outcome ---" | tee -a "$OUT"
  ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -aE "AttributeError|died unexpectedly|propose-graph|ALWAYS-EAGER|PROPOSE graphs|Traceback" \
    | sed 's/^[^ ]* *| *//' | tail -12 | tee -a "$OUT"
  down "$wt"; echo "" | tee -a "$OUT"
}

# 1. PRE-FIX, PERSIST_KV=0 -> the crash. 2. FIXED, same config -> serves + honest readout.
leg "PRE-FIX  persist_kv=0" "$OLD_WT" MINISGL_DFLASH_PERSIST_KV=0
leg "FIXED    persist_kv=0" "$NEW_WT" MINISGL_DFLASH_PERSIST_KV=0
# 3. FIXED, shipped default -> capture engaged, replay/eager split as before.
leg "FIXED    default"      "$NEW_WT"
echo "== FIX GATE 1 COMPLETE ==" | tee -a "$OUT"
