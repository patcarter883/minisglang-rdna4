#!/usr/bin/env bash
# FIX GATE 2 — DDTree could not be run AT ALL, so nothing about it was verifiable.
#
# Review finding: `MINISGL_DFLASH_DDTREE=1` on Laguna dies at BOOT with
#   AssertionError: SWA metadata missing (is_swa_hybrid not wired?)   (attention/rdna4.py)
# raised out of `capture_ddtree_verify_graphs`, because `_ddtree_verify_metadata_static`
# (attention/hip.py) populates no swa_* fields while the K+1 verify capture does. Laguna is the ONLY
# model whose DFlash drafter has a capturable propose, so this made the claim "DDTree's F1 propose is
# DFlash's, so it is captured by this work" unverifiable — and it was not disclosed. The crash is
# PRE-EXISTING (it reproduces on the phase parent), but "pre-existing" is not "unverifiable is fine".
#
# FIX: `capture_ddtree_verify_graphs` now DECLINES on an SWA-hybrid model with a warning instead of
# asserting. The tree-verify runs eager there (the scheduler's own `prepare_metadata` does build the
# SWA fields), which is slower but runnable — and therefore gateable.
#
# So: PRE-FIX must CRASH; FIXED must serve, must show DFlash PROPOSE capture engaged UNDER DDTree,
# and must be reproducible boot-to-boot (the noise floor without which nothing else means anything).
#
#   gpu-lease -n 2 -- bash tools/fixgate_ddtree.sh
set -uo pipefail

NEW_WT=${NEW_WT:-/home/pat/code/minisgl-rdna4-propose}
OLD_WT=${OLD_WT:-/home/pat/code/minisgl-rdna4-prefix753}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
OUT=${OUT:-$NEW_WT/tools/fixgate_ddtree.txt}
export MODEL="${MODEL:-laguna}" TP=2 MEM_RATIO="${MEM_RATIO:-0.93}" CONC=1 GRAPH_BS=8
export MINISGL_SPEC_TIMING=1 MINISGL_KV_FP8=0
: > "$OUT"

down() { ( cd "$1" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
alldown() { down "$NEW_WT"; down "$OLD_WT"; }
trap alldown EXIT INT TERM

wait_ready() { for _ in $(seq 1 240); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0
    sleep 2; done; return 1; }

drive() { python3 - <<'PY'
import hashlib, json, urllib.request
BASE = "http://localhost:1919"
PROMPT = "Write a Python function that reverses a singly linked list in place, then explain it."
try:
    M = json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=30).read())["data"][0]["id"]
except Exception as e:                                                       # noqa: BLE001
    print(f"  NO SERVER: {type(e).__name__}: {e}"); raise SystemExit
for MT in (64, 128, 256):
    for rep in (1, 2):          # in-boot repeat = the determinism control
        b = {"model": M, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": MT,
             "temperature": 0.0, "seed": 1234, "stream": False}
        r = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(b).encode(),
                                   headers={"Content-Type": "application/json"})
        try:
            d = json.loads(urllib.request.urlopen(r, timeout=1200).read())
        except Exception as e:                                               # noqa: BLE001
            print(f"  MT={MT} rep={rep} FAILED: {type(e).__name__}: {e}"); continue
        t = d["choices"][0]["message"]["content"] or ""
        print(f"  MT={MT:4d} rep={rep} n={d['usage']['completion_tokens']:4d} "
              f"md5={hashlib.md5(t.encode()).hexdigest()}  head={t[:56]!r}")
PY
}

leg() {  # leg <tag> <worktree> <extra env...>
  local tag=$1 wt=$2; shift 2
  echo "=== $tag  ($wt) ===" | tee -a "$OUT"
  alldown
  ( cd "$wt" && env MINISGL_IMAGE="$IMAGE" SPEC=dflash "$@" \
      docker compose --profile serve up -d >/dev/null 2>&1 )
  if wait_ready; then
    local c; c=$( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
    echo -n "  PROVENANCE engine/graph.py md5: " | tee -a "$OUT"
    docker exec "$c" sh -c "md5sum /engine/python/minisgl/engine/graph.py | cut -d' ' -f1" \
      2>/dev/null | tee -a "$OUT"
    echo -n "  PROVENANCE env DDTREE: " | tee -a "$OUT"
    docker exec "$c" sh -c "tr '\0' '\n' < /proc/1/environ | grep -E 'DDTREE' | tr '\n' ' '" \
      2>/dev/null | tee -a "$OUT"; echo "" | tee -a "$OUT"
    drive 2>&1 | tee -a "$OUT"
  else
    echo "  NEVER BECAME READY" | tee -a "$OUT"
  fi
  echo "  --- boot/serve evidence ---" | tee -a "$OUT"
  ( cd "$wt" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -aE "AssertionError|SWA metadata|ddtree-verify|DDTree|died unexpectedly|PROPOSE graphs|propose-graph|ALWAYS-EAGER" \
    | sed 's/^[^ ]* *| *//' | tail -10 | tee -a "$OUT"
  down "$wt"; echo "" | tee -a "$OUT"
}

leg "PRE-FIX  ddtree"   "$OLD_WT" MINISGL_DFLASH_DDTREE=1
leg "FIXED    ddtree"   "$NEW_WT" MINISGL_DFLASH_DDTREE=1
leg "FIXED    ddtree#2" "$NEW_WT" MINISGL_DFLASH_DDTREE=1   # cross-boot control
echo "== FIX GATE 2 COMPLETE ==" | tee -a "$OUT"
