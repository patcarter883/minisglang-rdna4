#!/usr/bin/env bash
# Profile the spec step with the ENGINE'S OWN torch-profiler window, then rank kernels by GPU time.
#
# Why not the IntelliKit tools here: kerncap captures via LD_PRELOAD=libkerncap.so, and this engine
# spawns its scheduler under setsid, so the preload never reaches the process that runs propose
# (same reason rpd was rejected — tools/rank_trace_kernels.py:5). metrix wraps rocprofv3, whose
# --pmc path hangs on the first dispatch on gfx1201. Both ARE usable against a standalone
# reproducer; neither reaches a live serve. MINISGL_PROFILE does.
set -uo pipefail
cd "$(dirname "$0")/.."
WT=/home/pat/code/minisgl-rdna4-propose
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
SCRATCH=${SCRATCH:-$HOME/.cache/minisgl-perf}
mkdir -p "$SCRATCH"
OUT=${OUT:-$SCRATCH/propose_profile.txt}
TRACE=propose_trace.json
: > "$OUT"

down() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
trap down EXIT INT TERM
down
rm -f "$WT/$TRACE"

export MODEL=laguna SPEC=dflash SPEC_K=16 TP=2 CONC=8 GRAPH_BS=8 MINISGL_SPEC_DEBUG=1
export MINISGL_PROFILE="/engine/$TRACE" MINISGL_PROFILE_SKIP=40 MINISGL_PROFILE_STEPS=50
( cd "$WT" && env MINISGL_IMAGE="$IMAGE" docker compose --profile serve up -d >/dev/null 2>&1 )
for _ in $(seq 1 400); do
  curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && break; sleep 2; done

# One bs=1 request, long enough to cover SKIP+STEPS=90 spec steps.
python3 - <<'PY' | tee -a "$OUT"
import json, urllib.request
BASE = "http://localhost:1919"
M = json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=60).read())["data"][0]["id"]
P = ("Write a Python function that reverses a singly linked list in place, then explain how it "
     "works, why it is O(n) time and O(1) space, and what happens on an empty list and on a "
     "single-node list. Then show how you would test it.")
b = {"model": M, "messages": [{"role": "user", "content": P}], "max_tokens": 512,
     "temperature": 0.0, "seed": 1234, "stream": False}
r = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(b).encode(),
                           headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(r, timeout=3600).read())
print(f"    driven: {d['usage']['completion_tokens']} completion tokens")
PY

sleep 5
if [ -s "$WT/$TRACE" ]; then
  cp "$WT/$TRACE" "$SCRATCH/$TRACE"
  echo "trace: $SCRATCH/$TRACE ($(du -h "$SCRATCH/$TRACE" | cut -f1))" | tee -a "$OUT"
  python3 "$WT/tools/rank_trace_kernels.py" "$SCRATCH/$TRACE" 2>&1 | tee -a "$OUT"
else
  echo "NO TRACE WRITTEN at $WT/$TRACE" | tee -a "$OUT"
fi
