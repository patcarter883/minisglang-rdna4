#!/usr/bin/env bash
# Which verify-width RUNG actually maximizes throughput?
#
# The adaptive controller maximizes accepted tokens per STEP. Serving cares about tokens per
# MILLISECOND. Those are not the same objective, and the 2026-08-01 long x N=8 run showed them
# pointing in OPPOSITE directions: as the controller climbed off rung 3, emitted/step rose
# 17.9 -> 20.5 while wall time per 50 steps rose 5-6s -> 11s, i.e. ~163 -> ~91 tok/s. Reason: at
# bs=8, rung 3 verifies M = 8*4 = 32 rows and rung 7 verifies M = 8*8 = 64, and past the M<=16
# decode-GEMV boundary the MoE pays expert fanout per row.
#
# So: pin each captured rung in turn (MINISGL_SPEC_VERIFY_WIDTH_PIN) and measure TRUE tok/s at
# bs=1 and bs=8. K is held at 16 throughout, so PROPOSE cost is identical across legs and the only
# variable is verify width. This is what decides whether the controller's censoring bug
# (spec/width.py:285, accepted stored at face value) is worth fixing or is accidentally load-bearing.
#
#   gpu-lease -n 2 -- bash tools/verify_width_rung_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

SCRATCH=${SCRATCH:-$HOME/.cache/minisgl-perf}
mkdir -p "$SCRATCH"
OUT=${OUT:-$SCRATCH/width_rung_sweep.txt}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
WT=/home/pat/code/minisgl-rdna4-propose
RUNGS=${RUNGS:-"3 7 15"}
NREQS=${NREQS:-"1 8"}
MAXTOK=${MAXTOK:-384}
: > "$OUT"

export MODEL="${MODEL:-laguna}" SPEC="${SPEC:-dflash}" SPEC_K="${SPEC_K:-16}" TP=2 \
       CONC="${CONC:-8}" GRAPH_BS="${GRAPH_BS:-8}" MINISGL_SPEC_DEBUG=1
# TEMP: spec must be measured SAMPLED, not greedy — greedy inflates acceptance at LATE draft
# positions and over-recommends K (and, here, width). Default 0.0 keeps the banked Laguna sweep
# comparable; set TEMP=0.8 for the honest serving measurement.
export DRIVE_TEMP="${TEMP:-0.0}"

down() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
trap down EXIT INT TERM

drive() { NREQ="$1" MAXTOK="$MAXTOK" PROMPTCLASS="${PROMPTCLASS:-short}" \
          LONGCODE="${LONGCODE:-$SCRATCH/longcode.txt}" python3 - <<'PY'
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE = "http://localhost:1919"
MODEL = json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=60).read())["data"][0]["id"]
MT, NREQ = int(os.environ["MAXTOK"]), int(os.environ["NREQ"])
# Same prompts as tools/perf_matrix.sh so numbers are comparable with the banked matrix. SHORT is
# well under the 512-token seed bound, so the prompt-seed path engages identically on every leg;
# LONG is over it, so every long leg runs UNSEEDED — which is the shipped behaviour after
# _spec_seed_fits, and is the configuration the row budget has to hold in.
SHORT = ("Write a Python function that reverses a singly linked list in place, then explain how it "
         "works, why it is O(n) time and O(1) space, and what happens on an empty list and on a "
         "single-node list. Then show how you would test it.")
if os.environ.get("PROMPTCLASS") == "long":
    code = open(os.environ["LONGCODE"]).read()
    BASE_PROMPT = ("Here is a module from a CUDA-graph-capturing LLM inference engine.\n\n```python\n"
                   + code + "\n```\n\nReview this code. Explain what the graph capture path does, "
                   "identify the invariants a caller must uphold, and point out anything that would "
                   "break if the batch size or sequence length changed between capture and replay.")
else:
    BASE_PROMPT = SHORT
def run(i):
    # Distinct suffix per request: one shared prefix would measure radix hits, not decode.
    txt = BASE_PROMPT if NREQ == 1 else f"{BASE_PROMPT}\n\n(Answer variant {i}: focus on point {i + 1}.)"
    b = {"model": MODEL, "messages": [{"role": "user", "content": txt}], "max_tokens": MT,
         "temperature": float(os.environ.get("DRIVE_TEMP") or "0.0"),
         "top_p": 1.0, "seed": 1234, "stream": False}
    r = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=3600).read())["usage"]["completion_tokens"]
# Warmup, discarded: a different string so it cannot seed a prefix-cache hit for the timed leg.
b = {"model": MODEL, "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
     "max_tokens": 16, "temperature": 0.0, "seed": 7, "stream": False}
urllib.request.urlopen(urllib.request.Request(
    f"{BASE}/v1/chat/completions", data=json.dumps(b).encode(),
    headers={"Content-Type": "application/json"}), timeout=600).read()
t = time.perf_counter()
with ThreadPoolExecutor(NREQ) as ex:
    res = list(ex.map(run, range(NREQ)))
w = time.perf_counter() - t
n = sum(res)
print(f"    NREQ={NREQ} completion_tokens={n} wall={w:.3f}s TPS={n / w:.2f}")
PY
}

for rung in $RUNGS; do
  echo "=== RUNG $rung (K=16, TP=2, CONC=8) ===" | tee -a "$OUT"
  down
  ( cd "$WT" && env MINISGL_IMAGE="$IMAGE" MINISGL_SPEC_VERIFY_WIDTH_PIN="$rung" \
      docker compose --profile serve up -d >/dev/null 2>&1 )
  C=$( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -qa serve )
  ready=0
  for _ in $(seq 1 400); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && { ready=1; break; }
    sleep 2
  done
  if [ "$ready" != 1 ]; then
    echo "  SERVE NEVER READY at rung $rung — skipping" | tee -a "$OUT"
    docker logs "$C" 2>&1 | grep -aiE "error|assert|Traceback|out of memory" | tail -10 | tee -a "$OUT"
    continue
  fi
  # PROVENANCE: assert from the engine's own log that this leg really ran the rung we asked for.
  # Without this the sweep could compare a build against itself and report a flat result as truth.
  docker logs "$C" 2>&1 | grep -aoE "PINNED to rung [0-9]+" | head -1 \
    | sed 's/^/  provenance: /' | tee -a "$OUT"
  for nq in $NREQS; do drive "$nq" | tee -a "$OUT"; done
  # Per-rung acceptance + realized width histogram, straight from the engine.
  docker logs "$C" 2>&1 | grep -aE "\[spec\] step=|verify-width\[" | tail -2 \
    | sed 's/^/  /' | tee -a "$OUT"
done

down
echo "=== sweep complete: $OUT ===" | tee -a "$OUT"
