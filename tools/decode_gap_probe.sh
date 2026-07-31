#!/usr/bin/env bash
# decode_gap_probe.sh — decompose the NON-GPU portion of a decode step.
#
# WHY: measured on Qwen3.6-35B TP=2 M=1 (2026-07-31), GPU-busy is 8.83 ms/step while client TPOT is
# 11.12 ms -> 2.30 ms (20.7%) of every step is not GPU work. That is larger than every addressable
# kernel except the dense GEMV, which is already at its bandwidth floor. Per-kernel optimisation
# cannot reach it, so it has to be split into its parts before anyone can act on it.
#
# The 2.30 ms is an UPPER BOUND on engine idle: TPOT is measured client-side over HTTP, so it also
# contains tokenizer/detokenizer, streaming and socket time. Separating that is the first job here.
#
# THREE MEASUREMENTS, one serve boot:
#   1. MINISGL_GRAPH_TIMING=1  -> the engine's own per-step host split: copy_from (I/O staging),
#      prepare_for_replay (attention + recurrent metadata rebuild — the eager work NOT in the graph),
#      and g.replay() (the launch itself). Averages printed every 100 replays on rank0.
#   2. py-spy on the LIVE scheduler process -> where Python wall actually goes, sampled, no restart
#      and no instrumentation. This is the piece we simply could not see before py-spy was installed.
#   3. server-side vs client-side step time -> how much of the 2.30 ms is HTTP/tokenizer, not engine.
#
# Run through the runner under a lease (TP=2 needs both cards):
#   gpu-lease -n 2 -- env INNER=/engine/tools/decode_gap_probe.sh MODEL=... TP=2 \
#     bash tools/run_bench_window.sh
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate

MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
TP="${TP:-2}"
PORT="${PORT:-21963}"
GRAPH="${GRAPH:-16}"
SECS="${SECS:-25}"                 # how long to hold decode load while sampling
OUT=/engine/tools/tp2_results
mkdir -p "$OUT"
LOG="$OUT/gap_probe.server.log"

echo "[probe] launching serve with MINISGL_GRAPH_TIMING=1 (host-cost split per step)"
local_pynccl=""; [ "$TP" -gt 1 ] && local_pynccl="--disable-pynccl"
setsid env MINISGL_GRAPH_TIMING=1 python -m minisgl \
  --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --host 0.0.0.0 --graph "$GRAPH" \
  --attention-backend hip $local_pynccl --memory-ratio 0.82 --max-running-requests 4 \
  > "$LOG" 2>&1 &
SRV=$!
stop() { [ -n "${SRV:-}" ] || return 0; kill -TERM -- "-$SRV" 2>/dev/null
         for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
         kill -KILL -- "-$SRV" 2>/dev/null; }
trap stop EXIT

for _ in $(seq 1 400); do
  python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "[probe] serve DIED:"; tail -30 "$LOG"; exit 1; }
  sleep 3
done
echo "[probe] ready"

# The scheduler is the process that actually runs the decode loop; the parent is a supervisor and will
# sit at ~0% (this is exactly the mistake that made a busy profiler look hung earlier today).
SCHED=$(pgrep -f 'minisgl' | while read -r p; do
          printf '%s %s\n' "$(ps -o %cpu= -p "$p" 2>/dev/null | tr -d ' ')" "$p"; done \
        | sort -rn | head -1 | awk '{print $2}')
echo "[probe] busiest minisgl pid = ${SCHED:-none}"

# Drive steady decode load in the background so there is something to sample.
python - "$PORT" "$SECS" <<'PY' &
import json, sys, time, urllib.request
port, secs = sys.argv[1], float(sys.argv[2])
end = time.time() + secs
n = 0
while time.time() < end:
    body = json.dumps({"model": "m", "temperature": 0.0, "max_tokens": 128,
                       "messages": [{"role": "user", "content": "Count slowly from one to forty."}]}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(r, timeout=120).read(); n += 1
    except Exception:
        break
print(f"[probe] drove {n} decode requests", flush=True)
PY
LOADER=$!

sleep 4   # let the load settle into steady-state decode before sampling
if [ -n "${SCHED:-}" ]; then
  echo "[probe] py-spy sampling the scheduler for $((SECS-8))s ..."
  py-spy record --pid "$SCHED" --duration $((SECS-8)) --rate 250 --subprocesses \
    --format speedscope --output "$OUT/gap_pyspy.speedscope.json" 2>&1 | tail -3
  echo "[probe] py-spy TOP (native+python, self time):"
  py-spy dump --pid "$SCHED" --locals 2>/dev/null | head -25
fi
wait $LOADER 2>/dev/null

echo
echo "=== ENGINE HOST-COST SPLIT (MINISGL_GRAPH_TIMING) ==="
grep -iE 'timing|copy_from|prepare_for_replay|replay' "$LOG" | tail -12
echo
echo "=== artifacts ==="
echo "  $OUT/gap_pyspy.speedscope.json   (open at https://www.speedscope.app)"
echo "  $LOG"
stop
