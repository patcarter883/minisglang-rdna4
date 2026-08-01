#!/usr/bin/env bash
# Focused capture of the NEW-tree failure on long x NREQ=8. The matrix harness's failure dump came
# back EMPTY (it shells out to `docker compose logs`, which returns nothing once the crash watchdog
# in server/launch.py has torn the project down), so this leg keeps the container name and reads
# `docker logs` directly, and it never calls `down` before the log is saved.
#
#   gpu-lease -n 2 -- bash tools/perf_crash_capture.sh
set -uo pipefail
cd "$(dirname "$0")/.."
# A per-session /tmp scratchpad is TMPFS: the first version of this script defaulted SCRATCH into
# one, and a reboot took the crash dump AND the long-prompt fixture with it, so the failure this
# script exists to capture had to be re-run from zero. Default somewhere that survives a reboot.
SCRATCH=${SCRATCH:-$HOME/.cache/minisgl-perf}
mkdir -p "$SCRATCH"
OUT="$SCRATCH/laguna_crash_detail.txt"
RAW="$SCRATCH/laguna_crash_rawlog.txt"
LONGCODE=${LONGCODE:-$SCRATCH/longcode.txt}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
WT=/home/pat/code/minisgl-rdna4-propose
: > "$OUT"

# Regenerate the >=3k-token "real code" prompt class from the repo module it describes, so the
# fixture is derived rather than stored. Byte-identical boot-to-boot; no orphaned tmpfs input.
[ -s "$LONGCODE" ] || head -c 12288 "$WT/python/minisgl/engine/graph.py" > "$LONGCODE"
echo "fixture: $LONGCODE ($(wc -c < "$LONGCODE") bytes)" | tee -a "$OUT"

cleanup() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
trap cleanup EXIT INT TERM

export MODEL=laguna SPEC=dflash TP=2 CONC=8 GRAPH_BS=8 MINISGL_SPEC_DEBUG=1
( cd "$WT" && env MINISGL_IMAGE="$IMAGE" docker compose --profile serve up -d >/dev/null 2>&1 )
# Resolve the container BEFORE the readiness wait, not after: the suspected failure is an OOM at
# propose capture, which is a BOOT crash. `ps -q serve` on a torn-down project returns empty, so
# resolving late would hand `docker logs` an empty id and silently lose the very traceback wanted.
C=$( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -qa serve )
echo "container=$C" | tee -a "$OUT"

READY=0
for _ in $(seq 1 400); do
  curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && { READY=1; break; }
  sleep 2
done
echo "ready=$READY" | tee -a "$OUT"

# NOTE the braces: a heredoc written after a PIPELINE binds to the LAST command in it, so
# `python3 - | tee <<'PY'` feeds the script to TEE and hands python an empty stdin. Group first.
if [ "$READY" = 1 ]; then
{ LONGCODE="$LONGCODE" python3 - <<'PY'
import json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE = "http://localhost:1919"
M = json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=60).read())["data"][0]["id"]
code = open(os.environ["LONGCODE"]).read()
P = ("Here is a module from a CUDA-graph-capturing LLM inference engine.\n\n```python\n" + code +
     "\n```\n\nReview this code. Explain what the graph capture path does, identify the invariants a "
     "caller must uphold, and point out anything that would break if the batch size or sequence "
     "length changed between capture and replay.")
def run(i):
    b = {"model": M, "messages": [{"role": "user", "content": f"{P}\n\n(Answer variant {i}: focus on point {i+1}.)"}],
         "max_tokens": 1600, "temperature": 0.0, "top_p": 1.0, "seed": 1234, "stream": False}
    r = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(b).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=3600).read())
        return f"req{i}: OK {d['usage']['completion_tokens']} tok"
    except Exception as e:                                                   # noqa: BLE001
        return f"req{i}: FAILED {type(e).__name__}: {e}"
with ThreadPoolExecutor(8) as ex:
    for line in ex.map(run, range(8)):
        print(line)
PY
} 2>&1 | tee -a "$OUT"
else
  echo "SERVE NEVER BECAME READY — the failure is at BOOT; skipping the load and dumping the log." \
    | tee -a "$OUT"
fi

echo "=== RAW ENGINE LOG (docker logs, container kept) ===" | tee -a "$OUT"
docker logs "$C" > "$RAW" 2>&1
echo "raw log: $RAW ($(wc -l < "$RAW") lines)" | tee -a "$OUT"
grep -aiE "error|assert|traceback|out of memory|died unexpectedly|Runtime|Exception|File \"" "$RAW" \
  | tail -40 | tee -a "$OUT"
