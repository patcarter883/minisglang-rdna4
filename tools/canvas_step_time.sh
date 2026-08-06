#!/usr/bin/env bash
# canvas_step_time.sh — the DiffusionGemma canvas step time, per arm, by DIFFERENCING the
# `[canvas-timing]` cumulative averages.
#
# `_StepTimer` reports a running mean every 10 steps, so the printed number at n=70 is the average
# over ALL 70 steps and is dominated by the first, cold ones. The step time is the DIFFERENCE of
# consecutive windows: (sum_n - sum_{n-10}) / 10. That is how docs §D11.4's 86.9 ms was taken and it
# is the only way these numbers compare to it.
#
# ARMS (each is one full boot):
#   fixed     the candidate as it will serve — MINISGL_ATTN_MAX_SPLITS UNSET
#   nosplit   same tree, split-K forced off — isolates what split-K costs from what the fix costs
#
# On the BASE tree, `fixed` cannot be run: the bit-exactness gate raises on the divergence, so the
# only base arm that completes a request is `nosplit`. That is the 86.9 ms certified-graph baseline.
#
#   gpu-lease -n 2 --timeout 7200 -- bash tools/canvas_step_time.sh
set -uo pipefail
WT=${WT:-/home/pat/code/minisgl-rdna4-splitctx}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:splitctx}
DG_MODEL=${DG_MODEL:-cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4}
SCRATCH=${SCRATCH:-$HOME/.cache/minisgl-perf}
ARMS=${ARMS:-fixed,nosplit}
MAXTOK=${MAXTOK:-1400}          # ~70+ canvas steps, enough for six differenced 10-step windows
CONC=${CONC:-1}
PORT=${PORT:-$((1920 + RANDOM % 900))}
mkdir -p "$SCRATCH"
RUN_ID=${RUN_ID:-$$$(od -An -N2 -tu2 </dev/urandom | tr -dc "0-9")}
OUT=${OUT:-$SCRATCH/canvas_step_time-$RUN_ID.txt}
: > "$OUT"
export COMPOSE_PROJECT_NAME="minisglst$RUN_ID"
export MINISGL_HOST_PORT="$PORT"

say() { echo "$@" | tee -a "$OUT"; }
say "=== canvas_step_time run=$RUN_ID image=$IMAGE wt=$WT"
say "=== engine=$(git -C "$WT" rev-parse --short HEAD) dirty=$(git -C "$WT" status --porcelain | wc -l) arms=$ARMS port=$PORT maxtok=$MAXTOK"

YMLF="$WT/docker-compose.st-$RUN_ID.yml"
write_yml() {
  { echo "services:"
    echo "  serve:"
    echo "    container_name: minisglst-$RUN_ID"
    echo "    command: [\"exec /engine/tools/serve.sh\"]"
    echo "    environment:"
    echo "      MINISGL_SWA_RADIX: \"0\""
    echo "      MINISGL_CANVAS_TIMING: \"1\""
    for kv in "$@"; do echo "      $kv"; done
  } > "$YMLF"
}
DC() { ( cd "$WT" && env MINISGL_IMAGE="$IMAGE" docker compose -f docker-compose.yml -f "$YMLF" --profile serve "$@" ); }
down() { DC down >/dev/null 2>&1; }
trap 'down; rm -f "$YMLF"' EXIT INT TERM

wait_ready() {
  local i c; c="minisglst-$RUN_ID"
  for i in $(seq 1 300); do
    if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" != "true" ]; then
      say "!! container not running after ~$((i*3))s"; docker logs "$c" 2>&1 | tail -30 | tee -a "$OUT"; return 1
    fi
    if docker logs "$c" 2>&1 | grep -qaE "Traceback \(most recent call last\)|torch.OutOfMemoryError"; then
      say "!! traceback in the log"; docker logs "$c" 2>&1 | grep -aA25 "Traceback" | tail -50 | tee -a "$OUT"; return 1
    fi
    curl -s --max-time 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { say "== ready after ~$((i*3))s"; return 0; }
    sleep 3
  done
  say "!! readiness timeout"; docker logs "$c" 2>&1 | tail -30 | tee -a "$OUT"; return 1
}

drive() {
  python3 - "$MAXTOK" "$PORT" <<'PY' 2>&1
import json, sys, time, urllib.request
body = json.dumps({"model": "m", "max_tokens": int(sys.argv[1]), "temperature": 1.0, "stream": False,
                   "messages": [{"role": "user", "content":
                                 "Write a detailed essay about the history of oceanography."}]}).encode()
rq = urllib.request.Request(f"http://localhost:{sys.argv[2]}/v1/chat/completions", body,
                            {"Content-Type": "application/json"})
t0 = time.perf_counter()
try:
    r = json.load(urllib.request.urlopen(rq, timeout=1800))
except Exception as e:
    print(f"req: FAILED {type(e).__name__}: {e}"); sys.exit(0)
ct = r.get("usage", {}).get("completion_tokens", 0)
dt = time.perf_counter() - t0
print(f"req: {ct} tok in {dt:.2f}s ({ct/dt:.1f} tok/s)")
PY
}

export MODEL="$DG_MODEL" SPEC=none TP=2 CONC="$CONC" GRAPH_BS="$CONC" EXTRA_ARGS=""

run_arm() {
  local arm="$1"; shift
  say ""; say "############ ARM $arm  ($*)"
  write_yml "$@"
  down
  if ! DC up -d >"$SCRATCH/up-st-$RUN_ID.log" 2>&1; then
    say "!! compose up FAILED"; tail -20 "$SCRATCH/up-st-$RUN_ID.log" | tee -a "$OUT"; return 1
  fi
  if wait_ready; then drive | tee -a "$OUT"; sleep 2; fi
  # The gate is the reason an arm may have produced no steps at all — surface it before the numbers.
  say "--- gate"
  docker logs "minisglst-$RUN_ID" 2>&1 | grep -aoE "graph vs eager\[matched regime\] max\|delta\|=[^ ]*( BIT-IDENTICAL)?" | head -2 | tee -a "$OUT"
  docker logs "minisglst-$RUN_ID" 2>&1 | grep -a "captured canvas step is NOT" | head -2 | tee -a "$OUT"
  say "--- differenced 10-step windows"
  docker logs "minisglst-$RUN_ID" 2>&1 | grep -a "\[canvas-timing\]" \
    | python3 "$WT/tools/canvas_timing_windows.py" | tee -a "$OUT"
  down
}

[[ ",$ARMS," == *",fixed,"*   ]] && run_arm fixed
[[ ",$ARMS," == *",nosplit,"* ]] && run_arm nosplit "MINISGL_ATTN_MAX_SPLITS: \"1\""

say ""; say "=== done. out=$OUT"
