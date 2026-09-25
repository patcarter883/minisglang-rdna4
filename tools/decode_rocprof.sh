#!/usr/bin/env bash
# decode_rocprof.sh — decompose a bs=1 AUTOREGRESSIVE decode step (SPEC=none) by GPU kernel.
#
# The decode sibling of canvas_rocprof.sh, with the same two legs off one boot config:
#   leg `base`  — no profiler: the served single-stream tok/s (sampled, temp 0.7), i.e. the TRUE step
#                 time every share below is a share of, plus the [hip-engage] ledger.
#   leg `trace` — rocprofv3 --kernel-trace --marker-trace --selected-regions, windowed on the plain
#                 decode loop's ROCTx ranges (normal_step#N / overlap_step#N). rocprofv3 inflates
#                 small kernels and the inter-kernel gaps, so the trace gives SHARES, not the wall.
# Same safety rules as canvas_rocprof.sh: a unique container/project/port per run, and the trace is
# written only on a normal interpreter exit, so the engine is bounded by MINISGL_EXIT_AFTER_STEPS and
# never signalled. Needs an image with a working rocprofv3 (Dockerfile.prof).
#
#   gpu-lease -n 2 -- bash tools/decode_rocprof.sh
#   python3 tools/canvas_trace_report.py <trace dir> --steps normal_step#,overlap_step#
set -uo pipefail
WT=${WT:-$(cd "$(dirname "$0")/.." && pwd)}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean-prof}
MODEL_ID=${MODEL_ID:-cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4}
SCRATCH=${SCRATCH:-$HOME/.cache/minisgl-perf}
RTX_SKIP=${RTX_SKIP:-100}    # past prefill + the first decode steps
RTX_STEPS=${RTX_STEPS:-60}
STEPS=${STEPS:-200}          # scheduler-loop iterations before the engine returns (trace leg)
MAXTOK=${MAXTOK:-512}
LEGS=${LEGS:-base,trace}
# TRACE_MODE=windowed (default): --marker-trace --selected-regions, collection gated on the ROCTx step
# window. MEASURED to distort the timeline ~7x (85 ms traced tokens vs a 12 ms real one — the host
# stalls between dispatches), so kernel DURATIONS are usable but gaps and collective waits are not.
# TRACE_MODE=full: --kernel-trace only over the whole process (model load and capture included —
# a big CSV), which keeps the real cadence; analyse the decode tail by the sampler's once-per-token
# kernel. This is also how a vLLM serve (no ROCTx markers) has to be traced, so use it for A/Bs.
TRACE_MODE=${TRACE_MODE:-windowed}
mkdir -p "$SCRATCH"
RUN_ID=${RUN_ID:-$$$(od -An -N2 -tu2 </dev/urandom | tr -dc "0-9")}
OUT=${OUT:-$SCRATCH/decode_rocprof-$RUN_ID.txt}
RPDIR=decout-$RUN_ID
: > "$OUT"
rm -rf "${WT:?}/$RPDIR"; mkdir -p "$WT/$RPDIR"
export COMPOSE_PROJECT_NAME="minisgldec$RUN_ID"
PORT=${PORT:-$((1920 + RANDOM % 900))}
export MINISGL_HOST_PORT="$PORT"
say() { echo "$@" | tee -a "$OUT"; }
say "=== decode_rocprof run=$RUN_ID image=$IMAGE model=$MODEL_ID engine=$(cd "$WT" && git rev-parse --short HEAD)"

YMLF="$WT/docker-compose.decode-$RUN_ID.yml"
write_yml() {  # $1 = command line, $2... = extra "KEY: val" env lines
  local cmd="$1"; shift
  { echo "services:"; echo "  serve:"
    echo "    container_name: minisgldec-$RUN_ID"
    echo "    command: [\"$cmd\"]"
    echo "    environment:"
    echo "      MINISGL_DECODE_ROCPROF_RUN: \"$RUN_ID\""
    for kv in "$@"; do echo "      $kv"; done
  } > "$YMLF"
}
# COMPOSE_EXTRA: an additional compose file (e.g. a kernel-package bind-mount override) for BOTH legs.
DC() { ( cd "$WT" && env MINISGL_IMAGE="$IMAGE" docker compose -f docker-compose.yml ${COMPOSE_EXTRA:+-f "$COMPOSE_EXTRA"} -f "$YMLF" --profile serve "$@" ); }
down() { DC down >/dev/null 2>&1; }
trap 'down; rm -f "$YMLF"' EXIT INT TERM

drive() {  # $1 = n sequential requests; prints tok/s of wall per request
  python3 - "$1" "$MAXTOK" "$PORT" <<'PY' 2>&1
import json, sys, time, urllib.request
n, maxtok, port = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
P = ("Write a Python module implementing an LRU cache class with type hints, docstrings, "
     "get/put/delete methods, a max-size eviction policy and a small unittest suite.")
for i in range(n):
    body = json.dumps({"model": "m", "max_tokens": maxtok, "temperature": 0.7, "top_p": 0.95,
                       "messages": [{"role": "user", "content": P}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=900))
    except Exception as e:
        print(f"req{i}: ended ({type(e).__name__}) — expected on the trace leg"); continue
    dt = time.perf_counter() - t0; ct = r["usage"]["completion_tokens"]
    print(f"req{i}: {ct} tok in {dt:.2f}s = {ct/dt:.1f} tok/s")
PY
}
wait_ready() {
  local i c="minisgldec-$RUN_ID"
  for i in $(seq 1 400); do
    if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" != "true" ] && [ "$i" -gt 5 ]; then
      say "!! container not running"; docker logs "$c" 2>&1 | tail -30 | tee -a "$OUT"; return 1; fi
    if docker logs "$c" 2>&1 | grep -qaE "Traceback \(most recent call last\)"; then
      say "!! traceback"; docker logs "$c" 2>&1 | grep -aA20 Traceback | tail -30 | tee -a "$OUT"; return 1; fi
    curl -s --max-time 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { say "== ready after ~$((i*3))s"; return 0; }
    sleep 3
  done
  say "!! readiness timeout"; return 1
}
export MODEL="$MODEL_ID" SPEC=none TP=2 CONC=1 GRAPH_BS=1

if [[ ",$LEGS," == *",base,"* ]]; then
  say ""; say "############ LEG base — served tok/s (no profiler) + hip-engage ledger"
  write_yml "exec /engine/tools/serve.sh"
  down; DC up -d >/dev/null 2>&1
  if wait_ready; then
    drive 1 >/dev/null
    drive 3 | tee -a "$OUT"
    docker logs "minisgldec-$RUN_ID" 2>&1 | grep -oaE "\[hip-engage\] .*" | sort -u | tee -a "$OUT"
    docker logs "minisgldec-$RUN_ID" > "$SCRATCH/decode_rocprof-$RUN_ID.base.log" 2>&1
  fi
  down
fi

if [[ ",$LEGS," == *",trace,"* ]]; then
  say ""; say "############ LEG trace — rocprofv3 windowed on decode steps $RTX_SKIP..$((RTX_SKIP+RTX_STEPS))"
  if [ "$TRACE_MODE" = full ]; then RPFLAGS="--kernel-trace"; else RPFLAGS="--kernel-trace --marker-trace --selected-regions"; fi
  write_yml "exec rocprofv3 $RPFLAGS --output-format csv -d /engine/$RPDIR -o dec -- /engine/tools/serve.sh" \
    "MINISGL_ROCTX: \"1\"" "MINISGL_ROCTX_SKIP: \"$RTX_SKIP\"" "MINISGL_ROCTX_STEPS: \"$RTX_STEPS\"" \
    "MINISGL_EXIT_AFTER_STEPS: \"$STEPS\""
  down; DC up -d >/dev/null 2>&1
  if wait_ready; then
    drive 2 >>"$OUT"
    for _ in $(seq 1 200); do
      [ "$(docker inspect -f '{{.State.Running}}' "minisgldec-$RUN_ID" 2>/dev/null)" = "true" ] || break
      sleep 3
    done
    docker logs "minisgldec-$RUN_ID" > "$SCRATCH/decode_rocprof-$RUN_ID.trace.log" 2>&1
  fi
  say "--- trace files"; find "$WT/$RPDIR" -type f -size +0 2>/dev/null | tee -a "$OUT"
  cp -r "$WT/$RPDIR" "$SCRATCH/" 2>/dev/null
fi
say ""; say "=== done. out=$OUT  trace=$SCRATCH/$RPDIR"
