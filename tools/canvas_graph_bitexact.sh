#!/usr/bin/env bash
# canvas_graph_bitexact.sh — boot the DiffusionGemma canvas serve and read the in-process
# `[canvas-graph]` bit-exactness gate. One boot per ARM; an arm is a collective-regime setting.
#
# WHAT THIS IS FOR. §D10.6 recorded `graph vs eager max|delta|=1.575e+01` against §D6.1's standing
# "bit-identical at every captured batch size" claim. That number was taken by a gate that compared
# the captured region against an eager forward running a DIFFERENT PROGRAM: `tp_overlap` is
# capture-transparent, so under capture every all_reduce is inline and Gemma4's FFN row split is off,
# while the eager path uses a side stream and (default `MINISGL_TP_AR_CHUNKS=2`) splits the FFN into
# two 128-row chunks. The gate now takes its reference inside `inline_collectives()` and reports the
# regime difference separately; these arms are what says which of the two the 1.575e+01 was.
#
#   arm `matched`   default serve config. The gate's own matched-regime reference is the claim; the
#                   `eager-fallback regime ... vs matched` field on the same line is the row split.
#   arm `chunks1`   MINISGL_TP_AR_CHUNKS=1 — branch overlap only, no row split. The regime field
#                   should collapse if the row split is what moved it.
#   arm `nooverlap` MINISGL_TP_OVERLAP=0 — no side stream at all. The regime field must be exactly 0
#                   (the eager fallback is then the same program as the graph), which is the control
#                   that says the scope is doing what it claims.
#   arm `nosplit`   MINISGL_ATTN_MAX_SPLITS=1 — THE PROOF. `prefill_split_policy` is keyed on the
#                   BLOCK-TABLE ROW WIDTH (`max_blocks * block_size`), which is the real context for
#                   the eager path (18 pages = 288) and the CAPTURED STATIC WIDTH for the graph
#                   (16384 pages = 262144). So the graph runs the 64-way split-K + reduce kernels on
#                   every layer and the eager forward runs the single-pass kernel — two different
#                   kernels on the same inputs. Forcing num_splits=1 everywhere removes that one
#                   variable without touching a line of engine code: if the delta collapses to 0,
#                   the split path IS the divergence.
#
#   gpu-lease -n 2 --timeout 7200 -- bash tools/canvas_graph_bitexact.sh
set -uo pipefail
WT=${WT:-/home/pat/code/minisgl-rdna4-cgfix}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:dgprof}
DG_MODEL=${DG_MODEL:-cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4}
SCRATCH=${SCRATCH:-$HOME/.cache/minisgl-perf}
SWA_RADIX=${SWA_RADIX:-0}      # PINNED: chunked-encoder divergence on partial prefix hits (docs §D3)
CONC=${CONC:-1}
ARMS=${ARMS:-matched,chunks1,nooverlap}
MAXTOK=${MAXTOK:-32}           # the gate fires on the first two canvas steps; nothing needs a long run
# A UNIQUE HOST PORT, not the 1919 default. Other agents on this box serve the same compose file on
# the same two cards; a collision makes `up -d` fail before the container exists, which this harness
# then reports as "container not running after ~3s" with an EMPTY docker log -- a failure that reads
# like a boot crash and is not one. Measured: that is exactly what the first run of this did.
PORT=${PORT:-$((1920 + RANDOM % 900))}
mkdir -p "$SCRATCH"
RUN_ID=${RUN_ID:-$$$(od -An -N2 -tu2 </dev/urandom | tr -dc "0-9")}
OUT=${OUT:-$SCRATCH/canvas_bitexact-$RUN_ID.txt}
: > "$OUT"
export COMPOSE_PROJECT_NAME="minisglbx$RUN_ID"
export MINISGL_HOST_PORT="$PORT"

say() { echo "$@" | tee -a "$OUT"; }
say "=== canvas_graph_bitexact run=$RUN_ID image=$IMAGE"
say "=== engine=$(cd "$WT" && git rev-parse --short HEAD) swa_radix=$SWA_RADIX conc=$CONC arms=$ARMS port=$PORT"

YMLF="$WT/docker-compose.bx-$RUN_ID.yml"
write_yml() {
  { echo "services:"
    echo "  serve:"
    echo "    container_name: minisglbx-$RUN_ID"
    echo "    command: [\"exec /engine/tools/serve.sh\"]"
    echo "    environment:"
    echo "      MINISGL_SWA_RADIX: \"$SWA_RADIX\""
    for kv in "$@"; do echo "      $kv"; done
  } > "$YMLF"
}
DC() { ( cd "$WT" && env MINISGL_IMAGE="$IMAGE" docker compose -f docker-compose.yml -f "$YMLF" --profile serve "$@" ); }
down() { DC down >/dev/null 2>&1; }
trap 'down; rm -f "$YMLF"' EXIT INT TERM

drive() {
  python3 - "$1" "$PORT" <<'PY' 2>&1
import json, sys, time, urllib.request
body = json.dumps({"model": "m", "max_tokens": int(sys.argv[1]), "temperature": 0.0,
                   "stream": False,
                   "messages": [{"role": "user", "content": "Write one sentence about the sea."}]}).encode()
rq = urllib.request.Request(f"http://localhost:{sys.argv[2]}/v1/chat/completions", body,
                            {"Content-Type": "application/json"})
t0 = time.perf_counter()
try:
    r = json.load(urllib.request.urlopen(rq, timeout=600))
except Exception as e:
    print(f"req: FAILED {type(e).__name__}: {e}"); sys.exit(0)
ct = r.get("usage", {}).get("completion_tokens", 0)
print(f"req: {ct} tok in {time.perf_counter()-t0:.2f}s")
PY
}

wait_ready() {  # "Container Up" is NOT ready. Poll health; bail on a traceback rather than burn the
                # timeout. NOTE: the bit-exactness gate now RAISES, so an arm that fails shows up
                # here as a traceback mid-request, not as a bad log line.
  local i c; c="minisglbx-$RUN_ID"
  for i in $(seq 1 300); do
    if [ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" != "true" ]; then
      say "!! container not running after ~$((i*3))s"; docker logs "$c" 2>&1 | tail -40 | tee -a "$OUT"; return 1
    fi
    if docker logs "$c" 2>&1 | grep -qaE "Traceback \(most recent call last\)|torch.OutOfMemoryError"; then
      say "!! traceback in the log"; docker logs "$c" 2>&1 | grep -aA25 "Traceback" | tail -60 | tee -a "$OUT"; return 1
    fi
    if curl -s --max-time 3 http://localhost:$PORT/health >/dev/null 2>&1 \
       || curl -s --max-time 3 http://localhost:$PORT/v1/models >/dev/null 2>&1; then
      say "== ready after ~$((i*3))s"; return 0
    fi
    sleep 3
  done
  say "!! readiness timeout"; docker logs "$c" 2>&1 | tail -40 | tee -a "$OUT"; return 1
}

export MODEL="$DG_MODEL" SPEC=none TP=2 CONC="$CONC" GRAPH_BS="$CONC" EXTRA_ARGS=""

run_arm() {  # $1 = arm name, rest = compose env lines
  local arm="$1"; shift
  say ""; say "############ ARM $arm  ($*)"
  write_yml "$@"
  down
  if ! DC up -d >"$SCRATCH/up-$RUN_ID.log" 2>&1; then
    say "!! compose up FAILED"; tail -20 "$SCRATCH/up-$RUN_ID.log" | tee -a "$OUT"; return 1
  fi
  if wait_ready; then
    drive "$MAXTOK" | tee -a "$OUT"
    sleep 2
  fi
  say "--- [canvas-graph]"
  docker logs "minisglbx-$RUN_ID" 2>&1 | grep -a "\[canvas-graph\]" | tee -a "$OUT"
  say "--- overlap provenance / gate failure"
  docker logs "minisglbx-$RUN_ID" 2>&1 | grep -aE "overlap ENGAGED|canvas-graph\] captured canvas step is NOT" | tee -a "$OUT"
  say "--- [hip-engage]"
  docker logs "minisglbx-$RUN_ID" 2>&1 | grep -oaE "\[hip-engage\] .*" | sort -u | tee -a "$OUT"
  down
}

[[ ",$ARMS," == *",matched,"* ]]   && run_arm matched
[[ ",$ARMS," == *",chunks1,"* ]]   && run_arm chunks1   "MINISGL_TP_AR_CHUNKS: \"1\""
[[ ",$ARMS," == *",nooverlap,"* ]] && run_arm nooverlap "MINISGL_TP_OVERLAP: \"0\""
[[ ",$ARMS," == *",nosplit,"* ]]   && run_arm nosplit   "MINISGL_ATTN_MAX_SPLITS: \"1\""

say ""; say "=== done. out=$OUT"
