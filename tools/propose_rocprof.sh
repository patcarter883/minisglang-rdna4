#!/usr/bin/env bash
# Decompose the SPEC step by GPU kernel, INSIDE the captured graphs.
#
# WHY A BOUNDED EXIT AND NO SIGNALS. rocprofv3 is NOT a separate process you can stop: it sets
# LD_PRELOAD and execs the target, so the tool is a library living inside the ENGINE process
# (measured 2026-08-01: `/proc/<pid>/exe -> python3.12`, cmdline `python -m minisgl ...`, and the
# profiler's handler firing on that same pid). Signalling "the profiler" signals the engine, and the
# injected handler treats SIGINT/SIGTERM as ERROR signals and aborts WITHOUT writing. Four shutdown
# strategies produced zero output before that was understood: SIGTERM via `compose stop`, SIGINT to
# PID 1, a pkill that self-matched, and running rocprofv3 as an in-container child.
#
# The trace is written by the tool's destructor during normal interpreter teardown, so the only way
# to get one out of a server is to make the server RETURN. MINISGL_EXIT_AFTER_STEPS does that.
#
#
# Needs minisgl-rdna4:lean-prof (stock lean core-dumps in rocprofiler_configure: torch vendors its
# own librocprofiler-sdk.so that registers after rocprofv3 closed its configuration window).
# The torch profiler cannot substitute: propose/verify replay from CUDA graphs, so MINISGL_PROFILE
# records the graph LAUNCH and zero kernels.
#
# ALWAYS lease BOTH cards (-n 2) even for a single-card test: it pins the run to GPU0 and, because
# the arbiter blocks until both are free, it SERIALISES profile runs for free. The lease is the lock;
# this script needs no other one.
#
# Every run also gets a UNIQUE compose project (container name) and a UNIQUE output dir. Without
# that, two runs leasing the same cards inherit the same COMPOSE_PROJECT_NAME from the arbiter, so
# one run's `down` trap tears down the OTHER run's container and both write the same rpv3out — which
# is exactly how four overlapping runs once produced three "identical" results from one trace.
#
#   gpu-lease -n 2 -- bash tools/propose_rocprof.sh
set -uo pipefail
WT=/home/pat/code/minisgl-rdna4-propose
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean-prof}
SCRATCH=${SCRATCH:-$HOME/.cache/minisgl-perf}
STEPS=${STEPS:-60}
mkdir -p "$SCRATCH"
RUN_ID=${RUN_ID:-$$$(od -An -N2 -tu2 </dev/urandom | tr -dc "0-9")}
RPDIR=rpv3out-$RUN_ID
OUT=${OUT:-$SCRATCH/propose_rocprof-$RUN_ID.txt}
: > "$OUT"
rm -rf "$WT/$RPDIR"; mkdir -p "$WT/$RPDIR"
# Unique container name per run (the arbiter sets COMPOSE_PROJECT_NAME from the LEASE, which is
# identical for every run on the same cards).
export COMPOSE_PROJECT_NAME="minisglprof$RUN_ID"

# The engine is the container process again. That was NOT possible before MINISGL_EXIT_AFTER_STEPS:
# rocprofv3 flushes only on a normal exit, and it is LD_PRELOADed into the engine process rather
# than being a separate one, so every signal-based stop landed in its error handler and aborted the
# write. With the engine ending itself, rocprofv3 finalizes and PID 1 exits on its own — so the
# CONTAINER STOPPING IS THE COMPLETION SIGNAL, instead of `sleep infinity` + docker exec + polling.
YMLF="$WT/docker-compose.rocprof-$RUN_ID.yml"   # per-run: a shared filename is another collision
cat > "$YMLF" <<YML
services:
  serve:
    # docker-compose.yml pins container_name to \${LEASE_NAME}-serve, which is IDENTICAL for every
    # run on the same cards — so COMPOSE_PROJECT_NAME cannot separate them and one run's `down` trap
    # kills another run's container. Override it per run.
    container_name: minisglprof-$RUN_ID
    command: ["exec rocprofv3 --kernel-trace --marker-trace --selected-regions --stats --output-format csv -d /engine/$RPDIR -o spec -- /engine/tools/serve.sh"]
YML
DC=(docker compose -f docker-compose.yml -f "$YMLF" --profile serve)
down() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" "${DC[@]}" down >/dev/null 2>&1 ); }
trap 'down; rm -f "$YMLF"' EXIT INT TERM
down

export MODEL=laguna SPEC=dflash SPEC_K=16 TP=2 CONC=8 GRAPH_BS=8 MINISGL_SPEC_DEBUG=1
# Marker-gated: annotate every spec step with propose/stage/verify_forward/accept ranges, and let
# roctxProfilerResume/Pause bound collection to steps [SKIP, SKIP+STEPS). No synchronize is added.
export MINISGL_ROCTX=1 MINISGL_ROCTX_SKIP=${RTX_SKIP:-20} MINISGL_ROCTX_STEPS=${RTX_STEPS:-20}
export MINISGL_EXIT_AFTER_STEPS=$STEPS
( cd "$WT" && env MINISGL_IMAGE="$IMAGE" "${DC[@]}" up -d >/dev/null 2>&1 )
C=$( cd "$WT" && MINISGL_IMAGE="$IMAGE" "${DC[@]}" ps -qa serve )
echo "run=$RUN_ID dir=$RPDIR container=$C  bound=$STEPS steps  markers=[SKIP=$MINISGL_ROCTX_SKIP,+$MINISGL_ROCTX_STEPS]" | tee -a "$OUT"

# The bound must sit BELOW the steps the driving request reaches: past it the loop blocks idle on
# receive_msg and would never get there. 60 << ~113 steps for a 384-token generation, so the engine
# exits MID-request. The HTTP call therefore fails, by design — we want the trace, not the answer.
ready=0
for _ in $(seq 1 400); do
  curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && { ready=1; break; }
  sleep 2
done
echo "ready=$ready" | tee -a "$OUT"
[ "$ready" = 1 ] || { tail -30 "$WT/rpv3out/engine.log" 2>&1 | tee -a "$OUT"; exit 1; }

python3 "$WT/tools/_rocprof_drive.py" 2>&1 | tee -a "$OUT"

# No signals anywhere. Wait for the engine to hit its bound and exit on its own.
# The container exits when the engine does. No signals, no polling a process tree.
echo "=== waiting for the container to exit on its own ===" | tee -a "$OUT"
for _ in $(seq 1 150); do
  [ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" = "true" ] || break
  sleep 2
done
echo "=== finalization ===" | tee -a "$OUT"
docker logs "$C" 2>&1 | grep -aiE "EXIT_AFTER_STEPS|Opened result|output generation" | tail -6 | tee -a "$OUT"
echo "=== output files ===" | tee -a "$OUT"
find "$WT/$RPDIR" -type f -size +0 2>/dev/null | tee -a "$OUT"
cp -r "$WT/$RPDIR" "$SCRATCH/" 2>/dev/null
