#!/usr/bin/env bash
# Runs INSIDE the minisgl-rdna4:lean container (launched by run_bench_window.sh under the lease).
# PRODUCTION serving benchmark: CUDA-graph decode capture ON (--graph), fused atomic-scatter MoE
# decode (MINISGL_MOE_SCATTER=1), HIP attention. Runs the prefill/decode/mixed x M matrix once.
# The scatter defaulted to 0 here for as long as the engine wrongly believed its atomicAdd could not
# be graph-captured. It captures fine and is worth +3.2% e2e (81.05 -> 83.60 tok/s, Qwen35B TP=2
# M=1, 2026-07-30), so this bench now measures the production path with the scatter ON.
set -uo pipefail
# Activate the serving venv: lean image (/opt/venv, the infra serving all day) or legacy (/app/.venv).
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate

MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
TP="${TP:-2}"
PORT="${PORT:-21009}"
MEMRATIO="${MEMRATIO:-0.82}"
MAXRUN="${MAXRUN:-24}"
GRAPH="${GRAPH:-24}"            # cuda_graph_max_bs; base rows default 24 so the sweep can reach M=24
                               # STILL graph-captured (M>GRAPH would fall back to eager). Spec rows
                               # override to their memory-safe value (MTP 8, DFlash 4).
# MINISGL_MOE_SCATTER. EMPTY (the default) means "do not override" -- the ENGINE's own default is
# authoritative, so this bench measures whatever production actually runs. Do NOT re-pin a value
# here: this line used to hardcode 0 and kept saying so in the header long after the engine default
# flipped to 1, which is how a bench silently starts measuring a path nobody ships.
MMS="${MOE_SCATTER:-}"
# Attention backend. 'hip' = native HIP flash (the only capture-capable GQA/MHA backend, for
# dense/Qwen). MLA models (GLM-4.7-Flash) MUST use 'auto' — the engine force-selects the capture-
# capable 'mla' backend (page_size 16) and a 'hip' override would just be re-overridden. The 'mla'
# backend's decode cudagraph capture is wired, so GRAPH>0 works for GLM too.
ATTN="${ATTN:-hip}"
BENCH_M="${BENCH_M:-1,2,4,8,16}"
# Spec-decode: SPEC=mtp | dflash | "" (base, no spec). SPEC_K = draft length (MTP 4, DFlash 15).
# DFLASH_MODEL = the drafter checkpoint (required for dflash). Adds --reasoning-parser auto so the
# thinking models terminate. Graph-captured verify like the compose serve profiles.
SPEC="${SPEC:-}"
SPEC_ARGS=""
case "$SPEC" in
  mtp)    SPEC_ARGS="--spec-algorithm mtp --spec-num-draft ${SPEC_K:-4} --reasoning-parser auto" ;;
  dflash) SPEC_ARGS="--spec-algorithm dflash --spec-draft-model-path ${DFLASH_MODEL:?DFLASH_MODEL required for SPEC=dflash} --spec-num-draft ${SPEC_K:-15} --reasoning-parser auto" ;;
  "")     : ;;
  *)      echo "[setup] unknown SPEC='$SPEC' (want mtp|dflash|empty)"; exit 1 ;;
esac
RESULTS=/engine/tools/tp2_results
mkdir -p "$RESULTS"

if [ "${SKIP_TRITON_COPY:-1}" = "1" ]; then
  echo "[setup] Triton-free native-HIP path (--attn hip + gdn_hip + native MoE) — NO Triton cache"
else
  echo "[setup] warm Triton cache (RO -> writable copy; only for the legacy triton_rdna4 backend) ..."
  mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
fi
# The lean serving image already ships every server dep; only pip-install if something is missing
# (and don't hard-fail offline — the import check below is the real gate).
if ! python -c "import msgpack,zmq,fastapi,uvicorn,pydantic,starlette,psutil,accelerate" 2>/dev/null; then
  echo "[setup] server deps missing — pip install ..."
  pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil || true
fi
python -c "import gdn_hip, moe_hip, tail_hip, mla_hip; print('[setup] hip pkgs import OK')" \
  || { echo '[setup] hip pkg import FAILED'; exit 1; }

SRV=""
launch() {  # $1 = log tag
  local tag="$1" log="$RESULTS/bench_$1.server.log"
  echo "[launch] attn=$ATTN graph_max_bs=$GRAPH moe_scatter=${MMS:-<engine default>} tag=$tag -> $log"
  local pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
  # Only export MINISGL_MOE_SCATTER when the caller actually asked for a value; an unset env var is
  # what lets the engine default stand (an empty one would read as "not 0", i.e. silently ON).
  local -a scat_env=(); [ -n "$MMS" ] && scat_env=(MINISGL_MOE_SCATTER="$MMS")
  # setsid => own process group, so stop() can kill the WHOLE engine tree (scheduler/worker subprocs);
  # a bare kill leaves them holding GPU+port and the next boot hangs.
  # --attn hip: the HIP attention backend (attn_hip prefill + attn_decode paged) is the ONLY
  # capture-capable backend; the default 'auto' resolves to triton_rdna4, whose cudagraph capture
  # is a Phase-4 stub (NotImplementedError). So production graph mode REQUIRES --attn hip.
  # --host 0.0.0.0: bind all interfaces so the published -p 1919 port is reachable from the host and
  # from Prometheus (host.docker.internal:1919). The default 127.0.0.1 binds container-loopback only,
  # so the serve is invisible to the monitored path (metrics never scraped).
  setsid env "${scat_env[@]}" python -m minisgl \
    --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --host 0.0.0.0 --graph "$GRAPH" \
    --attention-backend "$ATTN" $pynccl --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" \
    $SPEC_ARGS ${EP:+--enable-ep} \
    > "$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do   # graph capture adds boot time (captures each bs in the set)
    if python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null; then
      echo "[launch] ready"; return 0
    fi
    kill -0 "$SRV" 2>/dev/null || { echo "[launch] server PID $SRV DIED:"; tail -40 "$log"; return 1; }
    # Fail FAST on a worker/scheduler SUBPROCESS crash. The parent `python -m minisgl` survives a dead
    # scheduler child (kill -0 above still passes), so without this the loop polls the full ~20-min
    # timeout holding the shared GPU lease — the repeated "benchmark wedged" failure mode. The mp.Process
    # death banner ("Process minisgl-...:") + a traceback is a precise, low-false-positive crash signal;
    # OOM / no-HIP-GPU / CUDA-error cover the other fatal boot failures.
    if grep -qE '^Process minisgl-|^Traceback \(most recent call last\)|torch\.OutOfMemoryError|RuntimeError: No HIP GPUs|^[A-Za-z_.]*Error: |CUDA error:' "$log" 2>/dev/null; then
      echo "[launch] worker subprocess CRASHED — failing fast (see log):"; tail -40 "$log"; return 1
    fi
    sleep 3
  done
  echo "[launch] not ready in time:"; tail -40 "$log"; return 1
}
stop() {
  [ -n "$SRV" ] || return 0
  echo "[stop] terminating server process group -$SRV ..."
  kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null
  SRV=""
  for _ in $(seq 1 20); do
    python -c "import socket,sys; s=socket.socket(); r=s.connect_ex(('127.0.0.1',$PORT)); s.close(); sys.exit(0 if r!=0 else 1)" 2>/dev/null && break
    sleep 1
  done
  sleep 2
}
trap stop EXIT

echo "######## $MODEL  PRODUCTION (cuda-graph capture, fused atomic-scatter MoE decode) ########"
launch graph || exit 1
# confirm graph capture actually engaged (not a silent eager fallback)
grep -iE 'captur|cuda.?graph' "$RESULTS/bench_graph.server.log" | head -4 || true
# Concurrency ceiling for the sweep = min(max-running-requests, cuda-graph-max-bs): beyond MAXRUN a
# request QUEUES (not concurrent), beyond GRAPH a decode batch falls to EAGER (not the graph-captured
# production path). Cap at the smaller so every point is BOTH concurrent AND graph-captured.
CAP="$MAXRUN"; [ "${GRAPH:-0}" -gt 0 ] && [ "$GRAPH" -lt "$CAP" ] && CAP="$GRAPH"
echo "[bench] concurrency ceiling = min(MAXRUN=$MAXRUN, GRAPH=$GRAPH) = $CAP"
# WORKLOADS selects which rows run. The full matrix (prefill,decode,mixed) is the default; an A/B
# targeting one kernel sets WORKLOADS=decode so the run is minutes not tens of minutes AND so a
# single-section table can be parsed unambiguously (serve_ab.sh takes the tok/s of the row it asked
# for -- with three sections printed, "the M=1 row" is three different rows).
python /engine/tools/serve_matrix_bench.py --url "http://127.0.0.1:$PORT" \
  --label "$(basename "$MODEL") spec=${SPEC:-none} graph_max_bs=$GRAPH" --m "$BENCH_M" \
  --prefill-words "${PREFILL_WORDS:-480}" \
  --workloads "${WORKLOADS:-prefill,decode,mixed}" \
  --max-concurrency "$CAP"
stop
echo "[done] logs in $RESULTS/"
