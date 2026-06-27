#!/usr/bin/env bash
# Runs INSIDE the vllm22-w4a8:combined container (launched by run_bench_window.sh under the lease).
# PRODUCTION serving benchmark: CUDA-graph decode capture ON (--graph), graph-safe MoE decode
# (MINISGL_MOE_SCATTER=0 — the scatter/split-K atomicAdd is NOT graph-capturable), HIP attention.
# Runs the prefill/decode/mixed x M matrix once. (Split-K is irrelevant under graphs — same atomic
# constraint — so it is NOT benchmarked here.)
set -uo pipefail
source /app/.venv/bin/activate

MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}"
TP="${TP:-2}"
PORT="${PORT:-21009}"
MEMRATIO="${MEMRATIO:-0.82}"
MAXRUN="${MAXRUN:-24}"
GRAPH="${GRAPH:-16}"            # cuda_graph_max_bs; capture set = [1,2,4]+range(8,GRAPH+1,8)
MMS="${MOE_SCATTER:-0}"        # MINISGL_MOE_SCATTER: 0 = graph-safe gather_reduce decode path
# Attention backend. 'hip' = native HIP flash (the only capture-capable GQA/MHA backend, for
# dense/Qwen). MLA models (GLM-4.7-Flash) MUST use 'auto' — the engine force-selects the capture-
# capable 'mla' backend (page_size 16) and a 'hip' override would just be re-overridden. The 'mla'
# backend's decode cudagraph capture is wired, so GRAPH>0 works for GLM too.
ATTN="${ATTN:-hip}"
BENCH_M="${BENCH_M:-1,2,4,8,16}"
RESULTS=/engine/tools/tp2_results
mkdir -p "$RESULTS"

if [ "${SKIP_TRITON_COPY:-1}" = "1" ]; then
  echo "[setup] Triton-free native-HIP path (--attn hip + gdn_hip + native MoE) — NO Triton cache"
else
  echo "[setup] warm Triton cache (RO -> writable copy; only for the legacy triton_rdna4 backend) ..."
  mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
fi
echo "[setup] server deps (pip install, ~1 min) ..."
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil
python -c "import gdn_hip, moe_hip, tail_hip, mla_hip; print('[setup] hip pkgs import OK')" \
  || { echo '[setup] hip pkg import FAILED'; exit 1; }

SRV=""
launch() {  # $1 = log tag
  local tag="$1" log="$RESULTS/bench_$1.server.log"
  echo "[launch] attn=$ATTN graph_max_bs=$GRAPH moe_scatter=$MMS tag=$tag -> $log"
  local pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
  # setsid => own process group, so stop() can kill the WHOLE engine tree (scheduler/worker subprocs);
  # a bare kill leaves them holding GPU+port and the next boot hangs.
  # --attn hip: the HIP attention backend (attn_hip prefill + attn_decode paged) is the ONLY
  # capture-capable backend; the default 'auto' resolves to triton_rdna4, whose cudagraph capture
  # is a Phase-4 stub (NotImplementedError). So production graph mode REQUIRES --attn hip.
  setsid env MINISGL_MOE_SCATTER="$MMS" python -m minisgl \
    --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --graph "$GRAPH" \
    --attention-backend "$ATTN" $pynccl --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" \
    > "$log" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do   # graph capture adds boot time (captures each bs in the set)
    if python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null; then
      echo "[launch] ready"; return 0
    fi
    kill -0 "$SRV" 2>/dev/null || { echo "[launch] server PID $SRV DIED:"; tail -40 "$log"; return 1; }
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

echo "######## $MODEL  PRODUCTION (cuda-graph capture, graph-safe MoE) ########"
launch graph || exit 1
# confirm graph capture actually engaged (not a silent eager fallback)
grep -iE 'captur|cuda.?graph' "$RESULTS/bench_graph.server.log" | head -4 || true
python /engine/tools/serve_matrix_bench.py --url "http://127.0.0.1:$PORT" \
  --label "$(basename "$MODEL") graph_max_bs=$GRAPH" --m "$BENCH_M"
stop
echo "[done] logs in $RESULTS/"
