#!/usr/bin/env bash
# SWA graph-capture byte-identity + tok/s validation (runs INSIDE the lean container, one -n 2 lease).
# Phase 1: EAGER   (--cuda-graph-max-bs 0) -> short greedy (byte-identity ref) + long greedy (tok/s).
# Phase 2: CAPTURED (--cuda-graph-max-bs N) -> same prompts.
# GATE: short sha CAPTURED == short sha EAGER (capture must not change greedy output). bf16 KV.
set -u
source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
MODEL="${SWA_MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
RES=/engine/_swagraph_results.txt
: > "$RES"
SERVER_PID=0
LOG=/engine/_swagraph_server.log

start_server() {  # $1 = cuda-graph-max-bs (0 = eager); sets SERVER_PID, LOG
  LOG="/engine/_swagraph_server_g$1.log"
  # OVERLAP_OFF=1 (default) -> synchronous normal_loop (required for cross-run byte-identity).
  # OVERLAP_OFF=0 -> production overlap scheduling (tok/s-representative; no byte-identity claim).
  MINISGL_ATTN_HIP=1 MINISGL_KV_FP8=0 MINISGL_DISABLE_OVERLAP_SCHEDULING="${OVERLAP_OFF:-1}" \
    setsid python -m minisgl \
      --model "$MODEL" --host 0.0.0.0 --port 1919 \
      --cache-type radix --attention-backend hip --page-size 16 --tp 2 \
      --disable-pynccl --cuda-graph-max-bs "$1" --max-running-requests 2 \
      --memory-ratio 0.90 >"$LOG" 2>&1 &
  SERVER_PID=$!
}

wait_ready() {
  local i
  for i in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "!! server exited early"; tail -50 "$LOG"; return 1; fi
    if curl -sf http://localhost:1919/health >/dev/null 2>&1; then echo "== server ready after ~$((i*2))s"; return 0; fi
    sleep 2
  done
  echo "!! readiness timeout"; tail -50 "$LOG"; return 1
}

stop_server() {
  local i
  kill -9 -"$SERVER_PID" 2>/dev/null
  pkill -9 -f minisgl 2>/dev/null
  for i in $(seq 1 60); do
    if python -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',1920))!=0 else 1)" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "!! port 1920 still busy after 60s"
}

echo "########## PHASE 1: EAGER (--cuda-graph-max-bs 0) ##########"
start_server 0
if wait_ready; then
  python /engine/tools/swa_graph_client.py --mode eager --case short --max-tokens 32 | tee -a "$RES"
  python /engine/tools/swa_graph_client.py --mode eager --case long  --max-tokens 128 | tee -a "$RES"
fi
stop_server; sleep 5

echo "########## PHASE 2: CAPTURED (--cuda-graph-max-bs 2) ##########"
start_server 2
if wait_ready; then
  echo "--- capture / KV-pool log lines ---"
  grep -iE "capturing CUDA graphs|Free GPU memory (before|after) capturing|SWA ring KV|KV cache|num_pages|Free memory after init" "$LOG" | head -20
  python /engine/tools/swa_graph_client.py --mode captured --case short --max-tokens 32 | tee -a "$RES"
  python /engine/tools/swa_graph_client.py --mode captured --case long  --max-tokens 128 | tee -a "$RES"
fi
stop_server

echo "########## BYTE-IDENTITY GATE (short) + tok/s ##########"
sha_of() { grep "mode=$1 case=$2 " "$RES" | grep -o 'sha=[0-9a-f]*' | head -1; }
tps_of() { grep "mode=$1 case=$2 " "$RES" | grep -oE 'decode_tok_s=[0-9.]+' | head -1 | cut -d= -f2; }
es=$(sha_of eager short);   cs=$(sha_of captured short)
if [ -n "$es" ] && [ "$es" = "$cs" ]; then echo "PASS  short: captured==eager ($es) BYTE-IDENTICAL";
else echo "FAIL  short: eager=$es captured=$cs"; fi
el=$(tps_of eager long); cl=$(tps_of captured long)
echo "TOK/S long: eager=$el captured=$cl decode_tok_s"
echo "########## DONE ##########"
