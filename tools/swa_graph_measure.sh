#!/usr/bin/env bash
# SWA graph-capture host-cost A/B + tok/s + byte-identity (INSIDE lean container, one -n 2 lease).
# P1 EAGER (--cuda-graph-max-bs 0): short sha (byte-identity ref) + long tok/s.
# P2 CAPTURED, VECTORIZED metadata (MINISGL_SWA_METADATA_VEC=1) + GRAPH_TIMING: short sha + long tok/s
#    + [graph-timing] prepare_for_replay ms/step.
# P3 CAPTURED, EAGER metadata (MINISGL_SWA_METADATA_VEC=0) + GRAPH_TIMING: long tok/s + prepare ms/step.
# Compare P2 vs P3 prepare_for_replay ms to isolate the SWA-metadata host cost. Overlap OFF throughout.
set -u
source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
MODEL="${SWA_MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
RES=/engine/_swagraph_measure.txt
: > "$RES"
SERVER_PID=0
LOG=/engine/_swagraph_m.log

start_server() {  # $1=cuda-graph-max-bs  $2=SWA_METADATA_VEC  $3=GRAPH_TIMING
  LOG="/engine/_swagraph_m_g$1_vec$2.log"
  MINISGL_ATTN_HIP=1 MINISGL_KV_FP8=0 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
    MINISGL_SWA_METADATA_VEC="$2" MINISGL_GRAPH_TIMING="$3" \
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
    if curl -sf http://localhost:1919/health >/dev/null 2>&1; then echo "== ready ~$((i*2))s"; return 0; fi
    sleep 2
  done
  echo "!! readiness timeout"; tail -50 "$LOG"; return 1
}
stop_server() {
  local i
  kill -9 -"$SERVER_PID" 2>/dev/null; pkill -9 -f minisgl 2>/dev/null
  for i in $(seq 1 60); do
    if python -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',1920))!=0 else 1)" 2>/dev/null; then return 0; fi
    sleep 1
  done
}

echo "########## P1 EAGER ##########"
start_server 0 1 0
if wait_ready; then
  python /engine/tools/swa_graph_client.py --mode eager --case short --max-tokens 32 | tee -a "$RES"
  python /engine/tools/swa_graph_client.py --mode eager --case long  --max-tokens 256 | tee -a "$RES"
fi
stop_server; sleep 5

echo "########## P2 CAPTURED + VECTORIZED metadata + timing ##########"
start_server 2 1 1
if wait_ready; then
  python /engine/tools/swa_graph_client.py --mode capvec --case short --max-tokens 32 | tee -a "$RES"
  python /engine/tools/swa_graph_client.py --mode capvec --case long  --max-tokens 256 | tee -a "$RES"
  echo "--- P2 graph-timing (VECTORIZED) ---"; grep "\[graph-timing\]" "$LOG" | tail -3
fi
stop_server; sleep 5

echo "########## P3 CAPTURED + EAGER metadata + timing ##########"
start_server 2 0 1
if wait_ready; then
  python /engine/tools/swa_graph_client.py --mode capold --case long  --max-tokens 256 | tee -a "$RES"
  echo "--- P3 graph-timing (EAGER metadata) ---"; grep "\[graph-timing\]" "$LOG" | tail -3
fi
stop_server

echo "########## GATE + SUMMARY ##########"
sha_of() { grep "mode=$1 case=$2 " "$RES" | grep -o 'sha=[0-9a-f]*' | head -1; }
tps_of() { grep "mode=$1 case=$2 " "$RES" | grep -oE 'decode_tok_s=[0-9.]+' | head -1 | cut -d= -f2; }
es=$(sha_of eager short); vs=$(sha_of capvec short)
if [ -n "$es" ] && [ "$es" = "$vs" ]; then echo "PASS byte-identity: capvec short == eager ($es)";
else echo "FAIL byte-identity: eager=$es capvec=$vs"; fi
echo "TOK/S long: eager=$(tps_of eager long) capvec=$(tps_of capvec long) capold=$(tps_of capold long)"
echo "########## DONE ##########"
