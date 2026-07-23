#!/usr/bin/env bash
# Per-layer bisect: dump the last-token hidden hash at each layer for the SHORT (deterministic) case,
# reuse vs cold. First layer whose last_sha differs pinpoints the divergent op.
set -u
MODEL="${SWA_MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
SERVER_PID=0; LOG=/engine/_swa_dump.log

start() {  # $1 = SWA_RADIX
  LOG="/engine/_swa_dump_radix$1.log"
  MINISGL_SWA_RADIX="$1" MINISGL_ATTN_HIP=1 MINISGL_KV_FP8=0 MINISGL_SWA_DUMP=1 \
    setsid python -m minisgl --model "$MODEL" --host 0.0.0.0 --port 1919 \
      --cache-type radix --attention-backend hip --page-size 16 --tp 2 --disable-pynccl \
      --cuda-graph-max-bs 0 --max-running-requests 8 --memory-ratio 0.85 >"$LOG" 2>&1 &
  SERVER_PID=$!
}
ready() { local i; for i in $(seq 1 300); do kill -0 "$SERVER_PID" 2>/dev/null || { echo "died"; tail -30 "$LOG"; return 1; }; curl -sf localhost:1919/health >/dev/null 2>&1 && { echo "ready ${i}"; return 0; }; sleep 2; done; return 1; }
stop() { kill -9 -"$SERVER_PID" 2>/dev/null; pkill -9 -f minisgl 2>/dev/null; local i; for i in $(seq 1 60); do python -c "import socket,sys;s=socket.socket();sys.exit(0 if s.connect_ex(('127.0.0.1',1920))!=0 else 1)" 2>/dev/null && return 0; sleep 1; done; }

echo "### SWA-RADIX reuse (short) ###"
start 1; ready && { python /engine/tools/swa_radix_client.py --mode reuse --case short --max-tokens 48; }
stop; sleep 4
echo "### NAIVE cold (short) ###"
start 0; ready && { python /engine/tools/swa_radix_client.py --mode cold --case short --max-tokens 48; }
stop
echo "### DONE ###"
