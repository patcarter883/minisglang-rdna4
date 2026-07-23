#!/usr/bin/env bash
# Profile a CAPTURED bs=1 TP=2 Laguna decode step: dump a torch Chrome trace (skip 40 / active 50
# forward steps) via the engine's MINISGL_PROFILE hook, for host-side TraceLens breakdown.
# GPU-active vs CPU-gap, and WHAT fills the gap (_build_swa_metadata / TP all-reduce / compute).
set -u
source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
MODEL="${SWA_MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
OUTDIR=/engine/tools/_swagraph_prof
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
LOG=/engine/_swagraph_prof.log
SERVER_PID=0

# Overlap OFF: serial per-step structure (matches the byte-identity config) so the CPU metadata-build
# gap is visible on the critical path rather than hidden behind the next step's GPU work.
MINISGL_ATTN_HIP=1 MINISGL_KV_FP8=0 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
  MINISGL_PROFILE="$OUTDIR/decode.pt.trace.json.gz" MINISGL_PROFILE_SKIP=40 MINISGL_PROFILE_STEPS=50 \
  setsid python -m minisgl \
    --model "$MODEL" --host 0.0.0.0 --port 1919 \
    --cache-type radix --attention-backend hip --page-size 16 --tp 2 \
    --disable-pynccl --cuda-graph-max-bs 2 --max-running-requests 2 \
    --memory-ratio 0.90 >"$LOG" 2>&1 &
SERVER_PID=$!

ready=0
for i in $(seq 1 300); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "!! server exited early"; tail -50 "$LOG"; break; fi
  if curl -sf http://localhost:1919/health >/dev/null 2>&1; then echo "== ready ~$((i*2))s"; ready=1; break; fi
  sleep 2
done

if [ "$ready" = 1 ]; then
  # One long greedy generation (>=90 forward steps so the skip40+active50 window fully lands).
  python - <<'PY'
import json, urllib.request, time
body=json.dumps({"prompt":"A rigorous study of physics.","max_tokens":160,"ignore_eos":True}).encode()
r=urllib.request.Request("http://localhost:1919/generate",data=body,headers={"content-type":"application/json"})
t0=time.time()
with urllib.request.urlopen(r,timeout=900) as resp:
    for _ in resp: pass
print(f"gen done in {time.time()-t0:.2f}s")
PY
  # Wait for the profiler to flush the trace.
  for i in $(seq 1 30); do
    if grep -q "\[profile\] wrote" "$LOG"; then echo "== trace written"; break; fi
    sleep 1
  done
  grep -E "\[profile\]|SWA ring KV|Free GPU memory (before|after) capturing" "$LOG" | head
fi

kill -9 -"$SERVER_PID" 2>/dev/null; pkill -9 -f minisgl 2>/dev/null
for i in $(seq 1 60); do
  if python -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',1920))!=0 else 1)" 2>/dev/null; then break; fi
  sleep 1
done
echo "== trace files =="; ls -la "$OUTDIR"
echo "########## PROFILE DONE ##########"
