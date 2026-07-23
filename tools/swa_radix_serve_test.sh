#!/usr/bin/env bash
# SWA-radix serve validation (runs INSIDE the lean container under a single foreground -n 2 lease).
# Phase 1: SWA-radix ON  -> warm a prefix, REUSE it (long + short) -> record sha + TTFT.
# Phase 2: naive (SWA off) -> same B COLD -> record sha + TTFT.
# GATE: reuse sha == cold sha for both cases. bf16 KV (MINISGL_KV_FP8=0) required for byte-identity.
set -u
MODEL="${SWA_MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
RES=/engine/_swa_results.txt
: > "$RES"
SERVER_PID=0
LOG=/engine/_swa_server.log

start_server() {  # $1 = MINISGL_SWA_RADIX value; sets globals SERVER_PID (process-GROUP leader), LOG
  LOG="/engine/_swa_server_radix$1.log"
  # setsid -> the server + its TP-worker children form their OWN process group, so stop_server can
  # kill the WHOLE group (the workers hold the torch.distributed port 1920; a plain kill leaks them).
  # DISABLE_OVERLAP on BOTH configs: SWA-radix (reuse) already forces the synchronous normal_loop, so
  # the naive cold reference must too — else cold(overlap) vs reuse(normal) differ by ~1 ULP from
  # async-op timing (unrelated to prefix caching), the artifact that masked the true byte-identity.
  MINISGL_SWA_RADIX="$1" MINISGL_ATTN_HIP=1 MINISGL_KV_FP8=0 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
    setsid python -m minisgl \
      --model "$MODEL" --host 0.0.0.0 --port 1919 \
      --cache-type radix --attention-backend hip --page-size 16 --tp 2 \
      --disable-pynccl --cuda-graph-max-bs 0 --max-running-requests 8 \
      --memory-ratio 0.85 >"$LOG" 2>&1 &
  SERVER_PID=$!
}

wait_ready() {  # returns 0 when /health up, 1 if the process died / timed out
  local i
  for i in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "!! server exited early"; tail -40 "$LOG"; return 1; fi
    if curl -sf http://localhost:1919/health >/dev/null 2>&1; then echo "== server ready after ~$((i*2))s"; return 0; fi
    sleep 2
  done
  echo "!! readiness timeout"; tail -40 "$LOG"; return 1
}

stop_server() {  # kill the whole process GROUP (SERVER_PID is the setsid group leader)
  local i
  kill -9 -"$SERVER_PID" 2>/dev/null   # negative pid = the process group
  pkill -9 -f minisgl 2>/dev/null      # belt-and-braces: any stray worker
  # wait until the torch.distributed port 1920 is actually released before the next server binds it
  for i in $(seq 1 60); do
    if python -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1',1920))!=0 else 1)" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "!! port 1920 still busy after 60s"
}

echo "########## PHASE 1: SWA-RADIX ENABLED (reuse) ##########"
start_server 1
if wait_ready; then
  grep -i "SWA-radix prefix cache ENABLED" "$LOG" || echo "(warn: enable log not found)"
  python /engine/tools/swa_radix_client.py --mode reuse --case long  | tee -a "$RES" || echo "!! reuse/long failed"
  python /engine/tools/swa_radix_client.py --mode reuse --case short | tee -a "$RES" || echo "!! reuse/short failed"
  echo "--- SWA-radix HIT log lines ---"; grep -i "SWA-radix HIT" "$LOG" | head
fi
stop_server; sleep 5

echo "########## PHASE 2: NAIVE (cold reference) ##########"
start_server 0
if wait_ready; then
  python /engine/tools/swa_radix_client.py --mode cold --case long  | tee -a "$RES"
  python /engine/tools/swa_radix_client.py --mode cold --case short | tee -a "$RES"
fi
stop_server; sleep 3

echo "########## PHASE 3: NAIVE cold-vs-cold determinism control ##########"
start_server 0
if wait_ready; then
  # Re-run the same cold prompts on a fresh naive server; if these differ from Phase 2, the model's
  # long-prefill is non-deterministic run-to-run (a pre-existing fp8-MoE property, independent of the
  # prefix cache), and cross-run byte-identity is bounded by that noise floor.
  python /engine/tools/swa_radix_client.py --mode cold --case long  | sed 's/mode=cold/mode=cold2/' | tee -a "$RES"
  python /engine/tools/swa_radix_client.py --mode cold --case short | sed 's/mode=cold/mode=cold2/' | tee -a "$RES"
fi
stop_server

echo "########## BYTE-IDENTITY GATE (text sha) ##########"
sha_of() { grep "mode=$1 case=$2 " "$RES" | grep -o 'sha=[0-9a-f]*' | head -1; }
for case in long short; do
  r=$(sha_of reuse "$case"); c=$(sha_of cold "$case"); c2=$(sha_of cold2 "$case")
  det="det"; [ "$c" = "$c2" ] || det="NON-DET (cold!=cold2: $c vs $c2)"
  if [ -n "$r" ] && [ "$r" = "$c" ]; then echo "PASS  $case: reuse==cold ($r) BYTE-IDENTICAL  [model $det]";
  else echo "CHECK $case: reuse=$r cold=$c cold2=$c2  [model $det]"; fi
done
echo "--- TTFT (reuse prefill-skip vs cold) ---"; grep -oE "mode=[a-z0-9]+ case=[a-z]+ sha=[0-9a-f]+ ttft=[0-9.]+ total=[0-9.]+" "$RES"
echo "########## DONE ##########"
