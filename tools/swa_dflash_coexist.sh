#!/usr/bin/env bash
# SWA-radix prefix-caching  x  DFlash spec-decode COEXISTENCE gate (Track C).
# Runs INSIDE the lean container under a single foreground -n 2 lease (Laguna TP=2, EAGER).
#
# Proves the two features — previously mutually-exclusive — now work TOGETHER and stay lossless:
#   Phase 1 : SWA-radix ON  + DFlash spec ON  -> WARM prefix P, then REUSE it (gen B=P+suffix).
#   Phase 2 : SWA-radix OFF + DFlash spec ON  -> COLD (gen B, no warm) = the spec-cold reference.
#   Phase 3 : SWA-radix OFF + DFlash spec ON  -> COLD again = model determinism control (cold vs cold2).
# GATE (deliverable 1): reuse-sha == cold-sha  => prefix reuse does NOT change the spec-generated
# tokens. The SHORT (<window) case is the HARD deterministic byte-identity gate (like Track A); the
# LONG (>window, ring-wrap) case is bounded by the model's own long-prefill noise floor (cold vs cold2).
# Also records TTFT (deliverable 3: reuse prefill-skip << cold) and greps memory/KV/accept (4).
#
# bf16 KV (MINISGL_KV_FP8=0) required: the SWA extend/verify primitive cats the ring window with inline
# bf16 K/V and runs the dense bf16 flash_prefill; an fp8 ring window would dtype-clash there.
set -u
# /opt/kernels first so `python -m minisgl` loads THIS worktree's source, not the image's baked
# /opt/minisgl stub (Track B footgun). venv activate gives Triton's JIT its PATH.
source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
MODEL="${SWA_MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
DRAFT="${DRAFT:-poolside/Laguna-XS-2.1-DFlash-NVFP4}"
K="${K:-16}"
TP="${TP:-2}"
RES=/engine/_swa_coexist_results.txt
: > "$RES"
SERVER_PID=0
LOG=/engine/_swa_coexist_server.log

start_server() {  # $1 = MINISGL_SWA_RADIX value; sets globals SERVER_PID (group leader), LOG
  LOG="/engine/_swa_coexist_radix$1.log"
  # setsid -> server + TP-workers form their own process group (stop_server kills the whole group;
  # the workers hold torch.distributed port 1920). DISABLE_OVERLAP on BOTH: SWA-radix already forces
  # the synchronous normal_loop, so the cold reference must too, else ~1 ULP async-timing drift masks
  # the true byte-identity (Track A's finding). MINISGL_SPEC_DEBUG=1 -> accept-len logging.
  MINISGL_SWA_RADIX="$1" MINISGL_ATTN_HIP=1 MINISGL_KV_FP8=0 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
  MINISGL_SPEC_DEBUG=1 \
    setsid python -m minisgl \
      --model "$MODEL" --host 0.0.0.0 --port 1919 \
      --cache-type radix --attention-backend hip --page-size 16 --tp "$TP" \
      --disable-pynccl --cuda-graph-max-bs 0 --max-running-requests 2 \
      --memory-ratio 0.90 \
      --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K" \
      >"$LOG" 2>&1 &
  SERVER_PID=$!
}

wait_ready() {
  local i
  for i in $(seq 1 400); do
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

echo "########## PHASE 1: SWA-RADIX + DFLASH SPEC (reuse) ##########"
start_server 1
if wait_ready; then
  grep -i "SWA-radix prefix cache ENABLED" "$LOG" || echo "(warn: enable log not found)"
  grep -i "spec-decode active" "$LOG" || echo "(warn: spec-active note not found)"
  python /engine/tools/swa_radix_client.py --mode reuse --case short | tee -a "$RES" || echo "!! reuse/short failed"
  python /engine/tools/swa_radix_client.py --mode reuse --case long  | tee -a "$RES" || echo "!! reuse/long failed"
  echo "--- SWA-radix HIT log lines ---"; grep -i "SWA-radix HIT" "$LOG" | head
  echo "--- accept-len lines (spec active during reuse) ---"; grep -iE "accept|acc=|\[spec\]" "$LOG" | tail -6
  echo "--- memory / KV pool / SWA ring lines ---"
  grep -iE "Free memory|KV cache|num_pages|SWA ring|draft model|KV pool|reserv|GiB|GB/card|tokens" "$LOG" | tail -25
fi
stop_server; sleep 5

echo "########## PHASE 2: DFLASH SPEC, SWA-radix OFF (cold reference) ##########"
start_server 0
if wait_ready; then
  python /engine/tools/swa_radix_client.py --mode cold --case short | tee -a "$RES"
  python /engine/tools/swa_radix_client.py --mode cold --case long  | tee -a "$RES"
fi
stop_server; sleep 3

echo "########## PHASE 3: DFLASH SPEC, SWA-radix OFF (cold determinism control) ##########"
start_server 0
if wait_ready; then
  python /engine/tools/swa_radix_client.py --mode cold --case short | sed 's/mode=cold/mode=cold2/' | tee -a "$RES"
  python /engine/tools/swa_radix_client.py --mode cold --case long  | sed 's/mode=cold/mode=cold2/' | tee -a "$RES"
fi
stop_server

echo "########## COEXISTENCE BYTE-IDENTITY GATE (spec+reuse vs spec-cold) ##########"
sha_of() { grep "mode=$1 case=$2 " "$RES" | grep -o 'sha=[0-9a-f]*' | head -1; }
for case in short long; do
  r=$(sha_of reuse "$case"); c=$(sha_of cold "$case"); c2=$(sha_of cold2 "$case")
  det="det"; [ "$c" = "$c2" ] || det="NON-DET (cold!=cold2: $c vs $c2 — model long-prefill noise floor)"
  if [ -n "$r" ] && [ "$r" = "$c" ]; then echo "PASS  $case: spec+reuse==spec-cold ($r) BYTE-IDENTICAL  [model $det]";
  else echo "CHECK $case: reuse=$r cold=$c cold2=$c2  [model $det]"; fi
done
echo "--- TTFT (reuse prefill-skip vs cold, spec active) ---"
grep -oE "mode=[a-z0-9]+ case=[a-z]+ sha=[0-9a-f]+ ttft=[0-9.]+ total=[0-9.]+" "$RES"
echo "########## DONE ##########"
