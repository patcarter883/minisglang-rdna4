#!/usr/bin/env bash
# Gemma4 SWA-radix validation — the evidence required before SWA-radix may default ON.
#
# The byte-identity proof for SWA-radix (2e5d6d5e) was established on LAGUNA, whose sliding and full
# layers share one head_dim. Gemma4's do NOT (256/8 sliding vs 512/2 full), and the ring pool's
# geometry was recently re-derived from swa_head_dim/swa_num_kv_heads — the exact tensors the window
# snapshot/restore reads and writes. So the proof does not transfer by argument; it has to be re-run.
#
# Runs INSIDE the lean container, under ONE foreground `gpu-lease -n 2`. Four things get measured:
#   L) LOSSLESS   reuse-a-warm-prefix output == cold output, byte for byte, at three prefix
#                 lengths: xlong (~2600 tok, WRAPS Gemma4's 1024 ring), long (~780), short (~20).
#   T) TTFT       what the reuse actually buys on a long shared prefix.
#   P) DECODE     bs=1 tok/s with SWA-radix on vs off — SWA-radix forces the synchronous normal_loop,
#                 which is its one recorded cost, so it must be priced.
#   M) MEMORY     the boot-time snapshot-store reservation, the scheduler's derived cap, and the KV
#                 pool token count, on both legs.
#
# bf16 KV (MINISGL_KV_FP8=0) throughout: fp8 quantization noise is not the thing under test, and the
# byte-identity gate needs a deterministic cache.
set -u
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1

MODEL="${SWA_MODEL:-cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4}"
# Generation length for the byte-identity gate. This serve is NOT bit-reproducible against itself
# past ~32 output tokens (a pre-existing, documented property: batched-GEMM M-dependence + the fused
# MoE gemm2's atomic reduction order), so a longer sample measures the model's own chaos and reports
# it as a prefix-cache failure. Probe the floor first, then gate INSIDE it. 24 is inside; the earlier
# 48-token default produced cold != cold2 with no cache present at all.
MAXTOK="${MAXTOK:-24}"
RES=/engine/_gemma4_swa_results.txt
: > "$RES"
SERVER_PID=0
LOG=/engine/_gemma4_swa.log

start_server() {  # $1 = MINISGL_SWA_RADIX, $2 = tag, $3 = MINISGL_DISABLE_OVERLAP_SCHEDULING
  LOG="/engine/_gemma4_swa_$2.log"
  # setsid: the server + its TP workers form their own process group, so stop_server can kill the
  # WHOLE group. A plain kill leaks the workers, which hold the torch.distributed port.
  # MINISGL_MOE_G2FUSE / MINISGL_MOE_FLAG are inherited from the container env, not set here: the
  # fused MoE gemm2 reduces with atomics whose ORDER varies run to run, which makes the model
  # non-bit-reproducible against ITSELF and silently bounds any byte-identity gate. Set G2FUSE=0 on
  # the container for a losslessness run; leave it at the production default for a perf run.
  MINISGL_SWA_RADIX="$1" MINISGL_KV_FP8=0 MINISGL_DISABLE_OVERLAP_SCHEDULING="$3" \
    MODEL="$MODEL" TP=2 SPEC=none PORT=1919 CONC=4 \
    setsid bash /engine/tools/serve.sh >"$LOG" 2>&1 &
  SERVER_PID=$!
  echo "== phase $2 (MINISGL_SWA_RADIX=$1 DISABLE_OVERLAP=$3) pid=$SERVER_PID log=$LOG"
}

wait_ready() {
  local i
  for i in $(seq 1 200); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "!! server exited early"; tail -40 "$LOG"; return 1; fi
    if curl -sf http://localhost:1919/health >/dev/null 2>&1; then echo "== ready after ~$((i*3))s"; return 0; fi
    sleep 3
  done
  echo "!! readiness timeout"; tail -40 "$LOG"; return 1
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

memory_lines() {  # the four numbers the sizing argument rests on
  echo "--- [$1] memory accounting ---" | tee -a "$RES"
  grep -hoE "Reserved [0-9.]+ GiB for the [a-z-]+ snapshot store.*" "$LOG" | tee -a "$RES"
  grep -hoE "Reserved [0-9.]+ GiB for [A-Za-z /+]*state \([0-9]+ slots\).*" "$LOG" | tee -a "$RES"
  grep -hoE "(recurrent|SWA-window)-radix snapshot store: .*" "$LOG" | tee -a "$RES"
  grep -hoE "Allocating [0-9]+ tokens for KV cache.*" "$LOG" | tee -a "$RES"
  grep -hoE "(SWA-hybrid model: .*|SWA ring KV: .*)" "$LOG" | head -4 | tee -a "$RES"
}

########## L/T: byte-identity + TTFT. Overlap scheduling forced OFF on BOTH legs — SWA-radix forces
########## the synchronous normal_loop by construction, so an overlap-scheduled cold reference would
########## differ by ~1 ULP of async-op timing and mask the real result.
echo "########## PHASE L1: SWA-RADIX ON (reuse) ##########"
start_server 1 radixon_noovl 1
if wait_ready; then
  grep -i "SWA-radix prefix cache ENABLED" "$LOG" || echo "(warn: enable log not found)"
  memory_lines radixon
  for c in xlong long short; do
    python /engine/tools/swa_radix_client.py --mode reuse --case $c --max-tokens "$MAXTOK" | tee -a "$RES"
  done
  echo "--- SWA-radix HIT lines ---"; grep -ic "SWA-radix HIT" "$LOG"
fi
stop_server; sleep 5

echo "########## PHASE L2: SWA-RADIX OFF (cold reference, x2 for a determinism control) ##########"
# With SWA-radix off the scheduler forces 'naive' — no prefix cache at all — so re-running the same
# prompt on this same server is a genuine COLD repeat, not a cache hit. That is the determinism
# control: if cold != cold2 the model's long prefill is not run-to-run deterministic and the
# byte-identity gate is bounded by that noise floor rather than by the prefix cache.
start_server 0 radixoff_noovl 1
if wait_ready; then
  memory_lines radixoff
  for c in xlong long short; do
    python /engine/tools/swa_radix_client.py --mode cold --case $c --max-tokens "$MAXTOK" | tee -a "$RES"
  done
  for c in xlong long short; do
    python /engine/tools/swa_radix_client.py --mode cold --case $c --max-tokens "$MAXTOK" | sed 's/mode=cold/mode=cold2/' | tee -a "$RES"
  done
fi
stop_server; sleep 5

########## P: decode throughput.  (skipped when PHASES=lossless) Overlap scheduling at its DEFAULT here — the SWA-off leg gets the
########## overlap scheduler it would really run with, so the comparison prices SWA-radix honestly
########## (including the normal_loop it forces), instead of handicapping the baseline to match.
if [ "${PHASES:-all}" != "lossless" ]; then
echo "########## PHASE P1: SWA-RADIX ON — decode perf ##########"
start_server 1 radixon_perf 0
if wait_ready; then
  memory_lines radixon_perf
  python /engine/tools/serve_perf.py --conc 1 --maxtok 256 --container "" 2>&1 | tee -a "$RES" | tail -40
fi
stop_server; sleep 5

echo "########## PHASE P2: SWA-RADIX OFF — decode perf (the 43.9 tok/s baseline) ##########"
start_server 0 radixoff_perf 0
if wait_ready; then
  memory_lines radixoff_perf
  python /engine/tools/serve_perf.py --conc 1 --maxtok 256 --container "" 2>&1 | tee -a "$RES" | tail -40
fi
stop_server
fi

echo "########## BYTE-IDENTITY GATE ##########"
sha_of() { grep "mode=$1 case=$2 " "$RES" | grep -o 'sha=[0-9a-f]*' | head -1; }
for case in xlong long short; do
  r=$(sha_of reuse "$case"); c=$(sha_of cold "$case"); c2=$(sha_of cold2 "$case")
  det="deterministic"; [ "$c" = "$c2" ] || det="NON-DET (cold!=cold2: $c vs $c2)"
  if [ -n "$r" ] && [ "$r" = "$c" ]; then echo "PASS  $case: reuse==cold ($r) BYTE-IDENTICAL  [model $det]"
  else echo "FAIL  $case: reuse=$r cold=$c cold2=$c2  [model $det]"; fi
done
echo "--- TTFT ---"; grep -oE "mode=[a-z0-9]+ case=[a-z]+ sha=[0-9a-f]+ ttft=[0-9.]+ total=[0-9.]+" "$RES"
echo "########## DONE ##########"
