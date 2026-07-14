#!/usr/bin/env bash
# Train the num_spec=15 (full-width) DFlash drafter on the SAME fresh corpus the chain is capturing,
# in PARALLEL with the chain's num_spec=4 training. Waits UNLEASED for the shared capture to finish
# (so it doesn't hold a card idle), then leases a free card (card 1, while the chain holds card 0) and
# re-distills num_spec=15. Launch in background:
#   nohup bash tools/run_redistill_ns15_when_ready.sh > tools/_redistill_ns15.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
CHAINLOG=tools/_recap_redistill.log
echo "[ns15] $(date -u +%H:%M:%S) waiting (unleased) for the shared capture to finish..."
for _ in $(seq 1 180); do   # up to 60 min
  if grep -qiE "FATAL|server DIED|Traceback" "$CHAINLOG" 2>/dev/null && ! grep -q "capture wrote" "$CHAINLOG" 2>/dev/null; then
    echo "[ns15] chain capture failed before completion — aborting ns15."; exit 1
  fi
  grep -q "\[chain\] capture wrote" "$CHAINLOG" 2>/dev/null && break
  sleep 20
done
grep -q "\[chain\] capture wrote" "$CHAINLOG" 2>/dev/null || { echo "[ns15] capture never completed in time — aborting."; exit 1; }
n=$(ls /home/pat/code/_dflash_capture_minisgl_minv/seedbuf_*.pt 2>/dev/null | wc -l)
echo "[ns15] $(date -u +%H:%M:%S) capture done ($n seedbufs); leasing a card + training num_spec=15"
gpu-lease -n 1 --name dflash-ns15 -- env \
  NUM_SPEC=15 BATCH=128 EPOCHS=14 \
  INIT=/models_rw/ZAYA1-8B-DFlash-CCA-5L-init \
  OUT=/models_rw/ZAYA1-8B-DFlash-CCA-5L-minv-ns15 \
  SEEDBUF_HOST=/home/pat/code/_dflash_capture_minisgl_minv \
  bash tools/run_dflash_redistill.sh
echo "[ns15] $(date -u +%H:%M:%S) done rc=$? -> /home/pat/code/_models/ZAYA1-8B-DFlash-CCA-5L-minv-ns15"
