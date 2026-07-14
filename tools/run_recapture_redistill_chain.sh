#!/usr/bin/env bash
# OVERNIGHT: re-capture the ZAYA DFlash OPD corpus on the CURRENT (dense_gemm/MINV=1) serve, then
# re-distill the CCA drafter on it. The 15:18 corpus is stale (aux_minv_delta: L39 relL2 0.14 vs the
# current serve) — this regenerates on-policy aux, then trains. Runs both stages under ONE 1-card lease:
#   gpu-lease -n 1 --name dflash-recap -- bash tools/run_recapture_redistill_chain.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# --- config: FULL-SIZE capture (the 3737-prompt "big" set that made the original ~1.12M positions) ---
export CAPDIR="${CAPDIR:-/home/pat/code/_dflash_capture_minisgl_minv}"   # FRESH dir, not the stale one
export PROMPTS="${PROMPTS:-/home/pat/code/_dflash_capture_prompts_big.txt}"
export WORKERS="${WORKERS:-96}"; export MAXRUN="${MAXRUN:-96}"; export GENTOK="${GENTOK:-256}"
export MEMRATIO="${MEMRATIO:-0.9}"
# --- re-distill config (num_spec=4 like m4dss; batch 256 to stay well under the 15/512 OOM we hit) ---
export SEEDBUF_HOST="$CAPDIR"
export INIT="${INIT:-/models_rw/ZAYA1-8B-DFlash-CCA-5L-init}"
export OUT="${OUT:-/models_rw/ZAYA1-8B-DFlash-CCA-5L-minv}"   # captured under the M-invariant serve
export NUM_SPEC="${NUM_SPEC:-4}"; export EPOCHS="${EPOCHS:-14}"; export BATCH="${BATCH:-256}"

echo "=================================================================="
echo "[chain] HIP=${HIP_VISIBLE_DEVICES:-unset}  kernels=$(git -C /home/pat/code/rdna4-hip-kernels rev-parse --short HEAD 2>/dev/null)"
echo "[chain] STAGE 1 capture -> $CAPDIR  (prompts=$(wc -l < "$PROMPTS") workers=$WORKERS gentok=$GENTOK)"
echo "[chain] STAGE 2 redistill -> ${OUT/\/models_rw//home/pat/code/_models}  (num_spec=$NUM_SPEC epochs=$EPOCHS batch=$BATCH)"
echo "=================================================================="

# Fresh capture dir (never mix with a partial/old corpus).
if [ -d "$CAPDIR" ] && [ -n "$(ls -A "$CAPDIR" 2>/dev/null)" ]; then
  echo "[chain] WARNING $CAPDIR non-empty — archiving to $CAPDIR.old.$$"; mv "$CAPDIR" "$CAPDIR.old.$$"
fi
mkdir -p "$CAPDIR"

echo "[chain] ===== STAGE 1: CAPTURE ($(date -u +%H:%M:%S)) ====="
LEASE_NAME="${LEASE_NAME:-dflash-recap}" bash tools/run_dflash_capture.sh
n_seed=$(ls "$CAPDIR"/seedbuf_*.pt 2>/dev/null | wc -l)
cap_gb=$(du -sh "$CAPDIR" 2>/dev/null | cut -f1)
echo "[chain] capture wrote $n_seed seedbuf files ($cap_gb)"
if [ "$n_seed" -lt 10 ]; then
  echo "[chain] FATAL: capture produced too few seedbufs ($n_seed) — NOT starting re-distill."; exit 1
fi

echo "[chain] ===== STAGE 2: RE-DISTILL ($(date -u +%H:%M:%S)) ====="
LEASE_NAME="${LEASE_NAME:-dflash-recap}" bash tools/run_dflash_redistill.sh
rc=$?
echo "[chain] ===== DONE ($(date -u +%H:%M:%S)) rc=$rc  drafter -> ${OUT/\/models_rw//home/pat/code/_models} ====="
exit $rc
