#!/usr/bin/env bash
# HOST-side driver for tools/spec_seed_crossover.sh. Run under ONE lease:
#   gpu-lease -n 2 -- bash tools/spec_seed_crossover_driver.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export MINISGL_IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:zroute-merged}"
LEGS="${LEGS:-0 32 none}"
OUT=tools/seedx_raw.txt
: > "$OUT"
for T in $LEGS; do
  echo "=============================== XLEG TAIL=$T  $(date -Is) ===============================" | tee -a "$OUT"
  MINISGL_CMD="TAIL=$T DBG=2 GENTOK=${GENTOK:-512} bash /engine/tools/spec_seed_crossover.sh" \
    timeout 3600 docker compose --profile run run --rm run 2>&1 | tee -a "$OUT"
  echo "[driver] xleg TAIL=$T exit=${PIPESTATUS[0]}" | tee -a "$OUT"
  sleep 5
done
echo "[driver] XOVER ALL DONE $(date -Is)" | tee -a "$OUT"
