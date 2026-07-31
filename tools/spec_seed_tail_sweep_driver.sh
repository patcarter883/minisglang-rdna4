#!/usr/bin/env bash
# HOST-side driver for tools/spec_seed_tail_sweep.sh. Run under ONE lease:
#   gpu-lease -n 2 -- bash tools/spec_seed_tail_sweep_driver.sh
# Runs one container (one boot) per leg, sequentially, from THIS worktree (compose mounts `.`).
set -uo pipefail
cd "$(dirname "$0")/.."
export MINISGL_IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:zroute-merged}"
LEGS="${LEGS:-none 0 32 64 128 256 528}"
OUT=tools/seedtail_sweep_raw.txt
: > "$OUT"
for T in $LEGS; do
  echo "=============================== LEG TAIL=$T  $(date -Is) ===============================" | tee -a "$OUT"
  MINISGL_CMD="TAIL=$T DBG=${DBG:-2} bash /engine/tools/spec_seed_tail_sweep.sh" \
    timeout 3600 docker compose --profile run run --rm \
      -e TAIL="$T" -e DBG="${DBG:-2}" run 2>&1 | tee -a "$OUT"
  echo "[driver] leg TAIL=$T exit=${PIPESTATUS[0]}" | tee -a "$OUT"
  docker ps --format '{{.Names}}' | grep -i seedsweep && echo "WARNING stray container" | tee -a "$OUT"
  sleep 5
done
echo "[driver] ALL DONE $(date -Is)" | tee -a "$OUT"
