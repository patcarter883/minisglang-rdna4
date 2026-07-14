#!/usr/bin/env bash
# Queued pipeline: wait for the big capture to finish, then (each under its own 1-card lease, so they
# coordinate with every other agent on the box) validate SpecTr and re-distill the drafter on the fresh
# on-policy RXF seedbuf. Launch in the background:  bash tools/run_dflash_chain.sh
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[chain] waiting for the big capture to finish..."
until grep -q "big capture exited" /tmp/dfcap_big.out 2>/dev/null; do sleep 30; done
NPOS_FILES=$(ls /home/pat/code/_dflash_capture_minisgl/*.pt 2>/dev/null | wc -l)
CAP_MB=$(du -sm /home/pat/code/_dflash_capture_minisgl 2>/dev/null | cut -f1)
echo "[chain] capture done: $NPOS_FILES seedbuf files, ${CAP_MB}MB"
grep -E "done:.*captured" /tmp/dfcap_big.out 2>/dev/null | tail -1

echo "[chain] === step 1/2: validate SpecTr (incl Test 4: ddtree_walk_sampled) ==="
gpu-lease -n 1 --name specval -- bash tools/run_sampled_spec_validate.sh > /tmp/specval_chain.out 2>&1 || true
grep -E "Test 4|TV\(first|temp->0|VALIDATION" /tmp/specval_chain.out 2>/dev/null | tail -6

echo "[chain] === step 2/2: re-distill the CCA-DFlash drafter on the minisgl-RXF seedbuf ==="
gpu-lease -n 1 --name redistill -- bash tools/run_dflash_redistill.sh > /tmp/redistill.out 2>&1 || true
grep -iE "seed-records|examples|epoch|accept|acc=|diff_acc|saved|wrote|error|traceback" /tmp/redistill.out 2>/dev/null | tail -20
echo "[chain] DONE. new drafter: /home/pat/code/_models/ZAYA1-8B-DFlash-CCA-5L-minisgl-rxf"
