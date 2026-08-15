#!/usr/bin/env bash
# GATE-0 BASELINE: a properly-powered RAG (+ current tap) row on the in-distribution
# language-pivot ripple, so a future Path R verdict has a defensible comparator.
#
# Recorded RAG numbers are scattered 0.47-0.667 across n=3-13 with +/-0.10 noise at cohort=10.
# This run uses the VALIDATED training config (bind 1500 / tap 200, carry ~0.86) so the
# delivery gate clears 0.40 and the verdict is actually readable, and raises the raw edit
# pool so the FILTERED (base-composes n tap-delivers) set is large enough to mean something.
#
# Smoke calibration (bind 100 / tap 30, 16 edits): eval ~4s/edit, peak 13.17 GiB, filtered
# yield 3/16 ~= 19%. So 200 raw edits -> filtered n ~= 35-40, eval ~13min.
#
# Source = the PRESERVED worktree (clean, committed) per the source-isolation rule.
set -uo pipefail
ENGINE=/home/pat/code/memory-organ-tapmetric-preserve
DATA=/home/pat/code/memory-organ/data
CACHE=/home/pat/code/memory-organ/.probe_cache
OUT=/tmp/claude-1000/-home-pat-code-minisgl-rdna4/4a9a74d7-7891-4ece-97cb-ede55cf98bd3/scratchpad/ripple_baseline.out

echo "[base] start $(date -u +%H:%M:%S) HIP=$HIP_VISIBLE_DEVICES ROCR=$ROCR_VISIBLE_DEVICES" | tee "$OUT"
echo "[base] engine=$ENGINE (commit $(git -C $ENGINE log -1 --format=%h)) card=$LEASE_ROCR_DEVICES" | tee -a "$OUT"

docker run --rm --name ripple-baseline \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e PYTORCH_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 \
  -e CAM_SKIP_CEILING=1 -e CAM_PROBE_CACHE_DIR=/probe_cache -e CAM_PERSISTENT_EVAL_BATCH=4 \
  -e CAM_POOLED_SUBJ_KEY=1 -e CAM_SUBJ_ONLY_QUERY=1 -e CAM_LEARNED_KEY_POOL=1 \
  -e CAM_RIPPLE_EVAL_HOP=country \
  -v "$ENGINE":/engine:ro -v "$DATA":/data:ro -v "$CACHE":/probe_cache \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash titans:dev -lc \
  "source /app/.venv/bin/activate && timeout 5400 python /engine/cam/recall_mag.py \
     --store pk --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 --seed 20260625 \
     --batch 4 --bind-steps 1500 --steps 200 --phrasing counterfactual_multi \
     --multi-relations 6 --cf-probe-cap 21500 --dataset counterfact --data-dir /data \
     --tap-layers 12 --seg-len 48 --qa-seg 3 --save-anyway --conf-gate \
     --indist-ripple --mquake-limit 200 2>&1" 2>&1 | tee -a "$OUT"

echo "[base] exit=$? done $(date -u +%H:%M:%S)" | tee -a "$OUT"
