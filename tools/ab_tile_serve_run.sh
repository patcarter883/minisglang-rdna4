#!/usr/bin/env bash
# Host-side runner for ONE leg of the dense-tile/act-quant SERVE A/B. Run it under the arbiter:
#
#   gpu-lease -n 2 -- env LEG=BEFORE IMAGE=minisgl-rdna4:pre-tile \
#       REPO=/home/pat/code/minisgl-rdna4-abpretile bash tools/ab_tile_serve_run.sh
#
# One leg at a time, serially: two 35B TP=2 serves at once contend for board power/PSU/thermals and
# neither number is valid. No -p publish (the bench client lives inside the container), so this
# cannot collide with a serve another agent is running on 1919.
set -uo pipefail
LEG="${LEG:?LEG required}"
IMAGE="${IMAGE:?IMAGE required}"
REPO="${REPO:?REPO required — the ISOLATED worktree for this leg, never the shared tree}"
NAME="minisgl-abtile-${LEG}"

echo "[ab_tile] LEG=$LEG IMAGE=$IMAGE REPO=$REPO HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
docker run --rm --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e LEG="$LEG" -e REPS="${REPS:-3}" -e PORT="${PORT:-1919}" \
  -e MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}" -e TP="${TP:-2}" \
  -e MEMRATIO="${MEMRATIO:-0.90}" -e MAXRUN="${MAXRUN:-24}" -e GRAPH="${GRAPH:-16}" \
  -e NUM_PAGES="${NUM_PAGES:-16384}" \
  -e ATTN="${ATTN:-hip}" \
  -v "$REPO":/engine \
  -v /home/pat/models:/models \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH="/opt/kernels:/engine/python:/engine" \
  --entrypoint bash "$IMAGE" /engine/tools/_ab_tile_inner.sh
echo "[ab_tile] $LEG exited rc=$?"
