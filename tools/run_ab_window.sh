#!/usr/bin/env bash
# Functional A/B window: original AR ZAYA vs the TiDAR diffusion model. Launch UNDER the lease:
#   gpu-lease -n 1 -- bash tools/run_ab_window.sh
# Mounts /home/pat/models (original ZAYA1-8B-fp8) + /home/pat/code/_big (diffusion) + THIS worktree.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_ab_window] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e ORIG="${ORIG:-/models/ZAYA1-8B-fp8}" -e DIFF="${DIFF:-/big/zaya1-tidar-opd-fp8}" \
  -e NUM_DRAFT="${NUM_DRAFT:-4}" -e GENTOK="${GENTOK:-256}" -e NRUNS="${NRUNS:-3}" -e MEMRATIO="${MEMRATIO:-0.85}" \
  -e FUSED="${FUSED:-0}" -e SEG="${SEG:-0}" -e OLDMOE="${OLDMOE:-0}" -e TIME="${TIME:-0}" \
  -e W8A16="${W8A16:-0}" \
  -v "$PWD":/engine \
  -v /home/pat/models:/models:ro \
  -v /home/pat/code/_big:/big:ro \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/engine \
  --entrypoint bash vllm22-w4a8:combined /engine/tools/ab_zaya_vs_tidar.sh
echo "[run_ab_window] exited rc=$?"
