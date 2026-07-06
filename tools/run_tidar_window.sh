#!/usr/bin/env bash
# TiDAR self-draft spec-decode validation window. Launch UNDER the shared lease (single card):
#   gpu-lease -n 1 -- bash tools/run_tidar_window.sh
# Thin wrapper: forwards the lease's device env into vllm22-w4a8:combined and runs spec_tidar.sh.
# Mounts /home/pat/code/_big:/big (the fp8 TiDAR models) + THIS worktree as /engine (the B.0/B.1 code).
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_tidar_window] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e MODEL="${MODEL:-/big/zaya1-tidar-opd-fp8}" \
  -e NUM_DRAFT="${NUM_DRAFT:-4}" -e MAXTOK="${MAXTOK:-64}" -e MEMRATIO="${MEMRATIO:-0.85}" \
  -e FUSED="${FUSED:-0}" -e NOREP="${NOREP:-0}" -e OLDMOE="${OLDMOE:-0}" -e SEG="${SEG:-0}" \
  -e DUMP="${DUMP:-0}" -e PROFILE="${PROFILE:-0}" -e TIME="${TIME:-0}" -e W8A16="${W8A16:-0}" \
  -e GRAPH="${GRAPH:-0}" \
  -v "$PWD":/engine \
  -v /home/pat/code/_big:/big:ro \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/engine \
  --entrypoint bash vllm22-w4a8:combined /engine/tools/spec_tidar.sh
echo "[run_tidar_window] exited rc=$?"
