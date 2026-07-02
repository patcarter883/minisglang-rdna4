#!/usr/bin/env bash
# DFlash spec-decode validation window. Launch UNDER the shared lease (single card):
#   gpu-lease -n 1 -- bash tools/run_dflash_window.sh
# Thin wrapper: forwards the lease's device env into vllm22-w4a8:combined and runs spec_dflash.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_dflash_window] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e MODEL="${MODEL:-Qwen/Qwen3.5-4B}" -e DRAFT="${DRAFT:-z-lab/Qwen3.5-4B-DFlash}" \
  -e NUM_DRAFT="${NUM_DRAFT:-7}" -e WMMA="${WMMA:-1}" \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/engine \
  --entrypoint bash vllm22-w4a8:combined /engine/tools/spec_dflash.sh
echo "[run_dflash_window] exited rc=$?"
