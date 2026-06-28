#!/usr/bin/env bash
# Spec-length sweep window. Launch UNDER the shared lease (TP=2 -> both cards):
#   GLM:   ALGO_SET=glm  gpu-lease.sh -n 2 -- env ... bash tools/run_spec_len_sweep.sh
#   Qwen:  ALGO_SET=qwen gpu-lease.sh -n 2 -- env ... bash tools/run_spec_len_sweep.sh
# Forwards the lease device env + a CONFIGS list into vllm22-w4a8:combined and runs spec_len_sweep.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_spec_len_sweep] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset} MODEL=${MODEL:-?} TAG=${TAG:-out}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e MODEL="${MODEL:?set MODEL}" -e TP="${TP:-2}" -e ATTN="${ATTN:-auto}" \
  -e DRAFT="${DRAFT:-thoughtworks/GLM-4.7-Flash-Eagle3}" \
  -e MEMRATIO="${MEMRATIO:-0.85}" -e MAXRUN="${MAXRUN:-4}" -e MAXTOK="${MAXTOK:-256}" \
  -e CONFIGS="${CONFIGS:?set CONFIGS}" -e TAG="${TAG:-out}" \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/engine \
  --entrypoint bash vllm22-w4a8:combined -lc '
    set -e
    mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
    bash /engine/tools/spec_len_sweep.sh
  '
echo "[run_spec_len_sweep] exited rc=$?"
