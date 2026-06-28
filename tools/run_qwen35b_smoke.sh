#!/usr/bin/env bash
# Launch UNDER the shared lease (TP=2, both cards):
#   /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 2 -- bash tools/run_qwen35b_smoke.sh
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_qwen35b_smoke] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}" -e TP="${TP:-2}" \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/engine \
  --entrypoint bash vllm22-w4a8:combined -lc '
    set -e
    mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
    bash /engine/tools/qwen35b_smoke.sh
  '
echo "[run_qwen35b_smoke] exited rc=$?"
