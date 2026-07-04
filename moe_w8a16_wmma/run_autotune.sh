#!/usr/bin/env bash
# W8A16 tile autotune under a 1-card lease:  gpu-lease -n 1 -- bash moe_w8a16_wmma/run_autotune.sh
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_autotune] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -v "$PWD":/engine -w /engine -e PYTHONPATH=/engine \
  --entrypoint bash vllm22-w4a8:combined \
  -lc 'source /app/.venv/bin/activate && python moe_w8a16_wmma/moe_w8a16_autotune.py'
echo "[run_autotune] exited rc=$?"
