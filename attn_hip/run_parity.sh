#!/usr/bin/env bash
# Build + run attn_hip parity under a 1-card lease in the combined image.
set -euo pipefail
LEASE=/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh
WT=/home/pat/code/vllm-gfx1201-attn-hip
"$LEASE" -n 1 -- bash -c '
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    -v '"$WT"':/engine \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    --entrypoint bash vllm22-w4a8:combined -lc "
      source /app/.venv/bin/activate
      cd /engine/attn_hip
      echo === BUILD ===
      GPU_ARCHS=gfx1201 python setup.py build_ext --inplace 2>&1 | tail -25
      echo === PARITY ===
      cd /engine && PYTHONPATH=/engine python attn_hip/attn_hip_parity.py"'
