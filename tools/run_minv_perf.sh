#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -v "$PWD":/engine -v /home/pat/code/rdna4-hip-kernels:/kernels \
  -e PYTHONPATH=/kernels/_kernels:/engine/python:/engine \
  --entrypoint bash minisgl-rdna4:lean -lc "python /engine/tools/minv_perf.py"
echo "[minv-perf] done rc=$?"
