#!/usr/bin/env bash
# gpu-lease -n 1 --wait --name gemm-m -- bash tools/run_gemm_m.sh
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
  --entrypoint bash minisgl-rdna4:lean -lc "python /engine/tools/gemm_m_invariance.py"
echo "[gemm-m] done rc=$?"
