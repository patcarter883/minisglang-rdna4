#!/usr/bin/env bash
# Run the dense fused gemm+silu parity/bench in the lean image with the FUSION w4a8_fp8_wmma build ahead
# of /opt/kernels.  gpu-lease -n 1 --timeout 300 -- bash tools/w4a8_gemm_silu_parity_run.sh [tool.py]
set -uo pipefail
TOOL="${1:-tools/w4a8_gemm_silu_parity.py}"
IMG=minisgl-rdna4:lean
W4A8FIX=/home/pat/code/rdna4-hip-kernels-fusion/w4a8_fp8_wmma/torch-ext
WT=/home/pat/code/minisgl-rdna4-fusion
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" -e HF_HUB_OFFLINE=1 \
  -v "$W4A8FIX":/opt/w4a8fix -v "$WT":/engine \
  --entrypoint bash "$IMG" -lc \
    "PYTHONPATH=/opt/w4a8fix:/opt/kernels /opt/venv/bin/python /engine/$TOOL"
