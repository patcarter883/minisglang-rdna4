#!/usr/bin/env bash
# Run the fused moe_gemv_decode_silu parity in the lean image, with the FUSION w4a8_fp8_wmma build
# mounted AHEAD of /opt/kernels (so the modified .so wins) and tail_hip from /opt/kernels.
#   gpu-lease -n 1 --timeout 300 -- bash tools/moe_gemm1_silu_parity_run.sh
set -uo pipefail
IMG=minisgl-rdna4:lean
W4A8FIX=/home/pat/code/rdna4-hip-kernels-fusion/w4a8_fp8_wmma/torch-ext  # contains w4a8_fp8_wmma/ (fusion .so)
WT=/home/pat/code/minisgl-rdna4-fusion
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 \
  -v "$W4A8FIX":/opt/w4a8fix -v "$WT":/engine \
  --entrypoint bash "$IMG" -lc \
    'PYTHONPATH=/opt/w4a8fix:/opt/kernels /opt/venv/bin/python /engine/tools/moe_gemm1_silu_gemv_parity.py'
