#!/usr/bin/env bash
# Run tools/quant_m_invariance.py in the serve image against THIS worktree.
#   gpu-lease -n 1 --timeout 900 -- bash tools/quant_m_invariance_run.sh [extra args...]
set -uo pipefail
IMG="${QMINV_IMG:-minisgl-rdna4:gemma4}"
WT="${QMINV_WT:-/home/pat/code/minisgl-rdna4-qminv}"
docker run --rm --name qminv-$$ \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 \
  -v "$WT":/engine \
  --entrypoint bash "$IMG" -lc \
    "PYTHONPATH=/opt/kernels:/engine/python:/engine python /engine/${QMINV_TOOL:-tools/quant_m_invariance.py} $*"
