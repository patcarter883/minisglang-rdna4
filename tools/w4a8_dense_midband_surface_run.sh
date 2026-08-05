#!/usr/bin/env bash
# Run tools/w4a8_dense_midband_surface.py in the serve image against THIS worktree.
#   gpu-lease -n 1 --timeout 5400 -- bash tools/w4a8_dense_midband_surface_run.sh [extra args...]
set -uo pipefail
IMG="${MIDBAND_IMG:-minisgl-rdna4:gemma4}"
WT="${MIDBAND_WT:-/home/pat/code/minisgl-rdna4-midband}"
docker run --rm --name midband-$$ \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e MINISGL_HIP_ENGAGE_LOG=0 \
  -v "$WT":/engine \
  --entrypoint bash "$IMG" -lc \
    "PYTHONPATH=/opt/kernels:/engine/python:/engine python /engine/${MIDBAND_TOOL:-tools/w4a8_dense_midband_surface.py} $*"
