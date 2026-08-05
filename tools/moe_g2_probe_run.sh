#!/usr/bin/env bash
# Run a torch probe in the serve image against THIS worktree.
#   gpu-lease -n 1 -- bash tools/moe_g2_probe_run.sh tools/moe_g2_served_probe.py [args...]
set -uo pipefail
IMG="${G2_IMG:-minisgl-rdna4:post-tile}"
WT="${G2_WT:-/home/pat/code/minisgl-rdna4-g2sk}"
KERN="${G2_KERN:-}"          # optional: a kernels package dir to PREPEND to PYTHONPATH
TOOL="$1"; shift
PRE=""; [ -n "$KERN" ] && PRE="$KERN:"
docker run --rm --name g2probe-$$ \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -e HF_HUB_OFFLINE=1 -e MINISGL_HIP_ENGAGE_LOG=0 \
  ${G2_ENV:-} \
  -v "$WT":/engine ${G2_MOUNT:-} \
  --entrypoint bash "$IMG" -lc \
    "PYTHONPATH=${PRE}/opt/kernels:/engine/python:/engine ${G2_WRAP:-} python /engine/$TOOL $*"
