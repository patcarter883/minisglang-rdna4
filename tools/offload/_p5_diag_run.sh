#!/usr/bin/env bash
# Runs the P5 empty_cache diagnostic in the serve image against THIS worktree.
#   _p5_diag_run.sh <expandable|plain> <host|device>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:lean}"
MODE="${1:-expandable}"
BACKING="${2:-host}"
[[ "$REPO" == "/home/pat/code/minisgl-rdna4" ]] && { echo "REFUSING: shared tree" >&2; exit 3; }

CONF=()
if [[ "$MODE" == "expandable" ]]; then
  CONF=(-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True)
fi

exec docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES=0,1 -e MINISGL_P5_ROCR_DEVICES=0,1 \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e HF_HUB_OFFLINE=1 \
  -e DIAG_BACKING="$BACKING" \
  -e P5_CHOWN_UID="$(id -u)" -e P5_CHOWN_GID="$(id -g)" \
  "${CONF[@]}" \
  -v "$REPO":/engine \
  --entrypoint bash "$IMAGE" -lc \
  "unset HIP_VISIBLE_DEVICES; python /engine/tools/offload/_p5_diag_emptycache.py"
