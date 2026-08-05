#!/usr/bin/env bash
# Runner for the TP comms/compute-overlap op-level harnesses (regime sweep, control, bit-exactness).
#
# It used to bind-mount a hand-built custom_ar over the image's, because the then-current
# `minisgl-rdna4:gemma4` still shipped the SCALAR peer read and measuring against that would have
# reproduced exactly the stale number this work exists to replace. The rebuilt image (kernels 36d1ac4)
# bakes the vectorized `one_shot_ar`, so the mount is gone -- one less way for a .so built against a
# different image's ABI to produce a 0%-GPU wedge. Set KERNELS_CAR to re-introduce it for a kernel A/B.
#
# Usage (must already hold the lease, so the arbiter's device pair is in the environment):
#   gpu-lease -n 2 -- bash tools/tp_overlap_run.sh <name> <command...>
set -uo pipefail

ENGINE="${ENGINE:-$(cd "$(dirname "$0")/.." && pwd)}"
IMAGE="${MINISGL_IMAGE:-minisgl-rdna4:gemma4}"
NAME="$1"; shift

# Optional: mount a candidate custom_ar over the image's baked one (kernel A/B only).
CAR_MOUNT=()
if [ -n "${KERNELS_CAR:-}" ]; then
  [ -n "$(ls "$KERNELS_CAR"/custom_ar_C*.so 2>/dev/null)" ] || { echo "!! no .so at $KERNELS_CAR"; exit 1; }
  CAR_MOUNT=(-v "$KERNELS_CAR":/opt/kernels/custom_ar:ro)
fi

exec docker run --rm --name "$NAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 \
  -e MINISGL_TP_OVERLAP="${MINISGL_TP_OVERLAP:-}" \
  -e MINISGL_TP_OVERLAP_MIN_TOKENS="${MINISGL_TP_OVERLAP_MIN_TOKENS:-}" \
  -e MINISGL_TP_AR_CHUNKS="${MINISGL_TP_AR_CHUNKS:-}" \
  -v "$ENGINE":/engine \
  "${CAR_MOUNT[@]}" \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash "$IMAGE" -lc "$*"
