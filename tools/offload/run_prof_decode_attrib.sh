#!/usr/bin/env bash
# Drive tools/offload/prof_decode_attrib.py in the serve image at the COMMITTED operating point.
#
# The operating point is `tools/serve.sh`'s `qwen4exp` arm, minus one deliberate difference:
# MEM_RATIO stays at 0.90, which is what the capture A/B (79.86 ms/step) was measured at. 0.96 is
# the serve default and buys KV pages, not speed; changing it here would make this attribution
# incomparable with the number it is attributing.
#
# GPU lease is WAIVED for this task. ONE invocation takes BOTH cards; never run two.
set -euo pipefail

REPO="${REPO:-/home/pat/code/minisgl-rdna4-offload}"
IMAGE="${IMAGE:-minisgl-rdna4:m1b-20260903}"
OUT="${OUT:-/engine/docs/measurements/WEIGHT_OFFLOAD_2026-09-02/attrib/decode_attrib.json}"

mkdir -p "$REPO/docs/measurements/WEIGHT_OFFLOAD_2026-09-02/attrib"

docker run --rm -i \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e ROCR_VISIBLE_DEVICES="${ROCR:-0,1}" \
  -v "$REPO":/engine \
  -v /home/pat/.cache/hf-q4e:/model:ro \
  -v /home/pat/.cache/hf-ple:/ple:ro \
  -e HF_HUB_OFFLINE=1 \
  -e MINISGL_PLE_META_FILES=/ple/model-bf16-00010.safetensors \
  -e MINISGL_WEIGHT_ARENA_CHUNK_MIB="${CHUNK_MIB:-750}" \
  -e MINISGL_WEIGHT_ARENA_FLOOR_GIB="${FLOOR_GIB:-9}" \
  -e OUT="$OUT" \
  --entrypoint bash "$IMAGE" -s -- "$@" <<'INNER'
set -euo pipefail
export MINISGL_PLE_FILES="$(ls /ple/model-plefp8-*.safetensors | paste -sd:)"
export PYTHONPATH=/engine/python:/opt/kernels
exec python /engine/tools/offload/prof_decode_attrib.py --json "$OUT" "$@"
INNER
