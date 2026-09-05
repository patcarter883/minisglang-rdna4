#!/usr/bin/env bash
# Drive tools/offload/hc_capture_prize.py in the serve image at the COMMITTED operating point.
#
# Same operating point as tools/offload/run_prof_decode_attrib.sh (tools/serve.sh's `qwen4exp` arm
# at MEM_RATIO 0.90, which is what every 48-layer boot on record used) so the numbers this produces
# are on the same footing as QWEN4EXP_ENDGAME.md's.
#
# GPU lease is WAIVED for this task. ONE invocation takes BOTH cards; never run two.
set -euo pipefail

REPO="${REPO:-/home/pat/code/minisgl-rdna4-hcfuse}"
IMAGE="${IMAGE:-minisgl-rdna4:m1b-20260903}"
OUT="${OUT:-/engine/docs/measurements/HC_FUSION_2026-09-05/hc_capture_prize.json}"

mkdir -p "$REPO/docs/measurements/HC_FUSION_2026-09-05"

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
exec python /engine/tools/offload/hc_capture_prize.py --json "$OUT" "$@"
INNER
