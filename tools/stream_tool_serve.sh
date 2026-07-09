#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B MXFP4 (TP=2) + dflash spec on the LEAN image for streaming tool-call
# validation. Launch UNDER the shared lease (both cards), backgrounded so the lease is held for the
# container lifetime:
#   nohup gpu-lease -n 2 -- bash tools/stream_tool_serve.sh &
# HTTP API on host port 1919. Stop: docker kill minisgl_stream_tool  (frees the lease).
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4}"
DRAFT="${DRAFT:-z-lab/Qwen3.6-35B-A3B-DFlash}"
echo "[stream_tool_serve] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset} port=1919 MODEL=$MODEL"
docker run --rm --name minisgl_stream_tool \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -p 1919:1919 \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e MODEL="$MODEL" -e DRAFT="$DRAFT" \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  --entrypoint bash minisgl-rdna4:lean -lc '
    set -e
    exec env PYTHONPATH=/opt/kernels:/engine/python:/engine \
      MINISGL_KV_FP8=1 python -m minisgl \
      --model "$MODEL" --tensor-parallel-size 2 \
      --host 0.0.0.0 --port 1919 --disable-pynccl \
      --memory-ratio 0.82 --max-running-requests 4 \
      --attention-backend hip \
      --reasoning-parser auto \
      --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft 7
  ' > tools/stream_tool_serve.server.log 2>&1
echo "[stream_tool_serve] container exited rc=$?"
