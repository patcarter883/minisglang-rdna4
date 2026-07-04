#!/usr/bin/env bash
# Persistent GLM-4.7-Flash EAGLE3 serving endpoint for load testing:
#   TP=2, MLA, EAGLE3 spec (K=6, prefill-seed), verify CUDA-graph ON, FP8 KV cache, radix prefix
#   cache, ctx=40000, max concurrency=2 (fits 2 x 40000 in the 92032-token fp8 pool).
# Foreground (blocking) docker run UNDER a 2-card lease -> the lease is held for the server lifetime.
#   nohup gpu-lease -n 2 -- bash tools/glm_serve.sh &
# Stop: docker kill <container>  (frees the lease).  HTTP API on host port 1919.
set -uo pipefail
cd "$(dirname "$0")/.."
K="${K:-6}"  # EAGLE3 spec draft length (--spec-num-draft); override via env, e.g. K=3
echo "[glm_serve] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}  port=1919  K=$K"
docker run --rm --name glm_serve_eagle3 \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -p 1919:1919 \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e K="$K" \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/engine \
  --entrypoint bash vllm22-w4a8:combined -lc '
    set -e
    mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
    source /app/.venv/bin/activate
    pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
    exec env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 MINISGL_KV_FP8=1 \
      MINISGL_SPEC_PREFILL_SEED=1 python -m minisgl \
      --model QuantTrio/GLM-4.7-Flash-AWQ --tensor-parallel-size 2 \
      --host 0.0.0.0 --port 1919 --disable-pynccl \
      --graph 8 --memory-ratio 0.80 --max-seq-len-override 40000 --max-running-requests 2 \
      --cache-type radix \
      --spec-algorithm eagle3 --spec-draft-model-path thoughtworks/GLM-4.7-Flash-Eagle3 \
      --spec-num-draft "${K}"
  ' > tools/glm_serve.server.log 2>&1
echo "[glm_serve] container exited rc=$?"
