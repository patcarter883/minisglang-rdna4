#!/usr/bin/env bash
# seed_kv KV-parity (batched seed vs autoregressive step) for the prompt-prefill draft-KV seed.
# Launch UNDER the shared lease (single card):
#   gpu-lease -n 1 -- bash tools/run_seed_kv_parity.sh
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_seed_kv_parity] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/engine \
  --entrypoint bash vllm22-w4a8:combined -lc '
    set -e
    source /app/.venv/bin/activate
    mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
    pip install -q safetensors msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
    CKPT=$(ls -d /root/.cache/huggingface/hub/models--thoughtworks--GLM-4.7-Flash-Eagle3/snapshots/*/ | head -1)
    echo "[run_seed_kv_parity] EAGLE3_CKPT=$CKPT"
    EAGLE3_CKPT="$CKPT" python /engine/tools/seed_kv_parity.py
  '
echo "[run_seed_kv_parity] exited rc=$?"
