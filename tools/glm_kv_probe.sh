#!/usr/bin/env bash
# Probe: boot GLM-4.7-Flash EAGLE3 (fp8 KV, ctx=40000) just long enough to read the KV pool token
# capacity, then tear down. Launch UNDER the lease (TP=2):
#   gpu-lease -n 2 -- bash tools/glm_kv_probe.sh
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[probe] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
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
    mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
    source /app/.venv/bin/activate
    pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
    LOG=/engine/tools/glm_kv_probe.server.log
    setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 MINISGL_KV_FP8=1 \
      MINISGL_SPEC_PREFILL_SEED=1 python -m minisgl \
      --model QuantTrio/GLM-4.7-Flash-AWQ --tensor-parallel-size 2 --port 21972 --disable-pynccl \
      --graph 8 --memory-ratio 0.80 --max-seq-len-override 40000 --max-running-requests 4 \
      --spec-algorithm eagle3 --spec-draft-model-path thoughtworks/GLM-4.7-Flash-Eagle3 \
      --spec-num-draft 6 > "$LOG" 2>&1 &
    SRV=$!
    for _ in $(seq 1 200); do
      grep -q "Allocating .* tokens for KV cache" "$LOG" && break
      grep -q "Traceback (most recent call last)" "$LOG" && { echo "[probe] CRASH:"; tail -40 "$LOG"; break; }
      kill -0 "$SRV" 2>/dev/null || { echo "[probe] DIED:"; tail -40 "$LOG"; break; }
      sleep 2
    done
    echo "===== KV POOL + MLA DIMS ====="
    grep -E "Free memory before loading model|Allocating .* tokens for KV cache|Free memory after" "$LOG" || true
    grep -iE "kv_lora|qk_rope|num_layers|page size|Page size" "$LOG" | head || true
    kill -TERM -- "-$SRV" 2>/dev/null || true; sleep 2; kill -KILL -- "-$SRV" 2>/dev/null || true
  '
echo "[probe] done rc=$?"
