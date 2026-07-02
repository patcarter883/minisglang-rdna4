#!/usr/bin/env bash
# Prompt-prefill draft-KV seed validation window (GLM-4.7-Flash, TP=2). Launch UNDER the shared lease
# holding BOTH cards (TP=2):
#   gpu-lease -n 2 -- bash tools/run_prefill_seed_window.sh
#   ALGO=eagle3 gpu-lease -n 2 -- bash tools/run_prefill_seed_window.sh
# Thin wrapper: forwards the lease's device env into vllm22-w4a8:combined, copies the warm Triton
# cache to a writable throwaway (shared cache stays RO/uncorrupted), and runs spec_prefill_seed.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_prefill_seed_window] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset} ALGO=${ALGO:-mtp}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e ALGO="${ALGO:-mtp}" -e KDRAFT="${KDRAFT:-4}" \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/engine \
  --entrypoint bash vllm22-w4a8:combined -lc '
    set -e
    mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
    bash /engine/tools/spec_prefill_seed.sh
  '
echo "[run_prefill_seed_window] exited rc=$?"
