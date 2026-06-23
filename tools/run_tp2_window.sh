#!/usr/bin/env bash
# Phase 4 GPU window — TP=2 serve bring-up + parity probe (4-0 / 4-3 / 4-4).
#
# Launched UNDER the shared GPU lease, holding BOTH cards for the whole sequence:
#   /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 2 -- bash tools/run_tp2_window.sh
# (-n 2 = both cards, required for TP=2. The lease blocks until both are free, then runs this.)
#
# Runs ONE container for the whole escalation (pay image/cache warmup once) and drives
# tools/tp_serve_probe.py, which boots each server, greedy-generates, and diffs TP1 vs TP2.
#
# Triton cache: the concurrent `titans` training run mounts the SHARED
# .triton-cache-combined RW, and our TP=2 GDN kernels compile NEW shapes (conv_dim 4096,
# v_heads 16) — so we mount the shared cache READ-ONLY and COPY it to a container-local
# writable dir. Warm hits for unchanged kernels; new kernels written to the throwaway copy;
# the shared production cache cannot be corrupted.
set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p tools/tp2_results
echo "[run_tp2_window] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"

docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python \
  --entrypoint bash vllm22-w4a8:combined -lc '
    set -e
    source /app/.venv/bin/activate
    echo "[setup] copying warm Triton cache (isolated, RO source -> writable copy) ..."
    mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
    echo "[setup] installing server deps ..."
    pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil
    echo "[setup] rocm devices visible to torch:"
    python -c "import torch; print(\"  cuda.is_available=\", torch.cuda.is_available(), \"device_count=\", torch.cuda.device_count())"
    python /engine/tools/tp_serve_probe.py
  '
rc=$?
echo "[run_tp2_window] container exited rc=$rc; results in tools/tp2_results/"
exit $rc
