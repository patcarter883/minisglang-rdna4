#!/usr/bin/env bash
# Phase 4 GPU window — TP=2 serve bring-up + parity probe (4-0 / 4-3 / 4-4).
#
# Launched UNDER the shared GPU lease, holding BOTH cards for the whole sequence:
#   gpu-lease -n 2 -- bash tools/run_tp2_window.sh
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

# Args (if any) are forwarded to tp_serve_probe.py as config-name filters, e.g.
#   gpu-lease -n 1 -- bash tools/run_tp2_window.sh tp1   # single-card TP=1 validation
#   gpu-lease -n 2 -- bash tools/run_tp2_window.sh       # full TP=2 escalation
PROBE_ARGS="$*"

mkdir -p tools/tp2_results
echo "[run_tp2_window] HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} probe_args='${PROBE_ARGS}'"

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
    echo "[setup] copying warm Triton cache (isolated, RO source -> writable copy) ..."
    mkdir -p /root/.triton && cp -a /triton-ro/. /root/.triton/ 2>/dev/null || true
    echo "[setup] installing server deps ..."
    pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil
    echo "[setup] building gdn_hip HIP kernels (the GDN forward now imports torch.ops.gdn_hip) ..."
    ( cd /engine/gdn_hip && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace >/tmp/gdn_build.log 2>&1 \
      && python -c "import gdn_hip; print(\"  gdn_hip loaded OK\")" ) \
      || { echo "[setup] gdn_hip build/load FAILED:"; tail -25 /tmp/gdn_build.log; exit 1; }
    echo "[setup] rocm devices visible to torch:"
    python -c "import torch; print(\"  cuda.is_available=\", torch.cuda.is_available(), \"device_count=\", torch.cuda.device_count())"
    python /engine/tools/tp_serve_probe.py '"$PROBE_ARGS"'
  '
rc=$?
echo "[run_tp2_window] container exited rc=$rc; results in tools/tp2_results/"
exit $rc
