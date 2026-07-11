#!/usr/bin/env bash
# 2-card serve benchmark window (Task: serving matrix). Launch UNDER the shared lease:
#   gpu-lease -n 2 -- bash tools/run_bench_window.sh
# TP=1 single-card models: gpu-lease -n 1 -- env TP=1 MODEL=Qwen/Qwen3.5-4B bash tools/run_bench_window.sh
# Thin wrapper: forwards the lease's device env into the container and runs _bench_inner.sh there.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_bench_window] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset} MODEL=${MODEL:-default} TP=${TP:-2}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  --add-host host.docker.internal:host-gateway \
  -p "${METRICS_HOST_PORT:-1919}:1919" \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e PORT=1919 \
  -e MODEL="${MODEL:-cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit}" -e TP="${TP:-2}" \
  -e MEMRATIO="${MEMRATIO:-0.82}" -e MAXRUN="${MAXRUN:-24}" \
  -e GRAPH="${GRAPH:-16}" -e MOE_SCATTER="${MOE_SCATTER:-0}" -e BENCH_M="${BENCH_M:-1,2,4,8,16}" \
  -e ATTN="${ATTN:-hip}" \
  -e SPEC="${SPEC:-}" -e SPEC_K="${SPEC_K:-}" -e DFLASH_MODEL="${DFLASH_MODEL:-}" -e EP="${EP:-}" -e MINISGL_EP_DBG="${MINISGL_EP_DBG:-}" \
  -e MINISGL_SPEC_DEBUG="${MINISGL_SPEC_DEBUG:-}" -e MINISGL_SPEC_FORCE_N0="${MINISGL_SPEC_FORCE_N0:-}" \
  -e MINISGL_SPEC_PROPOSE_GRAPH="${MINISGL_SPEC_PROPOSE_GRAPH:-}" -e MINISGL_MTP_MAX_CTX="${MINISGL_MTP_MAX_CTX:-}" \
  -e MINISGL_SPEC_TIMING="${MINISGL_SPEC_TIMING:-}" -e MINISGL_SPEC_PROPOSE_NOCAPTURE="${MINISGL_SPEC_PROPOSE_NOCAPTURE:-}" \
  -e MINISGL_GRAPH_RESERVE_MARGIN_GB="${MINISGL_GRAPH_RESERVE_MARGIN_GB:-}" -e MINISGL_PROFILE="${MINISGL_PROFILE:-}" -e MINISGL_PROFILE_SKIP="${MINISGL_PROFILE_SKIP:-}" -e MINISGL_PROFILE_STEPS="${MINISGL_PROFILE_STEPS:-}" \
  -e SKIP_TRITON_COPY="${SKIP_TRITON_COPY:-1}" \
  -v "$PWD":/engine \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/kernels:/engine/python:/engine \
  --entrypoint bash "${MINISGL_IMAGE:-minisgl-rdna4:lean}" /engine/tools/_bench_inner.sh
echo "[run_bench_window] exited rc=$?"
