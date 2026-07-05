#!/usr/bin/env bash
# NORTH-STAR v0 launcher (cudagraph on CCA decode). Under the lease: gpu-lease -n 1 -- bash tools/run_cca_graph_v0.sh
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[run_cca_graph_v0] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e ORIG="${ORIG:-/models/ZAYA1-8B-fp8}" -e GENTOK="${GENTOK:-256}" -e NRUNS="${NRUNS:-3}" \
  -e MEMRATIO="${MEMRATIO:-0.85}" -e GRAPHS="${GRAPHS:-0 8}" \
  -v "$PWD":/engine \
  -v /home/pat/models:/models:ro \
  -v /home/pat/code/_big:/big:ro \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/triton-ro:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/engine/python:/engine \
  --entrypoint bash vllm22-w4a8:combined /engine/tools/cca_graph_v0.sh
echo "[run_cca_graph_v0] exited rc=$?"
