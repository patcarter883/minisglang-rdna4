#!/usr/bin/env bash
# OUTER wrapper: run the Qwen3.6-27B TP=2 coherence validation inside minisgl-rdna4:lean.
#   gpu-lease -n 2 -- bash tools/run_q27_lean.sh
# Mounts THIS worktree at /engine (per CLAUDE.md source-isolation), forwards the lease device pair.
set -uo pipefail
cd "$(dirname "$0")/.."
HIP="${HIP_VISIBLE_DEVICES:-${LEASE_HIP_DEVICES:-0}}"
ROCR="${ROCR_VISIBLE_DEVICES:-${LEASE_ROCR_DEVICES:-0}}"
echo "[run_q27_lean] HIP=$HIP ROCR=$ROCR PWD=$PWD"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP" -e ROCR_VISIBLE_DEVICES="$ROCR" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e MODEL="${MODEL:-cyankiwi/Qwen3.6-27B-AWQ-INT4}" -e TP="${TP:-2}" -e ATTN="${ATTN:-hip}" \
  -e GRAPH="${GRAPH:-0}" -e MEMRATIO="${MEMRATIO:-0.82}" -e MAXRUN="${MAXRUN:-8}" \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/kernels:/engine/python:/engine \
  --entrypoint bash minisgl-rdna4:lean -lc 'bash /engine/tools/q27_validate_lean.sh'
echo "[run_q27_lean] exited rc=$?"
