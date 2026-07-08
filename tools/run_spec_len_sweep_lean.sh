#!/usr/bin/env bash
# LEAN-IMAGE spec-length sweep launcher. Runs spec_len_sweep_lean.sh inside minisgl-rdna4:lean
# (the exact serving image), TP=2 -> both cards. Launch UNDER the shared lease:
#   MODEL=... TAG=... CONFIGS="none:0:0 mtp:1:0 ..." gpu-lease -n 2 -- bash tools/run_spec_len_sweep_lean.sh
# Forwards the lease device env (LEASE_*/HIP_/ROCR_ — accept either) into the container.
set -uo pipefail
cd "$(dirname "$0")/.."
HIP="${HIP_VISIBLE_DEVICES:-${LEASE_HIP_DEVICES:-0}}"
ROCR="${ROCR_VISIBLE_DEVICES:-${LEASE_ROCR_DEVICES:-0}}"
echo "[run_spec_len_sweep_lean] HIP=$HIP ROCR=$ROCR MODEL=${MODEL:-?} TAG=${TAG:-out}"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP" -e ROCR_VISIBLE_DEVICES="$ROCR" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e MODEL="${MODEL:?set MODEL}" -e TP="${TP:-2}" -e ATTN="${ATTN:-auto}" \
  -e DRAFT="${DRAFT:-thoughtworks/GLM-4.7-Flash-Eagle3}" \
  -e MEMRATIO="${MEMRATIO:-0.82}" -e MAXRUN="${MAXRUN:-4}" -e MAXTOK="${MAXTOK:-256}" \
  -e CONFIGS="${CONFIGS:?set CONFIGS}" -e TAG="${TAG:-out}" \
  -e GRAPH_SPEC="${GRAPH_SPEC:-16}" \
  -e MINISGL_NUM_NEXTN="${MINISGL_NUM_NEXTN:-}" -e MINISGL_MTP_LAYERS="${MINISGL_MTP_LAYERS:-}" \
  -e MINISGL_DISABLE_ROPE_INTERLEAVE="${MINISGL_DISABLE_ROPE_INTERLEAVE:-}" \
  -v "$PWD":/engine \
  -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/kernels:/engine/python:/engine \
  --entrypoint bash minisgl-rdna4:lean -lc 'bash /engine/tools/spec_len_sweep_lean.sh'
echo "[run_spec_len_sweep_lean] exited rc=$?"
