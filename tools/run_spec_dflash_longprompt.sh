#!/usr/bin/env bash
# Launcher for spec_dflash_longprompt_lean.sh inside minisgl-rdna4:lean, TP=2. Mounts THIS WORKTREE.
#   gpu-lease -n 2 -- bash <worktree>/tools/run_spec_dflash_longprompt.sh
set -uo pipefail
WORKTREE="$(cd "$(dirname "$0")/.." && pwd)"
HIP="${HIP_VISIBLE_DEVICES:-${LEASE_HIP_DEVICES:-0}}"
ROCR="${ROCR_VISIBLE_DEVICES:-${LEASE_ROCR_DEVICES:-0}}"
echo "[run_dflash_longprompt] WORKTREE=$WORKTREE HIP=$HIP ROCR=$ROCR"
TRITON_COPY="$WORKTREE/.triton-dflash-persist"; mkdir -p "$TRITON_COPY"
cp -an /home/pat/code/vllm-gfx1201/.triton-cache-combined/. "$TRITON_COPY/" 2>/dev/null || true
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP" -e ROCR_VISIBLE_DEVICES="$ROCR" -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e MODEL="${MODEL:-pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4}" \
  -e DRAFT="${DRAFT:-z-lab/Qwen3.6-35B-A3B-DFlash}" \
  -e TP="${TP:-2}" -e MEMRATIO="${MEMRATIO:-0.82}" -e NUM_DRAFT="${NUM_DRAFT:-8}" \
  -e GENTOK="${GENTOK:-200}" -e PCTX="${PCTX:-3500}" \
  -v "$WORKTREE":/engine -v "$TRITON_COPY":/root/.triton \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/kernels:/engine/python:/engine \
  --entrypoint bash minisgl-rdna4:lean -lc 'bash /engine/tools/spec_dflash_longprompt_lean.sh'
echo "[run_dflash_longprompt] exited rc=$?"
