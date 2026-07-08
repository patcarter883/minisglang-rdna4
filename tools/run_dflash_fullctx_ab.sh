#!/usr/bin/env bash
# Launcher for the DFlash full-context A/B (LEAN image + this worktree's source; kernels from /opt/kernels).
# Run under the shared lease:  gpu-lease -n 1 -- bash -c '.../run_dflash_fullctx_ab.sh'
set -uo pipefail
cd "$(dirname "$0")/.."
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e K="${K:-15}" -e GRAPH="${GRAPH:-4}" -e MAXREQ="${MAXREQ:-4}" \
  -v "$PWD":/engine \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  --entrypoint bash minisgl-rdna4:lean /engine/tools/dflash_fullctx_ab.sh
