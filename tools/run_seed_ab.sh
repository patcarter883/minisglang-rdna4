#!/usr/bin/env bash
# 2-card A/B window for the MTP prompt-prefill draft-KV seed. Launch UNDER the shared lease:
#   gpu-lease -n 2 --timeout 900 -- bash tools/run_seed_ab.sh
# Mounts THIS WORKTREE (isolated snapshot) as /engine and runs seed_ab_inner.sh in the lean image.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
echo "[run_seed_ab] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset} REPO=$REPO"
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
  -e MEMRATIO="${MEMRATIO:-0.80}" -e MAXRUN="${MAXRUN:-4}" -e GRAPH="${GRAPH:-4}" \
  -e ATTN="${ATTN:-hip}" -e SPEC_K="${SPEC_K:-4}" -e CONC="${CONC:-4}" \
  -e DECODE_TOKENS="${DECODE_TOKENS:-256}" -e MODE="${MODE:-all}" \
  -e MINISGL_SPEC_PROPOSE_GRAPH="${MINISGL_SPEC_PROPOSE_GRAPH:-}" -e MINISGL_MTP_MAX_CTX="${MINISGL_MTP_MAX_CTX:-}" \
  -e SKIP_TRITON_COPY=1 \
  -v "$REPO":/engine \
  -v /home/pat/models:/models \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH="${KERNEL_PYPATH:-/opt/kernels}:/engine/python:/engine" \
  --entrypoint bash "${MINISGL_IMAGE:-minisgl-rdna4:lean}" /engine/tools/seed_ab_inner.sh
echo "[run_seed_ab] exited rc=$?"
