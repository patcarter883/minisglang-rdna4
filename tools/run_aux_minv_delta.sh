#!/usr/bin/env bash
# Measure the DFlash-aux shift between the pre-dense_gemm capture-time path (MINV=0/rocBLAS) and the
# current serve (MINV=1/dense_gemm) -> tells us if the stored OPD corpus is still on-policy.
#   gpu-lease -n 1 -- bash tools/run_aux_minv_delta.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/ZAYA1-8B-RXF-h32}"
PROMPT="${PROMPT:-Explain in one sentence why the sky is blue.}"
CNAME="${LEASE_NAME:-aux-minv}-aux"
trap 'docker rm -f "$CNAME" >/dev/null 2>&1 || true' EXIT INT TERM
echo "[aux] kernels=$(git -C /home/pat/code/rdna4-hip-kernels rev-parse --short HEAD 2>/dev/null)"
docker run --rm --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 -e MINISGL_KV_FP8=1 -e MINISGL_MOE_SCATTER=0 \
  -e MODEL="$MODEL" -e PROMPT="$PROMPT" \
  -v "$PWD":/engine \
  -v /home/pat/code/rdna4-hip-kernels:/kernels:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash minisgl-rdna4:lean -lc '
    set -uo pipefail
    source /app/.venv/bin/activate 2>/dev/null || true
    export PYTHONPATH=/kernels/_kernels:/engine/python:/engine
    MINISGL_MINV_GEMM=1 MINISGL_ATTN_HIP=1 MINISGL_TAIL_HIP=1 \
      python /engine/tools/aux_minv_delta.py --model "$MODEL" --prompt "$PROMPT" --out /engine/tools/_aux_minv1.pt
    MINISGL_MINV_GEMM=0 MINISGL_ATTN_HIP=1 MINISGL_TAIL_HIP=1 \
      python /engine/tools/aux_minv_delta.py --model "$MODEL" --prompt "$PROMPT" --out /engine/tools/_aux_minv0.pt
    python /engine/tools/aux_minv_delta.py --compare /engine/tools/_aux_minv1.pt /engine/tools/_aux_minv0.pt
  '
echo "[aux] exited rc=$?"
