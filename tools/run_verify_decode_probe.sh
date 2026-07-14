#!/usr/bin/env bash
# Verify(prefill/extend-kernel) vs decode(decode-kernel) logit-divergence probe on RXF ZAYA. Runs the
# probe with the M-invariant dense_gemm fix ON (MINV=1) and OFF (MINV=0) so the divergence is attributed
# to the GEMM fix vs the residual (attention / fp8-KV / MoE). Lean image + LIVE kernels mount.
#   gpu-lease -n 1 -- bash tools/run_verify_decode_probe.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/ZAYA1-8B-RXF-h32}"
PROMPT="${PROMPT:-Explain in one sentence why the sky is blue.}"
N="${N:-24}"
CNAME="${LEASE_NAME:-verify-decode-probe}-probe"
trap 'docker rm -f "$CNAME" >/dev/null 2>&1 || true' EXIT INT TERM
echo "[vdprobe] HIP=${HIP_VISIBLE_DEVICES:-unset} kernels=$(git -C /home/pat/code/rdna4-hip-kernels rev-parse --short HEAD 2>/dev/null)"
docker run --rm --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 -e MINISGL_KV_FP8=1 -e MINISGL_MOE_SCATTER=0 \
  -e MODEL="$MODEL" -e PROMPT="$PROMPT" -e N="$N" \
  -v "$PWD":/engine \
  -v /home/pat/code/rdna4-hip-kernels:/kernels:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash minisgl-rdna4:lean -lc '
    set -uo pipefail
    source /app/.venv/bin/activate 2>/dev/null || true
    export PYTHONPATH=/kernels/_kernels:/engine/python:/engine
    for MINV in 1 0; do
      echo "======================= MINISGL_MINV_GEMM=$MINV ======================="
      MINISGL_MINV_GEMM=$MINV MINISGL_ATTN_HIP=1 MINISGL_TAIL_HIP=1 \
        python /engine/tools/verify_decode_logit_probe.py --model "$MODEL" --prompt "$PROMPT" --n "$N"
    done
  '
echo "[vdprobe] exited rc=$?"
