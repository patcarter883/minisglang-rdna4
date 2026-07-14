#!/usr/bin/env bash
# Persistent ZAYA-RXF serve with the re-distilled DFlash-CCA drafter (spec-decode ON). Launch under a
# 1-card lease; leaves the server up + reachable on $HOSTPORT for real use:
#   nohup gpu-lease -n 1 --name zaya-serve -- bash tools/run_zaya_dflash_serve.sh > tools/_zaya_serve.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/ZAYA1-8B-RXF-h32}"
DRAFT_HOST="${DRAFT_HOST:-/home/pat/code/_models/ZAYA1-8B-DFlash-CCA-5L-minv-ep4}"
NUM_DRAFT="${NUM_DRAFT:-4}"; HOSTPORT="${HOSTPORT:-1919}"; MEMRATIO="${MEMRATIO:-0.82}"; DP_SIZE="${DP_SIZE:-1}"
CNAME="${LEASE_NAME:-zaya-serve}-serve"
trap 'docker rm -f "$CNAME" >/dev/null 2>&1 || true' EXIT INT TERM
echo "[serve] HIP=${HIP_VISIBLE_DEVICES:-unset} draft=$DRAFT_HOST num_draft=$NUM_DRAFT -> host port $HOSTPORT"
docker run --rm --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 -e MINISGL_KV_FP8=1 -e MINISGL_MOE_SCATTER=0 \
  -e MINISGL_ATTN_HIP=1 -e MINISGL_TAIL_HIP=1 \
  -p "$HOSTPORT":1919 \
  -v "$PWD":/engine \
  -v /home/pat/code/rdna4-hip-kernels:/kernels:ro \
  -v "$DRAFT_HOST":/draft:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash minisgl-rdna4:lean -lc '
    set -uo pipefail
    source /app/.venv/bin/activate 2>/dev/null || true
    export PYTHONPATH=/kernels/_kernels:/engine/python:/engine
    exec python -m minisgl --model "'"$MODEL"'" --host 0.0.0.0 --port 1919 \
      --attention-backend hip --page-size 16 --tensor-parallel-size 1 --data-parallel-size "'"$DP_SIZE"'" --disable-pynccl \
      --cache-type naive --cuda-graph-max-bs 0 --memory-ratio "'"$MEMRATIO"'" \
      --spec-algorithm dflash --spec-draft-model-path /draft --spec-num-draft "'"$NUM_DRAFT"'" \
      --max-running-requests 32
  '
echo "[serve] exited rc=$?"
