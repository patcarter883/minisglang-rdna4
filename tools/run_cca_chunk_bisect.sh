#!/usr/bin/env bash
# CCA chunk-boundary bisect: single-pass vs chunked prefill, conv-output (pre-RoPE) vs stored-K.
# Run under a lease:  gpu-lease -n 1 --wait --name cca-bisect -- bash tools/run_cca_chunk_bisect.sh
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[cca-bisect] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset}"
MODEL="${MODEL:-/root/.cache/huggingface/ZAYA1-8B-RXF-h32}"
NTOK="${NTOK:-160}"

run() {
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
    -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 \
    -v "$PWD":/engine \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/pat/code/rdna4-hip-kernels:/kernels \
    -e PYTHONPATH=/kernels/_kernels:/engine/python:/engine \
    --entrypoint bash minisgl-rdna4:lean -lc "$1"
}

echo "[cca-bisect] === single-pass prefill ==="
run "python /engine/tools/cca_chunk_bisect.py --mode single  --model $MODEL --ntok $NTOK --max-extend 8192 --out /engine/tools/_cbz_single.pt"
echo "[cca-bisect] === chunked prefill (max_extend=48) ==="
run "python /engine/tools/cca_chunk_bisect.py --mode chunked --model $MODEL --ntok $NTOK --max-extend 48 --out /engine/tools/_cbz_chunk.pt"
echo "[cca-bisect] === compare ==="
run "python /engine/tools/cca_chunk_bisect.py --compare /engine/tools/_cbz_single.pt /engine/tools/_cbz_chunk.pt"
echo "[cca-bisect] done rc=$?"
