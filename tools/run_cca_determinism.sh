#!/usr/bin/env bash
# Determinism control: run single-pass prefill TWICE, compare. Isolates run-to-run nondeterminism
# (atomic reductions) from genuine chunk-boundary divergence.
# gpu-lease -n 1 --wait --name cca-det -- bash tools/run_cca_determinism.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/ZAYA1-8B-RXF-h32}"
NTOK="${NTOK:-160}"
run() {
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
    -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 \
    -v "$PWD":/engine -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/pat/code/rdna4-hip-kernels:/kernels \
    -e PYTHONPATH=/kernels/_kernels:/engine/python:/engine \
    --entrypoint bash minisgl-rdna4:lean -lc "$1"
}
echo "[det] single run A"; run "python /engine/tools/cca_chunk_bisect.py --mode single --model $MODEL --ntok $NTOK --max-extend 8192 --out /engine/tools/_cbz_A.pt"
echo "[det] single run B"; run "python /engine/tools/cca_chunk_bisect.py --mode single --model $MODEL --ntok $NTOK --max-extend 8192 --out /engine/tools/_cbz_B.pt"
echo "[det] compare A vs B (both single-pass -> any Δ is nondeterminism)"; run "python /engine/tools/cca_chunk_bisect.py --compare /engine/tools/_cbz_A.pt /engine/tools/_cbz_B.pt"
echo "[det] done rc=$?"
