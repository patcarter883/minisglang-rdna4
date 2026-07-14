#!/usr/bin/env bash
# gpu-lease -n 1 --wait --name cca-seamp -- bash tools/run_cca_seam_probe.sh
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
echo "[seamp] single"; run "python /engine/tools/cca_seam_probe.py --mode single  --model $MODEL --ntok $NTOK --max-extend 8192 --out /engine/tools/_seamp_single.pt"
echo "[seamp] chunked"; run "python /engine/tools/cca_seam_probe.py --mode chunked --model $MODEL --ntok $NTOK --max-extend 48 --out /engine/tools/_seamp_chunk.pt"
echo "[seamp] compare"; run "python /engine/tools/cca_seam_probe.py --compare /engine/tools/_seamp_single.pt /engine/tools/_seamp_chunk.pt"
echo "[seamp] done rc=$?"
