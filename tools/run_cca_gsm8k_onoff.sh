#!/usr/bin/env bash
# CCA prefix-cache validation: radix ON (recurrent radix reuse) vs OFF (naive), same model+prompts.
# With the M-invariant GEMM fix, reuse must match naive -> 0 answer divergence.
# gpu-lease -n 1 --wait --name cca-gsm8k -- bash tools/run_cca_gsm8k_onoff.sh
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/root/.cache/huggingface/ZAYA1-8B-RXF-h32}"
run() {
  docker run --rm \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}" \
    -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 \
    -e MINISGL_KV_FP8="${MINISGL_KV_FP8:-0}" \
    -v "$PWD":/engine -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
    -v /home/pat/code/rdna4-hip-kernels:/kernels \
    -e PYTHONPATH=/kernels/_kernels:/engine/python:/engine \
    --entrypoint bash minisgl-rdna4:lean -lc "$1"
}
# graph=0 for determinism; default max_extend so the shared few-shot prefix is REUSED via recurrent radix.
echo "[gsm8k] radix ON  (recurrent radix)"; run "python /engine/tools/cca_gsm8k_offline.py --radix on  --graph 0 --model $MODEL --out /engine/tools/_g_on_fp8.pt"
echo "[gsm8k] radix OFF (naive)";            run "python /engine/tools/cca_gsm8k_offline.py --radix off --graph 0 --model $MODEL --out /engine/tools/_g_off_fp8.pt"
echo "[gsm8k] compare ON vs OFF";            run "python /engine/tools/cca_gsm8k_offline.py --compare /engine/tools/_g_on_fp8.pt /engine/tools/_g_off_fp8.pt"
echo "[gsm8k] done rc=$?"
