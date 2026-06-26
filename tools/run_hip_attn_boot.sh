#!/usr/bin/env bash
# Boot a dense model with the fully Triton-free HIP attention backend (native HIP flash-prefill +
# paged flash-decode). Mounts both kernel worktrees so `import attn_hip` / `import attn_decode`
# resolve. BACKEND env selects 'hip' (default) or 'auto' (triton_rdna4) for an A/B token-diff.
set -euo pipefail
LEASE=/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh
ENGINE=/home/pat/code/minisgl-rdna4
PREFILL_WT=/home/pat/code/vllm-gfx1201-attn-hip
DECODE_WT=/home/pat/code/vllm-gfx1201-attn-decode
BACKEND="${BACKEND:-hip}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
"$LEASE" -n 1 -- bash -c '
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable \
    --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
    -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
    -v '"$ENGINE"':/engine \
    -v '"$PREFILL_WT"':/mnt/attn_prefill \
    -v '"$DECODE_WT"':/mnt/attn_decode \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    --entrypoint bash vllm22-w4a8:combined -lc "
      source /app/.venv/bin/activate
      pip install -q msgpack pyzmq prompt_toolkit accelerate 2>&1 | tail -1
      export PYTHONPATH=/engine/python:/engine:/mnt/attn_prefill:/mnt/attn_decode
      python /engine/tools/boot_smoke.py --model '"$MODEL"' --attn-backend '"$BACKEND"' \
        --max-tokens 32 --max-running-req 16 \
        --prompt \"The history of the Roman Empire spans many centuries. It began as a small\""'
