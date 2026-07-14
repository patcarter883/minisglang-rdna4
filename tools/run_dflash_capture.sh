#!/usr/bin/env bash
# Capture minisgl-ZAYA aux+tokens for DFlash drafter RE-distillation (on-policy fidelity fix).
# Boots a single-card ZAYA serve with the capture hook on (MINISGL_ZAYA_CAPTURE_DIR) + prefix cache
# OFF, runs the teacher-forcing driver, then stops. Base output is config-independent (TP=1 == DP+EP,
# byte-identical), so TP=1 capture is representative and avoids the EP serve-loop.
#
# Launch under the shared lease (single card), foreground so the lease frees on exit:
#   gpu-lease -n 1 -- bash tools/run_dflash_capture.sh
#
# Output: seedbuf_*.pt under $CAPDIR (host). Then re-distill (Phase 3):
#   python /home/pat/code/vllm-gfx1201-zaya-dflash/zaya/dflash/train_cca_drafter.py \
#     --init /home/pat/code/_models/ZAYA1-8B-DFlash-CCA-5L-init \
#     --seed-dir <CAPDIR> --out /home/pat/code/_models/ZAYA1-8B-DFlash-CCA-5L-minisgl --epochs 14
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-/models/ZAYA1-8B-fp8}"
CAPDIR="${CAPDIR:-/home/pat/code/_dflash_capture_minisgl}"
PROMPTS="${PROMPTS:-/home/pat/code/_capture_prompts.txt}"
GENTOK="${GENTOK:-256}"
WORKERS="${WORKERS:-16}"
mkdir -p "$CAPDIR"
echo "[capture] HIP=${HIP_VISIBLE_DEVICES:-unset} ROCR=${ROCR_VISIBLE_DEVICES:-unset} -> $CAPDIR"
docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 \
  -e MINISGL_ZAYA_CAPTURE_DIR=/capture \
  -e MINISGL_ZAYA_W8A16="${W8A16:-1}" -e MINISGL_MOE_SCATTER=0 \
  -e GENTOK="$GENTOK" -e WORKERS="$WORKERS" -e MODEL="$MODEL" \
  -v "$PWD":/engine \
  -v /home/pat/models:/models:ro \
  -v /home/pat/code/_big:/big:ro \
  -v /home/pat/code/rdna4-hip-kernels:/kernels:ro \
  -v "$CAPDIR":/capture \
  -v "$PROMPTS":/prompts.txt:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash "${MINISGL_IMAGE:-minisgl-rdna4:lean}" -lc '
    set -uo pipefail
    source /app/.venv/bin/activate 2>/dev/null || true
    export PYTHONPATH=/kernels/_kernels:/engine/python:/engine
    LOG=/engine/tools/dflash_capture.server.log
    # Boot ZAYA TP=1 with prefix cache OFF (teacher-force must be a full prefill).
    setsid env MINISGL_ZAYA_CAPTURE_DIR=/capture MINISGL_ZAYA_W8A16="'"${W8A16:-1}"'" \
      MINISGL_MOE_SCATTER=0 MINISGL_ATTN_HIP=1 MINISGL_TAIL_HIP=1 \
      python -m minisgl --model "$MODEL" --host 127.0.0.1 --port 1919 \
        --cache-type naive --attention-backend hip --page-size 16 --tp 1 --disable-pynccl \
        --cuda-graph-max-bs 0 --memory-ratio 0.85 > "$LOG" 2>&1 &
    SRV=$!
    for _ in $(seq 1 400); do
      python -c "import urllib.request;urllib.request.urlopen(\"http://127.0.0.1:1919/v1/models\",timeout=3)" 2>/dev/null && break
      kill -0 "$SRV" 2>/dev/null || { echo "[capture] server DIED:"; tail -60 "$LOG"; exit 1; }
      sleep 3
    done
    echo "[capture] server ready; running driver"
    python /engine/tools/capture_dflash_data.py --ports 1919 --workers "$WORKERS" \
      --gen-tokens "$GENTOK" --prompts-file /prompts.txt
    echo "[capture] driver done; files:"; ls -la /capture | tail -8
    kill -TERM -- -"$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null || true
  '
echo "[capture] exited rc=$?; seedbuf in $CAPDIR"
