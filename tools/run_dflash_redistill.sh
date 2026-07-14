#!/usr/bin/env bash
# Re-distill the ZAYA CCA-DFlash drafter on the FRESH on-policy minisgl-RXF seedbuf (the fidelity fix).
# train_cca_drafter.py is pure-torch (no vLLM/HIP), so it runs in the lean image; base embed/lm_head
# weight comes from the RXF checkpoint (--zaya; embed is un-quantized). Launch under a 1-card lease:
#   gpu-lease -n 1 -- bash tools/run_dflash_redistill.sh
set -uo pipefail
cd "$(dirname "$0")/.."
INIT="${INIT:-/models_rw/ZAYA1-8B-DFlash-CCA-5L-init}"
OUT="${OUT:-/models_rw/ZAYA1-8B-DFlash-CCA-5L-minisgl-rxf}"
NUM_SPEC="${NUM_SPEC:-4}"; EPOCHS="${EPOCHS:-14}"; BATCH="${BATCH:-512}"
echo "[redistill] HIP=${HIP_VISIBLE_DEVICES:-unset} -> $OUT (num_spec=$NUM_SPEC epochs=$EPOCHS)"
CNAME="${LEASE_NAME:-dflash-redistill}-redistill"
trap 'docker rm -f "$CNAME" >/dev/null 2>&1 || true' EXIT INT TERM
docker run --rm --name "$CNAME" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -e INIT="$INIT" -e OUT="$OUT" -e NUM_SPEC="$NUM_SPEC" -e EPOCHS="$EPOCHS" -e BATCH="$BATCH" \
  -v /home/pat/code/vllm-gfx1201-zaya-dflash:/trainrepo:ro \
  -v /home/pat/code/_models:/models_rw \
  -v "${SEEDBUF_HOST:-/home/pat/code/_dflash_capture_minisgl}":/seedbuf:ro \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface:ro \
  --entrypoint bash "${MINISGL_IMAGE:-minisgl-rdna4:lean}" -lc '
    set -uo pipefail
    source /app/.venv/bin/activate 2>/dev/null || true
    python /trainrepo/zaya/dflash/train_cca_drafter.py \
      --init "$INIT" --out "$OUT" --seed-dir /seedbuf \
      --zaya /root/.cache/huggingface/ZAYA1-8B-RXF-h32 \
      --num-spec "$NUM_SPEC" --epochs "$EPOCHS" --batch "$BATCH"
  '
echo "[redistill] exited rc=$?; drafter -> ${OUT/\/models_rw//home/pat/code/_models}"
