#!/usr/bin/env bash
# Capture minisgl-ZAYA aux+tokens for DFlash drafter RE-distillation (on-policy fidelity fix).
# Boots a single-card ZAYA serve with the capture hook on (MINISGL_ZAYA_CAPTURE_DIR) + prefix cache
# OFF (naive), runs the teacher-forcing driver, then stops.
#
# QUANT MUST MIRROR THE DEPLOYED SERVE — this is the whole point. ZAYA's MoE quant is precision-
# sensitive (the router argmaxes a balancing-biased softmax -> near-ties flip per quant -> different
# tokens -> different aux); a drafter captured under a different quant than it serves under is OOD.
# The DEPLOYED serve is the RXF W4A8 checkpoint (ZAYA1-8B-RXF-h32: rxf-pack-quantized, iq4_nl group32,
# hadamard32, int8 acts; routed to _RXFMoEMethod), KV_FP8=1, MOE_SCATTER=0, fp32 router. Defaults here
# MATCH it. (RXF ZAYA works now — the old "RXF serves garbage" handoff is obsolete.) W8A16 is an fp8-
# only knob and is IGNORED on the RXF path.
#
# TOPOLOGY: this captures at TP=1 for one card. The serve runs DP=2+EP; EP's all_reduce reconstructs the
# exact per-token MoE result and the router (expert SELECTION) is on the replicated gate, so TP=1 aux is
# representative (fp8 base was byte-identical TP=1 vs DP+EP). For a byte-exact topology match, capture on
# the DP=2 serve instead: add MINISGL_ZAYA_CAPTURE_DIR + --cache-type naive to the `zaya` compose profile
# (the _forward capture hook fires in the EP loop's prefill too; both replicas dump, slot=uid separates).
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
MODEL="${MODEL:-/root/.cache/huggingface/ZAYA1-8B-RXF-h32}"   # the DEPLOYED RXF serve checkpoint
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
  -e MINISGL_KV_FP8="${KV_FP8:-1}" -e MINISGL_MOE_SCATTER=0 \
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
    setsid env MINISGL_ZAYA_CAPTURE_DIR=/capture MINISGL_KV_FP8="'"${KV_FP8:-1}"'" \
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
