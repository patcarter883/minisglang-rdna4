#!/usr/bin/env bash
# Validate the dense mmq_fp8_gemm_silu ENGINE wiring end-to-end: serve GLM-4.7-Flash-AWQ (W4A8 MoE with a
# QUANTIZED merged gate_up SHARED EXPERT — the fused-silu target) TP=2 under GRAPH CAPTURE in the lean image
# with the FUSION w4a8 .so + MINISGL_DENSE_FUSED_SILU=1, confirm the fused op fires (engaged marker), then
# coherence-check greedy answers. Base model (no spec) to isolate the shared-expert path.
#   gpu-lease -n 2 --timeout 900 -- bash tools/glm_fused_silu_smoke.sh
set -uo pipefail
MODEL="${MODEL:-QuantTrio/GLM-4.7-Flash-AWQ}"
IMG=minisgl-rdna4:lean
WT=/home/pat/code/minisgl-rdna4-fusion
W4A8FIX=/home/pat/code/rdna4-hip-kernels-fusion/w4a8_fp8_wmma/torch-ext
PORT=21979; name=glmfusedsmoke
docker rm -f "$name" >/dev/null 2>&1 || true

echo ">>> serve up (GLM tp2, graph, FUSED dense gate_up+silu)"
docker run -d --name "$name" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e MINISGL_DENSE_FUSED_SILU=1 -e MINISGL_MOE_SCATTER=0 -e MINISGL_KV_FP8=1 \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -v "$W4A8FIX":/opt/w4a8fix -v "$WT":/engine \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -p "${PORT}:1919" \
  --entrypoint bash "$IMG" -lc "
    PYTHONPATH=/opt/w4a8fix:/opt/kernels:/engine/python:/engine /opt/venv/bin/python -m minisgl \
      --model '$MODEL' --host 0.0.0.0 --port 1919 --cache-type radix \
      --tensor-parallel-size 2 --disable-pynccl --graph 8 \
      --memory-ratio 0.80 --max-seq-len-override 40000 --max-running-requests 2" >/dev/null

for _ in $(seq 1 500); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  [ -z "$(docker ps -q -f name="$name")" ] && { echo "  DIED:"; docker logs --tail 60 "$name" 2>&1|tail -60; exit 1; }
  sleep 3
done
echo ">>> ready. coherence check (greedy):"
declare -A KW=( ["The capital of France is"]="paris" ["The chemical symbol for gold is"]="au"
                ["Water is made of hydrogen and"]="oxygen" ["The largest planet in our solar system is"]="jupiter"
                ["Two plus two equals"]="4|four" )
bad=0
for p in "${!KW[@]}"; do
  a=$(curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":48,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c 'import sys,json;m=json.load(sys.stdin)["choices"][0]["message"];print((m.get("content") or m.get("reasoning_content") or "").replace(chr(10)," ").lower())' 2>/dev/null)
  if echo "$a" | grep -qiE "${KW[$p]}"; then echo "  OK   [$p] -> ${a:0:70}"; else echo "  MISS [$p] (want ${KW[$p]}) -> ${a:0:90}"; bad=$((bad+1)); fi
done
echo ">>> engaged markers (fused dense gate_up+silu must fire):"
docker logs "$name" 2>&1 | grep -iE "mmq_fp8_gemm_silu|graphs captured|Capturing CUDA" | sort -u | head -6 | sed 's/^/    /'
docker rm -f "$name" >/dev/null 2>&1
echo ">>> GLM FUSED SILU SMOKE: $([ $bad = 0 ] && echo COHERENT || echo "INCOHERENT ($bad)")"
exit $bad
