#!/usr/bin/env bash
# Validate the gdn_decode_gated ENGINE wiring end-to-end: serve Qwen3.5-4B (GDN-dense) TP=1 under graph
# capture with the fusion gdn_hip .so + MINISGL_GDN_FUSED_NORM=1, confirm the fused op is actually loaded,
# then coherence-check greedy answers. A wiring bug (bad z/arg) would produce gibberish; bit-exactness of
# the op means correct wiring == coherent, model-quality output.  gpu-lease -n 1 -- bash <this>
set -uo pipefail
MODEL="${MINISGL_MODEL:-cyankiwi/Qwen3.5-4B-AWQ-BF16-INT4}"
IMG=minisgl-rdna4:lean
WT=/home/pat/code/minisgl-rdna4-fusion
GDNPKG=/home/pat/code/rdna4-hip-kernels-fusion/gdn/torch-ext   # contains gdn_hip/ (fusion .so)
PORT=19193; name=gdnsmoke
docker rm -f "$name" >/dev/null 2>&1 || true

echo ">>> confirm fusion gdn_hip exposes gdn_decode_gated (no GPU):"
docker run --rm -v "$GDNPKG":/opt/gdnfix --entrypoint bash "$IMG" \
  -lc 'PYTHONPATH=/opt/gdnfix /opt/venv/bin/python -c "import gdn_hip; print(\"  has gdn_decode_gated:\", hasattr(gdn_hip,\"gdn_decode_gated\"))"' 2>/dev/null

echo ">>> serve up (tp1, graph, FUSED gdn norm)"
docker run -d --name "$name" \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" -e ROCR_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES" \
  -e HF_HUB_OFFLINE=1 -e MINISGL_GDN_FUSED_NORM=1 \
  -v "$GDNPKG":/opt/gdnfix -v "$WT":/engine \
  -v /home/pat/.cache/huggingface:/root/.cache/huggingface -p "${PORT}:1919" \
  --entrypoint bash "$IMG" -lc "
    PYTHONPATH=/opt/gdnfix:/opt/kernels:/engine/python:/engine /opt/venv/bin/python -m minisgl \
      --model '$MODEL' --host 0.0.0.0 --port 1919 --cache-type radix --attention-backend hip \
      --page-size 16 --tp 1 --cuda-graph-max-bs 8 --max-running-requests 8 --memory-ratio 0.8" >/dev/null
for _ in $(seq 1 150); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  [ -z "$(docker ps -q -f name="$name")" ] && { echo "  DIED:"; docker logs --tail 40 "$name" 2>&1|tail -40; exit 1; }
  sleep 4
done
echo ">>> ready. coherence check (greedy):"
declare -A KW=( ["The capital of France is"]="paris" ["The chemical symbol for gold is"]="au"
                ["Water is made of hydrogen and"]="oxygen" ["The largest planet is"]="jupiter"
                ["Two plus two equals"]="4|four" )
bad=0
for p in "${!KW[@]}"; do
  a=$(curl -sf "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":40,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c 'import sys,json;m=json.load(sys.stdin)["choices"][0]["message"];print((m.get("content") or m.get("reasoning_content") or "").replace(chr(10)," ").lower())' 2>/dev/null)
  if echo "$a" | grep -qiE "${KW[$p]}"; then echo "  OK   [$p] -> ${a:0:70}"; else echo "  MISS [$p] (want ${KW[$p]}) -> ${a:0:90}"; bad=$((bad+1)); fi
done
docker logs "$name" 2>&1 | grep -iE "gdn_decode_gated|error|traceback" | head -4 | sed 's/^/    log: /'
docker rm -f "$name" >/dev/null 2>&1
echo ">>> GDN FUSED SMOKE: $([ $bad = 0 ] && echo COHERENT || echo "INCOHERENT ($bad)")"
exit $bad
