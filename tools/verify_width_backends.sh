#!/usr/bin/env bash
# Multi-width verify capture across the OTHER backbones.
#
# Laguna+DFlash (where the A/B is measured) is SWA/MHA over the RDNA4 backend with no recurrent
# state, so it exercises exactly ONE of the four per-width code paths. This drives the rest:
#
#   qwen35b-awq + MTP    GDN hybrid   -> GDNVerifyGraphCapture per-layer conv/ssm scratch, now
#                                        allocated at Qmax and sliced [:Q]  (K=4  -> ladder [2,4])
#   glm + EAGLE3         MLA          -> MLABackend per-width kbound + seq_idx, where
#                                        `seq_idx = arange(T)//qlen` is NOT a slice of the
#                                        max-width pattern and needs one buffer per width
#                                        (K=6  -> ladder [3,6])
#
# PASS = boots, captures the ladder, generates coherent text, and logs `verify-graph eager=0`.
#
#     gpu-lease -n 2 -- bash tools/verify_width_backends.sh
set -uo pipefail

WT=/home/pat/code/minisgl-rdna4-propose
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:lean}
OUT=${OUT:-$WT/tools/verify_width_backends.txt}
: > "$OUT"

down() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }
trap down EXIT INT TERM

wait_ready() { for _ in $(seq 1 500); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

leg() {  # leg <model> <spec> <mem_ratio> <extra-env...>
  local model=$1 spec=$2 mem=$3
  echo "=== MODEL=$model SPEC=$spec ===" | tee -a "$OUT"
  down
  ( cd "$WT" && env MINISGL_IMAGE="$IMAGE" MODEL="$model" SPEC="$spec" TP=2 CONC=4 GRAPH_BS=8 \
      MEM_RATIO="$mem" MINISGL_SPEC_TIMING=1 docker compose --profile serve up -d >/dev/null 2>&1 )
  if ! wait_ready; then
    echo "  FAILED to become ready" | tee -a "$OUT"
    ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs --tail 40 2>&1 ) \
      | tail -40 | tee -a "$OUT"; down; return 1
  fi
  python3 - 2>&1 <<'PY' | tee -a "$OUT"
import json, time, urllib.request
BASE="http://localhost:1919"
M=json.loads(urllib.request.urlopen(f"{BASE}/v1/models",timeout=30).read())["data"][0]["id"]
b={"model":M,"messages":[{"role":"user","content":"Write a Python function to reverse a linked list, then explain it."}],
   "max_tokens":700,"temperature":0.0,"seed":1234,"stream":False}
r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                         headers={"Content-Type":"application/json"})
t=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=1800).read()); w=time.perf_counter()-t
n=d["usage"]["completion_tokens"]
print(f"  TOKENS {n}  TPS {n/w:.2f}")
print("  TEXT:", (d["choices"][0]["message"]["content"] or "")[:120].replace("\n"," "))
PY
  ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
    | grep -aE "ADAPTIVE verify width|Capturing spec-verify|\[spec\] mean|\[spec-timing\]|Traceback|RuntimeError" \
    | sed 's/^[^ ]* *| *//' | tail -8 | tee -a "$OUT"
  down; echo "" | tee -a "$OUT"
}

leg qwen35b-awq mtp    0.80
leg glm         eagle3 0.80
echo "== BACKENDS COMPLETE ==" | tee -a "$OUT"
