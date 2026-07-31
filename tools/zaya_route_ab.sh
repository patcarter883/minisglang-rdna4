#!/usr/bin/env bash
# zaya_route_ab.sh — matched A/B for the fused ZAYA route (moe_hip.moe_topk_softmax_bias).
#
# WHAT MOVES. ZayaRouter.forward used to spend SEVEN torch dispatches per MoE layer on a 17-float
# row (bf16->fp32 copy, softmax, +balancing_biases, topk, gather, clamp, int64->int32), plus the two
# the MOD blend spends on `(expert_idx != ne).to(dtype)`. ZAYA1-8B has 80 layers of which the 40 ODD
# ones carry a router, so that is 280-360 launches per decode step, each ~2.6 us of essentially pure
# launch latency under graph replay. The fused route policy makes it ONE launch per layer.
#
# THE A/B IS ONE ENV VAR ON ONE BINARY. MINISGL_ZAYA_TORCH_ROUTE=1 takes the original torch chain,
# which is still in the file verbatim; unset takes the fused op. Same image, same mounted source,
# same weights, same graph capture — nothing but the route path differs, so there is no provenance
# question about which build each leg loaded. (Two images would reintroduce one.)
#
# WHY NOT `docker compose --profile serve`. Compose enumerates the environment it forwards, so a new
# MINISGL_* var cannot reach the container without editing docker-compose.yml — a shared file under
# concurrent edit. This replicates the `serve` service's mounts/devices/env exactly and adds the one
# var, touching nothing that is not this file.
#
# MEASUREMENT (the traps this harness exists to avoid):
#  * TRUE tok/s from usage.completion_tokens on a NON-STREAMING request. Counting SSE chunks measures
#    the transport.
#  * NREQ=1 AND NREQ=8. A dispatch-count lever is a per-STEP cost, so its share of the step shrinks
#    as the batch does more work per step; one concurrency cannot characterise it. (A real +20% lever
#    on this box first read "flat, 0.0%" from a CONC=1-only A/B.)
#  * COHERENCE, not just tok/s. A wrong router degrades text before it degrades throughput, and top-1
#    routing has no second expert to average the error away.
#  * TP=1, which is where the 16.9 ms step-time baseline this lever was sized against was measured.
#
# MUST be invoked UNDER the shared arbiter, which this script does NOT acquire itself:
#     NREQ=1 gpu-lease -n 1 -- bash tools/zaya_route_ab.sh
#     NREQ=8 gpu-lease -n 1 -- bash tools/zaya_route_ab.sh
set -uo pipefail

WT=/home/pat/code/minisgl-rdna4-specod
NREQ="${NREQ:-1}"
OUT=${OUT:-$WT/tools/zaya_route_ab_results_n$NREQ.txt}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:zroute}
PORT=${PORT:-1919}
NAME=zaya-route-ab
: > "$OUT"

down() { docker rm -f "$NAME" >/dev/null 2>&1; }
trap down EXIT

serve_up() {   # serve_up <extra -e args...>
  docker run -d --name "$NAME" \
    --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
    --ipc host --shm-size 16gb -p "$PORT":1919 \
    -v "$WT":/engine \
    -v "${HF_HOME:-/home/pat/.cache/huggingface}":/root/.cache/huggingface \
    -v "${MODELS_DIR:-/home/pat/models}":/models \
    -v "${DRAFT_DIR:-/home/pat/code/_models}":/drafts:ro \
    -w /engine \
    -e PYTHONPATH=/opt/kernels:/engine/python:/engine \
    -e HF_HUB_OFFLINE=0 -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e ROCR_VISIBLE_DEVICES="${LEASE_ROCR_DEVICES:-${ROCR_VISIBLE_DEVICES:-0}}" \
    -e HIP_VISIBLE_DEVICES="${LEASE_HIP_DEVICES:-0}" \
    -e MINISGL_KV_FP8=1 -e MINISGL_SPEC_MHA_PAGED=1 \
    -e MINISGL_MOE_ASYNC_AR=1 -e MINISGL_MOE_ASYNC_AR_MIN_TOKENS=512 \
    -e MINISGL_CUSTOM_AG_EP=1 -e MINISGL_CUSTOM_AR_EP=0 \
    -e MINISGL_GRAMMAR_THINK_GATE=1 -e MINISGL_MOE_W4A16=0 \
    -e MINISGL_TIDAR_MIX_BETA=1.0 -e MINISGL_SPEC_ONDEVICE=0 \
    -e MODEL="${MODEL:-zaya}" -e SPEC=none -e TP="${TP:-1}" -e DP=1 -e EP=0 \
    -e CONC="${CONC:-$NREQ}" -e GRAPH_BS="${GRAPH_BS:-8}" \
    "$@" \
    --entrypoint bash "$IMAGE" -lc 'exec /engine/tools/serve.sh' >/dev/null
}

bench() {
  NREQ="$NREQ" PORT="$PORT" python3 - <<'PY'
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE=f"http://localhost:{os.environ['PORT']}"
# Ask the server what it is serving rather than reconstructing serve.sh's model table here — the
# zaya arm resolves an ALIAS to a path (/models/ZAYA1-8B-fp8) and a stale copy would 404.
MODEL=json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=10).read())["data"][0]["id"]
NREQ=int(os.environ["NREQ"])
PROMPT=("Write a detailed technical explanation of how a B-tree index works, including "
        "insertion, node splitting, and range scans.")
def run(i=0, mt=384):
    # A distinct suffix per stream keeps the requests from sharing a radix-cache prefix, so all NREQ
    # really do decode concurrently instead of collapsing onto one cached sequence.
    txt = PROMPT if NREQ == 1 else f"{PROMPT} (variant {i})"
    b={"model":MODEL,"messages":[{"role":"user","content":txt}],"max_tokens":mt,
       "temperature":0.0,"stream":False}          # NON-streaming: usage.completion_tokens is truth
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=900).read())
    return d.get("usage",{}).get("completion_tokens",0), d["choices"][0]["message"]["content"]
def wave(mt=384):
    t=time.perf_counter()
    with ThreadPoolExecutor(NREQ) as ex:
        res=list(ex.map(lambda i: run(i, mt), range(NREQ)))
    return time.perf_counter()-t, sum(n for n,_ in res), res[0][1]
wave(64)                                             # warm (capture + radix)
w=[wave() for _ in range(3 if NREQ > 1 else 5)]
tps=sorted(n/t for t,n,_ in w)
label="AGGREGATE tok/s" if NREQ > 1 else "TRUE tok/s"
print(f"  {label} (NREQ={NREQ}): min={tps[0]:.2f} median={tps[len(tps)//2]:.2f} max={tps[-1]:.2f}")
print(f"  tokens={sum(n for _,n,_ in w):.0f} wall={sum(t for t,_,_ in w):.2f}s")
print("  sample:", " ".join(w[0][2].split())[:200])
PY
}

leg() {
  local name=$1; shift
  echo "=== $name ===" | tee -a "$OUT"
  down; sleep 2
  serve_up "$@"
  local ok=0
  for _ in $(seq 1 240); do
    curl -s --max-time 3 "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { ok=1; break; }
    docker ps -q -f name="$NAME" | grep -q . || break
    sleep 2
  done
  [ "$ok" = 1 ] || { echo "  FAILED to become ready" | tee -a "$OUT"; \
                     docker logs "$NAME" 2>&1 | tail -40 | tee -a "$OUT"; return 1; }

  # PROVENANCE, two independent witnesses.
  #  1. What the engine was TOLD. The quoted `sh -c` is load-bearing: `docker exec c tr ... <
  #     /proc/1/environ` redirects the HOST's pid 1 environ into the exec's stdin, so it never looks
  #     inside the container. Note also that a present-but-EMPTY var must read as OFF, which is why
  #     the model-side parse strips and compares against ("", "0", "false", "no").
  echo -n "  PID1 MINISGL_ZAYA_TORCH_ROUTE: " | tee -a "$OUT"
  docker exec "$NAME" sh -c "tr '\0' '\n' < /proc/1/environ | grep '^MINISGL_ZAYA_TORCH_ROUTE=' \
    || echo '(absent -> fused)'" 2>/dev/null | tee -a "$OUT"
  #  2. Which path actually RAN, from the engine's own one-shot engagement log. This fires during
  #     GRAPH CAPTURE, so its presence also proves the op survived capture rather than falling back.
  echo -n "  engage witness: " | tee -a "$OUT"
  docker logs "$NAME" 2>&1 | grep -oE 'hip-engage\] (moe_hip\.moe_topk_softmax_bias|zaya\.torch_route[^ ]*)' \
    | head -1 | tee -a "$OUT" || echo "(none)" | tee -a "$OUT"

  bench 2>&1 | tee -a "$OUT"
  echo | tee -a "$OUT"
}

echo "## ZAYA1-8B-fp8 TP=${TP:-1}, SPEC=none, graph-captured — fused route vs the torch chain" \
  | tee -a "$OUT"
leg "BASELINE: torch route chain (MINISGL_ZAYA_TORCH_ROUTE=1)" -e MINISGL_ZAYA_TORCH_ROUTE=1
leg "CANDIDATE: fused moe_topk_softmax_bias (default, no env)"
down
echo "results -> $OUT"
