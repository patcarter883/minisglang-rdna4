#!/usr/bin/env bash
# Matched A/B for the group-16 by-lane decode GEMV (CONTINUANCE §5 item 2).
#
# rdna4-hip-kernels 785f6a5 lets Int4Fp8GemvLoader take the BYLANE (N-on-lanes) tiling at
# group_size 16, which every NVFP4 shape was previously excluded from — Laguna's fused MoE gemm2 is
# K = moe_intermediate/TP = 256, group 16, and the kernel's own table has K-on-lanes at 43.6 us/MB
# vs BYLANE's 9.4 there.
#
# THE A/B IS ONE ENV VAR ON ONE BINARY. VLLM_W4A8_MOE_G2FUSE_BYLANE=0 forces the K-on-lanes sweep,
# which is byte-for-byte what group-16 got before the change. Same image, same weights, same engine
# source, same graph capture — so nothing but the tiling differs and there is no provenance question
# about which build each leg loaded. (Building two images and comparing them would reintroduce one.)
#
# Qwen is a NO-OP CONTROL, not a second win: Qwen3.6-35B-AWQ is group_size 32, so it already cleared
# the old gate and already runs BYLANE. Its two legs must come out equal; if they do not, the change
# leaked into a shape it should not touch.
#
# Graph-captured (GRAPH_BS=8), true tok/s from usage.completion_tokens (§1's measurement trap).
#
# MUST be invoked UNDER the shared arbiter, which this script does NOT acquire itself:
#     gpu-lease -n 2 -- bash tools/bylane16_ab.sh
set -uo pipefail

WT=/home/pat/code/minisgl-rdna4-specod
NREQ="${NREQ:-1}"                       # set before OUT — `set -u` would abort on the expansion below
OUT=${OUT:-$WT/tools/bylane16_ab_results_n$NREQ.txt}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:bl16}
: > "$OUT"

down() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }

bench() {
  local model=$1
  # NREQ is the point of this harness, not a detail. The fused MoE gemm2's M is the number of tokens
  # in the decode batch, and the isolated crossover is steep in M: at the Laguna g16 shape BYLANE
  # measures 0.85x at M=1, 2.19x at M=8, 3.02x at M=16 (local/bench_g2fuse_bylane.py). A CONC=1 A/B
  # therefore samples the ONE regime where the tiling does not pay. Run this at NREQ=1 and NREQ=8.
  MODEL_ID="$model" NREQ="${NREQ:-1}" python3 - <<'PY'
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE="http://localhost:1919"; MODEL=os.environ["MODEL_ID"]; NREQ=int(os.environ["NREQ"])
PROMPT=("Write a detailed technical explanation of how a B-tree index works, including "
        "insertion, node splitting, and range scans.")
def run(i=0, mt=384):
    # A distinct suffix per stream keeps the requests from sharing a radix-cache prefix, so all NREQ
    # really do decode concurrently instead of collapsing onto one cached sequence.
    txt = PROMPT if NREQ == 1 else f"{PROMPT} (variant {i})"
    b={"model":MODEL,"messages":[{"role":"user","content":txt}],"max_tokens":mt,
       "temperature":0.0,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    d=json.loads(urllib.request.urlopen(r,timeout=900).read())
    return d.get("usage",{}).get("completion_tokens",0), d["choices"][0]["message"]["content"]
def wave(mt=384):
    t=time.perf_counter()
    with ThreadPoolExecutor(NREQ) as ex:
        res=list(ex.map(lambda i: run(i, mt), range(NREQ)))
    return time.perf_counter()-t, sum(n for n,_ in res), res[0][1]
wave(64)                                             # warm
w=[wave() for _ in range(3 if NREQ > 1 else 5)]
tps=sorted(n/t for t,n,_ in w)
label="AGGREGATE tok/s" if NREQ > 1 else "TRUE tok/s"
print(f"  {label} (NREQ={NREQ}): min={tps[0]:.2f} median={tps[len(tps)//2]:.2f} max={tps[-1]:.2f}")
print(f"  tokens={sum(n for _,n,_ in w):.0f} wall={sum(t for t,_,_ in w):.2f}s")
# Coherence is part of the gate: a wrong high-half scale fold degrades output before it degrades
# throughput, and a tok/s-only A/B would call that a win.
print("  sample:", " ".join(w[0][2].split())[:140])
PY
}

leg() {
  local name=$1 model_id=$2; shift 2
  echo "=== $name ===" | tee -a "$OUT"
  down
  ( cd "$WT" && env MINISGL_IMAGE="$IMAGE" "$@" docker compose --profile serve up -d >/dev/null 2>&1 )
  local ok=0
  for _ in $(seq 1 180); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && { ok=1; break; }
    sleep 2
  done
  [ "$ok" = 1 ] || { echo "  FAILED to become ready" | tee -a "$OUT"; \
    ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs --tail 30 2>&1 | tail -30 ) \
      | tee -a "$OUT"; return 1; }
  # PROVENANCE, two independent witnesses. Neither is optional: the first run of this harness read
  # "(unset)" on BOTH legs because `docker exec <c> tr ... < /proc/1/environ` redirects the HOST's
  # PID 1 environ into the exec's stdin — the redirect is evaluated by the outer shell, so it never
  # looked inside the container at all. The quoted `sh -c` below is what actually reads the engine's
  # environment (§6: `docker exec <c> env` shows the exec shell, not PID 1).
  local c; c=$( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
  echo -n "  PID1 BYLANE env: " | tee -a "$OUT"
  docker exec "$c" sh -c "tr '\0' '\n' < /proc/1/environ | grep '^VLLM_W4A8_MOE_G2FUSE_BYLANE='" \
    2>/dev/null | tee -a "$OUT" || echo "(absent)" | tee -a "$OUT"
  # KERNEL-SIDE witness, which is the one that actually settles it. select_gemv_tiling warns exactly
  # once per process when it takes the K-on-lanes sweep at an under-filled K. Laguna's fused gemm2 is
  # K=256, so this line MUST appear on the BYLANE=0 leg and MUST NOT appear on the BYLANE-on leg. An
  # env var proves what the engine was told; this proves which tiling the kernel chose.
  echo -n "  kernel tiling witness (K=256 under-occupancy warning): " | tee -a "$OUT"
  if ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
       | grep -q 'gemv_decode.*WARNING: K=256'; then
    echo "PRESENT -> K-on-lanes" | tee -a "$OUT"
  else
    echo "absent -> BYLANE" | tee -a "$OUT"
  fi
  bench "$model_id" 2>&1 | tee -a "$OUT"; echo | tee -a "$OUT"
}

export TP=2 GRAPH_BS=8 SPEC=none
export NREQ
export CONC="${CONC:-$NREQ}"

echo "## Laguna NVFP4 (group 16) — the shape the change unlocks" | tee -a "$OUT"
export MEM_RATIO=0.96 MINISGL_SPEC_MHA_PAGED=1 MINISGL_SWA_RADIX=1
leg "laguna BYLANE=0 (baseline: K-on-lanes, pre-change behaviour)" poolside/Laguna-XS-2.1-NVFP4 \
    MODEL=laguna VLLM_W4A8_MOE_G2FUSE_BYLANE=0
leg "laguna BYLANE on (new default)"                                poolside/Laguna-XS-2.1-NVFP4 \
    MODEL=laguna

# The Qwen control is a statement about a shape the change must NOT touch, which one concurrency
# establishes; SKIP_QWEN=1 skips it on repeat runs at other NREQ.
if [ "${SKIP_QWEN:-0}" != "1" ]; then
  echo "## Qwen3.6-35B AWQ (group 32) — NO-OP CONTROL, both legs must match" | tee -a "$OUT"
  unset MEM_RATIO MINISGL_SPEC_MHA_PAGED MINISGL_SWA_RADIX
  leg "qwen BYLANE=0" cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit MODEL=qwen35b-awq VLLM_W4A8_MOE_G2FUSE_BYLANE=0
  leg "qwen BYLANE on (default)" cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit MODEL=qwen35b-awq
fi

down
echo "results -> $OUT"
