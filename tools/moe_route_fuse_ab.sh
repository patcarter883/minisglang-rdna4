#!/usr/bin/env bash
# Matched A/B for the FUSED sigmoid+bias MoE route (rdna4-hip-kernels df6b49d,
# moe_hip.moe_topk_sigmoid_bias) on the two models that run it: Laguna-XS-2.1 and GLM-4.7-Flash.
#
# WHAT IS BEING MEASURED. `LagunaSparseBlock._route` and `GLMSparseBlock._noaux_tc` (n_group=1) were
# TWELVE torch kernels per sparse layer over a single [T, E] row — 39 x 12 = 468 dispatches/step on
# Laguna (21.9% of ALL decode dispatches, ~1.9 us each of essentially pure launch latency) and
# 46 x 12 = 552 on GLM. They are now ONE launch. Scope is the ROUTE only: moe_align is still a
# separate call, so the honest claim is ~1020 route dispatches/step -> ~85, not ~1020 -> ~85 total.
#
# HISTORICAL — THIS HARNESS NO LONGER RUNS AS-IS (see the guard below). Kept for the recorded result
# (moe_route_fuse_ab_results_n{1,8}.txt) and as the template for the next single-binary A/B.
#
# The A/B was ONE ENV VAR ON ONE BINARY: MINISGL_MOE_ROUTE_TORCH_BASELINE=1 made
# quant/kernels.py:moe_route_sigmoid_bias run the verbatim twelve-op torch chain instead of the op.
# Same image, same weights, same engine source, same graph capture — nothing but the route differed,
# so there was no provenance question about which build each leg loaded (a second image would have
# reintroduced exactly that question). That shim was REMOVED for merge: the fused route is
# unconditionally on, and an env-selectable path nothing serves is unexercised code that rots.
#
# RUN IT AT NREQ=1 **AND** NREQ=8. This is a pure launch-latency lever, so its size tracks how
# gap-bound the step is. A CONC=1-only A/B on this box already misread a real +20.1% lever as
# "flat, 0.0%".
#
#     NREQ=1 gpu-lease -n 2 -- bash tools/moe_route_fuse_ab.sh
#     NREQ=8 gpu-lease -n 2 -- bash tools/moe_route_fuse_ab.sh
#     ONLY=glm NREQ=1 gpu-lease -n 2 -- bash tools/moe_route_fuse_ab.sh
#
# MUST be invoked UNDER the shared arbiter, which this script does NOT acquire itself.
set -uo pipefail

WT=/home/pat/code/minisgl-rdna4-specod

# GUARD. Without the torch-baseline shim BOTH legs run the FUSED route and this script reports a
# perfectly plausible ~0% — a false negative that reads exactly like a real measurement. Fail loudly
# instead. (This is the same class of silent-identical-legs bug that made an earlier microbench
# report a uniform 1.0x by comparing BYLANE against itself.)
if ! grep -q "_MOE_ROUTE_TORCH_BASELINE" "$WT/python/minisgl/quant/kernels.py" 2>/dev/null; then
  cat >&2 <<'MSG'
[moe_route_fuse_ab] ABORT: the torch-baseline shim is gone from quant/kernels.py, so both legs would
run the FUSED route and this harness would report ~0%. Restore _MOE_ROUTE_TORCH_BASELINE and
_moe_route_sigmoid_bias_torch (see the commit that removed them) plus the compose passthrough before
re-running. Recorded result of the original run: tools/moe_route_fuse_ab_results_n{1,8}.txt
MSG
  exit 2
fi
NREQ="${NREQ:-1}"                       # set before OUT — `set -u` would abort on the expansion below
OUT=${OUT:-$WT/tools/moe_route_fuse_ab_results_n$NREQ.txt}
IMAGE=${MINISGL_IMAGE:-minisgl-rdna4:route}
: > "$OUT"

down() { ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve down >/dev/null 2>&1 ); }

bench() {
  local model=$1
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
       "temperature":0.0,"stream":False}          # NON-streaming: true tok/s is usage.completion_tokens
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
# COHERENCE IS PART OF THE GATE. A wrong router degrades text long before it degrades throughput,
# and Laguna's gate deliberately uses minv_linear precisely because a 1-ulp logit shift can flip
# expert selection under prefix caching. Read the text; do not just diff the numbers.
print("  sample:", " ".join(w[0][2].split())[:180])
PY
}

leg() {
  local name=$1 model_id=$2; shift 2
  echo "=== $name ===" | tee -a "$OUT"
  down
  ( cd "$WT" && env MINISGL_IMAGE="$IMAGE" "$@" docker compose --profile serve up -d >/dev/null 2>&1 )
  local ok=0
  for _ in $(seq 1 240); do
    curl -s --max-time 3 http://localhost:1919/v1/models >/dev/null 2>&1 && { ok=1; break; }
    sleep 2
  done
  [ "$ok" = 1 ] || { echo "  FAILED to become ready" | tee -a "$OUT"; \
    ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs --tail 40 2>&1 | tail -40 ) \
      | tee -a "$OUT"; return 1; }
  local c; c=$( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve ps -q serve )
  # PROVENANCE, THREE INDEPENDENT WITNESSES. None is optional.
  # (1) the ENGINE's own environment. The quoted `sh -c` is load-bearing: the obvious
  #     `docker exec <c> tr ... < /proc/1/environ` form redirects the HOST's PID 1 into the exec's
  #     stdin, because the redirect is evaluated by the OUTER shell — it never looks in the
  #     container at all. Present-but-empty is NOT an override (atoi("") == 0).
  echo -n "  PID1 baseline env: " | tee -a "$OUT"
  docker exec "$c" sh -c "tr '\0' '\n' < /proc/1/environ | grep '^MINISGL_MOE_ROUTE_TORCH_BASELINE='" \
    2>/dev/null | tee -a "$OUT" || echo "(absent)" | tee -a "$OUT"
  # (2) the IMAGE really carries the kernel (a cached kernel layer presents as a silent wrong
  #     result, not an error).
  echo -n "  image carries the op: " | tee -a "$OUT"
  docker exec "$c" sh -c \
    "grep -c SigmoidBiasScore /opt/rdna4-hip-kernels/moe/moe_rocm/moe_route_core.h" \
    2>/dev/null | tee -a "$OUT" || echo "(absent)" | tee -a "$OUT"
  # (3) THE ONE THAT ACTUALLY SETTLES IT: the engage line the wrapper prints on rank 0 the first
  #     time the op is called (i.e. DURING graph capture). It MUST be present on the fused leg and
  #     absent on the baseline leg. An env var proves what the engine was told; this proves which
  #     route the model ran.
  echo -n "  [hip-engage] moe_topk_sigmoid_bias: " | tee -a "$OUT"
  if ( cd "$WT" && MINISGL_IMAGE="$IMAGE" docker compose --profile serve logs 2>&1 ) \
       | grep -q 'hip-engage.*moe_topk_sigmoid_bias'; then
    echo "PRESENT -> FUSED route fired" | tee -a "$OUT"
  else
    echo "absent -> torch chain" | tee -a "$OUT"
  fi
  bench "$model_id" 2>&1 | tee -a "$OUT"; echo | tee -a "$OUT"
}

# SPEC=none EXPLICITLY on both models: GLM's spec_default is eagle3 (serve.sh:59) and serve.sh:71
# fills an unset OR EMPTY SPEC from it, so omitting this benchmarks GLM under EAGLE3 against Laguna
# under plain decode. GRAPH_BS covers the concurrency — eager-only is never done.
export TP=2 SPEC=none
export NREQ
export CONC="${CONC:-$NREQ}"
export GRAPH_BS="${GRAPH_BS:-8}"

ONLY="${ONLY:-both}"

if [ "$ONLY" != "glm" ]; then
  echo "## Laguna-XS-2.1-NVFP4 — E=256, top_k=8, sf=2.5, 39 sparse layers" | tee -a "$OUT"
  # Laguna uniquely gets mem_default=0.85 and swa_hybrid=1 (which exports BOTH MINISGL_SWA_RADIX=1
  # and MINISGL_SPEC_MHA_PAGED=1). serve.sh sets those from the model table; keep them identical
  # across legs by not overriding them at all.
  leg "laguna TORCH BASELINE (12 kernels/layer)" poolside/Laguna-XS-2.1-NVFP4 \
      MODEL=laguna MINISGL_MOE_ROUTE_TORCH_BASELINE=1
  leg "laguna FUSED (1 launch/layer)"            poolside/Laguna-XS-2.1-NVFP4 \
      MODEL=laguna
fi

if [ "$ONLY" != "laguna" ]; then
  # GLM's win should be LARGER than Laguna's (46 sparse layers vs 39) even though its per-route work
  # is smaller — the lever is dispatch COUNT, not route arithmetic.
  echo "## GLM-4.7-Flash-AWQ — E=64, top_k=4, sf=1.8, 46 sparse layers, n_group=1" | tee -a "$OUT"
  leg "glm TORCH BASELINE (12 kernels/layer)" QuantTrio/GLM-4.7-Flash-AWQ \
      MODEL=glm MINISGL_MOE_ROUTE_TORCH_BASELINE=1
  leg "glm FUSED (1 launch/layer)"            QuantTrio/GLM-4.7-Flash-AWQ \
      MODEL=glm
fi

down
echo "results -> $OUT"
