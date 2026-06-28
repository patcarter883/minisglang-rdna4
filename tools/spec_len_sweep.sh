#!/usr/bin/env bash
# Spec-decode length sweep — wall-clock tokens/sec at batch=1 vs --spec-num-draft K, to find the
# optimal draft length and the uplift over the no-spec (CUDA-graph) baseline. Runs INSIDE
# vllm22-w4a8:combined; ONE container re-boots the python server per config (model reload only,
# ~1 min) so we pay image/pip/triton setup once. Driven by run_spec_len_sweep.sh under the lease.
#
# CONFIGS = space-separated  algo:k:seed  tuples. algo=none -> no-spec baseline (CUDA graph ON);
# algo=mtp|eagle3 -> spec (eager, --spec-num-draft k); seed=1 -> MINISGL_SPEC_PREFILL_SEED=1.
# Spec is eager-only (the engine disables graph capture when spec is set), so the baseline (graph ON)
# is the honest "what you'd serve without spec" reference — spec must beat it on net tokens/sec.
set -uo pipefail
source /app/.venv/bin/activate
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
python -c "import gdn_hip, moe_hip, tail_hip, mla_hip; print('[setup] hip pkgs OK')" \
  || { echo '[setup] hip import FAILED'; exit 1; }

MODEL="${MODEL:?set MODEL}"; TP="${TP:-2}"; PORT="${PORT:-21055}"
MEMRATIO="${MEMRATIO:-0.85}"; MAXRUN="${MAXRUN:-4}"; MAXTOK="${MAXTOK:-256}"
ATTN="${ATTN:-auto}"; DRAFT="${DRAFT:-thoughtworks/GLM-4.7-Flash-Eagle3}"
CONFIGS="${CONFIGS:?set CONFIGS=algo:k:seed ...}"; TAG="${TAG:-out}"
RES="/engine/tools/spec_sweep_${TAG}.tsv"; : > "$RES"

SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=algo $2=k $3=seed ; log path echoed via $LOG
  local algo="$1" k="$2" seed="$3"
  LOG="/engine/tools/spec_sweep.${TAG}.${algo}.${k}.${seed}.log"
  local spec="" graph="--graph 16" env_extra="MINISGL_MOE_SCATTER=0"
  local pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
  if [ "$algo" != "none" ]; then
    spec="--spec-algorithm $algo --spec-num-draft $k"; graph="--graph 0"  # spec is eager-only
    [ "$algo" = "eagle3" ] && spec="$spec --spec-draft-model-path $DRAFT"
    env_extra="$env_extra MINISGL_SPEC_DEBUG=1"
    [ "$seed" = "1" ] && env_extra="$env_extra MINISGL_SPEC_PREFILL_SEED=1"
  fi
  setsid env PYTHONPATH=/engine/python:/engine $env_extra python -m minisgl \
    --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" $graph $pynccl \
    --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" \
    --attention-backend "$ATTN" $spec > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 400); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot $algo k=$k seed=$seed] DIED:"; tail -30 "$LOG"; return 1; }
    sleep 3
  done; echo "[boot $algo k=$k seed=$seed] timeout:"; tail -30 "$LOG"; return 1
}

probe(){ PORT=$PORT MAXTOK=$MAXTOK python - <<'PY'
import json,os,time,urllib.request
PORT=os.environ["PORT"]; MAXTOK=int(os.environ["MAXTOK"])
prompt=("Explain in detail how a binary search tree works, including how insertion, lookup, and "
        "in-order traversal are performed, and discuss the average and worst-case time complexity.")
def gen():
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":MAXTOK,
                     "messages":[{"role":"user","content":prompt}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,
                             headers={"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=240)); el=time.time()-t
    n=(d.get("usage") or {}).get("completion_tokens") or len(d["choices"][0]["message"]["content"].split())
    return n, el
gen()  # warmup (prefill caches / first-step settle)
ns=els=0.0
for _ in range(3):
    n,el=gen(); ns+=n; els+=el
print(f"PROBE toks_per_s={ns/els:.2f} completion_tokens={int(ns)} elapsed={els:.2f}")
PY
}

for cfg in $CONFIGS; do
  IFS=: read -r algo k seed <<< "$cfg"
  echo "===== boot $algo k=$k seed=$seed ====="
  if boot "$algo" "$k" "$seed"; then
    out=$(probe); echo "$out"
    tps=$(echo "$out" | sed -n 's/.*toks_per_s=\([0-9.]*\).*/\1/p')
    accline=$(grep -E "\[spec\]" "$LOG" | tail -1)
    eps=$(echo "$accline" | sed -n 's/.*emitted\/step=\([0-9.]*\).*/\1/p')
    acc=$(echo "$accline" | sed -n 's/.*accept_rate=\([0-9.]*\).*/\1/p')
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$algo" "$k" "$seed" "${tps:-NA}" "${eps:-NA}" "${acc:-NA}" >> "$RES"
  else
    printf "%s\t%s\t%s\tBOOT_FAIL\t-\t-\n" "$algo" "$k" "$seed" >> "$RES"
  fi
  stop; sleep 2
done

echo "===== SWEEP RESULTS  model=$MODEL  tag=$TAG ====="
printf "algo\tk\tseed\ttoks/s\temit/step\taccept\n"; cat "$RES"
echo "[done]"
