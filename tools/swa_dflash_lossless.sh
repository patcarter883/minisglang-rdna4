#!/usr/bin/env bash
# SWA multi-query VERIFY losslessness gate for Laguna DFlash spec-decode (TP=2, EAGER).
#
# Laguna is is_swa_hybrid: 10 full-attn + 30 sliding (window=512). DFlash stages K+1 verify tokens per
# step; the sliding layers verify them through the SWA extend primitive (_gather_swa_windows +
# _swa_prefill_extend). The widened SWA ring stride (window + num_draft + 1) keeps the speculative
# block in slots disjoint from the live window so a rejected draft never corrupts the next step's gather.
#
# Three greedy serves over the SAME long prompts (>512 generated tokens => exercises ring wraparound):
#   SPEC      : real DFlash multi-token accept   (spec ring stride = W+K+1, extend/verify kernel)
#   FORCE_N0  : same spec staging+verify forward, commit only the bonus (1 tok/step) — isolates the
#               accept/commit/ring-rollback logic from kernel numerics (both go through the verify path)
#   PLAIN     : no spec at all (single-query decode, ring stride = W) — the end-to-end reference
# SPEC == FORCE_N0 (byte-identical) is the HARD gate for the verify/ring logic. SPEC == PLAIN is the
# end-to-end lossless-greedy confirmation.
set -uo pipefail
source /app/.venv/bin/activate 2>/dev/null || source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
# bf16 SWA ring: the extend/verify primitive (_swa_prefill_extend) cats the ring window with the inline
# new-token K/V and runs the DENSE bf16 flash_prefill — an fp8 ring window would dtype-clash there. bf16
# keeps the verify path numerically clean for the correctness gate.
export MINISGL_KV_FP8="${MINISGL_KV_FP8:-0}"

MODEL="${MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
DRAFT="${DRAFT:-poolside/Laguna-XS-2.1-DFlash-NVFP4}"
K="${K:-16}"
TP="${TP:-2}"
MAXTOK="${MAXTOK:-720}"
PORT=21955
LOG=/engine/tools/swa_dflash.server.log
SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; sleep 3; }
trap stop EXIT

boot(){ # $1 = extra env assignments (may be empty), $2... = extra minisgl args
  local envs="$1"; shift
  setsid env $envs PYTHONPATH=/opt/kernels:/engine/python:/engine python -m minisgl \
    --model "$MODEL" --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size "$TP" --disable-pynccl \
    --cache-type naive --attention-backend hip --page-size 16 \
    --cuda-graph-max-bs "${GRAPHBS:-0}" --max-running-requests "${MAXRUN:-2}" --memory-ratio "${MEMR:-0.90}" \
    "$@" > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED"; tail -40 "$LOG"; exit 1; }
    sleep 3
  done
  echo "server not ready"; tail -50 "$LOG"; exit 1
}

probe(){ PORT=$PORT OUT=$1 MAXTOK=$MAXTOK python - <<'PY'
import json,os,urllib.request
PORT,OUT,MAXTOK=os.environ["PORT"],os.environ["OUT"],int(os.environ["MAXTOK"])
# Long, repetitive, deterministic generations — reliably run past the 512 window (ring wraparound).
prompts=[
 "Count from 1 to 250, writing each number on its own line, like:\n1\n2\n3\n",
 "Write the multiplication table for 7, from 7x1 up to 7x60, one product per line.",
 "List the numbers 1 to 200, and for each say whether it is even or odd, one per line.",
 "Repeat the sentence 'The quick brown fox jumps over the lazy dog.' exactly 80 times, numbered.",
]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":MAXTOK,
                     "messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,
                             headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(r,timeout=600))
    txt=d["choices"][0]["message"]["content"]
    res.append(txt)
    print(f"  probe len={len(txt)} chars  finish={d['choices'][0].get('finish_reason')}")
json.dump(res,open(OUT,"w"))
PY
}

echo "============ 1/3  SPEC (DFlash multi-token accept) ============"
boot "MINISGL_SPEC_DEBUG=1" \
  --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K"
probe /engine/tools/ll.spec.json
echo "[spec] accept-len log lines:"; grep -E "\[spec\]|accept-len|acc=" "$LOG" | tail -6
echo "[mem] memory / KV-pool / SWA-ring lines:"
grep -iE "Free memory|KV cache|num_pages|SWA ring|draft model|KV pool|recurrent|reserv|tokens" "$LOG" | tail -20
cp "$LOG" /engine/tools/swa_dflash.spec.log
stop

echo "============ 2/3  FORCE_N0 (1 tok/step, same verify kernel) ============"
boot "MINISGL_SPEC_FORCE_N0=1" \
  --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K"
probe /engine/tools/ll.n0.json
stop

echo "============ 3/3  PLAIN (no spec, single-query decode) ============"
boot ""
probe /engine/tools/ll.plain.json
stop

echo "============ DIFF ============"
python - /engine/tools/ll.spec.json /engine/tools/ll.n0.json /engine/tools/ll.plain.json <<'PY'
import json,sys
spec=json.load(open(sys.argv[1])); n0=json.load(open(sys.argv[2])); plain=json.load(open(sys.argv[3]))
def cmp(a,b,label):
    ok=True
    for i,(x,y) in enumerate(zip(a,b)):
        m=(x==y); ok&=m
        tag='MATCH' if m else 'DIFF'
        print(f"  {label} prompt[{i}]: {tag}  (len {len(x)} vs {len(y)})")
        if not m:
            for j,(cx,cy) in enumerate(zip(x,y)):
                if cx!=cy:
                    print(f"      diverge@char{j}: spec=...{x[max(0,j-30):j+10]!r}  other=...{y[max(0,j-30):j+10]!r}")
                    break
    return ok
print("--- SPEC vs FORCE_N0 (HARD gate: accept/commit/ring logic) ---")
g1=cmp(spec,n0,"[spec|n0]")
print("--- SPEC vs PLAIN (end-to-end lossless-greedy) ---")
g2=cmp(spec,plain,"[spec|plain]")
print()
print("SPEC==FORCE_N0 :", "PASS" if g1 else "FAIL")
print("SPEC==PLAIN    :", "PASS" if g2 else "FAIL")
print("OVERALL        :", "PASS" if (g1 and g2) else "FAIL")
PY
echo "[done]"
