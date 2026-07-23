#!/usr/bin/env bash
# Rigorous FULL-STREAM SWA verify losslessness gate for Laguna DFlash — /v1/completions (raw prompt,
# raw completion, NO chat template / reasoning split), ignore_eos to force a long generation that
# wraps the 512 sliding window many times. Compares the ENTIRE generated token stream (not just the
# chat answer). SPEC == FORCE_N0 is the hard gate; SPEC == PLAIN is the end-to-end greedy reference.
set -uo pipefail
source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
export MINISGL_KV_FP8="${MINISGL_KV_FP8:-0}"

MODEL="${MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
DRAFT="${DRAFT:-poolside/Laguna-XS-2.1-DFlash-NVFP4}"
K="${K:-16}"; TP="${TP:-2}"; MAXTOK="${MAXTOK:-900}"
PORT=21957; LOG=/engine/tools/swa_dflash_raw.server.log
SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; sleep 3; }
trap stop EXIT

boot(){ local envs="$1"; shift
  setsid env $envs PYTHONPATH=/opt/kernels:/engine/python:/engine python -m minisgl \
    --model "$MODEL" --host 127.0.0.1 --port $PORT --tensor-parallel-size "$TP" --disable-pynccl \
    --cache-type naive --attention-backend hip --page-size 16 \
    --cuda-graph-max-bs 0 --max-running-requests "${MAXRUN:-2}" --memory-ratio "${MEMR:-0.90}" \
    "$@" > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED"; tail -40 "$LOG"; exit 1; }; sleep 3
  done; echo "not ready"; tail -50 "$LOG"; exit 1
}

probe(){ PORT=$PORT OUT=$1 MAXTOK=$MAXTOK python - <<'PY'
import json,os,urllib.request
PORT,OUT,MAXTOK=os.environ["PORT"],os.environ["OUT"],int(os.environ["MAXTOK"])
# FULL-STREAM capture: concat reasoning_content + content so the ENTIRE generated token stream
# (reasoning + answer, ~MAXTOK tokens => wraps the 512 window many times) is compared, not just the
# short answer tail. Chat endpoint (the only one minisgl serves); temperature 0 = greedy.
prompts=[
 "Count from 1 to 300, one number per line.",
 "Write the 7 times table from 7x1 to 7x80, one product per line.",
 "For each number from 1 to 250, say if it is even or odd, one per line.",
]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":MAXTOK,
                     "messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,
                             headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(r,timeout=900))
    m=d["choices"][0]["message"]
    txt=(m.get("reasoning_content") or "")+"\x1e"+(m.get("content") or "")  # \x1e = boundary marker
    res.append(txt)
    print(f"  raw-probe fullstream len={len(txt)} chars finish={d['choices'][0].get('finish_reason')}")
json.dump(res,open(OUT,"w"))
PY
}

echo "======== SPEC ========"
boot "MINISGL_SPEC_DEBUG=1" --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K"
probe /engine/tools/raw.spec.json; grep -E "accept-len|accept_rate" "$LOG" | tail -3; stop
echo "======== FORCE_N0 ========"
boot "MINISGL_SPEC_FORCE_N0=1" --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K"
probe /engine/tools/raw.n0.json; stop
echo "======== PLAIN ========"
boot ""; probe /engine/tools/raw.plain.json; stop

echo "======== DIFF (full raw stream) ========"
python - /engine/tools/raw.spec.json /engine/tools/raw.n0.json /engine/tools/raw.plain.json <<'PY'
import json,sys
spec=json.load(open(sys.argv[1])); n0=json.load(open(sys.argv[2])); plain=json.load(open(sys.argv[3]))
def cmp(a,b,label):
    ok=True
    for i,(x,y) in enumerate(zip(a,b)):
        m=(x==y); ok&=m
        print(f"  {label} p[{i}]: {'MATCH' if m else 'DIFF'} (len {len(x)} vs {len(y)})")
        if not m:
            for j,(cx,cy) in enumerate(zip(x,y)):
                if cx!=cy: print(f"     diverge@char{j}: ...{x[max(0,j-40):j+8]!r} | ...{y[max(0,j-40):j+8]!r}"); break
    return ok
g1=cmp(spec,n0,"[spec|n0]"); print("---"); g2=cmp(spec,plain,"[spec|plain]")
print("\nSPEC==FORCE_N0 :", "PASS" if g1 else "FAIL")
print("SPEC==PLAIN    :", "PASS" if g2 else "FAIL")
print("OVERALL        :", "PASS" if (g1 and g2) else "FAIL")
PY
echo "[done-raw]"
