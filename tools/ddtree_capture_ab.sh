#!/usr/bin/env bash
# DDTree tree-verify GRAPH CAPTURE validation (lean image, 35B MXFP4, TP=2, fp8 KV + fp8 DFlash).
# Three configs, each rebooted:
#   1. none            — no-spec baseline (lossless reference + tok/s floor)
#   2. dflash          — plain DFlash, LINEAR verify (already graph-captured; the ~103 tok/s bar)
#   3. dflash+ddtree   — DFlash DDTree draft-TREE verify UNDER GRAPH CAPTURE (this change)
# Confirms: (a) boot log shows "DDTREE-verify graphs captured" + "ddtree-verify GRAPH REPLAY engaged"
# (tree verify runs CAPTURED not eager), (b) sustained decode tok/s on a long deterministic gen,
# (c) losslessness (none == dflash == ddtree byte-identical on terminating prompts).
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
python -c "import moe_hip,tail_hip,gdn_hip,attn_hip,attn_decode,attn_prefill_paged;print('[setup] hip pkgs OK')" || exit 1

MODEL="${MODEL:-pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4}"
DRAFT="${DRAFT:-z-lab/Qwen3.6-35B-A3B-DFlash}"
TP="${TP:-2}"; PORT="${PORT:-21088}"; MEMRATIO="${MEMRATIO:-0.82}"
MAXRUN="${MAXRUN:-2}"; NUM_DRAFT="${NUM_DRAFT:-7}"; GRAPHBS="${GRAPHBS:-8}"; LONGTOK="${LONGTOK:-1800}"
# DDTree knobs (tree budget -> tree_qlen=budget+1 captured; mask width cap; captured bs cap).
DDBUDGET="${DDBUDGET:-24}"; DDMAXCTX="${DDMAXCTX:-4096}"; DDMAXBS="${DDMAXBS:-2}"; DDTOPK="${DDTOPK:-8}"
OUTDIR=/engine/tools

SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=algo(none|dflash) $2=ddtree(0|1) ; sets $LOG
  local algo="$1" ddtree="$2"
  LOG="$OUTDIR/ddtree_cap.${algo}.dd${ddtree}.log"
  local spec="" env_extra="MINISGL_MOE_SCATTER=0 MINISGL_KV_FP8=1"
  local pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
  if [ "$algo" != "none" ]; then
    spec="--spec-algorithm $algo --spec-num-draft $NUM_DRAFT --spec-draft-model-path $DRAFT"
    env_extra="$env_extra MINISGL_SPEC_DEBUG=1 MINISGL_DFLASH_QUANT=fp8 MINISGL_DFLASH_FULLCTX=1"
    if [ "$ddtree" = "1" ]; then
      env_extra="$env_extra MINISGL_DFLASH_DDTREE=1 MINISGL_DDTREE_BUDGET=$DDBUDGET \
        MINISGL_DDTREE_TOPK=$DDTOPK MINISGL_DDTREE_MAXCTX=$DDMAXCTX MINISGL_DDTREE_MAXBS=$DDMAXBS"
    fi
  fi
  echo "[boot $algo ddtree=$ddtree] -> $LOG"
  setsid env PYTHONPATH=/opt/kernels:/engine/python:/engine $env_extra python -m minisgl \
    --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --graph "$GRAPHBS" $pynccl \
    --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" --attention-backend hip $spec > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 600); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot $algo dd=$ddtree] DIED:"; tail -50 "$LOG"; return 1; }
    grep -q "Traceback (most recent call last)" "$LOG" && { echo "CRASH:"; tail -50 "$LOG"; return 1; }
    sleep 3
  done; echo "timeout"; tail -50 "$LOG"; return 1
}

lossless_probe(){ PORT=$PORT OUT=$1 python - <<'PY'
import json,os,urllib.request
PORT=os.environ["PORT"]; OUT=os.environ["OUT"]
prompts=["What is the capital of France? Answer in one short sentence.",
         "Name the first four planets from the Sun, comma-separated.",
         "Write one sentence about why the sky appears blue.",
         "What is 12 times 12? Give just the number."]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":160,"enable_thinking":False,
                     "messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(r,timeout=300))
    res.append({"text":d["choices"][0]["message"]["content"],
                "fr":d["choices"][0].get("finish_reason"),
                "n":(d.get("usage") or {}).get("completion_tokens")})
json.dump(res,open(OUT,"w"))
for i,x in enumerate(res): print(f"  [{i}] fr={x['fr']} n={x['n']} {x['text'][:70]!r}")
PY
}

long_probe(){ PORT=$PORT LONGTOK=$LONGTOK python - <<'PY'
import json,os,time,urllib.request
PORT=os.environ["PORT"]; LONGTOK=int(os.environ["LONGTOK"])
prompt=("Count from 1 to 1500, listing every integer in order separated by commas "
        "(1, 2, 3, 4, ...). Do not skip any number and do not stop early.")
body=json.dumps({"model":"m","temperature":0.0,"max_tokens":LONGTOK,"enable_thinking":False,
                 "messages":[{"role":"user","content":prompt}]}).encode()
r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
# warm one short call first so the tok/s excludes cold-start / first-step alloc.
w=json.dumps({"model":"m","temperature":0.0,"max_tokens":8,"enable_thinking":False,
              "messages":[{"role":"user","content":"hi"}]}).encode()
urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=w,headers={"Content-Type":"application/json"}),timeout=120)
t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=1200)); el=time.time()-t
n=(d.get("usage") or {}).get("completion_tokens") or 0
print(f"LONG toks_per_s={n/el:.2f} completion_tokens={n} elapsed={el:.2f}")
PY
}

run(){ # $1=algo $2=ddtree $3=label
  echo "===== $3 ====="
  boot "$1" "$2" || { echo "[$3] boot FAILED"; return 1; }
  lossless_probe "$OUTDIR/ll.$3.json"
  local L; L=$(long_probe); echo "$L"
  echo "[$3] capture: $(grep -oE 'DDTREE-verify graphs captured|ddtree-verify GRAPH REPLAY engaged.*|spec-verify graphs captured' "$LOG" | sort -u | tr '\n' ' | ')"
  echo "[$3] treelen: $(grep -oE '\[ddtree\] mean tree accept-len=[0-9.]+ over [0-9]+ reqs' "$LOG" | tail -1)"
  echo "[$3] acclen:  $(grep -oE '\[spec\] mean accept-len=[0-9.]+ over [0-9]+ reqs' "$LOG" | tail -1)"
  stop; sleep 2
}

run none   0 baseline
run dflash 0 dflash_linear
run dflash 1 dflash_ddtree

echo "===== LOSSLESSNESS (baseline vs dflash_linear vs dflash_ddtree, byte-exact) ====="
python - "$OUTDIR/ll.baseline.json" "$OUTDIR/ll.dflash_linear.json" "$OUTDIR/ll.dflash_ddtree.json" <<'PY'
import json,sys,os
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
a,b,c=load(sys.argv[1]),load(sys.argv[2]),load(sys.argv[3])
if not(a and b and c): print("  (missing json — a boot failed)"); raise SystemExit
okb=okc=True
for i in range(len(a)):
    mb=a[i]["text"]==b[i]["text"]; mc=a[i]["text"]==c[i]["text"]; okb&=mb; okc&=mc
    print(f"  prompt[{i}] base==dflash:{mb} base==ddtree:{mc} (n {a[i]['n']}/{b[i]['n']}/{c[i]['n']})")
    if not mc: print(f"    base ={a[i]['text']!r}\n    ddtree={c[i]['text']!r}")
print("DFLASH LINEAR LOSSLESS:", "PASS" if okb else "FAIL")
print("DFLASH DDTREE LOSSLESS:", "PASS" if okc else "FAIL")
PY
echo "[done]"
