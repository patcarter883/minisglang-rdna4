#!/usr/bin/env bash
# Prompt-prefill draft-KV seed validation (MINISGL_SPEC_PREFILL_SEED) on GLM-4.7-Flash (glm4_moe_lite,
# AWQ, MLA) at TP=2. Runs INSIDE vllm22-w4a8:combined under a 2-card lease (via run_prefill_seed_window.sh).
#
# For the chosen draft head (ALGO=mtp|eagle3) it boots TWICE — seed OFF, then seed ON — and checks:
#   1. OUTPUT INVARIANCE: seed-on output == seed-off output BYTE-FOR-BYTE. Both are lossless greedy
#      (the verify forward commits the target's argmax regardless of the drafts), so seeding must
#      change ONLY which drafts get proposed/accepted, NEVER the committed tokens. A DIFF means the
#      seed corrupted the verify path — a real bug.
#   2. ACCEPTANCE LIFT: the seed-on accept_rate should be >= seed-off (the early tokens now draft
#      with full prompt context instead of a cold/empty draft KV).
set -uo pipefail
source /app/.venv/bin/activate
pip install -q msgpack pyzmq prompt_toolkit accelerate fastapi uvicorn pydantic starlette psutil 2>&1 | tail -1
PYTHONPATH=/engine/python:/engine python -c "import mla_hip, moe_hip, tail_hip, swiglu_hip; print('[setup] hip pkgs OK')" \
  || { echo '[setup] hip import FAILED'; exit 1; }

ALGO="${ALGO:-mtp}"
MODEL="${MODEL:-QuantTrio/GLM-4.7-Flash-AWQ}"; PORT=21977; LOG=/engine/tools/seed_${ALGO}.server.log
DRAFT="${DRAFT:-thoughtworks/GLM-4.7-Flash-Eagle3}"
SPECARGS="--spec-algorithm $ALGO --spec-num-draft ${KDRAFT:-4}"
[ "$ALGO" = "eagle3" ] && SPECARGS="$SPECARGS --spec-draft-model-path $DRAFT"

SRV=""; stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 20); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1 = extra env assignment
  setsid env PYTHONPATH=/engine/python:/engine MINISGL_MOE_SCATTER=0 $1 python -m minisgl \
    --model "$MODEL" --tensor-parallel-size 2 --port $PORT --graph 0 --disable-pynccl \
    --memory-ratio 0.85 --max-running-requests 4 $SPECARGS > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "[boot] DIED:"; tail -60 "$LOG"; exit 1; }; sleep 3
  done; echo "[boot] not ready:"; tail -60 "$LOG"; exit 1
}
probe(){ PORT=$PORT OUT=$1 python - <<'PY'
import json,os,urllib.request
PORT,OUT=os.environ["PORT"],os.environ["OUT"]
# LONG prompts + SHORT generation: the prompt-prefill seed lifts EARLY-token acceptance, so a long
# prompt (more seeded context) with a short generation (early tokens dominate) is where the lever pays.
ctx=("You are reading a short report. Paris is the capital of France and its largest city. The Seine "
     "river runs through it. The Louvre museum is located there and the Eiffel Tower was completed in "
     "1889. The city is divided into twenty arrondissements arranged in a spiral. ")
prompts=[
 ctx+"Question: What is the capital of France? Answer in one word:",
 ctx+"Question: Which river runs through the city? Answer in one word:",
 ctx+"Question: In what year was the Eiffel Tower completed? Answer:",
 ctx+"Question: How many arrondissements is the city divided into? Answer:",
 ctx+"Question: Which famous museum is in the city? Answer:",
 ctx+"Summarize the report in one short sentence:",
 ctx+"List three facts from the report:",
 ctx+"Question: What shape are the arrondissements arranged in? Answer:",
]
res=[]
for p in prompts:
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":32,"messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    txt=json.load(urllib.request.urlopen(r,timeout=240))["choices"][0]["message"]["content"]
    res.append(txt); print(f"\n>>> ...{p[-48:]!r}\n<<< {txt[:120]!r}")
json.dump(res,open(OUT,"w"))
PY
}
acc(){ grep -E "\[spec\]" "$LOG" | tail -1; }  # last cumulative accept_rate / emitted-per-step line

echo "===== PREFILL-SEED ($ALGO): FORCE_N0 reference (plain greedy decode — accepts no drafts) ====="
boot "MINISGL_SPEC_FORCE_N0=1"; probe /engine/tools/seed_${ALGO}.n0.json; stop

echo "===== PREFILL-SEED ($ALGO): seed OFF (baseline cold draft KV) ====="
boot "MINISGL_SPEC_DEBUG=1"; probe /engine/tools/seed_${ALGO}.off.json
echo "[off] $(acc)"; stop

echo "===== PREFILL-SEED ($ALGO): seed ON (MINISGL_SPEC_PREFILL_SEED=1) ====="
boot "MINISGL_SPEC_DEBUG=1 MINISGL_SPEC_PREFILL_SEED=1"; probe /engine/tools/seed_${ALGO}.on.json
echo "[on]  seed banner:"; grep -i "prompt-prefill draft-KV seed ENABLED" "$LOG" | head -1
echo "[on]  $(acc)"; stop

echo "===== compare (oracle = FORCE_N0 plain decode) ====="
python - /engine/tools/seed_${ALGO}.n0.json /engine/tools/seed_${ALGO}.off.json /engine/tools/seed_${ALGO}.on.json <<'PY'
import json,sys
n0=json.load(open(sys.argv[1])); off=json.load(open(sys.argv[2])); on=json.load(open(sys.argv[3]))
def cmp(name,a):
    ok=all(x==y for x,y in zip(n0,a))
    for i,(x,y) in enumerate(zip(n0,a)):
        print(f"  {name} prompt[{i}]: {'MATCH' if x==y else 'DIFF'} (n0 {len(x)} vs {len(y)})")
    print(f"  -> {name} vs FORCE_N0 LOSSLESS:", "PASS" if ok else "FAIL (verify qlen-drift; compare seed-OFF too)")
    return ok
cmp("seed-OFF",off); print(); cmp("seed-ON ",on)
PY
echo "[done]"
