#!/usr/bin/env bash
# Isolate the persistent-draft-KV win: a LONG PROMPT (~PCTX tokens) makes the drafter's context P
# large from token 1, so persist=0 re-projects O(P) every decode step while persist=1 projects only
# the new tail. Compare decode tok/s of dflash persist=1 vs persist=0 at large P (baseline already
# measured elsewhere). Lean image, 35B MXFP4, TP=2, graph verify.
set -uo pipefail
source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
python -c "import moe_hip,tail_hip,gdn_hip,attn_hip,attn_decode,attn_prefill_paged;print('[setup] hip pkgs OK')" || exit 1

MODEL="${MODEL:-pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4}"
DRAFT="${DRAFT:-z-lab/Qwen3.6-35B-A3B-DFlash}"
TP="${TP:-2}"; PORT="${PORT:-21079}"; MEMRATIO="${MEMRATIO:-0.82}"
MAXRUN="${MAXRUN:-4}"; NUM_DRAFT="${NUM_DRAFT:-8}"; GENTOK="${GENTOK:-200}"; PCTX="${PCTX:-3500}"
OUTDIR=/engine/tools

SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -TERM -- "-$SRV" 2>/dev/null
  for _ in $(seq 1 25); do kill -0 "$SRV" 2>/dev/null||break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; }
trap stop EXIT

boot(){ # $1=persist ; sets $LOG
  local persist="$1"
  LOG="$OUTDIR/spec_dflash_longprompt.p${persist}.log"
  local pynccl=""; [ "$TP" -gt 1 ] && pynccl="--disable-pynccl"
  echo "[boot dflash persist=$persist] -> $LOG"
  setsid env PYTHONPATH=/opt/kernels:/engine/python:/engine \
    MINISGL_MOE_SCATTER=0 MINISGL_KV_FP8=1 MINISGL_SPEC_DEBUG=1 MINISGL_DFLASH_PERSIST_KV=$persist \
    python -m minisgl --model "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" --graph 16 $pynccl \
    --memory-ratio "$MEMRATIO" --max-running-requests "$MAXRUN" --attention-backend hip \
    --spec-algorithm dflash --spec-num-draft $NUM_DRAFT --spec-draft-model-path $DRAFT > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 500); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "DIED:"; tail -40 "$LOG"; return 1; }
    grep -q "Traceback (most recent call last)" "$LOG" && { echo "CRASH:"; tail -40 "$LOG"; return 1; }
    sleep 3
  done; echo "timeout"; tail -40 "$LOG"; return 1
}

probe(){ PORT=$PORT GENTOK=$GENTOK PCTX=$PCTX python - <<'PY'
import json,os,time,urllib.request
PORT=os.environ["PORT"]; GENTOK=int(os.environ["GENTOK"]); PCTX=int(os.environ["PCTX"])
# Build a ~PCTX-token prompt by repeating a neutral passage, then ask for a short continuation.
para=("The history of computing spans many centuries and involves countless contributors. "
      "Early mechanical calculators gave way to electromechanical relays, then vacuum tubes, "
      "transistors, and finally integrated circuits that pack billions of devices onto a chip. ")
prompt=(para*400)[:PCTX*4]  # ~4 chars/token heuristic; oversized then truncated by the model context
prompt="Summarize the following passage in exactly one paragraph.\n\n"+prompt+"\n\nSummary:"
def gen():
    body=json.dumps({"model":"m","temperature":0.0,"max_tokens":GENTOK,"enable_thinking":False,
                     "messages":[{"role":"user","content":prompt}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=900)); el=time.time()-t
    u=d.get("usage") or {}
    return u.get("prompt_tokens"),u.get("completion_tokens") or 0, el
pt,ct,_=gen()  # warmup (also fills prefix cache so 2nd run is pure decode)
pt,ct,el=gen()
print(f"PROBE prompt_tokens={pt} completion_tokens={ct} elapsed={el:.2f} toks_per_s={ct/el:.2f}")
PY
}

for p in 1 0; do
  echo "===== dflash persist=$p  (PCTX~$PCTX, gen $GENTOK) ====="
  if boot "$p"; then
    out=$(probe); echo "$out"
    echo "  accept: $(grep -E '\[spec\]' "$LOG" | tail -1)"
  fi
  stop; sleep 2
done
echo "[done]"
