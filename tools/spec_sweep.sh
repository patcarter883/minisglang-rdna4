#!/usr/bin/env bash
# Laguna DFlash accept-len TRUTH harness.
#
# Boots ONE in-container minisgl server with a HAND-WRITTEN argv that is byte-identical on every
# build under test (the two builds' compose services / serve.sh differ, so neither can be the
# comparison vehicle). Runs three prompt GROUPS non-streaming and reports, per group, accept-len
# computed BOTH ways from raw counter deltas plus the /metrics gauge (whose DEFINITION differs
# between builds -- reported and labelled, never used for the A/B).
#
# TRUE tok/s comes from usage.completion_tokens on a non-streaming request. Never count SSE chunks.
#
# Env: K MAXRUN GRAPHBS MEMR MINISGL_KV_FP8 MINISGL_SPEC_MHA_PAGED LEG
set -uo pipefail
source /app/.venv/bin/activate 2>/dev/null || source /opt/venv/bin/activate 2>/dev/null || true
export PYTHONPATH=/opt/kernels:/engine/python:/engine
export HF_HUB_OFFLINE=1
export MINISGL_KV_FP8="${MINISGL_KV_FP8:-0}"

MODEL="${MODEL:-poolside/Laguna-XS-2.1-NVFP4}"
DRAFT="${DRAFT:-poolside/Laguna-XS-2.1-DFlash-NVFP4}"
K="${K:-15}"; TP="${TP:-2}"; LEG="${LEG:-leg}"
PORT=21955
LOG=/engine/tools/spec_truth.$LEG.server.log
SRV=""
stop(){ [ -n "$SRV" ]||return 0; kill -KILL -- "-$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""; sleep 3; }
trap stop EXIT

git config --global --add safe.directory '*' 2>/dev/null
git config --global --add safe.directory /engine 2>/dev/null
echo "################ PROVENANCE (from INSIDE the container) — LEG=$LEG ################"
echo "-- host/container id --"; hostname; cat /etc/os-release 2>/dev/null | head -2
echo "-- IMAGE fingerprint (baked engine + kernels; equal across legs == same image) --"
echo -n "  /opt/minisgl/python digest : "; find /opt/minisgl/python -name '*.py' -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum
echo -n "  /opt/kernels        digest : "; find /opt/kernels -name '*.so' -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | md5sum
echo -n "  torch                      : "; python -c "import torch;print(torch.__version__, torch.version.hip)" 2>&1 | tail -1
echo "-- MOUNTED /engine (the build under test) --"
echo -n "  .git pointer               : "; (cat /engine/.git 2>/dev/null || echo "(dir)")
echo -n "  git rev-parse HEAD         : "; git -C /engine rev-parse HEAD 2>&1 | tail -1
echo -n "  git log -1                 : "; git -C /engine log -1 --oneline 2>&1 | tail -1
echo    "  git status --porcelain     :"; git -C /engine status --porcelain 2>&1 | head -20
echo -n "  /engine/python digest      : "; find /engine/python -name '*.py' -print0 | sort -z | xargs -0 md5sum | md5sum
echo -n "  scheduler.py md5           : "; md5sum /engine/python/minisgl/scheduler/scheduler.py
echo -n "  metrics.py md5             : "; md5sum /engine/python/minisgl/server/metrics.py
echo -n "  BUILD DISCRIMINATOR grep -c MINISGL_SPEC_MHA_PAGED engine.py : "
grep -c MINISGL_SPEC_MHA_PAGED /engine/python/minisgl/engine/engine.py
echo    "  mean_accept_len GAUGE DEFINITION in this build:"
grep -n -A3 'minisgl_spec_mean_accept_len' /engine/python/minisgl/server/metrics.py | head -12
echo "-- REQUESTED conditions --"
echo "  K=$K TP=$TP MAXRUN=${MAXRUN:-1} GRAPHBS=${GRAPHBS:-0} MEMR=${MEMR:-0.90}"
echo "  MINISGL_KV_FP8=$MINISGL_KV_FP8 MINISGL_SPEC_MHA_PAGED=${MINISGL_SPEC_MHA_PAGED:-<unset>}"
echo "-- SWEPT KNOBS (as seen by THIS shell; the engine inherits them) --"
echo "  GRAPHBS                    = ${GRAPHBS:-0}   (0 => EAGER, >0 => cuda-graph capture)"
echo "  MINISGL_SPEC_PREFILL_SEED  = ${MINISGL_SPEC_PREFILL_SEED:-<unset>}"
echo "  MINISGL_SPEC_SAMPLED       = ${MINISGL_SPEC_SAMPLED:-<unset>}"
echo "  DO_SAMPLED (extra temp>0 cells) = ${DO_SAMPLED:-0}   SAMPLE_TEMP=${SAMPLE_TEMP:-0.7}"
echo "-- STATIC GATE CHECK: does the DFlash proposer even declare supports_prefill_seed? --"
grep -n "supports_prefill_seed" /engine/python/minisgl/spec/dflash.py || echo "  (absent in spec/dflash.py -> inherits Proposer default)"
grep -n "supports_prefill_seed" /engine/python/minisgl/spec/base.py | head -3
echo "#################################################################################"

boot(){
  setsid env MINISGL_SPEC_DEBUG=1 PYTHONPATH=/opt/kernels:/engine/python:/engine python -m minisgl \
    --model "$MODEL" --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size "$TP" --disable-pynccl \
    --cache-type naive --attention-backend hip --page-size 16 \
    --cuda-graph-max-bs "${GRAPHBS:-0}" --max-running-requests "${MAXRUN:-1}" \
    --memory-ratio "${MEMR:-0.90}" \
    --spec-algorithm dflash --spec-draft-model-path "$DRAFT" --spec-num-draft "$K" \
    > "$LOG" 2>&1 &
  SRV=$!
  for _ in $(seq 1 300); do
    python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/v1',timeout=3)" 2>/dev/null && return 0
    kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED"; tail -60 "$LOG"; exit 1; }
    sleep 3
  done
  echo "server not ready"; tail -60 "$LOG"; exit 1
}

echo "[boot] launching..."
boot
echo "[boot] ready."
echo "-- ENGINE ARGV (real, from /proc) --"
for p in $(pgrep -f "python -m minisgl" | head -4); do
  echo -n "  pid $p : "; tr '\0' ' ' < /proc/$p/cmdline; echo
done
echo "-- ENGINE page_size / spec decisions from the server log --"
grep -iE "page_size|spec-decode|spec_num_draft|memory_ratio|cache_type|swa|window|draft model" "$LOG" | head -25
echo "-- KNOB ENGAGEMENT PROOF (engine's own log lines) --"
echo -n "  'prompt-prefill draft-KV seed ENABLED'  : "; grep -c "prompt-prefill draft-KV seed ENABLED" "$LOG"
echo -n "  'SAMPLED (rejection-sampling) verify ENABLED' : "; grep -c "SAMPLED (rejection-sampling) verify ENABLED" "$LOG"
echo    "  graph/verify-capture lines:"; grep -iE "graph|captur" "$LOG" | head -12
echo -n "  ENGINE pid1-ish env (spec knobs as the ENGINE sees them): "
for p in $(pgrep -f "python -m minisgl" | head -1); do
  tr '\0' '\n' < /proc/$p/environ | grep -E "MINISGL_SPEC_(SAMPLED|PREFILL_SEED)|MINISGL_SPEC_MHA_PAGED|MINISGL_KV_FP8" | tr '\n' ' '
done; echo

LEG="$LEG" PORT=$PORT python - <<'PY'
import json, os, time, urllib.request
PORT=os.environ["PORT"]; LEG=os.environ["LEG"]
BASE=f"http://127.0.0.1:{PORT}"

# ---- PROMPT A : the FOUR verbatim prompts e6ddb502's 8.1 was measured on (swa_dflash_lossless.sh) --
REPETITIVE=[
 "Count from 1 to 250, writing each number on its own line, like:\n1\n2\n3\n",
 "Write the multiplication table for 7, from 7x1 up to 7x60, one product per line.",
 "List the numbers 1 to 200, and for each say whether it is even or odd, one per line.",
 "Repeat the sentence 'The quick brown fox jumps over the lazy dog.' exactly 80 times, numbered.",
]
# ---- PROMPT A' : ordinary PROSE continuation (the harmonic-oscillator style) -----------------------
PROSE=[
 "A rigorous study of physics. The quantum harmonic oscillator exhibits discrete energy levels "
 "spaced evenly apart. Write a detailed continuation of this exposition, covering the ladder "
 "operators, the zero-point energy, and the classical correspondence limit.",
 "Write a detailed technical explanation of how a B-tree index works, including insertion, node "
 "splitting, and range scans.",
]
# ---- PROMPT B : REAL CODE ------------------------------------------------------------------------
CODE=[
 "Write a complete, production-quality C++ source file implementing a B-tree with configurable "
 "order, supporting insert with node splitting, delete with merging, point lookup, and an "
 "in-order range scan iterator. Include the full class definition and all method bodies. "
 "Output only code.",
 "Write a complete Python source file implementing a paged-attention kernel wrapper: a class that "
 "owns a KV page pool, allocates and frees pages per sequence, builds the page table tensor, and "
 "dispatches a decode and a prefill attention call, with full error handling and docstrings. "
 "Output only code.",
]

def metrics():
    raw=urllib.request.urlopen(f"{BASE}/metrics",timeout=15).read().decode(); o={}
    for l in raw.splitlines():
        if l.startswith("#") or not l.strip(): continue
        try: o[l.split("{")[0].split(" ")[0]]=float(l.rsplit(" ",1)[1])
        except Exception: pass
    return o

def run(p, mt, temp=0.0):
    b={"model":"m","messages":[{"role":"user","content":p}],"max_tokens":mt,
       "temperature":temp,"stream":False}
    r=urllib.request.Request(f"{BASE}/v1/chat/completions",data=json.dumps(b).encode(),
                             headers={"Content-Type":"application/json"})
    t=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=1800).read())
    w=time.perf_counter()-t
    u=d.get("usage",{})
    return w, u.get("completion_tokens",0), u.get("prompt_tokens",0), \
           d["choices"][0]["message"].get("content","")[:160], d["choices"][0].get("finish_reason")

K=("minisgl_spec_steps_total","minisgl_spec_emitted_tokens_total",
   "minisgl_spec_accepted_tokens_total","minisgl_spec_draft_tokens_total")

def group(name, prompts, mt, temp=0.0):
    print(f"\n===== CELL  leg={LEG}  prompt={name}  temperature={temp} =====", flush=True)
    time.sleep(1.5); m0=metrics()
    rows=[]
    for p in prompts:
        w,n,pt,head,fin=run(p,mt,temp)
        rows.append((w,n))
        print(f"  req: prompt_tok={pt} completion_tok={n} wall={w:.2f}s "
              f"tok/s={n/w:.2f} finish={fin}", flush=True)
        print(f"       head={head!r}", flush=True)
    time.sleep(1.5); m1=metrics()
    st,em,ac,dr=[m1.get(k,0)-m0.get(k,0) for k in K]
    tw=sum(w for w,_ in rows); tn=sum(n for _,n in rows)
    print(f"  --- {name} @ {LEG} ---")
    print(f"  TRUE tok/s (usage.completion_tokens): aggregate={tn/tw:.2f}  "
          f"per-req={[f'{n/w:.2f}' for w,n in rows]}")
    print(f"  tokens={tn:.0f}  wall={tw:.2f}s")
    print(f"  spec steps={st:.0f}  emitted={em:.0f}  accepted={ac:.0f}  drafted={dr:.0f}")
    if st>0:
        rps=(em-ac)/st
        print(f"  accept-len [emitted/steps]            = {em/st:.3f}   <-- the '8.1' and '2.58' basis")
        print(f"  accept-len [emitted/(emitted-accept)] = {em/(em-ac):.3f}   <-- current-build gauge form")
        print(f"  accept-len [1 + accepted/steps]       = {1+ac/st:.3f}   <-- swaverify gauge form")
        print(f"  accepted-drafts/step [accepted/steps] = {ac/st:.3f}   <-- '[spec] mean accept-len' basis")
        print(f"  accept RATE [accepted/drafted]        = {ac/dr:.3f}" if dr>0 else "")
        print(f"  reqs/step = (emitted-accepted)/steps  = {rps:.4f}   "
              f"{'OK (bs=1)' if abs(rps-1.0)<0.03 else '*** VOID: BATCH-INFLATED ***'}")
        print(f"  ms/step = {1000*tw/st:.2f}")
    else:
        print("  spec steps=0 (no spec ran!)")
    g=m1.get("minisgl_spec_mean_accept_len",float('nan'))
    print(f"  /metrics gauge minisgl_spec_mean_accept_len (CUMULATIVE, build-specific defn) = {g:.3f}")

run("hi", 32)   # warm
group("A_repetitive_e6ddb502_verbatim", REPETITIVE, 720)
group("A2_prose",                        PROSE,      512)
group("B_real_code",                     CODE,       720)
if os.environ.get("DO_SAMPLED")=="1":
    T=float(os.environ.get("SAMPLE_TEMP","0.7"))
    # Real traffic is SAMPLED. With MINISGL_SPEC_SAMPLED=0 a non-greedy req is NOT allowed to spec
    # at all (_req_spec_ok: sp.is_greedy or self._spec_sampled) -> expect spec steps == 0 here.
    group(f"S_repetitive_temp{T}", REPETITIVE, 720, T)
    group(f"S_real_code_temp{T}",  CODE,       720, T)
PY

echo "-- [spec] debug lines (reqs/step cross-check) --"
grep -E "\[spec\]" "$LOG" | tail -12
stop
echo "[done] leg=$LEG"
