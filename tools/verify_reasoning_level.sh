#!/usr/bin/env bash
# Verify the reasoning-level plumbing IN A RUNNING SERVE, not just in unit tests.
#
# Two halves, because either alone can mislead:
#   A. PLUMBING — exec into the serve and ask the server's own functions what they resolved. A
#      behavioural check alone cannot distinguish "the level reached the template" from "the model
#      happened to think longer this time".
#   B. BEHAVIOUR — drive /v1/chat/completions at each rung and measure reasoning_content length. The
#      plumbing check alone cannot prove the model actually responds to the rendered line.
#
#   CONTAINER=<name> PORT=<port> tools/verify_reasoning_level.sh
set -uo pipefail
CONTAINER="${CONTAINER:?set CONTAINER=<serve container name>}"
PORT="${PORT:-1919}"

echo "=============== A. PLUMBING (inside $CONTAINER) ==============="
docker exec -i -e PYTHONPATH=/opt/kernels:/engine/python:/engine \
  -e SERVED_MODEL="${SERVED_MODEL:?set SERVED_MODEL=<hf id the serve was launched with>}" \
  "$CONTAINER" python - <<'PY'
import json
from minisgl.server import api_server as A
from minisgl.utils import load_generation_config
# NOT _served_model_path(): get_global_state() is PER-PROCESS, and `docker exec` starts a new
# interpreter that never ran serve startup — it returns None there and every check below silently
# degrades to "nothing configured". Take the path from the environment the serve was launched with.
import os
mp = os.environ.get("SERVED_MODEL") or os.environ.get("MODEL")
print(f"served model_path      : {mp}")

# 1) sampling inheritance from the DECLARED base model (the repack ships none of the three)
g = load_generation_config(mp)
print(f"generation_config      : temperature={g.get('temperature')} top_p={g.get('top_p')} "
      f"top_k={g.get('top_k')}  inherited_from={g.get('_sampling_inherited_from')}")

# 2) which kwarg the TEMPLATE consumes (sentinel render, derived not tabulated)
lvl = A._template_level_kwarg(mp)
print(f"template level kwarg   : {lvl!r}")

# 3) what each standard rung resolves to: template kwargs + think budget
class R:
    def __init__(s, eff): s.reasoning_effort = eff
    reasoning = None; enable_thinking = None; thinking = None
    reasoning_max_tokens = None; chat_template_kwargs = None
print(f"{'effort':10s} {'chat_template_kwargs':58s} {'think_budget'}")
for eff in ("low", "medium", "high", "xhigh", "max", "none", None):
    r = R(eff)
    print(f"{str(eff):10s} {str(A._resolve_chat_template_kwargs(r, mp)):58s} "
          f"{A._resolve_think_budget(r, mp)}")

# 4) PROOF the level reaches the rendered prompt: render at two rungs and diff the reasoning line
from minisgl.utils import load_tokenizer
tok = load_tokenizer(mp)
msgs = [{"role": "user", "content": "x"}]
for eff in ("low", "xhigh"):
    kw = A._resolve_chat_template_kwargs(R(eff), mp) or {}
    kw.pop("enable_thinking", None)          # Muse's template ignores it; keep the render clean
    out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    line = [l for l in out.split("\n") if "Reasoning strength" in l]
    print(f"rendered @ effort={eff:6s}: {line}")
PY

echo
echo "=============== B. BEHAVIOUR (port $PORT) ==============="
PORT="$PORT" python3 - <<'PY'
import json, os, time, urllib.request
port = os.environ["PORT"]
PROMPT = ("Derive the closed form of the sum of the first n fourth powers, showing the finite-"
          "difference argument, then verify it for n=1..5.")
print(f"{'effort':8s} {'reasoning chars':>15s} {'content chars':>14s} {'completion tok':>14s} {'s':>7s}")
for eff in ("low", "medium", "high", "xhigh"):
    body = {"model": "m", "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 4000, "reasoning_effort": eff, "seed": 7}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                               json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(r, timeout=900))
    except Exception as e:
        print(f"{eff:8s}  REQUEST FAILED: {type(e).__name__}: {e}"); continue
    m = d["choices"][0]["message"]
    rc = m.get("reasoning_content") or ""
    print(f"{eff:8s} {len(rc):15d} {len(m.get('content') or ''):14d} "
          f"{d['usage']['completion_tokens']:14d} {time.time()-t0:7.1f}")
print("\nEXPECT: reasoning length to RISE with the rung. Flat across all four means the level is not "
      "reaching the template (the failure this change exists to fix).")
PY
