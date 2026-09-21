"""Tool-calling degeneration probe for Qwen3.8-Flash-Next (q4e) — the signature that ACTUALLY occurs.

WHY THIS EXISTS, AND WHY THE OLD INSTRUMENT IS BLIND. The degeneration hunt has been scored by
EMPTY-CONTENT RATE at `finish_reason=stop` over ~12 trivial one-line questions. Measured against
real traffic on 2026-09-20, that signature occurs **zero times**: across all eight Hermes sessions
served by `Qwen3.8-Flash-Next` (2026-09-06 .. 09-19, ~76 assistant turns, two of them clean
post-bring-up sessions) there is not one empty-content `stop` turn. Every arm ever "exonerated" by
that metric — image, transformers, kv sidecar, concurrency, reasoning parser, chat template — was
cleared against a number that does not track the production failure.

What DOES occur, from session cdb27addb762 (2026-09-19), read verbatim out of Hermes' state.db:

    id     think  content  tool call
    37077  23941     1936  browser_exec({"code": "x"})        <- content ends mid-sentence
    37079     52        0  {"code": "# placeholder\nprint('ok')"}
    37082      0        0  same placeholder
    37085      0        0  same placeholder
    37088      0      763  recovers; the model calls its own placeholder calls "a mistake"

Turn 37077 carries 23,941 chars of thinking, content that stops mid-sentence ("...but neither
specifies:"), thinking that stops mid-token ("[0xA5][CMD][LEN][PAY"), and `finish_reason=tool_calls`
— NOT `length`. The first junk call is at 37077, BEFORE Hermes injected any "your previous response
was truncated" continuation prompt, so the continuation machinery is downstream of the failure, not
its cause.

DETECTOR CALIBRATION, measured 2026-09-20 by replaying every stored assistant turn in Hermes'
state.db through `score_turn` (5 models, 2,536 turns). This is what each detector is WORTH — a
detector is only useful where the q4e column stands clear of the controls:

    detector          Flash-Next   Qwen3.6-35B   3.8-27B-MTP   3.8-27B   deepseek(cloud)
    turns                     76           418           651       128              1263
    junk_args             *7.9%*          0.2%          0.0%      0.8%              0.0%
    think_zero              9.2%          0.5%          2.5%      3.1%              2.0%
    think_collapse          2.6%          0.7%          4.8%      5.5%              6.3%
    empty_stop              0.0%          0.0%          0.0%      0.0%              0.0%

  * junk_args is THE metric: 7.9% against <=0.8% everywhere else, ~10-40x. Validated against the
    six hand-read junk turns (37027/37077/37079/37082/37085 in cdb27addb762, 34460 in
    30935df66949): 6/6 caught, ZERO other firings in 76 turns.
  * think_zero is secondary — elevated (9.2% vs 0.5-3.1%) but it co-occurs with junk_args rather
    than adding independent signal.
  * think_collapse is DIAGNOSTIC ONLY, NOT A DEFECT COUNT. It is HIGHER on the healthy controls
    (6.3% on cloud deepseek) than on q4e. It measures conversation style, not degeneration.
    Recorded so nobody re-derives it and mistakes it for a finding.
  * empty_stop — THE LEGACY METRIC — reads 0.0% on every model including the one that is failing.
    It is kept in the output for exactly that reason: a run that shows the old number at zero while
    junk_args fires is the proof that the old instrument was blind.
  * A `midsentence` detector was built and DROPPED. Every threshold either fired on nothing or
    fired on 18% of plainly healthy turns (thinking that ends "...let me check:" before a tool call
    is normal). Content/thinking termination is still RECORDED per turn as `content_unterminated`
    for manual reading; it is not scored. Presence of a detector is not coverage.

SO THIS PROBE DRIVES TOOL-CALLING CONVERSATIONS, NOT ONE-LINE QUESTIONS, and scores four detectors
plus the legacy one (kept precisely so a run can show the legacy metric reading zero while the real
one fires). It is an HTTP client: CPU-only, no GPU lease, no container — the serve under test holds
the lease. Point it at a serve and let it talk.

Counts alone have burned this investigation before (a loop regex and a missing-period regex were
both blind to letter-spelling, and a degenerate arm passed validation once), so every turn's FULL
text is written to the fixture directory. READ A CLEAN ARM'S TEXT before believing it.

PROVENANCE IS ASSERTED, NOT ASSUMED. `--expect-model` is required and is checked against
`/v1/models`; a mismatch aborts before a single probe runs, because an A/B whose two arms silently
served the same build is the failure mode that has cost this repo the most time. `--serve-log`
additionally records whether the boot emitted

    recurrent-radix: snapshotting GDN + PLE state together

which is the q4e-exclusive composite prefix-cache path (`scheduler.py:431`, landed fb86ba3c
2026-09-09). Its ABSENCE while the recurrent radix is on means a prefix hit restores GDN state and
leaves the PLE conv window and n-gram history stale — record it either way, it is free.

USAGE

    python3 tools/toolcall_degen_probe.py --expect-model Qwen3.8-Flash-Next --arm q4e-baseline
    python3 tools/toolcall_degen_probe.py --expect-model Qwen3.6-35B --arm control-36 \
        --port 1919 --serve-log /path/to/serve.log

The CONTROL ARM IS THE POINT. Qwen3.6 shares this engine but is GDN-without-PLE, so it does not get
the composite snapshot store; running it separates "this engine degenerates" from "this q4e-only
path degenerates". A q4e number with no control arm is not evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

FIXTURE_ROOT = "/home/pat/fixtures/minisgl-toolcall-degen"

# ---------------------------------------------------------------------------------------------
# Tools. Deliberately the two shapes the real sessions used: a code-executing tool (whose argument
# is where every observed junk call landed) and a search tool (a plain string argument, so a junk
# call there is distinguishable from "the model dislikes writing code").
# ---------------------------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browser_exec",
            "description": "Execute Python in the browser-automation sandbox. The `code` argument "
                           "must be real, runnable code that performs the requested step.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to run."}},
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return result snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"],
            },
        },
    },
]

# Canned tool results. FIXED so every arm sees byte-identical tool output: a probe whose tool
# results vary cannot attribute a rate difference to the engine.
CANNED = {
    "terminal": "stdout:\n/home/user/workspace\nagent_helpers.py\n(exit 0)",
    "read_file": "# agent_helpers.py\ndef page_info():\n    ...\n",
    "browser_exec": "stdout:\nOK\n(exit 0)",
    "web_search": ("1. STM32G4 current sensing app note AN5397 — shunt amplifier configuration.\n"
                   "2. SimpleFOC docs: InlineCurrentSense with STM32 ADC injected conversions.\n"
                   "3. X-NUCLEO-IHM16M1 user manual, section 4.3: shunt resistor jumper map."),
}

# ---------------------------------------------------------------------------------------------
# Tasks. `long_spec` mirrors the shape that failed in cdb27addb762 — an underspecified protocol
# design question that drives the model into long deliberation with tools available. `short_task`
# is the cheap control. `long_prefix` is the SAME short task behind a large filler prefix, which is
# the prefill-length discriminator: if the failure tracks prompt size rather than deliberation
# length, these two separate and the trivial-question harness was measuring the wrong axis.
# ---------------------------------------------------------------------------------------------
FILLER = ("Reference material (do not summarise; it is background only).\n" + "\n".join(
    f"  R{i:03d}. Register 0x{i:02X} — reserved; read-returns-zero on this silicon revision."
    for i in range(240)))

TASKS = {
    "long_spec": (
        "Two firmware documents describe the same serial link and disagree. Doc A frames it as "
        "[0xA5][CMD][LEN][PAYLOAD][CRC16]; Doc B gives CRC-CCITT poly 0x1021 init 0xFFFF but never "
        "states whether CRC_HI or CRC_LO goes on the wire first, and Doc A's worked example is "
        "self-inconsistent on that point. Decide the wire byte order, justify it against both "
        "documents, and write the framing brief. Use the tools to check anything you need."
    ),
    "short_task": (
        "Use web_search to find the shunt-resistor jumper map for the X-NUCLEO-IHM16M1, then tell "
        "me in one sentence which jumper selects the low-side shunt."
    ),
    "browser_task": (
        "Find the shunt-resistor jumper map for the X-NUCLEO-IHM16M1 board on ST's website and tell "
        "me which jumper selects the low-side shunt. Drive the browser yourself to get it — open the "
        "page, wait for it to load, and pull the relevant section out of the DOM."
    ),
    "long_prefix": (
        FILLER + "\n\nUse web_search to find the shunt-resistor jumper map for the "
        "X-NUCLEO-IHM16M1, then tell me in one sentence which jumper selects the low-side shunt."
    ),
}


# ---------------------------------------------------------------------------------------------
# PRODUCTION-SHAPED TOOLSET. The simplified TOOLS above did NOT reproduce the failure: 60/60 calls
# came back with substantive arguments (fixture 20260920-172021-q4edeg, 0/67 turns). Reading the
# real definitions out of Hermes (tools/browser_use_cli.py: BROWSER_EXEC_SCHEMA, _HEADER_BASE,
# _HELPERS_DIGEST) shows why that probe could not have reproduced it, and it is not subtle:
#
#   * EVERY junk turn ever observed, across EVERY model and 2,536 stored turns, is a `browser_exec`
#     call. Zero on any other tool. Scored per browser_exec call rather than per turn, the real
#     rates are Flash-Next 6/7, Qwen3.6-35B 1/4, Qwen3.8-27B 1/5, deepseek(cloud) 0/7 — so the
#     published "7.9% vs <=0.8%" table was largely measuring HOW OFTEN each model happened to reach
#     for this one tool, not how degenerate it is.
#   * The real description MANDATES the shape the detector scores as junk: "Start `code` with a
#     one-line comment describing the step for the user in plain language, max 60 chars
#     (e.g. `# Searching Amazon for paper towels`)". The observed junk — "# placeholder",
#     "# placeholder\nprint('ok')" — is that mandated comment with no body after it.
#   * `code` is the only argument in the whole toolset that is inherently MULTI-LINE. Under the
#     native XML form the model writes raw newlines inside <parameter=code>; under the JSON the
#     structural tag forces, the same body has to be emitted as \n escapes inside a string.
#
# So the probe has to offer THIS tool, with THIS description, or it is not testing the thing that
# fails. Descriptions are copied verbatim from Hermes rather than paraphrased — the mandated
# leading comment is the whole point and a paraphrase would lose it.
# ---------------------------------------------------------------------------------------------
_BE_DESC = (
    "Drive a real web browser via the Browser Use CLI: `code` runs as full Python (stdlib available) "
    "with pre-imported browser helpers; stdout comes back in the result. Start `code` with a one-line "
    "comment describing the step for the user in plain language, max 60 chars "
    "(e.g. `# Searching Amazon for paper towels`) \u2014 the UI shows it as the step label.\n\n"
    "STATE: the browser session and workspace persist across calls; Python variables do NOT (fresh "
    "interpreter each call). The workspace dir is $BH_AGENT_WORKSPACE (also `workspace` in every result); "
    "functions defined in agent_helpers.py there are auto-imported into every call. For multi-item tasks "
    "('all N products / every entry'), append each batch to a JSON/CSV file in the workspace, then read it "
    "back and aggregate in code \u2014 dedupe/count/sort with Python, not in your head \u2014 and verify the "
    "collected count against what was asked before answering.\n\n"
    "Batch each sub-procedure (navigate, wait, extract, act) into one call \u2014 do not spend a call per "
    "action \u2014 but for long extractions prefer several medium calls that append to workspace files over "
    "one giant call, so progress survives timeouts."
    "\n\nHELPERS (pre-imported): new_tab(url) opens/navigates (use for the FIRST navigation), goto_url(url) "
    "navigates the current tab, wait_for_load() after navigation, page_info() summarizes the current page "
    "state, js(expr) evaluates a JS expression and returns its value (js('document.title'); wrap function "
    "bodies as js('(() => {...})()') \u2014 a bare '() => {...}' returns the function itself, uncalled), "
    "fill_input(selector, text) types into inputs, click_at_xy(x, y) clicks viewport coordinates, "
    "capture_screenshot() saves and prints a screenshot path, cdp('Domain.method', **kwargs) is raw CDP \u2014 "
    "cdp('Accessibility.getFullAXTree')['nodes'] lists every element's role/name/backendDOMNodeId (filter "
    "in Python before printing; it is thousands of nodes), then cdp('DOM.getBoxModel', backendNodeId=n) "
    "gives click coordinates. ensure_real_tab() recovers from a stale/internal tab. Login walls: never guess "
    "credentials; see the vault note below if present, otherwise stop and ask the user."
)

HERMES_TOOLS = [
    {"type": "function", "function": {
        "name": "browser_exec",
        "description": _BE_DESC,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute using the pre-imported browser helpers. Use print(...) for any data you need back."},
                "session": {"type": "string", "description": "Named isolated browser session \u2014 its own daemon and (on cloud backends) own browser, so concurrent tasks don't share tabs. Reuse the same name on every related call; omit for the shared default session."},
                "timeout_s": {"type": "integer", "default": 120, "description": "Max seconds to wait for the code to finish (default 120, max 600)."},
            },
            "required": ["code"],
        }}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web and return result snippets.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "The search query."},
                                      "limit": {"type": "integer", "description": "Max results."}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "terminal",
        "description": "Run a shell command on the host and return its stdout/stderr.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "The shell command."}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from disk and return its contents.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Absolute path."}},
                       "required": ["path"]}}},
]

SYSTEM = ("You are a careful engineering assistant with tool access. Call a tool only when it "
          "advances the task, and always with real, complete arguments.")

# ---------------------------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------------------------
_NOOP_LINE = re.compile(r"^\s*(#.*|pass|\.\.\.|print\(\s*['\"](ok|test|hello|placeholder)['\"]\s*\))\s*$",
                        re.IGNORECASE)
# CJK terminators included deliberately: the first cut used ASCII only and flagged two perfectly
# well-formed Chinese turns ending in U+3002 as truncated. This model mixes scripts.
_SENTENCE_END = tuple('.!?:;"\')]}`' + '\u3002\uff01\uff1f\uff1b\uff1a\u300d\u300f\uff09\u2026')

# Parameters that are supposed to CARRY THE WORK. The placeholder test applies only to these: a
# tool call is junk when its payload argument is empty, not when an incidental one is short.
# Deriving this from the tool schema (rather than "any short value") is load-bearing — the first
# cut of this detector flagged `{"query": "...", "limit": 6}` as junk because the integer 6 is two
# characters long, and over-fired on 23 of 71 real turns. A detector that cries wolf on a third of
# ordinary traffic cannot measure a 6% effect.
CONTENT_PARAMS = ("code", "query", "command", "content", "text", "input", "script", "body")


def is_placeholder_value(val) -> bool:
    """True when a STRING argument carries no actual work.

    Non-strings are never placeholders — `limit: 6` and `stream: false` are real arguments.
    Substring-matching "placeholder" would flag legitimate code that merely mentions the word, so
    this strips comments and no-op statements and asks whether ANYTHING executable is left. The
    observed junk — `x`, `# placeholder`, `# placeholder\\nprint('ok')` — reduces to empty; a real
    `import` or assignment does not.
    """
    if not isinstance(val, str):
        return False
    s = val.strip()
    if not s:
        return True
    if len(s) <= 2:                      # the observed `{"code": "x"}`
        return True
    live = [ln for ln in s.splitlines() if ln.strip() and not _NOOP_LINE.match(ln)]
    return not live


def call_is_junk(call: dict, content_params=CONTENT_PARAMS) -> bool:
    """Judge ONE tool call. Unparseable arguments count: truncated JSON is its own defect."""
    raw = (call.get("function") or call).get("arguments")
    if isinstance(raw, str):
        try:
            args = __import__("json").loads(raw)
        except Exception:
            return True
    elif isinstance(raw, dict):
        args = raw
    else:
        return raw is None or raw == ""
    if not isinstance(args, dict):
        return is_placeholder_value(args)
    named = [v for k, v in args.items() if k.lower() in content_params]
    if named:
        return any(is_placeholder_value(v) for v in named)
    # Tool with no recognised payload parameter: fall back to "every string argument is empty".
    strs = [v for v in args.values() if isinstance(v, str)]
    return bool(strs) and all(is_placeholder_value(v) for v in strs)


# Scored as defects, in order of demonstrated discriminating power. Everything else in the
# per-turn record is diagnostic: written down, never totalled into a verdict.
PRIMARY = ("junk_args",)
SECONDARY = ("think_zero",)
DIAGNOSTIC = ("think_collapse", "content_unterminated", "empty_stop")
SCORED = PRIMARY + SECONDARY + DIAGNOSTIC


def score_turn(msg: dict, prior_think: list, finish: str) -> dict:
    """Score ONE assistant turn. `prior_think` is this conversation's earlier thinking lengths."""
    content = (msg.get("content") or "")
    think = (msg.get("reasoning_content") or msg.get("reasoning") or "")
    calls = msg.get("tool_calls") or []

    junk = any(call_is_junk(c) for c in calls)

    # The collapse state seen at 37082/37085: a tool call with no deliberation and no prose.
    think_zero = bool(calls) and not think.strip() and not content.strip()

    # A sharp drop against this conversation's own established baseline. Guarded on a real
    # baseline (>=3 prior turns, median >500 chars) so an ordinarily terse conversation cannot
    # manufacture collapses.
    collapse = False
    if len(prior_think) >= 3:
        med = statistics.median(prior_think)
        if med > 500 and len(think) < 0.05 * med:
            collapse = True

    # Recorded, not scored (see the calibration table): thinking or prose that does not land on a
    # terminator. 37077 ended "...but neither specifies:" — but so does every healthy turn that
    # introduces a tool call, which is why this is a field and not a defect.
    tail = (content.rstrip() or think.rstrip())
    unterminated = bool(tail) and finish != "length" and not tail.endswith(_SENTENCE_END)

    return {
        "junk_args": junk,
        "think_zero": think_zero,
        "think_collapse": collapse,
        "content_unterminated": unterminated,
        "empty_stop": finish == "stop" and not content.strip(),   # the legacy, blind metric
    }


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------
def post(url: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url: str, timeout: int = 15) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def assert_provenance(base: str, expect: str) -> str:
    """Abort unless the endpoint is serving the model this arm claims. An A/B whose arms silently
    served the same build is the most expensive failure mode in this repo's history."""
    try:
        served = [m["id"] for m in get(f"{base}/v1/models").get("data", [])]
    except Exception as e:
        sys.exit(f"PROVENANCE: cannot reach {base}/v1/models ({e}). Is the serve up?")
    matches = [s for s in served if expect.lower() in s.lower()]
    if not matches:
        sys.exit(f"PROVENANCE FAIL: expected a model matching {expect!r}, endpoint serves {served}. "
                 f"Refusing to run — an arm that is not the arm you think it is produces a green "
                 f"result for the wrong build.")
    # Return the MATCHING id, never served[0]: a multi-model endpoint (lemonade lists its whole
    # downloaded registry, not just the loaded checkpoint) can name a different model first, and
    # the probe would drive every request at a model that is not loaded — 6 conversations, 0
    # turns, all request errors (observed 2026-09-21 vs lemonade 11.9.0, fixture
    # 20260921-154617-lemonade-q4gguf).
    return matches[0]


def scan_serve_log(path: str) -> dict:
    """Record the q4e composite prefix-cache boot line. Absent + radix on == PLE state goes stale on
    every prefix hit (see the module docstring)."""
    out = {"path": path, "composite_line": None, "recurrent_radix": None}
    try:
        with open(path, errors="replace") as fh:
            text = fh.read()
    except Exception as e:
        out["error"] = str(e)
        return out
    m = re.search(r".*recurrent-radix: snapshotting .*", text)
    out["composite_line"] = m.group(0).strip() if m else None
    out["recurrent_radix"] = bool(re.search(r"recurrent_radix|recurrent-radix", text))
    return out


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
def run_conversation(base: str, model: str, task: str, turns: int, args, rec: dict) -> list:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASKS[task]}]
    prior_think: list = []
    log = []
    for t in range(turns):
        payload = {
            "model": model,
            "messages": msgs,
            "tools": args.toolset_tools,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.top_k > 0:
            payload["top_k"] = args.top_k          # not OpenAI-standard; minisgl honours it
        t0 = time.time()
        try:
            resp = post(f"{base}/v1/chat/completions", payload, args.timeout)
        except Exception as e:
            log.append({"turn": t, "error": str(e), "elapsed": round(time.time() - t0, 1)})
            break
        dt = round(time.time() - t0, 1)
        choice = resp["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason") or "-"
        sc = score_turn(msg, prior_think, finish)
        think = (msg.get("reasoning_content") or msg.get("reasoning") or "")
        content = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        log.append({
            "turn": t, "finish": finish, "elapsed": dt,
            "think_len": len(think), "content_len": len(content),
            "tool_calls": [{"name": (c.get("function") or {}).get("name"),
                            "arguments": (c.get("function") or {}).get("arguments")} for c in calls],
            "flags": sc,
            "usage": resp.get("usage"),
            # FULL text, deliberately. A clean arm must be READ, not trusted.
            "content": content, "reasoning": think,
        })
        prior_think.append(len(think))
        flags = ",".join(k for k, v in sc.items() if v) or "-"
        print(f"    t{t:<2} {finish:<11} think={len(think):>6} content={len(content):>5} "
              f"calls={len(calls)} {dt:>6.1f}s  {flags}")

        if not calls:
            break                                   # the model finished its answer
        echo = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}
        if args.echo_reasoning:
            rc = msg.get("reasoning_content") or msg.get("reasoning") or ""
            if rc:
                echo["reasoning_content"] = rc
        msgs.append(echo)
        for c in calls:
            name = (c.get("function") or {}).get("name")
            msgs.append({"role": "tool", "tool_call_id": c.get("id"),
                         "content": CANNED.get(name, "OK")})
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, help="label for this arm, e.g. q4e-baseline")
    ap.add_argument("--expect-model", required=True, help="substring the endpoint MUST be serving")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--tasks", default="long_spec,short_task,long_prefix")
    ap.add_argument("--toolset", choices=("simple", "hermes"), default="hermes",
                    help="hermes (default) offers the REAL browser_exec definition — the only tool "
                         "that has ever produced a junk argument. `simple` is the original 2-tool "
                         "set, kept only to reproduce the 0/67 null it produced.")
    ap.add_argument("--echo-reasoning", dest="echo_reasoning",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="replay reasoning_content into the history, as Hermes does "
                         "(hermes_state_messages.py:980-981). Default ON. --no-echo-reasoning is the "
                         "old stripped regime, in which this checkpoint's unguarded template renders "
                         "every prior assistant turn as the blank <think></think> NO-THINK marker — a "
                         "prompt shape that was absent from 5 of the 6 real junk turns.")
    ap.add_argument("--reps", type=int, default=4, help="conversations per task")
    ap.add_argument("--turns", type=int, default=8, help="max assistant turns per conversation")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20,
                    help="0 to omit — the bare-request path no client ever sets")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--serve-log", default=None, help="serve log to scan for the composite line")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    args.toolset_tools = HERMES_TOOLS if args.toolset == "hermes" else TOOLS
    base = f"http://{args.host}:{args.port}"
    served = assert_provenance(base, args.expect_model)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or os.path.join(FIXTURE_ROOT, f"{stamp}-{args.arm}")
    os.makedirs(outdir, exist_ok=True)

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        sha = "?"

    rec = {
        "arm": args.arm, "endpoint": base, "served_model": served, "expect_model": args.expect_model,
        "started": stamp, "harness_sha": sha, "params": vars(args),
        "serve_log": scan_serve_log(args.serve_log) if args.serve_log else None,
        "conversations": [],
    }
    print(f"arm={args.arm}  served={served}  out={outdir}")
    if rec["serve_log"]:
        cl = rec["serve_log"].get("composite_line")
        print(f"  serve-log composite line: {cl if cl else 'ABSENT  <-- PLE not in the snapshot'}")

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        for rep in range(args.reps):
            print(f"  [{task}] rep {rep + 1}/{args.reps}")
            log = run_conversation(base, served, task, args.turns, args, rec)
            rec["conversations"].append({"task": task, "rep": rep, "turns": log})

    # ---- summary -----------------------------------------------------------------------------
    keys = SCORED
    print(f"\n{'task':<14}{'turns':>6}" + "".join(f"{k:>21}" for k in keys))
    overall = {k: 0 for k in keys}
    total = 0
    by_task = {}
    for conv in rec["conversations"]:
        d = by_task.setdefault(conv["task"], {"turns": 0, **{k: 0 for k in keys}})
        for t in conv["turns"]:
            if "flags" not in t:
                continue
            d["turns"] += 1
            total += 1
            for k in keys:
                d[k] += int(t["flags"][k])
                overall[k] += int(t["flags"][k])
    for task, d in by_task.items():
        n = max(d["turns"], 1)
        print(f"{task:<14}{d['turns']:>6}" +
              "".join(f"{d[k]:>12} {d[k] / n * 100:>7.1f}%" for k in keys))
    n = max(total, 1)
    print(f"{'ALL':<14}{total:>6}" + "".join(f"{overall[k]:>12} {overall[k] / n * 100:>7.1f}%" for k in keys))
    rec["summary"] = {"total_turns": total, "counts": overall, "by_task": by_task}

    # PER-TOOL, because every junk argument ever observed — in production and here — belongs to ONE
    # tool. A per-turn rate divided by a tool mix is what made the original cross-model table read as
    # a 10-40x model difference when the models had simply reached for browser_exec different numbers
    # of times. Report the denominator that the defect actually lives in.
    per_tool = {}
    for conv in rec["conversations"]:
        for t in conv["turns"]:
            junk = bool(t.get("flags", {}).get("junk_args"))
            for tc in t.get("tool_calls", []):
                d = per_tool.setdefault(tc.get("name") or "?", {"calls": 0, "junk": 0})
                d["calls"] += 1
                d["junk"] += int(junk)
    print(f"\n{'tool':<18}{'calls':>7}{'junk':>7}{'rate':>9}")
    for name, d in sorted(per_tool.items(), key=lambda kv: -kv[1]["calls"]):
        print(f"{name:<18}{d['calls']:>7}{d['junk']:>7}{d['junk'] / max(d['calls'], 1) * 100:>8.1f}%")
    rec["summary"]["per_tool"] = per_tool

    path = os.path.join(outdir, "result.json")
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"\n  PRIMARY  {PRIMARY[0]} is the metric (q4e 7.9% vs <=0.8% on every control).")
    print(f"  DIAGNOSTIC {', '.join(DIAGNOSTIC)} are recorded, NOT defects — think_collapse runs "
          f"HIGHER on healthy controls, and empty_stop is the blind legacy metric.")
    print(f"\nfixture: {path}")
    print("READ THE TEXT of a clean arm before believing it — counts have been blind here before.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
