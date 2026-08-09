"""Regression test: a template that appends NO generation prompt leaves the model mid-span.

The bug this locks down, observed live (Hermes session 6c2c589d64c5, Gemma-4 26B on :1919):

Gemma-4's `chat_template.jinja` skips its whole `add_generation_prompt` block when the conversation
ends in a tool call or tool result — by design, because the model is meant to CONTINUE the model turn
that is already open. But "thinking off" is spelled by INJECTING a pre-closed empty span
(`<|channel>thought\\n<channel|>`) inside that very block, so on those turns the template cannot say
"off" and the model opens a thought channel of its own. Its completion therefore starts INSIDE a
reasoning span with no opener — measured raw output for such a prompt, first token included:

    'thought\\n<channel|><|tool_call>call:terminal{command:<|"|>git branch<|"|>}<tool_call|>'

`_prompt_thinking_state` classified every request from a stand-in `[{"role":"user"}]` conversation,
which always gets a full generation prompt back, so it reported "thinking off" for these turns. Two
things then went wrong, and only the first was visible:

* `parse` (non-streaming) splits on the closer whenever it is present and was RIGHT anyway
  (`reasoning_content='thought'`). The STREAMING splitter only looks for the closer once it is
  already active, so it streamed `thought\\n<channel|>` to the user as the answer. Streaming and
  non-streaming disagreed on identical bytes — half this file is that parity.
* `_thinking_active` was false too, so the reasoning-budget backstop never armed and a turn that
  never emitted its closer ran unbounded: 34,371 characters of chain-of-thought delivered as
  `content`, degenerating into one sentence repeated 676 times.

Two halves, run independently:

* PURE — parser semantics on the recorded raw outputs, no model files. Always runs.
* ARTIFACT — the real Gemma-4 chat template. Skipped when the checkpoint is not in the HF cache.

Run:  PYTHONPATH=python python3 tests/reasoning_tool_continuation_test.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from minisgl.server.reasoning import ReasoningParser  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


CHAN = ReasoningParser("<|channel>thought", "<channel|>")


def stream(parser: ReasoningParser, chunks: list[str], active: bool) -> tuple[str, str]:
    """Drive the streaming splitter and return the (reasoning, content) it delivers in total."""
    st = parser.stream_state(active)
    r = c = ""
    for ch in chunks:
        a, b = st.push(ch)
        r += a or ""
        c += b or ""
    a, b = st.flush()
    return r + (a or ""), c + (b or "")


# ---------------------------------------------------------------------------------------------
# PURE: streaming and non-streaming must agree, byte for byte, on the recorded outputs
# ---------------------------------------------------------------------------------------------
# Tokenised the way the model emits them — every delimiter is a single special token in this
# checkpoint, so the split points are real, not arbitrary.
CLOSED = ["thought", "\n", "<channel|>"]                      # empty thought, then a tool block
UNCLOSED = ["thought", "\n", "Thinking", " Process", ":", " I"]  # ran out before its closer

print("PURE 1: a closer with no opener is reasoning, on BOTH paths")
r_s, c_s = stream(CHAN, CLOSED, active=True)
r_p, c_p = CHAN.parse("".join(CLOSED), thinking_open=True)
check("streaming routes it to reasoning", (r_s.strip(), c_s) == ("thought", ""), f"got {(r_s, c_s)!r}")
check("non-streaming agrees", ((r_p or "").strip(), c_p) == (r_s.strip(), c_s),
      f"{(r_p, c_p)!r} != {(r_s, c_s)!r}")
check("nothing leaks to content", "channel" not in c_s and not c_s.startswith("thought"), repr(c_s))

print("PURE 2: an UNCLOSED span is reasoning too — never a 34k answer")
r_s, c_s = stream(CHAN, UNCLOSED, active=True)
r_p, c_p = CHAN.parse("".join(UNCLOSED), thinking_open=True)
check("streaming holds it all as reasoning", c_s == "" and r_s.startswith("thought"), repr((r_s, c_s)))
check("non-streaming agrees", ((r_p or "").strip(), c_p) == (r_s.strip(), c_s),
      f"{(r_p, c_p)!r} != {(r_s, c_s)!r}")

print("PURE 2b: the ONE remaining divergence is trailing whitespace — pinned, not fixed")
# `parse` does `pre.strip()`; the splitter can only lstrip its FIRST chunk, because whitespace that
# turns out to be trailing has already gone out on the wire by the time the closer arrives. Holding
# it back would mean buffering every whitespace run in the reasoning stream for a difference no
# client reads. Asserted so the gap stays exactly this wide: same text, `reasoning_content` only.
r_s, _ = stream(CHAN, CLOSED, active=True)
r_p, _ = CHAN.parse("".join(CLOSED), thinking_open=True)
check("divergence is whitespace-only", r_s.strip() == (r_p or "").strip() and r_s != r_p,
      f"expected a whitespace-only gap, got {r_s!r} vs {r_p!r}")

print("PURE 3: the OLD state is what leaked — this is the bug, asserted")
# `active=False` is what the stand-in probe produced for these turns. Kept as a live assertion so the
# test fails loudly if someone 'fixes' the splitter to swallow spans it should pass through: the
# splitter is CORRECT here, it was told the wrong thing.
_, leaked = stream(CHAN, CLOSED, active=False)
check("span-closed => passthrough (splitter is not at fault)", leaked == "thought\n<channel|>", repr(leaked))

print("PURE 4: a genuinely thinking-OFF turn still passes through untouched")
ANSWER = ["Hello", " there", "."]
r_s, c_s = stream(CHAN, ANSWER, active=False)
check("plain answer is content", (r_s, c_s) == ("", "Hello there."), repr((r_s, c_s)))

print("PURE 5: an ECHOED opener is stripped, span already open — markup never reaches the client")
# Classifying these turns as span-open (the fix above) means the splitter now starts ACTIVE on turns
# where the model ALSO writes the opener itself. `parse` drops such an echo; the splitter must too,
# or every streamed reasoning_content on those turns starts with raw `<|channel>thought`.
ECHO = ["<|channel>", "thought", "\n", "I", " think", ".", "<channel|>", "Hi", "."]
r_s, c_s = stream(CHAN, ECHO, active=True)
r_p, c_p = CHAN.parse("".join(ECHO), thinking_open=True)
check("opener stripped from reasoning", "channel" not in r_s, repr(r_s))
check("streaming == non-streaming", (r_s.strip(), c_s) == ((r_p or "").strip(), c_p),
      f"{(r_s, c_s)!r} != {(r_p, c_p)!r}")
check("answer still delivered", c_s == "Hi.", repr(c_s))

print("PURE 6: span open, opener NOT echoed — text still routes to reasoning, not content")
# The other half of the same branch: the probe rules out an opener, and what it was holding must go
# to reasoning because the prompt already opened the span. Releasing it as content was the old
# behaviour and would leak a prompt-opened chain of thought.
NOECHO = ["The", " user", " wants", "<channel|>", "Done", "."]
r_s, c_s = stream(CHAN, NOECHO, active=True)
r_p, c_p = CHAN.parse("".join(NOECHO), thinking_open=True)
check("held head goes to reasoning", r_s.strip() == "The user wants", repr(r_s))
check("streaming == non-streaming", (r_s.strip(), c_s) == ((r_p or "").strip(), c_p),
      f"{(r_s, c_s)!r} != {(r_p, c_p)!r}")

print("PURE 7: a completion SHORTER than the opener is not swallowed")
for active, want in ((False, ("", "<|c")), (True, ("<|c", ""))):
    r_s, c_s = stream(CHAN, ["<|c"], active=active)
    check(f"active={active}: flushed, not dropped", (r_s, c_s) == want, repr((r_s, c_s)))

print("PURE 8: a SECOND closer is markup, not answer text — the reasoning-budget backstop")
# β force-closes the span at the budget; the model does not notice, keeps thinking, and emits its
# OWN closer when it finishes. That one arrives in the content phase. Observed on gemma-4 at
# reasoning_effort=low: `…pointer overhead."<channel|>Radix trees save memory by…` shipped as the
# visible answer. A stream cannot rpartition, so the boundary legitimately differs from `parse` —
# but raw markup must not survive in either lane.
FORCED = ["I", " think", "<channel|>", "more", " scratch", "<channel|>", "The", " answer", "."]
r_s, c_s = stream(CHAN, FORCED, active=True)
check("no markup in content", "<channel|>" not in c_s, repr(c_s))
check("no markup in reasoning", "<channel|>" not in r_s, repr(r_s))
check("answer text survives intact", c_s == "more scratchThe answer.", repr(c_s))
check("reasoning is the pre-β scratch", r_s == "I think", repr(r_s))

print("PURE 9: a stale closer SPLIT across chunks is caught, and nothing is truncated")
r_s, c_s = stream(CHAN, ["I", "<channel|>", "answer", " a<chan", "nel|>b"], active=True)
check("split closer stripped", "<channel|>" not in c_s and "chan" not in c_s, repr(c_s))
check("surrounding text kept", c_s == "answer ab", repr(c_s))
# A held-back partial that turns out NOT to be a closer must still be delivered.
r_s, c_s = stream(CHAN, ["I", "<channel|>", "answer<"], active=True)
check("held partial flushed at stream end", c_s == "answer<", repr(c_s))

print("PURE 10: a completion that never opened a span is passed through byte for byte")
# The strip is armed only by having consumed a closer, so text that merely CONTAINS the delimiter on
# a non-reasoning turn is not silently edited.
r_s, c_s = stream(CHAN, ["the tag ", "<channel|>", " is literal"], active=False)
check("untouched when no span ever closed", c_s == "the tag <channel|> is literal", repr(c_s))


print("PURE 11: a RE-OPENED span is scratch, not answer — on BOTH lanes")
# The model answers, then opens a new thought channel. Observed live: an answer ending
# `…implementation of the Delta-Schema.<|channel>thought`. That markup shipped as the reply, the
# caller stored and replayed it, and the NEXT turn's prompt genuinely ended mid-thought-channel —
# that turn thought for 15,206 characters and delivered an empty answer. One leaked delimiter, one
# dead turn, so this is asserted on both lanes.
REOPEN = ["thought", "\n", "scratch", "<channel|>", "The", " answer", ".",
          "<|channel>", "thought", "\n", "more", " scratch"]
r_s, c_s = stream(CHAN, REOPEN, active=True)
r_p, c_p = CHAN.parse("".join(REOPEN), thinking_open=True)
check("streaming: answer only, no markup", c_s == "The answer.", repr(c_s))
check("streaming: re-opened scratch is reasoning", "more scratch" in r_s and "channel" not in r_s, repr(r_s))
check("non-streaming: answer only, no markup", c_p == "The answer.", repr(c_p))
check("non-streaming: re-opened scratch is reasoning",
      "more scratch" in (r_p or "") and "channel" not in (r_p or ""), repr(r_p))
check("lanes agree on content", c_s == c_p, f"{c_s!r} != {c_p!r}")

print("PURE 12: a re-opened span that CLOSES again returns to content")
CYCLE = ["thought", "\n", "a", "<channel|>", "X", "<|channel>", "thought", "b", "<channel|>", "Y"]
r_s, c_s = stream(CHAN, CYCLE, active=True)
check("both answer fragments delivered", c_s == "XY", repr(c_s))
check("both scratch fragments captured", "a" in r_s and "b" in r_s and "channel" not in r_s, repr(r_s))

print("PURE 13: an opener SPLIT across chunks is caught in the content phase too")
r_s, c_s = stream(CHAN, ["t", "<channel|>", "answer <|chan", "nel>thought hidden"], active=True)
check("split re-opener stripped from content", c_s == "answer " and "chan" not in c_s, repr(c_s))
check("text after it is reasoning", "hidden" in r_s, repr(r_s))


# ---------------------------------------------------------------------------------------------
# ARTIFACT: the real template — does `add_generation_prompt` append anything?
# ---------------------------------------------------------------------------------------------
MODEL = "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"
TOOLS = [{"type": "function", "function": {
    "name": "terminal", "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]
USER = {"role": "user", "content": "make a branch"}
ASSISTANT = {"role": "assistant", "content": "on it",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "terminal", "arguments": {"command": "git branch"}}}]}
RESULT = {"role": "tool", "tool_call_id": "c1", "name": "terminal", "content": "* main"}

print(f"\nARTIFACT: {MODEL}")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
except Exception as e:  # noqa: BLE001 — not cached / no transformers: the PURE half still stands
    print(f"  SKIP  checkpoint unavailable ({type(e).__name__})")
    tok = None

if tok is not None:
    def tail(messages: list[dict], **kw) -> str:
        on = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                                     add_generation_prompt=True, **kw)
        off = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                                      add_generation_prompt=False, **kw)
        i = 0
        while i < min(len(on), len(off)) and on[i] == off[i]:
            i += 1
        return on[i:]

    t_user = tail([USER])
    t_tool = tail([USER, ASSISTANT, RESULT])
    t_think = tail([USER], enable_thinking=True)
    check("user-terminated turn gets a generation prompt", t_user != "", repr(t_user))
    check("...and it is the pre-closed empty span (thinking off)",
          CHAN.prompt_state(t_user) == (False, False), repr(t_user))
    check("thinking-on leaves the opener to the model",
          CHAN.prompt_state(t_think) == (False, True), repr(t_think))
    check("TOOL-terminated turn gets NO generation prompt", t_tool == "", repr(t_tool))

    # ---- the fix itself, through the server's own classifier -------------------------------
    # `_prompt_thinking_state` is what feeds `parse(thinking_open=…)`, the streaming splitter's
    # initial state, and the reasoning-budget backstop. Drive it with real request objects; stub the
    # two globals it resolves lazily so no engine/GPU state is needed.
    print("\nARTIFACT: _prompt_thinking_state on real requests")
    from minisgl.server import api_server as A

    A._FRONTEND_TOKENIZER, A._FRONTEND_TOKENIZER_SET = tok, True
    A._REASONING_PARSER, A._REASONING_PARSER_SET = CHAN, True
    A._THINKING_STATE_CACHE.clear()

    def state(messages: list[dict], **body) -> tuple[bool, bool]:
        req = A.OpenAICompletionRequest(model="m", messages=messages, tools=TOOLS, **body)
        return A._prompt_thinking_state(req)

    check("user-terminated  -> thinking off", state([USER]) == (False, False), repr(state([USER])))
    check("enable_thinking  -> span not open, reasoning still possible",
          state([USER], enable_thinking=True) == (False, True),
          repr(state([USER], enable_thinking=True)))
    s_tool = state([USER, ASSISTANT, RESULT])
    check("TOOL-terminated   -> span OPEN (was (False, False): the leak)",
          s_tool == (True, True), repr(s_tool))
    # Distinct shapes must not collide in the cache — a stale hit would reintroduce the bug for
    # whichever shape lost the race.
    check("cache keeps the shapes apart",
          state([USER]) == (False, False) and state([USER, ASSISTANT, RESULT]) == (True, True))



print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
    sys.exit(1)
print("all checks passed")
