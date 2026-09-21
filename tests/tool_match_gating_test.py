"""MINISGL_TOOL_MATCH=think-gated — the in-span opener must not latch, hold, or EOS-guard.

THE DEFECT THIS PINS (2026-09-21, session ebbc1dd0903b). Qwen3.8-Flash-Next degenerated
mid-reasoning into template-register text and emitted a tool-call opener INSIDE its think span.
Raw-first matching latched it, which (a) held every later token from the client — the engine
warned "8,194 chars buffered inside an unclosed block" while the session streamed NOTHING for
11 minutes — and (b) armed the ToolCallGate EOS guard on a call that did not exist, so the model
could not end its turn while it looped at 18 tok/s. The client watchdog aborted at 924 s.

GATED mode is the checkpoint's own contract for a template that puts calls strictly AFTER the
reasoning span: the reasoning split runs first, the tool matcher sees only its content lane, the
mid-span opener stays inert in reasoning_content, and the scheduler-side gate suspends opener
matching until the span's close commits. The RAW default must keep its rescue semantics for
Laguna-shaped checkpoints (an opener without a span closer IS a call there) — these tests pin
BOTH behaviours so neither erodes the other.

PURE tests only — no engine, no model, no GPU. The think delimiters are synthetic (assembled
from parts so the literal strings never appear in this source) because ReasoningStreamState is
delimiter-agnostic; the tool opener/closer are pulled from api_server's real table at runtime.

Run:  PYTHONPATH=python PYTHONDONTWRITEBYTECODE=1 python3 tests/tool_match_gating_test.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from minisgl.scheduler.think_gate import ToolCallGate  # noqa: E402
from minisgl.server.reasoning import ReasoningStreamState  # noqa: E402

# Synthetic think delimiters, assembled so the literal tokens cannot be eaten by anything that
# processes this file's text (the same class of accident this feature guards against).
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"


def _tool_delims():
    """The REAL tool opener/closer this build matches, from api_server's own table."""
    from minisgl.server import api_server
    opener = next(iter(api_server._TOOL_OPENERS))
    return opener, api_server._TOOL_CLOSERS[opener]


# ---------------------------------------------------------------------------------------------
# 1. ToolCallGate: think-gated arming
# ---------------------------------------------------------------------------------------------
def test_gate_in_span_opener_is_inert():
    """An opener committed while the span is open must not arm EOS suppression; the span's close
    releases the suspension and the NEXT opener latches normally."""
    g = ToolCallGate(enabled=True, budget=64)
    EOS = 1
    # synthetic ids: think close = (80,), tool opener = (90,), closer = (91,)
    assert g.arm(7, openers=[(90,)], closers=[(91,)], eos_ids=[EOS], think_closers=[(80,)])
    for t in (11, 12, 13):
        g.commit(7, t)
    g.commit(7, 90)          # in-span opener: NOISE
    assert not g.is_open(7), "in-span opener must not open a block"
    assert not g.suppress_eos(7), "in-span opener must not suppress EOS"
    g.commit(7, 80)          # span closes
    g.commit(7, 90)          # genuine post-span opener
    assert g.is_open(7), "post-span opener must latch"
    assert g.suppress_eos(7), "post-span unclosed block must suppress EOS"
    g.commit(7, 91)          # block closes
    assert not g.is_open(7) and not g.suppress_eos(7)
    g.free(7)


def test_gate_ungated_arm_unchanged():
    """think_closers=() keeps the historical behaviour: any committed opener latches."""
    g = ToolCallGate(enabled=True, budget=64)
    assert g.arm(8, openers=[(90,)], closers=[(91,)], eos_ids=[1])
    g.commit(8, 90)
    assert g.is_open(8) and g.suppress_eos(8)
    g.free(8)


def test_gate_budget_bounded_post_span_only():
    """In-span tokens never count against the block budget; the budget still bounds a real
    post-span block (the anti-hang property the eos-guard pinned)."""
    g = ToolCallGate(enabled=True, budget=4)
    assert g.arm(9, openers=[(90,)], closers=[(91,)], eos_ids=[1], think_closers=[(80,)])
    for _ in range(50):
        g.commit(9, 11)      # long reasoning span — no budget consumed
    g.commit(9, 80)
    g.commit(9, 90)
    for i in range(3):
        g.commit(9, 11)
        assert g.suppress_eos(9)
    g.commit(9, 11)          # 4th token inside the block: budget spent
    assert not g.suppress_eos(9), "budget must still release EOS on an over-long block"
    g.free(9)


# ---------------------------------------------------------------------------------------------
# 2. ReasoningStreamState: the tool_openers override
# ---------------------------------------------------------------------------------------------
def _chunks(text, n):
    return [text[i:i + n] for i in range(0, len(text), n)]


def test_stream_gated_in_span_opener_stays_reasoning():
    """tool_openers=(): an opener inside the span does not end it — the text stays in the
    reasoning channel and nothing is routed to the tool parser's content lane."""
    opener, _ = _tool_delims()
    noise = f"plan the step.{opener}<function=junk>\n<parameter=q\n> x\n</parameter>\n</function>\n"
    body = THINK_OPEN + "reason first. " + noise + THINK_CLOSE + "\nanswer text"
    rs = ReasoningStreamState(THINK_OPEN, THINK_CLOSE, active=True, tool_openers=())
    r_all, c_all = [], []
    for ch in _chunks(body, 7):
        r, c = rs.push(ch)
        if r:
            r_all.append(r)
        if c:
            c_all.append(c)
    r_tail, c_tail = rs.flush()
    reasoning = "".join(r_all) + (r_tail or "")
    content = "".join(c_all) + (c_tail or "")
    assert THINK_CLOSE in body and content.lstrip() == "answer text", content
    assert opener in reasoning, "the in-span opener must stay VISIBLE in reasoning_content"
    assert "reason first." in reasoning and "junk" in reasoning


def test_stream_default_release_still_rescues_laguna_shape():
    """Default tool_openers: a mid-span opener still ends the span and routes the block to
    content — the Laguna rescue (vLLM/SGLang parity) must not regress."""
    opener, closer = _tool_delims()
    call = f"{opener}<function=web_search>\n<parameter=query\n> q\n</parameter>\n</function>\n{closer}\n"
    # Laguna shape: the span NEVER closes before the call.
    body = THINK_OPEN + "half a thought " + call
    rs = ReasoningStreamState(THINK_OPEN, THINK_CLOSE, active=True,
                              tool_openers=(opener,))
    r_all, c_all = [], []
    for ch in _chunks(body, 7):
        r, c = rs.push(ch)
        if r:
            r_all.append(r)
        if c:
            c_all.append(c)
    r_tail, c_tail = rs.flush()
    reasoning = "".join(r_all) + (r_tail or "")
    content = "".join(c_all) + (c_tail or "")
    assert "half a thought" in reasoning
    assert opener in content, "default mode must PRESERVE the opener into content for the tool parser"


def test_parse_gated_never_closed_span_has_no_calls():
    """Non-streaming parse, gated: an output whose span never closed is ALL reasoning with an
    empty body — an opener inside it yields no call."""
    from minisgl.server.reasoning import ReasoningParser
    opener, _ = _tool_delims()
    p = ReasoningParser(THINK_OPEN, THINK_CLOSE)
    p.tool_openers = (opener,)
    text = THINK_OPEN + "collapse " * 40 + f"{opener}<function=x>\n</function>\n"
    rc_gated, body_gated = p.parse(text, thinking_open=True, tool_openers=())
    assert body_gated == "" and rc_gated and "collapse" in rc_gated
    assert "function" in rc_gated, "gated parse keeps the noise INSIDE reasoning"
    rc_default, body_default = p.parse(text, thinking_open=True)
    assert body_default.startswith(opener), "default parse must still split at the opener"


# ---------------------------------------------------------------------------------------------
# 3. The ebbc1dd0903b pin: gated feed order vs raw feed order, end to end
# ---------------------------------------------------------------------------------------------
def _degenerate_transcript():
    """A faithful miniature of the collapsing turn: legit reasoning, an opener latched mid-span,
    looped register text, then (had the turn continued) a real post-span call."""
    opener, closer = _tool_delims()
    midspan_noise = (f"review the panel. {opener}<function=web_search>\n"
                     "<parameter=query\n> x\n</parameter>\n</function>\n")
    loop = "[omnibus middleware test — a continuation of the original conversation]\n" * 12
    real_call = (f"{THINK_CLOSE}\nnow the real step. {opener}<function=web_search>\n"
                 f"<parameter=query\n> shunt map\n</parameter>\n</function>\n{closer}\n")
    return THINK_OPEN + "review the panel. " + midspan_noise + loop + real_call


def test_gated_order_never_blinds_the_client():
    """THE regression pin. Gated order (reasoning split first, tool matcher on its content lane):
    an in-span opener never latches — held_chars stays ZERO while the span is open, everything
    streams as reasoning_content, and only the real POST-SPAN call is extracted."""
    from minisgl.server.api_server import ToolCallStreamState
    opener, closer = _tool_delims()
    body = _degenerate_transcript()
    rs = ReasoningStreamState(THINK_OPEN, THINK_CLOSE, active=True, tool_openers=())
    ts = ToolCallStreamState(0, allowed=frozenset({"web_search"}))
    held_max, calls, r_len, saw_noise = 0, [], 0, False
    for ch in _chunks(body, 13):
        r, c = rs.push(ch)
        if r:
            r_len += len(r)
            saw_noise = saw_noise or "omnibus" in r
        if c:
            c, tds = ts.push(c)
            calls.extend(tds)
        held_max = max(held_max, ts.held_chars)
    r_tail, c_tail = rs.flush()
    if c_tail:
        c_tail, tds = ts.push(c_tail)
        calls.extend(tds)
    hold, tds = ts.flush()
    calls.extend(tds)
    assert saw_noise, "the looped degenerate text must be VISIBLE in the reasoning stream"
    # The in-span noise opener must never have opened a block: the only held chars, if any, come
    # from the REAL call's own body — which closes within the transcript, so nothing is held at
    # flush time.
    assert ts.held_chars == 0
    # One call == two deltas (opener with empty arguments, then the arguments fragment).
    assert {c["index"] for c in calls} == {0}, calls
    assert calls[0]["function"]["name"] == "web_search", calls
    assert ts.emitted
    assert "shunt map" in calls[-1]["function"]["arguments"]


def test_raw_order_still_latches_the_in_span_opener():
    """The historical raw order (tool matcher on the RAW stream first) must keep doing exactly
    what it did on the incident — this is the behaviour Laguna needs, and it is why the mode is
    per-arm: the same transcript latches the noise opener and holds everything after it."""
    from minisgl.server.api_server import ToolCallStreamState
    body = _degenerate_transcript()
    ts = ToolCallStreamState(0, allowed=frozenset({"web_search"}))
    rs = ReasoningStreamState(THINK_OPEN, THINK_CLOSE, active=True, tool_openers=())
    held_max = 0
    for ch in _chunks(body, 13):
        nontool, tds = ts.push(ch)          # raw order: matcher first, on RAW text
        if nontool:
            rs.push(nontool)
        held_max = max(held_max, ts.held_chars)
    assert held_max > 1024, "raw order must still HOLD the in-span block (the incident behaviour)"
    assert ts.in_tool or ts.emitted, "raw order must still latch the noise opener"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
