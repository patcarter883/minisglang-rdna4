"""An unoffered tool NAME must not silently end an agent turn.

MEASURED DEFECT (Hermes session db682d3cae84, 2026-09-20 21:13:24, Qwen3.8-Flash-Next). After 19,363
chars of reasoning the model wrote its preamble and called `web_fetch` — a plausible neighbour of the
`web_search` / `web_extract` it WAS offered among 28 tools. `_tool_name_allowed` dropped it, the reply
went out 200 OK with `tool_calls: []` and finish_reason=stop, the agent loop read that as "the model
chose not to act", ended the turn, and the session sat idle until a human asked why. The only trace
was ONE log line, because the warning fires once per name per process.

The drop default (b33756ec, 2026-08-16) was written against a FLOOD — "a looping model can emit
thousands of calls to invented names" — and that flood grew its own defences the NEXT DAY
(MINISGL_TOOLCALL_RUNAWAY_LIMIT + ack/stream cancellation, tests/toolcall_runaway_kill_test.py). So
name validation is redundant for the case it was built for and harmful for the single-call case.

    docker exec: python -m pytest tests/unknown_tool_forward_test.py -q -o addopts=""
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

import minisgl.server.api_server as api  # noqa: E402

OFFERED = frozenset({"web_search", "web_extract", "terminal"})


@pytest.fixture(autouse=True)
def _reset():
    api._UNKNOWN_TOOL_WARNED.clear()
    api._UNKNOWN_TOOL_SEEN.clear()
    os.environ.pop("MINISGL_DROP_UNKNOWN_TOOLS", None)
    yield
    os.environ.pop("MINISGL_DROP_UNKNOWN_TOOLS", None)


def test_an_offered_name_is_always_allowed():
    assert api._tool_name_allowed("web_search", OFFERED) is True


def test_no_tools_offered_means_no_validation():
    """Unchanged: a request that offers nothing cannot have an 'unknown' name."""
    assert api._tool_name_allowed("anything", None) is True


def test_an_unoffered_name_is_FORWARDED_by_default():
    """THE FIX. `web_fetch` is exactly the name that killed db682d3cae84."""
    assert api._tool_name_allowed("web_fetch", OFFERED) is True


def test_the_drop_is_still_available_explicitly():
    os.environ["MINISGL_DROP_UNKNOWN_TOOLS"] = "1"
    assert api._tool_name_allowed("web_fetch", OFFERED) is False


def test_an_old_launch_line_setting_the_FORWARD_knob_still_forwards():
    """A serve still passing MINISGL_FORWARD_UNKNOWN_TOOLS=1 must not change meaning — forwarding is
    now the default, so the stale knob is inert rather than inverted."""
    os.environ["MINISGL_FORWARD_UNKNOWN_TOOLS"] = "1"
    try:
        assert api._tool_name_allowed("web_fetch", OFFERED) is True
    finally:
        os.environ.pop("MINISGL_FORWARD_UNKNOWN_TOOLS", None)


def test_every_occurrence_is_COUNTED_not_just_the_first():
    """The old path logged once per name per process and exported no metric, so a recurrence was
    indistinguishable from none — which is why the second dead-stop could not be attributed."""
    for _ in range(4):
        api._tool_name_allowed("web_fetch", OFFERED)
    api._tool_name_allowed("ghost_tool", OFFERED)
    assert api._UNKNOWN_TOOL_SEEN["web_fetch"] == 4
    assert api._UNKNOWN_TOOL_SEEN["ghost_tool"] == 1
    assert len(api._UNKNOWN_TOOL_WARNED) == 2          # still one WARNING per name
