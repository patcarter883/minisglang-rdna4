"""RSA must return the ANSWER, not the reasoning trace.

MEASURED DEFECT (2026-09-09, ZAYA1-8B-MXFP4). `Candidate.text` from the in-process client is the
RAW generation — `<think>...</think>` plus the answer — because it accumulates
`ack.incremental_output`, the same stream `/v1/chat/completions` splits into `reasoning_content` +
`content`. Nothing split it back, so every RSA selection branch returned the reasoning span. Every
rung of the effort ladder produced non-answers:

    low    -> "We need to decide what to output. The proble..."
    medium -> the prompt echoed back
    high   -> "=== Candidate 2 === We have a problem: ..."

The correct answer never appeared at any rung. RSA was returning worse output than a plain call
while costing 8-70x more.

    python3 -m pytest tests/rsa_answer_split_test.py -q -o addopts=""
"""

from __future__ import annotations

import pytest

from minisgl.rsa.core import answer_text, reasoning_trace  # noqa: E402

DELIM = "</think>"
RAW = "We need to work this out. The bat is x+1.00 ...</think>The ball costs $0.05."


def test_answer_text_returns_only_the_post_delimiter_answer():
    assert answer_text(RAW, DELIM) == "The ball costs $0.05."


def test_answer_text_is_the_exact_inverse_of_reasoning_trace():
    """Together they must partition the rollout — no token belongs to both, none is lost."""
    trace, ans = reasoning_trace(RAW, DELIM), answer_text(RAW, DELIM)
    assert DELIM not in trace and DELIM not in ans
    assert trace + DELIM + ans == RAW.replace(DELIM + "The", DELIM + "The")  # lstrip is the only edit
    assert RAW.startswith(trace) and RAW.endswith(ans)


def test_no_delimiter_means_the_whole_text_is_the_answer():
    """The HTTP client's `text` is ALREADY content-only, and a non-reasoning model emits no
    delimiter. Both must pass through unchanged — this is what makes the split safe on both
    backends, which disagree about what Candidate.text holds."""
    plain = "The ball costs $0.05."
    assert answer_text(plain, DELIM) == plain
    assert answer_text(plain, None) == plain
    assert answer_text(RAW, None) == RAW


def test_leading_whitespace_after_the_delimiter_is_trimmed():
    assert answer_text("think</think>\n\n  Answer: 5", DELIM) == "Answer: 5"


def test_the_scaffolding_leak_is_gone():
    """The exact observed symptom: aggregation scaffolding sits in the REASONING half, so a correct
    split must not return it."""
    leaked = "=== Candidate 2 === We have a problem: ...</think>$0.05"
    assert answer_text(leaked, DELIM) == "$0.05"
    assert "Candidate" not in answer_text(leaked, DELIM)


def test_only_the_first_delimiter_splits():
    """A rollout that mentions the delimiter inside its answer must not be re-split."""
    t = "reasoning</think>the tag </think> is literal"
    assert answer_text(t, DELIM) == "the tag </think> is literal"


def test_selection_returns_RAW_text_so_the_parser_owns_the_split():
    """THE DOUBLE-STRIP REGRESSION. `api_server` parses `final_text` with the same reasoning parser
    the plain lane uses. If RSA pre-strips the `</think>`, that parser sees an open span with no
    closer, takes its "never closed => all reasoning" branch, and returns content="" — measured
    2026-09-09: n=8 produced an empty answer after 67,585 completion tokens.

    So `_select` must return the RAW candidate text. `answer_text` stays internal: a predicate for
    "did this candidate answer" and the input to boxed extraction. One split, one owner.
    """
    import inspect
    from minisgl.rsa import core
    src = inspect.getsource(core._select)
    bad = [ln.strip() for ln in src.splitlines()
           if "return" in ln and "answer_text(" in ln]
    assert not bad, f"_select must not pre-strip the returned text: {bad}"


def test_answer_text_is_still_used_as_an_internal_predicate():
    """...but it must not be deleted either: dropping it would let an answerless rollout be voted
    on and returned, which is the bug it was added for."""
    import inspect
    from minisgl.rsa import core
    src = inspect.getsource(core._select)
    assert "answer_text(" in src, "answer_text no longer used to filter/extract"
