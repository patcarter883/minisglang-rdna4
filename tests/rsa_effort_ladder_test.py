"""The `reasoning_effort` knob must move RSA's WIDTH and DEPTH, not just a token budget.

MEASURED GAP (2026-09-10). RSA was opt-in through one channel only: the non-standard `rsa` field on
`/v1/chat/completions`. No OpenAI-shaped client sends it — Hermes sends `reasoning_effort` and
nothing else, and a custom provider's `extra_body` is provider-level, so it cannot vary per rung.
The result: a Hermes session pointed at a serve with RSA compiled in got ZERO rollouts at every
effort level. `max` bought a longer single completion, never a population.

The ladder's rungs are the ones measured on ZAYA1-8B-MXFP4 (tp=1 dp=2 ep=1, conc=64, tau=512,
beta=4096), where a plain call scored 3/6 on the probe and every RSA rung scored 3/3:

    low     n=2 T=1 k=2    603 s     14,156 completion tok
    medium  n=4 T=2 k=4   1158 s     42,072
    high    n=8 T=2 k=4   1144 s     72,847   <- n parallelises: +4 width, same wall clock
    max     n=8 T=3 k=4   1593 s    115,542   <- T does not: +1 round, +39% wall clock

    python3 -m pytest tests/rsa_effort_ladder_test.py -q -o addopts=""
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from minisgl.rsa.config import RSAParams  # noqa: E402
from minisgl.server.api_server import (  # noqa: E402
    OpenAICompletionRequest,
    _EFFORT_RSA,
    _rsa_from_effort,
)

MSG = [{"role": "user", "content": "hi"}]


def cfg(**over):
    """A server config stub carrying only what the resolver reads."""
    return SimpleNamespace(rsa_defaults=RSAParams(
        **{"effort_ladder": True, "tail_tokens": 512, "think_budget": 4096,
           "max_tokens": 6000, **over}))


def req(**kw):
    return OpenAICompletionRequest(model="ZAYA1-8B", messages=MSG, **kw)


@pytest.mark.parametrize("effort,n,k,t", [
    ("low", 2, 2, 1),
    ("medium", 4, 4, 2),
    ("high", 8, 4, 2),
    ("max", 8, 4, 3),
])
def test_each_rung_sets_its_measured_width_and_depth(effort, n, k, t):
    p = _rsa_from_effort(req(reasoning_effort=effort), cfg())
    assert p is not None, f"{effort} produced no RSA at all"
    assert (p.n, p.k, p.t) == (n, k, t)


def test_ladder_is_off_by_default():
    """RSA is a fan-out. A serve that did not ask for the ladder must not get one."""
    assert _rsa_from_effort(req(reasoning_effort="high"), cfg(effort_ladder=False)) is None


def test_disabled_rsa_defaults_beat_the_ladder():
    assert _rsa_from_effort(req(reasoning_effort="high"), cfg(enabled=False)) is None


@pytest.mark.parametrize("effort", ["none", "off", "disabled"])
def test_thinking_off_takes_the_plain_lane(effort):
    """Asking for NO reasoning is not a request for a population of reasoners."""
    assert _rsa_from_effort(req(reasoning_effort=effort), cfg()) is None


def test_minimal_is_a_rung_but_not_a_fan_out():
    """`minimal` is the LOWEST reasoning rung, not the absence of it — and n=1/T=1 is a plain call."""
    assert "minimal" in _EFFORT_RSA and _EFFORT_RSA["minimal"] is None
    assert _rsa_from_effort(req(reasoning_effort="minimal"), cfg()) is None


def test_no_effort_named_means_no_ladder():
    assert _rsa_from_effort(req(), cfg()) is None


def test_unknown_spelling_falls_through_rather_than_guessing():
    assert _rsa_from_effort(req(reasoning_effort="ludicrous"), cfg()) is None


@pytest.mark.parametrize("spelling", ["xhigh", "x-high", "extra high", "VERY_HIGH"])
def test_xhigh_clamps_down_to_high_never_up(spelling):
    """xhigh sits between `high` and `max` and has no rung; clamping UP would silently buy a round."""
    p = _rsa_from_effort(req(reasoning_effort=spelling), cfg())
    assert (p.n, p.k, p.t) == (8, 4, 2)


def test_openrouter_reasoning_object_is_read_too():
    p = _rsa_from_effort(req(reasoning={"effort": "max"}), cfg())
    assert (p.n, p.k, p.t) == (8, 4, 3)


@pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
def test_tau_beta_and_rollout_budget_are_NOT_laddered(effort):
    """Effort buys more SEARCH, not shorter thoughts.

    A rung that also shortened the workspace would spend the rollout compute and then truncate the
    reasoning it just produced — the user's objection to budget-shaped effort, and the reason beta
    stays at the serve's measured value on every rung.
    """
    p = _rsa_from_effort(req(reasoning_effort=effort), cfg())
    assert (p.tail_tokens, p.think_budget, p.max_tokens) == (512, 4096, 6000)


def test_beta_stays_far_above_tau_at_every_rung():
    """beta is the reasoning CHUNK, tau the tail carried forward; beta < tau is the degenerate
    regime run_markovian_rsa clamps and warns about (papers run 2-10x; this arm ships 8x)."""
    for effort in ("low", "medium", "high", "max"):
        p = _rsa_from_effort(req(reasoning_effort=effort), cfg())
        assert p.think_budget >= 2 * p.tail_tokens, effort


def test_defaults_other_than_width_and_depth_survive():
    p = _rsa_from_effort(req(reasoning_effort="high"), cfg(temperature=0.6, selection="majority",
                                                          max_concurrency=32))
    assert (p.temperature, p.selection, p.max_concurrency) == (0.6, "majority", 32)
