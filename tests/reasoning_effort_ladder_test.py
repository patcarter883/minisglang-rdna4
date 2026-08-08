"""Regression test: the reasoning-effort ladder maps to the template kwarg in BOTH directions.

`reasoning_effort` (and its OpenRouter-style aliases) is meant to be the portable way to ask for more
or less thinking, so a caller does not have to know that THIS checkpoint spells it
`chat_template_kwargs={"enable_thinking": …}`. Only the OFF half was wired.

An ON rung resolved a token BUDGET (`_resolve_think_budget`: xhigh -> 16384) and stopped there. It
never set `enable_thinking`, so a template that defaults to thinking-off — Gemma-4 — rendered a
pre-closed empty span and the model never opened a reasoning channel at all. Measured live on
gemma-4-26B, one prompt, `max_tokens=600`:

    reasoning_effort=xhigh   reasoning_content=0 chars, content=171
    reasoning_effort=high    reasoning_content=0 chars, content=171
    reasoning_effort=medium  reasoning_content=0 chars, content=171
    reasoning_effort=none    reasoning_content=0 chars, content=171   <- indistinguishable
    enable_thinking=true     reasoning_content=2118 chars, content=561

Every rung was a no-op, including the ones that asked for the most. The budget was real and useless:
a bound on a span that was never opened.

Run:  PYTHONPATH=python python3 tests/reasoning_effort_ladder_test.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from minisgl.server.api_server import (  # noqa: E402
    OpenAICompletionRequest,
    _resolve_chat_template_kwargs,
    _resolve_think_budget,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


def req(**body) -> OpenAICompletionRequest:
    return OpenAICompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **body)


def thinking(**body):
    """What `enable_thinking` this request resolves to (None = leave the template alone)."""
    return (_resolve_chat_template_kwargs(req(**body)) or {}).get("enable_thinking")


print("ON rungs open the span — this is the half that was missing")
ON = [
    ({"reasoning_effort": "low"}, 256),
    ({"reasoning_effort": "medium"}, 1024),
    ({"reasoning_effort": "high"}, 4096),
    ({"reasoning_effort": "xhigh"}, 16384),
    ({"reasoning_effort": "extra high"}, 16384),   # spelling-tolerant
    ({"reasoning_effort": "max"}, None),           # explicit "as long as you need" -> no budget
    ({"reasoning": {"enabled": True, "effort": "high"}}, 4096),
    ({"reasoning": {"enabled": True}}, None),      # enabled, no effort -> on, unbounded
    ({"thinking": True}, None),
]
for body, budget in ON:
    got = thinking(**body)
    check(f"{body} -> enable_thinking=True", got is True, f"got {got!r}")
    check(f"{body} -> budget {budget}", _resolve_think_budget(req(**body)) == budget,
          f"got {_resolve_think_budget(req(**body))!r}")

print("\nOFF rungs still close it (unchanged)")
OFF = [
    {"reasoning_effort": "none"},
    {"reasoning_effort": "minimal"},
    {"reasoning_effort": "off"},
    {"reasoning": {"enabled": False}},
    {"reasoning": {"exclude": True}},
    {"thinking": False},
]
for body in OFF:
    got = thinking(**body)
    check(f"{body} -> enable_thinking=False", got is False, f"got {got!r}")

print("\nAn explicit kwarg always wins over a rung that rode along")
check("enable_thinking=False beats reasoning_effort=high",
      thinking(enable_thinking=False, reasoning_effort="high") is False,
      repr(thinking(enable_thinking=False, reasoning_effort="high")))
check("enable_thinking=True beats reasoning_effort=none",
      thinking(enable_thinking=True, reasoning_effort="none") is True,
      repr(thinking(enable_thinking=True, reasoning_effort="none")))
check("chat_template_kwargs wins over the alias AND the rung",
      thinking(chat_template_kwargs={"enable_thinking": False},
               enable_thinking=True, reasoning_effort="high") is False,
      repr(thinking(chat_template_kwargs={"enable_thinking": False},
                    enable_thinking=True, reasoning_effort="high")))
check("OFF wins when a request somehow carries both",
      thinking(reasoning_effort="high", thinking=False) is False,
      repr(thinking(reasoning_effort="high", thinking=False)))

print("\nSaying nothing still says nothing — the template default is not overridden")
check("bare request leaves enable_thinking unset", thinking() is None, repr(thinking()))
check("...and resolves no kwargs at all", _resolve_chat_template_kwargs(req()) is None,
      repr(_resolve_chat_template_kwargs(req())))
check("an unrelated kwarg is passed through untouched",
      _resolve_chat_template_kwargs(req(chat_template_kwargs={"foo": 1})) == {"foo": 1},
      repr(_resolve_chat_template_kwargs(req(chat_template_kwargs={"foo": 1}))))
# `reasoning_max_tokens` is a BUDGET, not a mode: it bounds thinking that some other field turned on.
# Reading it as "on" would make a caller capping a runaway accidentally enable reasoning.
check("reasoning_max_tokens alone is a budget, not a switch",
      thinking(reasoning_max_tokens=4096) is None and _resolve_think_budget(req(reasoning_max_tokens=4096)) == 4096,
      repr(thinking(reasoning_max_tokens=4096)))

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
    sys.exit(1)
print("all checks passed")
