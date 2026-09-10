"""Regression test for the output cap applied to a request that sets NO `max_tokens`.

The bug this locks down: `OpenAICompletionRequest._coalesce_max_tokens` resolved an unset cap to
**16** on BOTH lanes. 16 is the documented default for `/v1/completions` (text); Chat Completions
defaults to the model maximum, and most clients — the OpenAI SDK, Hermes — send neither
`max_tokens` nor `max_completion_tokens`, so every chat request on this serve was capped at 16
tokens.

What that does to a thinking model is worse than a short answer. The reasoning-budget backstop caps
the think budget at 3/4 of max_tokens (scheduler `_maybe_arm_think_gate`), so the span was
force-closed at token 12 and the answer got four. Measured against a live serve 2026-09-10, every
reply came back `finish_reason="length"` carrying 0-18 characters and no tool_calls. An agent
harness reads that as truncation and retries: Hermes spent one continuation attempt per tool round
and then failed the turn with "Response remained truncated after 4 continuation attempts", having
paid a full prefill for each discarded call.

The chat default is FINITE rather than the true model maximum, and that is load-bearing: the prefill
manager reserves `output_len` KV tokens at admission (`scheduler/prefill.py::_try_allocate_one`), so
a cap of "the whole remaining context" reserves the whole remaining pool and nothing can be admitted
alongside it. An uncapped default would silently serialise the serve. THE_POOL_RESERVATION check
below pins that reasoning so it cannot be "simplified" into unboundedness.

All CPU, no GPU, no engine and no checkpoint.

Run:  PYTHONPATH=python python3 tests/default_max_tokens_test.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.pop("MINISGL_DEFAULT_MAX_TOKENS", None)

from minisgl.scheduler.utils import PendingReq  # noqa: E402
from minisgl.server import api_server as A  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


def chat(**kw) -> A.OpenAICompletionRequest:
    return A.OpenAICompletionRequest(model="m", messages=[A.Message(role="user", content="hi")], **kw)


def text(**kw) -> A.OpenAICompletionRequest:
    return A.OpenAICompletionRequest(model="m", prompt="hi", **kw)


# ---------------------------------------------------------------------------------------------
# THE BUG
# ---------------------------------------------------------------------------------------------
print("THE BUG: an unset cap on the chat lane must not be 16")
check("chat, nothing set -> NOT 16", chat().max_tokens != 16, f"{chat().max_tokens}")
check("chat, nothing set -> the chat default",
      chat().max_tokens == A._CHAT_DEFAULT_MAX_TOKENS, f"{chat().max_tokens}")
# On a model whose think budget is UNBOUNDED (Qwen3.8's template, and the `tool_choice: auto` tag is
# not a REQUIRED grammar, so `_maybe_arm_think_gate` does not substitute the server default) nothing
# force-closes the reasoning span — `think_gate.arm` clamps to 3/4 of max_tokens only for a bounded
# budget. So this cap alone has to cover the span AND the answer. `high` on the ladder asks for 4096.
check("the chat default covers a `high` reasoning span (4096) with as much again for the answer",
      A._CHAT_DEFAULT_MAX_TOKENS >= 2 * 4096, f"{A._CHAT_DEFAULT_MAX_TOKENS}")

print()
print("LANES: the two endpoints keep their OWN documented defaults")
check("text lane, nothing set -> 16 (the OpenAI /v1/completions default)", text().max_tokens == 16,
      f"{text().max_tokens}")
check("text and chat genuinely differ", text().max_tokens != chat().max_tokens)
# The lane is read the same way `_reject_malformed` reads it. A request carrying BOTH is the chat
# handler's raw-prompt form, which answers `chat.completion` — so it takes the chat default.
both = A.OpenAICompletionRequest(model="m", prompt="hi",
                                 messages=[A.Message(role="user", content="hi")])
check("prompt AND messages -> chat lane (it answers chat.completion)",
      both.max_tokens == A._CHAT_DEFAULT_MAX_TOKENS, f"{both.max_tokens}")

print()
print("COALESCE: an explicit cap always wins, in either spelling")
check("explicit max_tokens wins", chat(max_tokens=7).max_tokens == 7)
check("explicit max_completion_tokens wins", chat(max_completion_tokens=9).max_tokens == 9)
check("max_tokens beats max_completion_tokens (OpenAI's own precedence)",
      chat(max_tokens=7, max_completion_tokens=9).max_tokens == 7)
check("an explicit 16 is still honoured — the default changed, the field did not",
      chat(max_tokens=16).max_tokens == 16)
# `_reject_malformed` 400s on < 1; a client asking for 0 must still be rejected, not defaulted.
try:
    chat(max_tokens=0)
    check("max_tokens=0 is still a 4xx", False, "accepted")
except ValueError:
    check("max_tokens=0 is still a 4xx", True)

print()
print("KNOB: MINISGL_DEFAULT_MAX_TOKENS overrides the chat default only")
os.environ["MINISGL_DEFAULT_MAX_TOKENS"] = "1234"
check("knob is honoured on the chat lane", chat().max_tokens == 1234, f"{chat().max_tokens}")
check("knob does NOT touch the text lane", text().max_tokens == 16, f"{text().max_tokens}")
# env_int already tolerates junk; the floor is what keeps a junk-but-parseable value from turning
# every uncapped request into a 400 via `_reject_malformed`.
os.environ["MINISGL_DEFAULT_MAX_TOKENS"] = "0"
check("a 0 knob floors at 1 rather than 400ing every uncapped request", chat().max_tokens == 1,
      f"{chat().max_tokens}")
os.environ["MINISGL_DEFAULT_MAX_TOKENS"] = "not-a-number"
check("a junk knob falls back to the default", chat().max_tokens == A._CHAT_DEFAULT_MAX_TOKENS,
      f"{chat().max_tokens}")
os.environ.pop("MINISGL_DEFAULT_MAX_TOKENS", None)

print()
print("THE_POOL_RESERVATION: why the chat default is finite and not the model maximum")
# PendingReq.output_len IS sampling_params.max_tokens, and the prefill manager admits a request only
# if `extend_len + output_len + reserved_size <= available_size`. So the default is a direct charge
# against the KV pool at admission: make it the whole remaining context and the second concurrent
# request can never be admitted.
sp = A.SamplingParams(max_tokens=chat().max_tokens)
pending = PendingReq(uid=0, input_ids=[0] * 128, sampling_params=sp)
check("output_len is the coalesced max_tokens (the KV charge at admission)",
      pending.output_len == A._CHAT_DEFAULT_MAX_TOKENS, f"{pending.output_len}")
# Two concurrent requests must still fit a modest pool beside their prompts. 32k is small for this
# engine (the 27B here serves 225,872) and is the pessimistic case the default has to survive.
SMALL_POOL, PROMPT, CONCURRENCY = 32768, 4096, 2
check("two concurrent 4k-prompt requests still fit a 32k pool at the default cap",
      CONCURRENCY * (PROMPT + A._CHAT_DEFAULT_MAX_TOKENS) <= SMALL_POOL,
      f"{CONCURRENCY * (PROMPT + A._CHAT_DEFAULT_MAX_TOKENS)} > {SMALL_POOL}")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
