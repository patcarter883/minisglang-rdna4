"""Regression tests for the two in-flight runaway defenses added 2026-08-17.

The incident (Hermes session f59a0028faa6 + subagent deleg_4dd1d349, Qwen3.6-35B): a subagent's
non-streaming call entered runaway generation inside a tool-call block; the client timed out at
600s, but the non-streaming lane had no disconnect handling, so the engine kept decoding for another
2.5 minutes — ~45k tokens total to a dead socket, evicting the live session's radix pages. Two
defenses:

* `acks_with_cancellation` — the non-streaming analogue of `stream_with_cancellation`: a throttled
  `is_disconnected` poll between acks; on disconnect the inner `wait_for_ack` generator is closed,
  whose `finally` is the guaranteed abort path.
* the runaway budget — both chat lanes force-finish a generation once a SINGLE unclosed tool-call
  block exceeds MINISGL_TOOLCALL_RUNAWAY_LIMIT chars (default 65536), delivering the partial with
  finish_reason="length" instead of decoding to max_tokens.

PURE tests only — a scripted fake ack source and the real ToolCallStreamState; no engine, no model.

Run:  PYTHONPATH=python python3 tests/toolcall_runaway_kill_test.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from minisgl.server.api_server import FrontendManager, ToolCallStreamState  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------------------------------------
# PART 1: acks_with_cancellation — pass-through on a live client, abort-via-close on a dead one
# ---------------------------------------------------------------------------------------------
class FakeAck:
    def __init__(self, i: int, finished: bool = False) -> None:
        self.incremental_output = f"tok{i}"
        self.finished = finished


def make_state(n_acks: int, closed_flag: dict):
    """A stand-in FrontendManager whose wait_for_ack yields n scripted acks; its finally records
    whether the generator was CLOSED (the abort path) vs exhausted normally."""

    async def wait_for_ack(uid: int):
        try:
            for i in range(n_acks):
                yield FakeAck(i, finished=(i == n_acks - 1))
                await asyncio.sleep(0)  # a real suspension point between acks
            closed_flag["exhausted"] = True
        finally:
            closed_flag["finalized"] = True

    return types.SimpleNamespace(wait_for_ack=wait_for_ack)


class FakeRequest:
    def __init__(self, disconnected: bool) -> None:
        self._d = disconnected

    async def is_disconnected(self) -> bool:
        return self._d


async def drive(disconnected: bool, n_acks: int = 5):
    flag: dict = {}
    state = make_state(n_acks, flag)
    got = []
    async for ack in FrontendManager.acks_with_cancellation(state, 7, FakeRequest(disconnected)):
        got.append(ack.incremental_output)
    return got, flag


print("PART 1a: live client — every ack passes through, source exhausts normally")
got, flag = asyncio.run(drive(disconnected=False))
check("all acks delivered", got == [f"tok{i}" for i in range(5)], repr(got))
check("source ran to completion", flag.get("exhausted") is True)
check("source finalized", flag.get("finalized") is True)

print("PART 1b: dead client — iteration stops early and the source is CLOSED (the abort path)")
got, flag = asyncio.run(drive(disconnected=True))
# The first throttled poll runs after the first ack (last_check starts at 0.0, monotonic >> 0.5).
check("stopped after the first ack", got == ["tok0"], repr(got))
check("source closed early (abort path ran)", flag.get("finalized") is True
      and flag.get("exhausted") is not True)

# ---------------------------------------------------------------------------------------------
# PART 2: the runaway budget — held_chars on the measured shape, fed in stream-sized chunks
# ---------------------------------------------------------------------------------------------
LIMIT = 65536
OPEN = '<tool_call>{"name": "write_file", "arguments": {"path": "a.ts", "content": "'

print("PART 2a: an unclosed block crosses the budget while a closed one never holds")
st = ToolCallStreamState(uid=1)
st.push(OPEN)
tripped_at = None
fed = len(OPEN)
while fed < LIMIT + 4096:
    st.push("x" * 512)
    fed += 512
    if st.held_chars > LIMIT:
        tripped_at = fed
        break
check("unclosed block trips the budget", tripped_at is not None)
check("trips just past the limit, not late", tripped_at is not None and tripped_at <= LIMIT + 1024,
      f"tripped at {tripped_at}")

st2 = ToolCallStreamState(uid=2)
big_closed = OPEN + "y" * (LIMIT + 4096) + '"}}</tool_call>'
step = 4096
held_after_close = None
for i in range(0, len(big_closed), step):
    st2.push(big_closed[i:i + step])
held_after_close = st2.held_chars
check("closed block ends with nothing held", held_after_close == 0, f"held {held_after_close}")
check("closed block emitted its call", st2.emitted is True)

print("PART 2b: prose between calls never accumulates held chars")
st3 = ToolCallStreamState(uid=3)
worst = 0
for i in range(64):
    st3.push(f"plain prose chunk {i} with no markup at all. ")
    worst = max(worst, st3.held_chars)
check("no held chars outside a block", worst == 0, f"worst {worst}")

if FAILED:
    print(f"\nFAILED: {len(FAILED)}: {FAILED}")
    sys.exit(1)
print("\nall checks passed")
