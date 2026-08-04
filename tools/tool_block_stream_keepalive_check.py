"""CPU proof that an unclosed tool-call block no longer streams DEAD AIR.

Bug (measured 2026-08-04, Hermes session 722257e115e2 on Qwen3.6-35B-A3B-AWQ): the model opened a
`<tool_call>` and kept decoding into it without ever emitting the closer. `ToolCallStreamState.push`
holds the whole body while a block is open, so every `delta` came back empty and
`stream_chat_completions` yielded NOTHING — for 14 minutes and ~60,000 generated tokens the client
received zero bytes while the GPU sat at 100%. Both ends of the socket showed Recv-Q 0 and a frozen
byte counter: from the client there is no way to tell that from a hung server, so nothing timed out
and nothing retried.

Fix: while the tool buffer is holding, emit a periodic SSE COMMENT (`: …`) so the wire is provably
alive, and warn ONCE in the serve log when the held body crosses a size that no honest tool argument
reaches. The comment is not a `data:` line, so it never reaches an OpenAI client's delta stream —
the emitted token sequence is bit-identical, this only changes what goes out between tokens.

Run inside the serve container:
  PYTHONPATH=/engine/python python3 tools/tool_block_stream_keepalive_check.py
"""
from __future__ import annotations

import asyncio

from minisgl.server.api_server import (
    _TOOL_BLOCK_KEEPALIVE_S,
    FrontendManager,
    ToolCallStreamState,
)

FAILS: list = []


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILS.append(name)


class _Ack:
    """One scheduler ack: the raw incremental text plus the counters the streamer reads."""

    def __init__(self, out: str, finished: bool = False, finish_reason=None):
        self.incremental_output = out
        self.finished = finished
        self.finish_reason = finish_reason
        self.completion_tokens = 0
        self.prompt_tokens = 10
        self.error = None


class _Stub:
    """Stands in for FrontendManager: `stream_chat_completions` only ever calls `wait_for_ack`."""

    def __init__(self, acks, pace: float):
        self.acks, self.pace = acks, pace

    async def wait_for_ack(self, uid):
        for a in self.acks:
            await asyncio.sleep(self.pace)  # model decoding at some tokens/sec
            yield a


def run(acks, pace=0.02):
    async def _go():
        return [
            b.decode()
            async for b in FrontendManager.stream_chat_completions(
                _Stub(acks, pace), 1, None, ToolCallStreamState(1), False
            )
        ]

    out = asyncio.run(_go())
    return [c for c in out if c.startswith(":")], "".join(c for c in out if c.startswith("data:"))


def main() -> int:
    # The keepalive is time-based; pace the fake stream so the test spans a few intervals without
    # sleeping for the production 10s. 24 acks x (interval/4) ~= 6 intervals of dead air.
    pace = _TOOL_BLOCK_KEEPALIVE_S / 4

    print("\n1. unclosed <tool_call> — the wedge")
    acks = [_Ack('<tool_call>{"name":"write_file","arguments":{"content":"')]
    acks += [_Ack("lorem ipsum " * 40) for _ in range(24)]
    acks[-1].finished, acks[-1].finish_reason = True, "length"
    keepalives, data = run(acks, pace)
    check("keepalive comments went out while the block was open", len(keepalives) > 0)
    check("keepalives are SSE comments, never data lines", all(k.startswith(": ") for k in keepalives))
    check("trapped body still surfaces at flush", "lorem ipsum" in data)
    check("truncation is reported as finish_reason=length", '"finish_reason": "length"' in data)

    print("\n2. control — a normal tool call is byte-identical")
    keepalives, data = run(
        [_Ack('<tool_call>{"name":"f","arguments":{"a":1}}</tool_call>'), _Ack("", True, "stop")], pace
    )
    check("no keepalive on a block that closes promptly", keepalives == [])
    check("function name still emitted", '"name": "f"' in data)
    check("finish_reason is still tool_calls", '"finish_reason": "tool_calls"' in data)

    print("\n3. control — plain prose is byte-identical")
    keepalives, data = run([_Ack("hello "), _Ack("world"), _Ack("", True, "stop")], pace)
    check("no keepalive on ordinary content", keepalives == [])
    check("prose reaches the client intact", "hello" in data and "world" in data)

    print("\n4. the oversize warning latches once, then re-arms on the NEXT block")
    st = ToolCallStreamState(1)
    st.push('<tool_call>{"name":"write_file","arguments":{"content":"')
    check("held_chars tracks the trapped body", st.held_chars > 0)
    below = st.warn_if_oversize(threshold=64)
    st.push("x" * 200)
    crossed = st.warn_if_oversize(threshold=64)
    repeat = st.warn_if_oversize(threshold=64)
    check("silent below threshold", below is False)
    check("warns exactly once on crossing", crossed is True)
    check("does not warn again for the same block", repeat is False)
    st.push('"}}</tool_call>')
    check("held_chars clears when the block closes", st.held_chars == 0)
    check("re-armed for a later block", st._warned_oversize is False)

    st = ToolCallStreamState(1)
    st.push("just prose, no markup at all")
    check("prose never registers as held", st.held_chars == 0)

    print("\n" + ("FAILED: " + "; ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
