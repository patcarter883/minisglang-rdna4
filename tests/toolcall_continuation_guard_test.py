"""Regression test: reject continuations that resume inside an oversized unclosed tool-call block.

The loop this guard breaks, observed live (Hermes session 620a3c33becc, Qwen3.6-35B, 2026-08-17):
a completion hits max_tokens mid-tool-call, the client appends the partial block to the assistant
turn and re-sends, the model resumes the runaway and burns another full max_tokens — 8 legs x 8192
tokens overnight, ending only at a 65,510-char body the parser could not salvage. The guard 400s the
re-submission once the unclosed body is past MINISGL_TOOLCALL_CONT_LIMIT chars, while every benign
shape passes: closed blocks of any size, small truncated calls being legitimately resumed, prose that
merely quotes an opener, and unclosed blocks that are not in the final (continued) turn.

Run:  PYTHONPATH=python python3 tests/toolcall_continuation_guard_test.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from minisgl.server.api_server import (  # noqa: E402
    OpenAICompletionRequest,
    _reject_unclosed_toolcall_continuation,
)

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


LIMIT = 49152  # the guard's default


def req(messages: list[dict]) -> OpenAICompletionRequest:
    return OpenAICompletionRequest(model="m", messages=messages)


def verdict(messages: list[dict]):
    return _reject_unclosed_toolcall_continuation(req(messages))


def error_code(resp) -> str:
    return json.loads(bytes(resp.body))["error"]["code"]


USER = {"role": "user", "content": "build the app"}
# A runaway write_file whose arguments never close — the measured live shape, at guard-tripping size.
RUNAWAY = '<tool_call>{"name": "write_file", "arguments": {"path": "a.ts", "content": "' \
          + "x" * (LIMIT + 1024)

print("1: oversized unclosed block in the FINAL assistant turn is rejected")
r = verdict([USER, {"role": "assistant", "content": RUNAWAY}])
check("rejected", r is not None)
check("with the loop-breaking code", r is not None and error_code(r) == "unclosed_tool_call_continuation")

print("2: the same body CLOSED is a well-formed replay — allowed")
closed = RUNAWAY + '"}}</tool_call>'
check("allowed", verdict([USER, {"role": "assistant", "content": closed}]) is None)

print("3: a small truncated call is a legitimate resume — allowed")
small = '<tool_call>{"name": "write_file", "arguments": {"path": "a.ts", "content": "' + "y" * 2048
check("allowed", verdict([USER, {"role": "assistant", "content": small}]) is None)

print("4: an unclosed block NOT in the final turn is not being continued — allowed")
check("allowed", verdict([
    USER,
    {"role": "assistant", "content": RUNAWAY},
    {"role": "user", "content": "that looks stuck, stop"},
]) is None)

print("5: long prose that merely quotes an opener early is not a runaway block")
# Opener near the START of an otherwise huge closed-prose turn would hold everything after it —
# the shape a hard opener-regex would false-positive on. The guard still fires only because the
# block is genuinely unclosed AND oversized; quoting under the limit must pass.
prose = "the model emitted <tool_call> and then hung" + " padding" * 64
check("allowed", verdict([USER, {"role": "assistant", "content": prose}]) is None)

print("6: content-parts form is flattened before the scan — still rejected")
parts = [{"type": "text", "text": RUNAWAY[: len(RUNAWAY) // 2]},
         {"type": "text", "text": RUNAWAY[len(RUNAWAY) // 2:]}]
check("rejected", verdict([USER, {"role": "assistant", "content": parts}]) is not None)

print("7: MINISGL_TOOLCALL_CONT_LIMIT=0 disables the guard")
os.environ["MINISGL_TOOLCALL_CONT_LIMIT"] = "0"
try:
    check("allowed", verdict([USER, {"role": "assistant", "content": RUNAWAY}]) is None)
finally:
    del os.environ["MINISGL_TOOLCALL_CONT_LIMIT"]

print("8: a tightened limit fires on a correspondingly smaller body")
os.environ["MINISGL_TOOLCALL_CONT_LIMIT"] = "1024"
try:
    check("rejected", verdict([USER, {"role": "assistant", "content": small}]) is not None)
finally:
    del os.environ["MINISGL_TOOLCALL_CONT_LIMIT"]

print("9: no messages / raw-prompt requests pass through untouched")
check("allowed", _reject_unclosed_toolcall_continuation(
    OpenAICompletionRequest(model="m", prompt="raw")) is None)

if FAILED:
    print(f"\nFAILED: {len(FAILED)}: {FAILED}")
    sys.exit(1)
print("\nall checks passed")
