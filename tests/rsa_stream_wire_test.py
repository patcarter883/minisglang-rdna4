"""An RSA answer asked for as a stream must go out on the SSE wire.

MEASURED DEFECT (2026-09-10, live on ZAYA1-8B). `stream: true` + RSA returned
`Content-Type: application/json`, `object: chat.completion`, and ZERO `data:` frames — a
`chat.completion` object handed to a client that asked for `text/event-stream`. openai-python's
stream iterator cannot read that. It went unnoticed while RSA was reachable only through the
non-standard `rsa` field; the reasoning_effort ladder makes every effort-carrying request take this
lane, so any streaming client would have hit it.

RSA genuinely cannot stream incrementally — the answer does not exist until the last aggregation
round resolves — but `stream` is a statement about the WIRE, not about latency. The finished answer
is re-framed as SSE.

    python3 -m pytest tests/rsa_stream_wire_test.py -q -o addopts=""
"""

from __future__ import annotations

import asyncio
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

from minisgl.server.api_server import _rsa_as_sse  # noqa: E402

PAYLOAD = {
    "id": "chatcmpl-rsa-7",
    "object": "chat.completion",
    "created": 1788991868,
    "model": "ZAYA1-8B",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "42",
                    "reasoning_content": "six sevens"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 11, "completion_tokens": 9, "total_tokens": 20},
    "rsa": {"selection_method": "majority_vote", "effort": "high", "n": 8, "k": 4, "t": 2},
}


def frames(payload=PAYLOAD, include_usage=False):
    async def drain():
        return [c async for c in _rsa_as_sse(payload, include_usage)]
    return [c.decode() for c in asyncio.run(drain())]


def events(**kw):
    """Every `data:` line except the [DONE] sentinel, parsed."""
    out = []
    for f in frames(**kw):
        assert f.startswith("data: ") and f.endswith("\n\n"), repr(f)
        body = f[len("data: "):-2]
        if body != "[DONE]":
            out.append(json.loads(body))
    return out


def test_the_stream_is_sse_framed_and_terminated():
    fs = frames()
    assert all(f.startswith("data: ") for f in fs)
    assert fs[-1] == "data: [DONE]\n\n"


def test_every_chunk_is_a_chat_completion_chunk_carrying_its_identity():
    for e in events():
        assert e["object"] == "chat.completion.chunk"
        assert e["id"] == PAYLOAD["id"]
        assert e["model"] == PAYLOAD["model"]
        assert e["created"] == PAYLOAD["created"]


def test_the_answer_and_the_reasoning_both_reach_the_client():
    es = events()
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in es if e["choices"]) == "42"
    assert "".join(e["choices"][0]["delta"].get("reasoning_content", "")
                   for e in es if e["choices"]) == "six sevens"


def test_the_role_leads_and_the_finish_reason_rides_the_terminal_chunk():
    es = events()
    assert es[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert [e["choices"][0]["finish_reason"] for e in es[:-1]] == [None] * (len(es) - 1)
    assert es[-1]["choices"][0]["finish_reason"] == "stop"


def test_usage_rides_the_terminal_chunk_by_default():
    assert events()[-1]["usage"] == PAYLOAD["usage"]


def test_include_usage_moves_the_totals_to_a_dedicated_trailing_chunk():
    """OpenAI spec: usage is null on every content chunk, totals ride a chunk with empty choices."""
    es = events(include_usage=True)
    assert es[-1]["choices"] == [] and es[-1]["usage"] == PAYLOAD["usage"]
    assert all(e["usage"] is None for e in es[:-1])


def test_the_rsa_telemetry_survives_the_stream():
    """A streaming caller has no other channel to learn which rung ran."""
    assert events()[-1]["rsa"]["n"] == 8


def test_a_tool_call_streams_with_an_index():
    payload = json.loads(json.dumps(PAYLOAD))
    payload["choices"][0]["message"] = {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "f", "arguments": "{}"}}]}
    payload["choices"][0]["finish_reason"] = "tool_calls"
    es = events(payload=payload)
    tc = [e["choices"][0]["delta"]["tool_calls"] for e in es
          if e["choices"] and "tool_calls" in e["choices"][0]["delta"]]
    assert tc and tc[0][0]["index"] == 0
    assert es[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_an_empty_answer_emits_no_content_frame_but_still_terminates():
    payload = json.loads(json.dumps(PAYLOAD))
    payload["choices"][0]["message"] = {"role": "assistant", "content": ""}
    es = events(payload=payload)
    assert not any("content" in e["choices"][0]["delta"] for e in es if e["choices"])
    assert frames(payload=payload)[-1] == "data: [DONE]\n\n"
