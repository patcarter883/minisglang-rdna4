"""Regression test for the OpenAI TEXT-completions endpoint, `/v1/completions`.

The bug this locks down: `/v1/completions` was NOT SERVED AT ALL. The chat handler was named
`v1_completions` while being registered only at `/v1/chat/completions`, so the symbol asserted a
route the app never had and every `client.completions.create(...)` got a bare FastAPI 404. The name
is why it survived a read of the file.

The near-miss fix this also locks down is ALIASING — pointing `/v1/completions` at the chat handler.
It looks safe here (the chat handler really does accept a raw `prompt`, and the tokenizer applies a
chat template only to a messages LIST, so nothing would be silently templated) and it is still
wrong: that handler answers `{"object":"chat.completion","choices":[{"message":…}]}`, which has no
`text` key. A completions client reads `choices[0].text`, finds nothing, and renders an empty
answer off a 200. So the two endpoints must stay two DISTINCT handlers with two response objects,
which is what ROUTES 3 asserts.

Four halves, all CPU, no GPU, no engine and no checkpoint:

* ROUTES    — the registered path set, the handler names, and non-aliasing.
* REJECT    — every parameter the raw-prompt lane cannot honour is a 400, never accepted-and-ignored.
* STREAM    — the `text_completion` SSE wire shape, driven off a stub ack generator.
* HANDLER   — the whole non-streaming handler against a fake FrontendManager: verbatim output, the
              reasoning-backstop arming rule, and the sampling resolution.

Run:  PYTHONPATH=python python3 tests/completions_endpoint_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# The transparent-CAM hooks are env-gated and OFF here: this test is about the HTTP contract, and a
# CAM read would rewrite the prompt the verbatim assertions below depend on.
os.environ.pop("MINISGL_CAM_AUTO", None)
os.environ.pop("MINISGL_CAM_AUTO_WRITE", None)

from minisgl.message import UserReply  # noqa: E402
from minisgl.server import api_server as A  # noqa: E402
from minisgl.server.reasoning import ReasoningParser  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------------------------
print("ROUTES 1: the served path set")
SERVED = {r.path: r for r in A.app.routes if getattr(r, "endpoint", None) is not None}
for path in sorted(SERVED):
    r = SERVED[path]
    print(f"        {','.join(sorted(r.methods)):26} {path:24} -> {r.endpoint.__name__}")
check("/v1/completions is registered (it 404'd)", "/v1/completions" in SERVED)
check("/v1/chat/completions is registered", "/v1/chat/completions" in SERVED)
check("/v1/completions accepts POST",
      "POST" in SERVED.get("/v1/completions", type("x", (), {"methods": set()})).methods)

print("ROUTES 2: handler names name the protocol they serve")
check("chat route -> v1_chat_completions",
      SERVED["/v1/chat/completions"].endpoint.__name__ == "v1_chat_completions",
      SERVED["/v1/chat/completions"].endpoint.__name__)
check("text route -> v1_text_completions",
      SERVED["/v1/completions"].endpoint.__name__ == "v1_text_completions",
      SERVED["/v1/completions"].endpoint.__name__)

print("ROUTES 3: the two endpoints are NOT the same function")
check("distinct handlers (an alias would return chat.completion from /v1/completions)",
      SERVED["/v1/completions"].endpoint is not SERVED["/v1/chat/completions"].endpoint)

# ---------------------------------------------------------------------------------------------
# REJECT: a parameter this lane cannot honour is a 400, not a silent no-op
# ---------------------------------------------------------------------------------------------
print("REJECT: unsupported parameters are 400s with the offending `param` named")


def reject_param(**kw) -> str | None:
    """The `param` of the 400 `_reject_unsupported_text_completion` raises, or None if accepted."""
    resp = A._reject_unsupported_text_completion(A.OpenAICompletionRequest(model="m", **kw))
    if resp is None:
        return None
    assert resp.status_code == 400, resp.status_code
    return json.loads(resp.body)["error"]["param"]


REJECT_CASES = [
    (dict(messages=[{"role": "user", "content": "hi"}]), "messages", "chat array on the raw lane"),
    (dict(prompt="hi", tools=[{"type": "function", "function": {"name": "f"}}]), "tools",
     "tool specs render into the chat template, which this lane has none of"),
    (dict(prompt="hi", rsa=True), "rsa", "RSA drives chat rollouts"),
    (dict(prompt="hi", echo=True), "echo", "changes what the response contains"),
    (dict(prompt="hi", suffix="tail"), "suffix", "fill-in-the-middle"),
    (dict(prompt="hi", best_of=4), "best_of", "changes what is sampled"),
    (dict(prompt="hi", n=3), "n", "inherited from _reject_unsupported"),
    (dict(prompt="hi", logprobs=True), "logprobs", "inherited from _reject_unsupported"),
]
for kw, want, why in REJECT_CASES:
    got = reject_param(**kw)
    check(f"reject {want} ({why})", got == want, f"got param={got!r}")
check("a plain raw prompt is ACCEPTED", reject_param(prompt="The capital of France is") is None)
try:
    A.OpenAICompletionRequest(model="m")
    check("neither prompt nor messages -> 422 at validation", False, "accepted")
except Exception:
    check("neither prompt nor messages -> 422 at validation", True)

# ---------------------------------------------------------------------------------------------
# STREAM: the text_completion SSE wire shape
# ---------------------------------------------------------------------------------------------
print("STREAM: object/text/finish_reason/usage, and the include_usage trailing chunk")


class _StubState:
    """Just enough FrontendManager to drive `stream_text_completions` — it only calls wait_for_ack."""

    def __init__(self, acks):
        self._acks = acks

    async def wait_for_ack(self, uid):  # noqa: ANN001
        for ack in self._acks:
            yield ack


def ack(text: str, n: int, *, finished: bool = False, reason: str | None = None) -> UserReply:
    return UserReply(uid=7, incremental_output=text, finished=finished, completion_tokens=n,
                     prompt_tokens=5, finish_reason=reason)


def drain(acks, include_usage: bool) -> list:
    """Collect the SSE payloads (the terminal `[DONE]` sentinel included as the string 'DONE')."""
    async def go():
        out = []
        gen = A.FrontendManager.stream_text_completions(
            _StubState(acks), 7, "m", include_usage)
        async for raw in gen:
            line = raw.decode().strip()
            assert line.startswith("data: "), line
            payload = line[len("data: "):]
            out.append("DONE" if payload == "[DONE]" else json.loads(payload))
        return out
    return asyncio.run(go())


ACKS = [ack("Par", 1), ack("is", 2), ack(".", 3, finished=True, reason="stop")]
chunks = drain(ACKS, include_usage=False)
check("terminates with [DONE]", chunks[-1] == "DONE", repr(chunks[-1]))
body = chunks[:-1]
check("every chunk is object=text_completion",
      all(c["object"] == "text_completion" for c in body))
check("every chunk carries id/created/model (openai-python requires all five)",
      all({"id", "created", "model", "choices"} <= set(c) for c in body))
check("the streamed text reassembles to the completion",
      "".join(c["choices"][0]["text"] for c in body) == "Paris.")
check("no chat `delta` key anywhere (that is the other protocol)",
      not any("delta" in c["choices"][0] for c in body if c["choices"]))
check("finish_reason rides the terminal chunk only",
      [c["choices"][0]["finish_reason"] for c in body] == [None, None, None, "stop"],
      str([c["choices"][0]["finish_reason"] for c in body]))
check("content chunks omit `usage` entirely",
      all("usage" not in c for c in body[:-1]))
check("usage totals on the terminal chunk",
      body[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
      str(body[-1].get("usage")))

usage_chunks = drain(ACKS, include_usage=True)[:-1]
check("include_usage: totals move to a dedicated trailing chunk with empty choices",
      usage_chunks[-1]["choices"] == [] and usage_chunks[-1]["usage"]["total_tokens"] == 8)
check("include_usage: the finish chunk's usage is null (OpenAI spec)",
      usage_chunks[-2]["usage"] is None)

ERR = [UserReply(uid=7, incremental_output="", finished=True, error="prompt is too long")]
err_chunks = drain(ERR, include_usage=False)
check("an engine refusal mid-stream emits an explicit error event, not silence",
      isinstance(err_chunks[0], dict) and "error" in err_chunks[0] and err_chunks[-1] == "DONE",
      str(err_chunks))

# ---------------------------------------------------------------------------------------------
# HANDLER: the whole non-streaming route, against a fake FrontendManager
# ---------------------------------------------------------------------------------------------
print("HANDLER: verbatim output, the backstop arming rule, and sampling resolution")

# Gemma-4's asymmetric reasoning pair, injected directly so this half needs no checkpoint on disk.
CHAN = ReasoningParser("<|channel>thought", "<channel|>")
A._REASONING_PARSER, A._REASONING_PARSER_SET = CHAN, True
A._FRONTEND_TOKENIZER, A._FRONTEND_TOKENIZER_SET = None, True
# The checkpoint's own recommended sampling, as generation_config.json would carry it.
GEN_CONFIG = {"temperature": 1.0, "top_p": 0.95, "top_k": 50}
A.load_generation_config = lambda _path: dict(GEN_CONFIG)


class _Cfg:
    model_path = "fake/checkpoint"
    rsa_defaults = None


class _FakeState:
    """Records the SamplingParams the handler builds, then replays a canned generation."""

    config = _Cfg()

    def __init__(self, acks):
        self._acks = acks
        self.sent = None

    def new_user(self):
        return 42

    async def send_one(self, msg):
        self.sent = msg

    async def wait_for_ack(self, uid):
        for a in self._acks:
            yield a


class _FakeRequest:
    headers: dict = {}


def call(prompt: str, acks, **kw) -> tuple:
    """(response dict, the SamplingParams the handler sent to the engine)."""
    st = _FakeState(acks)
    A._GLOBAL_STATE = st
    req = A.OpenAICompletionRequest(model="m", prompt=prompt, **kw)
    resp = asyncio.run(A.v1_text_completions(req, _FakeRequest()))
    return resp, st.sent.sampling_params


RAW = "<|channel>thought\nadd them<channel|>4"
resp, sp = call("2+2 =", [ack(RAW, 9, finished=True, reason="stop")], max_tokens=40)
check("object is text_completion", resp["object"] == "text_completion", str(resp.get("object")))
check("the answer is in choices[0].text, not choices[0].message",
      resp["choices"][0]["text"] == RAW and "message" not in resp["choices"][0])
check("the completion is VERBATIM — reasoning delimiters are NOT split out",
      resp["choices"][0]["text"] == RAW,
      "a split would delete the scratch with nowhere to put it: /v1/completions has no "
      "reasoning_content field")
check("finish_reason is carried through", resp["choices"][0]["finish_reason"] == "stop")
check("usage totals", resp["usage"] == {"prompt_tokens": 5, "completion_tokens": 9,
                                        "total_tokens": 14}, str(resp["usage"]))

print("  -- the reasoning backstop arms only on a prompt that is itself mid-span")
# The backstop FORCE-EMITS the close delimiter at the budget, and the scheduler caps that budget at
# 3/4 of max_tokens. Arming it on `_thinking_active` (the chat lane's test, true merely because the
# model COULD open a span) would inject `<channel|>` into a 40-token plain completion at token 30 —
# invisible on the chat lane, where the reasoning split eats it, but plain corruption in `text`.
_, sp_plain = call("The capital of France is", [ack("Paris", 1, finished=True, reason="stop")])
check("plain prompt -> backstop NOT armed", sp_plain.think_close_delim is None,
      repr(sp_plain.think_close_delim))
check("plain prompt -> _thinking_active would have armed it (this is the trap)",
      A._thinking_active(A.OpenAICompletionRequest(model="m", prompt="The capital of France is")))
_, sp_open = call("2+2 =\n<|channel>thought\n", [ack("x", 1, finished=True, reason="stop")])
check("prompt left a span OPEN -> backstop armed with the close delimiter",
      sp_open.think_close_delim == "<channel|>", repr(sp_open.think_close_delim))

print("  -- sampling goes through _resolve_sampling, exactly like the chat lane")
check("the request's explicit values win",
      (sp_open.temperature, sp_open.top_p, sp_open.top_k) == (1.0, 1.0, -1),
      f"{(sp_open.temperature, sp_open.top_p, sp_open.top_k)}")
_, sp_greedy = call("hi", [ack("x", 1, finished=True, reason="stop")],
                    temperature=0.0, top_k=1, top_p=1.0)
check("the greedy triple survives to the engine", sp_greedy.is_greedy)
# THE HAZARD, pinned: `is_greedy` is `(temperature <= 0 or top_k == 1) and top_p == 1.0` (core.py), so
# greediness depends on top_p too. On the OpenAI lanes top_p DEFAULTS to 1.0 in the request model, so
# `temperature: 0` alone is greedy today; on /generate the same field is `| None` and INHERITS
# generation_config (0.95 here), which is not. If the OpenAI fields are ever changed to `| None` to
# pick up the checkpoint's recommended sampling, `temperature: 0` silently stops being greedy and
# every byte-identity gate built on it starts measuring the sampler. This check fails first.
_, sp_t0 = call("hi", [ack("x", 1, finished=True, reason="stop")], temperature=0.0)
check("temperature:0 alone IS greedy on the OpenAI lane (top_p defaults to 1.0, not inherited)",
      sp_t0.is_greedy, f"top_p resolved to {sp_t0.top_p} (generation_config says {GEN_CONFIG['top_p']})")
gt, gp, gk = A._resolve_sampling(A.GenerateRequest(prompt="hi", max_tokens=4), "fake/checkpoint")
check("/generate DOES inherit generation_config — the two lanes genuinely differ",
      (gt, gp, gk) == (1.0, 0.95, 50), f"{(gt, gp, gk)}")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
