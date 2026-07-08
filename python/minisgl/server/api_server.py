from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.rsa.config import merge_params
from minisgl.rsa.core import RSAError, run_markovian_rsa
from minisgl.rsa.inproc import InProcessBackendClient
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False
    # TRANSPARENT CAM per-request override: True/False forces ambient auto-write on/off for THIS call,
    # overriding MINISGL_CAM_AUTO_WRITE (None = server default). Suppress learning on a read-only turn.
    cam_write: bool | None = None


class Message(BaseModel):
    # "tool" carries a tool result back into the conversation (agent loop); assistant messages may
    # carry tool_calls with content=None.
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    # Tool-calling conversation history (echoed straight into the chat template):
    tool_calls: List[dict] | None = None  # on a prior assistant turn
    tool_call_id: str | None = None  # on a "tool" turn — which call this result answers
    name: str | None = None  # tool/function name on a "tool" turn


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    # TRANSPARENT CAM per-request override: True/False forces ambient auto-write on/off for THIS call,
    # overriding MINISGL_CAM_AUTO_WRITE (None = server default). Suppress learning on a read-only turn.
    cam_write: bool | None = None

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] | str = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    # Structured output. {"type": "json_object"} -> any valid JSON; {"type": "json_schema",
    # "json_schema": {"schema": {...}}} -> conform to the schema. None -> unconstrained.
    response_format: dict | None = None

    ignore_eos: bool = False

    # Tool / function calling (OpenAI-compatible). `tools` is the list of function specs
    # ({"type":"function","function":{name,description,parameters}}); they are injected into the
    # model's chat template (Qwen3 & friends are tool-trained) and any emitted <tool_call> blocks are
    # parsed back into `tool_calls` on the response. `tool_choice`: "auto" (default) | "none" (don't
    # offer tools) | "required" | {"type":"function","function":{"name":...}} to force a call.
    tools: List[dict] | None = None
    tool_choice: str | dict | None = None

    # Per-call Markovian-RSA control (in-engine, same port). Absent / null -> ordinary single
    # completion. `true` -> run RSA with the server's --rsa-* defaults. An object patches those
    # defaults for THIS call: {n, k, t, tail_tokens, max_tokens, agg_max_tokens, temperature,
    # selection, max_concurrency, max_retries, enabled}. `false` -> force a plain completion.
    rsa: bool | dict | None = None


def _grammar_from_response_format(rf: dict | None) -> str | None:
    """Map an OpenAI ``response_format`` to a SamplingParams.grammar spec ("json" or a JSON-schema
    string), or None if unconstrained / unrecognized."""
    if not rf:
        return None
    rtype = rf.get("type")
    if rtype == "json_object":
        return "json"
    if rtype == "json_schema":
        js = rf.get("json_schema") or {}
        schema = js.get("schema", js)
        return json.dumps(schema)
    return None


def _norm_stop(stop: list | str | None) -> List[str]:
    if not stop:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _normalize_tool_args(messages: List[dict]) -> None:
    """Chat templates expect a tool call's `function.arguments` to be a *mapping* (they call
    `.items()` on it), but the OpenAI wire format carries it as a JSON *string*. When tool-call
    history is replayed by a client, parse those strings back to dicts in place so the template
    renders (else jinja raises "Can only get item pairs from a mapping")."""
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except (json.JSONDecodeError, TypeError):
                    pass


def _tools_for_template(req: "OpenAICompletionRequest") -> List[dict] | None:
    """The tool specs to hand the chat template, honoring `tool_choice`. `"none"` withholds the tools
    entirely (so the model won't call one); everything else offers them and lets the model / template
    decide (a specific `{"function": {"name": ...}}` choice is offered as a hint, best-effort)."""
    if not req.tools:
        return None
    if req.tool_choice == "none":
        return None
    return req.tools


async def _cam_auto_augment(prompt, ns=None):
    """TRANSPARENT CAM read (MINISGL_CAM_AUTO=1): fold relevant remembered facts into the request context
    so /v1/chat and /generate use CAM with NO special params. `prompt` is a chat-messages list or a raw
    string. Retrieves cosine-matched facts for the query text and prepends them as a system note (chat) or
    a short preface (raw). No-op when auto is off, no CAM runtime, or nothing confidently matches (the
    store's tau threshold keeps it quiet on unrelated prompts). Costs one extra retrieve round-trip."""
    if os.environ.get("MINISGL_CAM_AUTO") != "1":
        return prompt
    try:
        from minisgl.cam import get_cam_runtime

        rt = get_cam_runtime()
    except Exception:  # noqa: BLE001
        return prompt
    if rt is None or not hasattr(rt, "retrieve"):
        return prompt
    if isinstance(prompt, list):
        query = " ".join(str(m.get("content") or "") for m in prompt if m.get("role") in ("user", "system"))
    else:
        query = str(prompt)
    if not query.strip():
        return prompt
    try:
        facts = await rt.retrieve(query, namespace=ns)
    except Exception as e:  # noqa: BLE001
        logger.debug("CAM auto-retrieve failed: %s", e)
        return prompt
    if not facts:
        return prompt
    note = "Relevant known facts (use if helpful):\n" + "\n".join(
        f"- {f.get('subject')}: {f.get('object')}" for f in facts)
    logger.debug("CAM auto-RAG: injected %d fact(s)", len(facts))
    if isinstance(prompt, list):
        return [{"role": "system", "content": note}, *prompt]
    return note + "\n\n" + str(prompt)


def _looks_like_fact_statement(text: str) -> bool:
    """Cheap pre-filter for the auto-write path: does `text` plausibly ASSERT a durable fact? Skips the
    (expensive) extraction generation on chit-chat and questions. Conservative — biased toward returning
    True so real facts are not dropped: only rejects the clear non-assertions (empty, a question, or no
    copula at all). "Zephyrina's mother tongue is Klingon" -> True; "hi" / "how are you?" / "summarize
    this" -> False. Disable with MINISGL_CAM_WRITE_HEURISTIC=0 (always extract)."""
    if os.environ.get("MINISGL_CAM_WRITE_HEURISTIC", "1") != "1":
        return True
    t = (text or "").strip()
    if not t or t.endswith("?"):
        return False                                  # empty or a question — not an assertion
    return re.search(r"\b(is|was|are|were)\b", t, re.IGNORECASE) is not None


def _auto_write_enabled(override: bool | None) -> bool:
    """Resolve whether ambient auto-write runs THIS turn: a per-request `cam_write` (True/False) overrides
    the server default (MINISGL_CAM_AUTO_WRITE=1). Lets a client suppress learning on a given call (e.g.
    a read-only task turn over an ingested store) or force it on for a one-off."""
    if override is not None:
        return bool(override)
    return os.environ.get("MINISGL_CAM_AUTO_WRITE") == "1"


async def _cam_auto_write(text: str, override: bool | None = None, ns: str = None) -> None:
    """TRANSPARENT CAM write: model-extract durable facts from `text` (the latest user turn) and remember
    them (mode='auto', so the store's freeze/no-clobber gates apply — a curated store is not overwritten
    by conversation). Gated by the per-request `cam_write` override or MINISGL_CAM_AUTO_WRITE=1. No-op when
    off / no runtime. Best-effort: extraction failures are swallowed. A cheap fact-statement heuristic
    (_looks_like_fact_statement) skips the extraction generation on chit-chat and questions."""
    if not _auto_write_enabled(override) or not (text and text.strip()):
        return
    if not _looks_like_fact_statement(text):
        logger.debug("CAM auto-write: %r is not a fact statement; skipping extraction.", text[:60])
        return
    try:
        from minisgl.cam import get_cam_runtime

        rt = get_cam_runtime()
    except Exception:  # noqa: BLE001
        return
    if rt is None or not hasattr(rt, "extract_facts"):
        return
    try:
        for subj, obj in await rt.extract_facts(text):
            await rt.remember(subj, obj, mode="auto", namespace=ns)   # ambient -> freeze/no-clobber gates
            logger.debug("CAM auto-write: remembered %r -> %r", subj, obj)
    except Exception as e:  # noqa: BLE001
        logger.debug("CAM auto-write failed: %s", e)


# Tool-trained models emit tool calls inside `<tool_call>...</tool_call>` blocks, but the INNER format
# varies by family. We parse both we've seen:
#   (A) Hermes JSON:  {"name": "fn", "arguments": {"k": v}}
#   (B) Qwen3 XML:    <function=fn><parameter=k>v</parameter></function>
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_XML_FN_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL)


def _coerce(val: str):
    """XML params arrive as strings; coerce JSON scalars/objects (numbers, bools, arrays), else keep
    the raw string."""
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        return val


def _parse_one_tool_call(inner: str) -> Tuple[str, dict] | None:
    """Parse one <tool_call> body (either format) -> (name, arguments_dict), or None."""
    inner = inner.strip()
    if inner.startswith("{"):  # (A) Hermes JSON
        try:
            call = json.loads(inner)
            if call.get("name"):
                return call["name"], call.get("arguments", {})
        except json.JSONDecodeError:
            pass
    fn = _XML_FN_RE.search(inner)  # (B) Qwen3 XML
    if fn:
        args = {k.strip(): _coerce(v.strip()) for k, v in _XML_PARAM_RE.findall(fn.group(2))}
        return fn.group(1).strip(), args
    return None


def _parse_tool_calls(text: str, uid: int) -> Tuple[str | None, List[dict]]:
    """Extract tool calls from a completion. Returns (content, tool_calls): `content` is the text with
    the <tool_call> blocks stripped (None if nothing but calls remain), `tool_calls` is the
    OpenAI-shaped list ([] when the model didn't call a tool)."""
    tool_calls: List[dict] = []
    for i, m in enumerate(_TOOL_CALL_BLOCK_RE.finditer(text)):
        parsed = _parse_one_tool_call(m.group(1))
        if parsed is None:
            continue  # malformed block -> ignore, leave it in the text
        name, args = parsed
        tool_calls.append(
            {
                "id": f"call_{uid}_{i}",
                "type": "function",
                # OpenAI carries arguments as a JSON *string*.
                "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
            }
        )
    if not tool_calls:
        return text, []
    content = _TOOL_CALL_BLOCK_RE.sub("", text).strip()
    return (content or None), tool_calls


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        prompt_tokens = completion_tokens = 0
        finish_reason = "stop"
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output
            completion_tokens = max(completion_tokens, ack.completion_tokens)
            prompt_tokens = ack.prompt_tokens or prompt_tokens
            if ack.finish_reason:
                finish_reason = ack.finish_reason

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "chat.completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # final chunk: finish_reason + usage (OpenAI carries usage on the terminal chunk)
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


# CAM edit-plane API (/cam/*): additive, OFF by default. Only mounted when MINISGL_CAM=1, and the
# import is guarded so a server without CAM loaded still starts and /generate + /v1/* are untouched.
if os.environ.get("MINISGL_CAM") == "1":
    try:
        from .cam_api import cam_router

        app.include_router(cam_router)
        logger.info("CAM edit-plane API mounted at /cam/*")
    except Exception as e:  # noqa: BLE001 - never let CAM wiring break the base server
        logger.warning("CAM API not mounted (MINISGL_CAM=1 but import failed): %s", e)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    _cam_ns = request.headers.get("x-cam-namespace")   # #6 per-tenant/session store (None -> default)
    await _cam_auto_write(req.prompt, override=req.cam_write, ns=_cam_ns)   # ambient write (gated)
    prompt = await _cam_auto_augment(req.prompt, ns=_cam_ns)   # TRANSPARENT CAM read (no-op unless enabled)
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()

    # In-engine Markovian RSA (opt-in per call via the `rsa` field). When enabled, the WHOLE
    # expand -> aggregate(K-subsets over T rounds) -> select loop runs server-side: each rollout is
    # an internal generation (InProcessBackendClient drives the same front-end primitive, no HTTP),
    # so N rollouts fan out across the scheduler / DP-EP replicas exactly like concurrent requests.
    # `merge_params` returns None for an absent/false/disabled `rsa`, which falls through to the
    # ordinary single-completion path below.
    rsa_params = merge_params(state.config.rsa_defaults, req.rsa) if req.rsa is not None else None
    if rsa_params is not None:
        if not req.messages:
            return JSONResponse(
                status_code=400,
                content={"error": "RSA requires `messages` (chat format), not a raw `prompt`"},
            )
        messages = [msg.model_dump() for msg in req.messages]
        client = InProcessBackendClient(state, state.config.model_path)
        try:
            result = await run_markovian_rsa(client, rsa_params, messages, req.model)
        except RSAError as e:
            return JSONResponse(status_code=502, content={"error": f"RSA failed: {e}"})
        finally:
            await client.close()
        return {
            "id": f"chatcmpl-rsa-{state.uid_counter}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.final_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
            "rsa": {
                "selection_method": result.selection_method,
                "n": rsa_params.n,
                "k": rsa_params.k,
                "t": rsa_params.t,
                "tail_tokens": rsa_params.tail_tokens,
                "temperature": rsa_params.temperature,
                "top_p": rsa_params.top_p,
                "top_k": rsa_params.top_k,
                "rounds": len(result.rounds),
                "population": len(result.population),
                "n_requests": result.usage.n_requests,
                "vote_detail": result.vote_detail,
            },
        }

    if req.messages:
        # exclude_none so tool-calling turns render cleanly (assistant content=None + tool_calls; a
        # "tool" result turn) — the chat template checks for absent keys, not explicit nulls.
        prompt = [msg.model_dump(exclude_none=True) for msg in req.messages]
        _normalize_tool_args(prompt)  # tool_call arguments: JSON string -> dict for the template
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TRANSPARENT CAM: learn facts from the latest user turn, then fold relevant known facts into context.
    if isinstance(prompt, list):
        _last_user = next((m.get("content") for m in reversed(prompt) if m.get("role") == "user"), None)
    else:
        _last_user = prompt
    _cam_ns = request.headers.get("x-cam-namespace")   # #6 per-tenant/session store (None -> default)
    await _cam_auto_write(_last_user or "", override=req.cam_write, ns=_cam_ns)   # gated ambient write
    prompt = await _cam_auto_augment(prompt, ns=_cam_ns)     # TRANSPARENT CAM read (no-op unless enabled)

    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            tools=_tools_for_template(req),
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                stop=_norm_stop(req.stop),
                grammar=_grammar_from_response_format(req.response_format),
            ),
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    prompt_tokens = completion_tokens = 0
    finish_reason = "stop"
    async for ack in state.wait_for_ack(uid):
        full_content += ack.incremental_output
        completion_tokens = max(completion_tokens, ack.completion_tokens)
        prompt_tokens = ack.prompt_tokens or prompt_tokens
        if ack.finish_reason:
            finish_reason = ack.finish_reason
        if ack.finished:
            break

    # Tool calling: if tools were offered, parse any <tool_call> blocks the model emitted into
    # OpenAI-shaped tool_calls and flip finish_reason. No tools offered -> plain text (untouched).
    message = {"role": "assistant", "content": full_content}
    if req.tools and finish_reason != "length":
        content, tool_calls = _parse_tool_calls(full_content, uid)
        if tool_calls:
            message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
            finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )



async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
