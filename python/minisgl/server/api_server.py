from __future__ import annotations

import ast
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
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.utils import load_generation_config
from minisgl.rsa.config import merge_params
from minisgl.rsa.core import RSAError, run_markovian_rsa
from minisgl.rsa.inproc import InProcessBackendClient
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    StatsFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger

from .metrics import BackendSnapshot, FrontendMetrics
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field, model_validator
from starlette.background import BackgroundTask

from .args import ServerArgs
from .reasoning import get_reasoning_parser

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
    # Symmetric per-request override for the TRANSPARENT auto-READ (retrieve+augment): True/False forces
    # it on/off for THIS call, overriding MINISGL_CAM_AUTO (None = server default). Lets a caller (or an
    # A/B harness) suppress the retrieve round-trip on a given turn.
    cam_read: bool | None = None


class Message(BaseModel):
    # "tool" carries a tool result back into the conversation (agent loop); assistant messages may
    # carry tool_calls with content=None.
    role: Literal["system", "user", "assistant", "tool"]
    # OpenAI-spec content: either a plain string OR an array of content parts
    # ([{"type":"text","text":...}, {"type":"image_url",...}]). Middleware (prompt-caching / memory
    # injection) commonly rewrites a string into the array form — spec-legal, so accept it and
    # flatten the text parts to a string below (rejecting it was a 422).
    content: "str | List[dict] | None" = None
    # Tool-calling conversation history (echoed straight into the chat template):
    tool_calls: List[dict] | None = None  # on a prior assistant turn
    tool_call_id: str | None = None  # on a "tool" turn — which call this result answers
    name: str | None = None  # tool/function name on a "tool" turn

    @model_validator(mode="after")
    def _flatten_content_parts(self) -> "Message":
        """OpenAI array-of-parts content -> a plain string the chat template consumes. Concatenates the
        `text` parts (in order); non-text parts (e.g. image_url) are ignored for this text model. A
        plain-string content is left untouched."""
        if isinstance(self.content, list):
            self.content = "".join(
                p.get("text", "") for p in self.content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        return self


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    # OpenAI renamed `max_tokens` -> `max_completion_tokens` (max_tokens is deprecated but still sent
    # by older clients). Accept BOTH and coalesce: max_tokens ?? max_completion_tokens ?? 16. Kept as
    # `int | None` so the validator can tell "unset" from an explicit value; downstream code reads the
    # coalesced int `max_tokens`.
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0

    # TRANSPARENT CAM per-request override: True/False forces ambient auto-write on/off for THIS call,
    # overriding MINISGL_CAM_AUTO_WRITE (None = server default). Suppress learning on a read-only turn.
    cam_write: bool | None = None
    # Symmetric per-request override for the TRANSPARENT auto-READ (retrieve+augment): True/False forces
    # it on/off for THIS call, overriding MINISGL_CAM_AUTO (None = server default).
    cam_read: bool | None = None

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

    # Reasoning / "thinking" control for chain-of-thought models. `chat_template_kwargs` is forwarded
    # verbatim to the chat template (vLLM-compatible), e.g. {"enable_thinking": false} to disable
    # thinking. `enable_thinking` is a convenience alias folded into chat_template_kwargs. When
    # thinking is on (the reasoning-model default), the completion's `<think>…</think>` scratch is
    # split out into `reasoning_content` on the response (see --reasoning-parser).
    chat_template_kwargs: dict | None = None
    enable_thinking: bool | None = None

    # Reasoning BUDGET (backstop): for a grammar-constrained + thinking request, cap the free reasoning
    # phase at this many tokens — after it, the scheduler force-emits the reasoning-close token so the
    # JSON schema engages (a rambling model that never emits a clean `</think>` still yields JSON).
    # `reasoning_max_tokens` is the explicit token count; the OpenAI `reasoning_effort`
    # ("low"/"medium"/"high") maps to a token budget; a `reasoning_max_tokens` in `chat_template_kwargs`
    # is also honored. Unset -> the server's MINISGL_THINK_BUDGET default.
    reasoning_max_tokens: int | None = None
    reasoning_effort: str | None = None

    # Per-call Markovian-RSA control (in-engine, same port). Absent / null -> ordinary single
    # completion. `true` -> run RSA with the server's --rsa-* defaults. An object patches those
    # defaults for THIS call: {n, k, t, tail_tokens, max_tokens, agg_max_tokens, temperature,
    # selection, max_concurrency, max_retries, enabled}. `false` -> force a plain completion.
    rsa: bool | dict | None = None

    @model_validator(mode="after")
    def _coalesce_max_tokens(self) -> "OpenAICompletionRequest":
        """max_tokens ?? max_completion_tokens ?? 16 — accept the OpenAI-renamed field. After this,
        `self.max_tokens` is always the resolved int the rest of the code reads."""
        if self.max_tokens is None:
            self.max_tokens = self.max_completion_tokens if self.max_completion_tokens is not None else 16
        return self


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


def _grammar_from_tools(req: "OpenAICompletionRequest") -> str | None:
    """Build a JSON-schema grammar for a FORCED tool call so `tools` gets the same constrained-decoding
    treatment `response_format` gets — the final answer is masked to a complete, valid call (name +
    schema-checked arguments) that terminates, instead of the model drafting prose to the token cap.

    Applies only when the request FORCES a call: `tool_choice: "required"` (any tool) or a specific
    `{"type":"function","function":{"name": …}}`. For `"auto"`/`"none"`/absent it returns None — auto
    must stay free to NOT call (and the XML wrapper parser handles those). The grammar constrains the
    output to `{"name": <one of the allowed tools>, "arguments": <that tool's parameters schema>}`."""
    tools = req.tools
    if not tools:
        return None
    choice = req.tool_choice
    forced_name: str | None = None
    if isinstance(choice, dict):
        forced_name = (choice.get("function") or {}).get("name")
    elif choice != "required":
        return None  # "auto" / "none" / None -> not forced (see _structural_tag_from_tools)
    if _TOOL_FORMAT == "zaya_xml":
        ebnf = _zaya_xml_grammar(tools, forced_name)  # native <zyphra_tool_call> XML, not JSON
        return json.dumps({"__ebnf__": ebnf}) if ebnf else None
    variants = _tool_call_variants(tools, forced_name)
    if not variants:
        return None
    return json.dumps(variants[0] if len(variants) == 1 else {"anyOf": variants})


def _ebnf_lit(s: str) -> str:
    """GBNF-escape a Python string into a double-quoted EBNF terminal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _zaya_xml_grammar(tools: List[dict], forced_name: str | None = None) -> str | None:
    """EBNF constraining ZAYA's NATIVE tool call to its trained format:
        <zyphra_tool_call>\\n<function=NAME>\\n(<parameter=P>\\nVALUE\\n</parameter>\\n)*</function>\\n</zyphra_tool_call>
    Function NAME is constrained to the allowed tools and parameter names to the known params (structure
    is guaranteed parseable by `_parse_tool_calls`); VALUES stay permissive ([^<]*) so the model isn't
    boxed on content. Any-order/any-subset params (a fixed order would reject valid calls). None if no
    tool matches. Respects ZAYA's RL training instead of forcing an unfamiliar JSON shape."""
    fns: List[str] = []
    pnames: set[str] = set()
    for t in tools:
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name or (forced_name and name != forced_name):
            continue
        fns.append(name)
        pnames.update((fn.get("parameters") or {}).get("properties", {}) or {})
    if not fns:
        return None
    fname_alt = " | ".join(_ebnf_lit(n) for n in fns)
    pname_alt = " | ".join(_ebnf_lit(p) for p in sorted(pnames)) if pnames else _ebnf_lit("_")
    return "\n".join([
        'root ::= "<zyphra_tool_call>\\n<function=" fname ">\\n" params "</function>\\n</zyphra_tool_call>"',
        f"fname ::= {fname_alt}",
        "params ::= param*",
        'param ::= "<parameter=" pname ">\\n" pval "\\n</parameter>\\n"',
        f"pname ::= {pname_alt}",
        "pval ::= [^<]*",
    ])


def _tool_call_variants(tools: List[dict], forced_name: str | None = None) -> List[dict]:
    """A JSON-schema per allowed tool: {name: const, arguments: that tool's parameters}."""
    variants = []
    for t in tools:
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name or (forced_name and name != forced_name):
            continue
        variants.append({
            "type": "object",
            "properties": {
                "name": {"const": name},
                "arguments": fn.get("parameters") or {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        })
    return variants


# Tool-call wrappers we constrain in `auto` mode: a trigger opener -> JSON call -> closer. The model
# stays free to answer in prose (no trigger); if it opens one of these, xgrammar forces the wrapped
# content to a schema-valid JSON call. MUST include ZAYA's native `<zyphra_tool_call>` — otherwise the
# trigger never fires for ZAYA (`<tool_call>` is NOT a substring of `<zyphra_tool_call>`), the auto
# grammar is effectively OFF, and the model free-forms into unparseable tool calls (the explore_do
# format chaos). Structural tags are JSON-schema-only, so the wrapped content is forced to JSON (ZAYA
# emits valid JSON when constrained — cf. the forced path); the XML `<function=…>` parser recovers any
# native-XML that still slips through. For a GUARANTEED native-XML forced call see `_zaya_xml_grammar`
# (MINISGL_TOOL_FORMAT=zaya_xml).
_TOOL_STRUCT_WRAPPERS = (
    ("<zyphra_tool_call>", "</zyphra_tool_call>"),
    ("<tool_call>", "</tool_call>"),
    ("<tools>", "</tools>"),
)
# Model-native forced-tool-call format. "json" (default) -> `{"name": …, "arguments": …}`; "zaya_xml" ->
# ZAYA's trained `<zyphra_tool_call><function=NAME><parameter=P>…</parameter></function></zyphra_tool_call>`
# via a custom EBNF (`_zaya_xml_grammar`). Set per serve (the ZAYA compose service sets zaya_xml); JSON is
# correct for every other family, so the default is a no-op change.
_TOOL_FORMAT = os.environ.get("MINISGL_TOOL_FORMAT", "json")


def _structural_tag_from_tools(req: "OpenAICompletionRequest") -> str | None:
    """`tool_choice: "auto"` (or default): build an xgrammar STRUCTURAL TAG so the model may answer in
    prose OR call a tool, and when it opens a recognized tool-call wrapper the arguments are forced to
    the tool's schema. Returns None for none/required/specific (handled by _grammar_from_tools) or no
    tools. Strictly >= the un-constrained auto path (free text is unaffected)."""
    tools = req.tools
    if not tools or (req.tool_choice not in (None, "auto")):
        return None
    variants = _tool_call_variants(tools)
    if not variants:
        return None
    call_schema = variants[0] if len(variants) == 1 else {"anyOf": variants}
    tags = [{"begin": b, "schema": call_schema, "end": e} for b, e in _TOOL_STRUCT_WRAPPERS]
    triggers = [b for b, _ in _TOOL_STRUCT_WRAPPERS]
    return json.dumps({"__structural_tag__": {"tags": tags, "triggers": triggers}})


def _parse_json_tool_call(body: str, uid: int) -> dict | None:
    """Parse a grammar-constrained tool call — a bare JSON object `{"name": …, "arguments": {…}}`
    (what `_grammar_from_tools` forces) — into an OpenAI `tool_calls` entry. None if it isn't that."""
    try:
        call = json.loads(body.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(call, dict) or not call.get("name"):
        return None
    args = call.get("arguments", {})
    return {
        "id": f"call_{uid}_0",
        "type": "function",
        "function": {"name": call["name"], "arguments": args if isinstance(args, str) else json.dumps(args)},
    }


def _grammar_think_gate_delim(req: "OpenAICompletionRequest") -> str | None:
    """Reasoning + structured output: when a request is BOTH grammar-constrained AND has thinking
    active, return the reasoning parser's close delimiter (e.g. "</think>") so the scheduler gates
    grammar enforcement until reasoning ends. Applying a JSON schema from token 0 would otherwise
    mask out the `<think>…</think>` scratch a reasoning model opens with (→ truncated / CoT-leaked
    output). Returns None — grammar from token 0, unchanged — when the request is unconstrained,
    thinking is off, or no reasoning parser is configured (non-reasoning models). Generic/config-
    driven: the delimiter comes from `--reasoning-parser` (auto/qwen3/deepseek/glm), no model-name
    branch."""
    if _grammar_from_response_format(req.response_format) is None:
        return None
    if not _thinking_active(req):
        return None
    parser = _reasoning_parser()
    if parser is None:
        return None
    return parser.end_token


def _reasoning_close_delim(req: "OpenAICompletionRequest") -> str | None:
    """The reasoning parser's close delimiter (e.g. "</think>") when thinking is active — UNCONDITIONAL,
    unlike `_grammar_think_gate_delim` which returns it only for grammar-constrained requests. RSA uses
    this to β-bound reasoning on EVERY rollout (the exploration rollouts are grammar-free), so the
    scheduler force-closes </think> at the budget and each rollout yields an answer instead of running
    reasoning to max_tokens. None when thinking is off or no reasoning parser is configured."""
    if not _thinking_active(req):
        return None
    parser = _reasoning_parser()
    return parser.end_token if parser is not None else None


def _resolve_think_budget(req: "OpenAICompletionRequest") -> int | None:
    """Per-request reasoning-token budget for the think-gate backstop, or None to use the server's
    MINISGL_THINK_BUDGET default. Precedence: explicit `reasoning_max_tokens` > the same key inside
    `chat_template_kwargs` > OpenAI `reasoning_effort` (low/medium/high -> a token budget). Only takes
    effect for grammar-constrained + thinking requests (the scheduler ignores it otherwise)."""
    if isinstance(req.reasoning_max_tokens, int) and req.reasoning_max_tokens > 0:
        return req.reasoning_max_tokens
    ck = req.chat_template_kwargs or {}
    ck_budget = ck.get("reasoning_max_tokens")
    if isinstance(ck_budget, int) and ck_budget > 0:
        return ck_budget
    if req.reasoning_effort:
        return {"low": 256, "medium": 1024, "high": 4096}.get(req.reasoning_effort.lower())
    return None


def _norm_stop(stop: list | str | None) -> List[str]:
    if not stop:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _resolve_sampling(req: "OpenAICompletionRequest", model_path: str) -> tuple:
    """Effective (temperature, top_p, top_k): the request value when the client set it, else the
    model author's `generation_config.json` default, else the neutral default. Lets a bare request
    inherit the model's recommended sampling (e.g. GLM-4.x top_p 0.95 / top_k 50) instead of the
    generic 1.0 / -1 that can push reasoning models toward degenerate output."""
    gen = load_generation_config(model_path)
    temperature = req.temperature if req.temperature is not None else float(gen.get("temperature", 1.0))
    top_p = req.top_p if req.top_p is not None else float(gen.get("top_p", 1.0))
    top_k = req.top_k if req.top_k is not None else int(gen.get("top_k", -1) or -1)
    return temperature, top_p, top_k


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


def _resolve_chat_template_kwargs(req: "OpenAICompletionRequest") -> dict | None:
    """Merge the request's `chat_template_kwargs` with the `enable_thinking` convenience alias into
    the kwargs forwarded to `apply_chat_template`. None -> template defaults (thinking ON for Qwen3)."""
    kwargs = dict(req.chat_template_kwargs or {})
    if req.enable_thinking is not None and "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = req.enable_thinking
    return kwargs or None


def _thinking_active(req: "OpenAICompletionRequest") -> bool:
    """Whether reasoning is expected in the output (thinking mode engaged). Governs streaming reasoning
    routing. Default True (reasoning models open `<think>` in the generation prompt); explicit
    enable_thinking=False (top-level or in chat_template_kwargs) turns it off."""
    if req.enable_thinking is False:
        return False
    if (req.chat_template_kwargs or {}).get("enable_thinking") is False:
        return False
    return True


_REASONING_PARSER = None
_REASONING_PARSER_SET = False


def _reasoning_parser():
    """Cached ReasoningParser built from the server's --reasoning-parser (None when disabled)."""
    global _REASONING_PARSER, _REASONING_PARSER_SET
    if not _REASONING_PARSER_SET:
        cfg = get_global_state().config
        _REASONING_PARSER = get_reasoning_parser(getattr(cfg, "reasoning_parser", "auto"))
        _REASONING_PARSER_SET = True
    return _REASONING_PARSER


async def _cam_auto_augment(prompt, ns=None, override=None):
    """TRANSPARENT CAM read (MINISGL_CAM_AUTO=1): fold relevant remembered facts into the request context
    so /v1/chat and /generate use CAM with NO special params. `prompt` is a chat-messages list or a raw
    string. Retrieves cosine-matched facts for the query text and prepends them as a system note (chat) or
    a short preface (raw). No-op when auto is off, no CAM runtime, or nothing confidently matches (the
    store's tau threshold keeps it quiet on unrelated prompts). Costs one extra retrieve round-trip.
    `override` (per-request cam_read) forces on/off, else the MINISGL_CAM_AUTO env default."""
    enabled = override if override is not None else (os.environ.get("MINISGL_CAM_AUTO") == "1")
    if not enabled:
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
        _t0 = time.perf_counter()
        facts = await rt.retrieve(query, namespace=ns)
        if os.environ.get("MINISGL_CAM_RETRIEVE_PROF") == "1":
            logger.info_rank0("[cam-prof] augment retrieve_wall=%.1fms facts=%d qlen=%d",
                              (time.perf_counter() - _t0) * 1e3, len(facts), len(query))
    except Exception as e:  # noqa: BLE001
        logger.debug("CAM auto-retrieve failed: %s", e)
        return prompt
    if not facts:
        return prompt
    note = "Relevant known facts (use if helpful):\n" + "\n".join(
        f"- {f.get('subject')}: {f.get('object')}" for f in facts)
    logger.debug("CAM auto-RAG: injected %d fact(s)", len(facts))
    if isinstance(prompt, list):
        # Merge the note into the request's OWN leading system message when it has one. Prepending a
        # SECOND system message produces two system turns, which templates that require the system turn
        # first reject (Qwen: "System message must be at the beginning") — and that used to crash the
        # tokenizer worker. Otherwise prepend a fresh system message.
        if prompt and isinstance(prompt[0], dict) and prompt[0].get("role") == "system":
            merged = note + "\n\n" + str(prompt[0].get("content") or "")
            return [{**prompt[0], "content": merged}, *prompt[1:]]
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


_CAM_WRITE_TASKS: set = set()   # strong refs so fire-and-forget write tasks aren't GC'd mid-flight


def _schedule_cam_auto_write(text: str, override: bool | None = None, ns: str = None) -> None:
    """Run the ambient CAM write OFF the request's critical path (default). The write may cost a
    fact-extraction generation (~2s), and the user's response must not wait on learning — so unless
    MINISGL_CAM_WRITE_SYNC=1, schedule it as a background task (self-contained: no request handle, so
    the disconnect-abort fire-and-forget hazard doesn't apply). Errors are swallowed (best-effort)."""
    if not _auto_write_enabled(override) or not (text and text.strip()):
        return
    try:
        task = asyncio.ensure_future(_cam_auto_write(text, override=override, ns=ns))
    except RuntimeError:      # no running loop (shouldn't happen in a handler) -> skip silently
        return
    _CAM_WRITE_TASKS.add(task)
    task.add_done_callback(_CAM_WRITE_TASKS.discard)


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


# Tool-trained models emit tool calls inside a wrapper block, but both the WRAPPER tag and the INNER
# format vary by family. Wrappers we've seen: `<tool_call>` (Hermes/Qwen3) and `<zyphra_tool_call>`
# (ZAYA/Zyphra). Inner formats we parse:
#   (A) Hermes JSON:  {"name": "fn", "arguments": {"k": v}}
#   (B) Qwen3 XML:    <function=fn><parameter=k>v</parameter></function>  (ZAYA nests this in its wrapper)
_TOOL_WRAPPERS = ("zyphra_tool_call", "tool_call", "tools")
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<(?:" + "|".join(_TOOL_WRAPPERS) + r")>\s*(.*?)\s*</(?:" + "|".join(_TOOL_WRAPPERS) + r")>",
    re.DOTALL,
)
# A bare `<function=…></function>` block (Qwen3 XML emitted WITHOUT a `<tool_call>` wrapper). Kept in
# lock-step with the streaming parser, which also accepts the unwrapped opener.
_BARE_FN_BLOCK_RE = re.compile(r"<function=[^>\s]+\s*>.*?</function>", re.DOTALL)
_XML_FN_RE = re.compile(r"<function=([^>\s(]+)\s*>(.*?)</function>", re.DOTALL)
_XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL)
# ZAYA DEVIATION recovery. Under RSA / long-context / quant the model drops its trained
# `<function=NAME><parameter=X>v</parameter></function>` form and emits a self-contained inline
# python-call instead — `<function=NAME(a='x', b=1)>` with args in parens, NO </function> and NO
# <parameter> blocks — often wrapped in SQUARE `[zyphra_tool_call]` rather than angle
# `<zyphra_tool_call>` (and frequently with no closer at all). Parse that so a recoverable-but-malformed
# call still lands as a real tool_call instead of leaking to `content`. The canonical parsers above stay
# the primary path; this is a fallback. `[/?zyphra_tool_call]` square wrappers are normalized to angle
# in `_parse_tool_calls` before the block scan.
_INLINE_FN_RE = re.compile(r"<function\s*=\s*([A-Za-z_][\w.]*)\s*\((.*?)\)\s*/?>", re.DOTALL)
_SQUARE_WRAP_RE = re.compile(r"\[(/?(?:" + "|".join(_TOOL_WRAPPERS) + r"))\]")
# Orphan wrapper open/close tags left in content after a call is parsed (e.g. the model emitted an opener
# but no closer around an inline function) — strip them so `content` isn't polluted with dangling markup.
_ORPHAN_WRAP_RE = re.compile(r"</?(?:" + "|".join(_TOOL_WRAPPERS) + r")>")
_KV_FALLBACK_RE = re.compile(r"([A-Za-z_]\w*)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,]+))")


def _parse_pycall_args(argstr: str) -> dict:
    """Parse inline python-call kwargs (`a='x', b=1, c=[1,2]`) into a dict. Prefer `ast` (safe literal
    eval per value); fall back to a permissive key=value regex when the args aren't clean literals
    (truncated string, unquoted value). Positional args are ignored (ZAYA emits kwargs)."""
    argstr = argstr.strip()
    if not argstr:
        return {}
    try:
        call = ast.parse(f"_f({argstr})", mode="eval").body
        out: dict = {}
        ok = True
        for kw in getattr(call, "keywords", []):
            if kw.arg is None:
                continue
            try:
                out[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                ok = False
                break
        if ok and out:
            return out
    except Exception:
        pass
    out = {}
    for m in _KV_FALLBACK_RE.finditer(argstr):
        v = next((g for g in m.groups()[1:] if g is not None), "")
        out[m.group(1)] = _coerce(v.strip())
    return out


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
    fn = _XML_FN_RE.search(inner)  # (B) Qwen3 XML: <function=NAME>…<parameter=…>…</function>
    if fn:
        args = {k.strip(): _coerce(v.strip()) for k, v in _XML_PARAM_RE.findall(fn.group(2))}
        return fn.group(1).strip(), args
    inl = _INLINE_FN_RE.search(inner)  # (C) ZAYA deviation: inline <function=NAME(a='x', b=1)>
    if inl:
        return inl.group(1).strip(), _parse_pycall_args(inl.group(2))
    return None


def _parse_tool_calls(text: str, uid: int) -> Tuple[str | None, List[dict]]:
    """Extract tool calls from a completion. Returns (content, tool_calls): `content` is the text with
    the <tool_call> blocks stripped (None if nothing but calls remain), `tool_calls` is the
    OpenAI-shaped list ([] when the model didn't call a tool)."""
    tool_calls: List[dict] = []

    def _add(inner: str) -> None:
        parsed = _parse_one_tool_call(inner)
        if parsed is None:
            return  # malformed block -> ignore, leave it in the text
        name, args = parsed
        tool_calls.append(
            {
                "id": f"call_{uid}_{len(tool_calls)}",
                "type": "function",
                # OpenAI carries arguments as a JSON *string*.
                "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)},
            }
        )

    # Normalize ZAYA's occasional SQUARE `[zyphra_tool_call]` wrapper to the angle form so the block
    # regex catches it (a deviation from the trained `<zyphra_tool_call>`; if the model also dropped the
    # closer, the bare/inline scans below still recover the inner `<function=…>`).
    text = _SQUARE_WRAP_RE.sub(r"<\1>", text)
    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        _add(m.group(1))
    # Bare blocks in whatever text remains after removing the wrapped blocks (so an inner block is not
    # double-counted): canonical `<function=…></function>`, then the inline `<function=NAME(...)>` form.
    remainder = _TOOL_CALL_BLOCK_RE.sub("", text)
    for m in _BARE_FN_BLOCK_RE.finditer(remainder):
        _add(m.group(0))
    for m in _INLINE_FN_RE.finditer(_BARE_FN_BLOCK_RE.sub("", remainder)):
        _add(m.group(0))
    if not tool_calls:
        return text, []
    content = _INLINE_FN_RE.sub("", _BARE_FN_BLOCK_RE.sub("", _TOOL_CALL_BLOCK_RE.sub("", text)))
    content = _ORPHAN_WRAP_RE.sub("", content).strip()  # drop any dangling wrapper opener/closer
    return (content or None), tool_calls


# --- streaming tool-call parsing ------------------------------------------------------------------
# The block openers we recognise, mapped to their closers. `<tool_call>` wraps either inner format
# (Hermes JSON or Qwen3 `<function=…>` XML); a bare `<function=…>` (no wrapper) is also accepted so
# the parser degrades to whatever the model actually emits. Detection mirrors the reasoning streamer:
# text before any opener flows through as `content`; once inside a block the markup is withheld and,
# on the closing tag, re-emitted as OpenAI streaming `delta.tool_calls`.
_TOOL_OPENERS = ("<tool_call>", "<zyphra_tool_call>", "<tools>", "<function=")
_TOOL_CLOSERS = {
    "<tool_call>": "</tool_call>",
    "<zyphra_tool_call>": "</zyphra_tool_call>",
    "<tools>": "</tools>",
    "<function=": "</function>",
}


def _earliest_opener(text: str) -> Tuple[int, str | None]:
    """Index + token of the earliest complete tool-block opener in ``text`` (``(-1, None)`` if none)."""
    best_idx, best_tok = -1, None
    for tok in _TOOL_OPENERS:
        j = text.find(tok)
        if j != -1 and (best_idx == -1 or j < best_idx):
            best_idx, best_tok = j, tok
    return best_idx, best_tok


def _opener_partial_len(text: str) -> int:
    """Largest k>0 such that ``text`` ends with a *strict* prefix of some opener (a start marker
    straddling a streaming boundary). 0 when no suffix could begin an opener. Complete openers are
    handled by ``_earliest_opener`` before this is consulted."""
    best = 0
    for tok in _TOOL_OPENERS:
        for k in range(min(len(text), len(tok) - 1), 0, -1):
            if text.endswith(tok[:k]):
                best = max(best, k)
                break
    return best


class ToolCallStreamState:
    """Incremental tool-call splitter for the streaming path — the tool-call analogue of
    ``ReasoningStreamState``. Feed each *content* chunk (post reasoning-split); get back
    ``(content_delta, tool_deltas)`` where ``content_delta`` is text to stream verbatim as
    ``delta.content`` (None if none this chunk) and ``tool_deltas`` is a list of OpenAI streaming
    ``delta.tool_calls`` entries (each a dict to wrap as its own chunk).

    A tool call is emitted as two deltas: an opener carrying ``index``/``id``/``type``/
    ``function.name`` (empty arguments), then the full ``function.arguments`` JSON string as one
    fragment. Arguments are not streamed token-by-token because the Qwen3 XML form only yields a
    well-formed JSON object once the whole block is parsed; a single complete fragment reassembles
    identically on any OpenAI client. Sequential blocks increment ``index``."""

    def __init__(self, uid: int) -> None:
        self.uid = uid
        self.buf = ""            # partial opener (outside a block) OR accumulating block body (inside)
        self.in_tool = False
        self.opener: str | None = None
        self.next_index = 0
        self.emitted = False     # any tool call emitted -> finish_reason becomes "tool_calls"

    def _parse_block(self, block: str) -> Tuple[str, dict] | None:
        if self.opener in ("<tool_call>", "<zyphra_tool_call>", "<tools>"):
            closer = _TOOL_CLOSERS[self.opener]
            inner = block[len(self.opener):-len(closer)]
            return _parse_one_tool_call(inner)
        return _parse_one_tool_call(block)  # <function=…></function>, regex finds the fn tag

    def _emit_call(self, block: str) -> List[dict]:
        parsed = self._parse_block(block)
        if parsed is None:
            return []  # malformed block -> drop it (never leak markup into content)
        name, args = parsed
        args_str = args if isinstance(args, str) else json.dumps(args)
        i = self.next_index
        self.next_index += 1
        self.emitted = True
        return [
            {"index": i, "id": f"call_{self.uid}_{i}", "type": "function",
             "function": {"name": name, "arguments": ""}},
            {"index": i, "function": {"arguments": args_str}},
        ]

    def push(self, delta: str) -> Tuple[str | None, List[dict]]:
        content_parts: List[str] = []
        tool_deltas: List[dict] = []
        text = self.buf + delta
        self.buf = ""
        while text:
            if not self.in_tool:
                idx, opener = _earliest_opener(text)
                if idx == -1:
                    keep = _opener_partial_len(text)
                    if keep:
                        content_parts.append(text[:-keep])
                        self.buf = text[-keep:]
                    else:
                        content_parts.append(text)
                    break
                if idx > 0:
                    content_parts.append(text[:idx])
                self.in_tool = True
                self.opener = opener
                text = text[idx:]  # keep the opener token as the head of the block buffer
            else:
                closer = _TOOL_CLOSERS[self.opener]  # type: ignore[index]
                cidx = text.find(closer)
                if cidx == -1:
                    self.buf = text  # block still open; hold the whole body
                    break
                block = text[: cidx + len(closer)]
                text = text[cidx + len(closer):]
                tool_deltas.extend(self._emit_call(block))
                self.in_tool = False
                self.opener = None
        return ("".join(content_parts) or None), tool_deltas

    def flush(self) -> Tuple[str | None, List[dict]]:
        """At stream end: an unclosed block (truncated mid-call) is dropped; a buffered partial opener
        turned out to be literal ``content`` and is emitted."""
        if self.in_tool:
            self.buf, self.in_tool, self.opener = "", False, None
            return None, []
        if self.buf:
            out, self.buf = self.buf, ""
            return out, []
        return None, []


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
    metrics: FrontendMetrics = field(default_factory=FrontendMetrics)
    # Strong refs to fire-and-forget cleanup tasks: a bare asyncio.create_task can be garbage-collected
    # before it runs (the disconnect-abort bug), so keep the task alive until it completes.
    _bg_tasks: set = field(default_factory=set)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        self.metrics.on_request_start(uid)
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            # Scheduler metrics snapshot (piggybacked on the detokenizer link) — feed /metrics, no uid.
            if isinstance(msg, StatsFrontendMsg):
                self.metrics.update_backend(
                    BackendSnapshot(
                        dp_rank=msg.dp_rank,
                        spec_draft_tokens=msg.spec_draft_tokens,
                        spec_accepted_tokens=msg.spec_accepted_tokens,
                        spec_emitted_tokens=msg.spec_emitted_tokens,
                        spec_steps=msg.spec_steps,
                        running_requests=msg.running_requests,
                        waiting_requests=msg.waiting_requests,
                        kv_tokens_total=msg.kv_tokens_total,
                        kv_tokens_used=msg.kv_tokens_used,
                        gdn_slots_total=msg.gdn_slots_total,
                        gdn_slots_used=msg.gdn_slots_used,
                        cam_facts=msg.cam_facts,
                        cam_namespaces=msg.cam_namespaces,
                        cam_evicted=msg.cam_evicted,
                        cam_max_bank_load=msg.cam_max_bank_load,
                        cam_crowded_banks=msg.cam_crowded_banks,
                        cam_recovered_from_backup=msg.cam_recovered_from_backup,
                        cam_index_nn_cos_max=msg.cam_index_nn_cos_max,
                        cam_last_save_age_s=msg.cam_last_save_age_s,
                    )
                )
                continue
            for reply in _unwrap_msg(msg):
                if reply.uid not in self.ack_map:
                    continue
                self.metrics.on_reply(
                    reply.uid,
                    reply.completion_tokens,
                    reply.prompt_tokens,
                    bool(reply.incremental_output),
                    reply.finished,
                )
                self.ack_map[reply.uid].append(reply)
                self.event_map[reply.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    def _spawn_bg(self, coro) -> None:
        """Schedule a fire-and-forget coroutine while holding a strong reference to its task, so the
        event loop can't garbage-collect it mid-flight (the disconnect-abort leak)."""
        t = asyncio.ensure_future(coro)
        self._bg_tasks.add(t)
        t.add_done_callback(self._bg_tasks.discard)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]
        finished = False
        try:
            while True:
                await event.wait()
                event.clear()

                pending = self.ack_map[uid]
                self.ack_map[uid] = []
                ack = None
                for ack in pending:
                    yield ack
                if ack and ack.finished:
                    finished = True
                    break
        finally:
            # GUARANTEED terminal cleanup on ANY exit — normal finish, client disconnect (GeneratorExit
            # when the response body iterator is aclose()d), or a mid-stream error. Every streaming path
            # (stream_generate, stream_chat_completions, and their RSA/plain callers) funnels through
            # here, so this is the one place that frees the per-request state and keeps
            # minisgl_requests_inflight honest. Previously the del below ran ONLY on a normal break, so a
            # disconnect leaked ack_map/event_map, and the metrics _req record leaked too (its only other
            # release was a fire-and-forget abort_user task that could be GC'd) — inflight then climbed
            # monotonically + the dicts grew unbounded over a long run. Sync (no await): safe under
            # GeneratorExit. If the request never finished (client bailed mid-generation), reconcile the
            # metrics (on_abort pops _req + counts the abort iff still present — idempotent, never
            # double-counts a finished req) and tell the backend to abort so it stops wasting compute + KV.
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)
            if not finished:
                self.metrics.on_abort(uid)
                self._spawn_bg(self.send_one(AbortMsg(uid=uid)))

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int, reasoning_stream=None, tool_stream=None):
        first_chunk = True
        prompt_tokens = completion_tokens = 0
        finish_reason = "stop"

        def _chunk(delta: dict) -> bytes:
            payload = {
                "id": f"cmpl-{uid}",
                "object": "chat.completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            return f"data: {json.dumps(payload)}\n\n".encode()

        async for ack in self.wait_for_ack(uid):
            delta: dict = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            tool_deltas: List[dict] = []
            if ack.incremental_output:
                # Reasoning models: route the pre-</think> scratch to `reasoning_content` and the
                # answer to `content`, in the streaming delta (buffers a partial closing tag).
                if reasoning_stream is not None:
                    r_delta, c_delta = reasoning_stream.push(ack.incremental_output)
                    if r_delta:
                        delta["reasoning_content"] = r_delta
                else:
                    c_delta = ack.incremental_output
                # Tool calling: split completed <tool_call>/<function=> blocks out of `content` and
                # re-emit them as OpenAI streaming `delta.tool_calls` (buffers a partial opener).
                if c_delta:
                    if tool_stream is not None:
                        content_out, tool_deltas = tool_stream.push(c_delta)
                        if content_out:
                            delta["content"] = content_out
                    else:
                        delta["content"] = c_delta
            completion_tokens = max(completion_tokens, ack.completion_tokens)
            prompt_tokens = ack.prompt_tokens or prompt_tokens
            if ack.finish_reason:
                finish_reason = ack.finish_reason

            # Emit the content/reasoning delta (if any), then one chunk per tool-call fragment.
            if delta:
                yield _chunk(delta)
            for td in tool_deltas:
                yield _chunk({"tool_calls": [td]})

            if ack.finished:
                break

        # final chunk: flush any buffered reasoning tail (model never closed </think>) and any tool
        # tail, then finish_reason + usage (OpenAI carries usage on the terminal chunk).
        final_delta: dict = {}
        if reasoning_stream is not None and (tail := reasoning_stream.flush()):
            final_delta["reasoning_content"] = tail
        if tool_stream is not None:
            c_tail, t_tail = tool_stream.flush()
            if c_tail:
                final_delta["content"] = final_delta.get("content", "") + c_tail
            for td in t_tail:
                yield _chunk({"tool_calls": [td]})
            if tool_stream.emitted and finish_reason != "length":
                finish_reason = "tool_calls"
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": final_delta, "index": 0, "finish_reason": finish_reason}],
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
        # `request.is_disconnected()` is an event-loop receive() round-trip; awaiting it on EVERY
        # token is pure per-token overhead. Poll it on a cadence instead — at most once per
        # _DISCONNECT_POLL_INTERVAL seconds. Detecting a disconnect up to one interval late is fine:
        # over-running the generation briefly is cheap, and the next poll aborts + cleans up.
        _DISCONNECT_POLL_INTERVAL = 0.5
        last_disconnect_check = 0.0
        try:
            async for chunk in generator:
                # detect if the client has disconnected (rate-limited)
                now = time.monotonic()
                if now - last_disconnect_check >= _DISCONNECT_POLL_INTERVAL:
                    last_disconnect_check = now
                    if await request.is_disconnected():
                        logger.info("Client disconnected for user %s", uid)
                        raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        self.metrics.on_abort(uid)
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
    if os.environ.get("MINISGL_CAM_WRITE_SYNC") == "1":
        await _cam_auto_write(req.prompt, override=req.cam_write, ns=_cam_ns)   # ambient write (gated)
    else:
        _schedule_cam_auto_write(req.prompt, override=req.cam_write, ns=_cam_ns)  # off critical path
    prompt = await _cam_auto_augment(req.prompt, ns=_cam_ns, override=req.cam_read)   # TRANSPARENT CAM read
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
        # Same message normalization as the plain lane so tool-call / tool-result history renders in
        # the template (exclude_none + JSON-string tool args -> dict).
        messages = [msg.model_dump(exclude_none=True) for msg in req.messages]
        _normalize_tool_args(messages)
        client = InProcessBackendClient(state, state.config.model_path)
        try:
            # Thread the SAME structured transports the plain lane uses: thinking-mode applies to every
            # rollout; the grammar (response_format / json_schema) + tools are applied to the FINAL
            # answer so RSA honors structured output / tool-calling instead of returning raw prose.
            # Constrained decoding for the FINAL answer: response_format wins; else a FORCED tool call
            # (tool_choice required / specific) gets its own JSON-schema grammar so the call's arguments
            # are schema-checked and terminate — the same treatment response_format gets (auto/none tool
            # choice stays grammar-free and is parsed from the XML wrapper below).
            rf_grammar = _grammar_from_response_format(req.response_format)
            forced_tool_grammar = _grammar_from_tools(req) if rf_grammar is None else None
            auto_tool_grammar = (
                _structural_tag_from_tools(req)
                if rf_grammar is None and forced_tool_grammar is None else None
            )
            result = await run_markovian_rsa(
                client, rsa_params, messages, req.model,
                chat_template_kwargs=_resolve_chat_template_kwargs(req),
                grammar=rf_grammar or forced_tool_grammar or auto_tool_grammar,
                tools=_tools_for_template(req),
                # UNCONDITIONAL close delim (not grammar-gated): RSA β-bounds reasoning on every
                # grammar-free rollout, not just the structured final answer.
                think_close_delim=_reasoning_close_delim(req),
                think_budget=_resolve_think_budget(req),
            )
        except RSAError as e:
            return JSONResponse(status_code=502, content={"error": f"RSA failed: {e}"})
        finally:
            await client.close()
        # Parse the final answer exactly like the plain lane: split the <think>…</think> reasoning out
        # of content into reasoning_content, then (if tools were offered) parse <tool_call> blocks into
        # OpenAI-shaped tool_calls. Fixes "thinking stays in content" + tool calls ignored on this lane.
        finish_reason = "stop"
        reasoning_content: str | None = None
        body = result.final_text
        parser = _reasoning_parser()
        if parser is not None and _thinking_active(req):
            # thinking_open=True: reached only when thinking is active, so an un-closed </think> means
            # a truncated chain-of-thought -> route it to reasoning_content, not the visible answer.
            reasoning_content, body = parser.parse(result.final_text, thinking_open=True)
        message: dict = {"role": "assistant", "content": body}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if forced_tool_grammar is not None:
            # Forced tool call: grammar constrained `body` to a complete call. zaya_xml -> native XML
            # (<function=…><parameter=…>), parsed by the wrapper parser; else JSON {"name","arguments"}.
            if '"__ebnf__"' in forced_tool_grammar:
                _tc_content, tool_calls = _parse_tool_calls(body, state.uid_counter)
                if tool_calls:
                    message["content"] = _tc_content
                    message["tool_calls"] = tool_calls
                    finish_reason = "tool_calls"
            else:
                tc = _parse_json_tool_call(body, state.uid_counter)
                if tc is not None:
                    message["content"] = None
                    message["tool_calls"] = [tc]
                    finish_reason = "tool_calls"
        elif req.tools and req.tool_choice != "none":
            # auto: the model chose; if it opened a wrapper its args were structural-tag-constrained.
            _tc_content, tool_calls = _parse_tool_calls(body, state.uid_counter)
            if tool_calls:
                message["content"] = _tc_content
                message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
        return {
            "id": f"chatcmpl-rsa-{state.uid_counter}",
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
                "think_budget": rsa_params.think_budget,
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
    if os.environ.get("MINISGL_CAM_WRITE_SYNC") == "1":
        await _cam_auto_write(_last_user or "", override=req.cam_write, ns=_cam_ns)   # gated ambient write
    else:
        _schedule_cam_auto_write(_last_user or "", override=req.cam_write, ns=_cam_ns)  # off critical path
    prompt = await _cam_auto_augment(prompt, ns=_cam_ns, override=req.cam_read)     # TRANSPARENT CAM read

    uid = state.new_user()
    # Constrained decoding: response_format wins; else a FORCED tool call (tool_choice required /
    # specific) gets its own JSON-schema grammar so the arguments are schema-checked and terminate.
    _pl_rf_grammar = _grammar_from_response_format(req.response_format)
    _pl_forced_tool_grammar = _grammar_from_tools(req) if _pl_rf_grammar is None else None
    _pl_auto_tool_grammar = (
        _structural_tag_from_tools(req)
        if _pl_rf_grammar is None and _pl_forced_tool_grammar is None else None
    )
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            tools=_tools_for_template(req),
            chat_template_kwargs=_resolve_chat_template_kwargs(req),
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                **dict(zip(("temperature", "top_p", "top_k"), _resolve_sampling(req, state.config.model_path))),
                stop=_norm_stop(req.stop),
                grammar=_pl_rf_grammar or _pl_forced_tool_grammar or _pl_auto_tool_grammar,
                # UNCONDITIONAL close delim (was grammar-only): β-bounds reasoning on the PLAIN lane
                # too, so a thinking request without response_format can't run reasoning to max_tokens
                # and truncate with no answer. For grammar requests this is the same delim (it also
                # gates the schema until </think>); for plain thinking it's a pure backstop.
                think_close_delim=_reasoning_close_delim(req),
                think_budget=_resolve_think_budget(req),
            ),
        )
    )

    if req.stream:
        parser = _reasoning_parser()
        reasoning_stream = (
            parser.stream_state(active=True)
            if parser is not None and _thinking_active(req)
            else None
        )
        # Stateful tool-call parser: only when tools are actually offered to the model (mirrors the
        # non-streaming path's `if req.tools`). `tool_choice:"none"` withholds the tools from the
        # template, so no blocks are emitted and this stays a no-op even when constructed.
        tool_stream = ToolCallStreamState(uid) if req.tools and req.tool_choice != "none" else None
        return StreamingResponse(
            state.stream_with_cancellation(
                state.stream_chat_completions(uid, reasoning_stream, tool_stream), request, uid
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response. Accumulate the incremental
    # chunks in a list and "".join once at the end — string `+=` in the loop is O(n^2) in the output
    # length for long completions.
    content_chunks: List[str] = []
    prompt_tokens = completion_tokens = 0
    finish_reason = "stop"
    async for ack in state.wait_for_ack(uid):
        content_chunks.append(ack.incremental_output)
        completion_tokens = max(completion_tokens, ack.completion_tokens)
        prompt_tokens = ack.prompt_tokens or prompt_tokens
        if ack.finish_reason:
            finish_reason = ack.finish_reason
        if ack.finished:
            break
    full_content = "".join(content_chunks)

    # Reasoning: split a thinking model's `<think>…</think>` scratch out of the answer into a
    # separate reasoning_content field (the opening tag is in the prompt, so the completion carries
    # only the closing </think> + answer). No-op when disabled / thinking off. thinking_open=True:
    # thinking is active here, so an output with NO closing </think> is a truncated chain-of-thought
    # -> route it entirely to reasoning_content, not the visible answer.
    reasoning_content: str | None = None
    body = full_content
    parser = _reasoning_parser()
    if parser is not None and _thinking_active(req):
        reasoning_content, body = parser.parse(full_content, thinking_open=True)

    # Tool calling: if tools were offered, parse any <tool_call> blocks the model emitted (AFTER the
    # reasoning split) into OpenAI-shaped tool_calls and flip finish_reason. No tools -> untouched.
    message: dict = {"role": "assistant", "content": body}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if _pl_forced_tool_grammar is not None:
        # Forced tool call: zaya_xml -> native XML (wrapper parser); else JSON {"name","arguments"}.
        if '"__ebnf__"' in _pl_forced_tool_grammar:
            content, tool_calls = _parse_tool_calls(body, uid)
            if tool_calls:
                message["content"] = content
                message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
        else:
            tc = _parse_json_tool_call(body, uid)
            if tc is not None:
                message["content"] = None
                message["tool_calls"] = [tc]
                finish_reason = "tool_calls"
    elif req.tools and finish_reason != "length":
        content, tool_calls = _parse_tool_calls(body, uid)
        if tool_calls:
            message["content"] = content
            message["tool_calls"] = tool_calls
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


@app.get("/health")
async def health():
    """Liveness probe. 200 once the frontend is up and wired to the backend (global state
    initialized). uvicorn only accepts connections after the lifespan startup that populates the
    global state and launches the backend, so a 200 here means the server is serving. Agents /
    monitors / load balancers should poll this."""
    state = get_global_state()
    return {"status": "ok", "model": state.config.model_path}


@app.get("/metrics")
async def metrics():
    """Prometheus text-exposition endpoint (hand-rolled; no prometheus_client dependency). Frontend
    counters/histograms + the latest per-DP-replica scheduler snapshot. See server/metrics.py."""
    state = get_global_state()
    return PlainTextResponse(
        state.metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


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
                **dict(zip(("temperature", "top_p", "top_k"), _resolve_sampling(req, state.config.model_path))),
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
        metrics=FrontendMetrics(config.model_path),
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
        # uvicorn's per-request HTTP access log (one INFO line per POST /v1/... or /generate) is the
        # ~200-lines-per-run noise; default it OFF. MINISGL_HTTP_ACCESS_LOG=1 re-enables it.
        _access_log = os.environ.get("MINISGL_HTTP_ACCESS_LOG", "0") != "0"
        uvicorn.run(app, host=host, port=port, access_log=_access_log)
    else:
        asyncio.run(shell())
