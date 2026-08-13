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
from minisgl.utils import load_generation_config, load_tokenizer
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
from .reasoning import resolve_reasoning_parser

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
    # Mirror the OpenAI lane so /generate is not a second-class citizen (it already mirrors
    # temperature/top_p/top_k). Neutral 0.0 default = penalty path skipped entirely.
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # Sampling. Unset (None) inherits the checkpoint's generation_config.json via _resolve_sampling,
    # exactly as the OpenAI endpoints already do. /generate previously built a bare SamplingParams()
    # and so served GREEDY (SamplingParams defaults temperature=0.0 / top_k=-1 / top_p=1.0) even
    # though this checkpoint asks for do_sample=true, temperature 1.0, top_k 20, top_p 0.95 — a
    # silent divergence from /v1/chat/completions, and it meant every /generate benchmark measured
    # the argmax path instead of the top-k/top-p sampler the real serve runs.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    # TRANSPARENT CAM per-request override: True/False forces ambient auto-write on/off for THIS call,
    # overriding MINISGL_CAM_AUTO_WRITE (None = server default). Suppress learning on a read-only turn.
    cam_write: bool | None = None
    # Symmetric per-request override for the TRANSPARENT auto-READ (retrieve+augment): True/False forces
    # it on/off for THIS call, overriding MINISGL_CAM_AUTO (None = server default). Lets a caller (or an
    # A/B harness) suppress the retrieve round-trip on a given turn.
    cam_read: bool | None = None
    # Per-request RNG seed (None = the process RNG, i.e. unchanged). The knob that makes a
    # BLOCK-DIFFUSION generation reproducible: that path samples a whole canvas from noise on every
    # denoising step, so it has no greedy mode and `temperature 0` does not pin it. Inert for the
    # autoregressive path, whose reproducibility switch is the greedy triple.
    seed: int | None = None


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
    # OpenAI stream_options, e.g. {"include_usage": true}. When include_usage is set, a spec-compliant
    # streaming response emits a FINAL chunk with an empty `choices` array carrying the `usage` totals
    # (before [DONE]) — the shape strict clients (langchain usage_metadata, budget guards) parse.
    stream_options: dict | None = None
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
    # reasoning_effort ladder: none/off/minimal -> thinking OFF; low/medium/high -> 256/1024/4096;
    # extra_high (a.k.a. "extra high"/"xhigh") -> 16384; max/maximum/unlimited -> unbounded.
    # An unrecognised value is a 400 rather than a silent fallback.
    reasoning_effort: str | None = None
    # OpenRouter-style {"enabled": bool, "effort": str, "exclude": bool}, and a bare `thinking` bool.
    # Accepted as aliases so a client does not have to know THIS server's preferred spelling.
    reasoning: dict | None = None
    thinking: bool | None = None
    # Accepted and inert (no behavioural meaning for a local single-tenant server). Declared so they
    # are visibly ignored rather than silently swallowed by the extra-field policy.
    user: str | None = None
    store: bool | None = None
    metadata: dict | None = None
    service_tier: str | None = None
    parallel_tool_calls: bool | None = None
    # Declared ONLY so the request can be REJECTED with a clear reason. The engine has no support for
    # these; accepting them silently (the old behaviour) means answering a different question than the
    # one asked. See _reject_unsupported.
    logprobs: bool | None = None
    top_logprobs: int | None = None
    logit_bias: dict | None = None
    # TEXT-completions-only parameters (/v1/completions). Declared for the SAME reason as the block
    # above — so they can be rejected instead of silently swallowed by the extra-field policy. Each
    # one changes the answer: `echo` changes what the response contains (prompt + completion, not
    # completion), `suffix` changes what is generated (fill-in-the-middle), `best_of` changes what is
    # sampled (n candidates, return the best). Accepting any of them and doing nothing returns a 200
    # that plausibly answers a DIFFERENT question, which is the exact failure _reject_unsupported
    # exists to prevent. Unused by the chat lane, where they are not part of the protocol.
    echo: bool | None = None
    suffix: str | None = None
    best_of: int | None = None
    # `seed` USED to sit in the group above — declared so it could be rejected, but never actually
    # checked by _reject_unsupported, so it was accepted and ignored. It is now HONOURED, and only
    # where it means something: the BLOCK-DIFFUSION canvas, the one path here with no greedy mode
    # (its canvas is drawn from noise and every denoising step draws a multinomial, so
    # `temperature 0` pins nothing and two identical requests return different text). It reaches
    # SamplingParams.seed -> CanvasManager.begin. On the autoregressive path it is inert, because
    # reproducibility there is the greedy triple, not an RNG seed.
    seed: int | None = None

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

    @model_validator(mode="after")
    def _reject_malformed(self) -> "OpenAICompletionRequest":
        """Reject requests that cannot be served, as 4xx, BEFORE they reach the engine.

        `max_tokens <= 0` KILLED THE WHOLE SERVE — measured 2026-08-02: a single
        `{"max_tokens": -5}` dropped the connection and the container came back reloading the model
        from scratch. Nothing downstream bounded it (`reasoning_max_tokens` is checked `> 0` a few
        lines below, but the main field never was), so a negative budget reached the scheduler and
        took the process with it. That is a remote DoS from an unauthenticated malformed request —
        any client can end the serve, and every other in-flight request dies with it.

        Empty/absent `messages` with no `prompt` was a 500 for the same reason: the guard existed
        only on the RSA lane, so the ordinary lane fell through to a raw-prompt branch with nothing
        to read. A malformed CLIENT request must never surface as a server error.
        """
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not self.messages and not getattr(self, "prompt", None):
            raise ValueError("either `messages` (chat) or `prompt` (completion) is required")
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
    fmt = _resolve_tool_format()
    if fmt == "zaya_xml":
        ebnf = _zaya_xml_grammar(tools, forced_name)  # native <zyphra_tool_call> XML, not JSON
        return json.dumps({"__ebnf__": ebnf}) if ebnf else None
    if fmt == "gemma_native":
        ebnf = _gemma_native_grammar(tools, forced_name)  # native <|tool_call>call:…, not JSON
        return json.dumps({"__ebnf__": ebnf}) if ebnf else None
    if fmt == "atem":
        ebnf = _atem_xml_grammar(tools, forced_name)  # native <atem:invoke> XML, not JSON
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


def _atem_xml_grammar(tools: List[dict], forced_name: str | None = None) -> str | None:
    """EBNF constraining Muse-Glimmer's NATIVE tool call to its trained ATEM format:

        <atem:function_calls>\\n<atem:invoke name="NAME">\\n
        (<atem:parameter name="P">VALUE</atem:parameter>\\n)*
        </atem:invoke>\\n</atem:function_calls>

    Mirrors `_zaya_xml_grammar`: NAMES are constrained to the allowed tools and known parameters (so
    the structure is guaranteed parseable by `_parse_tool_calls`) while VALUES stay permissive so the
    model is not boxed in on content. Params are any-order/any-subset — a fixed order would reject
    valid calls.

    Only ONE `<atem:invoke>` is emitted under constraint even though the format admits several in a
    block (that is how it expresses parallel calls). Forcing a call is a request for *a* call; the
    unconstrained "auto" path still parses as many as the model emits.

    `pval` excludes `<` for the same reason ZAYA's does — it is what terminates the value — so a
    value containing a literal `<` cannot be emitted under constraint. That is a real limitation of
    a delimiter-terminated format, shared with every other native grammar here, and it applies only
    when a call is FORCED."""
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
        'root ::= "<atem:function_calls>\\n<atem:invoke name=\\"" fname "\\">\\n" params '
        '"</atem:invoke>\\n</atem:function_calls>"',
        f"fname ::= {fname_alt}",
        "params ::= param*",
        'param ::= "<atem:parameter name=\\"" pname "\\">" pval "</atem:parameter>\\n"',
        f"pname ::= {pname_alt}",
        "pval ::= [^<]*",
    ])


def _gemma_native_grammar(tools: List[dict], forced_name: str | None = None) -> str | None:
    """EBNF constraining Gemma-4's NATIVE tool call to its trained format:

        <|tool_call>call:NAME{key:value,key:value}<tool_call|>

    Read off the checkpoint's own `chat_template.jinja` rather than guessed. The call is rendered by
    `'<|tool_call>call:' + name + '{'` then `key ':' format_argument(value, escape_keys=False)` joined
    by `,`, then `'}<tool_call|>'`. `format_argument` is the value grammar, and `escape_keys=False`
    PROPAGATES into nested mappings, so every key at every depth is BARE:

        str      -> `<|"|>text<|"|>`      bool -> `true` / `false`
        mapping  -> `{k:v,…}`            sequence -> `[a,b]`        anything else -> raw

    The body is JSON-SHAPED but is NOT JSON — bare keys, and strings delimited by the `<|"|>` special
    token rather than `"`. That is why this needs an EBNF and cannot ride the structural-tag path,
    which is JSON-schema-only (see `_TOOL_STRUCT_WRAPPERS`). Forcing the JSON shape here would make a
    forced call come out in a format the checkpoint was never trained to emit AND that its own
    template cannot render back into a prompt.

    Function NAME is constrained to the allowed tools and top-level keys to those tools' declared
    properties, so the result is guaranteed parseable by `_parse_gemma_tool_call`. Params are
    any-order/any-subset (a fixed order would reject valid calls) and values stay permissive so the
    model is not boxed on content — the same trade `_zaya_xml_grammar` makes.

    One documented limit: a string value cannot contain the two-character sequence `<|`, because the
    closing `<|"|>` has to be recognisable. `<` alone is fine (`ls <file`, `a < b`), which is the
    common case a flat `[^<]*` would have broken."""
    fns: List[str] = []
    keys: set[str] = set()
    for t in tools:
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name or (forced_name and name != forced_name):
            continue
        fns.append(name)
        keys.update((fn.get("parameters") or {}).get("properties", {}) or {})
    if not fns:
        return None
    fname_alt = " | ".join(_ebnf_lit(n) for n in fns)
    key_alt = " | ".join(_ebnf_lit(k) for k in sorted(keys)) if keys else _ebnf_lit("_")
    q = _ebnf_lit('<|"|>')
    return "\n".join([
        f'root ::= "<|tool_call>call:" fname "{{" args "}}<tool_call|>"',
        f"fname ::= {fname_alt}",
        'args ::= (arg ("," arg)*)?',
        'arg ::= key ":" value',
        f"key ::= {key_alt}",
        "value ::= str | bool | num | arr | obj",
        f"str ::= {q} schar* {q}",
        # Anything but the closer. `<` is allowed unless followed by `|`, so shell redirects and
        # comparisons survive; only the literal `<|` is out of reach.
        'schar ::= [^<] | "<" [^|]',
        'bool ::= "true" | "false"',
        'num ::= "-"? [0-9]+ ("." [0-9]+)?',
        'arr ::= "[" (value ("," value)*)? "]"',
        # Nested mappings: `escape_keys=False` propagates, so these keys are bare too. They are NOT
        # tool property names (they are whatever the argument object holds), hence a permissive ident.
        'obj ::= "{" (okv ("," okv)*)? "}"',
        'okv ::= ident ":" value',
        "ident ::= [a-zA-Z_] [a-zA-Z0-9_]*",
    ])


# Which native tool-call format this checkpoint speaks. DERIVED from the chat template by default
# (see `_derive_tool_format`) rather than named: `MINISGL_TOOL_FORMAT` used to default to "json", so a
# family whose native call is not JSON got the JSON shape unless an operator knew to set the variable
# — which is how Gemma-4 ended up with no native grammar at all despite the template plainly rendering
# one. An explicit value still wins, for a serve that needs to pin the format (the ZAYA compose
# service sets zaya_xml).
_TOOL_FORMAT = os.environ.get("MINISGL_TOOL_FORMAT", "auto")
_DERIVED_TOOL_FORMAT: str | None = None
_DERIVED_TOOL_FORMAT_SET = False

# Probe call used to make the template SHOW its tool-call wrapper. Content is irrelevant — the wrapper
# is emitted by the template's tool_calls branch, which keys on the message shape, not on the values.
_TOOL_PROBE_MESSAGES = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {"k": "v"}}}]},
]


def _derive_tool_format() -> str | None:
    """Render a tool call through the checkpoint's own template and read the format back out.

    The template is the fact: it is what turns a `tool_calls` message into bytes, so whatever it emits
    around the call IS this checkpoint's native format. Only a shape the forced path can actually
    CONSTRAIN is reported — today Gemma-4's `<|tool_call>call:NAME{…}` and Muse-Glimmer's ATEM XML;
    everything else returns None and keeps the JSON default, so this can never downgrade a family it
    does not recognise."""
    tok = _frontend_tokenizer()
    if tok is None:
        return None
    try:
        rendered = tok.apply_chat_template(
            _TOOL_PROBE_MESSAGES, tokenize=False, add_generation_prompt=False)
    except Exception as e:  # noqa: BLE001 — a template that rejects the probe tells us nothing
        logger.debug("tool-format probe render failed: %s", e)
        return None
    if "<|tool_call>call:" in rendered:
        return "gemma_native"
    if "<atem:invoke" in rendered:
        return "atem"
    return None


def _resolve_tool_format() -> str:
    """`MINISGL_TOOL_FORMAT` when set to anything but "auto", else the derived format, else "json"."""
    global _DERIVED_TOOL_FORMAT, _DERIVED_TOOL_FORMAT_SET
    if _TOOL_FORMAT and _TOOL_FORMAT != "auto":
        return _TOOL_FORMAT
    if not _DERIVED_TOOL_FORMAT_SET:
        _DERIVED_TOOL_FORMAT_SET = True
        _DERIVED_TOOL_FORMAT = _derive_tool_format()
        logger.info("native tool-call format: %s", _DERIVED_TOOL_FORMAT or "json (no native form derived)")
    return _DERIVED_TOOL_FORMAT or "json"


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
#
# GEMMA-4 IS DELIBERATELY ABSENT, and it is the one family that cannot simply be added. Its wrapper
# `<|tool_call>` … `<tool_call|>` would be a fine trigger, but a structural tag can only constrain its
# body to a JSON SCHEMA (`grammar.py`: `xgr.StructuralTagItem(begin, schema, end)`), and Gemma's body
# is not JSON — bare keys, and strings delimited by the `<|"|>` special token. Listing it here would
# force the model to emit JSON inside its native wrapper: a shape it was never trained to produce and
# that its own template cannot render back into a prompt on the next turn. So `auto` mode stays
# UNCONSTRAINED for Gemma-4 (the model picks the format; `_parse_gemma_tool_call` reads it), and the
# native format is guaranteed on the FORCED path instead, via `_gemma_native_grammar`. Closing this
# properly needs per-tag EBNF support in xgrammar, which structural tags do not have today.
# Muse-Glimmer's ATEM XML is omitted for the SAME reason: its body is `<atem:parameter name="k">v`
# elements, not JSON. It likewise stays unconstrained under `auto` (parsed by the `_ATEM_INVOKE_RE`
# scan) and is guaranteed on the forced path by `_atem_xml_grammar`.
_TOOL_STRUCT_WRAPPERS = (
    ("<zyphra_tool_call>", "</zyphra_tool_call>"),
    ("<tool_call>", "</tool_call>"),
    ("<tools>", "</tools>"),
)


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


def _reasoning_answer_delim(req: "OpenAICompletionRequest") -> str | None:
    """The model's own ANSWER-TURN header, as an additional "reasoning ended" marker for the
    scheduler's think gate. Never a force target — only a release trigger.

    Needed by a template whose generation prompt stops MID-HEADER: Muse-Glimmer's ends
    `<|start|>assistant`, so the model itself writes the recipient and a reply with no reasoning
    begins ` to=user<|message|>` — it never opens a reasoning turn and so never emits the closer.
    Without this the gate would stay shut for the whole reply and the β backstop would splice a turn
    header into the middle of a legitimate answer. "" for every family whose generation prompt ends at
    a turn boundary, which is the common case.

    SAFETY: a header that is a SUFFIX of the reasoning opener would match at the START of the
    reasoning turn and release the gate immediately — exactly the single-token bug this whole change
    fixes, reintroduced one level up. (Muse: opener ` to=self<|message|>` vs header
    ` to=user<|message|>` — no suffix relation, so the header is kept.)"""
    if not _thinking_active(req):
        return None
    parser = _reasoning_parser()
    if parser is None:
        return None
    header = parser.turn_header
    if not header:
        return None
    if parser.start_token and parser.start_token.endswith(header):
        return None
    return header


def _reasoning_close_wildcard(req: "OpenAICompletionRequest") -> tuple[str | None, str | None]:
    """`(prefix, suffix)` bracketing a VARIABLE recipient inside the close delimiter, so the gate also
    recognises reasoning that ends by routing to a TOOL (`…assistant to=<toolname><|message|>`) rather
    than to the user. `(None, None)` when the checkpoint's closer is a fixed literal."""
    if not _thinking_active(req):
        return None, None
    parser = _reasoning_parser()
    if parser is None or not (parser.end_prefix and parser.end_suffix):
        return None, None
    return parser.end_prefix, parser.end_suffix


# reasoning_effort -> token budget for the think-gate backstop.
#   * OFF rungs disable thinking outright (handled in _resolve_chat_template_kwargs, not here).
#   * "max" means UNBOUNDED: thinking on, no budget — distinct from omitting the field only in that
#     it is an explicit choice rather than a default.
# Spellings are normalised (case, spaces, hyphens -> underscores) because clients disagree:
# "extra high" / "extra-high" / "xhigh" all mean the same thing.
_EFFORT_OFF = frozenset({"none", "off", "minimal", "no", "false", "disabled"})
_EFFORT_BUDGET = {
    "low": 256,
    "medium": 1024,
    "high": 4096,
    "extra_high": 16384,
    "xhigh": 16384,
    "x_high": 16384,
    "very_high": 16384,
    "max": None,          # explicit "think as long as you need"
    "maximum": None,
    "unlimited": None,
}


def _norm_effort(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("-", " ").replace("_", " ").split())


def _effort_is_off(req: "OpenAICompletionRequest") -> bool:
    """True when the client asked for NO reasoning via reasoning_effort or an OpenRouter-style alias."""
    if req.reasoning_effort and _norm_effort(req.reasoning_effort) in _EFFORT_OFF:
        return True
    r = req.reasoning
    if isinstance(r, dict):
        if r.get("enabled") is False or r.get("exclude") is True:
            return True
        eff = r.get("effort")
        if eff and _norm_effort(eff) in _EFFORT_OFF:
            return True
    return req.thinking is False


def _effort_is_on(req: "OpenAICompletionRequest") -> bool:
    """True when the client asked FOR reasoning — the mirror of ``_effort_is_off``.

    An ON rung (`reasoning_effort` low/medium/high/xhigh/max, `reasoning={"enabled": true}` or an
    effort inside it, or a bare `thinking: true`) is a request for the model to think, not merely a
    token budget for thinking that may or may not happen. Without this the ladder was one-sided: OFF
    was honored and ON was silently dropped, so a caller asking for MORE reasoning on a checkpoint
    whose template defaults to thinking-off got a 16384-token budget for a span the template never
    opened — measured on Gemma-4, `reasoning_effort` xhigh, high, medium and none all returned an
    identical answer with `reasoning_content` empty. Only `enable_thinking` worked, which is the
    field the ladder exists to spare callers from knowing about.

    `enable_thinking=false` still wins: it is checked first in `_resolve_chat_template_kwargs`, so an
    explicit opt-out is never overridden by an effort rung that came along for the ride."""
    if req.thinking is True:
        return True
    if req.reasoning_effort and _norm_effort(req.reasoning_effort) in _EFFORT_BUDGET:
        return True
    r = req.reasoning
    if isinstance(r, dict):
        if r.get("enabled") is True:
            return True
        eff = r.get("effort")
        if eff and _norm_effort(eff) in _EFFORT_BUDGET:
            return True
    return False


def _resolve_think_budget(req: "OpenAICompletionRequest") -> int | None:
    """Per-request reasoning-token budget, or None for unbounded (the server's MINISGL_THINK_BUDGET
    default still applies when nothing is set). Precedence: explicit `reasoning_max_tokens` > the same
    key inside `chat_template_kwargs` > `reasoning_effort` > `reasoning.effort`.

    Applies to plain requests too, not just grammar-constrained ones — the scheduler arms the think
    gate for any thinking request (see _maybe_arm_think_gate). The old docstring claimed otherwise
    and was stale, which made this knob look inert when it is not."""
    if isinstance(req.reasoning_max_tokens, int) and req.reasoning_max_tokens > 0:
        return req.reasoning_max_tokens
    ck = req.chat_template_kwargs or {}
    ck_budget = ck.get("reasoning_max_tokens")
    if isinstance(ck_budget, int) and ck_budget > 0:
        return ck_budget
    effort = req.reasoning_effort
    if not effort and isinstance(req.reasoning, dict):
        effort = req.reasoning.get("effort")
    if effort:
        return _EFFORT_BUDGET.get(_norm_effort(effort))
    return None


def _norm_stop(stop: list | str | None) -> List[str]:
    if not stop:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _tool_stop_keep(req) -> List[str]:
    """Tool-call closers to stop on but KEEP in the output (an INCLUSIVE stop — unlike a normal stop
    string, which the detokenizer trims). Laguna (and other tool-trained models that don't emit EOS
    after a call) over-generate: after the first `</tool_call>` they keep going, repeating the call +
    emitting garbage until max_tokens. Stopping at the closer yields one clean call
    (finish_reason=tool_calls); keeping the closer leaves the `<tool_call>…</tool_call>` block
    parseable. Empty unless tools are offered (then the closer only appears if the model calls a
    tool)."""
    if getattr(req, "tools", None) and getattr(req, "tool_choice", None) != "none":
        return ["</tool_call>", "</zyphra_tool_call>"]
    return []


def _resolve_sampling(req: "OpenAICompletionRequest | GenerateRequest", model_path: str) -> tuple:
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


def _reject_unsupported(req: "OpenAICompletionRequest") -> JSONResponse | None:
    """400 for parameters this engine cannot honour, instead of accepting and ignoring them.

    Silently ignoring is the worse failure: the caller gets a 200 and a confidently wrong answer to a
    different question. Measured before this existed — presence_penalty 0.0 vs 2.0 produced
    byte-identical output, n=3 returned 1 choice, and reasoning_effort="none" produced UNBOUNDED
    reasoning. Every one of those looked like success."""
    def bad(msg: str, param: str, code: str = "unsupported_parameter") -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": {
            "message": msg, "type": "invalid_request_error", "param": param, "code": code}})

    if req.logprobs or req.top_logprobs is not None:
        return bad("logprobs / top_logprobs are not supported by this server: returning per-token "
                   "logprobs would require carrying them through the sampler, scheduler and "
                   "detokenizer on every token. Omit the field.", "logprobs")
    if req.logit_bias:
        return bad("logit_bias is not supported by this server.", "logit_bias")
    if req.n is not None and req.n != 1:
        return bad(f"n={req.n} is not supported: this server returns a single choice. Issue n "
                   "separate requests instead.", "n")
    effort = req.reasoning_effort or (req.reasoning or {}).get("effort")
    if effort:
        norm = _norm_effort(effort)
        if norm not in _EFFORT_OFF and norm not in _EFFORT_BUDGET:
            return bad(
                f"reasoning_effort={effort!r} is not recognised. Supported: "
                "none/off/minimal (no reasoning), low, medium, high, extra_high, max.",
                "reasoning_effort", "invalid_value")
    return None


def _reject_unsupported_text_completion(req: "OpenAICompletionRequest") -> JSONResponse | None:
    """`_reject_unsupported`, plus the things that are meaningless on the RAW-prompt lane.

    Same principle as its sibling: a parameter this endpoint cannot honour is a 400, never an
    accepted-and-ignored field. The additions are all cases where the chat lane's machinery has no
    counterpart here, and where the silent behaviour is the confusing one:

    * `messages` — there is no chat template on this lane, so a messages array would be flattened to
      nothing or stringified. The caller wants /v1/chat/completions.
    * `tools` — tool specs are rendered INTO the chat template. With no template they are never shown
      to the model, so the model answers in prose, no `<tool_call>` block is ever emitted, and the
      caller sees `tool_calls: null` with no indication that its tools were dropped on the floor.
    * `rsa` — the in-engine RSA loop drives chat rollouts; the chat lane already 400s a raw `prompt`
      for the same reason.
    * `echo` / `suffix` / `best_of` — see the request-model comment.
    """
    bad = _reject_unsupported(req)
    if bad is not None:
        return bad

    def reject(msg: str, param: str) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": {
            "message": msg, "type": "invalid_request_error", "param": param,
            "code": "unsupported_parameter"}})

    if req.messages:
        return reject("`messages` is not accepted by /v1/completions: this endpoint continues a RAW "
                      "`prompt` with no chat template applied. Use /v1/chat/completions.", "messages")
    if not isinstance(req.prompt, str) or not req.prompt:
        return reject("`prompt` (a non-empty string) is required by /v1/completions.", "prompt")
    if req.tools:
        return reject("`tools` are not supported by /v1/completions: tool specs are rendered into the "
                      "chat template, and this endpoint applies no template — the model would never "
                      "see them. Use /v1/chat/completions.", "tools")
    if req.rsa:
        return reject("`rsa` requires `messages` (chat format), not a raw `prompt`.", "rsa")
    if req.echo:
        return reject("`echo` is not supported: this server returns the completion only.", "echo")
    if req.suffix is not None:
        return reject("`suffix` (fill-in-the-middle) is not supported by this server.", "suffix")
    if req.best_of is not None and req.best_of != 1:
        return reject(f"best_of={req.best_of} is not supported: this server samples a single "
                      "candidate.", "best_of")
    return None


def _resolve_chat_template_kwargs(req: "OpenAICompletionRequest") -> dict | None:
    """Merge the request's `chat_template_kwargs` with the `enable_thinking` convenience alias into
    the kwargs forwarded to `apply_chat_template`. None -> template defaults (thinking ON for Qwen3 /
    Poolside, whose reasoning format is now parsed — see _reasoning_parser + _REASONING_DELIMITERS)."""
    kwargs = dict(req.chat_template_kwargs or {})
    if req.enable_thinking is not None and "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = req.enable_thinking
    # The reasoning ladder maps to the template kwarg in BOTH directions. OFF first, so an explicit
    # opt-out wins over an effort rung that rode along on the same request:
    #
    # * reasoning_effort=none/minimal, reasoning={"enabled":false}, thinking=false -> enable_thinking
    #   False. Previously accepted and IGNORED — and for reasoning_effort="none" the effect was the
    #   OPPOSITE of the request (no budget -> unbounded thinking), which is how a client asking for
    #   less reasoning got the most possible.
    # * reasoning_effort=low/medium/high/xhigh/max, reasoning={"enabled":true}, thinking=true ->
    #   enable_thinking True. Same defect, other direction, and it outlived the first fix: the rung
    #   set a token BUDGET but never opened the span, so on a checkpoint whose template defaults to
    #   thinking-off every rung produced an identical answer with reasoning_content empty. A budget
    #   for reasoning that cannot happen is not a knob, and `enable_thinking` — the field the ladder
    #   exists so callers need not know about — was the only thing that worked.
    if "enable_thinking" not in kwargs:
        if _effort_is_off(req):
            kwargs["enable_thinking"] = False
        elif _effort_is_on(req):
            kwargs["enable_thinking"] = True
    return kwargs or None


def _thinking_opted_out(req: "OpenAICompletionRequest") -> bool:
    """The request explicitly asked for no reasoning: `enable_thinking=false` (top-level or inside
    chat_template_kwargs) or an OFF reasoning-effort rung. Checked ahead of the prompt derivation so
    the opt-out holds even for a template that silently ignores the kwarg."""
    if req.enable_thinking is False:
        return True
    if (req.chat_template_kwargs or {}).get("enable_thinking") is False:
        return True
    return _effort_is_off(req)


_FRONTEND_TOKENIZER = None
_FRONTEND_TOKENIZER_SET = False


def _frontend_tokenizer():
    """The served checkpoint's tokenizer, loaded ONCE in the frontend process (None if it can't be).

    The frontend does not tokenize — the tokenizer workers do — but it does need the checkpoint's
    CHAT TEMPLATE, both to derive the reasoning delimiters and to read whether a request's generation
    prompt leaves a reasoning span open. Rendering is the only way to get either: the delimiters are
    produced by template LOGIC (branches on enable_thinking, on the last turn's role), so reading the
    Jinja source or a config field would be guessing at what it emits instead of observing it.
    Costs one CPU-side tokenizer load and a handful of cached renders; no GPU state."""
    global _FRONTEND_TOKENIZER, _FRONTEND_TOKENIZER_SET
    if not _FRONTEND_TOKENIZER_SET:
        _FRONTEND_TOKENIZER_SET = True
        try:
            _FRONTEND_TOKENIZER = load_tokenizer(get_global_state().config.model_path)
        except Exception as e:  # noqa: BLE001 — reasoning splitting must never block serving
            logger.warning("could not load the tokenizer in the frontend (%s); reasoning delimiters "
                           "fall back to the legacy <think>/</think> pair", e)
            _FRONTEND_TOKENIZER = None
    return _FRONTEND_TOKENIZER


# Rendered generation prompts, keyed by the resolved chat_template_kwargs AND the request's trailing
# message SHAPE. The kwargs alone are not enough: the `add_generation_prompt` branch also keys on what
# the conversation ends WITH. Gemma-4 skips the branch entirely when the last thing rendered was a
# tool call/result (`chat_template.jinja`: `if ns.prev_message_type != 'tool_response' and ... !=
# 'tool_call'`), because the model is meant to CONTINUE the already-open model turn. Keying on kwargs
# only meant every request was classified from a `[user]`-shaped probe, so a tool-continuation turn was
# read as "thinking off" — see `_prompt_thinking_state`. Shape has a handful of distinct values per
# kwargs set, so the cache stays small and no conversation is re-rendered per request.
_PROMPT_PROBE_CACHE: Dict[tuple, str | None] = {}

# How many trailing messages define the shape. The generation-prompt branch looks at the tail of the
# conversation (the last non-tool role, and whether a tool call/result is still open), never at its
# head or at message TEXT, so a short suffix is a sound cache key.
_SHAPE_TAIL = 4

# `(span_open, reasoning_possible)` per (chat_template_kwargs, tail shape). Caching the ANSWER rather
# than the rendered string is what keeps the extra render off the per-request path: a conversation is
# rendered once per shape it ever ends in, and never re-rendered as it grows.
_THINKING_STATE_CACHE: Dict[tuple, Tuple[bool, bool]] = {}

# Sentinel for "omit the `usage` key entirely", which is not the same as "usage: null".
_NO_USAGE = object()


def _tail_shape(req: "OpenAICompletionRequest") -> tuple:
    """Structural signature of the conversation's TAIL — the only thing `add_generation_prompt`
    branches on. Roles plus "did this turn carry tool calls", never message TEXT, so it is a sound
    cache key and two different conversations that end the same way share one render."""
    msgs = req.messages or []
    return tuple(
        (m.role, bool(m.tool_calls), m.content is not None) for m in msgs[-_SHAPE_TAIL:]
    )


def _generation_prompt_tail(req: "OpenAICompletionRequest", kwargs: dict | None) -> str | None:
    """What `add_generation_prompt` APPENDS for THIS request — rendered from the request's own
    messages, not from a stand-in conversation.

    Returns the appended text, `""` when the template appended NOTHING, or None if it cannot render.
    The empty string is the case this function exists for: a template may decline to open a new turn
    (Gemma-4 after a tool call/result — the model is meant to continue the model turn already open),
    and that is invisible to a probe rendered from a `[user]` conversation, which always gets a full
    generation prompt back.

    Diffing the two renders rather than reading the whole prompt also keeps history OUT of the answer:
    only the appended tail is inspected for delimiters, so an earlier turn carrying a stray delimiter
    (e.g. reasoning markup that leaked into an assistant `content` and got replayed) cannot flip the
    state of a prompt that in fact ends with a clean generation prompt."""
    tok = _frontend_tokenizer()
    if tok is None or not req.messages:
        return None
    messages = [msg.model_dump(exclude_none=True) for msg in req.messages]
    _normalize_tool_args(messages)
    try:
        on = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **(kwargs or {}))
        off = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **(kwargs or {}))
    except Exception as e:  # noqa: BLE001 — a template that rejects the render tells us nothing
        logger.debug("generation-prompt render failed for %s: %s", kwargs, e)
        return None
    i = 0
    while i < min(len(on), len(off)) and on[i] == off[i]:
        i += 1
    return on[i:]


def _probe_generation_prompt(kwargs: dict | None) -> str | None:
    """Render a STAND-IN generation prompt under `kwargs`, cached. None if it can't render.

    The fallback for a request with no messages of its own to render (the raw-completion lane's
    callers). Prefer `_generation_prompt_tail`, which reads the actual request."""
    key = tuple(sorted((k, repr(v)) for k, v in (kwargs or {}).items()))
    if key not in _PROMPT_PROBE_CACHE:
        rendered = None
        tok = _frontend_tokenizer()
        if tok is not None:
            try:
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": "hi"}], tokenize=False,
                    add_generation_prompt=True, **(kwargs or {}),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("generation-prompt probe render failed for %s: %s", kwargs, e)
        _PROMPT_PROBE_CACHE[key] = rendered
    return _PROMPT_PROBE_CACHE[key]


def _prompt_thinking_state(req: "OpenAICompletionRequest") -> Tuple[bool, bool]:
    """`(span_open, reasoning_possible)` for THIS request, read off its rendered generation prompt.

    This replaces a DEFAULT. The old code assumed thinking was on for every request, on the theory
    that "reasoning models open `<think>` in the generation prompt" — true for Qwen3/GLM, false for
    everything else, and when it is false the parser routes the WHOLE completion into
    `reasoning_content` and returns `content=""`. That is a silent, total output loss: generation is
    perfect, every OpenAI client reading `.content` gets an empty string, and nothing logs an error.
    Gemma-4 hit it on every request.

    The ground truth is the rendered prompt — a span is open iff the template injected an opener and
    did not close it. Reading it needs no model-name branch, no allow-list and no new config field.
    Falls back to the old optimistic default only when nothing renders, i.e. when we truly cannot
    tell.

    THREE states, not two, because a template can also decline to start a turn at all:

    * generation prompt ABSENT (`tail == ""`) — the template appended nothing, so the model is
      CONTINUING a turn that is already open and whatever convention that turn carries is still in
      force. Gemma-4 does this after every tool call/result, and it is why "thinking off" cannot be
      expressed there: off is spelled by INJECTING a pre-closed empty span
      (`<|channel>thought\\n<channel|>`) inside the very branch that got skipped. The model duly opens
      a thought channel and the completion begins mid-span — measured, its first token is literally
      `thought`, with no opener. Reported as `(True, True)`: treat the span as open, which is what
      makes the streaming splitter route that scratch to `reasoning_content` instead of streaming it
      to the user as the answer, and what arms the reasoning-budget backstop that force-closes a
      runaway (observed unfixed: 34k characters of chain-of-thought delivered as `content`, the last
      27k of it a single sentence repeated 676 times).
    * generation prompt PRESENT — read the delimiters out of the appended tail, as before.
    * nothing renders — the old optimistic default.

    Cached on `(kwargs, tail shape)`: the branch keys on the conversation's trailing structure, never
    on message text, so one render answers every request that ends the same way."""
    parser = _reasoning_parser()
    if parser is None:
        return False, False
    # The raw-completion lane has no template: the request's `prompt` IS the rendered prompt, so read
    # the delimiters straight out of it (a raw prompt ending in `<think>` really is mid-reasoning).
    if req.messages is None and isinstance(req.prompt, str):
        return parser.prompt_state(req.prompt)
    kwargs = _resolve_chat_template_kwargs(req)
    key = (tuple(sorted((k, repr(v)) for k, v in (kwargs or {}).items())), _tail_shape(req))
    if key not in _THINKING_STATE_CACHE:
        tail = _generation_prompt_tail(req, kwargs)
        if tail is None:
            # No messages of our own to render (or the template refused): fall back to the stand-in.
            rendered = _probe_generation_prompt(kwargs)
            state = (True, True) if rendered is None else parser.prompt_state(rendered)
        elif not tail.strip():
            state = (True, True)  # no new turn -> the model continues the one already open
        else:
            state = parser.prompt_state(tail)
        _THINKING_STATE_CACHE[key] = state
    return _THINKING_STATE_CACHE[key]


def _thinking_open(req: "OpenAICompletionRequest") -> bool:
    """Is the model INSIDE a reasoning span at completion token 0? Feeds `parse(thinking_open=…)` and
    the streaming splitter's initial state — i.e. it decides where an output with NO closing
    delimiter goes, which is the difference between an answer and an empty `content`."""
    if _thinking_opted_out(req):
        return False
    return _prompt_thinking_state(req)[0]


def _thinking_active(req: "OpenAICompletionRequest") -> bool:
    """Whether reasoning can still appear in this completion — the gate for the reasoning-token budget
    and the grammar think-gate. Weaker than `_thinking_open`: also true when the prompt carries no
    delimiter at all, because the template left the opener to the MODEL (Gemma-4 with thinking on,
    older Qwen3 templates) and the gate must stay armed for a span that has not opened YET. False
    once the prompt has CLOSED the span, which is how every family spells thinking-off."""
    if _thinking_opted_out(req):
        return False
    return _prompt_thinking_state(req)[1]


_REASONING_PARSER = None
_REASONING_PARSER_SET = False


def _reasoning_parser():
    """Cached ReasoningParser for the served checkpoint, resolved ONCE (cascade: see
    `resolve_reasoning_parser`). On the default "auto" the delimiters are DERIVED from the model's own
    chat template rather than looked up by family name — a name table is a model-name branch by
    another spelling, and a checkpoint with no row silently fell through to `<think>`/`</think>`,
    matched nothing, and had its whole reply misrouted. An explicit --reasoning-parser still wins."""
    global _REASONING_PARSER, _REASONING_PARSER_SET
    if not _REASONING_PARSER_SET:
        cfg = get_global_state().config
        try:
            declared = load_generation_config(cfg.model_path).get("reasoning_parser")
        except Exception:  # noqa: BLE001
            declared = None
        _REASONING_PARSER, how = resolve_reasoning_parser(
            _frontend_tokenizer(),
            requested=getattr(cfg, "reasoning_parser", "auto"),
            declared=declared,
        )
        _REASONING_PARSER_SET = True
        if _REASONING_PARSER is None:
            logger.info("reasoning extraction DISABLED (%s)", how)
        else:
            # The answer-turn header is logged alongside the pair because it is now load-bearing for
            # DECODING (it releases the scheduler's reasoning gate), not just for stripping text.
            logger.info("reasoning delimiters %r … %r (answer-turn header %r) (%s)",
                        _REASONING_PARSER.start_token, _REASONING_PARSER.end_token,
                        _REASONING_PARSER.turn_header, how)
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
# (F) Muse-Glimmer native ATEM XML:
#   <atem:function_calls><atem:invoke name="NAME">
#     <atem:parameter name="KEY">VALUE</atem:parameter>…
#   </atem:invoke>…</atem:function_calls>
# Scanned per-INVOKE rather than per-block, because ONE block expresses N parallel calls — which the
# shared `_add(inner)` path (one body -> one call) cannot represent. The block regex exists only to
# strip the wrapper out of `content` afterwards.
_ATEM_BLOCK_RE = re.compile(r"<atem:function_calls>.*?</atem:function_calls>", re.DOTALL)
_ATEM_INVOKE_RE = re.compile(r'<atem:invoke\s+name="([^"]*)"\s*>(.*?)</atem:invoke>', re.DOTALL)
_ATEM_PARAM_RE = re.compile(
    r'<atem:parameter\s+name="([^"]*)"\s*>(.*?)</atem:parameter>', re.DOTALL
)
# An unclosed ATEM block (model truncated mid-call). Recovered like the other unclosed wrappers.
_ATEM_UNCLOSED_RE = re.compile(r"<atem:function_calls>(?!.*</atem:function_calls>)(.*)$", re.DOTALL)
# Dangling ATEM wrapper tags left after the invokes are lifted out.
_ATEM_ORPHAN_RE = re.compile(r"</?atem:function_calls>")
_SQUARE_WRAP_RE = re.compile(r"\[(/?(?:" + "|".join(_TOOL_WRAPPERS) + r"))\]")
# (D) Laguna native tool args: `<arg_key>NAME</arg_key><arg_value>VALUE</arg_value>` pairs (NOT the
# Qwen3 `<parameter=…>` form). The call is `<tool_call>fname<arg_key>…</arg_key><arg_value>…` — a bare
# function-name head (no `<function=>` wrapper) followed by these pairs. Parse both so the call lands
# as a real tool_call instead of leaking `<arg_value>` markup into content.
_ARG_KV_RE = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>", re.DOTALL)
# Orphan wrapper open/close tags left in content after a call is parsed (e.g. the model emitted an opener
# but no closer around an inline function) — strip them so `content` isn't polluted with dangling markup.
_ORPHAN_WRAP_RE = re.compile(r"</?(?:" + "|".join(_TOOL_WRAPPERS) + r")>|<\|tool_call>|<tool_call\|>")
# (E) GEMMA-4 native. The wrapper is PIPE-INSIDE and ASYMMETRIC — `<|tool_call>` … `<tool_call|>` —
# so no `_TOOL_WRAPPERS` entry matches, and `<tool_call>` is NOT a substring of `<|tool_call>`:
# exactly the trap the `_TOOL_STRUCT_WRAPPERS` comment already records for ZAYA. Both delimiters are
# real special tokens in the checkpoint tokenizer, as is `<|"|>`, which is how the template spells a
# STRING DELIMITER. Per the checkpoint's chat_template.jinja:
#     '<|tool_call>call:' + name + '{' + key ':' format_argument(value) … + '}<tool_call|>'
# where format_argument renders str -> `<|"|>s<|"|>`, bool -> true/false, mapping -> `{k:v,…}`,
# sequence -> `[a,b]`, anything else raw. `escape_keys=False` propagates from the top level, so EVERY
# key is BARE — the body is JSON-shaped but is NOT JSON. Live example (session facc711200c0, which
# leaked the whole block into `content` with tool_calls=[]):
#     <|tool_call>call:web_extract{urls:[<|"|>https://a.aliexpress.com/_mPWq9G7<|"|>]}<tool_call|>
_GEMMA_TOOL_BLOCK_RE = re.compile(r"<\|tool_call>\s*(.*?)\s*<tool_call\|>", re.DOTALL)
_GEMMA_TOOL_UNCLOSED_RE = re.compile(r"<\|tool_call>(?!.*<tool_call\|>)(.*)$", re.DOTALL)
_GEMMA_CALL_HEAD_RE = re.compile(r"^call:\s*([A-Za-z_][\w.]*)\s*(?=\{)")
_GEMMA_BAREWORD_RE = re.compile(r"[A-Za-z_]\w*")
_GEMMA_QUOTE = '<|"|>'
# What a function name may look like. Used to stop the permissive bare-name branch (D) from turning
# an unparsed body into a tool call named after its own leading junk.
_TOOL_NAME_RE = re.compile(r"[A-Za-z_][\w.\-]*")
# A wrapper opener with NO matching closer anywhere after it — the truncated-mid-call shape. Group 1 is
# the block body (opener to end of text) to hand to the same inner parsers.
_UNCLOSED_WRAP_RE = re.compile(
    r"<(?:" + "|".join(_TOOL_WRAPPERS) + r")>(?!.*</(?:" + "|".join(_TOOL_WRAPPERS) + r")>)(.*)$",
    re.DOTALL,
)
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


def _gemma_args_to_json(text: str) -> str:
    """Gemma-4 argument-object syntax -> JSON text, in one pass.

    Two things differ from JSON and both must be handled together, because you cannot tell a key from
    a string without tracking where the string delimiters are:
      * strings are delimited by the `<|"|>` TOKEN, not by `"`, and their contents are raw (a URL with
        a `"` or a newline in it must be re-escaped, which is why this rebuilds them via json.dumps
        rather than swapping the delimiter for a quote character);
      * keys are BARE (`urls:` not `"urls":`), since the template propagates escape_keys=False.
    Literals true/false/null are barewords too, so a bareword only becomes a key when the next
    non-space character is ':'. Everything else (braces, brackets, commas, numbers) passes through."""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith(_GEMMA_QUOTE, i):
            j = text.find(_GEMMA_QUOTE, i + len(_GEMMA_QUOTE))
            if j < 0:
                raise ValueError("unterminated <|\"|> string")
            out.append(json.dumps(text[i + len(_GEMMA_QUOTE) : j]))
            i = j + len(_GEMMA_QUOTE)
            continue
        m = _GEMMA_BAREWORD_RE.match(text, i)
        if m:
            word = m.group(0)
            k = m.end()
            rest = k
            while rest < n and text[rest].isspace():
                rest += 1
            if word not in ("true", "false", "null") and rest < n and text[rest] == ":":
                out.append(json.dumps(word))  # bare KEY -> quoted
            else:
                out.append(word)  # true/false/null, or a bareword we leave for json to reject
            i = k
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _parse_gemma_tool_call(inner: str) -> Tuple[str, dict] | None:
    """(E) Gemma-4 native: `call:NAME{k:v,…}` -> (name, args). None if it isn't that shape.

    The object is decoded with raw_decode rather than by brace-counting: once the strings are proper
    JSON, raw_decode tracks nesting AND string contents correctly, so a `}` inside an argument value
    cannot terminate the object early."""
    head = _GEMMA_CALL_HEAD_RE.match(inner.strip())
    if not head:
        return None
    body = inner.strip()[head.end() :]
    try:
        args, _ = json.JSONDecoder().raw_decode(_gemma_args_to_json(body))
    except (ValueError, json.JSONDecodeError):
        return None
    return (head.group(1), args) if isinstance(args, dict) else None


def _parse_one_tool_call(inner: str) -> Tuple[str, dict] | None:
    """Parse one <tool_call> body (either format) -> (name, arguments_dict), or None."""
    inner = inner.strip()
    # (E) Gemma-4 native `call:NAME{…}`. MUST be tried before (D): its body starts with neither `{`
    # nor `<`, so (D)'s bare-name branch would otherwise claim it and split on the first `<` — which
    # lands INSIDE the `<|"|>` string delimiter, yielding a tool call literally named
    # `call:web_extract{urls:[` with no arguments. A wrong call is worse than an unparsed one.
    gemma = _parse_gemma_tool_call(inner)
    if gemma is not None:
        return gemma
    if inner.startswith("{"):  # (A) Hermes JSON
        try:
            call = json.loads(inner)
            if call.get("name"):
                return call["name"], call.get("arguments", {})
        except json.JSONDecodeError:
            pass
    # (F) Muse-Glimmer ATEM XML. Only the FIRST invoke, because this function's contract is one body
    # -> one call. That is exact for the streaming path (which is where it is reached from: a block
    # is withheld until its closer, then handed here), and the NON-streaming path never relies on it
    # — `_parse_tool_calls` scans invokes directly, so a parallel call yields every one of them.
    atem = _ATEM_INVOKE_RE.search(inner)
    if atem and atem.group(1).strip():
        return (
            atem.group(1).strip(),
            {k.strip(): _coerce(v) for k, v in _ATEM_PARAM_RE.findall(atem.group(2))},
        )
    fn = _XML_FN_RE.search(inner)  # (B) Qwen3 XML: <function=NAME>…<parameter=…>…</function>
    if fn:
        args = {k.strip(): _coerce(v.strip()) for k, v in _XML_PARAM_RE.findall(fn.group(2))}
        return fn.group(1).strip(), args
    inl = _INLINE_FN_RE.search(inner)  # (C) ZAYA deviation: inline <function=NAME(a='x', b=1)>
    if inl:
        return inl.group(1).strip(), _parse_pycall_args(inl.group(2))
    # (D) Laguna native: `fname<arg_key>k</arg_key><arg_value>v</arg_value>…` — a bare function-name
    # head then arg_key/arg_value pairs (no `<function=>` wrapper). The head is the text before the
    # first tag; args from the pairs (empty for a no-arg call). Guarded to not shadow A/B/C (which
    # start with `{` or `<`), so this only fires on the bare-name form.
    # The head must be a PLAUSIBLE FUNCTION NAME, not merely non-empty. Without this, any body that
    # reached (D) unparsed became a tool call named after its own leading junk — e.g. a Gemma-4 body
    # whose arguments were malformed split on the `<` of the `<|"|>` delimiter and produced a call
    # literally named `call:bad{x:someBareword}`. Refusing is correct there: an unparsed call is
    # visible, a WRONG call is dispatched.
    if inner and not inner.startswith(("{", "<")):
        name = inner.split("<", 1)[0].strip()
        if name and _TOOL_NAME_RE.fullmatch(name):
            args = {k.strip(): _coerce(v.strip()) for k, v in _ARG_KV_RE.findall(inner)}
            return name, args
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
    # (F) ATEM invokes, scanned FIRST and independently: one `<atem:function_calls>` block can carry
    # several `<atem:invoke>`s (that is how the format spells parallel calls), so each invoke becomes
    # its own tool call rather than being funnelled through the one-body-one-call `_add`.
    for m in _ATEM_INVOKE_RE.finditer(text):
        name = m.group(1).strip()
        if not name:
            continue
        # Values are NOT stripped: the format's own instructions state that spaces in string values
        # are significant. `_coerce` still lifts JSON/number/bool forms, and it tolerates the
        # surrounding whitespace a multi-line value carries.
        args = {k.strip(): _coerce(v) for k, v in _ATEM_PARAM_RE.findall(m.group(2))}
        tool_calls.append(
            {
                "id": f"call_{uid}_{len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        _add(m.group(1))
    # (E) Gemma-4's asymmetric `<|tool_call>…<tool_call|>` block — a separate scan because its closer
    # is not `</opener>`, so it cannot ride the shared wrapper alternation.
    for m in _GEMMA_TOOL_BLOCK_RE.finditer(text):
        _add(m.group(1))
    # Bare blocks in whatever text remains after removing the wrapped blocks (so an inner block is not
    # double-counted): canonical `<function=…></function>`, then the inline `<function=NAME(...)>` form.
    remainder = _GEMMA_TOOL_BLOCK_RE.sub("", _TOOL_CALL_BLOCK_RE.sub("", text))
    for m in _BARE_FN_BLOCK_RE.finditer(remainder):
        _add(m.group(0))
    for m in _INLINE_FN_RE.finditer(_BARE_FN_BLOCK_RE.sub("", remainder)):
        _add(m.group(0))
    # UNCLOSED wrapper recovery. `_TOOL_CALL_BLOCK_RE` hard-requires a matching closer, so a model that
    # emits `<tool_call>{"name": …` and then runs out of tokens (or drifts into a different tool syntax
    # mid-call) matched nothing at all and the raw markup leaked into `content` as if it were prose.
    # Take everything from the dangling opener to end-of-text and try the same inner parsers: when the
    # JSON/XML is already complete this recovers a real tool call, and when it genuinely is truncated we
    # say so in the log instead of silently passing markup off as the model's answer.
    if not tool_calls:
        # An ATEM block that opened but whose LAST invoke never closed: the complete invokes before
        # it were already taken above, so reaching here means none completed. Retry the tail against
        # the invoke scanner in case only the outer `</atem:function_calls>` is missing.
        atem_open = _ATEM_UNCLOSED_RE.search(text)
        if atem_open:
            for m in _ATEM_INVOKE_RE.finditer(atem_open.group(1)):
                name = m.group(1).strip()
                if not name:
                    continue
                args = {k.strip(): _coerce(v) for k, v in _ATEM_PARAM_RE.findall(m.group(2))}
                tool_calls.append(
                    {
                        "id": f"call_{uid}_{len(tool_calls)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                )
            if tool_calls:
                return (text[: atem_open.start()].strip() or None), tool_calls
    if not tool_calls:
        unclosed = _GEMMA_TOOL_UNCLOSED_RE.search(text) or _UNCLOSED_WRAP_RE.search(text)
        if unclosed:
            before = len(tool_calls)
            _add(unclosed.group(1))
            if len(tool_calls) > before:
                return (text[: unclosed.start()].strip() or None), tool_calls
            logger.warning(
                "tool-call block opened but never closed and its body did not parse (%d chars) — "
                "leaving it in content; the model most likely truncated or mixed tool syntaxes mid-call",
                len(unclosed.group(1)),
            )
    if not tool_calls:
        return text, []
    content = _ATEM_BLOCK_RE.sub("", _GEMMA_TOOL_BLOCK_RE.sub("", text))
    content = _INLINE_FN_RE.sub("", _BARE_FN_BLOCK_RE.sub("", _TOOL_CALL_BLOCK_RE.sub("", content)))
    # Lift out any invoke that survived (an ATEM block whose wrapper was malformed) plus the
    # wrapper tags themselves, so no `<atem:…>` markup is ever passed off as the model's prose.
    content = _ATEM_ORPHAN_RE.sub("", _ATEM_INVOKE_RE.sub("", content))
    content = _ORPHAN_WRAP_RE.sub("", content).strip()  # drop any dangling wrapper opener/closer
    return (content or None), tool_calls


# --- streaming tool-call parsing ------------------------------------------------------------------
# The block openers we recognise, mapped to their closers. `<tool_call>` wraps either inner format
# (Hermes JSON or Qwen3 `<function=…>` XML); a bare `<function=…>` (no wrapper) is also accepted so
# the parser degrades to whatever the model actually emits. Detection mirrors the reasoning streamer:
# text before any opener flows through as `content`; once inside a block the markup is withheld and,
# on the closing tag, re-emitted as OpenAI streaming `delta.tool_calls`.
# ORDER MATTERS: `<|tool_call>` must precede `<tool_call>`. They are different strings (the pipe sits
# INSIDE the angle bracket) so neither contains the other, but keeping the Gemma opener first makes the
# asymmetry explicit to anyone extending this table — its closer is `<tool_call|>`, NOT `</…>`.
_TOOL_OPENERS = (
    "<|tool_call>",
    "<tool_call>",
    "<zyphra_tool_call>",
    "<atem:function_calls>",
    "<tools>",
    "<function=",
)
_TOOL_CLOSERS = {
    "<|tool_call>": "<tool_call|>",  # Gemma-4: asymmetric, pipe-inside
    "<tool_call>": "</tool_call>",
    "<zyphra_tool_call>": "</zyphra_tool_call>",
    # Muse-Glimmer ATEM. The closer is the BLOCK's, not the invoke's, so a parallel call (several
    # `<atem:invoke>`s in one block) is withheld and emitted as one complete set — matching the
    # non-streaming parser, which also scans invokes within the whole block.
    "<atem:function_calls>": "</atem:function_calls>",
    "<tools>": "</tools>",
    "<function=": "</function>",
}


# Seconds of dead air tolerated while a tool-call block is buffering before an SSE keepalive comment
# goes out. Short enough to beat the usual 30-60s proxy/client idle timeouts, long enough that a
# normal (fast-closing) block never emits one.
_TOOL_BLOCK_KEEPALIVE_S = 10.0


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
        # Set by flush() when the stream ended inside a tool-call block that could NOT be parsed, so
        # its raw markup was surfaced as content. The caller MUST NOT report finish_reason="stop"
        # then: the turn did not complete, and a client told "stop" renders that markup to the user
        # as the final answer instead of continuing or retrying.
        self.unparsed_tail = False
        # One-shot latch for the oversize-block warning (see `held_chars`).
        self._warned_oversize = False

    @property
    def held_chars(self) -> int:
        """How much output is currently trapped in an unclosed block. Zero outside a block (the
        partial-opener buffer is a handful of chars and always drains). While this is nonzero the
        stream emits NOTHING to the client, so the caller uses it to keep the wire alive."""
        return len(self.buf) if self.in_tool else 0

    def warn_if_oversize(self, threshold: int = 8192) -> bool:
        """True exactly once, when the held body first crosses ``threshold``. A tool call whose
        arguments run to kilobytes with no closer in sight is the signature of a model that latched
        an opener and kept decoding — measured 2026-08-04: ~60k tokens generated, zero delivered,
        because `push` holds the whole body and the client just sees a dead connection. Surfacing it
        in the serve log is the only way that failure is visible while it is happening."""
        if self.in_tool and not self._warned_oversize and len(self.buf) >= threshold:
            self._warned_oversize = True
            return True
        return False

    # `<function=…>` is the one opener that is part of its own body — the regex parsers need the tag
    # itself — so it is the ONLY case where the block is passed through whole. Every other opener is a
    # WRAPPER and must be stripped. This was previously an explicit allow-list of wrapper openers,
    # which silently excluded Gemma-4: its block reached the inner parsers with `<|tool_call>` still
    # attached, matched nothing, and `_emit_call` dropped it as malformed — so a valid call vanished
    # from a STREAMED response entirely (no content, no tool_calls, finish_reason=stop). Keyed on the
    # exception instead of the members, so the next wrapper family is handled by default.
    _WHOLE_BLOCK_OPENER = "<function="

    def _parse_block(self, block: str) -> Tuple[str, dict] | None:
        if self.opener == self._WHOLE_BLOCK_OPENER:
            return _parse_one_tool_call(block)  # regex finds the fn tag inside
        inner = block[len(self.opener):]
        closer = _TOOL_CLOSERS[self.opener]
        if inner.endswith(closer):  # tolerate a block handed over without its closer
            inner = inner[: -len(closer)]
        return _parse_one_tool_call(inner)

    def _parse_unclosed(self, buf: str) -> Tuple[str, dict] | None:
        """Parse a block the stream ended INSIDE (opener seen, closer never arrived). Same inner
        formats as ``_parse_block``; the only difference is there is no closer to strip."""
        if self.opener == self._WHOLE_BLOCK_OPENER:
            return _parse_one_tool_call(buf)
        return _parse_one_tool_call(buf[len(self.opener):])

    def _emit_parsed(self, name: str, args) -> List[dict]:
        args_str = args if isinstance(args, str) else json.dumps(args)
        i = self.next_index
        self.next_index += 1
        self.emitted = True
        return [
            {"index": i, "id": f"call_{self.uid}_{i}", "type": "function",
             "function": {"name": name, "arguments": ""}},
            {"index": i, "function": {"arguments": args_str}},
        ]

    def _emit_call(self, block: str) -> List[dict]:
        parsed = self._parse_block(block)
        if parsed is None:
            return []  # malformed block -> drop it (never leak markup into content)
        return self._emit_parsed(*parsed)

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
                self._warned_oversize = False  # re-arm: a LATER block can go oversize too
        return ("".join(content_parts) or None), tool_deltas

    def flush(self) -> Tuple[str | None, List[dict]]:
        """At stream end, drain whatever is still buffered — NEVER silently. A block the stream ended
        inside (hit the token cap mid-call) used to be discarded outright, which returned
        ``finish_reason="length"`` with empty content AND zero tool_calls: the caller could not tell a
        truncated call from a model that chose to say nothing, and multi-KB arguments vanished. Now the
        partial block is parsed if it is already complete enough to parse (the common case — the cap
        lands in trailing whitespace or the closer), and otherwise surfaced verbatim as ``content`` so
        the bytes reach the caller and the truncation is visible. A buffered partial opener turned out
        to be literal ``content`` and is emitted."""
        if self.in_tool:
            buf = self.buf
            parsed = self._parse_unclosed(buf)
            self.buf, self.in_tool, self.opener = "", False, None
            if parsed is not None:
                return None, self._emit_parsed(*parsed)
            self.unparsed_tail = bool(buf)
            logger.warning(
                "stream ended inside an unclosed tool-call block (%d chars) whose body did not parse; "
                "surfacing it as content rather than dropping it, and reporting finish_reason=length "
                "so the caller treats the turn as truncated. Cause is either the token cap or the "
                "model emitting a stop token mid-call — check completion_tokens against max_tokens",
                len(buf),
            )
            return (buf or None), []
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
    # SERVABLE context, not the checkpoint's. These are min(model max_position, KV pool) — for this
    # 35B the checkpoint says 262,144 while the pool allows 73,872. Published because a client with
    # nothing to read must guess, and the guess is wrong in the dangerous direction: Hermes
    # auto-detect settles on 131,072 and then sends prompts the engine can only reject. Both spellings
    # are emitted deliberately — `max_model_len` is what vLLM publishes (so vLLM-shaped clients find
    # it), `context_length` is what Ollama-shaped clients read.
    max_model_len: int | None = None
    context_length: int | None = None


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
    # Servable context, learned from the scheduler's first stats snapshot (it depends on the KV pool,
    # which only the scheduler knows). None until then, and /v1/models then omits it rather than
    # publishing the checkpoint's number — which would be wrong in the direction that hurts.
    max_seq_len: int | None = None

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
                if msg.max_seq_len:
                    self.max_seq_len = int(msg.max_seq_len)
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
                        prefill_seconds=msg.prefill_seconds,
                        prefix_cache_hit_tokens=msg.prefix_cache_hit_tokens,
                        prefix_cache_prompt_tokens=msg.prefix_cache_prompt_tokens,
                        prefill_computed_tokens=msg.prefill_computed_tokens,
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
            if getattr(ack, "error", None):
                yield f"data: {json.dumps({'error': ack.error})}\n".encode()
                yield "data: [DONE]\n".encode()
                return
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(
        self, uid: int, reasoning_stream=None, tool_stream=None, include_usage: bool = False
    ):
        first_chunk = True
        prompt_tokens = completion_tokens = 0
        finish_reason = "stop"
        # Wall-clock of the last byte written. While `tool_stream` is holding an unclosed block it
        # emits nothing, so without this the socket goes silent for as long as the model keeps
        # decoding into that block — indistinguishable, from the client, from a hung server.
        last_write = time.monotonic()

        def _chunk(delta: dict) -> bytes:
            payload = {
                "id": f"cmpl-{uid}",
                "object": "chat.completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            return f"data: {json.dumps(payload)}\n\n".encode()

        async for ack in self.wait_for_ack(uid):
            if getattr(ack, "error", None):
                # Headers are already sent, so this cannot become a 4xx — emit an explicit error
                # event so the client sees a REASON instead of an empty stream that just stops.
                _err = json.dumps({"error": {"message": ack.error,
                                             "type": "invalid_request_error",
                                             "code": "context_length_exceeded"}})
                yield f"data: {_err}\n\n".encode()
                yield "data: [DONE]\n\n".encode()
                return
            delta: dict = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            tool_deltas: List[dict] = []
            if ack.incremental_output:
                # Tool calls are their OWN channel — detect them on the RAW stream FIRST (before the
                # reasoning split), so a <tool_call> the model emits (Laguna emits them WITHOUT ever
                # closing </think>, so the reasoning splitter would otherwise trap the whole block in
                # reasoning_content) is pulled out as delta.tool_calls. The non-tool remainder then
                # goes through the reasoning split (pre-</think> scratch -> reasoning_content, answer
                # -> content). Buffers a partial opener across chunks.
                if tool_stream is not None:
                    nontool, tool_deltas = tool_stream.push(ack.incremental_output)
                else:
                    nontool = ack.incremental_output
                if nontool:
                    if reasoning_stream is not None:
                        r_delta, c_delta = reasoning_stream.push(nontool)
                        if r_delta:
                            delta["reasoning_content"] = r_delta
                        if c_delta:
                            delta["content"] = c_delta
                    else:
                        delta["content"] = nontool
            completion_tokens = max(completion_tokens, ack.completion_tokens)
            prompt_tokens = ack.prompt_tokens or prompt_tokens
            if ack.finish_reason:
                finish_reason = ack.finish_reason

            # Emit the content/reasoning delta (if any), then one chunk per tool-call fragment.
            if delta:
                yield _chunk(delta)
            for td in tool_deltas:
                yield _chunk({"tool_calls": [td]})
            if delta or tool_deltas:
                last_write = time.monotonic()
            elif tool_stream is not None and tool_stream.held_chars:
                # Nothing to emit because the whole body is trapped in an unclosed tool block. Keep
                # the wire alive with an SSE COMMENT — legal per the SSE spec, ignored by every
                # OpenAI client (it is not a `data:` line, so it never reaches the delta stream), and
                # enough to stop idle-timeout proxies and client stall detectors from firing on a
                # server that is in fact still decoding.
                if time.monotonic() - last_write >= _TOOL_BLOCK_KEEPALIVE_S:
                    last_write = time.monotonic()
                    yield b": tool-call block open\n\n"
                if tool_stream.warn_if_oversize():
                    logger.warning(
                        "uid=%s has %d chars buffered inside an unclosed %s block after %d completion "
                        "tokens — nothing has been streamed to the client since the opener. Either the "
                        "model is generating a very large tool argument or it latched an opener it will "
                        "never close; the turn will only surface when max_tokens is hit.",
                        uid, tool_stream.held_chars, tool_stream.opener, completion_tokens,
                    )

            if ack.finished:
                break

        # final chunk: flush any buffered reasoning tail (model never closed </think>) and any tool
        # tail, then finish_reason + usage (OpenAI carries usage on the terminal chunk).
        final_delta: dict = {}
        # Tool tail FIRST (it fed off the RAW stream): emit any final tool fragment, then a buffered
        # partial-opener that turned out to be literal text still flows through the reasoning split.
        nontool_tail = None
        if tool_stream is not None:
            nontool_tail, t_tail = tool_stream.flush()
            for td in t_tail:
                yield _chunk({"tool_calls": [td]})
            if tool_stream.emitted and finish_reason != "length":
                finish_reason = "tool_calls"
            elif tool_stream.unparsed_tail:
                # A tool call was cut off mid-emission and its markup is going out as content. Saying
                # "stop" would assert the model finished normally, so the client renders raw
                # `<tool_call>{…` to the user and ends the turn. "length" is the truthful signal and
                # is what makes an agent harness treat this as a partial turn to continue or retry.
                finish_reason = "length"
        if reasoning_stream is not None:
            if nontool_tail:
                r2, c2 = reasoning_stream.push(nontool_tail)
                if r2:
                    final_delta["reasoning_content"] = r2
                if c2:
                    final_delta["content"] = c2
            # flush() returns BOTH tails: buffered reasoning (a partial close tag) and buffered
            # content (a head held back while it might have been a model-side opener). Dropping the
            # content one would silently truncate a reply shorter than the opening delimiter.
            r_tail, c_tail = reasoning_stream.flush()
            if r_tail:
                final_delta["reasoning_content"] = final_delta.get("reasoning_content", "") + r_tail
            if c_tail:
                final_delta["content"] = final_delta.get("content", "") + c_tail
        elif nontool_tail:
            final_delta["content"] = nontool_tail
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "chat.completion.chunk",
            "choices": [{"delta": final_delta, "index": 0, "finish_reason": finish_reason}],
            # OpenAI spec: in include_usage mode `usage` is null on every content chunk (incl. this
            # finish chunk) and the totals ride a dedicated trailing chunk (below). Otherwise keep the
            # totals here (back-compat for clients that read usage off the finish chunk).
            "usage": None if include_usage else usage,
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        if include_usage:
            # Spec-compliant final usage chunk: choices is an empty array, usage carries the totals.
            # This is the chunk langchain usage_metadata / budget guards look for.
            usage_chunk = {
                "id": f"cmpl-{uid}",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_text_completions(self, uid: int, model: str, include_usage: bool = False):
        """SSE for /v1/completions — the TEXT-completion wire shape, which is a different object from
        the chat stream above and cannot be produced by it.

        Every chunk is `{"object":"text_completion","choices":[{"index":0,"text":…}]}`; there is no
        `delta`, no `role`, and no reasoning/tool channel to split into (see the route docstring: this
        lane returns the raw continuation verbatim). `id`/`created`/`model` ride EVERY chunk, unlike
        `stream_chat_completions` which omits `created`/`model` — openai-python's `Completion` model
        declares all five as required, so a chunk missing them is a client-side ValidationError rather
        than a rendered token."""
        created = int(time.time())
        prompt_tokens = completion_tokens = 0
        finish_reason = "stop"

        def _chunk(text: str, finish: str | None, usage: dict | None = _NO_USAGE) -> bytes:
            payload = {
                "id": f"cmpl-{uid}",
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
            }
            # `usage` is OMITTED on content chunks and PRESENT (possibly null) on the terminal one —
            # an explicit null is how the spec says "the totals are not here", and a client that
            # reads `chunk.usage` on the finish chunk must see the key, not a KeyError.
            if usage is not _NO_USAGE:
                payload["usage"] = usage
            return f"data: {json.dumps(payload)}\n\n".encode()

        async for ack in self.wait_for_ack(uid):
            if getattr(ack, "error", None):
                # Headers are already out, so this can no longer become a 4xx. Emit an explicit error
                # event: an empty stream that simply stops is indistinguishable from a healthy but
                # short completion, so the caller would treat a REFUSED request as a valid empty answer.
                _err = json.dumps({"error": {"message": ack.error,
                                             "type": "invalid_request_error",
                                             "code": "context_length_exceeded"}})
                yield f"data: {_err}\n\n".encode()
                yield "data: [DONE]\n\n".encode()
                return
            completion_tokens = max(completion_tokens, ack.completion_tokens)
            prompt_tokens = ack.prompt_tokens or prompt_tokens
            if ack.finish_reason:
                finish_reason = ack.finish_reason
            if ack.incremental_output:
                yield _chunk(ack.incremental_output, None)
            if ack.finished:
                break

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        # Terminal chunk carries finish_reason with an empty text, mirroring the chat lane's contract
        # (and OpenAI's): usage rides here unless include_usage asked for the spec's dedicated
        # trailing chunk, in which case it is null here and the totals follow with empty `choices`.
        yield _chunk("", finish_reason, None if include_usage else usage)
        if include_usage:
            yield f"data: {json.dumps({'id': f'cmpl-{uid}', 'object': 'text_completion', 'created': created, 'model': model, 'choices': [], 'usage': usage})}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming text completion for user %s", uid)

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
    # Start consuming the scheduler link IMMEDIATELY, not on the first outgoing request.
    # `_create_listener_once` used to fire only from `send_one`, so on an idle serve nothing drained
    # recv_tokenizer and the scheduler's stats snapshots were never processed: /metrics read 0 and
    # /v1/models could not advertise the servable context until traffic happened to arrive. A client
    # asking "how much context do I have?" does so BEFORE sending anything, which is precisely when
    # the answer was unavailable.
    try:
        get_global_state()._create_listener_once()
    except Exception:  # noqa: BLE001 - never block startup on the metrics link
        logger.warning("could not start the scheduler listener at startup", exc_info=True)
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
                presence_penalty=req.presence_penalty,
                frequency_penalty=req.frequency_penalty,
                max_tokens=req.max_tokens,
                seed=req.seed,
                # unset -> the checkpoint's generation_config default (same resolution the OpenAI
                # endpoints use), NOT the greedy SamplingParams default.
                **dict(zip(("temperature", "top_p", "top_k"),
                           _resolve_sampling(req, state.config.model_path))),
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
async def v1_chat_completions(req: OpenAICompletionRequest, request: Request):
    # Renamed from `v1_completions`. The old name asserted a route this app did not have: the only
    # registration was (and is) `/v1/chat/completions`, so `/v1/completions` was a bare FastAPI 404
    # and the misleading symbol is what made the hole look filled on a read of the file. The real
    # `/v1/completions` is now its own handler below — this one is chat, and answers `chat.completion`.
    _bad = _reject_unsupported(req)
    if _bad is not None:
        return _bad
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
                think_answer_delim=_reasoning_answer_delim(req),
                think_close_prefix=_reasoning_close_wildcard(req)[0],
                think_close_suffix=_reasoning_close_wildcard(req)[1],
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
        if parser is not None:
            # Always attempt the split; `thinking_open` decides only where an output with NO closing
            # delimiter goes. Gating the CALL on thinking state was the other half of the bug: it made
            # "is a span open" and "should we parse at all" the same flag, so turning one off also
            # threw away the split for output that plainly contains the closing delimiter.
            reasoning_content, body = parser.parse(
                result.final_text, thinking_open=_thinking_open(req))
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
                presence_penalty=req.presence_penalty,
                frequency_penalty=req.frequency_penalty,
                max_tokens=req.max_tokens,
                seed=req.seed,
                **dict(zip(("temperature", "top_p", "top_k"), _resolve_sampling(req, state.config.model_path))),
                stop=_norm_stop(req.stop),
                stop_keep=_tool_stop_keep(req),
                grammar=_pl_rf_grammar or _pl_forced_tool_grammar or _pl_auto_tool_grammar,
                # UNCONDITIONAL close delim (was grammar-only): β-bounds reasoning on the PLAIN lane
                # too, so a thinking request without response_format can't run reasoning to max_tokens
                # and truncate with no answer. For grammar requests this is the same delim (it also
                # gates the schema until </think>); for plain thinking it's a pure backstop.
                think_close_delim=_reasoning_close_delim(req),
                think_answer_delim=_reasoning_answer_delim(req),
                think_close_prefix=_reasoning_close_wildcard(req)[0],
                think_close_suffix=_reasoning_close_wildcard(req)[1],
                think_budget=_resolve_think_budget(req),
            ),
        )
    )

    if req.stream:
        parser = _reasoning_parser()
        # `active` is the DERIVED span state, not a default: it says whether the model starts inside
        # a reasoning span. A closed start is not "no splitting" — the splitter still watches for the
        # model opening its own span and still splits on a close delimiter it meets mid-stream.
        reasoning_stream = (
            parser.stream_state(active=_thinking_open(req)) if parser is not None else None
        )
        # Stateful tool-call parser: only when tools are actually offered to the model (mirrors the
        # non-streaming path's `if req.tools`). `tool_choice:"none"` withholds the tools from the
        # template, so no blocks are emitted and this stays a no-op even when constructed.
        tool_stream = ToolCallStreamState(uid) if req.tools and req.tool_choice != "none" else None
        include_usage = bool((req.stream_options or {}).get("include_usage"))
        return StreamingResponse(
            state.stream_with_cancellation(
                state.stream_chat_completions(uid, reasoning_stream, tool_stream, include_usage),
                request, uid,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response. Accumulate the incremental
    # chunks in a list and "".join once at the end — string `+=` in the loop is O(n^2) in the output
    # length for long completions.
    content_chunks: List[str] = []
    prompt_tokens = completion_tokens = 0
    finish_reason = "stop"
    rejected: str | None = None
    async for ack in state.wait_for_ack(uid):
        if getattr(ack, "error", None):
            rejected = ack.error
            break
        content_chunks.append(ack.incremental_output)
        completion_tokens = max(completion_tokens, ack.completion_tokens)
        prompt_tokens = ack.prompt_tokens or prompt_tokens
        if ack.finish_reason:
            finish_reason = ack.finish_reason
        if ack.finished:
            break
    if rejected is not None:
        # The engine refused this request (e.g. prompt longer than the KV pool). Answer with a real
        # 4xx: returning an empty 200 would look like the model chose to say nothing, and the old
        # behaviour — no reply at all — hung the caller until its own timeout.
        return JSONResponse(status_code=400, content={"error": {
            "message": rejected, "type": "invalid_request_error", "param": "messages",
            "code": "context_length_exceeded"}})
    full_content = "".join(content_chunks)

    # Tool calls are extracted from the RAW output FIRST, BEFORE the reasoning split. A reasoning
    # model (Laguna/poolside) emits its <tool_call> block WITHOUT ever closing </think>, so splitting
    # reasoning first (thinking_open=True) traps the ENTIRE block — markup and all — inside
    # reasoning_content, leaving `body` empty: tool_calls come back null and the caller sees raw
    # `<arg_value>…</arg_value></tool_call>` leak into the payload (the spine acceptance battery's
    # signature). Mirror the streaming path (ToolCallStreamState runs pre-reasoning-split): pull the
    # calls out of the raw text, THEN reasoning-split only the non-tool remainder.
    tool_calls: List[dict] | None = None
    remainder = full_content
    # Set only when a forced bare-JSON call was recovered from AFTER a think block: the reasoning was
    # split off here, so it has to be carried across or it would be dropped with the remainder.
    reasoning_prefix: str | None = None
    if _pl_forced_tool_grammar is not None:
        # Forced tool call: zaya_xml -> native XML (wrapper parser); else JSON {"name","arguments"}.
        if '"__ebnf__"' in _pl_forced_tool_grammar:
            _c, _tc = _parse_tool_calls(full_content, uid)
            if _tc:
                tool_calls, remainder = _tc, (_c or "")
        else:
            # The forced grammar emits a BARE JSON object, so `_parse_json_tool_call` needs the whole
            # string to be that object — but on a thinking model the grammar only takes effect after the
            # `<think>` scratch, so the raw text is `…reasoning…</think>{"name":…}` and `json.loads`
            # fails, demoting a forced call to plain content with finish_reason="stop". Raw first (the
            # d71cde22 rationale: a WRAPPED call can sit inside an unclosed think block), then retry on
            # the reasoning-stripped body for this bare form.
            tc = _parse_json_tool_call(full_content, uid)
            if tc is None:
                _p = _reasoning_parser()
                if _p is not None:
                    _rc, _body = _p.parse(full_content, thinking_open=_thinking_open(req))
                    tc = _parse_json_tool_call(_body, uid)
                    if tc is not None:
                        reasoning_prefix = _rc
            if tc is not None:
                tool_calls, remainder = [tc], ""
    elif req.tools:
        # Parsed even when finish_reason == "length": skipping truncated output guaranteed that any
        # markup already emitted leaked into `content` as prose. `_parse_tool_calls` recovers complete
        # blocks and logs the genuinely-truncated ones, so attempting it is strictly better than not.
        _c, _tc = _parse_tool_calls(full_content, uid)
        if _tc:
            tool_calls, remainder = _tc, (_c or "")

    # Reasoning: split the model's scratch reasoning out of the (tool-stripped) remainder into a
    # separate reasoning_content field, on THIS checkpoint's delimiters (derived from its chat
    # template, e.g. `<think>…</think>` for Qwen3/GLM, `<|channel>thought…<channel|>` for Gemma-4).
    # `thinking_open` is derived from the rendered prompt and decides only where an output with NO
    # closing delimiter goes; the split itself is always attempted, so a completion that plainly
    # contains the closing delimiter is split whatever the request asked for.
    reasoning_content: str | None = None
    body = remainder
    parser = _reasoning_parser()
    if parser is not None:
        reasoning_content, body = parser.parse(remainder, thinking_open=_thinking_open(req))
    if reasoning_prefix is not None:
        reasoning_content = reasoning_prefix if not reasoning_content else reasoning_prefix + reasoning_content

    message: dict = {"role": "assistant", "content": (body or None) if tool_calls else body}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
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


@app.post("/v1/completions")
async def v1_text_completions(req: OpenAICompletionRequest, request: Request):
    """OpenAI TEXT-completions: continue a RAW prompt, verbatim, with no chat template.

    THIS ROUTE DID NOT EXIST. The chat handler above was named `v1_completions` while registered only
    at `/v1/chat/completions`, so every `client.completions.create(...)` — and every `POST
    /v1/completions` — got a bare FastAPI 404. That reads to a caller as a broken server rather than
    an unimplemented endpoint, and the misleading handler name is what kept it unnoticed.

    Re-pointing the chat handler at this path would have been the wrong fix, and NOT for the reason
    it first looks like. The chat handler already accepts a raw `prompt` (the tokenizer applies the
    chat template only to a messages LIST — tokenize.py `isinstance(msg.text, list)`), so an alias
    would not have silently templated the prompt. The damage is the RESPONSE: that lane always
    answers `{"object":"chat.completion","choices":[{"message":{…}}]}`, which has no `text` key at
    all. A text-completions client reads `choices[0].text`, finds nothing, and either raises a
    validation error or renders an empty completion — a 200 that lost the answer. Two protocols,
    two response objects, two handlers; the request model is shared because the REQUEST fields
    genuinely overlap.

    Deliberately NOT inherited from the chat lane:

    * The reasoning split. `/v1/completions` has no `reasoning_content` field to split INTO, so
      routing the model's scratch out of `text` would delete it with nowhere to put it — the exact
      silent total-output-loss that 450f8ab9 fixed, in the other direction. The contract here is the
      raw continuation, delimiters and all, so the caller can parse it however it likes. Note the
      derivation itself still behaves on this lane: `_prompt_thinking_state` has a raw branch that
      reads `parser.prompt_state(req.prompt)` directly (the prompt IS the rendered prompt here), so
      `thinking_open` is honest — we simply do not act on it by splitting.
    * The reasoning-budget backstop, EXCEPT when the raw prompt is genuinely mid-span. The backstop
      FORCE-EMITS the close delimiter once the budget is spent, and the scheduler caps that budget at
      3/4 of max_tokens (scheduler `_maybe_arm_think_gate`) — so on the chat lane's `_thinking_active`
      test, which is True merely because the model COULD open a span, a plain 40-token raw completion
      would have `<channel|>` injected at token 30 by a server the caller never asked to edit its
      output. On the chat lane that injection is invisible (the reasoning split eats it); here it
      would land in `text`. So arm it only on `_thinking_open`, i.e. the prompt itself left a span
      open, where force-closing is a legitimate continuation of what the prompt started.
    * Tool-call parsing (see `_reject_unsupported_text_completion`: tools are a 400, not a no-op).

    Everything else IS shared, deliberately: `_resolve_sampling` (so an unset field inherits the
    checkpoint's generation_config.json exactly as the chat lane does — and note that inheritance is
    why `temperature: 0` alone is NOT greedy on a checkpoint shipping top_p 0.95, since `is_greedy`
    in core.py is `(temperature <= 0 or top_k == 1) and top_p == 1.0`), the stop/penalty/seed/
    ignore_eos plumbing, `response_format` grammars (sampler-level, template-independent), the
    transparent-CAM hooks, and the engine-refusal 400.
    """
    _bad = _reject_unsupported_text_completion(req)
    if _bad is not None:
        return _bad
    state = get_global_state()
    prompt: str = req.prompt  # type: ignore[assignment]  # guaranteed a non-empty str by the reject above

    # TRANSPARENT CAM, same as /generate and the chat lane's raw-prompt branch: learn from the prompt,
    # then fold relevant known facts back in. Both halves are off unless enabled, and both take a
    # per-request override so a caller that needs a literally-untouched prompt can say so.
    _cam_ns = request.headers.get("x-cam-namespace")
    if os.environ.get("MINISGL_CAM_WRITE_SYNC") == "1":
        await _cam_auto_write(prompt, override=req.cam_write, ns=_cam_ns)
    else:
        _schedule_cam_auto_write(prompt, override=req.cam_write, ns=_cam_ns)
    prompt = await _cam_auto_augment(prompt, ns=_cam_ns, override=req.cam_read)

    # Arm the reasoning backstop ONLY for a prompt that is itself mid-span — see the docstring.
    _think_delim = _reasoning_close_delim(req) if _thinking_open(req) else None
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                presence_penalty=req.presence_penalty,
                frequency_penalty=req.frequency_penalty,
                max_tokens=req.max_tokens,
                seed=req.seed,
                **dict(zip(("temperature", "top_p", "top_k"),
                           _resolve_sampling(req, state.config.model_path))),
                stop=_norm_stop(req.stop),
                grammar=_grammar_from_response_format(req.response_format),
                think_close_delim=_think_delim,
                think_answer_delim=_reasoning_answer_delim(req) if _think_delim else None,
                think_close_prefix=(_reasoning_close_wildcard(req)[0] if _think_delim else None),
                think_close_suffix=(_reasoning_close_wildcard(req)[1] if _think_delim else None),
                think_budget=_resolve_think_budget(req) if _think_delim else None,
            ),
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(
                state.stream_text_completions(
                    uid, req.model, bool((req.stream_options or {}).get("include_usage"))),
                request, uid,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: accumulate into a list and "".join once — `+=` in the loop is O(n^2) in the
    # output length (same reason as the chat lane).
    chunks: List[str] = []
    prompt_tokens = completion_tokens = 0
    finish_reason = "stop"
    rejected: str | None = None
    async for ack in state.wait_for_ack(uid):
        if getattr(ack, "error", None):
            rejected = ack.error
            break
        chunks.append(ack.incremental_output)
        completion_tokens = max(completion_tokens, ack.completion_tokens)
        prompt_tokens = ack.prompt_tokens or prompt_tokens
        if ack.finish_reason:
            finish_reason = ack.finish_reason
        if ack.finished:
            break
    if rejected is not None:
        # The engine refused (e.g. prompt longer than the KV pool). A real 4xx: an empty 200 would
        # look like the model chose to emit nothing, and returning nothing at all hangs the caller
        # until its own timeout.
        return JSONResponse(status_code=400, content={"error": {
            "message": rejected, "type": "invalid_request_error", "param": "prompt",
            "code": "context_length_exceeded"}})

    return {
        "id": f"cmpl-{uid}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "text": "".join(chunks),
                "logprobs": None,
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
    ctx = state.max_seq_len
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path,
                                     max_model_len=ctx, context_length=ctx)])


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
                presence_penalty=req.presence_penalty,
                frequency_penalty=req.frequency_penalty,
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
