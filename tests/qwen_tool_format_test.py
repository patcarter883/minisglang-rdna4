"""Qwen3.6/3.8 native tool-call format must be DERIVED, and the auto tag must not force JSON into it.

MEASURED DEFECT (2026-09-20). `_derive_tool_format` knew Gemma-4's, Muse-Glimmer's and ZAYA's
wrappers but NOT Qwen's, so every Qwen3.6/3.8 checkpoint — including Qwen3.8-Flash-Next — fell
through to the "json" default. Both constrained paths then forced a JSON call on a model whose own
template renders XML and whose system prompt, emitted by that same template, says:

    "If you choose to call a function ONLY reply in the following format with NO suffix:
     <tool_call>\\n<function=example_function_name>\\n<parameter=…>"
    "Function calls MUST follow the specified format"

Measured against xgrammar on the live Flash-Next tokenizer: the auto structural tag ACCEPTED
`<tool_call>{"name":…,"arguments":{…}}` and REJECTED the model's own instructed form at the FIRST
token after the wrapper (the newline). Sampled unconstrained through /v1/completions, the model
emits exactly that rejected form. So a tools request masked the model off its trained format mid
call, at the ARGUMENT region — which is why a failing turn's reasoning and content stay coherent
while only the arguments degenerate (`{"code": "# placeholder"}`, seen in Hermes sessions
30935df66949 and cdb27addb762).

    docker exec: python -m pytest tests/qwen_tool_format_test.py -q -o addopts=""
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

import minisgl.server.api_server as api  # noqa: E402

# What the shipped Qwen3.6 / Qwen3.8 templates render for an assistant tool call.
QWEN_RENDERED = ("<|im_start|>assistant\n<tool_call>\n<function=f>\n"
                 "<parameter=k>\nv\n</parameter>\n</function>\n</tool_call><|im_end|>\n")

TOOLS = [
    {"type": "function", "function": {"name": "browser_exec", "parameters": {
        "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "web_search", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}}}},
]


class _StubTok:
    def __init__(self, rendered: str):
        self._rendered = rendered

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return self._rendered


class _Req:
    def __init__(self, tools, tool_choice=None):
        self.tools = tools
        self.tool_choice = tool_choice


@pytest.fixture(autouse=True)
def _reset_derivation_cache():
    """`_resolve_tool_format` memoises; each case must derive afresh."""
    api._DERIVED_TOOL_FORMAT = None
    api._DERIVED_TOOL_FORMAT_SET = False
    yield
    api._DERIVED_TOOL_FORMAT = None
    api._DERIVED_TOOL_FORMAT_SET = False


def _with_template(monkeypatch, rendered: str):
    monkeypatch.setattr(api, "_frontend_tokenizer", lambda: _StubTok(rendered))


def test_qwen_wrapper_derives_qwen_xml(monkeypatch):
    _with_template(monkeypatch, QWEN_RENDERED)
    assert api._derive_tool_format() == "qwen_xml"


def test_zaya_still_wins_its_own_wrapper(monkeypatch):
    """ZAYA's `<zyphra_tool_call>` also contains `<function=`. It must NOT be read as Qwen — that
    would swap its forced grammar's wrapper and constrain it to a tag it never emits."""
    _with_template(monkeypatch,
                   "<zyphra_tool_call>\n<function=f>\n</function>\n</zyphra_tool_call>")
    assert api._derive_tool_format() == "zaya_xml"


def test_other_families_and_the_json_default_are_untouched(monkeypatch):
    _with_template(monkeypatch, "<|tool_call>call:f{}")
    assert api._derive_tool_format() == "gemma_native"
    api._DERIVED_TOOL_FORMAT_SET = False
    _with_template(monkeypatch, '<atem:invoke name="f">')
    assert api._derive_tool_format() == "atem"
    api._DERIVED_TOOL_FORMAT_SET = False
    _with_template(monkeypatch, '{"name": "f", "arguments": {}}')
    assert api._derive_tool_format() is None


def test_the_auto_path_carries_no_grammar_at_all(monkeypatch):
    """THE DEFECT, and the scope of its fix.

    `auto` used to get an xgrammar structural tag (b3456b04) triggering on `<tool_call>` and forcing
    the wrapped body to a JSON schema. MEASURED on Qwen3.8-Flash-Next with the real Hermes
    browser_exec definition, identical probe either side:

        constrained    browser_exec  7/28 junk arguments (25.0%);  code bodies max 369 chars
        unconstrained  browser_exec  0/25 junk arguments ( 0.0%);  code bodies max 5426 chars

    The grammar layer is for STRUCTURED calls — `response_format` and a forced `tool_choice`. There
    is no auto-path builder left to call, for ANY family: of the 24 checkpoints on this box, ZERO are
    JSON-native inside `<tool_call>` (10 are Qwen XML, 3 are Laguna/GLM `<arg_key>/<arg_value>`), so
    the tag corrupted 13 and helped none. The nine it spared were spared by wrapper SPELLING, not by
    design, which is why this is removed rather than extended with another family arm."""
    assert not hasattr(api, "_structural_tag_from_tools")
    assert not hasattr(api, "_TOOL_STRUCT_WRAPPERS")


def test_auto_stays_unconstrained_even_for_a_json_bodied_checkpoint(monkeypatch):
    """No family enumeration survives: `auto` is unconstrained regardless of derived format."""
    _with_template(monkeypatch, '{"name": "f", "arguments": {}}')
    assert api._grammar_from_tools(_Req(TOOLS)) is None          # auto -> not forced -> no grammar


def test_forced_call_uses_the_native_xml_grammar(monkeypatch):
    _with_template(monkeypatch, QWEN_RENDERED)
    g = api._grammar_from_tools(_Req(TOOLS, tool_choice="required"))
    assert g is not None
    ebnf = json.loads(g)["__ebnf__"]
    assert ebnf.startswith('root ::= "<tool_call>\\n<function=" fname ">\\n" params '
                           '"</function>\\n</tool_call>"')
    assert '"browser_exec" | "web_search"' in ebnf


def test_generalised_grammar_reproduces_the_zaya_body_exactly():
    """One body, parameterised by the wrapper — the ZAYA output must be byte-identical to the
    dedicated function it replaced, or this refactor silently changed ZAYA's forced grammar."""
    expected = "\n".join([
        'root ::= "<zyphra_tool_call>\\n<function=" fname ">\\n" params '
        '"</function>\\n</zyphra_tool_call>"',
        'fname ::= "browser_exec" | "web_search"',
        "params ::= param*",
        'param ::= "<parameter=" pname ">\\n" pval "\\n</parameter>\\n"',
        'pname ::= "code" | "query"',
        "pval ::= [^<]*",
    ])
    got = api._wrapped_xml_grammar(
        TOOLS, None, "<zyphra_tool_call>", "</zyphra_tool_call>")
    assert got == expected
