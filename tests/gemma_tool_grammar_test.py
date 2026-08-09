"""Regression test: Gemma-4's forced tool call is constrained to its NATIVE format, not JSON.

`_forced_tool_grammar` had two shapes — ZAYA's native XML (behind `MINISGL_TOOL_FORMAT=zaya_xml`) and
JSON for everyone else. Gemma-4 fell into "everyone else", so `tool_choice: required` forced

    {"name": "terminal", "arguments": {"command": "ls"}}

out of a checkpoint whose template renders

    <|tool_call>call:terminal{command:<|"|>ls<|"|>}<tool_call|>

— a shape it was never trained to emit, and one its own chat template cannot render back into a
prompt on the next turn. The selection was also an env var defaulting to "json", so the fix could
only ever reach a serve whose operator already knew; it is now DERIVED from the template, which is
what turns a `tool_calls` message into bytes and therefore knows the answer.

Three halves:

* PURE — grammar text and the derivation cascade, no model files, no xgrammar. Always runs.
* COMPILE — the EBNF actually compiles under the installed xgrammar. Skipped if xgrammar is absent.
* ROUNDTRIP — a call written in the grammar's own language parses back through the SAME parser the
  serve uses (`_parse_tool_calls`). This is the half that would have caught a grammar that compiles
  but emits something the server then fails to read.

Run:  PYTHONPATH=python python3 tests/gemma_tool_grammar_test.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from minisgl.server import api_server as A  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


TOOLS = [
    {"type": "function", "function": {
        "name": "terminal", "description": "run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                        "limit": {"type": "integer"}}}}},
]

print("PURE 1: the grammar is Gemma's native shape, read off the template")
g = A._gemma_native_grammar(TOOLS)
check("wraps in the native asymmetric pair",
      'root ::= "<|tool_call>call:" fname "{" args "}<tool_call|>"' in g, repr(g.split("\n")[0]))
check("function names are constrained to the allowed tools",
      'fname ::= "terminal" | "read_file"' in g, repr([l for l in g.split("\n") if l.startswith("fname")]))
check("keys are the tools' declared properties, bare (escape_keys=False)",
      'key ::= "command" | "limit" | "path"' in g, repr([l for l in g.split("\n") if l.startswith("key ")]))
check("strings use the <|\"|> special token, not a quote",
      'str ::= "<|\\"|>" schar* "<|\\"|>"' in g, repr([l for l in g.split("\n") if l.startswith("str ")]))
check("no JSON object shape anywhere", '"name"' not in g and '"arguments"' not in g)
check("forced_name narrows to one tool",
      'fname ::= "read_file"' in A._gemma_native_grammar(TOOLS, "read_file"))
check("no matching tool -> no grammar", A._gemma_native_grammar(TOOLS, "nope") is None)

print("PURE 2: a `<` inside a string survives — only the literal `<|` cannot")
# A flat `[^<]*` (the ZAYA precedent) would have made `ls <file` or `a < b` ungrammatical, and the one
# tool most likely to need it is the shell.
check("schar admits '<' unless followed by '|'", 'schar ::= [^<] | "<" [^|]' in g,
      repr([l for l in g.split("\n") if l.startswith("schar")]))

print("PURE 3: format selection is DERIVED, and an explicit env still wins")
check("default is auto (was 'json', which silently excluded Gemma)",
      os.environ.get("MINISGL_TOOL_FORMAT") is not None or A._TOOL_FORMAT == "auto", A._TOOL_FORMAT)
_saved = A._TOOL_FORMAT
try:
    A._TOOL_FORMAT = "json"
    check("explicit json wins over derivation", A._resolve_tool_format() == "json")
    A._TOOL_FORMAT = "zaya_xml"
    check("explicit zaya_xml wins over derivation", A._resolve_tool_format() == "zaya_xml")
finally:
    A._TOOL_FORMAT = _saved

print("\nARTIFACT: derive the format from the real Gemma-4 template")
MODEL = "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
except Exception as e:  # noqa: BLE001
    print(f"  SKIP  checkpoint unavailable ({type(e).__name__})")
    tok = None

if tok is not None:
    A._FRONTEND_TOKENIZER, A._FRONTEND_TOKENIZER_SET = tok, True
    A._DERIVED_TOOL_FORMAT, A._DERIVED_TOOL_FORMAT_SET = None, False
    _saved = A._TOOL_FORMAT
    A._TOOL_FORMAT = "auto"
    try:
        check("template renders the native wrapper -> gemma_native",
              A._resolve_tool_format() == "gemma_native", repr(A._resolve_tool_format()))
        # And the forced path now hands the engine an EBNF rather than a JSON schema.
        req = A.OpenAICompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}],
            tools=TOOLS, tool_choice="required")
        spec = A._grammar_from_tools(req)
        check("forced tool_choice yields an __ebnf__ spec", spec is not None and '"__ebnf__"' in spec,
              repr(spec)[:120])
        check("...and the response lane routes __ebnf__ to the wrapper parser",
              spec is not None and '"__ebnf__"' in spec)
        # auto mode must stay unconstrained-but-tagged, NOT switched to the EBNF (see the
        # _TOOL_STRUCT_WRAPPERS comment: a hard grammar would forbid answering in prose).
        req_auto = A.OpenAICompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], tools=TOOLS)
        check("auto mode is not forced into the EBNF", A._grammar_from_tools(req_auto) is None)
    finally:
        A._TOOL_FORMAT = _saved

print("\nCOMPILE: the EBNF is accepted by the installed xgrammar")
try:
    import xgrammar as xgr
except Exception as e:  # noqa: BLE001
    print(f"  SKIP  xgrammar unavailable ({type(e).__name__})")
    xgr = None
if xgr is not None:
    try:
        xgr.Grammar.from_ebnf(A._gemma_native_grammar(TOOLS))
        check("grammar compiles", True)
    except Exception as e:  # noqa: BLE001
        check("grammar compiles", False, f"{type(e).__name__}: {e}")

print("\nROUNDTRIP: a call in the grammar's language parses through the serve's own parser")
# Exactly what the template renders, byte for byte (see chat_template.jinja: format_argument).
SAMPLES = [
    ('<|tool_call>call:terminal{command:<|"|>ls -R<|"|>}<tool_call|>',
     "terminal", {"command": "ls -R"}),
    ('<|tool_call>call:read_file{limit:20,path:<|"|>a/b.py<|"|>}<tool_call|>',
     "read_file", {"limit": 20, "path": "a/b.py"}),
    # the `<` that a flat [^<]* would have forbidden
    ('<|tool_call>call:terminal{command:<|"|>sort < in.txt<|"|>}<tool_call|>',
     "terminal", {"command": "sort < in.txt"}),
    ('<|tool_call>call:terminal{command:<|"|>x<|"|>}<tool_call|>',
     "terminal", {"command": "x"}),
]
for raw, want_name, want_args in SAMPLES:
    content, calls = A._parse_tool_calls(raw, 1)
    ok = (len(calls) == 1
          and calls[0]["function"]["name"] == want_name
          and json.loads(calls[0]["function"]["arguments"]) == want_args
          and not (content or "").strip())
    got = (calls[0]["function"] if calls else None, content)
    check(f"{want_name}({', '.join(want_args)})", ok, repr(got)[:160])

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
    sys.exit(1)
print("all checks passed")
