#!/usr/bin/env python3
"""Gemma-4 native tool-call parsing — the format minisgl silently dropped until 2026-08-07.

Gemma-4 emits tool calls in its OWN wrapper, which is pipe-inside and ASYMMETRIC:

    <|tool_call>call:NAME{key:value,...}<tool_call|>

None of the pre-existing wrappers match it (`<tool_call>` is not a substring of `<|tool_call>`), and
the body is not JSON either: strings are delimited by the `<|"|>` TOKEN and keys are BARE, because
the checkpoint's chat_template.jinja propagates escape_keys=False from the top level. The result was
that a perfectly well-formed tool call from the model was returned to the client as prose, with
tool_calls=[] — observed live in Hermes session facc711200c0, where the model correctly asked to
fetch a URL and the caller saw only markup.

Run: PYTHONPATH=python python3 tools/test_gemma4_tool_parser.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from minisgl.server.api_server import _parse_tool_calls  # noqa: E402

Q = '<|"|>'  # the string-delimiter special token


def case(name, text, expect_calls, expect_content=...):
    content, tcs = _parse_tool_calls(text, 7)
    got = [(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in tcs]
    ok = got == expect_calls and (expect_content is ... or content == expect_content)
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"         got calls   = {got}")
        print(f"         want calls  = {expect_calls}")
        if expect_content is not ...:
            print(f"         got content = {content!r}")
            print(f"         want content= {expect_content!r}")
    return ok


def main():
    results = []

    # The exact string from Hermes session facc711200c0.
    results.append(case(
        "REAL session facc711200c0",
        "<|tool_call>call:web_extract{urls:[" + Q + "https://a.aliexpress.com/_mPWq9G7" + Q + "]}<tool_call|>",
        [("web_extract", {"urls": ["https://a.aliexpress.com/_mPWq9G7"]})],
        expect_content=None,          # nothing but the call -> content must be None, not markup
    ))

    results.append(case(
        "prose before the call is preserved",
        "Let me look that up.<|tool_call>call:search{query:" + Q + "pool dosing" + Q + ",top_k:5}<tool_call|>",
        [("search", {"query": "pool dosing", "top_k": 5})],
        expect_content="Let me look that up.",
    ))

    results.append(case(
        "bool / number / float / nested object",
        "<|tool_call>call:cfg{on:true,n:3,ratio:1.5,opts:{deep:false,tag:" + Q + "x" + Q + "}}<tool_call|>",
        [("cfg", {"on": True, "n": 3, "ratio": 1.5, "opts": {"deep": False, "tag": "x"}})],
    ))

    # A string whose CONTENT contains the JSON metacharacters. This is why the parser rebuilds strings
    # with json.dumps instead of swapping <|"|> for a quote character: a raw `"` or `}` inside the
    # value would otherwise terminate the object early or produce invalid JSON.
    results.append(case(
        "string containing braces, comma, colon and a double-quote",
        "<|tool_call>call:note{body:" + Q + 'a{b},c:d " e' + Q + "}<tool_call|>",
        [("note", {"body": 'a{b},c:d " e'})],
    ))

    results.append(case("no arguments", "<|tool_call>call:ping{}<tool_call|>", [("ping", {})]))

    results.append(case(
        "two calls in one message",
        "<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:" + Q + "z" + Q + "}<tool_call|>",
        [("a", {"x": 1}), ("b", {"y": "z"})],
    ))

    # Truncated mid-call (hit the token budget before the closer): recovered by the unclosed-wrapper
    # path rather than leaking raw markup into content.
    results.append(case(
        "truncated: opener with no closer still recovers",
        "<|tool_call>call:web_extract{urls:[" + Q + "https://x" + Q + "]}",
        [("web_extract", {"urls": ["https://x"]})],
    ))

    # A bareword value that is NOT true/false/null must not be silently accepted as a key or a
    # literal — the call is malformed and should be REFUSED, not turned into a wrong call.
    results.append(case(
        "malformed body is refused, not guessed",
        "<|tool_call>call:bad{x:someBareword}<tool_call|>",
        [],
    ))

    # --- regressions: the other families must be untouched -------------------------------------
    results.append(case(
        "regression: Hermes/Qwen3 JSON wrapper",
        '<tool_call>{"name": "foo", "arguments": {"a": 1}}</tool_call>',
        [("foo", {"a": 1})],
    ))
    results.append(case(
        "regression: Qwen3 XML",
        "<tool_call><function=foo><parameter=a>1</parameter></function></tool_call>",
        [("foo", {"a": 1})],
    ))
    results.append(case(
        "regression: plain prose is not mangled",
        "Just a normal answer with no tools.",
        [],
        expect_content="Just a normal answer with no tools.",
    ))

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
