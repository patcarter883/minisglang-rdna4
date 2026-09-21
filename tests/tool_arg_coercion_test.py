"""A declared-string tool argument must reach the client as that exact string.

THE DEFECT. `_coerce` ran `json.loads` over every raw XML/KV parameter value and kept whatever came
back, guessing the type from the CONTENT. The XML and arg_key/arg_value formats have no quoting
convention — `<parameter=code>` simply contains the code — so any string that happened to look like
JSON was silently re-typed before it ever reached the client:

    {"command": "true"}  -> True      a shell command became a boolean
    {"code": "123"}      -> 123       a one-line program became an int
    {"code": "[1, 2]"}   -> [1, 2]    a list literal became an actual list

Coercion is still needed for the non-string params these formats carry (`limit: 6`, `timeout_s`,
`local: false`), so the fix is to consult the schema the caller already sent, not to stop coercing.
This path became load-bearing when the auto-path JSON grammar was removed: the model now emits its
native XML and this parser is what reads it back.

    docker exec: python -m pytest tests/tool_arg_coercion_test.py -q -o addopts=""
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest  # noqa: E402

import minisgl.server.api_server as api  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "terminal", "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "execute_code", "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}, "local": {"type": "boolean"}}}}},
]
PT = api._tool_param_types(TOOLS)


def _xml(fn, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<function={fn}>\n{body}</function>"


@pytest.mark.parametrize("value", ["true", "false", "null", "123", "1.5", "[1, 2]", '{"a": 1}',
                                   '"quoted"', "-0"])
def test_a_declared_string_survives_verbatim(value):
    """Every one of these used to come back as a non-string."""
    name, args = api._parse_one_tool_call(_xml("terminal", command=value), PT)
    assert name == "terminal"
    assert args["command"] == value, f"{value!r} was re-typed to {args['command']!r}"


def test_real_code_is_untouched():
    code = 'from hermes_tools import terminal\nr = terminal("echo \'hi\'")\nprint(r)'
    _, args = api._parse_one_tool_call(_xml("execute_code", code=code), PT)
    assert args["code"] == code


def test_declared_non_strings_STILL_coerce():
    """The reason coercion exists at all — these formats carry no types of their own."""
    _, args = api._parse_one_tool_call(_xml("terminal", command="ls", timeout_s="120"), PT)
    assert args["timeout_s"] == 120 and isinstance(args["timeout_s"], int)
    _, args = api._parse_one_tool_call(_xml("execute_code", code="x=1", local="false"), PT)
    assert args["local"] is False


def test_with_NO_schema_the_fallback_is_conservative():
    """An unknown tool or undeclared param: keep text that might be text. Objects, arrays and the
    three bare literals stay unambiguous in a format whose strings are unquoted."""
    assert api._coerce("123") == "123"          # was 123
    assert api._coerce('"quoted"') == '"quoted"'  # was 'quoted' - the quotes were eaten
    assert api._coerce("true") is True
    assert api._coerce("[1, 2]") == [1, 2]
    assert api._coerce("not json") == "not json"


def test_the_inline_pycall_format_uses_its_OWN_quoting():
    """ZAYA's `<function=f(a='true', b=1)>` quotes strings, so the source says the type."""
    args = api._parse_pycall_args("a='true', b=1, c=[1,2]")
    assert args["a"] == "true" and isinstance(args["a"], str)
    assert args["b"] == 1 and args["c"] == [1, 2]
