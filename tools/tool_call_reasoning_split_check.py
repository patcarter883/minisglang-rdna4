"""CPU proof for the non-streaming tool-call / reasoning-split ordering fix.

Bug (spine acceptance battery): a reasoning model (Laguna/poolside) emits its <tool_call> block
WITHOUT closing </think>. The old non-stream path reasoning-split FIRST, trapping the whole block
(markup + args) in reasoning_content, so tool_calls came back null and the caller saw raw
`<arg_value>…</arg_value></tool_call>` leak into the argument payload.

Fix: extract tool calls from the RAW text FIRST, then reasoning-split only the non-tool remainder
(mirrors the streaming ToolCallStreamState ordering). This test replicates that two-step sequence
and asserts clean extraction across the reproduced shapes.

Run inside the serve container:
  PYTHONPATH=/engine/python python3 tools/tool_call_reasoning_split_check.py
"""
from __future__ import annotations

import json

from minisgl.server.api_server import _parse_tool_calls
from minisgl.server.reasoning import get_reasoning_parser

PARSER = get_reasoning_parser("poolside_v1")  # <think> … </think>


def assemble(raw: str, has_tools: bool = True, thinking: bool = True):
    """Replicate the FIXED non-stream assembly: tools-from-raw, then reasoning-split the remainder."""
    tool_calls = None
    remainder = raw
    if has_tools:
        _c, _tc = _parse_tool_calls(raw, 1)
        if _tc:
            tool_calls, remainder = _tc, (_c or "")
    reasoning = None
    body = remainder
    if thinking and PARSER is not None:
        reasoning, body = PARSER.parse(remainder, thinking_open=True)
    content = (body or None) if tool_calls else body
    return {"reasoning_content": reasoning, "content": content, "tool_calls": tool_calls}


def check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'}  {name}")
    assert cond, name


# --- Case A: the exact reproduced Laguna output — tool call, NO </think> --------------------------
RAW_A = ('I\'ll create the task with the specified title and execution requirements.'
         '<tool_call>create_task'
         '<arg_key>title</arg_key><arg_value>Ship release</arg_value>'
         '<arg_key>execution_requirements</arg_key><arg_value>{"cpu": 4, "gpu": 1}</arg_value>'
         '</tool_call>')
print("Case A: tool call, no </think> (the reproduced bug)")
a = assemble(RAW_A)
check("tool_calls extracted (not null)", a["tool_calls"] is not None and len(a["tool_calls"]) == 1)
tc = a["tool_calls"][0]
fn = tc["function"]
args = json.loads(fn["arguments"])
check("function name = create_task", fn["name"] == "create_task")
check("title arg clean", args.get("title") == "Ship release")
check("object arg present", "execution_requirements" in args)
# the money assertion: NO markup leaked into ANY argument value
raw_args = fn["arguments"]
check("no </arg_value> leak in args", "</arg_value>" not in raw_args)
check("no </tool_call> leak in args", "</tool_call>" not in raw_args)
check("reasoning captured the prefix", (a["reasoning_content"] or "").startswith("I'll create the task"))
check("content is None (only a call)", a["content"] is None)

# --- Case B: reasoning CLOSED with </think>, then answer text + tool call --------------------------
RAW_B = ('let me think about it</think>Here is the call.'
         '<tool_call>create_task<arg_key>title</arg_key><arg_value>X</arg_value></tool_call>')
print("Case B: closed </think> + answer + tool call")
b = assemble(RAW_B)
check("tool_calls extracted", b["tool_calls"] is not None and len(b["tool_calls"]) == 1)
check("reasoning = pre-</think>", b["reasoning_content"] == "let me think about it")
check("content = answer text (call stripped)", (b["content"] or "").strip() == "Here is the call.")
check("no markup leak", "</arg_value>" not in b["tool_calls"][0]["function"]["arguments"])

# --- Case C: plain reasoning, NO tools ------------------------------------------------------------
RAW_C = "thinking...</think>The answer is 42."
print("Case C: plain reasoning, no tools")
c = assemble(RAW_C, has_tools=False)
check("no tool_calls", c["tool_calls"] is None)
check("reasoning split", c["reasoning_content"] == "thinking...")
check("content = answer", c["content"] == "The answer is 42.")

# --- Case D: tool call only, no reasoning prefix, no </think> -------------------------------------
RAW_D = '<tool_call>ping<arg_key>host</arg_key><arg_value>1.2.3.4</arg_value></tool_call>'
print("Case D: bare tool call, no reasoning")
d = assemble(RAW_D)
check("tool_calls extracted", d["tool_calls"] is not None)
check("host arg clean", json.loads(d["tool_calls"][0]["function"]["arguments"])["host"] == "1.2.3.4")
check("content None", d["content"] is None)

print("\nALL CASES PASS — tool calls extracted from raw before reasoning split; no markup leak into args.")
