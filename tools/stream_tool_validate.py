#!/usr/bin/env python3
"""Validate the streaming tool-call parser against a live minisgl serve (host-side, stdlib only).

Checks:
  1. A streaming tools request emits proper delta.tool_calls (index/id/name/arguments fragments),
     NO raw <function=/<tool_call> markup in any delta.content, finish_reason:tool_calls.
  2. Reassembled streamed tool_calls == the non-streaming parse of the same prompt.
  3. A streaming request WITHOUT tools is unchanged (content streams, no tool_calls, finish stop).
"""
import json, sys, urllib.request

BASE = "http://127.0.0.1:1919/v1/chat/completions"
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}
MSGS = [{"role": "user", "content": "What is the weather in Paris? Call get_weather."}]
# Disable thinking for the stream-vs-non-stream equality checks so both runs are deterministic
# (greedy). A separate check exercises thinking-on to prove the reasoning->tool_calls compose order.
NOTHINK = {"enable_thinking": False}


def post(payload):
    req = urllib.request.Request(
        BASE, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=600)


def run_stream(with_tools, thinking=False):
    payload = {"model": "m", "messages": MSGS, "stream": True, "max_tokens": 500, "temperature": 0.0,
               "chat_template_kwargs": {"enable_thinking": thinking}}
    if with_tools:
        payload["tools"] = [WEATHER_TOOL]
    resp = post(payload)
    content = ""
    reasoning = ""
    calls = {}  # index -> assembled call
    finish = None
    n_tool_deltas = 0
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        obj = json.loads(data)
        choice = obj["choices"][0]
        delta = choice.get("delta", {})
        if delta.get("content"):
            content += delta["content"]
        if delta.get("reasoning_content"):
            reasoning += delta["reasoning_content"]
        for td in delta.get("tool_calls", []):
            n_tool_deltas += 1
            i = td["index"]
            c = calls.setdefault(i, {"id": None, "type": None, "name": "", "arguments": ""})
            if td.get("id"):
                c["id"] = td["id"]
            if td.get("type"):
                c["type"] = td["type"]
            fn = td.get("function", {})
            if fn.get("name"):
                c["name"] += fn["name"]
            if fn.get("arguments"):
                c["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    ordered = [calls[i] for i in sorted(calls)]
    return {"content": content, "reasoning": reasoning, "calls": ordered,
            "finish": finish, "n_tool_deltas": n_tool_deltas}


def run_nonstream(with_tools, thinking=False):
    payload = {"model": "m", "messages": MSGS, "stream": False, "max_tokens": 500, "temperature": 0.0,
               "chat_template_kwargs": {"enable_thinking": thinking}}
    if with_tools:
        payload["tools"] = [WEATHER_TOOL]
    obj = json.loads(post(payload).read().decode())
    choice = obj["choices"][0]
    msg = choice["message"]
    calls = []
    for tc in msg.get("tool_calls") or []:
        calls.append({"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]})
    return {"content": msg.get("content") or "", "finish": choice["finish_reason"], "calls": calls}


def main():
    failures = []

    print("=== 1) STREAMING + tools ===")
    s = run_stream(with_tools=True)
    print("  finish_reason:", s["finish"])
    print("  reasoning chars:", len(s["reasoning"]))
    print("  content:", repr(s["content"][:200]))
    print("  tool_call deltas:", s["n_tool_deltas"])
    print("  reassembled calls:", json.dumps(s["calls"], indent=None))
    leaked = ("<function=" in s["content"]) or ("<tool_call>" in s["content"])
    if leaked:
        failures.append("MARKUP LEAKED into streaming delta.content")
    if not s["calls"]:
        failures.append("no tool_calls emitted in streaming mode")
    else:
        c0 = s["calls"][0]
        if not c0["id"] or c0["type"] != "function" or not c0["name"]:
            failures.append(f"first tool_call missing id/type/name: {c0}")
        try:
            json.loads(c0["arguments"])
        except Exception as e:
            failures.append(f"tool_call arguments not valid JSON: {c0['arguments']!r} ({e})")
    if s["finish"] != "tool_calls":
        failures.append(f"streaming finish_reason={s['finish']!r}, expected tool_calls")

    print("\n=== 2) NON-STREAMING + tools (same prompt) ===")
    ns = run_nonstream(with_tools=True)
    print("  finish_reason:", ns["finish"])
    print("  calls:", json.dumps(ns["calls"], indent=None))
    # compare reassembled streamed calls vs non-streamed (name + parsed args, order)
    def norm(calls):
        out = []
        for c in calls:
            try:
                args = json.loads(c["arguments"])
            except Exception:
                args = c["arguments"]
            out.append((c["name"], args))
        return out
    if norm(s["calls"]) != norm(ns["calls"]):
        failures.append(f"streamed calls != non-streamed calls:\n    stream={norm(s['calls'])}\n    nonstr={norm(ns['calls'])}")
    else:
        print("  MATCH: streamed tool_calls == non-streamed parse")

    print("\n=== 3) STREAMING without tools (regression) ===")
    p = run_stream(with_tools=False)
    print("  finish_reason:", p["finish"])
    print("  content:", repr(p["content"][:200]))
    print("  tool_call deltas:", p["n_tool_deltas"])
    if p["n_tool_deltas"] != 0:
        failures.append("plain streaming emitted tool_calls unexpectedly")
    if not p["content"].strip():
        failures.append("plain streaming produced no content")
    if p["finish"] not in ("stop", "length"):
        failures.append(f"plain streaming finish_reason={p['finish']!r}")

    print("\n=== 4) STREAMING + tools + THINKING (compose order) ===")
    t = run_stream(with_tools=True, thinking=True)
    print("  finish_reason:", t["finish"])
    print("  reasoning chars:", len(t["reasoning"]))
    print("  content:", repr(t["content"][:120]))
    print("  reassembled calls:", json.dumps(t["calls"]))
    leaked_t = ("<function=" in t["content"]) or ("<tool_call>" in t["content"]) \
        or ("</think>" in t["content"]) or ("<function=" in t["reasoning"])
    if leaked_t:
        failures.append("thinking+tools: markup leaked into content/reasoning")
    if not t["reasoning"].strip():
        failures.append("thinking+tools: no reasoning_content streamed")
    if not t["calls"] or t["finish"] != "tool_calls":
        failures.append(f"thinking+tools: expected reasoning then tool_calls, got finish={t['finish']} calls={t['calls']}")

    print("\n" + ("=" * 50))
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
