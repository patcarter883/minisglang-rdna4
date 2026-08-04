"""Prefix-sharing traffic for measuring what the recurrent-radix snapshot store actually buys.

The store costs VRAM out of the same budget the KV pool is sized from (0.375 GiB by default, 16.4 MiB
a snapshot on the 35B), and what it buys is prefill it does not have to redo. Whether that trade is
worth it is a property of the TRAFFIC, so this drives the two shapes that actually share prefixes on
this box and reports the only number that matters to a user: time to first token.

  * AGENT   — one long shared system block (tool schemas + instructions), then N different questions.
              Every request after the first re-sends the identical prefix. This is what a coding or
              tool-using agent does on every single turn.
  * CHAT    — a multi-turn conversation, each turn re-sending the whole history. Pure prefix
              EXTENSION: turn k's prompt is turn k-1's prompt plus two messages.

Requests are issued SEQUENTIALLY on purpose: concurrent ones race to insert the same radix node and
would measure the scheduler, not the cache.

    python3 tools/rec_radix_traffic.py --base http://127.0.0.1:1919 --out leg.json

TTFT is measured from the STREAM (first delta chunk of any kind — a reasoning model emits
`reasoning_content` before `content`, so filtering on "content" never fires), which is the metric a
prefix cache moves; decode speed is unaffected by it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request

# ~2.6k tokens of tool-schema-shaped system block: the realistic shared prefix on this box.
_TOOL = """You have access to the following tools. Use them when they help.

- name: read_file
  description: Read a file from the repository and return its contents with line numbers.
  parameters: {path: string (absolute), offset: integer (optional), limit: integer (optional)}
- name: write_file
  description: Write content to a file, creating parent directories as needed.
  parameters: {path: string (absolute), content: string, mode: string (optional)}
- name: run_command
  description: Execute a shell command in the workspace and capture stdout, stderr and exit code.
  parameters: {command: string, timeout_ms: integer (optional), cwd: string (optional)}
- name: search_code
  description: Search the repository for a regular expression and return matching lines with context.
  parameters: {pattern: string, path: string (optional), max_results: integer (optional)}
- name: list_directory
  description: List the entries of a directory, one per line, with type and size.
  parameters: {path: string, recursive: boolean (optional)}
"""


def system_block(repeat: int) -> str:
    return (
        "You are a careful engineering assistant working in a large HIP/PyTorch repository.\n"
        + _TOOL * repeat
        + "\nAnswer concisely. Prefer concrete file paths and measured numbers over description.\n"
    )


def stream_ttft(base: str, messages: list[dict], max_tokens: int, timeout: int) -> tuple[float, float, int]:
    """Returns (ttft_s, total_s, completion_tokens). TTFT = first delta of ANY kind."""
    body = {
        "model": "x", "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.0, "top_p": 1.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                ntok = d["usage"].get("completion_tokens", ntok)
            for ch in d.get("choices", []):
                delta = ch.get("delta") or {}
                if ttft is None and any(delta.get(k) for k in ("content", "reasoning_content")):
                    ttft = time.time() - t0
    return (ttft if ttft is not None else time.time() - t0), time.time() - t0, ntok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:1919")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--agent-reqs", type=int, default=8)
    ap.add_argument("--chat-turns", type=int, default=6)
    ap.add_argument("--prefix-repeat", type=int, default=6, help="copies of the tool block (~430 tok each)")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    sysmsg = {"role": "system", "content": system_block(a.prefix_repeat)}
    rows = []

    print(f"[{a.label}] AGENT: {a.agent_reqs} requests sharing one system prefix")
    questions = [
        "Which tool would you use to find every call site of a function?",
        "Name the parameter that bounds run_command's runtime.",
        "What does list_directory return for each entry?",
        "Which tool creates parent directories?",
        "How would you read only lines 100-200 of a file?",
        "Which tool reports an exit code?",
        "What is the optional parameter of search_code that limits output?",
        "Which two tools take an absolute path?",
    ]
    for i in range(a.agent_reqs):
        q = questions[i % len(questions)]
        ttft, tot, n = stream_ttft(a.base, [sysmsg, {"role": "user", "content": q}], a.max_tokens, a.timeout)
        rows.append({"phase": "agent", "i": i, "ttft": round(ttft, 3), "total": round(tot, 3), "tok": n})
        print(f"   req {i}: TTFT {ttft:.3f}s  total {tot:.3f}s")

    print(f"[{a.label}] CHAT: {a.chat_turns} turns, each re-sending the history")
    convo = [sysmsg]
    for t in range(a.chat_turns):
        convo = convo + [{"role": "user", "content": f"Step {t}: name one more thing to check, briefly."}]
        ttft, tot, n = stream_ttft(a.base, convo, a.max_tokens, a.timeout)
        rows.append({"phase": "chat", "i": t, "ttft": round(ttft, 3), "total": round(tot, 3), "tok": n})
        print(f"   turn {t}: TTFT {ttft:.3f}s  total {tot:.3f}s")
        convo = convo + [{"role": "assistant", "content": "noted."}]

    def summarize(phase, skip_first):
        v = [r["ttft"] for r in rows if r["phase"] == phase][1 if skip_first else 0:]
        return {"n": len(v), "mean": round(statistics.mean(v), 3),
                "median": round(statistics.median(v), 3), "min": round(min(v), 3), "max": round(max(v), 3)}

    # The FIRST agent request populates the cache; it is a miss by construction in every leg, so the
    # comparable number is the mean over the rest.
    summary = {"label": a.label, "agent_warm": summarize("agent", True),
               "agent_all": summarize("agent", False), "chat": summarize("chat", False)}
    print(f"\n[{a.label}] agent TTFT (excl. first): {summary['agent_warm']}")
    print(f"[{a.label}] chat  TTFT: {summary['chat']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
