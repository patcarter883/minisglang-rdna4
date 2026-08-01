"""Drive one bs=1 spec-decode request at the serve, for tools/propose_rocprof.sh.

Separate file rather than a heredoc: the caller is itself inside a heredoc, and a nested one
terminates the outer at the first matching delimiter.

The request is EXPECTED to fail. MINISGL_EXIT_AFTER_STEPS cuts the engine off mid-generation so the
profiler gets a normal interpreter exit; the connection dies with it. The trace is the artifact, not
the completion.
"""
import json
import urllib.request

BASE = "http://localhost:1919"
PROMPT = (
    "Write a Python function that reverses a singly linked list in place, then explain how it "
    "works, why it is O(n) time and O(1) space, and what happens on an empty list and on a "
    "single-node list. Then show how you would test it."
)

try:
    model = json.loads(urllib.request.urlopen(f"{BASE}/v1/models", timeout=60).read())["data"][0]["id"]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 384,
        "temperature": 0.0,
        "seed": 1234,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    used = json.loads(urllib.request.urlopen(req, timeout=600).read())["usage"]["completion_tokens"]
    print(f"    driven: {used} tokens (request COMPLETED — bound may be too high to force an exit)")
except Exception as exc:  # noqa: BLE001
    print(f"    request cut off by the step bound (EXPECTED): {type(exc).__name__}")
