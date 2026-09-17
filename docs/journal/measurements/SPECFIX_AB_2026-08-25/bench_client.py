"""Sampled M=1 decode bench against a RUNNING minisgl serve (read-only over HTTP, no lease).

Replicates the 2026-08-19 dspark protocol: coding prompt, SAMPLED seeds 101/202/303, M=1,
1200 tok. TRUE tok/s = usage.completion_tokens / wall (never SSE chunks). Sampling params are
deliberately OMITTED so the engine applies the checkpoint generation_config (the real serving
path); only `seed` is sent for repeatability. /metrics is scraped before/after for the spec
accept counters.
"""
import json, re, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1919"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout"
SEEDS = [101, 202, 303]
MAXTOK = 1200

PROMPT = ("Write a complete Python module implementing an LRU cache with per-entry TTL, "
          "thread safety, and a decorator interface. Include type hints, docstrings, and a "
          "small unit-test class covering eviction order, TTL expiry, and concurrent access.")


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.read().decode()


def metrics():
    out = {}
    try:
        for line in get("/metrics").splitlines():
            m = re.match(r"^(minisgl_spec_\w+|minisgl_decode_\w+)(?:{[^}]*})?\s+([0-9.eE+-]+)$", line)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    except Exception as e:
        out["_error"] = str(e)
    return out


model = json.loads(get("/v1/models"))["data"][0]["id"]
# warmup (graph replay + lazy alloc) — discarded
post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "warmup: say ok"}],
                              "max_tokens": 128, "seed": 1})
m0 = metrics()
runs = []
for seed in SEEDS:
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": model,
                                      "messages": [{"role": "user", "content": PROMPT}],
                                      "max_tokens": MAXTOK, "seed": seed})
    wall = time.time() - t0
    ct = r.get("usage", {}).get("completion_tokens", 0)
    runs.append({"seed": seed, "completion_tokens": ct, "wall_s": round(wall, 2),
                 "tok_s": round(ct / wall, 2) if wall > 0 else 0.0})
    print(f"  seed {seed}: {ct} tok in {wall:.1f}s = {ct/wall:.2f} tok/s", file=sys.stderr)
m1 = metrics()

toks = sorted(r["tok_s"] for r in runs)
result = {
    "model": model, "runs": runs,
    "median_tok_s": toks[len(toks) // 2],
    "spread_tok_s": round(toks[-1] - toks[0], 2),
    "metrics_delta": {k: m1.get(k, 0) - m0.get(k, 0) for k in sorted(set(m0) | set(m1))
                      if not k.startswith("_")},
}
with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps({k: result[k] for k in ("median_tok_s", "spread_tok_s")}), file=sys.stderr)
