"""Multi-turn cache-hit bench for the prefix-cache seed A/B (read-only over HTTP).

Exercises the traffic class the seed_prefill fix targets: a long shared prefix (system + prior
turns, radix-cached) plus a short new turn whose suffix length passes the seed gate (<=512 tok).
On the BASELINE code the drafter's KV is seeded misaligned on every cache-hit turn (accept-len
collapses, tok/s drops); on the FIX it seeds correctly. Turn 1 is the cold control — the two legs
should match there.

One conversation, 6 turns, sampled via the checkpoint generation_config (only `seed` sent).
Per-turn: TRUE tok/s (usage.completion_tokens / wall), cached_tokens if reported, and the
/metrics spec-counter deltas (accept-len per turn).
"""
import json, re, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:1919"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/dev/stdout"
MAXTOK = 400

CONTEXT = ("You are reviewing a large Python codebase for an LLM inference engine. "
           "Key modules: scheduler (batch formation, KV paging, radix prefix cache), "
           "attention backends (paged MHA, MLA latent, sliding-window), speculative decoding "
           "(draft proposers, verify, acceptance), and quantized GEMM kernels. ") * 40  # ~3k tok

TURNS = [
    "Summarize the main responsibilities of the scheduler module in 4 bullet points.",
    "How does a radix prefix cache reduce prefill cost? Give a concrete example.",
    "Explain the verify step of speculative decoding and why acceptance rate matters.",
    "What are the tradeoffs between paged and contiguous KV layouts?",
    "Describe how sliding-window attention interacts with a prefix cache.",
    "Write a short docstring for a function that evicts cold radix-cache pages.",
]


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
            m = re.match(r"^(minisgl_spec_\w+)(?:{[^}]*})?\s+([0-9.eE+-]+)$", line)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    except Exception as e:
        out["_error"] = str(e)
    return out


model = json.loads(get("/v1/models"))["data"][0]["id"]
post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "warmup: say ok"}],
                              "max_tokens": 64, "seed": 1})

msgs = [{"role": "system", "content": CONTEXT}]
turns = []
for i, q in enumerate(TURNS):
    msgs.append({"role": "user", "content": q})
    m0 = metrics()
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": model, "messages": msgs,
                                      "max_tokens": MAXTOK, "seed": 100 + i})
    wall = time.time() - t0
    m1 = metrics()
    u = r.get("usage", {})
    ct = u.get("completion_tokens", 0)
    acc = m1.get("minisgl_spec_accepted_tokens_total", 0) - m0.get("minisgl_spec_accepted_tokens_total", 0)
    emi = m1.get("minisgl_spec_emitted_tokens_total", 0) - m0.get("minisgl_spec_emitted_tokens_total", 0)
    reply = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    msgs.append({"role": "assistant", "content": reply})
    turn = {"turn": i + 1, "completion_tokens": ct, "wall_s": round(wall, 2),
            "tok_s": round(ct / wall, 2) if wall > 0 else 0.0,
            "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": u.get("cached_tokens"),
            "spec_accepted": acc, "spec_emitted": emi,
            "accept_len": round(emi / (emi - acc), 3) if emi > acc else None}
    turns.append(turn)
    print(f"  turn {i+1}: {turn['tok_s']} tok/s accept_len={turn['accept_len']} "
          f"cached={turn['cached_tokens']}", file=sys.stderr)

hot = [t["tok_s"] for t in turns[1:]]
hot_al = [t["accept_len"] for t in turns[1:] if t["accept_len"]]
result = {"model": model, "turns": turns,
          "cold_tok_s": turns[0]["tok_s"],
          "hot_median_tok_s": sorted(hot)[len(hot) // 2],
          "hot_mean_accept_len": round(sum(hot_al) / len(hot_al), 3) if hot_al else None}
with open(OUT, "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps({k: result[k] for k in ("cold_tok_s", "hot_median_tok_s", "hot_mean_accept_len")}),
      file=sys.stderr)
