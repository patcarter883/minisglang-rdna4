"""bs=1 spec-decode benchmark: decode + e2e tok/s WITH the acceptance stats that explain them.

`bench_decode.py` reports throughput but not why. A spec row is uninterpretable without accept
length: 92.9 tok/s can mean "great drafter, cheap verify" or "mediocre drafter, very cheap verify",
and the fix is different in each case. This deltas vLLM's spec counters across exactly the benched
requests so every row carries its own attribution:

    emitted tokens/step  = 1 + accepted/drafts        (how much the drafter buys)
    spec step time       = emitted / decode           (what a propose+verify step costs)
    step-cost multiplier = spec_step / plain_step     (vs --plain-tps, the no-spec baseline)

Spec is a WIN only when the multiplier is below emitted-tokens/step.

MEASURE SAMPLED, NOT GREEDY. Defaults here are this checkpoint's own generation_config
(temperature=1.0, top_k=20, top_p=0.95) — what a real client actually gets. `temperature=0.0` makes
verification an exact-match against a deterministic argmax against a drafter that IS the target
checkpoint, so acceptance sits near its ceiling and every spec number is inflated. The inflation is
NOT uniform: it is largest at the LATE draft positions, which is exactly where a bigger K earns its
keep, so a greedy sweep systematically over-recommends K. Real serving is rejection sampling against
a distribution. Pass --temperature 0 only to reproduce a historical greedy row.

Sampling adds run-to-run variance, hence --trials 5 by default; the spread is printed so a small
difference can be told apart from noise.

Usage:
    python tools/vhip_patches/bench_spec_decode.py [--port 8000] [--gen 200] [--trials 5]
                                                   [--plain-tps N] [--temperature 1.0]
                                                   [--top-k 20] [--top-p 0.95] [--label ...]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

PROMPT = (
    "Write a detailed technical explanation of how a gated delta-net linear-attention layer "
    "maintains its recurrent state across tokens. Be thorough and precise."
)

# Cumulative counters; a delta across the benched requests is those requests' own contribution.
_SPEC = "vllm:spec_decode_num_"
_KEYS = (
    f"{_SPEC}drafts_total",
    f"{_SPEC}draft_tokens_total",
    f"{_SPEC}accepted_tokens_total",
    "vllm:request_decode_time_seconds_sum",
)


def _get(url: str, timeout: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def counters(port: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in _get(f"http://localhost:{port}/metrics").splitlines():
        if line.startswith("#"):
            continue
        for key in _KEYS:
            if line.startswith(key):
                out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
        # per-position acceptance: position=N is "how often the Nth draft token survived"
        if line.startswith(f"{_SPEC}accepted_tokens_per_pos_total"):
            pos = line.split('position="', 1)[1].split('"', 1)[0]
            out[f"pos{pos}"] = out.get(f"pos{pos}", 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def post(port: int, model: str, max_tokens: int, samp: dict) -> tuple[int, float]:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,  # sampling can hit EOS early; pin length so decode rate is comparable
        "stream": False,
        **samp,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return out["usage"]["completion_tokens"], time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--plain-tps", type=float, default=None,
                    help="no-spec decode tok/s at the SAME config AND SAME sampling, for the "
                         "step-cost multiplier. A greedy baseline here invalidates the ratio.")
    # this checkpoint's generation_config.json (vLLM logs it as overriding its own defaults)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--label", default="result")
    args = ap.parse_args()

    samp = {"temperature": args.temperature}
    if args.temperature > 0:
        samp["top_k"], samp["top_p"] = args.top_k, args.top_p
    print(f"sampling: {samp}" + ("   *** GREEDY — inflates acceptance, not a serving number ***"
                                 if args.temperature == 0 else ""))

    model = json.loads(_get(f"http://localhost:{args.port}/v1/models"))["data"][0]["id"]
    post(args.port, model, 16, samp)  # warm: first call pays graph/JIT one-offs

    a = counters(args.port)
    tok = 0
    wall = 0.0
    per = []
    for _ in range(args.trials):
        c0 = counters(args.port)
        n, w = post(args.port, model, args.gen, samp)
        c1 = counters(args.port)
        d1 = c1["vllm:request_decode_time_seconds_sum"] - c0["vllm:request_decode_time_seconds_sum"]
        per.append((n - 1) / d1 if d1 > 0 else float("nan"))
        tok += n
        wall += w
    b = counters(args.port)

    def d(k: str) -> float:
        return b.get(k, 0.0) - a.get(k, 0.0)

    dt = d("vllm:request_decode_time_seconds_sum")
    # one token per request is the prefill's, not decode's
    decode = (tok - args.trials) / dt if dt > 0 else float("nan")
    e2e = tok / wall

    print(f"\n{args.label}:")
    print(f"  tokens {tok}   decode {decode:6.1f} tok/s   e2e {e2e:6.1f} tok/s")
    print(f"  per-trial decode: {' '.join(f'{x:.1f}' for x in per)}"
          f"   (spread {max(per) - min(per):.1f})")

    drafts = d(f"{_SPEC}drafts_total")
    if drafts <= 0:
        print("  spec: NO DRAFTS — speculation is off or never engaged")
        return

    dtok = d(f"{_SPEC}draft_tokens_total")
    acc = d(f"{_SPEC}accepted_tokens_total")
    emitted = 1.0 + acc / drafts
    k = round(dtok / drafts)
    pos = " ".join(
        f"pos{i} {d(f'pos{i}') / drafts * 100:.1f}%" for i in range(k) if f"pos{i}" in b
    )
    print(f"  drafts {drafts:.0f}  draft-tok {dtok:.0f}  accepted {acc:.0f}  (K={k})")
    print(f"  acceptance {acc / dtok * 100:.1f}%   ({pos})")
    print(f"  emitted tok/step {emitted:.3f} of max {k + 1}")

    step_ms = emitted / decode * 1000
    print(f"  spec step {step_ms:.2f} ms")
    if args.plain_tps:
        plain_ms = 1000.0 / args.plain_tps
        mult = step_ms / plain_ms
        verdict = "WIN" if emitted > mult else "LOSS"
        print(f"  plain step {plain_ms:.2f} ms (@{args.plain_tps} tok/s)")
        print(f"  step-cost {mult:.2f}x vs {emitted:.2f}x tokens -> {verdict} "
              f"{decode / args.plain_tps * 100 - 100:+.1f}%")


if __name__ == "__main__":
    main()
