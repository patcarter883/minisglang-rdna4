#!/usr/bin/env python
"""Controlled A/B probe for the MTP prompt-prefill draft-KV seed lever. Stdlib only.

Measures, against a running minisgl OpenAI server:
  * REAL-token throughput from ``usage.completion_tokens`` (NOT SSE-chunk counts — the chunk meter
    undercounts spec ~2.4x). Non-streaming requests so the server reports exact completion_tokens.
  * Greedy determinism fingerprint (text of a fixed prompt set) so seed-on vs seed-off vs base can be
    diffed for losslessness.
  * Decode POWER (amdgpu_power_watts) sampled from Prometheus during a sustained decode.

Usage:
  seed_probe.py --url http://127.0.0.1:1919 --tag seed_on --out /path/out.json \
      [--prom http://host.docker.internal:9090] [--conc 4] [--decode-tokens 256]
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.parse
import urllib.request

# Fixed greedy prompt set — coherence + determinism fingerprint (identical across seed on/off/base).
_PROMPTS = [
    "Explain, step by step, how a modern CPU executes a program: fetch, decode, execute, pipelining, "
    "branch prediction, caches, and out-of-order execution.",
    "Write a short factual paragraph about the causes of the French Revolution.",
    "Describe how photosynthesis converts sunlight into chemical energy in plants.",
    "List the first ten prime numbers and briefly explain what a prime number is.",
    "Summarize the plot structure of a classic three-act screenplay.",
]


def _chat(url, prompt, max_tokens, ignore_eos, stream=False, barrier=None):
    payload = json.dumps({
        "model": "", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": ignore_eos, "stream": stream,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait()
    if not stream:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read().decode("utf-8", "ignore"))
        txt = d["choices"][0]["message"]["content"]
        ct = int(d.get("usage", {}).get("completion_tokens", 0))
        return txt, ct
    # streaming (used only for the sustained power-decode; return chunk count)
    n = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                n += 1
    return "", n


def greedy_fingerprint(url):
    """Greedy (temp=0) generations over the fixed prompt set — the losslessness fingerprint."""
    out = []
    for i, p in enumerate(_PROMPTS):
        txt, ct = _chat(url, p, max_tokens=128, ignore_eos=False)
        out.append({"id": i, "completion_tokens": ct, "text": txt})
    return out


def throughput(url, conc, decode_tokens, prom=None):
    """REAL-token throughput: `conc` concurrent greedy ignore_eos requests, tok/s from summed
    usage.completion_tokens over the wall from barrier release to the last completion. Optionally
    samples Prometheus amdgpu_power_watts (daemon poller) DURING the sustained decode so decode power
    is measured on a run that reliably completes (no fragile long single-stream)."""
    barrier = threading.Barrier(conc)
    results = [None] * conc

    def worker(i):
        txt, ct = _chat(url, _PROMPTS[i % len(_PROMPTS)], max_tokens=decode_tokens,
                        ignore_eos=True, barrier=barrier)
        results[i] = (ct, time.perf_counter())

    samples = {"0": [], "1": []}
    stop = threading.Event()

    def poller():
        while not stop.is_set():
            v = _prom_power(prom)
            for g in ("0", "1"):
                if g in v:
                    samples[g].append(v[g])
            stop.wait(1.0)

    pt = None
    if prom:
        pt = threading.Thread(target=poller, daemon=True)
        pt.start()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    stop.set()
    t1 = max(r[1] for r in results)
    total_ct = sum(r[0] for r in results)
    wall = t1 - t0
    out = {"conc": conc, "total_completion_tokens": total_ct, "wall_s": round(wall, 3),
           "real_tok_s": round(total_ct / wall, 2) if wall > 0 else 0.0}
    if prom:
        pw = {}
        for g in ("0", "1"):
            s = [x for x in samples[g] if x > 15]  # drop idle-floor samples before decode ramps
            if s:
                pw[g] = {"median_w": round(statistics.median(s), 1), "max_w": round(max(s), 1),
                         "n": len(s)}
        out["power"] = pw
    return out


def _prom_power(prom):
    vals = {}
    for gpu in ("0", "1"):
        try:
            q = urllib.parse.quote(f'amdgpu_power_watts{{gpu="{gpu}"}}')
            with urllib.request.urlopen(f"{prom}/api/v1/query?query={q}", timeout=5) as r:
                d = json.loads(r.read().decode())
            res = d.get("data", {}).get("result", [])
            if res:
                vals[gpu] = float(res[0]["value"][1])
        except Exception:
            pass
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prom", default="http://host.docker.internal:9090")
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--aa", action="store_true", help="run fingerprint twice (determinism floor)")
    a = ap.parse_args()

    result = {"tag": a.tag}
    print(f"[probe {a.tag}] greedy fingerprint (run 1) ...", flush=True)
    result["fingerprint"] = greedy_fingerprint(a.url)
    if a.aa:
        # A/A determinism floor: same config, same prompts, twice. Any mismatch is run-to-run GPU FP
        # non-determinism (non-associative reductions at logit near-ties), NOT a spec/seed effect.
        print(f"[probe {a.tag}] greedy fingerprint (run 2, A/A) ...", flush=True)
        fp2 = greedy_fingerprint(a.url)
        result["fingerprint_aa"] = fp2
        matches = sum(1 for x, y in zip(result["fingerprint"], fp2) if x["text"] == y["text"])
        result["aa_text_match"] = f"{matches}/{len(fp2)}"
        print(f"[probe {a.tag}] A/A intra-config text match = {matches}/{len(fp2)} "
              f"(mismatches => GPU FP non-determinism floor)", flush=True)
    prom = None if a.skip_power else a.prom
    print(f"[probe {a.tag}] throughput conc={a.conc} decode_tokens={a.decode_tokens} "
          f"(power={'on' if prom else 'off'}) ...", flush=True)
    result["throughput"] = throughput(a.url, a.conc, a.decode_tokens, prom=prom)
    print(f"[probe {a.tag}] throughput = {result['throughput']}", flush=True)
    print(f"[probe {a.tag}] throughput conc=1 decode_tokens={a.decode_tokens} ...", flush=True)
    result["throughput_c1"] = throughput(a.url, 1, a.decode_tokens, prom=None)
    print(f"[probe {a.tag}] throughput_c1 = {result['throughput_c1']}", flush=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[probe {a.tag}] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
