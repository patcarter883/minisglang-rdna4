import json, time, urllib.request, concurrent.futures, sys

URL = "http://127.0.0.1:1919/v1/chat/completions"
MODEL = "QuantTrio/GLM-4.7-Flash-AWQ"
MAXTOK = 256

PROMPTS = [
    "Write a detailed technical explanation of how TCP congestion control works, covering slow start, congestion avoidance, fast retransmit, and fast recovery. Be thorough.",
    "Explain in depth how a B-tree database index works: structure, insertion, search, range scans, and why it suits disk-based storage. Be thorough.",
    "Describe the full lifecycle of an HTTP request in a modern web stack, from DNS resolution through TLS handshake, load balancing, app server, and response. Be thorough.",
]

def one(prompt):
    body = json.dumps({"model": MODEL, "temperature": 0.0, "max_tokens": MAXTOK,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    el = time.time() - t0
    return {"text": d["choices"][0]["message"]["content"], "elapsed": el}

def run(n, label):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(one, PROMPTS[:n]))
    wall = time.time() - t0
    for i, r in enumerate(res):
        r["chars"] = len(r["text"])
    return {"label": label, "n": n, "wall": wall, "results": res}

# warmup (primes graphs/caches)
one(PROMPTS[0])
out = []
out.append(run(1, "concurrency=1 (baseline)"))
out.append(run(2, "concurrency=2 (full)"))
json.dump(out, open("tools/glm_loadtest_raw.json", "w"))
for o in out:
    print(f"\n### {o['label']}: wall={o['wall']:.2f}s")
    for i, r in enumerate(o["results"]):
        print(f"  req{i}: elapsed={r['elapsed']:.2f}s chars={r['chars']}")
print("\n(raw saved -> tools/glm_loadtest_raw.json; exact token counts next)")
