"""bs=1 A/B for the CAM transparent-read penalty. Stdlib only (host torch is a broken CUDA build).

Fires single-stream chat completions with cam_read False (no retrieve) vs True (retrieve fires) at
fixed gen lengths, so tps_false/tps_true = the 'halving' factor. Server-side [cam-prof] logs give the
retrieve phase breakdown + round-trip wall (read separately from docker logs).
"""
import json, time, urllib.request, statistics as st

URL = "http://localhost:1919/v1/chat/completions"
# A prompt with proper-noun spans + a phrase subject, moderate length -> realistic candidate count.
PROMPT = ("I was reading about Wolfgang Amadeus Mozart and Marie Antoinette yesterday, and also about "
          "the capital of Zorbia and the mother tongue of Lionel Jospin. Can you tell me what you know "
          "about Danielle Darrieux, Oleg Kotov, and the city of Reykjavik in a couple of sentences?")


def call(max_tokens, cam_read):
    body = json.dumps({
        "model": "x", "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0.0,
        "cam_read": cam_read, "cam_write": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    usage = d.get("usage", {})
    ct = usage.get("completion_tokens") or max_tokens
    return wall, ct


def bench(max_tokens, cam_read, reps=5):
    call(max_tokens, cam_read)  # warm
    walls, cts = [], []
    for _ in range(reps):
        w, ct = call(max_tokens, cam_read)
        walls.append(w); cts.append(ct)
    wall = st.median(walls); ct = st.median(cts)
    return wall, ct, ct / wall


print(f"prompt words={len(PROMPT.split())}\n")
for N in (16, 64, 256):
    wf, cf, tf = bench(N, False)
    wt, ct_, tt = bench(N, True)
    print(f"max_tokens={N:4d}  cam_read=OFF: {wf*1e3:7.0f}ms {tf:6.1f}tok/s | "
          f"cam_read=ON: {wt*1e3:7.0f}ms {tt:6.1f}tok/s | "
          f"penalty(off/on tps)={tf/tt:4.2f}x  wall_add={ (wt-wf)*1e3:6.0f}ms", flush=True)
