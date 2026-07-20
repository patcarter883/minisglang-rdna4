"""bs=1 A/B for the CAM auto-WRITE path (extract_facts). Statement-shaped prompt (passes the fact
heuristic) so auto-write actually fires; cam_write True vs False isolates the extract_facts generation
cost. Read is held off. Stdlib only.
"""
import json, time, urllib.request, statistics as st

URL = "http://localhost:1919/v1/chat/completions"
# STATEMENT (assertion) so _looks_like_fact_statement passes -> extract_facts fires when cam_write on.
PROMPT = ("Zephyrina Marsh's mother tongue is Klingon, and Oleg Kotov was born in the city of Reykjavik. "
          "Danielle Darrieux speaks Japanese and Lionel Jospin speaks Dutch.")


def call(max_tokens, cam_write):
    body = json.dumps({
        "model": "x", "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0.0,
        "cam_read": False, "cam_write": cam_write,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    return time.perf_counter() - t0, (d.get("usage", {}).get("completion_tokens") or max_tokens)


def bench(max_tokens, cam_write, reps=5):
    call(max_tokens, cam_write)
    walls, cts = [], []
    for _ in range(reps):
        w, ct = call(max_tokens, cam_write)
        walls.append(w); cts.append(ct)
    wall = st.median(walls); ct = st.median(cts)
    return wall, ct / wall


print(f"prompt words={len(PROMPT.split())} (statement -> triggers extract_facts)\n")
for N in (16, 64):
    wf, tf = bench(N, False)
    wt, tt = bench(N, True)
    print(f"max_tokens={N:4d}  cam_write=OFF: {wf*1e3:7.0f}ms {tf:6.1f}tok/s | "
          f"cam_write=ON: {wt*1e3:7.0f}ms {tt:6.1f}tok/s | "
          f"penalty(off/on tps)={tf/tt:4.2f}x  wall_add={(wt-wf)*1e3:6.0f}ms", flush=True)
