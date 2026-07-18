"""Replay a real-trace load against the serve.

Modes:
  sequence (RECOMMENDED for benchmarking) — completion-driven replay that PRESERVES the capture's
     concurrency but DROPS its wall-clock timing. The capture's real [offset_s, offset_s+dur_s]
     intervals are partitioned into concurrency LANES (greedy interval assignment; #lanes = the
     trace's real peak concurrency). All lanes run in parallel; WITHIN a lane each request fires as
     soon as the prior one completes (no idle gaps, no clock). So the fan-out width / overlap
     structure is honoured, but runs are deterministic and comparable across kernel changes. Streams
     each request -> per-request TTFT (prefill latency) + decode tok/s.
       python replay_load.py <jsonl> <port> sequence [model]
  schedule — fire each request at its real offset_s/SPEED, reproducing the trace's concurrency
     envelope (fan-out bursts + idle gaps). SPEED compresses wall clock, preserves arrival SHAPE.
       python replay_load.py <jsonl> <port> schedule [speed=8] [model]
  saturate — CONC workers loop back-to-back for DUR seconds (max throughput; NOT representative of
     real shape — use only to force sustained decode for kernel profiling).
       python replay_load.py <jsonl> <port> saturate <conc> <dur_s> [model]

CLEANUP: the capture contains timed-out requests that the agent re-sent later (same question, doubled
`<objective>` tag, ending in a ptok=0 failure). DROP_IDX (indices in offset-sorted order) removes
those retry bursts so the replay reflects the intended traffic, not the timeout noise. Override on the
CLI with `--keep-all` to replay the raw trace.
"""
import sys, json, time, threading, urllib.request

# Timed-out / re-sent retry bursts to drop (offset-sorted indices). Evidence:
#  71      : dur 305s (client timeout), ptok=0 FAILED — "database migrations" objective
#  72-75   : "exception handling" objective x4 in a row (retry burst) -> re-sent clean at idx 84
#  76-82   : "database migrations" objective x7 in a row (retries of 71) -> re-sent clean at idx 83
#  118     : dur 0s, ptok=0 empty FAILURE (tail of the RainGauge assembly retry storm)
# The clean single-`<objective>` re-sends at 83-87 are KEPT.
DROP_IDX = {71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 118}

args = [a for a in sys.argv[1:] if a != "--keep-all"]
KEEP_ALL = "--keep-all" in sys.argv
JSONL, PORT, MODE = args[0], int(args[1]), args[2]
DEFMODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"

reqs = [json.loads(l) for l in open(JSONL)]
reqs.sort(key=lambda r: r.get("offset_s", 0))
if not KEEP_ALL:
    reqs = [r for i, r in enumerate(reqs) if i not in DROP_IDX]
    print(f"[cleanup] dropped {len(DROP_IDX)} timed-out/re-sent requests -> {len(reqs)} kept "
          f"(pass --keep-all to replay raw)")

lock = threading.Lock(); acc = {"n": 0, "err": 0, "out": 0}


def _post(req, model, stream):
    body = json.dumps({"model": model, "messages": req["messages"],
                       "max_tokens": min(req.get("max_tokens", 512), 1024),
                       "temperature": 0.7, "stream": stream}).encode()
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=600)


def fire(req, model):  # non-stream (schedule/saturate): count output tokens only
    try:
        with _post(req, model, False) as resp:
            d = json.loads(resp.read()); ot = d.get("usage", {}).get("completion_tokens", 0)
            with lock: acc["n"] += 1; acc["out"] += ot
    except Exception:
        with lock: acc["err"] += 1


def fire_streamed(req, model):
    """Stream one request -> (ttft_s, total_s, out_tokens). ttft = first content token = prefill."""
    t0 = time.time(); ttft = None; n = 0
    try:
        with _post(req, model, True) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except Exception:
                    continue
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
        return ttft if ttft is not None else (time.time() - t0), time.time() - t0, n
    except Exception:
        return None, time.time() - t0, 0


def _pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs); k = max(0, min(len(xs) - 1, round((p / 100) * (len(xs) - 1))))
    return xs[k]


def assign_lanes(rs):
    """Greedy interval partition on the real [offset, offset+dur] intervals -> list of lanes, each a
    list of request-indices in order. #lanes = the trace's real peak concurrency; each lane's requests
    did NOT overlap in the capture, so replaying a lane back-to-back preserves that non-overlap while
    the parallel lanes reproduce the concurrency width."""
    lane_end = []      # last real end-time per lane
    lane_reqs = []     # request list per lane
    for i, r in enumerate(rs):
        s = r.get("offset_s", 0.0); e = s + r.get("dur_s", 0.0)
        placed = False
        for L in range(len(lane_end)):
            if lane_end[L] <= s + 1e-6:      # lane free by this request's real start
                lane_end[L] = e; lane_reqs[L].append(i); placed = True; break
        if not placed:
            lane_end.append(e); lane_reqs.append([i])
    return lane_reqs


if MODE == "sequence":
    model = args[3] if len(args) > 3 else DEFMODEL
    lanes = assign_lanes(reqs)
    ttfts, tputs = [], []
    print(f"sequence replay: {len(reqs)} calls across {len(lanes)} concurrency lanes (real peak "
          f"concurrency); lanes run in parallel, each fires back-to-back on completion; timing dropped")

    def run_lane(idx_list):
        for i in idx_list:
            ttft, tot, n = fire_streamed(reqs[i], model)
            if n == 0:
                with lock: acc["err"] += 1
            else:
                dec = (n - 1) / max(1e-6, (tot - ttft)) if n > 1 else None
                with lock:
                    acc["n"] += 1; acc["out"] += n; ttfts.append(ttft)
                    if dec is not None: tputs.append(dec)

    t_all = time.time()
    ths = [threading.Thread(target=run_lane, args=(L,), daemon=True) for L in lanes]
    for t in ths: t.start()
    for t in ths: t.join(timeout=3600)
    el = time.time() - t_all
    print(f"done: {acc['n']} ok, {acc['err']} err, {acc['out']} out-tok over {el:.0f}s")
    print(f"  TTFT (prefill) s : p50={_pct(ttfts,50):.2f} p90={_pct(ttfts,90):.2f} max={max(ttfts,default=float('nan')):.2f}")
    print(f"  decode tok/s     : p50={_pct(tputs,50):.1f} p90={_pct(tputs,90):.1f}  (per-request, concurrent)")
    print(f"  overall throughput: {acc['out']/el:.1f} out-tok/s across {len(lanes)} lanes")

elif MODE == "schedule":
    speed = float(args[3]) if len(args) > 3 else 8.0
    model = args[4] if len(args) > 4 else DEFMODEL
    t0 = time.time(); ths = []
    print(f"schedule replay: {len(reqs)} calls, real span {reqs[-1]['offset_s']:.0f}s at SPEED={speed} "
          f"(~{reqs[-1]['offset_s']/speed:.0f}s wall), fan-out bursts + idle gaps preserved")
    for req in reqs:
        due = t0 + req.get("offset_s", 0) / speed
        d = due - time.time()
        if d > 0: time.sleep(d)
        t = threading.Thread(target=fire, args=(req, model), daemon=True); t.start(); ths.append(t)
    for t in ths: t.join(timeout=600)
    el = time.time() - t0
    print(f"done: {acc['n']} ok, {acc['err']} err, {acc['out']} out-tok over {el:.0f}s")

else:  # saturate
    import itertools
    CONC, DUR = int(args[3]), float(args[4])
    model = args[5] if len(args) > 5 else DEFMODEL
    reqs.sort(key=lambda r: -r.get("max_tokens", 0)); pool = itertools.cycle(reqs)
    stop = time.time() + DUR
    def worker():
        while time.time() < stop:
            with lock: req = next(pool)
            fire(req, model)
    ts = [threading.Thread(target=worker, daemon=True) for _ in range(CONC)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join(timeout=DUR + 300)
    el = time.time() - t0
    print(f"saturate: {acc['n']} ok, {acc['err']} err, {acc['out']} out-tok over {el:.0f}s "
          f"=> {acc['out']/el:.1f} tok/s @conc {CONC}")
