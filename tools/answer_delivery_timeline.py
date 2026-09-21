#!/usr/bin/env python3
"""ONE TICK of the answer-delivery timeline: how long does a q4e serve take to stop delivering?

The 2026-09-22 reboot A/B proved the q4e serve loses answer delivery over its own UPTIME (chat-lane
no_answer 28/30 on a 15 h boot, 0/30 fresh, same commit — docs/journal/Q4E_DEGENERATION_2026-09-22.md).
It did NOT find the mechanism. This walks the curve: run a small probe on a schedule, record the
counts beside the serve's uptime and its accumulating internal counters, and see what moves WITH the
failure rate.

WHY A SMALL SAMPLE PER TICK. The full A/B arm is 60 requests and ~10 minutes; run every 30 minutes it
would be a load test, not an observation, and the probe would become a cause. A tick is `--reps 2`
over the 5 prompts on the CHAT LANE ONLY — 10 requests. That is enough: the two measured states are
~93% and 0%, so a 10-sample sees a >=30% rate with ~97% probability, and 0/10 vs 9/10 is
unmistakable. The raw lane is dropped because it is the A/B's control, not a time series, and one of
its cells (raw/capitals) is empty by prompt design in both arms.

WHAT TO CORRELATE. Recorded per tick: serve uptime, request/token totals, empty_completions, KV pool
use, prefix-cache hit ratio, and the `[expert-cache]` counter line (fill, observed_h, promotions,
evictions, inflight, free). Note from the first fresh-serve reading that `fill=0.987 inflight=25
free=0` are IDENTICAL on a healthy boot and a broken one — so those three are already known not to be
the signature on their own. Look for what DIVERGES, not for what merely looks alarming.

The tick is a CPU-only HTTP client: no GPU lease, no container. It shares the serve with real traffic
(max_running_req=2), so a tick can add latency to a concurrent agent turn — keep the schedule sparse.

Appends one row to `<out>/timeline.tsv` (header written once) and keeps every tick's full fixture
under `<out>/ticks/`. Prints an `ONSET` line the first time the rate crosses --onset-threshold, so a
log watcher can catch the transition without parsing the TSV.

    tools/answer_delivery_timeline.py --out /home/pat/fixtures/minisgl-answer-delivery-timeline

`--container` defaults to `auto`: the container publishing the serve port is resolved PER TICK,
because the name carries the lease label and changes on every relaunch. Nothing serving -> the tick
exits non-zero and writes no row, rather than measuring a container that no longer exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "answer_delivery_probe.py")

# Columns, in order. Keep append-only: a new column goes on the END so old rows stay parseable.
COLUMNS = [
    "ts_utc", "uptime_s", "container", "engine_sha", "engine_dirty",
    "n", "no_answer", "eos_in_think", "answer_wrong", "digit_noise", "cjk_leak", "loop",
    "requests_total", "gen_tokens_total", "empty_completions_total",
    "kv_pool_used", "kv_pool_total", "prefix_hit_ratio",
    "ec_fill", "ec_observed_h", "ec_promotions", "ec_evictions", "ec_inflight", "ec_free",
    "ec_deferred", "ec_throttled", "ec_stale_pub", "fixture",
]

_METRIC_MAP = {
    "requests_total": "minisgl_requests_total",
    "gen_tokens_total": "minisgl_generation_tokens_total",
    "empty_completions_total": "minisgl_empty_completions_total",
    "kv_pool_used": "minisgl_kv_pool_used_tokens",
    "kv_pool_total": "minisgl_kv_pool_total_tokens",
    "prefix_hit_ratio": "minisgl_prefix_cache_hit_ratio",
}
# Fields lifted off the serve's periodic `[expert-cache] …` line.
_EC_FIELDS = ["fill", "observed_h", "promotions", "evictions", "inflight", "free",
              "deferred", "throttled", "stale_pub"]


def scrape_metrics(base_url: str) -> dict:
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=15) as fh:
            body = fh.read().decode()
    except Exception:
        return {}
    out = {}
    for col, name in _METRIC_MAP.items():
        m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}}\s+(\S+)$", body, re.M)
        if m:
            out[col] = m.group(1)
    return out


def scrape_expert_cache(container: str) -> dict:
    """Last periodic `[expert-cache]` line. `docker logs --tail` on a chatty container is cheap; the
    line is emitted every MINISGL_EXPERT_CACHE_REPORT (default 200) ticks."""
    try:
        body = subprocess.run(["docker", "logs", "--tail", "400", container],
                              capture_output=True, text=True, timeout=60)
        lines = [l for l in (body.stdout + body.stderr).splitlines() if "[expert-cache] slots=" in l]
    except Exception:
        return {}
    if not lines:
        return {}
    last = lines[-1]
    return {f"ec_{k}": (m.group(1) if (m := re.search(rf"\b{k}=(\S+)", last)) else "")
            for k in _EC_FIELDS}


def resolve_container(name: str, base_url: str) -> str:
    """`--container auto` -> whichever container publishes the serve's port right now.

    The container name carries the lease label (`lease-<name>-serve`), so it CHANGES on every
    relaunch. A scheduled tick with a hardcoded name would keep running against a container that no
    longer exists and quietly stop measuring — the silent-stale failure this repo keeps paying for.
    Resolve it per tick instead, and fail loudly when nothing is serving.
    """
    if name != "auto":
        return name
    port = re.search(r":(\d+)", base_url)
    port = port.group(1) if port else "1919"
    out = subprocess.run(["docker", "ps", "--filter", f"publish={port}",
                          "--format", "{{.Names}}"], capture_output=True, text=True, timeout=30)
    names = [n for n in out.stdout.split() if n]
    if len(names) != 1:
        sys.exit(f"--container auto: expected exactly one container publishing {port}, got {names}")
    return names[0]


def uptime_seconds(container: str) -> tuple[str, int | str]:
    try:
        started = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.StartedAt}}", container],
            capture_output=True, text=True, check=True, timeout=30).stdout.strip()
        t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
        return started, int((datetime.now(timezone.utc) - t0).total_seconds())
    except Exception as exc:
        return f"<{exc!r}>", ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:1919/v1")
    ap.add_argument("--expect-model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--container", default="auto",
                    help="serve container name, or 'auto' to resolve whichever container publishes "
                         "the serve port right now (the name changes on every relaunch)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--onset-threshold", type=float, default=0.3,
                    help="no_answer fraction that counts as degraded (default 0.3)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "ticks"), exist_ok=True)
    args.container = resolve_container(args.container, args.base_url)
    started, up_s = uptime_seconds(args.container)

    # The probe owns provenance assertion (--expect-model aborts on a mismatch) and fixture writing.
    cmd = [sys.executable, PROBE, "--base-url", args.base_url,
           "--expect-model", args.expect_model, "--container", args.container,
           "--arm", f"tick-up{up_s}s", "--reps", str(args.reps),
           "--max-tokens", str(args.max_tokens), "--lanes", "chat",
           "--out", os.path.join(args.out, "ticks")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # A provenance failure or a dead serve must be LOUD and must not write a row that looks like
        # a clean measurement.
        sys.stderr.write(f"TICK FAILED rc={proc.returncode}\n{proc.stdout[-2000:]}\n"
                         f"{proc.stderr[-2000:]}\n")
        return proc.returncode
    fixture = next((l.split("fixture:", 1)[1].strip()
                    for l in reversed(proc.stdout.splitlines()) if l.startswith("fixture:")), "")
    with open(os.path.join(fixture, "result.json")) as fh:
        res = json.load(fh)
    chat = res["summary"]["by_lane"]["chat"]
    serve = res["provenance"]["serve"]

    row = {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "uptime_s": up_s, "container": args.container,
        "engine_sha": (serve.get("engine_sha") or "")[:12],
        "engine_dirty": "dirty" if serve.get("engine_dirty") else "clean",
        "n": chat["n"], **{k: chat.get(k, "") for k in
                           ("no_answer", "eos_in_think", "answer_wrong",
                            "digit_noise", "cjk_leak", "loop")},
        **scrape_metrics(args.base_url), **scrape_expert_cache(args.container),
        "fixture": os.path.basename(fixture),
    }

    tsv = os.path.join(args.out, "timeline.tsv")
    new = not os.path.exists(tsv)
    with open(tsv, "a") as fh:
        if new:
            fh.write("\t".join(COLUMNS) + "\n")
        fh.write("\t".join(str(row.get(c, "")) for c in COLUMNS) + "\n")

    frac = chat["no_answer"] / chat["n"] if chat["n"] else 0.0
    state = "DEGRADED" if frac >= args.onset_threshold else "healthy"
    print(f"[tick] up={up_s}s no_answer={chat['no_answer']}/{chat['n']} "
          f"eos_in_think={chat['eos_in_think']} {state} -> {tsv}")
    if state == "DEGRADED":
        # Grep-able transition marker: the whole point of the exercise.
        prior = [l.split("\t") for l in open(tsv).read().splitlines()[1:-1]]
        idx = COLUMNS.index("no_answer")
        n_idx = COLUMNS.index("n")
        was_clean = all(
            (int(p[idx]) / int(p[n_idx])) < args.onset_threshold
            for p in prior if len(p) > n_idx and p[n_idx].isdigit() and int(p[n_idx])
        )
        if was_clean:
            print(f"ONSET: first degraded tick at uptime {up_s}s "
                  f"({up_s / 3600:.2f} h), no_answer={chat['no_answer']}/{chat['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
