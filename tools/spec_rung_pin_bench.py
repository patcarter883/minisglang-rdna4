"""bs=1 tok/s at a PINNED verify rung, with provenance.

Answers one question: does a faster GDN verify kernel move the tokens-per-MILLISECOND optimum of
the spec-decode width ladder? The controller picks a rung from ACCEPTANCE (`want = E[A]+1`), which
cannot see cost at all, so the rung it picks and the rung that serves fastest are not the same
question. `MINISGL_SPEC_VERIFY_WIDTH_PIN` forces a rung so each one's true rate is measurable.

WHAT IT ASSERTS, because a pinned leg that silently ran a different rung is worse than no data:
  * the verify-width histogram is ~100% at the rung this leg claims to have pinned;
  * the GDN verify arm that ran is the one this image is supposed to dispatch;
  * tok/s comes from usage.completion_tokens / wall, never from counting SSE chunks (one chunk
    carries a whole accepted block under spec, so chunk-counting under-reports by accept-len);
  * every rep uses a DISTINCT prompt suffix, so no rep is measuring a radix prefix-cache hit.

Usage: python3 tools/spec_rung_pin_bench.py --pin 7 --leg baseline --out <dir> [--reps 5]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import statistics
import sys
import time
import urllib.request

PROMPT = ("Explain how speculative decoding works in a language model inference engine, "
          "covering the drafter, the verify step, and why acceptance matters.")


def gen(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.7, "seed": 1234}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return d["usage"]["completion_tokens"], time.perf_counter() - t0


def logs(container, n=6000):
    o = subprocess.run(["docker", "logs", "--tail", str(n), container],
                       capture_output=True, text=True, timeout=120)
    return o.stdout + o.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:1919")
    ap.add_argument("--container", default="lease-gpu0-1-serve")
    ap.add_argument("--pin", type=int, required=True)
    ap.add_argument("--leg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--maxtok", type=int, default=512)
    ap.add_argument("--expect-arm", default=None,
                    help="GDN verify arm this leg MUST engage, e.g. gdn_hip.gdn_prefill_verify_wmma. "
                         "Without it a two-leg A/B cannot tell 'new vs old' from 'old vs itself'.")
    a = ap.parse_args()

    model = json.loads(urllib.request.urlopen(a.base + "/v1/models", timeout=30).read())["data"][0]["id"]
    boot = logs(a.container, 20000)

    # provenance BEFORE measuring: the boot line must name this leg's pin.
    m = re.search(r"PINNED to rung (\d+)", boot)
    pinned = int(m.group(1)) if m else 0
    arms = sorted(set(re.findall(r"gdn_hip\.gdn_(?:prefill_verify_wmma|prefill_verify|verify_replay)", boot)))
    ladder = re.search(r"ADAPTIVE verify width \w+ widths=\[([0-9, ]+)\]", boot)

    # FAIL BEFORE SPENDING THE MEASUREMENT. A leg that booted the wrong pin, or whose engine never
    # dispatched to the kernel this leg is supposed to be testing, is not a slow leg -- it is a
    # different experiment wearing this leg's name. Both have happened here: a pin silently came up
    # at 15 when 7 was asked for, and an entire 4-leg A/B ran the recurrent verify on BOTH images
    # because the engine source was shared and only the kernel package differed.
    if pinned != a.pin:
        print(f"ABORT: asked for rung {a.pin}, serve booted PINNED to rung {pinned}", file=sys.stderr)
        return 1
    if a.expect_arm and a.expect_arm not in arms:
        print(f"ABORT: leg '{a.leg}' must engage {a.expect_arm}, but the engine engaged {arms}. "
              f"Check the MOUNTED SOURCE TREE, not just the image: compose mounts .:/engine, so the "
              f"kernel package and the dispatch that calls it come from different places.",
              file=sys.stderr)
        return 1

    gen(a.base, model, "Say hello.", 32)                       # warmup, discarded
    before = logs(a.container)
    tps = []
    for i in range(a.reps):
        t, w = gen(a.base, model, PROMPT + f"\n\n(Rep {i}: emphasise point {i+1}.)", a.maxtok)
        tps.append(t / w)
    after = logs(a.container)

    def widths(txt):
        w = re.findall(r"verify-width\[([^\]]*)\]", txt)
        if not w:
            return {}
        return {int(k): int(v.split("(")[0]) for k, v in
                (p.split(":", 1) for p in w[-1].split())}
    w0, w1 = widths(before), widths(after)
    delta = {k: w1.get(k, 0) - w0.get(k, 0) for k in set(w0) | set(w1)}
    delta = {k: v for k, v in delta.items() if v > 0}
    tot = sum(delta.values()) or 1
    ran = max(delta, key=delta.get) if delta else -1
    share = 100.0 * delta.get(ran, 0) / tot

    acc = re.findall(r"mean accept-len=([0-9.]+).*?committed/verify=([0-9.]+)", after)
    rec = {
        "leg": a.leg, "pin_requested": a.pin, "pin_reported_by_serve": pinned,
        "ladder": ladder.group(1) if ladder else None,
        "gdn_verify_arms_engaged": arms,
        "width_actually_run": ran, "width_share_pct": round(share, 1),
        "width_delta": {str(k): v for k, v in sorted(delta.items())},
        "tok_s_median": round(statistics.median(tps), 2),
        "tok_s_min": round(min(tps), 2), "tok_s_max": round(max(tps), 2),
        "reps": a.reps,
        "accept_len": float(acc[-1][0]) if acc else None,
        "committed_per_verify": float(acc[-1][1]) if acc else None,
    }
    if rec["committed_per_verify"] and rec["tok_s_median"]:
        rec["step_ms"] = round(rec["committed_per_verify"] / rec["tok_s_median"] * 1000, 2)

    rec["expect_arm"] = a.expect_arm
    rec["arm_ok"] = (a.expect_arm in arms) if a.expect_arm else None
    ok = ((pinned == a.pin) and (ran == a.pin) and share > 95.0
          and (rec["arm_ok"] is not False))
    rec["provenance_ok"] = ok
    path = f"{a.out}/leg-{a.leg}-pin{a.pin}.json"
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))
    if not ok:
        print(f"\nPROVENANCE FAILED: asked for rung {a.pin}, serve reported {pinned}, "
              f"ran {ran} at {share:.1f}% — this leg is NOT a measurement of rung {a.pin}.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
