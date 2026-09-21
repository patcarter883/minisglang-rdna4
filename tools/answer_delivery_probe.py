#!/usr/bin/env python3
"""Does a chat turn DELIVER AN ANSWER? — the one-line-question counterpart to toolcall_degen_probe.

WHY A SECOND PROBE. `toolcall_degen_probe.py` drives tool-calling conversations and scores argument
corruption. It is blind to the failure this one measures, and so is every engine metric:

    a turn that reasons for 24 tokens, emits <|im_end|> INSIDE its <think> span, and returns
    `content: ""` is reported by the engine as finish_reason="stop" and counted in
    `minisgl_requests_success_total`. `minisgl_empty_completions_total` does NOT count it, because
    the reasoning tokens are output. The client gets nothing and monitoring sees a success.

That is what hit Hermes session 5d26424a4be3 (2026-09-21 23:02): a fresh ~70-token context on the
q4e arm, zero delivered content, the frontend's truncation-continuation retry re-fed the fragment,
and the turn ran 1,493 s before the client's repetition guard killed it.

The engine mechanism is not a bug to fix here but it IS the reason a plain prompt has no floor:
`_resolve_think_budget` returns THINK_BUDGET_UNBOUNDED for any template that consumes
`reasoning_effort` (the whole Qwen3 family — this checkpoint renders `xhigh` for an unset value),
and `ThinkGate.suppress_eos` returns False under an unbounded budget BY CONTRACT ("Unbounded
thinking has to mean the model may also stop on its own"). So on this arm nothing suppresses a
mid-think EOS and nothing forces `</think>`: there is no engine-side guarantee that an answer is
ever produced. Pass --reasoning-max-tokens N to arm both mechanisms and measure the difference.

TWO LANES, BECAUSE ONE OF THEM IS THE CONTROL. The `chat` lane is the failing path. The `raw` lane
sends the same question to /v1/completions as plain text — no chat template, no <think> span. On a
serve where the weights are healthy the raw lane answers correctly even while the chat lane returns
nothing, which is what separates "the model is broken" from "the chat/think path is broken". An arm
whose RAW lane is also wrong is a different (worse) finding, so both are always scored.

DETERMINISM IS NOT THE SIGNAL. This engine's greedy floor is ~4-8 tokens on the chat lane
(kernel-atomics near-tie argmax flips, measured in docs/journal/Q4E_DEGENERATION_2026-09-21.md §3),
so two identical requests legitimately differ. Never score "the arms diverged". Score whether an
answer arrived and whether it was right, over enough reps to separate the rates.

PROVENANCE IS ASSERTED, NOT ASSUMED (same contract as toolcall_degen_probe): --expect-model is
required and checked against /v1/models, and --container records the serve's image, boot time and
the git sha of the tree it has mounted at /engine. An A/B whose two arms silently served the same
build is the failure mode that has cost this repo the most time; an A/B whose arms differ in the
source tree as well as the thing under test is the second.

DETECTORS. All countable, all recorded per turn, full text always written to the fixture dir.

  * no_answer      — content (stripped) is empty. THE HEADLINE DEFECT.
  * answer_wrong   — content is non-empty but a required substring is missing.
  * eos_in_think   — no_answer AND finish_reason=="stop" AND reasoning non-empty: the model ended
                     the turn inside its reasoning span. Distinguishes the mid-think EOS from a
                     length cut.
  * digit_noise    — invented digits in the output of a prompt that CONTAINS NO DIGITS: a bare run
                     of >=6, or one token carrying >=8 digits in total (see has_digit_noise). This is
                     NOT §6's `id_noise`: that detector is tool-arguments-only and scores fused
                     forms (`hex::`, `_40hex_`) exclusively, which free prose never produces. The
                     free-text signature is a real identifier decorated with an invented digit run
                     (`@docs/ui-plan` -> `docs/22909-74486-01`; `Internal reasoning step 1:
                     51424964`). Its own false-positive rate is measurable on this probe's own
                     digit-free prompts on a healthy arm — read it there before trusting a rate.
  * cjk_leak       — CJK codepoints in the output of an all-ASCII English prompt.
  * loop           — the most-repeated 10-gram in the turn occurs >= --loop-threshold times.

Counts alone have burned this investigation before, so every turn's FULL text goes to the fixture
directory. READ A CLEAN ARM'S TEXT before believing it.

USAGE

    tools/answer_delivery_probe.py \
        --base-url http://localhost:1919/v1 --expect-model Qwen3.8-Flash-Next \
        --container lease-hitdiv-serve --arm before-reboot --reps 8 \
        --out /home/pat/fixtures/minisgl-answer-delivery

CPU-only: an HTTP client. No GPU lease, no container of its own — the serve under test holds the
lease.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

# ---------------------------------------------------------------------------------------------
# The prompt set. Each carries a machine-checkable expectation so "an answer arrived" and "the
# answer was right" are separate counts. `digit_free` marks the prompts the digit_noise detector is
# allowed to score (a prompt containing digits makes a digit run in the answer legitimate).
# ---------------------------------------------------------------------------------------------
PROMPTS = [
    {
        "id": "capitals",
        "text": "List the capital cities of France, Japan, Italy, Egypt and Canada. One per line.",
        # Checked case-insensitively; all five must be present.
        "require_all": ["paris", "tokyo", "rome", "cairo", "ottawa"],
        "digit_free": True,
    },
    {
        "id": "arith",
        "text": "What is 17 + 26? Answer with just the number.",
        "require_all": ["43"],
        "digit_free": False,
    },
    {
        "id": "echo",
        "text": "Repeat this exactly, and nothing else: the quick brown fox jumps over the lazy dog",
        "require_all": ["quick brown fox jumps over the lazy dog"],
        "digit_free": True,
    },
    {
        "id": "ocean",
        "text": "Write exactly one sentence about the ocean.",
        # No content requirement beyond delivery: scores no_answer only.
        "require_all": [],
        "digit_free": True,
    },
    {
        "id": "count",
        "text": "Count from 1 to 10, separated by commas. Output only the numbers.",
        "require_all": ["1", "10"],
        "digit_free": False,
    },
]

_CJK = re.compile(r"[　-鿿＀-￯]")
# A bare run of >=6 digits, OR a single whitespace-delimited token whose TOTAL digit content is >=8.
# The second clause is load-bearing and a naive `\d{6,}` misses every real sample: the observed
# signature is a hyphenated invented id, not one long run — `docs/22909-74486-01` (12 digits, longest
# run 5) and `ui-0000446015-000046` (16 digits) both slip past a run-length rule. Scored only on
# prompts marked digit_free, where no answer has a legitimate reason to carry eight digits. Known and
# accepted over-fire: a spelled-out date (`01/06/2017` = 8) counts — on a digit-free prompt that is
# already the signature, and the 2026-09-21 23:02 collapse emitted exactly that, ~100 times.
_DIGIT_RUN = re.compile(r"\d{6,}")


def has_digit_noise(text: str) -> bool:
    if _DIGIT_RUN.search(text):
        return True
    return any(sum(c.isdigit() for c in tok) >= 8 for tok in text.split())


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def assert_model(base_url: str, expect: str, timeout: float) -> dict:
    """Abort before a single probe runs if the endpoint is not serving the expected model. An A/B
    whose arms silently served the same build is this repo's most expensive recurring mistake."""
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as fh:
        listing = json.load(fh)
    ids = [m.get("id") for m in listing.get("data") or []]
    if expect not in ids:
        sys.exit(f"provenance FAIL: /v1/models lists {ids}, expected {expect!r}")
    entry = next(m for m in listing["data"] if m.get("id") == expect)
    return {"models_listed": ids, "entry": entry}


def container_provenance(name: str | None) -> dict:
    """Image, boot time, and the git sha of the tree the serve has mounted at /engine. The engine
    source is hot-mounted, so the image tag alone does not identify the code under test."""
    if not name:
        return {}
    out: dict = {"container": name}
    try:
        fmt = "{{.Config.Image}}\t{{.State.StartedAt}}\t{{.State.Status}}"
        img, started, status = subprocess.run(
            ["docker", "inspect", "--format", fmt, name],
            capture_output=True, text=True, check=True,
        ).stdout.strip().split("\t")
        out.update(image=img, started_at=started, status=status)
        mounts = json.loads(subprocess.run(
            ["docker", "inspect", "--format", "{{json .Mounts}}", name],
            capture_output=True, text=True, check=True,
        ).stdout)
        engine = next((m["Source"] for m in mounts if m.get("Destination") == "/engine"), None)
        out["engine_source"] = engine
        if engine:
            for key, args in (("engine_sha", ["rev-parse", "HEAD"]),
                              ("engine_dirty", ["status", "--short"])):
                r = subprocess.run(["git", "-C", engine] + args, capture_output=True, text=True)
                out[key] = r.stdout.strip() if r.returncode == 0 else f"<{r.returncode}>"
    except Exception as exc:  # provenance is recorded best-effort; never blocks a run
        out["provenance_error"] = repr(exc)[:200]
    return out


def max_ngram_repeat(text: str, n: int = 10) -> tuple[int, str]:
    words = text.split()
    if len(words) < n:
        return 0, ""
    grams = Counter(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    return count, gram


def score(prompt: dict, reasoning: str, content: str, finish: str | None,
          loop_threshold: int) -> dict:
    body = content.strip()
    whole = f"{reasoning}\n{content}"
    no_answer = not body
    flags = {
        "no_answer": no_answer,
        "answer_wrong": bool(body) and any(
            r.lower() not in body.lower() for r in prompt["require_all"]
        ),
        "eos_in_think": no_answer and finish == "stop" and bool(reasoning.strip()),
        "digit_noise": prompt["digit_free"] and has_digit_noise(whole),
        "cjk_leak": bool(_CJK.search(whole)),
    }
    count, gram = max_ngram_repeat(whole)
    flags["loop"] = count >= loop_threshold
    return flags, {"max_10gram_count": count, "max_10gram": gram[:120]}


def run_chat(base_url: str, model: str, prompt: dict, max_tokens: int,
             reasoning_max_tokens: int | None, timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt["text"]}],
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
        "top_k": 1,
    }
    if reasoning_max_tokens:
        body["reasoning_max_tokens"] = reasoning_max_tokens
    t0 = time.time()
    r = _post(base_url.rstrip("/") + "/chat/completions", body, timeout)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    return {
        "lane": "chat",
        "latency_s": round(time.time() - t0, 2),
        "finish_reason": ch.get("finish_reason"),
        "reasoning": msg.get("reasoning_content") or "",
        "content": msg.get("content") or "",
        "usage": r.get("usage") or {},
    }


def run_raw(base_url: str, model: str, prompt: dict, max_tokens: int, timeout: float) -> dict:
    """Control lane: same question as plain text, no chat template and no <think> span. A healthy
    serve answers here even when the chat lane delivers nothing."""
    body = {
        "model": model,
        "prompt": prompt["text"] + "\nAnswer: ",
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
        "top_k": 1,
    }
    t0 = time.time()
    r = _post(base_url.rstrip("/") + "/completions", body, timeout)
    ch = r["choices"][0]
    return {
        "lane": "raw",
        "latency_s": round(time.time() - t0, 2),
        "finish_reason": ch.get("finish_reason"),
        "reasoning": "",
        "content": ch.get("text") or "",
        "usage": r.get("usage") or {},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:1919/v1")
    ap.add_argument("--expect-model", required=True,
                    help="checked against /v1/models; a mismatch aborts before any probe")
    ap.add_argument("--container", default=None,
                    help="serve container name, for image/boot/engine-sha provenance")
    ap.add_argument("--arm", required=True, help="arm label, e.g. before-reboot / after-reboot")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--reasoning-max-tokens", type=int, default=None,
                    help="arm ThinkGate's EOS hold + beta backstop (unbounded by default on this "
                         "checkpoint family); omit to measure the shipped behaviour")
    ap.add_argument("--lanes", default="chat,raw")
    ap.add_argument("--loop-threshold", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", required=True, help="fixture root (a real directory, never tmpfs)")
    args = ap.parse_args()

    lanes = [l.strip() for l in args.lanes.split(",") if l.strip()]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(args.out, f"{stamp}-{args.arm}")
    os.makedirs(outdir, exist_ok=True)

    prov = {
        "arm": args.arm,
        "utc": stamp,
        "base_url": args.base_url,
        "expect_model": args.expect_model,
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "lanes": lanes,
        "endpoint": assert_model(args.base_url, args.expect_model, args.timeout),
        "serve": container_provenance(args.container),
    }
    print(f"[provenance] {json.dumps(prov['serve'])}")

    turns: list[dict] = []
    for prompt in PROMPTS:
        for lane in lanes:
            for rep in range(args.reps):
                try:
                    if lane == "chat":
                        res = run_chat(args.base_url, args.expect_model, prompt,
                                       args.max_tokens, args.reasoning_max_tokens, args.timeout)
                    else:
                        res = run_raw(args.base_url, args.expect_model, prompt,
                                      args.max_tokens, args.timeout)
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    res = {"lane": lane, "error": repr(exc)[:300], "reasoning": "",
                           "content": "", "finish_reason": None, "usage": {}}
                flags, extra = score(prompt, res["reasoning"], res["content"],
                                     res.get("finish_reason"), args.loop_threshold)
                turn = {"prompt_id": prompt["id"], "rep": rep, **res, **extra, "flags": flags}
                turns.append(turn)
                mark = "".join(k[0].upper() for k, v in flags.items() if v) or "-"
                print(f"  {prompt['id']:9s} {lane:4s} rep{rep} "
                      f"{str(res.get('finish_reason')):6s} "
                      f"ctok={res.get('usage', {}).get('completion_tokens', '?'):>4} [{mark}]")

    # Rates per lane, and per (lane, prompt) so a single bad prompt cannot hide behind an average.
    summary: dict = {"by_lane": {}, "by_lane_prompt": {}}
    for lane in lanes:
        rows = [t for t in turns if t["lane"] == lane]
        summary["by_lane"][lane] = {
            "n": len(rows),
            **{k: sum(1 for t in rows if t["flags"][k]) for k in rows[0]["flags"]},
        }
        for prompt in PROMPTS:
            sub = [t for t in rows if t["prompt_id"] == prompt["id"]]
            if sub:
                summary["by_lane_prompt"][f"{lane}/{prompt['id']}"] = {
                    "n": len(sub),
                    **{k: sum(1 for t in sub if t["flags"][k]) for k in sub[0]["flags"]},
                }

    with open(os.path.join(outdir, "result.json"), "w") as fh:
        json.dump({"provenance": prov, "summary": summary, "turns": turns}, fh, indent=1)
    # Full text, always — a count that looks clean has been wrong before.
    with open(os.path.join(outdir, "turns.txt"), "w") as fh:
        for t in turns:
            fh.write(f"===== {t['prompt_id']} / {t['lane']} / rep{t['rep']} "
                     f"finish={t.get('finish_reason')} flags="
                     f"{[k for k, v in t['flags'].items() if v]}\n")
            fh.write(f"--- reasoning ---\n{t['reasoning']}\n--- content ---\n{t['content']}\n\n")

    print(f"\n[{args.arm}] {json.dumps(summary['by_lane'], indent=1)}")
    print(f"fixture: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
