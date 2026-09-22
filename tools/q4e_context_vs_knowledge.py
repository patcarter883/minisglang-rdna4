#!/usr/bin/env python3
"""KNOWLEDGE vs CONTEXT RETRIEVAL on a worked q4e serve — which one actually breaks?

THE HYPOTHESIS (Pat, 2026-09-22). The degradation is not general: the model answers fine from its own
parameters and fails when it must RETRIEVE SOMETHING IT WAS GIVEN. The evidence that prompted it, all
from the same degraded serve:

    knowledge   "What is 17 + 26?"                      -> 43                      CORRECT
    knowledge   "capital cities of France, Japan, ..."  -> Paris/Tokyo/Rome/...    CORRECT
    knowledge   raw "The capital city of Italy is"      -> Rome, Cairo, Ottawa...  CORRECT
    retrieval   "Repeat this exactly: the quick brown fox ..."
                                                        -> "Are all objects non-homogeneous?"
    retrieval   "@docs/ui-plan @ui"                     -> docs/22909-74486-01

If that split is real and widens with load, the defect is in the machinery that reads the PROMPT —
attention/KV (fp8 KV, page reuse, the radix tree), the PLE n-gram path, or the 36 GDN recurrent
layers — and NOT in the weights, the sampler, or the chat template. That is a large narrowing, so it
deserves a direct measurement rather than an argument from five anecdotes.

DESIGN. Two arms, same serve, same dose point, interleaved so drift cannot favour one:

  * K (knowledge)  — answerable with no context at all. The CONTROL. If K degrades too, the defect is
                     general and this whole framing is wrong.
  * R (retrieval)  — the answer is planted IN the prompt and exists nowhere else: a random code the
                     model cannot know. Scored by exact match of that code.

R is swept over NEEDLE DEPTH (fraction through the filler) and CONTEXT LENGTH, because those
discriminate mechanisms that a single retrieval number cannot: KV-page or radix corruption should be
depth- and length-dependent, whereas a global numerical fault should hit every depth equally.

A third arm, R-ident, plants real identifiers and demands them back verbatim — the literal
`@docs/ui-plan -> docs/22909-74486-01` failure, made countable.

X-AXIS IS CUMULATIVE PREFILL, NOT UPTIME. Prometheus on the degraded boot: all 3.26M prompt tokens
were processed 17:59-21:00, the serve was then COMPLETELY IDLE until the 23:02 failure, and the first
empty completion landed at 18:20 with 680k prompt tokens on the clock. Work done, not time elapsed —
so this tool DRIVES load between dose points instead of waiting on a clock.

CPU-only HTTP client; it holds no lease but it does occupy the serve it measures.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

# ---- K arm: answerable from parameters alone. Kept trivially checkable and unambiguous.
KNOWLEDGE = [
    ("What is 17 + 26? Reply with just the number.", ["43"]),
    ("What is the capital city of Italy? Reply with just the city name.", ["rome"]),
    ("What is the chemical symbol for gold? Reply with just the symbol.", ["au"]),
    ("How many days are in a leap year? Reply with just the number.", ["366"]),
    ("What is the capital city of Japan? Reply with just the city name.", ["tokyo"]),
]

# ---- R-ident arm: the shapes the real failure chewed on.
PLANTED = [
    "@docs/ui-plan",
    "core/src/mc_velocity.c",
    "mc_velocity_update",
    "tests/test_velocity.c",
    "ui/panels/ConnectionPanel.tsx",
]
_ID_SHAPED = re.compile(r"[@\w][\w./@-]{5,}")


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def ask(base_url: str, model: str, content: str, max_tokens: int, timeout: float) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "stream": False, "temperature": 0.0, "top_k": 1}
    t0 = time.time()
    r = _post(base_url.rstrip("/") + "/chat/completions", body, timeout)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    return {"latency_s": round(time.time() - t0, 2), "finish_reason": ch.get("finish_reason"),
            "reasoning": msg.get("reasoning_content") or "", "content": msg.get("content") or "",
            "usage": r.get("usage") or {}}


_DOSE_METRICS = {
    # THE DOSE AXIS IS *COMPUTED* PREFILL, NOT SUBMITTED. On the degraded boot
    # `minisgl_prompt_tokens_total` read 3,263,790 while
    # `minisgl_prefill_computed_tokens_total` read only 731,822 — the prefix cache absorbed 4.5x of
    # it, doing no work. Driving to a SUBMITTED target would mostly serve cache hits and never reach
    # the state that breaks, so the load loop and the x-axis both use computed tokens. Submitted and
    # generated are recorded alongside, because which of the three actually drives the defect is the
    # open question and throwing two of them away would decide it by accident.
    "computed": "minisgl_prefill_computed_tokens_total",
    "submitted": "minisgl_prompt_tokens_total",
    "generated": "minisgl_generation_tokens_total",
    "requests": "minisgl_requests_total",
}


def dose_counters(base_url: str) -> dict:
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=15) as fh:
            body = fh.read().decode()
    except Exception:
        return {k: -1 for k in _DOSE_METRICS}
    out = {}
    for key, name in _DOSE_METRICS.items():
        m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}}\s+(\S+)$", body, re.M)
        out[key] = int(float(m.group(1))) if m else -1
    return out


def prompt_tokens(base_url: str) -> int:
    """The dose reading: COMPUTED prefill tokens (see _DOSE_METRICS)."""
    return dose_counters(base_url)["computed"]


def filler_lines(n: int, seed: int) -> list[str]:
    """Neutral, non-repeating filler. Numbered prose lines rather than source code, so the needle is
    not competing with real identifiers and a retrieval miss cannot be blamed on distractors."""
    rng = random.Random(seed)
    topics = ["harbour", "lantern", "meadow", "compass", "quarry", "thistle", "ferry", "almanac",
              "orchard", "beacon", "trellis", "cistern", "pelican", "sundial", "willow"]
    return [f"Note {i:05d}: the {rng.choice(topics)} was inspected and recorded without incident."
            for i in range(n)]


def make_needle(depth: float, lines: int, seed: int) -> tuple[str, str]:
    """A prompt whose answer exists ONLY in the prompt. Returns (prompt, expected code)."""
    rng = random.Random(seed)
    code = (f"{rng.randint(1000, 9999)}-"
            + "".join(rng.choices(string.ascii_uppercase, k=3))
            + f"-{rng.randint(1000, 9999)}")
    body = filler_lines(lines, seed)
    at = min(len(body) - 1, max(0, int(len(body) * depth)))
    body[at] = f"Note {at:05d}: the authorisation code for the east gate is {code} — record it."
    return ("\n".join(body)
            + "\n\nWhat is the authorisation code for the east gate? "
              "Reply with just the code, exactly as written."), code


def score_knowledge(reply: str, want: list[str]) -> dict:
    low = (reply or "").lower()
    return {"no_answer": not (reply or "").strip(),
            "wrong": bool((reply or "").strip()) and not any(w in low for w in want)}


def score_needle(reply: str, code: str) -> dict:
    body = (reply or "").strip()
    return {"no_answer": not body,
            "wrong": bool(body) and code not in body,
            # A near-miss — right shape, wrong characters — is the corruption signature rather than a
            # plain failure to find it, so count it separately.
            "corrupted": bool(body) and code not in body
                         and bool(re.search(r"\d{3,4}-[A-Z]{2,4}-\d{3,4}", body))}


def score_ident(reply: str) -> tuple[dict, dict]:
    body = reply or ""
    low = body.lower()
    altered, missing = [], []
    for ident in PLANTED:
        if ident.lower() in low:
            continue
        stem = re.split(r"[./@]", ident.strip("@"))[-1].split(".")[0].lower()
        (altered if stem and stem in low else missing).append(ident)
    exact = {p.lower() for p in PLANTED}
    # An invented identifier is one that is NOT a planted identifier yet carries a 3+ digit run.
    # Earlier version excluded any token merely CONTAINING a known component, which threw away the
    # headline case: `docs/22909-74486-01` contains `docs`, so the real failure scored as clean.
    invented = [t for t in _ID_SHAPED.findall(body)
                if t.lower() not in exact and re.search(r"\d{3,}", t)]
    return ({"no_answer": not body.strip(), "id_altered": bool(altered),
             "id_missing": bool(missing), "id_invented": bool(invented)},
            {"altered": altered, "missing": missing, "invented": invented[:8]})


def load_corpus(repo: str, approx_chars: int) -> str:
    parts, total = [], 0
    for root, _d, files in os.walk(os.path.join(repo, "python", "minisgl")):
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            try:
                parts.append(open(os.path.join(root, f), encoding="utf-8", errors="replace").read())
            except OSError:
                continue
            total += len(parts[-1])
            if total >= approx_chars:
                return "".join(parts)[:approx_chars]
    return "".join(parts)[:approx_chars]


def drive_load(base_url: str, model: str, corpus: str, target: int, chunk: int,
               timeout: float) -> int:
    start = prompt_tokens(base_url)
    sent = 0
    while True:
        now = prompt_tokens(base_url)
        if now < 0 or now - start >= target:
            return now
        off = (sent * chunk) % max(1, len(corpus) - chunk)
        try:
            _post(base_url.rstrip("/") + "/chat/completions",
                  # max_tokens 256, not 32: the degraded boot accumulated ~101k GENERATED tokens
                  # alongside its 732k computed prefill, and which of the two drives the defect is
                  # unknown. A prefill-only load would silently fail to reproduce a decode-driven
                  # fault and the null result would look like evidence against the hypothesis.
                  {"model": model, "max_tokens": 256, "stream": False, "temperature": 0.0,
                   "top_k": 1, "messages": [{"role": "user", "content":
                                             "Summarise in one word.\n\n" + corpus[off:off + chunk]}]},
                  timeout)
        except Exception as exc:
            sys.stderr.write(f"  load failed (continuing): {repr(exc)[:110]}\n")
        sent += 1
        if sent % 5 == 0:
            print(f"  [load] {sent} reqs, prompt_tokens {now:,} (+{now - start:,})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:1919/v1")
    ap.add_argument("--expect-model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3, help="reps per cell per dose point")
    ap.add_argument("--dose", type=int, default=125_000,
                    help="COMPUTED prefill tokens to add between dose points; the degraded boot\n                         reached 731,822 computed in total, and its first empty completion at 189,627")
    ap.add_argument("--points", type=int, default=7)
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--lines", default="120,900", help="filler lines -> short/long context")
    ap.add_argument("--chunk-chars", type=int, default=90_000)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()

    with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=30) as fh:
        ids = [m.get("id") for m in json.load(fh).get("data") or []]
    if args.expect_model not in ids:
        sys.exit(f"provenance FAIL: /v1/models lists {ids}, expected {args.expect_model!r}")

    depths = [float(d) for d in args.depths.split(",")]
    lines_opts = [int(x) for x in args.lines.split(",")]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(args.out, stamp)
    os.makedirs(outdir, exist_ok=True)
    corpus = load_corpus(args.repo, args.chunk_chars * 12)

    tsv = os.path.join(outdir, "dose.tsv")
    cols = ["point", "computed", "submitted", "generated", "requests",
            "K_n", "K_wrong", "K_empty",
            "R_n", "R_wrong", "R_corrupted", "R_empty", "Rident_n", "Rident_altered",
            "Rident_invented", "Rident_missing"]
    with open(tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")

    turns: list[dict] = []
    seed = 0
    for point in range(args.points):
        if point:
            print(f"[dose {point}] +{args.dose:,} prompt tokens...", flush=True)
            drive_load(args.base_url, args.expect_model, corpus, args.dose,
                       args.chunk_chars, args.timeout)
        ctr = dose_counters(args.base_url)
        pt = ctr["computed"]
        K = {"n": 0, "wrong": 0, "empty": 0}
        R = {"n": 0, "wrong": 0, "corrupted": 0, "empty": 0}
        I = {"n": 0, "altered": 0, "invented": 0, "missing": 0}
        for rep in range(args.reps):
            # K and R interleaved, so any within-point drift cannot favour one arm.
            for q, want in KNOWLEDGE:
                try:
                    res = ask(args.base_url, args.expect_model, q, 300, args.timeout)
                except Exception as exc:
                    sys.stderr.write(f"  K failed: {repr(exc)[:110]}\n"); continue
                f = score_knowledge(res["content"], want)
                K["n"] += 1; K["wrong"] += f["wrong"]; K["empty"] += f["no_answer"]
                turns.append({"arm": "K", "point": point, "prompt_tokens": pt, "q": q,
                              "want": want, **res, "flags": f})
            for nlines in lines_opts:
                for depth in depths:
                    seed += 1
                    prompt, code = make_needle(depth, nlines, seed)
                    try:
                        res = ask(args.base_url, args.expect_model, prompt,
                                  args.max_tokens, args.timeout)
                    except Exception as exc:
                        sys.stderr.write(f"  R failed: {repr(exc)[:110]}\n"); continue
                    f = score_needle(res["content"], code)
                    R["n"] += 1; R["wrong"] += f["wrong"]
                    R["corrupted"] += f["corrupted"]; R["empty"] += f["no_answer"]
                    turns.append({"arm": "R", "point": point, "prompt_tokens": pt,
                                  "depth": depth, "lines": nlines, "code": code, **res, "flags": f})
            listing = "\n".join(f"- {p}" for p in PLANTED)
            try:
                res = ask(args.base_url, args.expect_model,
                          "These files are under review:\n" + listing +
                          "\n\nList every file path above, exactly as written, one per line. "
                          "Do not add, renumber, abbreviate or invent anything.",
                          args.max_tokens, args.timeout)
                f, detail = score_ident(res["content"])
                I["n"] += 1; I["altered"] += f["id_altered"]
                I["invented"] += f["id_invented"]; I["missing"] += f["id_missing"]
                turns.append({"arm": "Rident", "point": point, "prompt_tokens": pt,
                              **res, **detail, "flags": f})
            except Exception as exc:
                sys.stderr.write(f"  Rident failed: {repr(exc)[:110]}\n")
            print(f"  p{point} rep{rep} K_wrong={K['wrong']}/{K['n']} "
                  f"R_wrong={R['wrong']}/{R['n']} Rident_alt={I['altered']}/{I['n']}", flush=True)
        with open(tsv, "a") as fh:
            fh.write("\t".join(str(x) for x in [
                point, ctr["computed"], ctr["submitted"], ctr["generated"], ctr["requests"],
                K["n"], K["wrong"], K["empty"],
                R["n"], R["wrong"], R["corrupted"], R["empty"],
                I["n"], I["altered"], I["invented"], I["missing"]]) + "\n")
        with open(os.path.join(outdir, "result.json"), "w") as fh:
            json.dump({"args": vars(args), "turns": turns}, fh, indent=1)
        with open(os.path.join(outdir, "turns.txt"), "w") as fh:
            for t in turns:
                fh.write(f"===== {t['arm']} point{t['point']} pt={t['prompt_tokens']} "
                         f"flags={[k for k, v in t['flags'].items() if v]}\n"
                         f"--- content ---\n{t['content']}\n\n")
        print(f"[dose {point}] pt={pt:,} K_wrong={K['wrong']}/{K['n']} "
              f"R_wrong={R['wrong']}/{R['n']} R_corrupted={R['corrupted']} "
              f"Rident_altered={I['altered']}/{I['n']} invented={I['invented']}", flush=True)

    print(f"\nfixture: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
