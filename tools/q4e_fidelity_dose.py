#!/usr/bin/env python3
"""IDENTIFIER FIDELITY vs CUMULATIVE PREFILL — does a worked q4e serve start getting names wrong?

THE DEFECT THIS MEASURES. Hermes session 5d26424a4be3 asked for a UI review of `@docs/ui-plan @ui`
and the model rendered those as `docs/22909-74486-01` and `ui-0000446015-000046`. That is the failure
that matters: not an empty turn, but a turn whose CONTENT IS WRONG — a real identifier decorated with
invented digits, reported with total confidence. Same shape as the 2026-09-21 journal §5 tool-arg
confabulation (`uuid::name`, `_40hex_NNN`, `head 161 166`).

WHY THE EARLIER PROBE WAS BLIND TO IT. `answer_delivery_probe.py` asks "list the capital cities of
France, Japan…" — a prompt with NO identifiers in it, so there is nothing for the model to conflate
and the detector fired once in 30 turns on a serve that was catastrophically broken. Presence of a
detector is not coverage: a fidelity defect needs a prompt that CARRIES identifiers, and a score that
counts how many come back altered. Every prompt here is built around that.

WHY DOSE-RESPONSE AGAINST PREFILL, NOT UPTIME. The 2026-09-22 reboot A/B showed a bounce fixes the
defect, and I first read that as an uptime effect. It is not — or at least the evidence cannot say so,
because across a reboot uptime and cumulative work reset together. Prometheus settles it for the
degraded boot: all 3.26M prompt tokens were processed between 17:59 and 21:00, the serve was then
COMPLETELY IDLE from 21:00 to the 23:02 failure (`minisgl_prompt_tokens_total` flat), and the first
empty completion appeared at 18:20 with 680k prompt tokens on the clock — 21 minutes in. Whatever
accumulates, it accumulates with WORK DONE, and two idle hours added nothing. So the x-axis is
cumulative prompt tokens, and this tool DRIVES that axis rather than waiting on a clock (a 30-minute
tick of short prompts would need ~98 days to reach 3.3M — the instrument would have read "healthy"
forever and confirmed the wrong hypothesis).

    load burst -> probe -> row, keyed on minisgl_prompt_tokens_total.

Detectors, all countable, per probe reply:

  * id_altered   — an identifier from the prompt that came back MODIFIED (the headline defect:
                   `docs/ui-plan` -> `docs/22909-74486-01`). Matched by stem, so a decorated form is
                   attributed to the identifier it corrupted rather than counted as a miss.
  * id_missing   — an identifier the reply never mentions at all. Separate from altered: dropping a
                   name is a different failure from inventing one, and conflating them would let a
                   truncated reply score as corruption.
  * id_invented  — an identifier-shaped token in the reply matching NOTHING in the prompt.
  * canary_leak  — text from an EARLIER request in this run appearing in this reply. The
                   cross-request-contamination test: journal §1 ruled out CONCURRENT injection via
                   `requests_inflight` max 1, which says nothing about a stale page reused after a
                   request finished. Unique per request, so a hit is unambiguous.
  * no_answer    — empty content, carried over so the delivery defect stays visible too.

Full text of every reply is written to the fixture dir. A rate that looks clean has been wrong before;
read the text.

CPU-only HTTP client: no GPU lease, no container. The serve under test holds the lease. It does drive
real load, so it is NOT free — it occupies the serve it measures.
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

# Identifiers the probe plants and then demands back verbatim. Deliberately the shapes the real
# failure chewed on: an @-mention, a dotted path, a C source path, a snake_case symbol, a short hex.
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


def prompt_tokens(base_url: str) -> int:
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=15) as fh:
            body = fh.read().decode()
    except Exception:
        return -1
    m = re.search(r"^minisgl_prompt_tokens_total\{[^}]*\}\s+(\S+)$", body, re.M)
    return int(float(m.group(1))) if m else -1


def load_corpus(repo: str, approx_chars: int) -> str:
    """Realistic filler: this repo's own source. Real code at a realistic size, reproducible, and
    nothing like the probe prompts, so the load cannot itself seed a probe answer."""
    parts, total = [], 0
    for root, _dirs, files in os.walk(os.path.join(repo, "python", "minisgl")):
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
    return "".join(parts)[:approx_chars] if parts else ("filler. " * (approx_chars // 8))


def drive_load(base_url: str, model: str, corpus: str, target_delta: int,
               chunk_chars: int, timeout: float, verbose: bool = True) -> int:
    """Push prompt tokens until the serve's counter has risen by ``target_delta``. Mirrors the real
    traffic shape that produced the failure: large prefill, small generation."""
    start = prompt_tokens(base_url)
    sent = 0
    while True:
        now = prompt_tokens(base_url)
        if now < 0 or now - start >= target_delta:
            return now
        off = (sent * chunk_chars) % max(1, len(corpus) - chunk_chars)
        body = {
            "model": model,
            "messages": [{"role": "user", "content":
                          "Summarise in one word what this code does.\n\n"
                          + corpus[off:off + chunk_chars]}],
            "max_tokens": 32, "stream": False, "temperature": 0.0, "top_k": 1,
        }
        try:
            _post(base_url.rstrip("/") + "/chat/completions", body, timeout)
        except Exception as exc:
            sys.stderr.write(f"  load request failed (continuing): {repr(exc)[:120]}\n")
        sent += 1
        if verbose and sent % 5 == 0:
            print(f"  [load] {sent} requests, prompt_tokens {now:,} (+{now - start:,})", flush=True)


def score(reply: str, canaries: list[str]) -> tuple[dict, dict]:
    body = reply or ""
    low = body.lower()
    altered, missing = [], []
    for ident in PLANTED:
        if ident.lower() in low:
            continue
        # Stem = the distinctive word inside the identifier. Present-but-not-verbatim means the model
        # reproduced the name in a CORRUPTED form, which is the defect; absent entirely is a miss.
        stem = re.split(r"[./@]", ident.strip("@"))[-1].split(".")[0].lower()
        (altered if stem and stem in low else missing).append(ident)
    known = {p.lower() for p in PLANTED}
    known |= {w.lower() for p in PLANTED for w in re.split(r"[./@-]", p) if w}
    invented = [t for t in _ID_SHAPED.findall(body)
                if t.lower() not in known
                and not any(t.lower() in k or k in t.lower() for k in known)
                and re.search(r"\d{3,}", t)]
    leak = [c for c in canaries if c in body]
    flags = {
        "no_answer": not body.strip(),
        "id_altered": bool(altered),
        "id_missing": bool(missing),
        "id_invented": bool(invented),
        "canary_leak": bool(leak),
    }
    return flags, {"altered": altered, "missing": missing,
                   "invented": invented[:8], "leak": leak}


def probe_once(base_url: str, model: str, canary: str, max_tokens: int,
               timeout: float) -> dict:
    listing = "\n".join(f"- {p}" for p in PLANTED)
    content = (
        f"Reference tag {canary}.\n\n"
        "These files are under review:\n" + listing + "\n\n"
        "List every file path from the list above, exactly as written, one per line. "
        "Do not add, renumber, abbreviate or invent anything."
    )
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "stream": False, "temperature": 0.0, "top_k": 1}
    t0 = time.time()
    r = _post(base_url.rstrip("/") + "/chat/completions", body, timeout)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    return {"latency_s": round(time.time() - t0, 2), "finish_reason": ch.get("finish_reason"),
            "reasoning": msg.get("reasoning_content") or "", "content": msg.get("content") or "",
            "usage": r.get("usage") or {}, "canary": canary}


def serve_provenance(container: str, base_url: str) -> dict:
    if container == "auto":
        port = (re.search(r":(\d+)", base_url) or [None, "1919"])[1]
        out = subprocess.run(["docker", "ps", "--filter", f"publish={port}", "--format",
                              "{{.Names}}"], capture_output=True, text=True)
        names = [n for n in out.stdout.split() if n]
        if len(names) != 1:
            sys.exit(f"--container auto: expected one container on {port}, got {names}")
        container = names[0]
    info = {"container": container}
    try:
        fmt = "{{.Config.Image}}\t{{.State.StartedAt}}"
        img, started = subprocess.run(["docker", "inspect", "--format", fmt, container],
                                      capture_output=True, text=True, check=True
                                      ).stdout.strip().split("\t")
        info.update(image=img, started_at=started)
        t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
        info["uptime_s"] = int((datetime.now(timezone.utc) - t0).total_seconds())
    except Exception as exc:
        info["provenance_error"] = repr(exc)[:200]
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:1919/v1")
    ap.add_argument("--expect-model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--container", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=6, help="fidelity probes per dose point")
    ap.add_argument("--dose", type=int, default=400_000,
                    help="prompt tokens of load to add between dose points")
    ap.add_argument("--points", type=int, default=9,
                    help="dose points (point 0 is the unloaded baseline)")
    ap.add_argument("--chunk-chars", type=int, default=90_000,
                    help="approx chars per load request (~30k tokens, the real traffic shape)")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()

    with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=30) as fh:
        ids = [m.get("id") for m in json.load(fh).get("data") or []]
    if args.expect_model not in ids:
        sys.exit(f"provenance FAIL: /v1/models lists {ids}, expected {args.expect_model!r}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(args.out, stamp)
    os.makedirs(outdir, exist_ok=True)
    prov = serve_provenance(args.container, args.base_url)
    print(f"[provenance] {json.dumps(prov)}")
    corpus = load_corpus(args.repo, args.chunk_chars * 12)
    print(f"[corpus] {len(corpus):,} chars")

    tsv = os.path.join(outdir, "dose.tsv")
    cols = ["point", "prompt_tokens", "uptime_s", "n", "no_answer", "id_altered", "id_missing",
            "id_invented", "canary_leak"]
    with open(tsv, "w") as fh:
        fh.write("\t".join(cols) + "\n")

    canaries: list[str] = []
    all_turns: list[dict] = []
    for point in range(args.points):
        if point:
            print(f"[dose {point}] driving +{args.dose:,} prompt tokens...", flush=True)
            drive_load(args.base_url, args.expect_model, corpus, args.dose,
                       args.chunk_chars, args.timeout)
        pt = prompt_tokens(args.base_url)
        up = serve_provenance(prov["container"], args.base_url).get("uptime_s", "")
        rows = []
        for rep in range(args.reps):
            canary = "CANARY-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=10))
            try:
                res = probe_once(args.base_url, args.expect_model, canary,
                                 args.max_tokens, args.timeout)
            except Exception as exc:
                sys.stderr.write(f"  probe failed: {repr(exc)[:160]}\n")
                continue
            flags, detail = score(res["content"], canaries)
            canaries.append(canary)
            rows.append(flags)
            all_turns.append({"point": point, "prompt_tokens": pt, "rep": rep,
                              **res, **detail, "flags": flags})
            mark = ",".join(k for k, v in flags.items() if v) or "clean"
            print(f"  p{point} rep{rep} pt={pt:,} {mark}", flush=True)
        if not rows:
            continue
        agg = {k: sum(1 for r in rows if r[k]) for k in rows[0]}
        with open(tsv, "a") as fh:
            fh.write("\t".join(str(x) for x in
                               [point, pt, up, len(rows)] + [agg[k] for k in cols[4:]]) + "\n")
        with open(os.path.join(outdir, "result.json"), "w") as fh:
            json.dump({"provenance": prov, "args": vars(args), "turns": all_turns}, fh, indent=1)
        with open(os.path.join(outdir, "turns.txt"), "w") as fh:
            for t in all_turns:
                fh.write(f"===== point{t['point']} rep{t['rep']} pt={t['prompt_tokens']} "
                         f"flags={[k for k, v in t['flags'].items() if v]}\n"
                         f"altered={t['altered']} missing={t['missing']} invented={t['invented']}\n"
                         f"--- reasoning ---\n{t['reasoning']}\n--- content ---\n{t['content']}\n\n")
        print(f"[dose {point}] pt={pt:,} " + " ".join(f"{k}={agg[k]}/{len(rows)}" for k in cols[4:]),
              flush=True)

    print(f"\nfixture: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
