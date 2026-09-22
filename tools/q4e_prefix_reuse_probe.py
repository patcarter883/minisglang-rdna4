#!/usr/bin/env python3
"""PREFIX-REUSING retrieval probe — the load shape that actually broke the serve.

WHY THIS EXISTS. `q4e_context_vs_knowledge.py` drove 265,172 computed prefill tokens of large
INDEPENDENT prompts into a fresh serve, past the 189,627 at which real traffic first failed, and
every check stayed clean. The dose axis was not the point. The real traffic ran **3,263,790 submitted
against 731,822 computed — 78% prefix-cache hits**: many conversations, each resuming from a deep
shared prefix, turn after turn. The synthetic load rotated corpus offsets specifically to AVOID reuse,
so its requests never resumed from cached state, and the one mechanism the failing session's log
shows firing in the minute before it broke was never exercised at all:

    recurrent-radix HIT: uid=94 restored recurrent state at cached_len=64

On q4e that path is composite — the boot warns "snapshotting GDN + PLE state together (shared slot
space); a prefix hit restores both or neither". So a prefix hit does not merely skip attention
prefill: it RESTORES the recurrent state, GDN's SSM and PLE's conv window and n-gram history, from a
snapshot. If a restore ever delivers state that does not match the tokens the request actually has,
the model's representation of its own context is wrong while its weights are untouched — fluent,
confident, and wrong about what it was shown. That is the signature under investigation.

WHAT THIS DRIVES THAT THE OTHER PROBE DID NOT:

  * `--sessions N` distinct conversations, each with its own long shared prefix, round-robined so
    every turn after the first HITS that prefix;
  * N deliberately exceeds the snapshot store's frame count (the serve boots with `frames=22`), so
    frames churn and `drops` rises — `alloc` returning None is a drop, and the claim under test is
    that a drop degrades to a safe re-prefill rather than to a mismatched restore;
  * the needle lives IN THE SHARED PREFIX, not in the per-turn tail. That is the whole point: the
    tokens carrying the answer are exactly the tokens a prefix hit does not recompute, so a bad
    restore loses them while a correct one cannot.

SCORING is retrieval of a code that exists ONLY in that session's prefix, plus a cross-session
CONTAMINATION check: every session's code is distinct, so another session's code appearing in this
one's answer is unambiguous evidence of state crossing between sequences — which no amount of
"the model was confused" can explain.

A KNOWLEDGE CONTROL runs alongside at every checkpoint. If knowledge degrades too, the defect is not
retrieval-specific and this framing is wrong; say so rather than reporting only the retrieval number.

THREE FALSE-POSITIVE MECHANISMS, all of which fired here before they were fixed. Every "retrieval
failure" this probe produced on 2026-09-22 turned out to be one of them; none was the model.

1. **max_tokens too small.** This is a THINKING model: it spends the budget reasoning and the answer
   is cut MID-CODE. A run at `--max-tokens 120` scored 26/64 failures whose outputs were
   `wants 1994-AUC-2417 got '1994-AUC-24'`, `wants 4863-XTE-6133 got '4863-XTE-'` — each one PROOF
   that retrieval worked. `finish_reason == "length"` is now scored as `truncated` and excluded.
2. **reasoning_max_tokens too small.** The fix for (1) introduced this. Capping the think span makes
   the beta backstop force `</think>` mid-sentence, and the fragment the model had typed so far
   becomes the answer — with `finish_reason == "stop"`, so the truncation check in (1) does NOT
   catch it. Observed: reasoning containing `The code is "5632-FMT-8131".` twice, then
   `So I will output: 5632` force-closed at 126 tokens, `content: '5632'`. Default is now 0.
3. **Reading the wrong fixture directory.** `ls -dt | head -1` picked a KILLED run's directory and
   reported "0 turns recorded" while the live run was writing rows normally, which looked like a
   hung serve and prompted a wedge investigation. Check the path the run actually printed.

A corollary worth keeping: the model's REASONING is the ground truth for whether retrieval happened.
When content looks wrong, read `reasoning_content` before scoring a defect — in every case above the
reasoning held the correct answer verbatim.

CPU-only HTTP client. It holds no lease but it fully occupies the serve it measures.
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

KNOWLEDGE = [
    ("What is 17 + 26? Reply with just the number.", "43"),
    ("What is the capital city of Italy? Reply with just the city name.", "rome"),
    ("What is the chemical symbol for gold? Reply with just the symbol.", "au"),
]


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def ask(base_url: str, model: str, messages: list, max_tokens: int, timeout: float,
        reasoning_max_tokens: int | None = None) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "stream": False, "temperature": 0.0, "top_k": 1}
    if reasoning_max_tokens:
        # Cap the THINK span, not the reply. This is a thinking model: with a small `max_tokens` it
        # spends the whole budget reasoning and the answer is truncated MID-CODE, which scores as a
        # retrieval failure while actually proving retrieval WORKED (`wants 1994-AUC-2417 got
        # 1994-AUC-24`). A run at max_tokens=120 produced 26/64 such artifacts before this existed.
        body["reasoning_max_tokens"] = reasoning_max_tokens
    t0 = time.time()
    r = _post(base_url.rstrip("/") + "/chat/completions", body, timeout)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    return {"latency_s": round(time.time() - t0, 2), "finish_reason": ch.get("finish_reason"),
            "reasoning": msg.get("reasoning_content") or "", "content": msg.get("content") or "",
            "usage": r.get("usage") or {}}


def counters(base_url: str) -> dict:
    root = base_url.rstrip("/").removesuffix("/v1")
    out = {}
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=15) as fh:
            body = fh.read().decode()
    except Exception:
        return out
    for key, name in (("computed", "minisgl_prefill_computed_tokens_total"),
                      ("submitted", "minisgl_prompt_tokens_total"),
                      ("hit_tokens", "minisgl_prefix_cache_hit_tokens_total"),
                      ("hit_ratio", "minisgl_prefix_cache_hit_ratio"),
                      ("requests", "minisgl_requests_total")):
        m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}}\s+(\S+)$", body, re.M)
        if m:
            out[key] = float(m.group(1))
    return out


def snapshot_stats(container: str) -> dict:
    """Frame-store pressure, off the serve log. `drops` is the number under test: a drop must mean
    'no snapshot stored' (safe re-prefill), never a mismatched restore."""
    try:
        txt = subprocess.run(["docker", "logs", "--tail", "3000", container],
                             capture_output=True, text=True, timeout=60)
        lines = [l for l in (txt.stdout + txt.stderr).splitlines()
                 if "recurrent-radix snapshot store" in l]
    except Exception:
        return {}
    if not lines:
        return {}
    last = lines[-1]
    out = {}
    for k in ("frames", "in_use", "high_water", "drops"):
        m = re.search(rf"\b{k}=(\d+)", last)
        if m:
            out[k] = int(m.group(1))
    return out


def rec_radix_hits(container: str) -> int:
    try:
        txt = subprocess.run(["docker", "logs", "--tail", "6000", container],
                             capture_output=True, text=True, timeout=60)
        return sum(1 for l in (txt.stdout + txt.stderr).splitlines()
                   if "recurrent-radix HIT" in l)
    except Exception:
        return -1


def build_prefix(session_id: int, lines: int, rng: random.Random) -> tuple[str, str]:
    """A long per-session preamble carrying that session's unique code.

    The code sits in the PREFIX, which is precisely the span a prefix hit does not recompute. A
    correct restore cannot lose it; a mismatched one cannot keep it.
    """
    code = (f"{rng.randint(1000, 9999)}-"
            + "".join(rng.choices(string.ascii_uppercase, k=3))
            + f"-{rng.randint(1000, 9999)}")
    topics = ["harbour", "lantern", "meadow", "compass", "quarry", "thistle", "ferry",
              "almanac", "orchard", "beacon", "trellis", "cistern", "pelican", "sundial"]
    body = [f"Dossier {session_id:03d} line {i:05d}: the {rng.choice(topics)} was logged clean."
            for i in range(lines)]
    body[len(body) // 2] = (f"Dossier {session_id:03d} line {len(body)//2:05d}: "
                            f"the vault authorisation code is {code} — this is the only record.")
    return "\n".join(body), code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:1919/v1")
    ap.add_argument("--expect-model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--container", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=32,
                    help="distinct conversations; keep it ABOVE the snapshot store's frame count "
                         "(the serve boots with frames=22) so frames churn and drops rise")
    ap.add_argument("--turns", type=int, default=6, help="turns per session after the first")
    ap.add_argument("--lines", type=int, default=400, help="prefix lines (~15 tokens each)")
    # THE ONLY VALID CONFIGURATION, and both cheaper ones are BOOBY-TRAPPED (see the module
    # docstring's THREE FALSE-POSITIVE MECHANISMS). Generous max_tokens, NO reasoning cap.
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--reasoning-max-tokens", type=int, default=0,
                    help="cap the THINK span. LEAVE AT 0: any cap guillotines the chain of thought "
                         "mid-sentence and the fragment typed so far is served as the answer, which "
                         "scores as a retrieval failure on a turn that retrieved correctly")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    with urllib.request.urlopen(args.base_url.rstrip("/") + "/models", timeout=30) as fh:
        ids = [m.get("id") for m in json.load(fh).get("data") or []]
    if args.expect_model not in ids:
        sys.exit(f"provenance FAIL: /v1/models lists {ids}, expected {args.expect_model!r}")

    container = args.container
    if container == "auto":
        port = (re.search(r":(\d+)", args.base_url) or [None, "1919"])[1]
        names = subprocess.run(["docker", "ps", "--filter", f"publish={port}", "--format",
                                "{{.Names}}"], capture_output=True, text=True).stdout.split()
        if len(names) != 1:
            sys.exit(f"--container auto: expected one container on {port}, got {names}")
        container = names[0]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join(args.out, stamp)
    os.makedirs(outdir, exist_ok=True)

    rng = random.Random(args.seed)
    prefixes, codes = [], []
    for s in range(args.sessions):
        p, c = build_prefix(s, args.lines, rng)
        prefixes.append(p)
        codes.append(c)
    code_set = set(codes)
    print(f"[setup] {args.sessions} sessions x {args.lines} lines, {args.turns} reuse turns each; "
          f"container={container}")

    turns_log: list[dict] = []
    tsv = os.path.join(outdir, "reuse.tsv")
    with open(tsv, "w") as fh:
        fh.write("\t".join(["turn_idx", "session", "turn", "computed", "hit_ratio", "radix_hits",
                            "frames", "in_use", "high_water", "drops",
                            "retrieved", "wrong", "empty", "contaminated_by", "truncated"]) + "\n")

    n = 0
    fails = 0
    truncs = 0
    for t in range(args.turns):
        for s in range(args.sessions):
            # Turn 1 of each session establishes the prefix; later turns RESUME it, which is what
            # makes this a prefix-reuse test rather than another long-prompt test.
            msgs = [{"role": "user", "content": prefixes[s]
                     + f"\n\nAcknowledge dossier {s:03d} in one word."}]
            if t > 0:
                msgs += [{"role": "assistant", "content": "Acknowledged."},
                         {"role": "user", "content":
                          f"What is the vault authorisation code in dossier {s:03d}? "
                          f"Reply with just the code, exactly as written."}]
            else:
                msgs[0]["content"] = (prefixes[s] +
                                      f"\n\nWhat is the vault authorisation code in dossier "
                                      f"{s:03d}? Reply with just the code, exactly as written.")
            try:
                res = ask(args.base_url, args.expect_model, msgs, args.max_tokens, args.timeout,
                          args.reasoning_max_tokens)
            except Exception as exc:
                sys.stderr.write(f"  turn failed s={s} t={t}: {repr(exc)[:140]}\n")
                continue
            body = (res["content"] or "").strip()
            want = codes[s]
            got = want in body
            # TRUNCATED != WRONG. A reply cut at max_tokens that carries a PREFIX of the right code
            # retrieved correctly and merely ran out of room; counting it as a miss inverts the
            # measurement. Scored separately and excluded from the failure count.
            truncated = (res.get("finish_reason") == "length") and not got
            # CONTAMINATION: another session's code in this session's answer. Unambiguous.
            other = [c for c in code_set if c != want and c in body]
            c = counters(args.base_url)
            snap = snapshot_stats(container)
            n += 1
            if not got and not truncated:
                fails += 1
            if truncated:
                truncs += 1
            with open(tsv, "a") as fh:
                fh.write("\t".join(str(x) for x in [
                    n, s, t, int(c.get("computed", -1)), round(c.get("hit_ratio", -1), 4),
                    rec_radix_hits(container) if n % 10 == 0 else "",
                    snap.get("frames", ""), snap.get("in_use", ""), snap.get("high_water", ""),
                    snap.get("drops", ""), int(got), int(bool(body) and not got),
                    int(not body), ";".join(other)]) + "\n")
            turns_log.append({"session": s, "turn": t, "want": want, "got": got,
                              "contaminated_by": other, **res})
            if other:
                print(f"  !! CONTAMINATION s={s} t={t}: answer carries {other} (wants {want})",
                      flush=True)
            elif truncated:
                print(f"  ~~ TRUNC s={s} t={t} wants {want} got {body[:40]!r} "
                      f"(retrieval OK, ran out of tokens)", flush=True)
            elif not got:
                print(f"  .. MISS s={s} t={t} wants {want} got {body[:60]!r}", flush=True)
        # Knowledge control once per sweep — if this degrades too, the retrieval framing is wrong.
        kbad = 0
        for q, wantk in KNOWLEDGE:
            try:
                rk = ask(args.base_url, args.expect_model, [{"role": "user", "content": q}],
                         150, args.timeout)
                if wantk not in (rk["content"] or "").lower():
                    kbad += 1
            except Exception:
                pass
        snap = snapshot_stats(container)
        print(f"[sweep {t}] retrieval_fail={fails}/{n} truncated={truncs}  "
              f"K_wrong={kbad}/{len(KNOWLEDGE)}  "
              f"snapshot={snap}  hit_ratio={counters(args.base_url).get('hit_ratio')}", flush=True)
        with open(os.path.join(outdir, "result.json"), "w") as fh:
            json.dump({"args": vars(args), "container": container, "turns": turns_log}, fh, indent=1)

    with open(os.path.join(outdir, "turns.txt"), "w") as fh:
        for t in turns_log:
            fh.write(f"===== s{t['session']} t{t['turn']} want={t['want']} got={t['got']} "
                     f"contaminated_by={t['contaminated_by']}\n--- content ---\n{t['content']}\n\n")
    print(f"\nretrieval failures {fails}/{n} (truncated, excluded: {truncs})\n"
          f"fixture: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
