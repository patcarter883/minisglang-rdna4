#!/usr/bin/env python3
"""Score ONE Hermes session's assistant turns through `toolcall_degen_probe`'s detectors.

Why this exists separately from `toolcall_degen_replay_validate.py`: that script is the CALIBRATION
gate (fixed model list, fixed known-junk turn ids, re-run after any detector edit). This one is the
FIELD instrument -- point it at a session that is running right now and it tells you whether the
degeneration signature has appeared yet, with zero GPU load and no contention with the serve.

It imports `score_turn` rather than reimplementing it, deliberately. A forked detector drifts from
its calibration and then "0 firings" stops meaning anything -- the calibration (3/3 incident calls
caught, 0 firings across 3,034 control turns on 4 models) belongs to THAT function, not to this
script.

Read the metric hierarchy before reading the output:
  PRIMARY    junk_args   -- the REAL signature. q4e 7.9% vs <=0.8% on every control.
  SECONDARY  think_zero, id_noise
  DIAGNOSTIC think_collapse, content_unterminated, empty_stop
`empty_stop` is the legacy BLIND metric: 0% across 2,536 real turns while the model was visibly
broken. Do not read it as health.

    python3 tools/hermes_session_score.py --session a34e1344af0a
    python3 tools/hermes_session_score.py --session a34e1344af0a --since 40000 --verbose

Read-only against ~/.hermes/state.db. No GPU, no serve, no lease.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toolcall_degen_probe import DIAGNOSTIC, PRIMARY, SECONDARY, score_turn  # noqa: E402

DB = os.path.expanduser("~/.hermes/state.db")
KEYS = PRIMARY + SECONDARY + DIAGNOSTIC


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", required=True, help="session id, or a unique prefix of one")
    ap.add_argument("--db", default=DB)
    ap.add_argument("--since", type=int, default=0,
                    help="only score message ids > this (to diff against an earlier run)")
    ap.add_argument("--verbose", action="store_true", help="print every firing turn")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    sids = [r[0] for r in con.execute("select id from sessions where id like ?",
                                      (f"%{args.session}%",))]
    if not sids:
        print(f"no session matching {args.session!r}", file=sys.stderr)
        return 2
    if len(sids) > 1:
        print(f"ambiguous: {sids}", file=sys.stderr)
        return 2
    sid = sids[0]
    model = con.execute("select model from sessions where id=?", (sid,)).fetchone()[0]

    rows = con.execute(
        """select id, content, tool_calls, finish_reason, reasoning, reasoning_content, token_count
           from messages where session_id=? and role='assistant' and id > ?
           order by id""", (sid, args.since)).fetchall()

    counts = {k: 0 for k in KEYS}
    firing = []
    prior_think: list = []
    n_calls = 0
    for mid, content, tool_calls, finish, reasoning, rc, ntok in rows:
        try:
            calls = json.loads(tool_calls) if tool_calls else []
        except Exception:
            calls = []
        n_calls += len(calls or [])
        msg = {"content": content or "", "tool_calls": calls,
               "reasoning_content": rc or reasoning or ""}
        s = score_turn(msg, prior_think, finish or "")
        prior_think.append(len(msg["reasoning_content"]))
        hits = [k for k in KEYS if s.get(k)]
        for k in hits:
            counts[k] += 1
        # PRIMARY/SECONDARY firings are the ones worth eyeballing; a DIAGNOSTIC-only turn is noise.
        if any(k in PRIMARY + SECONDARY for k in hits):
            firing.append((mid, hits, calls, (content or "")[:120]))

    n = len(rows)
    print(f"session {sid}  model {model!r}")
    print(f"{n} assistant turns scored (ids > {args.since}), {n_calls} tool calls\n")
    if not n:
        print("no turns yet")
        return 0
    width = max(len(k) for k in KEYS)
    for group, label in ((PRIMARY, "PRIMARY"), (SECONDARY, "SECONDARY"), (DIAGNOSTIC, "diagnostic")):
        for k in group:
            c = counts[k]
            flag = "  <-- FIRED" if c and k in PRIMARY + SECONDARY else ""
            print(f"  {label:10s} {k:<{width}s} {c:4d}  {100.0*c/n:5.1f}%{flag}")
    print(f"\nlast message id: {rows[-1][0]}   (pass --since {rows[-1][0]} next run to see only new turns)")
    if firing:
        print(f"\n{len(firing)} PRIMARY/SECONDARY firing turn(s):")
        for mid, hits, calls, snippet in firing[: (None if args.verbose else 10)]:
            print(f"  id={mid} {hits}")
            for c in (calls or [])[:3]:
                fn = (c or {}).get("function") or {}
                print(f"      {fn.get('name')!r} args={str(fn.get('arguments'))[:200]!r}")
            if snippet.strip():
                print(f"      content: {snippet!r}")
    else:
        print("\nno PRIMARY/SECONDARY firings -- the degeneration signature has NOT appeared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
