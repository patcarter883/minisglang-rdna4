#!/usr/bin/env python3
"""Rate of tool calls whose TARGET DOES NOT EXIST -- the confabulated-reference signature.

WHY THIS EXISTS. `toolcall_degen_probe`'s detectors are all SHAPE-based: `junk_args` and
`id_noise` fire on identifier-shaped noise (invented UUIDs, 40/64-hex strings, `shorthex::`,
a command stuffed into a `path` slot). A model that invents a PLAUSIBLE document name --
"the spec is in docs/ARCHITECTURE.md", when no such file exists -- produces arguments that are
structurally perfect. Every shape detector reads it as clean. The failure is semantic, and the
panel was blind to it (observed directly: a Flash-Next session hallucinated document names on
its SECOND turn while `junk_args`/`id_noise`/`think_zero` all read 0.0%).

THE ENVIRONMENT ALREADY KNOWS. A nonexistent target comes back as a tool ERROR. So this probe
does not guess at shape; it reads the tool result's STATUS FIELD and asks whether the target
resolved. That is the environment's own verdict, not a heuristic about text.

MATCH ON THE STATUS FIELD, NEVER ON FREE TEXT. The first cut of this probe regex'd the whole
result body for "does not exist" and scored 11.4% on a control session -- because it was matching
those words inside the CONTENTS of files that had been read successfully. A metric that counts
something other than what it claims is how `empty_stop` read 0% across 2,536 turns while the
model was visibly broken. Parse the JSON, require success is False or a non-empty `error`, and
classify only that string.

DELIBERATELY NOT COUNTED (these are not confabulation):
  * Web 404s. Guessing a documentation URL during research is normal behaviour, and web results
    dominate any naive not-found match (172 sessions, mostly ROCm/GitHub doc 404s).
  * "Could not find a match for old_string" -- a stale edit, not an invented target.
  * "BLOCKED: You have called read_file on this exact region 3 times" -- a loop guard firing.
  * "Refusing to write ..." -- a policy refusal; the path exists fine.

CALIBRATION (2026-09-22, ~/.hermes/state.db, 12,253 tool-result messages, sessions with >=20
filesystem calls). THE BASE RATE IS NOT ZERO -- do not read a bare nonzero as proof of anything:

    poolside/Laguna-XS-2.1-NVFP4     30 calls    7 missing   23.3%
    deepseek/deepseek-v4-pro         23 calls    3 missing   13.0%
    (unnamed "model")                25 calls    2 missing    8.0%
    poolside/laguna-m.1:free         24 calls    1 missing    4.2%
    Qwen3.8-27B                      43 calls    1 missing    2.3%
    deepseek/deepseek-v4.1-flash   1045 calls    1 missing    0.1%
    deepseek-v4-pro, gemini-3.1-flash-lite, laguna-m.1, glm-5.3, owl-alpha: 0.0%

WHAT IT ACTUALLY MEASURES, AND THE LIMITATION. It counts SPECULATIVE TARGETS, which covers two
different behaviours this probe CANNOT separate:
  (a) benign exploration -- guessing a conventional path in an unfamiliar repo
      ("packages/bedrock/src/index.ts"), which is how Laguna-XS reaches 23.3%; and
  (b) confabulation -- inventing a document that the task implies should exist
      ("audit.md", "audit_report.md" in a workspace that has neither).
Both are real hits; only (b) is degeneration. So USE IT AS A CROSS-ARM COMPARISON ON THE SAME
TASK, never against an absolute threshold. Two arms running the same agent workload differ only
in the engine, and then a rate gap is attributable. One number on its own is not evidence.

A separate genuine defect it surfaced: deepseek-v4-pro emitting `File not found: ` with an EMPTY
path -- a malformed argument, not a wrong one.

    python3 tools/missing_target_probe.py --session 91c96f430a55
    python3 tools/missing_target_probe.py --all --min-calls 20

Read-only against ~/.hermes/state.db. No GPU, no serve, no lease.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sqlite3
import sys

DB = os.path.expanduser("~/.hermes/state.db")

#: Tools whose target is a path on disk. `terminal`/`execute_code` are excluded on purpose: a
#: shell command that fails to find a file is usually probing for it, which is legitimate.
FS_TOOLS = {"read_file", "patch", "search_files", "write_file", "skill_view"}

#: Classifies an ERROR STRING (never a whole result body) as "the target was not there".
NOT_EXIST = re.compile(
    r"no such file|file not found|does not exist|failed to (read|open|write) file|"
    r"enoent|cannot find|no files? (found|match)",
    re.I,
)


def scan(db_path: str, session: str | None, min_calls: int):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    q = ("select m.session_id, m.tool_name, m.content, s.model "
         "from messages m left join sessions s on s.id = m.session_id "
         "where m.role='tool' and m.tool_name is not null")
    args: tuple = ()
    if session:
        q += " and m.session_id like ?"
        args = (session + "%",)
    per = collections.defaultdict(lambda: {"calls": 0, "missing": 0, "model": None, "hits": []})
    for sid, tool, content, model in con.execute(q, args):
        if tool not in FS_TOOLS:
            continue
        rec = per[sid]
        rec["calls"] += 1
        rec["model"] = model
        try:
            res = json.loads(content or "")
        except Exception:
            continue
        if not isinstance(res, dict):
            continue
        if res.get("success") is False or res.get("error"):
            err = str(res.get("error") or "")
            if NOT_EXIST.search(err):
                rec["missing"] += 1
                rec["hits"].append((tool, err[:160]))
    return {s: r for s, r in per.items() if r["calls"] >= min_calls or (session and r["calls"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="session id or unique prefix")
    ap.add_argument("--all", action="store_true", help="every session, as a baseline table")
    ap.add_argument("--db", default=DB)
    ap.add_argument("--min-calls", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if not a.session and not a.all:
        ap.error("pass --session or --all")

    per = scan(a.db, a.session, a.min_calls)
    if not per:
        print("no filesystem tool calls yet")
        return 0

    rows = sorted(per.items(), key=lambda kv: -(kv[1]["missing"] / max(1, kv[1]["calls"])))
    print(f"{'session':14} {'model':34} {'fs':>5} {'missing':>8} {'rate':>7}")
    for sid, r in rows:
        rate = 100.0 * r["missing"] / max(1, r["calls"])
        print(f"{sid[:12]:14} {str(r['model'])[:34]:34} {r['calls']:5d} {r['missing']:8d} {rate:6.1f}%")
        if (a.verbose or a.session) and r["hits"]:
            for tool, err in r["hits"][:20]:
                print(f"    {tool}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
