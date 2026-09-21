#!/usr/bin/env python3
"""Compare two answer_delivery_probe fixtures arm-by-arm, and REFUSE to compare incomparable arms.

The comparison this exists for: "is the q4e serve's answer-delivery failure a code property, or does
it develop over a boot's uptime?" Same commit, same mounts, same checkpoint — one variable, the boot.

PROVENANCE IS CHECKED, NOT PRINTED AND IGNORED. The arms must agree on model, max_tokens,
reasoning_max_tokens, lanes and reps, and their `engine_sha` must match; they must NOT share a
container boot (`started_at`), or there is no A/B. A mismatch is reported as a BLOCKING error rather
than a footnote under the numbers — an A/B whose arms silently differed in the source tree, or
silently shared a build, is this repo's most expensive recurring mistake.

Reads: two fixture directories (each holding result.json). Writes nothing; prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Provenance keys that must be EQUAL for the arms to be comparable.
_MUST_MATCH = ["expect_model", "max_tokens", "reasoning_max_tokens", "reps", "lanes"]
# Detectors, in reporting order. no_answer first: it is the headline defect.
_FLAGS = ["no_answer", "eos_in_think", "answer_wrong", "digit_noise", "cjk_leak", "loop"]


def load(path: str) -> dict:
    fn = path if path.endswith(".json") else os.path.join(path, "result.json")
    with open(fn) as fh:
        return json.load(fh)


def check_comparable(a: dict, b: dict, la: str, lb: str) -> list[str]:
    pa, pb = a["provenance"], b["provenance"]
    errs = []
    for k in _MUST_MATCH:
        if pa.get(k) != pb.get(k):
            errs.append(f"{k}: {la}={pa.get(k)!r} vs {lb}={pb.get(k)!r}")
    sa, sb = pa.get("serve", {}), pb.get("serve", {})
    if sa.get("engine_sha") != sb.get("engine_sha"):
        errs.append(f"engine_sha: {la}={sa.get('engine_sha')} vs {lb}={sb.get('engine_sha')} "
                    "— arms differ in SOURCE, not just in the thing under test")
    for lbl, s in ((la, sa), (lb, sb)):
        if s.get("engine_dirty"):
            errs.append(f"{lbl}: engine tree DIRTY at probe time ({s['engine_dirty'][:80]!r}) "
                        "— the served source is not the recorded sha")
    if sa.get("started_at") and sa.get("started_at") == sb.get("started_at"):
        errs.append(f"both arms ran against the SAME container boot ({sa['started_at']}) "
                    "— this is not an A/B")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--allow-incomparable", action="store_true",
                    help="print the table anyway; the errors are still reported")
    args = ap.parse_args()

    a, b = load(args.before), load(args.after)
    la = a["provenance"]["arm"]
    lb = b["provenance"]["arm"]

    for lbl, d in ((la, a), (lb, b)):
        s = d["provenance"].get("serve", {})
        print(f"[{lbl}] container={s.get('container')} image={s.get('image')} "
              f"booted={s.get('started_at')}\n"
              f"{' ' * (len(lbl) + 3)}engine={s.get('engine_source')} sha={s.get('engine_sha')} "
              f"dirty={s.get('engine_dirty') or 'clean'}")
    errs = check_comparable(a, b, la, lb)
    if errs:
        print("\nPROVENANCE FAIL:")
        for e in errs:
            print(f"  ! {e}")
        if not args.allow_incomparable:
            return 2
        print("  (continuing under --allow-incomparable)")

    for scope in ("by_lane", "by_lane_prompt"):
        rows = sorted(set(a["summary"][scope]) | set(b["summary"][scope]))
        if not rows:
            continue
        print(f"\n{scope}   n     " + "  ".join(f"{f[:11]:>11s}" for f in _FLAGS))
        for key in rows:
            ra = a["summary"][scope].get(key, {})
            rb = b["summary"][scope].get(key, {})
            n = f"{ra.get('n', 0)}/{rb.get('n', 0)}"
            cells = []
            for f in _FLAGS:
                va, vb = ra.get(f, 0), rb.get(f, 0)
                mark = "  " if va == vb else ("->" if vb < va else "!!")
                cells.append(f"{va:>3}{mark}{vb:<3}".rjust(11))
            print(f"  {key:22s} {n:5s} " + "  ".join(cells))
    print(f"\ncolumns: {la} -> {lb}   ('->' improved, '!!' worse)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
