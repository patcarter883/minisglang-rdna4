"""The PLE once-per-forward ledger invariant — evaluated, which it never was.

`PLERuntime` has always incremented `prepares` / `commits` / `discards` / `commit_noops` and
documented the invariant they satisfy:

    prepares == commits + discards      and      commit_noops == 0

and until `ledger_fault()` existed, NOTHING IN THE ENGINE READ THEM. The counters were a comment.
That matters because of what the invariant catches: a forward that ran with nothing staged leaves
every slot in it re-hashing a stale token context, so its n-gram features FREEZE. Nothing crashes
and the weights are untouched — the model keeps answering fluently from its parameters while
misreading what it was GIVEN. Fluent, confident, wrong.

`ledger_fault` is exercised here against a stand-in rather than a live `PLERuntime`, because
constructing one wants a device and an n-gram source; the method reads five plain attributes and
nothing else, which is exactly why it can be checked this way.

Run (in the serve image, CPU only — the module imports torch):
    PYTHONPATH=/engine/python python3 /engine/tests/ple_ledger_test.py
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from minisgl.ple.runtime import PLERuntime

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK   ' if cond else 'FAIL '} {name}" + (f": {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def rt(*, prepares=0, commits=0, discards=0, noops=0, batch=None, begun=None, pending=None):
    return SimpleNamespace(prepares=prepares, commits=commits, discards=discards,
                           commit_noops=noops, batch=batch, _begun=begun, _pending=pending)


def main() -> int:
    fault = PLERuntime.ledger_fault

    print("HEALTHY LEDGERS report no fault")
    check("fresh runtime", fault(rt()) is None)
    check("balanced prepares/commits", fault(rt(prepares=10, commits=10)) is None)
    check("commits + discards (capture stages and discards)",
          fault(rt(prepares=10, commits=7, discards=3)) is None)

    print("\nTHE FAILURE THE COUNTERS EXIST FOR")
    f = fault(rt(prepares=10, commits=10, noops=1))
    check("commit_noops != 0 is a fault", f is not None)
    check("and it says the history is frozen", bool(f) and "frozen" in f, repr(f))

    f = fault(rt(prepares=10, commits=9))
    check("staged-but-never-committed is a fault", f is not None)
    check("and it names the conv-state skew", bool(f) and "conv state" in f, repr(f))

    print("\nMID-FORWARD IS NOT A FAULT — the pairing is legitimately open")
    # This is the property that stops the check firing every single step: `commit_staged` runs after
    # the forward, so between `prepare` and it the counters are unequal BY DESIGN. A checker without
    # this would cry fault on every healthy forward and get switched off, which is worse than not
    # having one.
    check("batch staged -> quiet", fault(rt(prepares=10, commits=9, batch=object())) is None)
    check("reads in flight -> quiet", fault(rt(prepares=10, commits=9, begun=object())) is None)
    check("noops are also suppressed mid-forward",
          fault(rt(prepares=1, commits=0, noops=5, batch=object())) is None)

    print("\nPRECEDENCE: commit_noops is reported over the arithmetic")
    # Both conditions hold at once here. noops is the more specific and more actionable diagnosis,
    # so it must win; reporting only the subtraction would send the reader hunting a lost discard.
    f = fault(rt(prepares=10, commits=8, noops=2))
    check("noops wins", bool(f) and "commit_noops" in f, repr(f))

    print("\nledger() reports the raw counters")
    led = PLERuntime.ledger(rt(prepares=3, commits=2, discards=1, noops=0, pending=object()))
    check("counters round-trip", led == {"prepares": 3, "commits": 2, "discards": 1,
                                         "commit_noops": 0, "pending": True}, repr(led))

    print("\n" + ("all passed" if not FAILED else f"FAILED: {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
