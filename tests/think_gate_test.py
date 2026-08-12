"""Regression test for the reasoning ("think") gate's SEQUENCE matching.

The bug this locks down: the gate was a single token id, and a multi-token close delimiter was
reduced to its LAST id ("a rare, tolerable approximation"). On Muse-Glimmer the reasoning opener,
the closer and the answer-turn header ALL end in the same token (`<|message|>`, 200023), so the gate
opened on the third token of the OPENING delimiter. Every structured-output request then had its
schema engaged inside the reasoning turn and came back with `content: ""` and the whole answer in
`reasoning_content`.

Sections:

* MUSE       — the real Muse-Glimmer token ids. The shared-trailing-token case, i.e. the bug.
* SYNTHETIC  — the same shape, minimal ids, plus partial-match / force-alignment behaviour.
* WILDCARD   — the recipient-wildcard closer (`<|eom|><|start|>assistant to=<any><|message|>`).
* OVERLAP    — self-overlapping patterns, which a reset-on-divergence cursor gets wrong.
* PARITY     — single-token `</think>` must behave EXACTLY as the old int-valued gate did.
* PURITY     — `scan` mutates nothing; rejected drafts cannot advance the gate (TP lockstep).

The module under test is imported BY PATH and asserted to be torch-free, so this runs on a host with
no GPU (and on one whose torch cannot even import).

Run:  PYTHONPATH=python python3 tests/think_gate_test.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MOD_PATH = Path(__file__).resolve().parent.parent / "python/minisgl/scheduler/think_gate.py"

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILED.append(name)


def _load():
    src = MOD_PATH.read_text()
    for banned in ("import torch", "from torch", "import os"):
        assert banned not in src, f"think_gate.py must stay free of {banned!r}"
    spec = importlib.util.spec_from_file_location("_think_gate", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_think_gate"] = mod   # @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


TG = _load().ThinkGate

# ---------------------------------------------------------------- MUSE (real ids)
# Derived at boot by server/reasoning.py from the checkpoint's own chat template, then tokenized.
MUSE_OPEN = (328, 19669, 200023)                                  # " to=self<|message|>"
MUSE_CLOSE = (200007, 200022, 140680, 328, 76976, 200023)         # "<|eom|>…assistant to=user<|message|>"
MUSE_HEADER = (328, 76976, 200023)                                # " to=user<|message|>"


def muse() -> None:
    print("\nMUSE — real Muse-Glimmer ids (open/close/header all end in 200023)")

    g = TG()
    g.arm(1, force_seq=MUSE_CLOSE, release_seqs=(MUSE_HEADER,), budget=50)

    # THE BUG: emitting the reasoning OPENER must not release the gate, even though its last token is
    # the same 200023 the closer ends with.
    rel = [g.commit(1, t) for t in MUSE_OPEN]
    check("opener does not release the gate", rel == [False, False, False], f"{rel}")
    check("still armed after opener", g.is_armed(1))

    for t in range(900, 920):  # 20 tokens of reasoning prose
        g.commit(1, t)
    check("prose does not release", g.is_armed(1))
    check("reasoning counted eagerly", g.count_of(1) == 23, f"{g.count_of(1)}")

    rel = [g.commit(1, t) for t in MUSE_CLOSE]
    check("closer releases on its LAST token only", rel == [False] * 5 + [True], f"{rel}")
    check("gate released", not g.is_armed(1) and g.is_done(1))

    # A direct answer (no reasoning turn at all) must also release, or the β backstop would splice a
    # turn header into the middle of a legitimate reply.
    g2 = TG()
    g2.arm(2, force_seq=MUSE_CLOSE, release_seqs=(MUSE_HEADER,), budget=50)
    rel = [g2.commit(2, t) for t in MUSE_HEADER]
    check("answer-turn header releases", rel == [False, False, True], f"{rel}")

    # Backstop: the whole 6-token delimiter is emitted, one token per step, counting nothing.
    g3 = TG()
    g3.arm(3, force_seq=MUSE_CLOSE, release_seqs=(MUSE_HEADER,), budget=5)
    for t in range(900, 905):
        g3.commit(3, t)
    forced = []
    while g3.is_armed(3):
        f = g3.forced_next(3)
        forced.append(f)
        g3.commit(3, f)
    check("backstop forces the full close sequence", tuple(forced) == MUSE_CLOSE, f"{forced}")
    check("forced tokens are not counted", g3.count_of(3) is None)  # cleared on release

    g4 = TG()
    g4.arm(4, force_seq=MUSE_CLOSE, budget=1)
    g4.commit(4, 999)
    ahead = [g4.forced_at(4, j) for j in range(8)]
    check("forced_at walks then clamps",
          ahead == list(MUSE_CLOSE) + [200023, 200023], f"{ahead}")


# ---------------------------------------------------------------- SYNTHETIC
S_OPEN, S_CLOSE, S_HEADER = (1, 2, 9), (7, 8, 1, 3, 9), (1, 3, 9)  # header is a SUFFIX of close


def synthetic() -> None:
    print("\nSYNTHETIC — shared trailing token, header is a suffix of close")

    g = TG()
    g.arm(1, force_seq=S_CLOSE, release_seqs=(S_HEADER,), budget=100)
    check("bare shared token does not release", not g.commit(1, 9))
    check("opener does not release", [g.commit(1, t) for t in S_OPEN] == [False] * 3)

    g2 = TG()
    g2.arm(2, force_seq=S_CLOSE, release_seqs=(S_HEADER,), budget=100)
    check("close releases at its last index", g2.commit_many(2, S_CLOSE) == 4)

    g3 = TG()
    g3.arm(3, force_seq=S_CLOSE, release_seqs=(S_HEADER,), budget=100)
    check("header alone releases", g3.commit_many(3, S_HEADER) == 2)

    # Partial match that then diverges: nothing may be lost from the budget count.
    g4 = TG()
    g4.arm(4, force_seq=S_CLOSE, budget=100)
    for t in (7, 8, 1, 55):
        g4.commit(4, t)
    check("partial-then-diverge keeps the gate", g4.is_armed(4))
    check("partial-then-diverge loses no count", g4.count_of(4) == 4, f"{g4.count_of(4)}")

    # Force alignment: the model already emitted `7,8` before the budget expired, so the backstop
    # must resume at index 2 rather than restarting the delimiter.
    g5 = TG()
    g5.arm(5, force_seq=S_CLOSE, budget=2)
    g5.commit(5, 7)
    g5.commit(5, 8)
    check("force resumes at the partial match", g5.forced_next(5) == 1, f"{g5.forced_next(5)}")
    emitted = []
    while g5.is_armed(5):
        f = g5.forced_next(5)
        emitted.append(f)
        g5.commit(5, f)
    check("force completes exactly one delimiter", emitted == [1, 3, 9], f"{emitted}")

    # A non-forced token mid-run invalidates the run.
    g6 = TG()
    g6.arm(6, force_seq=S_CLOSE, budget=1)
    g6.commit(6, 42)
    check("force starts at 0", g6.forced_next(6) == 7)
    g6.commit(6, 7)
    check("force advances", g6.forced_next(6) == 8)
    g6.commit(6, 77)                       # not what was forced
    check("interrupted force re-derives", g6.forced_next(6) == 7, f"{g6.forced_next(6)}")
    check("interrupted force counted the stray token", g6.count_of(6) == 2, f"{g6.count_of(6)}")


# ---------------------------------------------------------------- WILDCARD
def wildcard() -> None:
    print("\nWILDCARD — recipient-variable closer (head … <=gap … tail)")

    head, tail = (200007, 200022, 140680, 328), (200023,)   # "<|eom|><|start|>assistant to=" … "<|message|>"
    pat = (head, tail, 8)

    g = TG()
    g.arm(1, force_seq=MUSE_CLOSE, release_seqs=(pat, MUSE_HEADER), budget=100)
    # Route to a TOOL: the recipient tokens differ from `to=user`, so only the wildcard matches.
    stream = list(head) + [50001, 50002] + list(tail)
    check("tool-routed closer releases", g.commit_many(1, stream) == len(stream) - 1, f"{stream}")

    g2 = TG()
    g2.arm(2, force_seq=MUSE_CLOSE, release_seqs=(pat,), budget=100)
    check("head alone does not release", g2.commit_many(2, list(head)) is None)
    check("head then a too-long gap does not release",
          g2.commit_many(2, [7000] * 9 + list(tail)) is None)

    g3 = TG()
    g3.arm(3, force_seq=MUSE_CLOSE, release_seqs=(pat,), budget=100)
    check("bare tail does not release", g3.commit_many(3, list(tail)) is None)


# ---------------------------------------------------------------- OVERLAP
def overlap() -> None:
    print("\nOVERLAP — self-overlapping patterns (the reset-on-divergence cursor trap)")

    g = TG()
    g.arm(1, force_seq=(4, 4, 5), budget=100)
    check("4,4,4,5 releases on the final 5", g.commit_many(1, [4, 4, 4, 5]) == 3)

    g2 = TG()
    g2.arm(2, force_seq=(4, 4), budget=100)
    check("4,4 releases at index 1", g2.commit_many(2, [4, 4]) == 1)

    g3 = TG()
    g3.arm(3, force_seq=(4, 4), budget=100)
    check("4,3,4,4 releases at index 3", g3.commit_many(3, [4, 3, 4, 4]) == 3)


# ---------------------------------------------------------------- PARITY
class _OldGate:
    """The pre-fix single-token gate, reimplemented from scheduler.py as it stood at 07c58123."""

    def __init__(self, tid: int, budget: int) -> None:
        self.tid, self.budget, self.count, self.open = tid, budget, 0, True

    def over_budget(self) -> bool:
        return self.open and self.count >= self.budget

    def forced_next(self):
        return self.tid if self.over_budget() else None

    def suppress_eos(self) -> bool:
        return self.open and not self.over_budget()

    def commit(self, tok: int) -> bool:
        if not self.open:
            return False
        if tok == self.tid:
            self.open = False
            return True
        self.count += 1
        return False


def parity() -> None:
    print("\nPARITY — single-token </think> must behave exactly as the old int gate")

    THINK = 11
    # A scripted stream: prose, a stray near-miss, then a natural close; and a second run that never
    # closes so the budget backstop fires.
    for label, stream, budget in (
        ("natural close", [500 + i for i in range(40)] + [THINK] + [7, 7, 7], 1024),
        ("budget backstop", [500 + i for i in range(200)], 32),
    ):
        new, old = TG(), _OldGate(THINK, budget)
        new.arm(1, force_seq=(THINK,), budget=budget)
        diverged = ""
        for step, tok in enumerate(stream):
            if (new.forced_next(1), new.suppress_eos(1)) != (old.forced_next(), old.suppress_eos()):
                diverged = f"step {step}: query mismatch"
                break
            # Both gates emit the forced token instead of the sampled one when over budget.
            f = new.forced_next(1)
            t = f if f is not None else tok
            if new.commit(1, t) != old.commit(t):
                diverged = f"step {step}: release mismatch"
                break
            if new.count_of(1) not in (None, old.count):
                diverged = f"step {step}: count {new.count_of(1)} vs {old.count}"
                break
            if not new.is_armed(1):
                break
        check(f"single-token parity — {label}", not diverged, diverged)


# ---------------------------------------------------------------- PURITY / TP
def purity() -> None:
    print("\nPURITY — scan is pure; rejected drafts cannot advance the gate")

    g = TG()
    g.arm(1, force_seq=MUSE_CLOSE, release_seqs=(MUSE_HEADER,), budget=100)
    g.commit_many(1, [900, 901, 902])
    before = (list(g._st[1].window), g._st[1].count, g._st[1].forcing, g.is_armed(1))
    idxs = [g.scan(1, list(MUSE_CLOSE)) for _ in range(5)]
    after = (list(g._st[1].window), g._st[1].count, g._st[1].forcing, g.is_armed(1))
    check("scan is repeatable", idxs == [5] * 5, f"{idxs}")
    check("scan mutates nothing", before == after, f"{before} != {after}")

    # TP lockstep: rank A probes a speculative chain whose tail is later REJECTED; rank B never sees
    # it. After both commit the same authoritative run, their state must be identical.
    a, b = TG(), TG()
    for g_ in (a, b):
        g_.arm(7, force_seq=MUSE_CLOSE, release_seqs=(MUSE_HEADER,), budget=100)
    a.scan(7, [200007, 200022, 55555])          # drafted, then rejected — probe only
    keep = [900, 901]
    a.commit_many(7, keep)
    b.commit_many(7, keep)
    check("rejected draft leaves no trace",
          (a._st[7].window, a._st[7].count) == (b._st[7].window, b._st[7].count),
          f"{a._st[7].window} vs {b._st[7].window}")

    # EOS buried in a delimiter -> suppression must be dropped, or the request can never finish.
    g2 = TG()
    g2.arm(8, force_seq=MUSE_CLOSE, budget=100, eos_ids=(200007,))
    check("eos inside a delimiter drops suppression", not g2.suppress_eos(8))
    g3 = TG()
    g3.arm(9, force_seq=MUSE_CLOSE, budget=100, eos_ids=(200001, 200008))
    check("disjoint eos keeps suppression", g3.suppress_eos(9))

    # Disabled gate is inert in every direction.
    off = TG(enabled=False)
    check("disabled: arm is a no-op", off.arm(1, force_seq=MUSE_CLOSE) is False)
    check("disabled: queries neutral",
          not off.any_armed() and off.forced_next(1) is None
          and not off.suppress_eos(1) and off.scan(1, [1]) is None and not off.commit(1, 1))

    # Lifecycle.
    g4 = TG()
    g4.arm(3, force_seq=(11,))
    g4.clear(3)
    g4.clear(3)
    check("arm after clear is refused (reasoning already ended)", g4.arm(3, force_seq=(11,)) is False)
    g4.free(3)
    check("arm after free re-arms", g4.arm(3, force_seq=(11,)) is True)

    # Budget cap must reserve room for the answer AND for the force run.
    g5 = TG()
    g5.arm(4, force_seq=MUSE_CLOSE, budget=1024, max_tokens=64)
    check("budget capped with force headroom", g5.budget_of(4) == 42, f"{g5.budget_of(4)}")


if __name__ == "__main__":
    muse()
    synthetic()
    wildcard()
    overlap()
    parity()
    purity()
    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    sys.exit(1 if FAILED else 0)
