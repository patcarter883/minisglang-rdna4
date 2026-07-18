"""Unit test for write-side semantic dedup: exercises the REAL CAMMemory._write index logic (exact
update / paraphrase merge / distinct append) with controlled subject keys, so no checkpoint/model is
needed. Run: PYTHONPATH=python python tools/cam_write_dedup_test.py"""
import os
import sys
import types

import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, _SRC)
from minisgl.cam.memory import CAMMemory, _NsState  # noqa: E402

_loaded = sys.modules["minisgl.cam.memory"].__file__
assert _loaded.startswith(_SRC), f"imported the wrong memory.py: {_loaded} (want under {_SRC})"


def make_cam(dedup_tau):
    """A minimal object carrying just what _write's POINTER branch touches, with the REAL unbound
    methods bound to it. _subj_key returns a caller-provided unit vector keyed by the id-tuple."""
    self = types.SimpleNamespace()
    self.pointer_only = True
    self.write_dedup_tau = dedup_tau
    self.max_facts = 0
    self._audit = []
    self._audit_max = 2000
    self._dirty = False
    self._keymap = {}                                    # id-tuple -> unit key vector (test-controlled)
    self._index_dtype = torch.float16
    st = _NsState([])
    self._state = lambda ns=None: st
    self._subj_key = lambda ids: self._keymap[tuple(int(x) for x in ids)]
    self._write = CAMMemory._write.__get__(self)
    self._dedup_match = CAMMemory._dedup_match.__get__(self)
    self._key_matrix = CAMMemory._key_matrix.__get__(self)
    self._maybe_evict = CAMMemory._maybe_evict.__get__(self)
    self._audit_add = CAMMemory._audit_add.__get__(self)
    return self, st


def unit(*v):
    t = torch.tensor(v, dtype=torch.float32)
    return t / t.norm()


def run():
    ok = 0
    total = 0

    def check(name, cond):
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # --- Scenario: dedup ON (tau 0.82), GTE-like keys -------------------------------------------------
    cam, st = make_cam(0.82)
    kM = unit(1.0, 0.0, 0.0)                              # "Mozart"
    kMp = unit(0.90, 0.20, 0.0)                           # paraphrase, cos(kM,kMp) ~ 0.976 -> MERGE
    kMp = kMp                                             # (keep name)
    kL = unit(0.79, 0.61, 0.0)                            # "Leopold Mozart", cos(kM,kL) ~ 0.79 -> NO merge
    kE = unit(0.0, 0.0, 1.0)                              # "Einstein", orthogonal -> append
    cam._keymap = {(1,): kM, (2,): kMp, (3,): kL, (4,): kE}

    cam._write([1], [100])                                # store Mozart -> 100
    check("first write appends", len(st.subj_tuple) == 1 and st.subj_objs[0] == [100])

    cam._write([1], [101])                                # exact re-remember (same ids) -> update object
    check("exact re-remember updates in place", len(st.subj_tuple) == 1 and st.subj_objs[0] == [101])

    print(f"    cos(Mozart, paraphrase)={float(kM @ cam._keymap[(2,)]):.3f}  "
          f"cos(Mozart, Leopold)={float(kM @ kL):.3f}")
    cam._write([2], [102])                                # paraphrase of Mozart -> MERGE onto entry 0
    check("paraphrase MERGES (no duplicate)", len(st.subj_tuple) == 1)
    check("merge takes latest object", st.subj_objs[0] == [102])
    check("merge keeps first-seen subject key anchored", st.subj_tuple[0] == (1,))
    check("fact anchored under first-seen key", (1,) in st.facts and (2,) not in st.facts)
    check("merge audited", any(a["op"] == "merge" for a in cam._audit))

    cam._write([3], [103])                               # Leopold Mozart (cos 0.79 to anchor < 0.82) -> APPEND
    check("distinct-but-related (Leopold) APPENDS, no silent merge", len(st.subj_tuple) == 2)
    check("Leopold stored separately", [103] in st.subj_objs)

    cam._write([4], [104])                               # Einstein -> APPEND
    check("unrelated subject APPENDS", len(st.subj_tuple) == 3)

    # --- Scenario: dedup OFF (tau 1.0) -> paraphrase must NOT merge (base-embed default) --------------
    cam2, st2 = make_cam(1.0)
    cam2._keymap = {(1,): kM, (2,): kMp}
    cam2._write([1], [200])
    cam2._write([2], [201])
    check("dedup OFF: paraphrase appends (no merge)", len(st2.subj_tuple) == 2)

    print(f"\nWRITE-DEDUP UNIT: {ok}/{total}")
    print("DEDUP TEST PASS" if ok == total else "DEDUP TEST FAIL")
    return ok == total


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
