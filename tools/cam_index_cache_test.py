"""Unit test for the cached subject-key matrix: after every index mutation (append, in-place update,
eviction, delete) the cache must equal a fresh torch.stack(subj_keys) and deliver the same argmax — in
particular the append+evict-same-length case that a length-only check would miss. Drives the REAL
_write/_maybe_evict/_key_matrix with controlled keys. Run: PYTHONPATH=python python tools/cam_index_cache_test.py"""
import os
import sys
import types

import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, _SRC)
from minisgl.cam.memory import CAMMemory, _NsState  # noqa: E402


def unit(*v):
    t = torch.tensor(v, dtype=torch.float32)
    return t / t.norm()


def make(max_facts=0, dedup_tau=1.0):
    self = types.SimpleNamespace()
    self.pointer_only = True
    self.write_dedup_tau = dedup_tau
    self.max_facts = max_facts
    self._audit = []; self._audit_max = 100; self._dirty = False
    self._keymap = {}
    st = _NsState([])
    self._state = lambda ns=None: st
    self._subj_key = lambda ids: self._keymap[tuple(int(x) for x in ids)]
    for m in ("_write", "_key_matrix", "_dedup_match", "_maybe_evict", "_audit_add"):
        setattr(self, m, getattr(CAMMemory, m).__get__(self))
    return self, st


def fresh(st):
    return torch.stack(st.subj_keys) if st.subj_keys else None


def run():
    ok = tot = 0

    def check(n, c):
        nonlocal ok, tot
        tot += 1; ok += bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    def cache_matches(self, st):
        m = self._key_matrix(st); f = fresh(st)
        if m is None and f is None:
            return True
        return m is not None and f is not None and torch.equal(m, f)

    # --- appends: cache tracks growth -----------------------------------------------------------------
    cam, st = make()
    cam._keymap = {(i,): unit(*[1.0 if j == i else 0.0 for j in range(6)]) for i in range(6)}
    for i in range(4):
        cam._write([i], [100 + i])
        check(f"cache == fresh after append {i}", cache_matches(cam, st))

    # --- in-place exact update: key changes, same length ---------------------------------------------
    cam._keymap[(1,)] = unit(0.0, 1.0, 0.5, 0, 0, 0)          # change subject 1's key
    cam._write([1], [999])                                     # exact re-remember -> updates key in place
    check("cache reflects in-place key update", cache_matches(cam, st))
    check("in-place update kept length", len(st.subj_keys) == 4)

    # --- THE edge case: append + LRU evict net to the SAME length (length check alone would go stale) --
    cam2, st2 = make(max_facts=3)
    cam2._keymap = {(i,): unit(*[1.0 if j == i else 0.0 for j in range(6)]) for i in range(6)}
    for i in range(3):
        cam2._write([i], [i])
    _ = cam2._key_matrix(st2)                                  # prime the cache at length 3
    cam2._write([5], [5])                                      # append (len 4) then evict LRU (back to len 3)
    check("append+evict kept length 3", len(st2.subj_keys) == 3)
    check("cache NOT stale after append+evict", cache_matches(cam2, st2))
    # the newest subject (5) must be deliverable via the cache; the evicted one (0) must not win its own query
    simK = cam2._key_matrix(st2)
    q5 = cam2._keymap[(5,)]
    j = int((simK @ q5).argmax())
    check("newest subject addressable via cache", st2.subj_tuple[j] == (5,))
    check("evicted subject gone from index", (0,) not in st2.subj_tuple)

    print(f"\nINDEX CACHE: {ok}/{tot}")
    return ok == tot


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
