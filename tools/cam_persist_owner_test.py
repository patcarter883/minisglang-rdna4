"""Unit test for rank0-authoritative persistence: a non-owner rank (_persist_owner=False) must never
write the store to disk (autosave/save no-op), while an owner does. Mocks only model bits.
Run: PYTHONPATH=python python tools/cam_persist_owner_test.py"""
import os
import sys
import tempfile
import types

import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, _SRC)
from minisgl.cam.memory import CAMMemory, _NsState  # noqa: E402


def make(owner, path):
    self = types.SimpleNamespace()
    self._persist_owner = owner
    self.store_path = path
    self._dirty = True
    self._last_save = 0.0
    self._save_interval = 5.0
    self.pointer_only = True
    self.n_banks, self.k_slots, self.mem_dim = 4, 8, 16
    self.meta = {"base_model": "t"}
    st = _NsState([], False)
    st.facts = {(1,): {"object_ids": [9], "base_p": 0.0, "used": 1}}
    self._ns_states = {"default": st}
    self.autosave = CAMMemory.autosave.__get__(self)
    self.save = CAMMemory.save.__get__(self)
    self.snapshot = CAMMemory.snapshot.__get__(self)
    return self


def run():
    ok = tot = 0

    def check(n, c):
        nonlocal ok, tot
        tot += 1; ok += bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}")

    d = tempfile.mkdtemp()

    # non-owner: no disk writes at all
    p1 = os.path.join(d, "nonowner.pt")
    no = make(False, p1)
    check("non-owner autosave(force) -> False", no.autosave(force=True) is False)
    check("non-owner autosave wrote NO file", not os.path.exists(p1))
    check("non-owner save() -> -1", no.save() == -1)
    check("non-owner save wrote NO file", not os.path.exists(p1))

    # owner: writes
    p2 = os.path.join(d, "owner.pt")
    yes = make(True, p2)
    check("owner autosave(force) -> True", yes.autosave(force=True) is True)
    check("owner autosave wrote the file", os.path.exists(p2))
    yes2 = make(True, os.path.join(d, "owner2.pt"))
    check("owner save() -> fact count", yes2.save() == 1)
    check("owner save wrote the file", os.path.exists(os.path.join(d, "owner2.pt")))

    print(f"\nPERSIST OWNER: {ok}/{tot}")
    return ok == tot


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
