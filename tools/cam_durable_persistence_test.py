"""Unit test for durable CAM persistence: exercises the REAL CAMMemory.snapshot/restore file handling
(atomic tmp+fsync+os.replace, .bak retention, restore fallback) with mocked model bits — no checkpoint.
Asserts a torn or missing primary store recovers from .bak instead of silently losing every fact.
Run: PYTHONPATH=python python tools/cam_durable_persistence_test.py"""
import os
import sys
import tempfile
import types

import torch
import torch.nn.functional as F

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, _SRC)
from minisgl.cam.memory import CAMMemory, _NsState  # noqa: E402

_loaded = sys.modules["minisgl.cam.memory"].__file__
assert _loaded.startswith(_SRC), f"wrong memory.py: {_loaded}"


def make_cam():
    """A stand-in carrying just what snapshot/restore touch, with the REAL methods bound."""
    self = types.SimpleNamespace()
    self.pointer_only = True
    self.n_banks, self.k_slots, self.mem_dim = 4, 8, 16
    self.meta = {"base_model": "test"}
    self.device = "cpu"
    store = types.SimpleNamespace(init_state=lambda b, dev, dtype=None: torch.zeros(1, 4))
    self.adapter = types.SimpleNamespace(store=store)
    self._subj_key = lambda ids: F.normalize(torch.tensor([float(sum(ids) % 7) + 1.0, 1.0, 0.0]), dim=-1)
    self.snapshot = CAMMemory.snapshot.__get__(self)
    self.restore = CAMMemory.restore.__get__(self)
    st = _NsState([], False)
    st.facts = {(1, 2): {"object_ids": [100], "base_p": 0.0, "used": 1},
                (3,): {"object_ids": [101, 102], "base_p": 0.0, "used": 2}}
    st.seq = 2
    self._ns_states = {"default": st}
    return self


def run():
    ok = tot = 0

    def check(name, cond):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    d = tempfile.mkdtemp()
    path = os.path.join(d, "store.pt")

    cam = make_cam()
    n = cam.snapshot(path)                                   # save #1
    check("snapshot returns fact count", n == 2)
    check("primary written", os.path.isfile(path))
    check("temp cleaned up (atomic)", not os.path.exists(path + ".tmp"))
    check("no .bak on first save", not os.path.exists(path + ".bak"))

    cam.snapshot(path)                                       # save #2 -> retains #1 as .bak
    check(".bak retained on second save", os.path.isfile(path + ".bak"))

    r = make_cam(); r._ns_states = {}
    check("round-trip restore count", r.restore(path) == 2)
    check("round-trip facts intact", set(r._ns_states["default"].facts) == {(1, 2), (3,)})
    check("index rebuilt", len(r._ns_states["default"].subj_keys) == 2)

    # TORN primary (simulate a kill mid-write leaving garbage) -> must recover from .bak
    with open(path, "wb") as f:
        f.write(b"\x00torn half-written garbage")
    r2 = make_cam(); r2._ns_states = {}
    check("torn primary -> recovers from .bak", r2.restore(path) == 2)

    # MISSING primary (crash between the two renames) -> must recover from .bak
    os.remove(path)
    r3 = make_cam(); r3._ns_states = {}
    check("missing primary -> recovers from .bak", r3.restore(path) == 2)

    # BOTH gone -> raises (caller starts empty), does NOT hang or silently mis-restore
    os.remove(path + ".bak")
    r4 = make_cam(); r4._ns_states = {}
    try:
        r4.restore(path); raised = False
    except Exception:
        raised = True
    check("both gone -> raises (not silent)", raised)

    print(f"\nDURABLE PERSISTENCE: {ok}/{tot}")
    return ok == tot


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
