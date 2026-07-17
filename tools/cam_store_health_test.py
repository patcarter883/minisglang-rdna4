"""Unit test for store-health observability: the cosine-index crowding gauge (_index_crowding) and the
new stats fields (index_size, index_nn_cos_*, last_save_age_s, recovered_from_backup). Mocks only the
model-independent attrs. Run: PYTHONPATH=python python tools/cam_store_health_test.py"""
import os
import sys
import types

import torch
import torch.nn.functional as F

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, _SRC)
from minisgl.cam.memory import CAMMemory, _NsState  # noqa: E402


def unit(*v):
    t = torch.tensor(v, dtype=torch.float32)
    return t / t.norm()


def make(keys, *, recovered=False, last_save=0.0):
    self = types.SimpleNamespace()
    self.n_banks = 4
    self.write_policy = "no-clobber"
    self.deliver_tau = 0.70
    self.max_facts = 0
    self.store_path = "/tmp/store.pt"
    self._dirty = True
    self._last_save = last_save
    self._recovered_from_bak = recovered
    st = _NsState([], False)
    st.subj_keys = list(keys)
    st.subj_tuple = [(i,) for i in range(len(keys))]
    st.facts = {(i,): {"object_ids": [i], "used": i} for i in range(len(keys))}
    st.evicted = 0
    self._ns_states = {"default": st}
    self._state = lambda ns=None: st
    self.stats = CAMMemory.stats.__get__(self)
    self._index_crowding = CAMMemory._index_crowding.__get__(self)
    return self, st


def run():
    ok = tot = 0

    def check(name, cond):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    # orthogonal keys -> near-zero crowding
    ortho = [unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)]
    cam, _ = make(ortho)
    m, mx = cam._index_crowding(cam._state())
    check(f"orthogonal keys low crowding (mean={m}, max={mx})", m is not None and m < 0.1 and mx < 0.1)

    # two near-duplicate keys -> high max crowding (collision risk)
    crowd = [unit(1, 0, 0), unit(0.99, 0.14, 0), unit(0, 0, 1)]
    cam2, _ = make(crowd)
    m2, mx2 = cam2._index_crowding(cam2._state())
    check(f"near-duplicate keys high max crowding (max={mx2})", mx2 > 0.95)

    # <2 keys -> None
    cam3, _ = make([unit(1, 0, 0)])
    m3, mx3 = cam3._index_crowding(cam3._state())
    check("single key -> None crowding", m3 is None and mx3 is None)

    # stats surfaces the health fields
    s = cam.stats()
    check("stats has index_size", s.get("index_size") == 3)
    check("stats has index_nn_cos_mean", "index_nn_cos_mean" in s and isinstance(s["index_nn_cos_mean"], float))
    check("stats has deliver_tau", s.get("deliver_tau") == 0.70)
    check("stats last_save_age_s None when never saved", s.get("last_save_age_s") is None)
    check("stats recovered_from_backup False", s.get("recovered_from_backup") is False)

    # recovered + saved store surfaces the alert + an age
    cam4, _ = make(ortho, recovered=True, last_save=1.0)
    s4 = cam4.stats()
    check("recovered_from_backup True surfaces", s4.get("recovered_from_backup") is True)
    check("last_save_age_s is a number when saved", isinstance(s4.get("last_save_age_s"), float))

    # sampling path (n > sample): build 400 keys, ensure it runs and returns floats
    many = [F.normalize(torch.randn(8), dim=-1) for _ in range(400)]
    cam5, _ = make(many)
    m5, mx5 = cam5._index_crowding(cam5._state(), sample=64)
    check("sampled crowding on large index returns floats", isinstance(m5, float) and isinstance(mx5, float))
    check("stats index_size reflects large store", cam5.stats().get("index_size") == 400)

    print(f"\nSTORE HEALTH: {ok}/{tot}")
    return ok == tot


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
