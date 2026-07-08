"""CPU pointer-roundtrip check (#100 serving): remember multi-token objects, deliver them back via the
store-addressed id-bank. Deterministic (addressing only needs write-slot==read-slot), so it validates
the serving wiring with random embeds + a REAL checkpoint — no GPU / no model semantics. Mount /ckpt."""
import sys, torch
sys.path.insert(0, "/minisgl/python")
from minisgl.cam.memory import CAMMemory

V, H = 4096, 2560
emb = torch.nn.Embedding(V, H); lm = torch.randn(V, H)
mem = CAMMemory("/ckpt", emb, lm)
assert mem.enabled, "FAIL: constructed DISABLED"
print(f"[ptr] loaded ckpt | n_banks={mem.n_banks} mt_cap={mem.mt_cap}")

def _remember(subj, obj):
    pl = torch.randn(V); pl[obj[0]] = -20.0        # base-low on first tok -> write gate passes
    mem.set_pending_object(obj)
    return mem.remember(subj, pl)

subj, obj = [11, 22, 33], [5, 9, 13, 21]           # 4-token object
assert _remember(subj, obj), "FAIL: remember gate rejected a base-low fact"
got = mem.deliver_object_ids(subj)
print(f"[ptr] remember {obj} -> deliver {got}")
assert got == obj, f"POINTER FAIL: {got} != {obj}"

subj2, obj2 = [44, 55, 66], [7, 3, 99]             # second subject, different bucket/slots
_remember(subj2, obj2)
assert mem.deliver_object_ids(subj2) == obj2, "FAIL: subject2 delivery"
assert mem.deliver_object_ids(subj) == obj, "FAIL: subject1 corrupted by subject2 (collision)"
print("[ptr] multi-subject: no cross-talk")

assert mem.forget(subj), "FAIL: forget"
assert mem.deliver_object_ids(subj) == [], f"FAIL: forget left {mem.deliver_object_ids(subj)}"
assert mem.deliver_object_ids(subj2) == obj2, "FAIL: forget clobbered a survivor"
print("[ptr] forget clears the pointer; survivor intact")
print("[ptr] POINTER SERVING ROUNDTRIP OK")
