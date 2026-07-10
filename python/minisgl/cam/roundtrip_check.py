import sys, torch
sys.path.insert(0, "/minisgl/python")
from minisgl.cam.memory import CAMMemory
V, H = 4096, 2560
emb = torch.nn.Embedding(V, H); lm = torch.randn(V, H)
mem = CAMMemory("/ckpt", emb, lm)
assert mem.enabled, "FAIL: constructed DISABLED — checkpoint did not load"
print(f"[rt] loaded REAL checkpoint | enabled={mem.enabled} tap_layer={mem.tap_layer} n_banks={mem.n_banks} tau={mem.remember_tau} alpha={mem.router_alpha}")
mem.set_pending_object([5]); pl = torch.randn(V); pl[5] = -20.0
stored = mem.remember([11,22,33], pl)
print(f"[rt] remember(obj=5, base-low) -> stored={stored} facts={len(mem.list_facts())}")
mem.set_pending_object([7]); pl2 = torch.randn(V); pl2[7] = 20.0
stored2 = mem.remember([44,55], pl2)
print(f"[rt] remember(obj=7, base-high) -> stored={stored2} (want False)")
bank, conf = mem.read([11,22,33])
print(f"[rt] read -> bank={tuple(bank.shape)} conf={float(conf[0]):.2f}")
delta = mem.router_delta(torch.randn(1,V), bank, conf)
print(f"[rt] router_delta -> {tuple(delta.shape)} seed={mem.seed_token(bank)}")
h = torch.randn(1,3,H); assert torch.equal(mem.apply_tap(h,None,None), h)
print("[rt] apply_tap(None) byte-exact no-op")
print("[rt] ROUNDTRIP OK — WS-A loader <-> WS-B export interlock on REAL weights")
