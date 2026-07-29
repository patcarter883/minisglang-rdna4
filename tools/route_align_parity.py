"""Fused route+align vs the three separate kernels: exact-equivalence + speed."""
import time, torch, moe_hip

def sep(g, K, renorm, E, BM, want_s2r):
    tw, ti = moe_hip.moe_topk_softmax(g, K, renorm)
    sid, eid, ntp = moe_hip.moe_align(ti, E, BM)
    s2r = None
    if want_s2r:
        numel = g.shape[0] * K
        s2r = torch.full((numel,), -1, dtype=torch.int32, device=g.device)
        n = int(ntp[0])
        rows = torch.arange(n, dtype=torch.int32, device=g.device)
        slots = sid[:n]
        m = slots < numel
        s2r[slots[m].long()] = rows[m]
    return tw, ti, sid, eid, ntp, s2r

def bench(fn, iters=150, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)*1e6/iters

E, K, BM = 256, 8, 16
torch.manual_seed(0)
print(f"{'M':>3}{'renorm':>8}{'tw==':>7}{'ti==':>7}{'sid==':>7}{'eid==':>7}{'ntp==':>7}{'s2r==':>7}{'sep us':>9}{'fused us':>10}{'x':>7}")
print("-"*82)
for M in (1,2,8,32):
    for renorm in (True, False):
        g = (torch.randn(M,E,device="cuda")*2).float()
        a = sep(g,K,renorm,E,BM,True)
        b = moe_hip.moe_route_align(g,K,renorm,E,BM,True)
        eq = [bool(torch.equal(x,y)) for x,y in zip(a[:5],b[:5])] + [bool(torch.equal(a[5],b[5]))]
        ts = bench(lambda: sep(g,K,renorm,E,BM,False))
        tf = bench(lambda: moe_hip.moe_route_align(g,K,renorm,E,BM,False))
        print(f"{M:>3}{str(renorm):>8}" + "".join(f"{str(e):>7}" for e in eq) + f"{ts:>9.1f}{tf:>10.1f}{ts/tf:>6.2f}x")
g = (torch.randn(1,E,device="cuda")*2).float()
s = bench(lambda: sep(g,K,True,E,BM,False)); f = bench(lambda: moe_hip.moe_route_align(g,K,True,E,BM,False))
print(f"\nx40 layers @M=1: separate {s*40/1000:.3f} ms/step -> fused {f*40/1000:.3f} ms/step (saves {(s-f)*40/1000:.3f})")
