"""Fused moe_topk_softmax vs the pure-torch router chain it replaces: parity + speed."""
import time
import torch
import moe_hip


def ref(g, k, renorm):
    probs = torch.softmax(g.float(), dim=-1)
    tw, ti = torch.topk(probs, k, dim=-1)
    if renorm:
        tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
    return tw.contiguous(), ti.to(torch.int32).contiguous()


def bench(fn, iters=200, warmup=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


print(f"{'shape':<26}{'renorm':>7}{'max|dW|':>11}{'ids match':>11}{'torch us':>10}{'fused us':>10}{'x':>7}")
print("-" * 84)
torch.manual_seed(0)
for M in (1, 2, 8, 32):
    for E, K in ((256, 8), (128, 4)):
        for renorm in (True, False):
            g = (torch.randn(M, E, device="cuda") * 2.0).float()
            rw, ri = ref(g, K, renorm)
            fw, fi = moe_hip.moe_topk_softmax(g, K, renorm)
            dW = (fw - rw).abs().max().item()
            same = bool((fi == ri).all().item())
            t_ref = bench(lambda: ref(g, K, renorm))
            t_fus = bench(lambda: moe_hip.moe_topk_softmax(g, K, renorm))
            print(f"{'M=%d E=%d K=%d' % (M, E, K):<26}{str(renorm):>7}{dW:>11.2e}{str(same):>11}"
                  f"{t_ref:>10.1f}{t_fus:>10.1f}{t_ref/t_fus:>6.2f}x")
print("\nx40 MoE layers at M=1, E=256, K=8 (the served shape):")
g = (torch.randn(1, 256, device="cuda") * 2.0).float()
a = bench(lambda: ref(g, 8, True)); b = bench(lambda: moe_hip.moe_topk_softmax(g, 8, True))
print(f"  torch chain {a*40/1000:.3f} ms/step -> fused {b*40/1000:.3f} ms/step "
      f"(saves {(a-b)*40/1000:.3f} ms/step)")
