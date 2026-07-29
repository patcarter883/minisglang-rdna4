"""Does the fused router emit an OUT-OF-RANGE expert id on non-finite logits?

CUDA-graph capture runs the model on DUMMY inputs, which can contain NaN/Inf. The top-k seeds
bv=-INFINITY, bi=E (out of range) and selects with `v > bv || (v == bv && e < bi)` — both false for
NaN, so a thread scanning only NaNs keeps bi=E. If that survives the block reduce, topk_ids contains
E, and moe_align's `atomicAdd(&cnt[topk_ids[t]], 1)` writes OUT OF BOUNDS in shared memory.
"""
import torch, moe_hip

E, K = 256, 8
cases = {
    "all NaN": torch.full((1, E), float("nan")),
    "all -inf": torch.full((1, E), float("-inf")),
    "mixed NaN": torch.where(torch.arange(E) % 2 == 0, torch.full((E,), float("nan")), torch.randn(E)).view(1, E),
    "normal": torch.randn(1, E),
}
bad = False
for name, g in cases.items():
    g = g.cuda().float()
    tw, ti = moe_hip.moe_topk_softmax(g, K, True)
    torch.cuda.synchronize()
    mn, mx = int(ti.min()), int(ti.max())
    oob = mx >= E or mn < 0
    bad |= oob
    print(f"{name:<12} ids[min={mn}, max={mx}]  OUT-OF-RANGE={oob}  ids={ti.flatten().tolist()[:8]}")
print("\nHYPOTHESIS CONFIRMED — router can emit id==E" if bad else "\nnot reproduced")
