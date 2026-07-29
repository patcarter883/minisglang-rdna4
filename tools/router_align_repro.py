"""Capture the SERVE sequence: fused router -> moe_align, the way w4a8_moe chains them."""
import torch, moe_hip

E, K, BM = 256, 8, 16
for M in (8, 4, 2, 1):
    g = torch.randn(M, E, device="cuda").float()

    def seq():
        tw, ti = moe_hip.moe_topk_softmax(g, K, True)
        tw = tw.to(torch.float32).contiguous()
        ti = ti.to(torch.int32).contiguous()
        return moe_hip.moe_align(ti, E, BM)

    seq(); torch.cuda.synchronize(); print(f"M={M} eager OK", flush=True)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            seq()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    print(f"M={M} capturing router->align...", flush=True)
    with torch.cuda.graph(gr):
        out = seq()
    print(f"M={M} CAPTURED", flush=True)
    gr.replay(); torch.cuda.synchronize()
    print(f"M={M} REPLAY OK ntp={int(out[2][0])}", flush=True)
print("PAIR OK")
