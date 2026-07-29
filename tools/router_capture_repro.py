"""Minimal repro: does moe_topk_softmax hang under CUDA graph capture?

The serve wedged at "Capturing graphs: bs=8, 0%" with both ranks alive. This isolates the op in a
plain torch.cuda.graph() capture — no model, no TP — so the failure can be iterated on in seconds.
"""
import torch
import moe_hip

E, K = 256, 8
for M in (1, 2, 8):
    g = torch.randn(M, E, device="cuda").float()
    # eager first: known-good
    tw, ti = moe_hip.moe_topk_softmax(g, K, True)
    torch.cuda.synchronize()
    print(f"M={M} eager OK", flush=True)

    # warm up on a side stream (required before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            moe_hip.moe_topk_softmax(g, K, True)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    print(f"M={M} warmup OK", flush=True)

    gr = torch.cuda.CUDAGraph()
    print(f"M={M} entering capture...", flush=True)
    with torch.cuda.graph(gr):
        ctw, cti = moe_hip.moe_topk_softmax(g, K, True)
    print(f"M={M} CAPTURED OK", flush=True)
    gr.replay()
    torch.cuda.synchronize()
    print(f"M={M} REPLAY OK  ids_match={bool((cti==ti).all())}", flush=True)
print("ALL OK — capture is not the problem")
