"""Compare minisgl vs HF per-layer last-token hidden states (cos-sim + rel-err)."""
import sys

import torch

a = torch.load(sys.argv[1])  # minisgl stack [num_layers+1, hidden]
b = torch.load(sys.argv[2])  # hf stack
print(f"minisgl {tuple(a.shape)}  hf {tuple(b.shape)}")
n = min(a.shape[0], b.shape[0])
for i in range(n):
    x, y = a[i].float(), b[i].float()
    cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
    rel = ((x - y).norm() / (y.norm() + 1e-9)).item()
    flag = "  <-- DIVERGE" if cos < 0.99 else ""
    print(f"hs[{i:02d}]: cos={cos:.5f} rel={rel:.3e} |ms|={x.norm():.2f} |hf|={y.norm():.2f}{flag}")
