import torch
s=torch.load('/engine/tools/_cbz_single.pt'); c=torch.load('/engine/tools/_cbz_chunk.pt')
for name in ('QK',):
  for lid in (0,1):
    a=s[name][lid]; b=c[name][lid]; n=min(a.shape[0],b.shape[0])
    a=a[:n].reshape(n,-1); b=b[:n].reshape(n,-1); w=min(a.shape[1],b.shape[1])
    d=(a[:,:w]-b[:,:w]).abs().max(dim=1).values
    nz=(d>1e-6).nonzero().flatten().tolist()
    print(f"[{name}] L{lid} rows={n} n_divergent={len(nz)}")
    print(f"   divergent token idxs: {nz[:40]}")
    print(f"   |Δ| at those: {[round(d[i].item(),4) for i in nz[:20]]}")
