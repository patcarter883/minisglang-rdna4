import torch
s=torch.load('/engine/tools/_cbz_single.pt'); c=torch.load('/engine/tools/_cbz_chunk.pt')
print("single rows", next(iter(s['QK'].values())).shape, "chunk rows", next(iter(c['QK'].values())).shape)
for lid in (0,):
    a=s['QK'][lid]; b=c['QK'][lid]
    print(f"\nlayer{lid} QK shape single{tuple(a.shape)} chunk{tuple(b.shape)}")
    # per-token max abs diff, show region 0..50
    d=(a-b).abs().max(dim=1).values
    print("tokens 0..49 |Δ|:", [f"{i}:{d[i]:.4f}" for i in range(50) if d[i]>1e-6])
    print("all divergent:", (d>1e-6).nonzero().flatten().tolist())
    # is chunk1 region [0:48) identical? check token 33 value magnitude
    print("token33 single[:6]:", a[33,:6].tolist())
    print("token33 chunk [:6]:", b[33,:6].tolist())
    print("token32 |Δ|:", (a[32]-b[32]).abs().max().item(), " token34 |Δ|:", (a[34]-b[34]).abs().max().item())
