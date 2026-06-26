"""Disambiguate the attn_hip parity failure: is the error driven by causal masking or by GQA?
Runs the 2x2 (causal x gqa) and localizes where max|Δ| lands (head/row/col)."""
from __future__ import annotations
import torch, torch.nn.functional as F
import attn_hip  # noqa

DEV = "cuda"
torch.manual_seed(0)


def ref_attention(q, k, v, scale, causal, sliding_window):
    S, Hq, D = q.shape
    Hk = k.shape[1]; rep = Hq // Hk
    qf = q.float().permute(1, 0, 2)
    kf = k.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    i = torch.arange(S, device=q.device)
    if causal:
        mask = i[:, None] < i[None, :]
        if sliding_window > 0:
            mask = mask | ((i[:, None] - i[None, :]) >= sliding_window)
        attn = attn.masked_fill(mask[None], float("-inf"))
    out = torch.matmul(F.softmax(attn, dim=-1), vf)
    return out.permute(1, 0, 2).contiguous()


def run(name, S, Hq, Hk, D, causal, sw=0):
    scale = D ** -0.5
    q = torch.randn(S, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(S, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(S, Hk, D, device=DEV, dtype=torch.bfloat16)
    got = torch.ops.attn_hip.flash_prefill(q, k, v, scale, causal, sw).float()
    ref = ref_attention(q, k, v, scale, causal, sw)
    diff = (got - ref).abs()
    dmax = diff.max().item()
    # locate
    idx = diff.argmax().item()
    Hd = D
    pos = idx // (Hq * Hd); h = (idx // Hd) % Hq; d = idx % Hd
    # per-head max
    perhead = diff.amax(dim=(0, 2))  # [Hq]
    gqa = Hq // Hk
    print(f"  {name:38s} gqa={gqa} causal={causal} sw={sw}: max|Δ|={dmax:.3e} "
          f"@ (pos={pos},head={h},d={d})")
    print(f"     per-head max|Δ|: {[round(x,4) for x in perhead.tolist()]}")
    # is the error concentrated in row 0 / early rows?
    perrow = diff.amax(dim=(1, 2))  # [S]
    top = torch.topk(perrow, min(5, S))
    print(f"     worst rows (pos): {[(int(i),round(float(perrow[i]),4)) for i in top.indices.tolist()]}")


def ref_bf16P(q, k, v, scale, causal, sliding_window):
    """EXACT mirror of the kernel's arithmetic: fp32 softmax, then round P->bf16, then
    (bf16P @ fp32V) / sum(bf16P). If the kernel matches THIS (not the fp32 ref), the kernel
    is correct and the residual vs fp32 is the unavoidable bf16-P floor."""
    S, Hq, D = q.shape
    Hk = k.shape[1]; rep = Hq // Hk
    qf = q.float().permute(1, 0, 2)
    kf = k.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    vf = v.float().repeat_interleave(rep, dim=1).permute(1, 0, 2)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * scale
    i = torch.arange(S, device=q.device)
    if causal:
        mask = i[:, None] < i[None, :]
        if sliding_window > 0:
            mask = mask | ((i[:, None] - i[None, :]) >= sliding_window)
        attn = attn.masked_fill(mask[None], float("-inf"))
    # Mirror the kernel EXACTLY: subtract rowmax, exp the UN-normalized scores, round THAT to
    # bf16 (this is the P fed to the WMMA), then l = sum(bf16 exp) and out = (bf16P @ V)/l.
    # (NB: rounding the *normalized* softmax instead — as a naive bf16 ref does — rounds at a
    # different point and can differ from the kernel by ~2x the bf16 floor, a false alarm.)
    m = attn.amax(dim=-1, keepdim=True)
    e = torch.exp(attn - m)
    eb = e.bfloat16().float()
    l = eb.sum(-1, keepdim=True).clamp_min(1e-20)
    out = torch.matmul(eb, vf) / l
    return out.permute(1, 0, 2).contiguous()


def cmp_floor(name, S, Hq, Hk, D, causal, sw=0):
    scale = D ** -0.5
    q = torch.randn(S, Hq, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(S, Hk, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(S, Hk, D, device=DEV, dtype=torch.bfloat16)
    got = torch.ops.attn_hip.flash_prefill(q, k, v, scale, causal, sw).float()
    rf = ref_attention(q, k, v, scale, causal, sw)
    rf_bf16 = rf.bfloat16().float()   # fair floor: the kernel RETURNS bf16, so round the ref too
    cos = F.cosine_similarity(got.flatten(), rf.flatten(), dim=0).item()
    print(f"  {name}: vs-fp32={ (got-rf).abs().max():.3e}  "
          f"vs-bf16(fp32ref)={ (got-rf_bf16).abs().max():.3e}  "
          f"cos={cos:.6f}  |out|max={rf.abs().max():.3f}")


def main():
    print("=== kernel-faithful FLOOR test (single KV block, S<=32: no online confound) ===")
    cmp_floor("causal gqa=1 Hq8/Hk8 S32 ", 32, 8, 8, 128, 1)
    cmp_floor("causal gqa=4 Hq8/Hk2 S32 ", 32, 8, 2, 128, 1)
    cmp_floor("causal gqa=1 Hq4/Hk4 S16 ", 16, 4, 4, 128, 1)
    cmp_floor("noncausal gqa=1 Hk8 S32   ", 32, 8, 8, 128, 0)
    print("=== 2x2 causal x gqa disambiguation (D128 S96) ===")
    run("causal  gqa=1  Hq8/Hk8 ", 96, 8, 8, 128, causal=1)
    run("noncausal gqa=8 Hq16/Hk2", 96, 16, 2, 128, causal=0)
    run("causal  gqa=8  Hq16/Hk2", 96, 16, 2, 128, causal=1)
    run("noncausal gqa=1 Hq8/Hk8 ", 96, 8, 8, 128, causal=0)
    print("=== minimal: single kv block, causal gqa>1 ===")
    run("causal gqa=4 Hq8/Hk2 S32", 32, 8, 2, 128, causal=1)
    run("causal gqa=1 Hq2/Hk2 S32", 32, 2, 2, 128, causal=1)


if __name__ == "__main__":
    main()
