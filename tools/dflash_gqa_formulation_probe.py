"""Probe: which GQA formulation is BIT-IDENTICAL to repeat_interleave + einsum, and what does it cost?

The shipped grouped einsum ("bkgd,skd->bkgs") changes the underlying bmm shape from
(batch=H, M=B, K=hd) to (batch=Hkv, M=B*g, K=hd) and the AV product from (batch=H, K=S) to
(batch=Hkv, K=S) — a different rocBLAS partition, hence a ~1-bf16-ULP move. Candidate fix: broadcast
matmul with a stride-0 group axis, which keeps batch=H and the original (M, N, K) exactly.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/engine/python")

B, H, HKV, HD = 16, 64, 8, 128
G = H // HKV


def legs(q, K, V, mask):
    out = {}

    # --- reference: materialise the group-expanded K/V (what ships today at d276137c) -------------
    Kr = K.repeat_interleave(G, dim=1)
    Vr = V.repeat_interleave(G, dim=1)
    s = torch.einsum("bhd,shd->bhs", q, Kr) * (HD ** -0.5)
    s = s + mask.unsqueeze(1)
    p = s.softmax(dim=-1).to(V.dtype)
    out["ref"] = torch.einsum("bhs,shd->bhd", p, Vr)

    # --- A: grouped einsum (the shipped rewrite) --------------------------------------------------
    q4 = q.view(B, HKV, G, HD)
    s = torch.einsum("bkgd,skd->bkgs", q4, K) * (HD ** -0.5)
    s = s + mask[:, None, None, :]
    p = s.softmax(dim=-1).to(V.dtype)
    out["grouped_einsum"] = torch.einsum("bkgs,skd->bkgd", p, V).reshape(B, H, HD)

    # --- C: broadcast matmul with a STRIDE-0 group axis -------------------------------------------
    # [HKV, G, B, HD] @ [HKV, 1->G, HD, S] : batch H, M=B, N=S, K=HD  -- the reference's exact shape.
    qc = q.view(B, HKV, G, HD).permute(1, 2, 0, 3)              # [HKV,G,B,HD]
    Kc = K.permute(1, 2, 0).unsqueeze(1)                        # [HKV,1,HD,S]
    s = torch.matmul(qc, Kc) * (HD ** -0.5)                     # [HKV,G,B,S]
    s = s + mask.permute(0, 1).unsqueeze(0).unsqueeze(0)        # [1,1,B,S]
    p = s.softmax(dim=-1).to(V.dtype)
    Vc = V.permute(1, 0, 2).unsqueeze(1)                        # [HKV,1,S,HD]
    out["bcast_matmul"] = torch.matmul(p, Vc).permute(2, 0, 1, 3).reshape(B, H, HD)
    return out


def bench(fn, iters=7):
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):  # MIN-of-N
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize()
        a.record(); fn(); b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b))
    return best


def main() -> int:
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(7)
    for S in (80, 272, 528, 1040, 4112):
        q = torch.randn((B, H, HD), generator=g, device=dev, dtype=torch.float32).bfloat16()
        K = torch.randn((S, HKV, HD), generator=g, device=dev, dtype=torch.float32).bfloat16()
        V = torch.randn((S, HKV, HD), generator=g, device=dev, dtype=torch.float32).bfloat16()
        mask = torch.zeros((B, S), device=dev, dtype=torch.float32)
        r = legs(q, K, V, mask)
        ref = r["ref"]
        line = [f"S={S:>5}"]
        for name in ("grouped_einsum", "bcast_matmul"):
            eq = torch.equal(ref, r[name])
            d = (ref.float() - r[name].float()).abs().max().item()
            line.append(f"{name}: eq={str(eq):>5} dmax={d:.2e}")
        # cost
        def _ref():
            Kr = K.repeat_interleave(G, dim=1); Vr = V.repeat_interleave(G, dim=1)
            s = torch.einsum("bhd,shd->bhs", q, Kr) * (HD ** -0.5) + mask.unsqueeze(1)
            return torch.einsum("bhs,shd->bhd", s.softmax(-1).to(V.dtype), Vr)

        def _bc():
            qc = q.view(B, HKV, G, HD).permute(1, 2, 0, 3)
            s = torch.matmul(qc, K.permute(1, 2, 0).unsqueeze(1)) * (HD ** -0.5)
            s = s + mask.unsqueeze(0).unsqueeze(0)
            p = s.softmax(-1).to(V.dtype)
            return torch.matmul(p, V.permute(1, 0, 2).unsqueeze(1)).permute(2, 0, 1, 3).reshape(B, H, HD)

        def _ge():
            s = torch.einsum("bkgd,skd->bkgs", q.view(B, HKV, G, HD), K) * (HD ** -0.5)
            s = s + mask[:, None, None, :]
            return torch.einsum("bkgs,skd->bkgd", s.softmax(-1).to(V.dtype), V).reshape(B, H, HD)

        line.append(f"ms ref={bench(_ref):.3f} ge={bench(_ge):.3f} bc={bench(_bc):.3f}")
        print("  ".join(line), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
