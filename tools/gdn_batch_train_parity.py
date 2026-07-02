"""Parity: the BATCHED native prefill-train (_prefill_train_batch, one varlen op call for all B seqs)
vs the per-sequence loop (_prefill_train_one_seq stacked) — the path it replaces in NativeGDNShim.
Checks the forward output AND the full backward (dL/d_input + dL/d_params) match. This is the correctness
gate for the native-GDN speed fix (kills the per-seq Python loop that made native ~3x slower than fla).

Run under a GPU lease in the combined/CAM image:
    PYTHONPATH=/minisgl/python:/minisgl python /minisgl/tools/gdn_batch_train_parity.py
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

import os
from minisgl.gdn.hf_patch import NativeGDNShim

DEV = "cuda"
MODEL = os.environ.get("CAM_BASE_MODEL", "Qwen/Qwen3.5-4B")


def _real_gdn_layer():
    """A NativeGDNShim built from a REAL Qwen3.5 GDN layer (trained weights — a random layer overflows the
    gated-delta scan). Mirrors cam_native_gdn_validate's setup."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    for _p, m in model.named_modules():
        if type(m).__name__ == "Qwen3_5GatedDeltaNet":
            return NativeGDNShim(m).to(DEV).ms
    raise RuntimeError("no Qwen3_5GatedDeltaNet layer found")


def _metrics(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (F.cosine_similarity(a, b, dim=0).item(),
            (a - b).abs().max().item(),
            ((a - b).norm() / (b.norm() + 1e-9)).item())


def main():
    assert torch.cuda.is_available(), "needs a GPU (run under a lease)"
    torch.manual_seed(0)
    # a small but representative GDN layer (Qwen3.5 geometry: GQA, head dims)
    ms = _real_gdn_layer()               # real trained GDN layer (bf16)
    for p in ms.parameters():
        p.requires_grad_(True)           # exercise param grads too
    Hd = ms.in_proj_qkvz.weight.shape[1]
    B, T = 6, 40
    x = torch.randn(B, T, Hd, dtype=torch.bfloat16, device=DEV)
    go = torch.randn(B, T, ms.out_proj.weight.shape[0], dtype=torch.bfloat16, device=DEV)   # out = hidden_size

    def run(batched):
        for p in ms.parameters():
            if p.grad is not None:
                p.grad = None
        xi = x.detach().clone().requires_grad_(True)
        out = ms._prefill_train_batch(xi) if batched else torch.stack(
            [ms._prefill_train_one_seq(xi[i]) for i in range(B)], dim=0)
        out.backward(go)
        pg = {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in ms.named_parameters()}
        return out.detach(), xi.grad.detach(), pg

    print("=== batched vs per-seq-loop prefill-train parity (B=%d T=%d) ===" % (B, T))
    out_l, gx_l, pg_l = run(False)
    print(f"[diag] loop:    out finite={torch.isfinite(out_l).all().item()} norm={out_l.float().norm():.3f} "
          f"| gx finite={torch.isfinite(gx_l).all().item()}")
    out_b, gx_b, pg_b = run(True)
    print(f"[diag] batched: out finite={torch.isfinite(out_b).all().item()} norm={out_b.float().norm():.3f} "
          f"| gx finite={torch.isfinite(gx_b).all().item()}")
    fc, fm, fr = _metrics(out_b, out_l)
    print(f"forward   : cos={fc:.6f} max|Δ|={fm:.3e} relL2={fr:.3e}")
    xc, xm, xr = _metrics(gx_b, gx_l)
    print(f"dL/d_input: cos={xc:.6f} max|Δ|={xm:.3e} relL2={xr:.3e}")
    worst = (1.0, "")
    for n in pg_l:
        if pg_l[n] is None or pg_b[n] is None:
            continue
        c, m, r = _metrics(pg_b[n], pg_l[n])
        print(f"dL/d_{n:22s}: cos={c:.6f} max|Δ|={m:.3e} relL2={r:.3e}")
        if c < worst[0]:
            worst = (c, n)
    ok = fc > 0.9995 and xc > 0.999 and worst[0] > 0.999
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} — batched matches the per-seq loop "
          f"(fwd cos {fc:.5f}, dL/dx cos {xc:.5f}, worst param-grad cos {worst[0]:.5f} @ {worst[1]})")


if __name__ == "__main__":
    main()
