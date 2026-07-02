"""End-to-end validation of the HF-Qwen3.5 -> native gdn_hip patch (minisgl.gdn.hf_patch).

Loads the real frozen Qwen3.5-4B, runs a training-style forward+backward through the stock HF layers
(fla -> torch fallback in this image = the reference), then patches every gated-delta-net layer to the
native gdn_hip path and re-runs. Compares forward (logits) and dL/d_inputs_embeds. Confirms the patched
model (a) runs without hang, (b) matches the reference forward, (c) matches the reference gradient — i.e.
the frozen base behaves the same and the tap's gradient path is intact, but through gdn_hip not fla.

Run under a GPU lease in the combined image:
    PYTHONPATH=/engine/python:/engine python /engine/tools/cam_native_gdn_validate.py
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

DEV = "cuda"
MODEL = os.environ.get("CAM_BASE_MODEL", "Qwen/Qwen3.5-4B")


def _metrics(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (F.cosine_similarity(a, b, dim=0).item(),
            (a - b).abs().max().item(),
            ((a - b).norm() / (b.norm() + 1e-9)).item())


def main():
    from transformers import AutoModelForCausalLM

    from minisgl.gdn.hf_patch import patch_qwen3_5_gdn

    assert torch.cuda.is_available(), "needs a GPU (run under a lease)"
    print(f"=== CAM native-GDN patch validation ({MODEL}, device={torch.cuda.get_device_name()}) ===")
    torch.manual_seed(0)

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    H = model.config.hidden_size if hasattr(model.config, "hidden_size") else \
        model.config.text_config.hidden_size
    T = 24
    x0 = torch.randn(1, T, H, device=DEV, dtype=torch.bfloat16)
    proj = torch.randn(H, device=DEV, dtype=torch.float32)  # fixed scalarizer for L

    def run(tag):
        x = x0.clone().requires_grad_(True)
        # last-hidden scalarizer keeps memory tiny and exercises the full stack incl. every GDN layer
        out = model(inputs_embeds=x, output_hidden_states=True, use_cache=False).hidden_states[-1]
        L = (out.float().reshape(T, -1) @ proj).sum()
        (g,) = torch.autograd.grad(L, x)
        print(f"  [{tag}] fwd last-hidden norm={out.float().norm().item():.4f}  "
              f"grad norm={g.float().norm().item():.4f}")
        return out.detach(), g.detach()

    print("--- reference: stock HF layers (fla -> torch fallback) ---")
    ref_out, ref_grad = run("ref ")

    print("--- patching gated-delta-net layers -> native gdn_hip ---")
    n = patch_qwen3_5_gdn(model)
    print(f"  patched {n} Qwen3_5GatedDeltaNet layers")
    assert n > 0, "no GDN layers patched — class-name detection failed"

    print("--- patched: native gdn_hip path ---")
    new_out, new_grad = run("nat ")

    fcos, fdmax, frel = _metrics(new_out, ref_out)
    gcos, gdmax, grel = _metrics(new_grad, ref_grad)
    print("\n" + "=" * 70)
    print(f"  forward  (last hidden) : cos={fcos:.6f}  max|Δ|={fdmax:.3e}  relL2={frel:.3e}")
    print(f"  backward (dL/d_embeds) : cos={gcos:.6f}  max|Δ|={gdmax:.3e}  relL2={grel:.3e}")
    # 24 stacked GDN layers compound the per-layer ~1e-3 wmma/bf16 gap; require strong agreement, not exact
    ok = fcos >= 0.99 and gcos >= 0.99 and torch.isfinite(new_grad).all().item()
    print("RESULT:", "PASS — patched model matches fla reference in fwd AND grad (no fla, no hang)"
          if ok else "FAIL (see above)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
