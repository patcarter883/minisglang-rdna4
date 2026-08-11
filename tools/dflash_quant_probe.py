"""Per-tensor quantization error for a DFlash drafter: NVFP4 (W4) vs fp8-per-channel.

Why measure rather than assume. Meta's own GGUF build of this drafter is NOT uniform — it is
Q4_K for most tensors but Q6_K for `ffn_down`, and F32 for every norm. That is them saying which
tensors are quantization-sensitive. This checks that claim against the actual weights, so the
mixed-precision carve-out in `_PlainLinear` is evidence-led rather than cargo-culted.

Error is reported as relative L2 (‖Q(w)−w‖ / ‖w‖), which is scale-free and comparable across
tensors of different magnitude. It is a PROXY: what actually matters for a drafter is acceptance
rate, and only a serve measures that. But a tensor that quantizes far worse than its neighbours is
the one to spend bits on.

CPU-only, no GPU lease. Run:
  docker run --rm -v <worktree>:/engine -v ~/.cache/huggingface:/root/.cache/huggingface \
    --entrypoint bash minisgl-rdna4:lean -lc \
    'PYTHONPATH=/engine/python:/engine:/opt/kernels python /engine/tools/dflash_quant_probe.py'
"""
from __future__ import annotations

import glob
import os
import sys

import torch
from safetensors import safe_open

from minisgl.quant.nvfp4 import dequantize_nvfp4_folded, quantize_nvfp4_rtn

DEFAULT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B-assistant/snapshots/*/"
)


def fp8_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel fp8 e4m3 RTN — exactly what `_PlainLinear.load_quant('fp8')` does."""
    amax = w.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    s = amax / 448.0
    return (w.float() / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def main() -> int:
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    if folder is None:
        hits = glob.glob(DEFAULT)
        if not hits:
            sys.exit("drafter not cached; pass its folder as argv[1]")
        folder = hits[0]
    paths = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    if not paths:
        sys.exit(f"no .safetensors in {folder}")

    rows: list[tuple[str, float, float, int]] = []
    with safe_open(paths[0], framework="pt") as h:
        keys = [k for k in h.keys() if h.get_slice(k).get_shape().__len__() == 2]
        for k in sorted(keys):
            w = h.get_tensor(k).float()
            if w.shape[1] % 16:
                continue  # group-16 needs K % 16 == 0
            packed, scales = quantize_nvfp4_rtn(w)
            n4 = rel_l2(dequantize_nvfp4_folded(packed, scales), w)
            n8 = rel_l2(fp8_roundtrip(w), w)
            rows.append((k, n4, n8, w.numel()))

    print(f"{'tensor':44s} {'nvfp4':>8s} {'fp8':>8s} {'ratio':>7s}   params")
    print("-" * 82)
    for k, n4, n8, numel in rows:
        print(f"{k:44s} {n4:8.4f} {n8:8.4f} {n4 / max(n8, 1e-9):7.1f}x  {numel / 1e6:7.1f}M")

    # Group by leaf so the per-KIND pattern is visible — that is what a carve-out keys on.
    print("\nby leaf kind (mean relative L2):")
    kinds: dict[str, list[tuple[float, float, int]]] = {}
    for k, n4, n8, numel in rows:
        leaf = k.split(".")[-2] if "." in k else k
        kinds.setdefault(leaf, []).append((n4, n8, numel))
    worst = None
    for leaf, vs in sorted(kinds.items()):
        m4 = sum(v[0] for v in vs) / len(vs)
        m8 = sum(v[1] for v in vs) / len(vs)
        tot = sum(v[2] for v in vs)
        print(f"  {leaf:22s} nvfp4={m4:.4f}  fp8={m8:.4f}   {tot / 1e6:7.1f}M params")
        if worst is None or m4 > worst[1]:
            worst = (leaf, m4)
    print(f"\nworst leaf under nvfp4: {worst[0]} ({worst[1]:.4f})")

    # Size of the carve-out: keeping a leaf at fp8 costs 8 bits/param instead of ~4.5.
    tot = sum(r[3] for r in rows)
    for leaf in sorted(kinds):
        sz = sum(v[2] for v in kinds[leaf])
        allv4 = tot * 0.5625 / 2**30
        mixed = ((tot - sz) * 0.5625 + sz * 1.0) / 2**30
        print(f"  keep {leaf:22s} at fp8 -> {mixed:.2f} GiB (vs {allv4:.2f} all-nvfp4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
