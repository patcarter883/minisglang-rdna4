"""Phase 3d-1 — CPU check for partial rotary in RotaryEmbedding.

Qwen3.5 uses partial rotary (rotary_dim=64 of head_dim=256): rotate the first
rotary_dim dims of each head with NeoX rotate-half, pass the rest through. This
asserts minisgl's RotaryEmbedding._apply matches the HF-style reference for BOTH
the partial case AND the full case (rotary_dim == head_dim regression), and that
the non-rotary tail is left bit-identical. Pure torch on CPU — no GPU, no weights.

Run with a torch-capable interpreter, e.g.:
    /home/pat/.conda/envs/sglang-triton36/bin/python tools/rotary_partial_test.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_rotary():
    """Load layers/rotary.py in isolation (stub the `.base` relative import) so the
    test does not drag in the full minisgl package (whose __init__ pulls transformers)."""
    pkg = types.ModuleType("rt_pkg")
    pkg.__path__ = []  # mark as package so relative import works
    base = types.ModuleType("rt_pkg.base")
    base.StateLessOP = type("StateLessOP", (), {})
    sys.modules["rt_pkg"] = pkg
    sys.modules["rt_pkg.base"] = base
    path = Path(__file__).resolve().parent.parent / "python/minisgl/layers/rotary.py"
    spec = importlib.util.spec_from_file_location("rt_pkg.rotary", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rt_pkg.rotary"] = mod
    spec.loader.exec_module(mod)
    return mod.RotaryEmbedding


def _ref_rope(x, positions, *, head_size, rotary_dim, base):
    """HF-style reference partial NeoX rope on x: (n, num_heads, head_size)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
    freqs = positions[:, None].float() * inv_freq[None, :]  # (n, rotary_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (n, rotary_dim)
    cos = emb.cos()[:, None, :]
    sin = emb.sin()[:, None, :]
    x = x.float()
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    def rotate_half(t):
        h = t.shape[-1] // 2
        return torch.cat([-t[..., h:], t[..., :h]], dim=-1)

    out_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([out_rot, x_pass], dim=-1)


def _case(RotaryEmbedding, *, head_size, rotary_dim, base, n=7, num_heads=3, tag=""):
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_size, rotary_dim, max_position_embeddings=4096, base=base)
    positions = torch.randint(0, 4096, (n,))
    q = torch.randn(n, num_heads, head_size)
    k = torch.randn(n, num_heads, head_size)

    q_out, k_out = rope.forward(positions, q.reshape(n, -1), k.reshape(n, -1))
    q_out = q_out.reshape(n, num_heads, head_size)
    k_out = k_out.reshape(n, num_heads, head_size)

    q_ref = _ref_rope(q, positions, head_size=head_size, rotary_dim=rotary_dim, base=base)
    k_ref = _ref_rope(k, positions, head_size=head_size, rotary_dim=rotary_dim, base=base)

    dq = (q_out.float() - q_ref).abs().max().item()
    dk = (k_out.float() - k_ref).abs().max().item()
    # tail must be identical to the input (passed through unrotated)
    tail = 0.0
    if rotary_dim < head_size:
        tail = (q_out[..., rotary_dim:].float() - q.float()[..., rotary_dim:]).abs().max().item()
    print(f"  [{tag}] head={head_size} rotary={rotary_dim} base={base}: "
          f"max|dq|={dq:.2e} max|dk|={dk:.2e} tail|d|={tail:.2e}")
    assert dq < 1e-4 and dk < 1e-4, (dq, dk)
    assert tail == 0.0, tail


def main() -> None:
    RotaryEmbedding = _load_rotary()
    print("partial rotary (Qwen3.5: rotary_dim=64 of head_dim=256):")
    _case(RotaryEmbedding, head_size=256, rotary_dim=64, base=10000000, tag="partial")
    print("full rotary regression (rotary_dim == head_dim):")
    _case(RotaryEmbedding, head_size=128, rotary_dim=128, base=1000000, tag="full-128")
    _case(RotaryEmbedding, head_size=64, rotary_dim=64, base=10000, tag="full-64")
    print("[OK] RotaryEmbedding partial + full rotary match the HF reference; tail untouched.")


if __name__ == "__main__":
    main()
