"""Phase 3b-3 CPU pre-check: isolate split / reshape / gate-order bugs WITHOUT a GPU.

The real `QwenGatedDeltaNetAttention` cannot be stood up on CPU in this image
(its `__init__` routes through `is_cpu()` -> `register_cpu_gdn_attention_ops()`,
which may not import), so instead of comparing against a live real layer we
compare `minisgl.gdn.layer.QwenGatedDeltaNet`'s pure-torch plumbing methods
against INLINE replications of the reference's non-interleaved (Qwen3.5) logic,
copied verbatim from `qwen_gdn_linear_attn.py`:

  * `_split_qkvz_ba`            vs prepare_gdn_attention_core_inputs (lines 709-718)
  * `_rearrange_mixed_qkv`      vs rearrange_mixed_qkv               (lines 808-842)
  * `_output_projection` order  vs _output_projection                (lines 863-869)

These are tp_size==1, so the reference's `// self.tp_size` is a no-op. A mismatch
here is a split/reshape/gate-order bug that would otherwise masquerade as a
numeric divergence in the (expensive) GPU harness.

Runs CPU-only. In this repo torch only imports inside the combined image, so run
it there (no GPU lease required — it touches no card):
  docker run --rm -v "$PWD":/engine --entrypoint bash vllm22-w4a8:combined -lc \
    'source /app/.venv/bin/activate && PYTHONPATH=/engine/python \
       python /engine/tools/gdn_layer_parity_cpu.py'
"""

from __future__ import annotations

import torch

from minisgl.gdn.layer import QwenGatedDeltaNet

DT = torch.float32
results: list[tuple[str, bool, float]] = []


def _cmp(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    md = (a.float() - b.float()).abs().max().item()
    exact = torch.equal(a, b)
    results.append((name, exact, md))
    print(f"  [{'PASS' if exact else 'FAIL'}] {name:32s} max|Δ|={md:.3e} "
          f"exact={exact}", flush=True)


def main() -> None:
    HK, HV, DK, DV, KCONV, HIDDEN = 16, 32, 128, 128, 4, 2048
    key_dim, value_dim = DK * HK, DV * HV
    layer = QwenGatedDeltaNet(
        hidden_size=HIDDEN, num_k_heads=HK, num_v_heads=HV,
        head_k_dim=DK, head_v_dim=DV, conv_kernel_size=KCONV,
        dtype=DT, device="cpu",
    )
    n = 7
    torch.manual_seed(0)
    qkvz = torch.randn(n, key_dim * 2 + value_dim * 2, dtype=DT)
    ba = torch.randn(n, 2 * HV, dtype=DT)

    # ---- 1. qkvz/ba split (vs prepare_gdn_attention_core_inputs 709-718) ----
    m_qkv, m_z, m_b, m_a = layer._split_qkvz_ba(qkvz, ba, n)
    qkv_size = key_dim * 2 + value_dim
    z_size = value_dim
    r_qkv, r_zflat = qkvz.split([qkv_size, z_size], dim=-1)
    r_z = r_zflat.reshape(n, -1, DV)
    r_b, r_a = ba.chunk(2, dim=-1)
    _cmp("split.mixed_qkv", m_qkv, r_qkv)
    _cmp("split.z", m_z, r_z)
    _cmp("split.b", m_b, r_b)
    _cmp("split.a", m_a, r_a)

    # ---- 2. decode qkv rearrange (vs rearrange_mixed_qkv 808-842) ----
    mixed_qkv = torch.randn(n, key_dim * 2 + value_dim, dtype=DT)
    m_q, m_k, m_v = layer._rearrange_mixed_qkv(mixed_qkv, n)
    rq, rk, rv = mixed_qkv.split([key_dim, key_dim, value_dim], dim=-1)
    ref_q = rq.reshape(1, n, -1, DK)
    ref_k = rk.reshape(1, n, -1, DK)
    ref_v = rv.reshape(1, n, -1, DV)
    _cmp("rearrange.q", m_q, ref_q)
    _cmp("rearrange.k", m_k, ref_k)
    _cmp("rearrange.v", m_v, ref_v)

    # ---- 3. output-projection reshape/flatten ORDER (vs _output_projection 863-869) ----
    # Stub the triton norm with identity so this stays CPU-only; we are checking the
    # reshape/flatten/out_proj plumbing order, not the norm math (covered on GPU).
    core = torch.randn(n, HV, DV, dtype=DT)
    z = torch.randn(n, HV, DV, dtype=DT)

    class _IdNorm(torch.nn.Module):
        def forward(self, c, zz):  # identity (skips the triton norm math)
            return c

    orig_norm = layer.norm
    layer.norm = _IdNorm()
    try:
        m_out = layer._output_projection(core.clone(), z.clone(), n)
    finally:
        layer.norm = orig_norm
    # reference replication with the SAME identity norm + the SAME out_proj weights
    z_shape_og = z.shape
    rc = core.clone().reshape(-1, core.shape[-1])
    rc = rc  # identity norm
    rc = rc.reshape(z_shape_og).flatten(-2)
    r_out = layer.out_proj(rc)
    _cmp("output_proj.reshape_order", m_out, r_out)
    # shape sanity: flatten(-2) must give head-major value_dim
    ok_shape = tuple(m_out.shape) == (n, HIDDEN)
    results.append(("output_proj.shape", ok_shape, 0.0))
    print(f"  [{'PASS' if ok_shape else 'FAIL'}] output_proj.shape            "
          f"-> {tuple(m_out.shape)} (want {(n, HIDDEN)})", flush=True)

    print("\n=== CPU pre-check: split / reshape / gate-order ===")
    all_ok = all(ok for _, ok, _ in results)
    print("RESULT:", "PASS — plumbing matches reference" if all_ok
          else "FAIL — split/reshape divergence")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
