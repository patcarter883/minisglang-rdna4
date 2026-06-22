"""Phase 3M-2 — validate awq_to_op_layout on REAL Qwen3.6-35B-A3B (AWQ-gemm) expert tensors.

The qwen3_5_moe routed experts are AWQ-gemm int4 (g32, asymmetric); `_GroupedAWQExperts.post_load`
runs `awq_to_op_layout` per expert. This test checks that conversion on a few real expert matrices
(gate/up/down across different experts):
  * `ref_dequant`  — dequant straight from the AWQ checkpoint tensors: undo the gemm interleave
    ([0,2,4,6,1,3,5,7]) to natural N order, then w = scale * (q - z) grouped by arange(K)//g;
  * `op_dequant`   — dequant from the op-native buffers awq_to_op_layout emits (natural packing,
    transposed to (N,K)); an INDEPENDENT unpacker.
max|Δ| == 0 proves the layout transform (deinterleave, transpose, repack, asymmetric zeros) is a
faithful re-encoding on the real 35B expert shapes. (This is the same awq_to_op_layout proven
end-to-end on the dense 7B-AWQ MVP; here it is exercised on the grouped MoE expert geometry.)

CPU-only; run in the combined image (needs torch; no GPU lease):
    PYTHONPATH=/engine/python python /engine/tools/qwen3_5_moe_awq_convert_test.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import torch
from safetensors import safe_open

from minisgl.quant.kernels import _REVERSE_AWQ_PACK_ORDER, awq_to_op_layout

REPO = Path.home() / ".cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit"
BITS, GROUP = 4, 32
PREFIX = "model.language_model.layers.0.mlp.experts."


def _open_canonical():
    snap = (REPO / "refs/main").read_text().strip()  # the AWQ snapshot
    files = glob.glob(str(REPO / "snapshots" / snap / "*.safetensors"))
    handles = [safe_open(f, framework="pt") for f in files]
    index = {k: h for h in handles for k in h.keys()}
    return lambda k: index[k].get_tensor(k)


def ref_dequant(qweight, scales, qzeros) -> torch.Tensor:
    """AWQ-gemm asymmetric dequant from checkpoint tensors -> (K, N) float32."""
    pf, mask = 32 // BITS, (1 << BITS) - 1
    K, Np = qweight.shape
    N = Np * pf
    rev = torch.tensor(_REVERSE_AWQ_PACK_ORDER[:pf], dtype=torch.long)
    shifts = torch.arange(0, 32, BITS, dtype=torch.int32)
    q = (qweight.unsqueeze(-1) >> shifts) & mask           # (K, Np, pf)
    q = q[:, :, rev].reshape(K, N).to(torch.float32)       # natural N order
    z = (qzeros.unsqueeze(-1) >> shifts) & mask            # (G, Np, pf)
    z = z[:, :, rev].reshape(qzeros.shape[0], N).to(torch.float32)
    gidx = torch.arange(K) // GROUP
    s = scales.to(torch.float32)[gidx]                     # (K, N)
    zf = z[gidx]                                           # (K, N)
    return s * (q - zf)


def op_dequant(w_packed, scales_op, zeros_op, K) -> torch.Tensor:
    """Dequant from op-native buffers -> (N, K) float32 (independent, natural packing)."""
    pf, mask = 32 // BITS, (1 << BITS) - 1
    N = w_packed.shape[0]
    shifts = torch.arange(0, 32, BITS, dtype=torch.int32)
    q = ((w_packed.unsqueeze(-1) >> shifts) & mask).reshape(N, K).to(torch.float32)  # (N, K)
    z = (zeros_op.unsqueeze(-1) >> shifts) & mask          # (N//pf, G, pf)
    G = zeros_op.shape[1]
    z = z.permute(0, 2, 1).reshape(N, G).to(torch.float32)  # (N, G)
    gidx = torch.arange(K) // GROUP
    s = scales_op.to(torch.float32)[:, gidx]               # (N, K)
    zf = z[:, gidx]                                        # (N, K)
    return s * (q - zf)


def check(get, name) -> None:
    qw, sc, qz = get(name + ".qweight"), get(name + ".scales"), get(name + ".qzeros")
    K, Np = qw.shape
    N = Np * (32 // BITS)
    w_ref = ref_dequant(qw, sc, qz)                        # (K, N)
    w_packed, scales_op, zeros_op = awq_to_op_layout(qw, sc, qz, bits=BITS)
    assert tuple(w_packed.shape) == (N, K // 8)
    assert tuple(scales_op.shape) == (N, K // GROUP)
    assert tuple(zeros_op.shape) == (N // 8, K // GROUP)
    w_op = op_dequant(w_packed, scales_op, zeros_op, K)    # (N, K)
    d = (w_op.t() - w_ref).abs().max().item()
    print(f"  {name.split('experts.')[1]}: K={K} N={N} |dequant|~{w_ref.abs().mean():.3e}  "
          f"max|Δ(op,ref)|={d:.3e}")
    assert d == 0.0, f"{name}: op-layout dequant != AWQ reference (max|Δ|={d})"


def main() -> None:
    get = _open_canonical()
    print("=== awq_to_op_layout faithfulness on real 35B experts (op-layout vs AWQ dequant) ===")
    for name in (
        PREFIX + "0.gate_proj",
        PREFIX + "5.up_proj",
        PREFIX + "200.down_proj",
    ):
        check(get, name)
    print("[OK] all expert matrices: op-layout dequant bit-identical to the AWQ reference.")


if __name__ == "__main__":
    main()
