"""Phase 2M-2 — validate gptq_to_op_layout on REAL Qwen1.5-MoE GPTQ-Int4 tensors.

For a few real checkpoint matrices (dense q_proj, an expert gate_proj, the shared-expert down_proj):
  * `ref_dequant` — dequant straight from the GPTQ checkpoint tensors via the standard formula
    w = scale * (q - (unpacked_qzeros + 1)), grouped by g_idx = arange(K)//group_size;
  * `op_dequant` — dequant from the op-native buffers that gptq_to_op_layout produces;
the two are INDEPENDENT unpackers, so max|Δ| == 0 proves the layout transform (input-packing,
transpose, +1 zero fold, repack) is a faithful re-encoding. Also asserts the symmetric constant
(zero point 8). The GPTQ FORMULA itself is the end-to-end oracle in 2M-4 (token parity vs vLLM).

CPU-only; run in the combined image (needs torch; no GPU lease):
    PYTHONPATH=/engine/python python /engine/tools/qwen2_moe_gptq_convert_test.py
"""
from __future__ import annotations

import glob
import torch
from safetensors import safe_open

from minisgl.quant.kernels import gptq_to_op_layout

MODEL_GLOB = (
    "/root/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4/"
    "snapshots/*/*.safetensors"
)
BITS, GROUP = 4, 128


def _open_all():
    handles = [safe_open(f, framework="pt") for f in glob.glob(MODEL_GLOB)]
    index = {k: h for h in handles for k in h.keys()}
    return lambda k: index[k].get_tensor(k)


def ref_dequant(qweight, scales, qzeros) -> torch.Tensor:
    """Standard GPTQ dequant from checkpoint tensors -> (K, N) float32."""
    pf, mask = 32 // BITS, (1 << BITS) - 1
    Kp, N = qweight.shape
    K = Kp * pf
    shifts = torch.arange(0, 32, BITS, dtype=torch.int32)
    q = (qweight.unsqueeze(-1) >> shifts) & mask          # (Kp, N, pf)
    q = q.permute(0, 2, 1).reshape(K, N).to(torch.float32)
    z = ((qzeros.unsqueeze(-1) >> shifts) & mask).reshape(qzeros.shape[0], N) + 1  # (G, N)
    gidx = torch.arange(K) // GROUP
    s = scales.to(torch.float32)[gidx]                    # (K, N)
    zf = z.to(torch.float32)[gidx]                        # (K, N)
    return s * (q - zf)


def op_dequant(w_packed, scales_op, zeros_op, K) -> torch.Tensor:
    """Dequant from op-native buffers -> (N, K) float32 (independent unpacker)."""
    pf, mask = 32 // BITS, (1 << BITS) - 1
    N = w_packed.shape[0]
    shifts = torch.arange(0, 32, BITS, dtype=torch.int32)
    q = ((w_packed.unsqueeze(-1) >> shifts) & mask).reshape(N, K).to(torch.float32)  # (N, K)
    z = (zeros_op.unsqueeze(-1) >> shifts) & mask          # (N//pf, G, pf)
    G = zeros_op.shape[1]
    z = z.permute(0, 2, 1).reshape(N, G).to(torch.float32)  # (N, G)
    gidx = torch.arange(K) // GROUP
    s = scales_op.to(torch.float32)[:, gidx]              # (N, K)
    zf = z[:, gidx]                                       # (N, K)
    return s * (q - zf)


def check(get, name) -> None:
    qw, sc, qz = get(name + ".qweight"), get(name + ".scales"), get(name + ".qzeros")
    Kp, N = qw.shape
    K = Kp * (32 // BITS)
    # symmetric constant: every unpacked qzero must be 7 (-> zero point 8)
    uz = ((qz.unsqueeze(-1) >> torch.arange(0, 32, BITS, dtype=torch.int32)) & 0xF)
    assert int(uz.min()) == int(uz.max()) == 7, f"{name}: qzeros not constant 7"

    w_ref = ref_dequant(qw, sc, qz)                       # (K, N)
    w_packed, scales_op, zeros_op = gptq_to_op_layout(qw, sc, qz, bits=BITS)
    assert tuple(w_packed.shape) == (N, K // 8)
    assert tuple(scales_op.shape) == (N, K // GROUP)
    assert tuple(zeros_op.shape) == (N // 8, K // GROUP)
    w_op = op_dequant(w_packed, scales_op, zeros_op, K)   # (N, K)

    d = (w_op.t() - w_ref).abs().max().item()
    print(f"  {name}: K={K} N={N} |dequant|~{w_ref.abs().mean():.3e}  max|Δ(op,ref)|={d:.3e}")
    assert d == 0.0, f"{name}: op-layout dequant != reference (max|Δ|={d})"


def main() -> None:
    get = _open_all()
    print("=== gptq_to_op_layout faithfulness (op-layout dequant vs GPTQ-checkpoint dequant) ===")
    for name in (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.5.down_proj",
        "model.layers.0.mlp.shared_expert.down_proj",
    ):
        check(get, name)
    print("[OK] all matrices: op-layout dequant bit-identical to the GPTQ reference; zero point 8.")


if __name__ == "__main__":
    main()
