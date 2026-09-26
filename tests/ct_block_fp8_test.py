"""compressed-tensors BLOCKWISE fp8 + class-level targets — the Qwen3.8-Flash-Next-MXFP4-FP8 shape.

That checkpoint mixes three schemes in one file and broke four separate assumptions in the config
layer, every one of them SILENTLY (plausible, finite, wrong numbers rather than a load error):

 1. `strategy: "block"` / `block_structure: [128,128]` were not parsed at all, so a group whose
    `group_size` is null fell through to the `else 32` default and was described as group-32 — a
    scheme the file does not contain.
 2. A group with no `format` of its own INHERITED the top-level one. The file says
    `format: mxfp4-pack-quantized` at the top while group_1 is 8-bit float, so its fp8 attention and
    GDN projections were tagged MXFP4 and routed to the e2m1 nibble decode.
 3. The scale is named `weight_scale_inv` (the DeepSeek blockwise name), which is NOT a suffix of
    `.weight_scale`, so `_QSUFFIX` missed it and those modules were classified UNQUANTIZED — the
    loader then asks for a bf16 `.weight` the file ships as F8_E4M3.
 4. `targets: ["Linear"]` is a torch CLASS name meaning "every nn.Linear", not a module-name
    pattern. Matched as a name it selects nothing, so every routed expert resolved to UNQUANTIZED
    while the file ships `weight_packed` for them.

And fixing (4) exposed a fifth: this file ships GENERAL-BEFORE-SPECIFIC (catch-all group_0, then the
fp8 regexes), the opposite of Qwen3.8-27B-NVFP4, so first-match-wins on declaration order handed
attention and GDN back to MXFP4. Both layouts have to work, which is why a class-level target is
ordered LAST rather than the file being expected to declare an order it does not control.

Run (in the serve image, CPU only):
    PYTHONPATH=/opt/kernels:/engine/python:/engine python3 /engine/tests/ct_block_fp8_test.py
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from minisgl.quant.config import QuantConfig
from minisgl.quant.method import Fp8BlockLinearMethod, create_linear_method

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK   ' if cond else 'FAIL '} {name}" + (f": {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# The real file's quantization_config, trimmed to what the config layer reads.
FLASHNEXT_MXFP4 = {
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "ignore": [
        r"re:.*\.ple\..*", r"re:.*hyper_connection.*", r"re:.*\.mlp\.gate$",
        r"re:.*\.linear_attn\.in_proj_a.*", "lm_head", r"re:.*norm.*",
    ],
    "config_groups": {
        # GENERAL first — the layout that broke declaration-order resolution.
        "group_0": {"targets": ["Linear"],
                    "weights": {"num_bits": 4, "type": "float", "symmetric": True,
                                "strategy": "group", "group_size": 32},
                    "input_activations": None},
        "group_1": {"targets": [r"re:.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
                                r"re:.*\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$"],
                    "weights": {"num_bits": 8, "type": "float", "symmetric": True,
                                "strategy": "block", "block_structure": [128, 128]},
                    "input_activations": {"num_bits": 8, "type": "float", "strategy": "group",
                                          "group_size": 128, "dynamic": True}},
    },
}

# A SPECIFIC-BEFORE-GENERAL layout with both groups explicit (the Qwen3.8-27B-NVFP4 shape). Its
# declaration order is already correct and must survive untouched.
TWENTYSEVEN_B = {
    "quant_method": "compressed-tensors",
    "format": "mixed-precision",
    "ignore": [],
    "config_groups": {
        "group_0": {"targets": [r"re:.*layers\.(56|57)\.mlp\.(gate|up|down)_proj$"],
                    "weights": {"num_bits": 8, "type": "float", "symmetric": True,
                                "strategy": "channel"},
                    "input_activations": {"num_bits": 8, "type": "float", "dynamic": True}},
        "group_1": {"targets": [r"re:.*mlp\.(gate|up|down)_proj$"],
                    "weights": {"num_bits": 4, "type": "float", "symmetric": True,
                                "strategy": "group", "group_size": 16},
                    "input_activations": None},
    },
}


def parse(d):
    return QuantConfig.from_hf(SimpleNamespace(quantization_config=d))


def method_of(q, name: str) -> str:
    sc = q.for_module(name)
    return "UNQUANTIZED" if sc is None else create_linear_method(sc).__class__.__name__


def groups() -> None:
    q = parse(FLASHNEXT_MXFP4)
    check("three-scheme file parses", q is not None)
    check("ct_groups populated without format:mixed-precision", len(q.ct_groups) == 2,
          f"got {len(q.ct_groups)}")
    by_fmt = {sc.ct_format for _t, sc in q.ct_groups}
    check("the 8-bit group is float-quantized, NOT the inherited mxfp4",
          "float-quantized" in by_fmt, str(by_fmt))
    blk = [sc for _t, sc in q.ct_groups if sc.weight_strategy == "block"]
    check("strategy parsed", len(blk) == 1)
    check("block_structure parsed", bool(blk) and blk[0].block_structure == (128, 128),
          str(blk[0].block_structure) if blk else "none")
    check("is_fp8_block true for it", bool(blk) and blk[0].is_fp8_block)
    check("is_fp8_block false for the 4-bit group",
          not any(sc.is_fp8_block for _t, sc in q.ct_groups if sc.bits == 4))


def routing() -> None:
    q = parse(FLASHNEXT_MXFP4)
    want = {
        # the MXFP4 bulk — reached only because ["Linear"] becomes a catch-all
        "model.layers.5.mlp.experts.7.gate_proj": "MxFp4LinearMethod",
        "model.layers.5.mlp.shared_expert.down_proj": "MxFp4LinearMethod",
        # the blockwise-fp8 group — reached only because the catch-all is ordered LAST
        "model.layers.3.self_attn.q_proj": "Fp8BlockLinearMethod",
        "model.layers.3.self_attn.o_proj": "Fp8BlockLinearMethod",
        "model.layers.0.linear_attn.in_proj_qkv": "Fp8BlockLinearMethod",
        "model.layers.0.linear_attn.out_proj": "Fp8BlockLinearMethod",
        # the ignore list still wins over both
        "model.layers.0.linear_attn.in_proj_a": "UNQUANTIZED",
        "model.layers.5.mlp.gate": "UNQUANTIZED",
        "model.layers.1.ple.key_proj": "UNQUANTIZED",
        "lm_head": "UNQUANTIZED",
    }
    for n, exp in want.items():
        got = method_of(q, n)
        check(f"{n} -> {exp}", got == exp, f"got {got}")


def declaration_order_preserved() -> None:
    # Both groups explicit: the NARROW rule is declared first and must stay first, or the 8 fp8 MLP
    # layers get served as NVFP4. This is the regression the catch-all reordering must not cause.
    q = parse(TWENTYSEVEN_B)
    narrow = method_of(q, "model.layers.56.mlp.gate_proj")
    broad = method_of(q, "model.layers.10.mlp.gate_proj")
    check("narrow explicit group still wins", narrow == "Fp8W8A8LinearMethod", f"got {narrow}")
    check("general explicit group still applies elsewhere",
          broad in ("NvFp4LinearMethod", "MxFp4LinearMethod"), f"got {broad}")


def _tile_ref(w8, sc, bn, bk):
    """Independent reference: per-element tile lookup. Deliberately not the implementation's
    expression — a shared `repeat_interleave` spelling would agree with itself even transposed."""
    N, K = w8.shape
    ref = torch.empty(N, K, dtype=torch.float32)
    for i in range(N):
        for j in range(K):
            ref[i, j] = w8[i, j].float() * sc[i // bn, j // bk].float()
    return ref


def native() -> None:
    """128-block: served NATIVE — the fp8 bytes stay, the tile scale becomes a (K/128, N) plane."""
    m = Fp8BlockLinearMethod(SimpleNamespace(block_structure=(128, 128)))
    layer = SimpleNamespace()
    N, K = 256, 384
    m.create_weights(layer, N, K)
    check("declares fp8 weight", layer.weight.dtype == torch.float8_e4m3fn)
    check("declares the DeepSeek scale name at tile shape",
          tuple(layer.weight_scale_inv.shape) == (N // 128, K // 128),
          str(tuple(layer.weight_scale_inv.shape)))
    torch.manual_seed(0)
    layer.weight = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn)
    layer.weight_scale_inv = (torch.rand(N // 128, K // 128) + 0.5).to(torch.bfloat16)
    w8, sc = layer.weight.clone(), layer.weight_scale_inv.clone()
    m.process_weights_after_load(layer)
    check("weight kept as 1-byte e4m3 (not widened)",
          layer._w_op.dtype == torch.uint8 and torch.equal(layer._w_op, w8.view(torch.uint8)))
    check("scale plane is group-major (K/128, N) f32",
          layer._scales_op.dtype == torch.float32 and tuple(layer._scales_op.shape) == (K // 128, N),
          str(tuple(layer._scales_op.shape)))
    # What the cores compute per element: byte * plane[k // 128, n]. It must equal the tile lookup.
    plane_full = layer._scales_op.t().repeat_interleave(128, dim=1)[:, :K]      # (N, K)
    served = layer._w_op.view(torch.float8_e4m3fn).float() * plane_full
    check("plane reproduces the per-element tile scale exactly",
          torch.equal(served, _tile_ref(w8, sc, 128, 128)))
    check("checkpoint tensors released after load",
          not hasattr(layer, "weight") and not hasattr(layer, "weight_scale_inv"))


def dequant_fallback() -> None:
    """A K-block other than 128 keeps the exact load-time dequant (the tiled core's group is 128)."""
    m = Fp8BlockLinearMethod(SimpleNamespace(block_structure=(128, 64)))
    layer = SimpleNamespace()
    N, K = 256, 384
    m.create_weights(layer, N, K)
    torch.manual_seed(1)
    layer.weight = (torch.randn(N, K) * 0.3).to(torch.float8_e4m3fn)
    layer.weight_scale_inv = (torch.rand(N // 128, K // 64) + 0.5).to(torch.bfloat16)
    w8, sc = layer.weight.clone(), layer.weight_scale_inv.clone()
    m.process_weights_after_load(layer)
    check("non-128 K-block dequantized to bf16", layer.weight.dtype == torch.bfloat16)
    check("bit-exact vs per-element reference",
          torch.equal(layer.weight.float(), _tile_ref(w8, sc, 128, 64).to(torch.bfloat16).float()))
    check("scale released after load", not hasattr(layer, "weight_scale_inv"))

    # A TP split that cuts a block must be refused, not silently misaligned.
    try:
        m.create_weights(SimpleNamespace(), 200, 384)
        check("non-divisible shape refused", False, "accepted")
    except ValueError:
        check("non-divisible shape refused", True)


def main() -> int:
    print("GROUP PARSING")
    groups()
    print("\nPER-MODULE ROUTING (general-before-specific + class-level catch-all)")
    routing()
    print("\nNO REGRESSION for the specific-before-general layout")
    declaration_order_preserved()
    print("\nBLOCK DEQUANTIZATION")
    native()
    dequant_fallback()
    print("\n" + ("all passed" if not FAILED else f"FAILED: {FAILED}"))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
