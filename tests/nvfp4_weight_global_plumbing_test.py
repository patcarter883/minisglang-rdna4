"""Every place the NVFP4 `weight_global` leaf must be known about, asserted without a GPU.

WHY THIS EXISTS. `nvfp4_leaf_splits` was widened to `return True` (84b1ede, FORMAT_MATRIX.md G14),
which made the two-level NVFP4 scale universal and so introduced a NEW TENSOR KIND — `.weight_global`
— into every module of every NVFP4 checkpoint. Four separate places enumerate tensor kinds by name,
none of them knew about it, and all four shipped:

  1. `_QWEN35_CONCAT`            the fused-GDN field list -> KeyError 'in_proj_qkvz.weight_global'
  2. `_shard_qwen3_5`            no TP rule -> an unsharded (N,) global against sharded (N/2, K)
  3. `NvFp4LinearMethod`         block scale declared uint8 while the leaf producer emits e4m3
  4. `cast_checkpoint_tensor`    fp32 global cast to the model dtype -> the int32 bitcast HALVES it

NONE of these are visible to a type checker, and none were caught by the dense load-path test that
shipped in the same commit, because that test walked the DENSE path and every one of these defects
needed either the FUSED GDN projection or TP>1 to appear. Defect 4 in particular was invisible on
dense linears: a `BaseOP` parameter is declared fp32 and `layers/base.py::_coerce_dtype` casts the
incoming tensor back, so `gate_up_proj.weight_global` survived the identical bad cast unharmed --
`GDNLinearAttn` loads with `assign=True`, which takes the incoming dtype and has nothing to restore
it. The same wrong cast was therefore harmless in one module type and fatal in another.

So this file asserts the PLUMBING, not the arithmetic: it walks a synthetic NVFP4 GDN projection
through the real loader functions at TP=2 and checks the global's LENGTH and DTYPE at each hop. It
needs no GPU and no checkpoint, so it runs anywhere and in a fraction of a second.

Run:  PYTHONPATH=python python3 tests/nvfp4_weight_global_plumbing_test.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from minisgl.models.weight import (  # noqa: E402
    _QWEN35_CONCAT,
    _shard_qwen3_5,
    cast_checkpoint_tensor,
)
from minisgl.quant import nvfp4  # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


# Qwen3.8-27B-MTP-NVFP4 geometry — the checkpoint that caught all four.
KEY_HEAD, N_KEY = 128, 16      # key_dim   = 2048
VAL_HEAD, N_VAL = 128, 48      # value_dim = 6144
KEY_DIM, VALUE_DIM = KEY_HEAD * N_KEY, VAL_HEAD * N_VAL
QKV_N = KEY_DIM * 2 + VALUE_DIM      # 10240, matches in_proj_qkv weight_packed [10240, 2560]
Z_N = VALUE_DIM                      # 6144,  matches in_proj_z   weight_packed [6144, 2560]
FUSED_N = QKV_N + Z_N                # 16384 -> 8192 per rank at TP=2
GROUP = 16
TP = 2

CFG = SimpleNamespace(
    linear_key_head_dim=KEY_HEAD, linear_num_key_heads=N_KEY,
    linear_value_head_dim=VAL_HEAD, linear_num_value_heads=N_VAL,
    quant=SimpleNamespace(bits=4, group_size=GROUP),
)

print("DEFECT 4 — cast_checkpoint_tensor must NOT follow the model dtype for `.weight_global`")
for model_dtype in (torch.bfloat16, torch.float16):
    g = torch.rand(FUSED_N, dtype=torch.float32)
    out = cast_checkpoint_tensor("model.layers.0.linear_attn.in_proj_qkvz.weight_global", g, model_dtype)
    check(f"stays fp32 under model dtype {model_dtype}", out.dtype == torch.float32,
          f"got {out.dtype}")
    # The failure was not "wrong precision", it was a BITCAST length change. Assert the consequence
    # directly, because that is what the kernel actually rejects.
    check(f"int32 bitcast keeps N under {model_dtype}",
          out.contiguous().view(torch.int32).numel() == FUSED_N,
          f"got {out.contiguous().view(torch.int32).numel()}, want {FUSED_N}")
# and the rule must not have caught something it shouldn't
w = torch.rand(8, 8, dtype=torch.float32)
check("an ordinary .weight still follows the model dtype",
      cast_checkpoint_tensor("model.layers.0.mlp.gate_proj.weight", w, torch.bfloat16).dtype
      == torch.bfloat16)

print()
print("DEFECT 1 — the fused-GDN concat map must carry the global leaf")
for part in ("in_proj_qkv", "in_proj_z"):
    key = f".linear_attn.{part}.weight_global"
    hit = _QWEN35_CONCAT.get(key)
    check(f"{part}.weight_global is in _QWEN35_CONCAT", hit is not None)
    if hit:
        check(f"{part}.weight_global merges into in_proj_qkvz on dim 0",
              hit[0] == ".linear_attn.in_proj_qkvz.weight_global" and hit[2] == 0, f"{hit}")
for part in ("in_proj_b", "in_proj_a"):
    check(f"{part}.weight_global is in _QWEN35_CONCAT",
          f".linear_attn.{part}.weight_global" in _QWEN35_CONCAT)

print()
print("DEFECT 2 — TP shard rules, per module, on the (N,) global")
cases = [
    ("in_proj_qkv", QKV_N, QKV_N // TP, "col-parallel, head-block split"),
    ("in_proj_z", Z_N, Z_N // TP, "col-parallel"),
    ("out_proj", FUSED_N, FUSED_N, "ROW-parallel -> replicates (output N is full width)"),
]
for part, n_in, n_want, why in cases:
    g = torch.rand(n_in, dtype=torch.float32)
    got = _shard_qwen3_5(f"model.layers.0.linear_attn.{part}.weight_global", g, 0, TP, CFG)
    check(f"{part:<12} {n_in} -> {got.numel()} ({why})", got.numel() == n_want,
          f"want {n_want}")
    check(f"{part:<12} shard preserves fp32", got.dtype == torch.float32, f"got {got.dtype}")

print()
print("END TO END — the fused projection, the shape that actually broke")
# split_nvfp4_scale turns each part's per-TENSOR scalar into its own per-OUTPUT-CHANNEL vector,
# which is the reason a plain dim-0 concat of the two parts is correct at all.
qkv_blk = torch.randint(0, 255, (QKV_N, 2560 // GROUP), dtype=torch.uint8)
z_blk = torch.randint(0, 255, (Z_N, 2560 // GROUP), dtype=torch.uint8)
qkv_leaves = nvfp4.nvfp4_leaf_scales(
    "model.layers.0.linear_attn.in_proj_qkv", qkv_blk,
    torch.tensor([3.0], dtype=torch.float32), global_field="weight_global_scale")
z_leaves = nvfp4.nvfp4_leaf_scales(
    "model.layers.0.linear_attn.in_proj_z", z_blk,
    torch.tensor([7.0], dtype=torch.float32), global_field="weight_global_scale")
leaf = {n: t for n, t in qkv_leaves + z_leaves}

check("the checkpoint's per-TENSOR scalar became a per-CHANNEL vector (qkv)",
      leaf["model.layers.0.linear_attn.in_proj_qkv.weight_global"].numel() == QKV_N,
      "a raw [1] scalar here is what would make the concat yield [2]")
check("...and for z, with its own DIFFERENT scalar",
      leaf["model.layers.0.linear_attn.in_proj_z.weight_global"].numel() == Z_N)
check("the two parts carry different values (not one global reused)",
      not torch.allclose(leaf["model.layers.0.linear_attn.in_proj_qkv.weight_global"][:1],
                         leaf["model.layers.0.linear_attn.in_proj_z.weight_global"][:1]))

# shard each part as the loader does (at READ, before the concat), then fuse.
sq = _shard_qwen3_5("model.layers.0.linear_attn.in_proj_qkv.weight_global",
                    leaf["model.layers.0.linear_attn.in_proj_qkv.weight_global"], 0, TP, CFG)
sz = _shard_qwen3_5("model.layers.0.linear_attn.in_proj_z.weight_global",
                    leaf["model.layers.0.linear_attn.in_proj_z.weight_global"], 0, TP, CFG)
fused = torch.cat([sq, sz], dim=0)
check(f"fused in_proj_qkvz global = {fused.numel()} (want {FUSED_N // TP} per rank)",
      fused.numel() == FUSED_N // TP)
# The whole failure in one line: this is the number the kernel checks.
fused_cast = cast_checkpoint_tensor(
    "model.layers.0.linear_attn.in_proj_qkvz.weight_global", fused, torch.bfloat16)
check("after the model-dtype cast, the int32 bitcast STILL has one entry per output channel",
      fused_cast.contiguous().view(torch.int32).numel() == FUSED_N // TP,
      f"got {fused_cast.contiguous().view(torch.int32).numel()}; "
      f"this is the exact kernel error 'N=8192; got 4096'")

print()
print("DEFECT 3 — the block scale's declared dtype must match what the leaf producer emits")
blk = leaf["model.layers.0.linear_attn.in_proj_qkv.weight_scale"]
check("split arm emits the block scale as float8_e4m3fn",
      blk.dtype == torch.float8_e4m3fn, f"got {blk.dtype}")
try:
    from minisgl.quant.method import NvFp4LinearMethod

    holder = SimpleNamespace()
    NvFp4LinearMethod(SimpleNamespace(group_size=GROUP)).create_weights(holder, 512, 2560)
    check("NvFp4LinearMethod declares the SAME dtype it will be handed",
          holder.weight_scale.dtype == blk.dtype,
          f"declared {holder.weight_scale.dtype} vs leaf {blk.dtype} — "
          f"_coerce_dtype hard-fails a quantized mismatch rather than casting")
    check("NvFp4LinearMethod declares weight_global fp32",
          holder.weight_global.dtype == torch.float32, f"got {holder.weight_global.dtype}")
except Exception as e:  # noqa: BLE001
    check("NvFp4LinearMethod.create_weights is inspectable", False, f"{type(e).__name__}: {e}")

print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASS")
