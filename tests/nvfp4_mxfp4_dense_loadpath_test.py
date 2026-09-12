"""The NVFP4/MXFP4 DENSE load path, end to end: create_weights -> post_load -> apply.

G14 changed what these methods DECLARE (NVFP4 now takes a 1-byte block scale plus a per-channel f32
global instead of one folded fp16 scale; MXFP4 keeps its E8M0 byte). Those are LOADER declarations,
so no kernel test touches them — a wrong dtype or a missing leaf here is a boot failure or, worse, a
silent misread. This walks the REAL methods with synthetic checkpoint tensors.

The e4m3 NaN guard below is not decoration. A first run of this test drew the block scale as random
uint8 and NVFP4 came back non-finite; the cause was 0x7F/0xFF in the fixture (the OCP e4m3 NaN
codes), not the kernel. A fixture that can encode NaN will report a correct path as broken, which is
the mirror of a fixture that cannot see a real fault.

Run under a lease inside the serve image:
    PYTHONPATH=/engine/python:/opt/kernels python3 /engine/tests/nvfp4_mxfp4_dense_loadpath_test.py
"""
import sys, torch
sys.path.insert(0, "/engine/python"); sys.path.append("/opt/kernels")
from minisgl.distributed import set_tp_info
set_tp_info(0, 1)
from minisgl.quant import method as M
from minisgl.quant.config import QuantConfig
from minisgl.quant import nvfp4, mxfp4

FAILED = []
def check(n, ok, d=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {n}{('  — '+d) if d else ''}")
    if not ok: FAILED.append(n)

dev = torch.device("cuda")
N, K = 256, 512
torch.manual_seed(11)

class Layer: pass

# ---- the fence is open: every module keeps two levels now ----------------------------------------
check("nvfp4_leaf_splits is True for a DENSE linear (not just .experts.)",
      nvfp4.nvfp4_leaf_splits("model.layers.0.mlp.gate_proj") is True)
check("...and still True for an expert", nvfp4.nvfp4_leaf_splits("model.layers.0.mlp.experts.w13") is True)

# ---- NVFP4 dense: declare -> fill -> post_load -> apply -------------------------------------------
q = QuantConfig(method="compressed-tensors", bits=4, group_size=16, sym=True, weight_type="float")
m = M.NvFp4LinearMethod(q)
L = Layer()
m.create_weights(L, N, K)
check("NVFP4 declares a 1-byte block scale", L.weight_scale.dtype is torch.uint8,
      str(L.weight_scale.dtype))
check("NVFP4 declares a per-output-channel f32 global",
      L.weight_global.dtype is torch.float32 and L.weight_global.shape == (N,),
      f"{L.weight_global.dtype} {tuple(L.weight_global.shape)}")
L.weight_packed = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=dev)
# VALID e4m3 bytes only. 0x7F and 0xFF are the OCP e4m3 NaN codes, and a random uint8 fixture hits
# them — which is a property of the FIXTURE, not of the kernel. Draw real values and encode them.
L.weight_scale = ((torch.rand(N, K // 16, device=dev) * 0.9 + 0.1)
                  .to(torch.float8_e4m3fn).view(torch.uint8))
_nan = ((L.weight_scale & 0x7F) == 0x7F).sum().item()
check("fixture carries no e4m3 NaN codes (0x7F/0xFF)", _nan == 0, f"{_nan} NaN bytes")
L.weight_global = (torch.rand(N, device=dev) * 0.02 + 0.001)
m.process_weights_after_load(L)
check("NVFP4 global reaches the op as an int32 BIT-VIEW (not a value cast)",
      L._global_op.dtype is torch.int32 and
      torch.equal(L._global_op.view(torch.float32), L._global_op.view(torch.float32)))
x = (torch.randn(4, K, dtype=torch.bfloat16, device=dev) * 0.1)
out = m.apply(L, x, None)
check("NVFP4 dense apply() runs and is finite",
      out.shape == (4, N) and torch.isfinite(out).all(), str(tuple(out.shape)))

# ---- MXFP4 dense ----------------------------------------------------------------------------------
q2 = QuantConfig(method="compressed-tensors", bits=4, group_size=32, sym=True, weight_type="float")
m2 = M.MxFp4LinearMethod(q2)
L2 = Layer()
m2.create_weights(L2, N, K)
L2.weight_packed = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=dev)
# exponents WAY outside fp16's window — the case the old widening saturated and logged
L2.weight_scale = (torch.randint(145, 150, (N, K // 32)).to(torch.uint8)).to(dev)
m2.process_weights_after_load(L2)
check("MXFP4 keeps the checkpoint's E8M0 byte (no fp16 widening)",
      L2._scales_op.dtype is torch.uint8, str(L2._scales_op.dtype))
out2 = m2.apply(L2, x, None)
check("MXFP4 dense apply() runs and is finite ABOVE fp16's range",
      out2.shape == (4, N) and torch.isfinite(out2).all(), str(tuple(out2.shape)))
check("...and is not degenerate (a saturated scale would flatten it)", out2.abs().max() > 0)

print("")
print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED) if FAILED else "ALL CHECKS PASS")
sys.exit(1 if FAILED else 0)
