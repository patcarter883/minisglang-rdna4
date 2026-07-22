"""Validate the NVFP4 (group-16 e2m1) kernel path numerically against the golden dequant, using a
REAL poolside/Laguna-XS-2.1-NVFP4 expert weight. Answers: does the group-16 e2m1 W4A8 kernel compute
correctly (the NVFP4-pathway headline)? cos ~0.99+ = correct (fp8-activation error only); cos low =
the group-16 kernel mis-decodes."""
import glob, json
import torch
import safetensors
from minisgl.quant import nvfp4, kernels


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-20)).item()


def find_file(snap, key):
    wm = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
    return snap + wm[key]


def load(snap, base):
    # base e.g. model.layers.1.mlp.experts.0.down_proj
    out = {}
    for fld in ("weight_packed", "weight_scale", "weight_global_scale"):
        k = f"{base}.{fld}"
        f = find_file(snap, k)
        with safetensors.safe_open(f, framework="pt", device="cuda") as h:
            out[fld] = h.get_tensor(k)
    return out


def check(snap, base, dev):
    t = load(snap, base)
    wp, ws, wg = t["weight_packed"], t["weight_scale"], t["weight_global_scale"]
    N, Khalf = wp.shape
    K = Khalf * 2
    print(f"\n=== {base}  N={N} K={K}  group=16  (packed {tuple(wp.shape)}, scale {tuple(ws.shape)} {ws.dtype}, global {wg.item():.4g}) ===")
    # golden bf16 weight from the RAW checkpoint tensors
    golden = nvfp4.dequant_reference(wp, ws, wg).to(torch.float32)  # [N,K]
    # folded -> op layout (what the served kernel consumes)
    folded = nvfp4.fold_nvfp4_scale(ws, wg)  # [N,K//16] fp16
    conv = nvfp4.convert_nvfp4_weight(wp, folded)
    w_op, scales_op, g = conv["w_packed"], conv["scales"], conv["group_size"]
    print(f"  fold: group_size={g}  w_op {tuple(w_op.shape)} {w_op.dtype}  scales_op {tuple(scales_op.shape)} {scales_op.dtype}")
    for M in (1, 4, 64):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.1
        try:
            ko = kernels.w4a8_linear(x, w_op, scales_op, None, g, weight_is_e2m1=True).float()  # [M,N]
        except Exception as e:
            print(f"  M={M:3d}  KERNEL RAISED: {e}")
            continue
        # reference: activation quantized to fp8 e4m3 (W4A8) then @ golden.T — matches the kernel's act path
        ref = (x.float() @ golden.T)  # ideal (no act-quant); kernel has fp8-act error
        c = cos(ko, ref)
        rel = ((ko - ref).norm() / (ref.norm() + 1e-20)).item()
        print(f"  M={M:3d}  cos(kernel, ideal)={c:.5f}  rel_err={rel:.4f}  kernel[0,:4]={ko[0,:4].tolist()}  ref[0,:4]={ref[0,:4].tolist()}")


if __name__ == "__main__":
    dev = "cuda"
    snap = glob.glob("/root/.cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots/*/")[0]
    print("snapshot:", snap)
    # down_proj (row-parallel shape, K=inter=512) and gate_proj (K=hidden=2048) of a routed expert
    check(snap, "model.layers.1.mlp.experts.0.down_proj", dev)
    check(snap, "model.layers.1.mlp.experts.0.gate_proj", dev)
    check(snap, "model.layers.1.mlp.shared_expert.down_proj", dev)
    print("\nDONE_NVFP4_CHECK")
