"""GPU microbench: confirm the e2m1 W4A8 kernel is correct at group_size=16 (NVFP4's block size).

This is the ONE thing static analysis can't fully prove — that the generic runtime-group_size e2m1
instance (the same one MXFP4-at-32 uses) stages/aligns correctly with BK=16. Build a synthetic
NVFP4-shaped weight (E2M1 codes + fp16 per-16-group scale), run kernels.w4a8_linear(weight_is_e2m1=
True, group_size=16) and the grouped kernels.w4a8_moe, and compare against a high-precision reference.
The kernel fp8-quantizes activations internally, so expect ~fp8-activation error (cos-sim > 0.99), not
bit-exactness.

Run (needs a card):
    gpu-lease -n 1 -- docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE --ipc host \
      --shm-size 16gb -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
      -v "$PWD":/engine -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
      --entrypoint bash minisgl-rdna4:lean -lc \
      'source /opt/venv/bin/activate && PYTHONPATH=/opt/kernels:/engine/python python /engine/tools/nvfp4_group16_microbench.py'
"""
import torch
import torch.nn.functional as F

from minisgl.quant import kernels
from minisgl.quant.mxfp4 import FP4_E2M1_LUT, pack_codes_to_int32

DEV = "cuda"
LUT = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32, device=DEV)


def build_e2m1_weight(N, K, group_size, dtype=torch.float16):
    """Random E2M1 codes + a random positive fp16 per-group scale -> (w_packed int32 (N,K/8),
    scales (N,K/group) fp16, W_real (N,K) f32 reference)."""
    codes = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=DEV)
    w_packed = pack_codes_to_int32(codes)
    scales = (0.01 + 0.05 * torch.rand(N, K // group_size, device=DEV)).to(dtype)
    W_real = LUT[codes.to(torch.int64)] * scales.float().repeat_interleave(group_size, dim=-1)
    return w_packed, scales, W_real


def cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def bench_dense():
    N, K, M, g = 4096, 2048, 32, 16
    w_packed, scales, W_real = build_e2m1_weight(N, K, g)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
    y_ref = (x.float() @ W_real.t()).to(torch.bfloat16)
    y = kernels.w4a8_linear(x, w_packed, scales, None, g, weight_is_e2m1=True).to(torch.bfloat16)
    rel = (y.float() - y_ref.float()).norm() / y_ref.float().norm().clamp(min=1e-9)
    print(f"[dense g16] out {tuple(y.shape)}  cos={cos(y, y_ref):.5f}  rel-err={rel:.4f}  "
          f"(expect ~fp8-act level; cos>0.99 = PASS)")
    return cos(y, y_ref) > 0.99


def bench_moe():
    E, N, K, M, top_k, g = 8, 1024, 2048, 16, 2, 16
    w13p, w13s, W13 = build_e2m1_weight(E * N, K, g)  # reuse builder then reshape per-expert
    w13p = w13p.view(E, N, K // 8); w13s = w13s.view(E, N, K // g)
    w2p, w2s, W2 = build_e2m1_weight(E * (K // 2), N, g)
    w2p = w2p.view(E, K // 2, N // 8); w2s = w2s.view(E, K // 2, N // g)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
    router = torch.randn(M, E, device=DEV)
    try:
        y = kernels.w4a8_moe(x, w13p, w13s, None, w2p, w2s, None, router, top_k, True,
                             weight_is_e2m1=True)
        ok = torch.isfinite(y).all().item()
        print(f"[moe g16]   out {tuple(y.shape)}  finite={ok}  (ran without a group%32 assert = PASS)")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[moe g16]   FAILED: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    d = bench_dense()
    m = bench_moe()
    print(f"\nGROUP-16 e2m1 KERNEL: dense={'PASS' if d else 'FAIL'}  moe={'PASS' if m else 'FAIL'}")
