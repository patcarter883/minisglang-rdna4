"""Numeric parity for the native mla_hip decode kernel (GPU).

MLA absorbed decode: q = [q_nope_absorbed (LATENT) ‖ q_rope (ROPE)]; the paged latent cache stores
[c_KV (LATENT) ‖ k_rope (ROPE)] SHARED across heads. score[j] = (q·cache[j])*scale over LATENT+ROPE;
out = Σ_j softmax(score)[j] * c_KV[j] (V = first LATENT dims). Checked vs an fp32 reference; the
kernel is fp32-internal/bf16-out so we judge cos-sim + max|Δ| within 2 bf16 ULP.

Run inside vllm22-w4a8:combined under a 1-card lease (see attn_decode_parity.py for the docker line).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import mla_hip  # noqa: F401  loads the .so + registers torch.ops.mla_hip.*

DEV = "cuda"
torch.manual_seed(0)
LATENT, ROPE = 512, 64
QK = LATENT + ROPE
COS_MIN = 0.9995
ULP_RTOL, ULP_ATOL = 0.016, 0.004


def check(name, B, H, ctx_lens, block_size=16, sw=0) -> bool:
    scale = QK ** -0.5
    S = max(ctx_lens)
    q = torch.randn(B, H, QK, device=DEV, dtype=torch.bfloat16)
    cache_d = torch.randn(B, S, QK, device=DEV, dtype=torch.bfloat16)        # dense latent (ckv‖krope)
    bps = (S + block_size - 1) // block_size
    num_blocks = B * bps + 3
    perm = torch.randperm(num_blocks, device=DEV).int()
    latent_cache = torch.zeros(num_blocks, block_size, QK, device=DEV, dtype=torch.bfloat16)
    block_table = torch.zeros(B, bps, device=DEV, dtype=torch.int32)
    ctx = torch.tensor(ctx_lens, device=DEV, dtype=torch.int32)
    for b in range(B):
        for lb in range(bps):
            phys = int(perm[b * bps + lb].item())
            block_table[b, lb] = phys
            for off in range(block_size):
                j = lb * block_size + off
                if j < ctx_lens[b]:
                    latent_cache[phys, off] = cache_d[b, j]

    got = torch.ops.mla_hip.mla_decode(q, latent_cache, block_table, ctx, scale, sw, 0).float()

    ref = torch.empty(B, H, LATENT, device=DEV)
    for b in range(B):
        cl = ctx_lens[b]
        cb = cache_d[b, :cl].float()                          # [cl, QK]
        ckv = cb[:, :LATENT]                                  # [cl, LATENT]  (V)
        scores = torch.einsum("hd,kd->hk", q[b].float(), cb) * scale   # [H, cl]
        if sw > 0:
            kpos = torch.arange(cl, device=DEV)
            scores = scores.masked_fill(((cl - 1 - kpos) >= sw)[None], float("-inf"))
        ref[b] = torch.einsum("hk,kd->hd", F.softmax(scores, dim=-1), ckv)   # [H, LATENT]

    ref_b = ref.bfloat16().float()
    viol = ((got - ref_b).abs() - (ULP_ATOL + ULP_RTOL * ref_b.abs())).clamp(min=0).max().item()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    ok = (cos >= COS_MIN) and (viol <= 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:30s} cos={cos:.6f}  ulp_viol={viol:.2e}")
    return ok


def check_fp8(name, B, H, ctx_lens, block_size=16, sw=0, descale=0.25) -> bool:
    """fp8 latent-cache decode. The latent cache is e4m3-quantized (store=value/descale, dequant
    =fp8*descale); the fp32 reference is computed on the DEQUANTIZED values, so the e4m3 rounding
    cancels and what's measured is the KERNEL's fp32-accumulate error (same bar as the bf16 path)."""
    scale = QK ** -0.5
    S = max(ctx_lens)
    q = torch.randn(B, H, QK, device=DEV, dtype=torch.bfloat16)
    cache_d = torch.randn(B, S, QK, device=DEV)                              # dense latent, fp32
    # per-tensor e4m3 quantize, then dequantize for the reference (quant error is in both -> cancels)
    cache_q = (cache_d / descale).to(torch.float8_e4m3fn)                    # [B,S,QK] e4m3
    cache_deq = cache_q.float() * descale                                    # what the kernel sees
    bps = (S + block_size - 1) // block_size
    num_blocks = B * bps + 3
    perm = torch.randperm(num_blocks, device=DEV).int()
    latent_cache = torch.zeros(num_blocks, block_size, QK, device=DEV, dtype=torch.float8_e4m3fn)
    block_table = torch.zeros(B, bps, device=DEV, dtype=torch.int32)
    ctx = torch.tensor(ctx_lens, device=DEV, dtype=torch.int32)
    for b in range(B):
        for lb in range(bps):
            phys = int(perm[b * bps + lb].item())
            block_table[b, lb] = phys
            for off in range(block_size):
                j = lb * block_size + off
                if j < ctx_lens[b]:
                    latent_cache[phys, off] = cache_q[b, j]

    got = torch.ops.mla_hip.mla_decode_fp8(q, latent_cache, block_table, ctx, scale,
                                           descale, descale, sw, 0).float()

    ref = torch.empty(B, H, LATENT, device=DEV)
    for b in range(B):
        cl = ctx_lens[b]
        cb = cache_deq[b, :cl]                                # [cl, QK]  (dequantized)
        ckv = cb[:, :LATENT]                                  # [cl, LATENT]  (V)
        scores = torch.einsum("hd,kd->hk", q[b].float(), cb) * scale   # [H, cl]
        if sw > 0:
            kpos = torch.arange(cl, device=DEV)
            scores = scores.masked_fill(((cl - 1 - kpos) >= sw)[None], float("-inf"))
        ref[b] = torch.einsum("hk,kd->hd", F.softmax(scores, dim=-1), ckv)   # [H, LATENT]

    ref_b = ref.bfloat16().float()
    viol = ((got - ref_b).abs() - (ULP_ATOL + ULP_RTOL * ref_b.abs())).clamp(min=0).max().item()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    ok = (cos >= COS_MIN) and (viol <= 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:30s} cos={cos:.6f}  ulp_viol={viol:.2e}")
    return ok


# ---------------------------------------------------------------------------
# PREFILL (materialized MLA): dense varlen MHA, asymmetric qk_head_dim=192 / v_head_dim=128.
# q,k:[total,H,192]  v:[total,H,128] packed varlen via cu_seqlens_q/k; causal carries the prefix
# offset (prefix_len = k_len - q_len), so cold prefill (q_len==k_len) and extend (prefix>0) both work.
QK_DIM, V_DIM = 192, 128
COS_MIN_PRE = 0.9995


def check_prefill(name, qkv_lens, H=16, sw=0, causal=1) -> bool:
    """qkv_lens: list of (q_len, k_len) per sequence (k_len >= q_len; prefix = k_len - q_len)."""
    scale = QK_DIM ** -0.5
    q_lens = [a for a, _ in qkv_lens]
    k_lens = [b for _, b in qkv_lens]
    total_q, total_k = sum(q_lens), sum(k_lens)
    q = torch.randn(total_q, H, QK_DIM, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(total_k, H, QK_DIM, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(total_k, H, V_DIM, device=DEV, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device=DEV, dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], device=DEV, dtype=torch.int32)
    max_q = max(q_lens)

    got = torch.ops.mla_hip.mla_prefill(q, k, v, cu_q, cu_k, scale, causal, sw, max_q).float()

    ref = torch.empty(total_q, H, V_DIM, device=DEV)
    for b in range(len(qkv_lens)):
        ql, kl = q_lens[b], k_lens[b]
        prefix = kl - ql
        qo, ko = int(cu_q[b].item()), int(cu_k[b].item())
        qb = q[qo:qo + ql].float()                                  # [ql,H,192]
        kb = k[ko:ko + kl].float()                                  # [kl,H,192]
        vb = v[ko:ko + kl].float()                                  # [kl,H,128]
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale       # [H,ql,kl]
        qpos = prefix + torch.arange(ql, device=DEV)                # [ql]
        kpos = torch.arange(kl, device=DEV)                         # [kl]
        if causal:
            mask = kpos[None, :] > qpos[:, None]                    # [ql,kl]
            if sw > 0:
                mask = mask | ((qpos[:, None] - kpos[None, :]) >= sw)
            scores = scores.masked_fill(mask[None], float("-inf"))
        attn = F.softmax(scores, dim=-1)                            # [H,ql,kl]
        ref[qo:qo + ql] = torch.einsum("hqk,khd->qhd", attn, vb)    # [ql,H,128]

    ref_b = ref.bfloat16().float()
    viol = ((got - ref_b).abs() - (ULP_ATOL + ULP_RTOL * ref_b.abs())).clamp(min=0).max().item()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    ok = (cos >= COS_MIN_PRE) and (viol <= 1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} cos={cos:.6f}  ulp_viol={viol:.2e}")
    return ok


def main() -> None:
    print("=== mla_hip decode parity (vs fp32 absorbed-MLA, LATENT=512 ROPE=64) ===")
    ok = True
    ok &= check("B1 H16 ctx[128]", 1, 16, [128])
    ok &= check("B1 H16 ctx[2048] (long)", 1, 16, [2048])
    ok &= check("B4 H16 ctx[100,250,37,1]", 4, 16, [100, 250, 37, 1])
    ok &= check("B1 H128 ctx[512]", 1, 128, [512])
    ok &= check("B1 H16 ctx[37] (ragged)", 1, 16, [37])
    ok &= check("B1 H16 ctx[1000] bs32", 1, 16, [1000], block_size=32)
    ok &= check("SWA=256 B1 H16 ctx[1024]", 1, 16, [1024], sw=256)
    print("--- fp8 latent cache ---")
    ok &= check_fp8("fp8 B1 H16 ctx[128]", 1, 16, [128])
    ok &= check_fp8("fp8 B1 H16 ctx[2048] (long)", 1, 16, [2048])
    ok &= check_fp8("fp8 B4 H16 ctx[100,250,37,1]", 4, 16, [100, 250, 37, 1])
    ok &= check_fp8("fp8 B1 H128 ctx[512]", 1, 128, [512])
    ok &= check_fp8("fp8 B1 H16 ctx[1000] bs32", 1, 16, [1000], block_size=32)
    ok &= check_fp8("fp8 SWA=256 B1 H16 ctx[1024]", 1, 16, [1024], sw=256)
    print("--- prefill (materialized MHA, qk192/v128) ---")
    ok &= check_prefill("cold B1 q=k=128", [(128, 128)])
    ok &= check_prefill("cold B1 q=k=512", [(512, 512)])
    ok &= check_prefill("cold B1 q=k=37 (ragged)", [(37, 37)])
    ok &= check_prefill("cold B3 q=k=[100,250,37]", [(100, 100), (250, 250), (37, 37)])
    ok &= check_prefill("extend B1 q=64 k=512 (prefix=448)", [(64, 512)])
    ok &= check_prefill("extend B2 q=[32,16] k=[200,300]", [(32, 200), (16, 300)])
    ok &= check_prefill("cold B1 H128 q=k=256", [(256, 256)], H=128)
    ok &= check_prefill("SWA=128 cold B1 q=k=512", [(512, 512)], sw=128)
    ok &= check_prefill("non-causal B1 q=k=128", [(128, 128)], causal=0)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
