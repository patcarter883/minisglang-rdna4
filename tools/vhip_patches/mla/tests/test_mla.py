"""Green the mla_hip kernel-builder package on gfx1201: forward parity vs a pure-torch MLA reference.

Runnable standalone (no pytest needed) inside the ROCm torch image after local/build_local.sh:
    python tests/test_mla.py
Exits nonzero on the first failure. Covers, against an fp32 absorbed-/materialized-MLA reference:
  * mla_decode            — absorbed paged-latent decode (bf16 I/O, fp32-internal).
  * mla_decode_fp8        — e4m3 latent cache; the reference runs on the dequantized cache so the
                            fp8 rounding cancels and the KERNEL's accumulate error is what's judged.
  * mla_prefill           — materialized varlen MHA, asymmetric qk_head_dim=192 / v_head_dim=128,
                            causal with a prefix offset (cold + chunked-extend).
  * mla_verify            — absorbed multi-query decode (speculative verify), per-query causal bound.
  * mla_verify_fp8        — e4m3 latent cache verify.
The kernel is fp32-internal/bf16-out, so parity is judged by cos-sim + a bf16-ULP violation bar;
fp8 variants are tolerated at the same bar (dequant cancels) plus a ~1e-1 rel-error safety net.
"""
import os
import sys

import torch
import torch.nn.functional as F

# import the built package from torch-ext/ (local build) — same package the Hub ships.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "torch-ext"))

import mla_hip as M  # noqa: E402

DEV = "cuda"
LATENT, ROPE = 512, 64
QK = LATENT + ROPE
COS_MIN = 0.9995
ULP_RTOL, ULP_ATOL = 0.016, 0.004
FP8_REL = 1e-1  # extra safety net for the fp8 paths
FAILS = []


def _judge(name, got, ref, cos_min=COS_MIN, rel_cap=None):
    got = got.detach().float()
    ref = ref.detach().float()
    ref_b = ref.bfloat16().float()
    viol = ((got - ref_b).abs() - (ULP_ATOL + ULP_RTOL * ref_b.abs())).clamp(min=0).max().item()
    cos = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    ok = (cos >= cos_min) and (viol <= 1e-6)
    if rel_cap is not None:  # fp8: accept either the ULP bar or a loose rel-error bar
        ok = (cos >= cos_min) and (viol <= 1e-6 or rel <= rel_cap)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} cos={cos:.6f}  ulp_viol={viol:.2e}  rel={rel:.2e}")
    if not ok:
        FAILS.append(name)


# ----------------------------------------------------------------- paged latent cache builder
def _build_paged_cache(B, ctx_lens, block_size, fp8=False, descale=0.25):
    S = max(ctx_lens)
    dtype = torch.float32 if fp8 else torch.bfloat16
    cache_d = torch.randn(B, S, QK, device=DEV, dtype=dtype)
    bps = (S + block_size - 1) // block_size
    num_blocks = B * bps + 3
    perm = torch.randperm(num_blocks, device=DEV).int()
    cache_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    latent_cache = torch.zeros(num_blocks, block_size, QK, device=DEV, dtype=cache_dtype)
    block_table = torch.zeros(B, bps, device=DEV, dtype=torch.int32)
    cache_q = (cache_d / descale).to(torch.float8_e4m3fn) if fp8 else cache_d
    for b in range(B):
        for lb in range(bps):
            phys = int(perm[b * bps + lb].item())
            block_table[b, lb] = phys
            for off in range(block_size):
                j = lb * block_size + off
                if j < ctx_lens[b]:
                    latent_cache[phys, off] = cache_q[b, j]
    # what the kernel effectively attends over (dequantized for fp8)
    cache_ref = (cache_q.float() * descale) if fp8 else cache_d.float()
    return cache_ref, latent_cache, block_table


# ----------------------------------------------------------------- decode parity
def test_decode(fp8=False):
    tag = "mla_decode_fp8" if fp8 else "mla_decode"
    print(f"\n== {tag} parity (absorbed, LATENT={LATENT} ROPE={ROPE}) ==")
    torch.manual_seed(0)
    B, H, ctx_lens, block_size, sw, descale = 4, 16, [128, 250, 37, 1], 16, 0, 0.25
    scale = QK ** -0.5
    q = torch.randn(B, H, QK, device=DEV, dtype=torch.bfloat16)
    cache_ref, latent_cache, block_table = _build_paged_cache(B, ctx_lens, block_size, fp8, descale)
    ctx = torch.tensor(ctx_lens, device=DEV, dtype=torch.int32)

    if fp8:
        # per-tensor descale as a 1-elem fp32 DEVICE tensor, read [0] in-kernel (graph-safe).
        dsc = torch.tensor([descale], dtype=torch.float32, device=DEV)
        got = M.mla_decode_fp8(q, latent_cache, block_table, ctx, scale, dsc, dsc, sw, 0)
    else:
        got = M.mla_decode(q, latent_cache, block_table, ctx, scale, sw, 0)

    ref = torch.empty(B, H, LATENT, device=DEV)
    for b in range(B):
        cl = ctx_lens[b]
        cb = cache_ref[b, :cl]                                # [cl, QK]
        ckv = cb[:, :LATENT]                                  # [cl, LATENT]  (V)
        scores = torch.einsum("hd,kd->hk", q[b].float(), cb) * scale
        ref[b] = torch.einsum("hk,kd->hd", F.softmax(scores, dim=-1), ckv)
    _judge(tag, got, ref, rel_cap=FP8_REL if fp8 else None)


# ----------------------------------------------------------------- verify parity (multi-query)
def test_verify(fp8=False):
    tag = "mla_verify_fp8" if fp8 else "mla_verify"
    print(f"\n== {tag} parity (absorbed multi-query) ==")
    torch.manual_seed(1)
    seqs, H, block_size, sw, descale = [(100, 5), (37, 4), (512, 3)], 16, 16, 0, 0.25
    scale = QK ** -0.5
    B = len(seqs)
    ctx_lens = [c + q for c, q in seqs]
    cache_ref, latent_cache, block_table = _build_paged_cache(B, ctx_lens, block_size, fp8, descale)
    total_q = sum(q for _, q in seqs)
    q = torch.randn(total_q, H, QK, device=DEV, dtype=torch.bfloat16)
    seq_idx, kbound = [], []
    for b, (c, ql) in enumerate(seqs):
        for qi in range(ql):
            seq_idx.append(b)
            kbound.append(c + qi + 1)
    q_seq_idx = torch.tensor(seq_idx, device=DEV, dtype=torch.int32)
    q_kbound = torch.tensor(kbound, device=DEV, dtype=torch.int32)

    if fp8:
        dsc = torch.tensor([descale], dtype=torch.float32, device=DEV)  # device per-tensor descale
        got = M.mla_verify_fp8(q, latent_cache, block_table, q_seq_idx, q_kbound, scale,
                               dsc, dsc, sw, 0)
    else:
        got = M.mla_verify(q, latent_cache, block_table, q_seq_idx, q_kbound, scale, sw, 0)

    ref = torch.empty(total_q, H, LATENT, device=DEV)
    r = 0
    for b, (c, ql) in enumerate(seqs):
        for qi in range(ql):
            cl = c + qi + 1
            cb = cache_ref[b, :cl]
            scores = torch.einsum("hd,kd->hk", q[r].float(), cb) * scale
            ref[r] = torch.einsum("hk,kd->hd", F.softmax(scores, dim=-1), cb[:, :LATENT])
            r += 1
    _judge(tag, got, ref, rel_cap=FP8_REL if fp8 else None)


# ----------------------------------------------------------------- prefill parity (materialized MHA)
def test_prefill():
    print("\n== mla_prefill parity (materialized MHA, qk192/v128) ==")
    torch.manual_seed(2)
    qk_dim, v_dim, H, sw, causal = 192, 128, 16, 0, 1
    # (q_len, k_len) per sequence; prefix = k_len - q_len (cold + extend).
    qkv_lens = [(128, 128), (37, 37), (64, 512), (32, 200)]
    scale = qk_dim ** -0.5
    q_lens = [a for a, _ in qkv_lens]
    k_lens = [b for _, b in qkv_lens]
    total_q, total_k = sum(q_lens), sum(k_lens)
    q = torch.randn(total_q, H, qk_dim, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(total_k, H, qk_dim, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(total_k, H, v_dim, device=DEV, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], device=DEV, dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(k_lens).cumsum(0).tolist()], device=DEV, dtype=torch.int32)
    max_q = max(q_lens)

    got = M.mla_prefill(q, k, v, cu_q, cu_k, scale, causal, sw, max_q)

    ref = torch.empty(total_q, H, v_dim, device=DEV)
    for b in range(len(qkv_lens)):
        ql, kl = q_lens[b], k_lens[b]
        prefix = kl - ql
        qo, ko = int(cu_q[b].item()), int(cu_k[b].item())
        qb = q[qo:qo + ql].float()
        kb = k[ko:ko + kl].float()
        vb = v[ko:ko + kl].float()
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale
        qpos = prefix + torch.arange(ql, device=DEV)
        kpos = torch.arange(kl, device=DEV)
        if causal:
            mask = kpos[None, :] > qpos[:, None]
            scores = scores.masked_fill(mask[None], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ref[qo:qo + ql] = torch.einsum("hqk,khd->qhd", attn, vb)
    _judge("mla_prefill", got, ref)


def main():
    print(f"device: {torch.cuda.get_device_properties(0).gcnArchName} | torch {torch.__version__}")
    test_decode(fp8=False)
    test_prefill()
    test_verify(fp8=False)
    print("\n--- fp8 latent cache ---")
    test_decode(fp8=True)
    test_verify(fp8=True)
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("ALL GREEN")


if __name__ == "__main__":
    main()
