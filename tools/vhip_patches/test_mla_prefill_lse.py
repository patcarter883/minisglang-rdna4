"""Parity test for mla_hip.mla_prefill_lse — the LSE output added for vLLM's chunked-context merge.

Checks three things against a float32 torch reference, on the two MLA shapes the kernel templates:
  1. mla_prefill (no LSE) is unchanged — the new optional output must not perturb the old path.
  2. mla_prefill_lse's attention output matches mla_prefill exactly.
  3. the LSE itself matches logsumexp(q@k^T * scale) in NATURAL log, FA's [num_heads, total_q] layout.

(3) is the one that matters: an LSE that is off by a constant factor (log2 vs ln) still *looks*
plausible on its own but silently corrupts every chunked-prefill merge, which is exactly the case a
plain output-only parity test would pass.
"""

import torch
import mla_hip

torch.manual_seed(0)
DEV = "cuda"


def reference(q, k, v, cu_q, cu_k, scale, causal):
    """fp32 varlen reference. Returns (out[total_q,H,V], lse[H,total_q]) in natural log."""
    H = q.shape[1]
    out = torch.zeros(q.shape[0], H, v.shape[2], dtype=torch.float32, device=q.device)
    lse = torch.full((H, q.shape[0]), float("-inf"), dtype=torch.float32, device=q.device)
    for s in range(len(cu_q) - 1):
        q0, q1 = int(cu_q[s]), int(cu_q[s + 1])
        k0, k1 = int(cu_k[s]), int(cu_k[s + 1])
        qs = q[q0:q1].float().transpose(0, 1)          # [H, Lq, QK]
        ks = k[k0:k1].float().transpose(0, 1)          # [H, Lk, QK]
        vs = v[k0:k1].float().transpose(0, 1)          # [H, Lk, V]
        scores = torch.einsum("hqd,hkd->hqk", qs, ks) * scale
        if causal:
            lq, lk = q1 - q0, k1 - k0
            prefix = lk - lq                            # query t sees keys [0, prefix+t]
            pos = torch.arange(lq, device=q.device)[:, None] + prefix
            mask = torch.arange(lk, device=q.device)[None, :] > pos
            scores = scores.masked_fill(mask, float("-inf"))
        lse[:, q0:q1] = torch.logsumexp(scores, dim=-1)
        out[q0:q1] = torch.softmax(scores, dim=-1).matmul(vs).transpose(0, 1)
    return out, lse


def run(name, qk_nope, qk_rope, v_dim, seqlens_q, seqlens_k, causal):
    H, QK = 4, qk_nope + qk_rope
    cu_q = torch.tensor([0, *torch.cumsum(torch.tensor(seqlens_q), 0).tolist()],
                        dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, *torch.cumsum(torch.tensor(seqlens_k), 0).tolist()],
                        dtype=torch.int32, device=DEV)
    tq, tk = int(cu_q[-1]), int(cu_k[-1])
    q = torch.randn(tq, H, QK, dtype=torch.bfloat16, device=DEV)
    k = torch.randn(tk, H, QK, dtype=torch.bfloat16, device=DEV)
    v = torch.randn(tk, H, v_dim, dtype=torch.bfloat16, device=DEV)
    scale = QK ** -0.5
    max_q = max(seqlens_q)

    o_plain = mla_hip.mla_prefill(q, k, v, cu_q, cu_k, scale, int(causal), 0, max_q)
    o_lse, lse = mla_hip.mla_prefill_lse(q, k, v, cu_q, cu_k, scale, int(causal), 0, max_q)
    ref_o, ref_lse = reference(q, k, v, cu_q, cu_k, scale, causal)

    same = torch.equal(o_plain, o_lse)
    o_err = (o_lse.float() - ref_o).abs().max().item()
    finite = torch.isfinite(ref_lse)
    lse_err = (lse[finite] - ref_lse[finite]).abs().max().item()
    inf_ok = bool((~torch.isfinite(lse[~finite])).all()) if (~finite).any() else True

    print(f"{name:36s} out_match={same}  out_maxerr={o_err:.4f}  "
          f"lse_maxerr={lse_err:.4f}  neg_inf_ok={inf_ok}")
    assert same, f"{name}: mla_prefill_lse output diverged from mla_prefill"
    assert o_err < 0.06, f"{name}: attention output off reference by {o_err}"
    assert lse_err < 0.06, f"{name}: LSE off reference by {lse_err} (log-base bug?)"
    assert inf_ok, f"{name}: empty rows must be -inf"
    return True


ok = True
# GLM-4.7-Flash: qk 192+64=256, v 256
ok &= run("GLM causal (cold prefill)", 192, 64, 256, [37, 8], [37, 8], True)
ok &= run("GLM causal (with prefix)", 192, 64, 256, [12, 5], [40, 33], True)
ok &= run("GLM non-causal (ctx chunk)", 192, 64, 256, [12, 5], [40, 33], False)
ok &= run("GLM non-causal (chunk<query)", 192, 64, 256, [20, 9], [7, 3], False)
# DeepSeek: qk 128+64=192, v 128
ok &= run("DeepSeek causal", 128, 64, 128, [33, 17], [33, 17], True)
ok &= run("DeepSeek non-causal (chunk)", 128, 64, 128, [33, 17], [64, 40], False)
print("\nALL PARITY CHECKS PASSED" if ok else "FAILED")
