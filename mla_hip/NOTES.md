# mla_hip — native HIP Multi-head Latent Attention (MLA) for gfx1201

The kernel DeepSeek-style **MLA** models (e.g. GLM-4.7-Flash / Glm4MoeLite) need to serve Triton-free.
MLA compresses the KV cache to a per-token **latent** (no per-head K/V) — `[c_KV (kv_lora_rank=512) ‖
k_rope (qk_rope_head_dim=64)] = 576` dims, SHARED across all query heads. None of the GQA/MHA kernels
(`attn_decode`, `attn_hip`, `attn_prefill_paged`) apply; vLLM even routes MLA through a separate
`mla/` backend.

## v0: absorbed DECODE
The efficient form (vLLM/SGLang). W_UK absorbed into the q up-proj, W_UV into the output proj — OUTSIDE
this kernel. The kernel gets a per-head absorbed query `q = [q_nope_absorbed (512) ‖ q_rope (64)]` and:
```
score[j] = (q · cache[j]) * scale          # dot over all 576 dims (nope-latent + rope)
attn     = softmax_j(score)                # causal
out      = Σ_j attn[j] * c_KV[j]           # V = the FIRST 512 dims of the cache (the latent)
```
Output is the per-head 512-dim latent context; the W_UV up-projection to v_head_dim is external.

Structurally a flash-DECODE (M=1, pure FMA + warp-shuffle online softmax, like `attn_decode`) with an
**asymmetric qk_dim=576 / v_dim=512** and **num_kv_heads=1** (shared latent) — no WMMA. Each lane owns
576/32=18 q/k dims and 512/32=16 output (latent) dims; V reuses the latent part of the loaded key.

## Op
`torch.ops.mla_hip.mla_decode(q[B,H,576], latent_cache[num_blocks,bs,576], block_table[B,max_blocks]
int32, context_lens[B] int32, scale, sliding_window, kv_block_stride=0) -> [B,H,512]`. Paged
(`block_table` + `kv_block_stride`; 0 = contiguous, or pass a stride for an interleaved/strided cache).

## v1: materialized PREFILL (`mla_prefill_kernels.hip`)
`torch.ops.mla_hip.mla_prefill(q[total_q,H,192], k[total_kv,H,192], v[total_kv,H,128],
cu_seqlens_q[S+1] int32, cu_seqlens_k[S+1] int32, scale, causal, sliding_window, max_seqlen_q)
-> [total_q,H,128]`. DeepSeek/GLM MLA prefill runs the MATERIALIZED form: the model layer up-projects
the latent KV to per-head K (W_UK) and V (W_UV) and concatenates decoupled RoPE, so the kernel sees a
regular varlen MHA — EXCEPT the QK^T contract dim (192 = 128 nope + 64 rope) ≠ the PV output dim
(128). num_kv_heads == num_q_heads (no GQA). Reuses attn_prefill_paged's WMMA core + smem online
softmax; K/V are PACKED varlen (cu_seqlens_k), not paged. Causal carries the prefix offset
(prefix_len = k_len - q_len) so cold prefill (q_len==k_len) AND chunked-extend (prefix>0) both work.
LDS: symmetric BR=BC=32 overflows 64 KB (~73.6 KB) at qk192/v128, so v0 uses **BR=32, BC=16**
(~59.8 KB; M_TILES=2 / N_TILES=1). Parity all-pass at the strict 2-ULP bar (cold/extend/ragged/H128/
SWA/non-causal). fp8 prefill: NOT YET.

## Status / scope
DECODE v0: bf16 latent cache, paged, causal + optional SWA, kv_lora_rank=512 + qk_rope=64 (GLM/DeepSeek);
fp8 latent cache variant landed. PREFILL v1: bf16 materialized qk192/v128 (above).
Validate with `mla_hip_parity.py` (fp32 reference). **NOT YET (the rest of MLA serving):**
1. **fp8 prefill** — materialized K/V in e4m3 (same dequant-on-smem-load pattern as attn_prefill_paged_fp8).
2. **The W_UK/W_UV absorption + decoupled-RoPE** — these live in the model layer (Python), not here.
3. Other ranks/dims (add template instantiations for LATENT/ROPE != 512/64, qk/v != 192/128).
4. **DeepSeek model + latent KV-cache integration** (#13) — the next program step that wires these ops.

## Wiring (later)
vLLM MLA goes through the `mla/` backend (NOT `triton_attn`), so the existing attention patch won't
intercept it — MLA needs its own routing hook into that backend. Out of scope for this scaffold.
