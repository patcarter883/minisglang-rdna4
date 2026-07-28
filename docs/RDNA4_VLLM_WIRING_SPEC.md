# RDNA4 HIP kernels -> vLLM 0.24: complete wiring spec

Requirements (binding, set 2026-07-28): **every** HIP op that has a real vLLM call site is wired;
**no Triton fallback** anywhere on those paths; **no monkeypatching** — registration through
supported hooks; highest-performing variant; **no hacks, no casts, no copies**.

## Why this exists

Three separate wirings were found INERT on the flagship model (Qwen3.6-35B-A3B-AWQ, bs=1 decode) on
2026-07-28. Every one of them measured "this lever does nothing" — because it was never connected:

| wiring | why it never executed |
|---|---|
| `awq_dense_hip` | built + validated during GLM bring-up, never added to the qwen plugin list |
| w4a8 grouped MoE | `moe_crossover_cache.json` puts every decode M outside the engage window (`M<48`) |
| `tail_vllm` RMSNorm / RMSNormGated / MRoPE | `CustomOp.register_oot` only dispatches when `CustomOp.enabled()`; the serve runs `custom_ops=['+sparse_attn_indexer','none']`, so `forward_hip` is never reached |

Consequence: on this model at bs=1 the only HIP code actually executing is GDN + paged attention.
Stock vLLM measures 92 tok/s decode; our stack 83.3. We were not losing to Triton — we were mostly
not running.

MEASURED corollary: forcing the fp8-activation MoE on at bs=1 (`VLLM_ROCM_W4A8_FORCE=on`) gives
83.4 vs 83.4 — our fp8-act MoE is worth nothing at decode.

UPDATE (2026-07-28, profile-driven): the decode MoE was running **stock Triton
`fused_moe_kernel_gptq_awq`, the #1 decode kernel at ~26% of GPU time** — `_moe_should_engage`
consults a crossover cache that only ever described the fp8-ACTIVATION kernel, whose decode window
is empty. W4A16 (fp16 acts, no act-quant) is now wired and is 70x more accurate (rel err 0.0006 vs
0.045), but it is 14% SLOWER, and entirely because of gemm2's short-K/wide-N shape. See
`CONTINUE_rdna4_vllm_wiring.md` for the full breakdown and the GEMV-by-lanes fix.

## Architecture

ONE plugin package `rdna4_vllm`, registered via a `vllm.general_plugins` entry point (the mechanism
`tail_vllm` / `gdn_vllm` / `w4a8_vllm` already use). `register()` runs from
`vllm/v1/worker/worker_base.py:247`, before the model is built and before `Sampler` is constructed.

Two seams have NO registration hook in 0.24. They are handled as **baked source patches in a derived
image**, NOT runtime monkeypatches — precedent: `rocm_hip_attn.py` already ships as an in-tree fork.

| seam | mechanism |
|---|---|
| `LogitsProcessor` (LM head) | `PluggableLayer.register_oot(name="LogitsProcessor")` — supported |
| `SiluAndMul`, `GeluAndMul`, `RMSNorm`, `RMSNormGated`, `MRotaryEmbedding` | `CustomOp.register_oot` — supported (**requires `-O.custom_ops=+...`**) |
| AWQ dense | `AutoAWQConfig.get_quant_method` routing + `_POSSIBLE_KERNELS[ROCM]` — supported |
| MoE experts | `register_moe*` registries — supported |
| GDN | `PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")` — supported |
| `store_kv` | override `do_kv_cache_update` on `RocmHipAttentionImpl` — we own the class |
| **sampler** | **baked patch** of `v1/sample/ops/topk_topp_sampler.py` (plain `nn.Module`, no hook) |
| **`rocm_unquantized_gemm`** | **baked patch** of `model_executor/layers/utils.py` (`direct_register_custom_op` binds the impl by reference; a second registration raises) |

## No-fallback policy

Every route raises on an unsupported shape/dtype instead of deferring. This is a deliberate reversal:
the `_moe_should_engage` crossover and the `_POSSIBLE_KERNELS` fall-through are exactly how we
shipped kernels that never ran. Validation happens at `process_weights_after_loading` where possible,
so an unsupported model fails at BOOT, not silently at step 3000.

## No casts / no copies

Audit of what must go:
* `vllm_oot_slotfix.py` spec path: `mqkv_spec.float().contiguous()`, `a_spec.float().contiguous()`,
  `b_spec.float().contiguous()` — cast + copy, 3x per layer per step. The gdn kernels are
  `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, ...)`, so they take the activations natively.
* `_split_conv_qkv` / spec split: 3x `.reshape(...).contiguous()` per layer (19.1 us/layer measured).
  Fix by having the conv kernel write the already-split q/k/v layout, or by making the recurrence
  kernels stride-aware over the split view (same class of fix as the ssm-state stride work).
* W4A16 MoE: `x.to(float16)` is a no-op only when the serve is fp16. Under bf16 it is a real cast ->
  REFUSE instead (bf16 + W4A16 is a genuine gap; the kernel is fp16-typed).
* Scales: `MoeWNA16Method` allocates in `params_dtype`; fp16 serve -> already fp16, no cast. bf16
  serve -> convert ONCE at load, never per step.

## Op inventory and status

Wireable, with a real vLLM call site:

| op(s) | site | status |
|---|---|---|
| `attn_decode.flash_decode_paged[_fp8]`, `attn_prefill_paged.flash_prefill_paged[_fp8]` | `rocm_hip_attn.py` in-tree fork | **already wired** |
| `tail_hip.rms_norm`, `rms_norm_add`, `rope`(mrope), `gdn_hip.rmsnorm_gated` | `CustomOp` OOT | registered but **INERT** until `custom_ops=+...` |
| `tail_hip.silu_and_mul`, `gelu_and_mul` | `activation.py` `SiluAndMul`/`GeluAndMul` | TODO |
| `tail_hip.store_kv` | `RocmHipAttentionImpl.do_kv_cache_update` (currently `triton_reshape_and_cache_flash`) | TODO — highest-ranked |
| `fp8_wmma.mmq_regdirect_w4a16_moe_gemv` | `MoeWNA16Method` | **wired + parity-green, DEFAULT OFF** — accuracy 70x better than fp8-act, but -14% tok/s: gemm2 (K=256, N=2048) runs at 39 GB/s vs gemm1's 310. Needs the GEMV-by-lanes tiling. |
| `fp8_wmma.dense_bf16_gemv`, `dense_gemm.dense_gemm_rd/_pipe` | `LogitsProcessor._get_logits` | **wired + CONFIRMED ENGAGED** — 680 GB/s, at bandwidth, worth ~0. Do not tune. |
| `sampler_hip.top_k_top_p_sampling_from_logits` | `TopKTopPSampler` | TODO — baked patch |
| `gdn_hip.gdn_decode_conv[_gated]`, `gdn_decode_gated` | our own `_forward_core_gdn_hip` | partially (conv fused; gated norm still separate) |
| `gdn_hip.gdn_prefill_verify`, `causal_conv1d_fwd_verify` | spec branch | **wired + fused publish (this session)** |
| `moe.moe_align`, `moe_splitk.*` | our own `moe_experts.py` | TODO |
| `mla.*` (5) | `TRITON_MLA` via `register_backend` | wired only via mounted `mla_vllm`; NOT baked |

Cannot be wired — no call site exists (clean negative, do not re-derive):
* `cca` (8 ops) — ZAYA is not in vLLM 0.24's model zoo.
* `gdn.rmsnorm_gated_bwd`, `gdn.causal_conv1d_bwd` — training-only.
* `attn_decode.flash_decode` (non-paged) — v1 is always paged.
* `custom_ar.alloc_shared/get_ipc_handle/open_ipc_handle/flag_probe` — host-side helpers.
* `custom_ar.one_shot_ar` / `all_gather_p2p` — a call site exists but the prototype is SHELVED:
  measured 0% win, plus corruption under cudagraph and deadlock under `--enforce-eager`.

## Landing order (cost-ranked, each measured independently)

1. `custom_ops=+rms_norm,+rms_norm_add,+silu_and_mul` — zero code, turns on wiring already written.
2. `store_kv` — only per-layer decode op still genuinely Triton; we own the override point.
3. W4A16 MoE — no repack for decode; directly attacks the fp8-act degradation.
4. `gdn_decode_conv_gated` + remove the spec-path casts/copies.
5. LM head (after `dense_gemm` is built into the image) + sampler (baked patch).
