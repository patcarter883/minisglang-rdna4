# minisgl-rdna4 — RDNA4 (gfx1201) native serving engine

A fork of [mini-SGLang](https://github.com/sgl-project/mini-sglang) re-targeted from NVIDIA/CUDA to
AMD RDNA4 (gfx1201, Radeon RX 9070 XT), built around the `w4a8_fp8_wmma` int4-weight/fp8-activation
WMMA kernel and the tuned RDNA4 `triton_attn`, extended with GDN (Gated Delta Net) hybrid support.

- **Forked from upstream:** `9a91cfafe754aa85daee49998176275667eb58f2` (mini-SGLang `main`, 2026-05-17).
- **Full design + rationale:** `vllm-gfx1201/docs/RDNA4_ENGINE_DESIGN.md` (the north star).
- **Companion analysis:** `vllm-gfx1201/docs/MINISGLANG_PATHWAYS.md`.

## Governing principles (do not violate)

1. **Maximise RDNA4 strengths** — native fp8/int4 WMMA, 3D flash-decode, `waves_per_eu` tuning.
2. **Nothing dequants to F16.** I/O = bf16, compute = fp8 (e4m3**fn**, NOT fnuz), accumulate = f32.
   Banned = per-element F16 dequant of quantized weights/activations. fp32 accumulate + bf16
   activations are REQUIRED, not violations.
3. **Clean, tidy, agent+human-maintainable.** Keep mini-SGLang's style: small typed modules,
   Protocol-based backends, `__dict__`-introspection weight loading. No special-case spaghetti.
4. **W4A8 kernel is consumed as a dependency** from `vllm-gfx1201/w4a8_fp8_wmma/` (single source of
   truth) — call the raw `torch.ops.w4a8_fp8_wmma.*` ops; NEVER copy its csrc.

## Equivalence oracle

Greedy **token-identical vs `vllm22-w4a8:combined`** (same kernels) — tighter than vs HF. Reuse the
`vllm-gfx1201/patches/run_het_e2e_combined.sh` pattern. Stand up before Phase 2.

## Phase status

| Phase | What | Status |
|---|---|---|
| 0 | Fork + strip NVIDIA deps + boot Qwen2.5-0.5B bf16 eager TP=1, greedy-equiv | **IN PROGRESS** |
| 1 | Tuned RDNA4 `triton_attn` backend + fp8-KV | todo |
| 2 | W4A8 dense (`LinearMethod`) + MoE backend + weight-loader fix → 7B-AWQ | todo |
| ★ | GATE: re-decide 35B GDN port | — |
| 3 | GDN hybrid (3a state cache → 3b layer numerics → 3c scheduler split → 3d serve) | todo |
| 4 | RCCL TP + het-TP (re-derive ratio) + decode HIP graphs + parity | todo |

## Change log (what we've diverged from upstream + why)

_Phase 0 — dependency strip. NVIDIA imports in mini-SGLang are all lazy; replace with torch/lifted
Triton. Targets (file:line from audit):_
- [x] `layers/norm.py` flashinfer rmsnorm/fused_add_rmsnorm → fp32-internal torch RMSNorm.
- [x] `layers/rotary.py` flashinfer apply_rope → torch NeoX rotate-half (fp32 internal, reuses
      `_cos_sin_cache`).
- [x] `layers/activation.py` flashinfer silu/gelu_and_mul → fp32-internal gated torch activation.
- [x] `engine/sample.py` flashinfer.sampling → torch softmax + per-row top-k/top-p + multinomial
      (greedy still argmax). NOTE: stochastic sampling won't be RNG-identical to flashinfer; only
      greedy is the token-identity oracle.
- [x] `kvcache/mha_pool.py` store_cache `.cu` → torch scatter (bf16 path; fp8 KV write lands in P1).
- [x] `layers/embedding.py` index `.cu` → torch vocab-parallel gather + mask + all_reduce.
- [x] `kernel/radix.py` fast_compare_key `.cu` → torch common-prefix-length (matches std::mismatch).
- [ ] `moe/fused.py:16,71` sgl_kernel topk_softmax/moe_align → DEFERRED (MoE-only path; our W4A8 MoE
      backend replaces it in Phase 2; not on the dense boot path).
- [ ] `engine/engine.py:223` + `utils/arch.py:12` — ROCm `auto` falls to `fi`. DEFERRED to Phase 1:
      add the RDNA4 `triton_attn` backend + select it on ROCm (no backend exists yet to select).
- [x] collectives: default `TorchDistributedImpl` = RCCL on ROCm — zero change at TP=1 (verified).

_Precision note: all shims are fp32-internal / bf16-I-O (honor "no F16"); they are correctness-first
placeholders, validated by token-diff vs the combined image once Phase 1 attention lands._

_Verification: `python -m py_compile` passes on all edited files; no top-level NVIDIA imports remain
(the residual `flashinfer`/`sgl_kernel` refs are the lazy CUDA attention backends — replaced in
P1 — and the deferred MoE path)._
