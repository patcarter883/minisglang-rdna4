# moe_hip — native HIP moe_align_block_size for gfx1201

W4A8 grouped-MoE needs routed tokens grouped per expert with each expert's run padded to a multiple
of `block_size`, so the grouped GEMM can process whole `[block_size]`-row tiles. This replaces the
`from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size` host op
on the serve path (removes the vLLM dependency for this step; native HIP, Triton-free).

## Op
`torch.ops.moe_hip.moe_align(topk_ids[M,top_k] int32, num_experts, block_size)
-> (sorted_ids[P] int32, expert_ids[cdiv(P,bs)] int32, num_tokens_post_pad[1] int32)`.
Drop-in for `moe_align_block_size(topk_ids, block_size, num_experts, None, pad_sorted_ids=True)`.

## Output contract (bit-compatible with vLLM, pad_sorted_ids=True)
- `P = max_num_tokens_padded = round_up(numel + E*(bs-1), bs)`, clamped to `numel*bs` when
  `numel < E` (numel = topk_ids.numel()). Computed identically in C++ so the P-row GEMM intermediates
  and `ident = arange(P)` match the reference exactly.
- `sorted_ids`: flat topk indices (0..numel-1) in per-expert contiguous runs, each run padded UP to a
  multiple of bs with the **sentinel = numel** (an out-of-range token the GEMM skips).
- `expert_ids`: expert index per block, ascending by expert; unused tail blocks zeroed (GEMM bounds
  by ntp so they're never read).
- `num_tokens_post_pad`: Σ_e round_up(count[e], bs).

Token ORDER within an expert's run is **arbitrary** — the downstream gather/scatter-reduce sums per
ORIGINAL token, so the final MoE output is invariant to intra-expert order. We guarantee a VALID
alignment, not bit-identical ordering.

## Implementation
Single workgroup, single launch (at decode `numel = M*top_k` is tiny → launch-bound, so one kernel
beats a multi-launch count/scan/scatter): sentinel-fill → count per expert (shared atomics) →
serial exclusive-scan + expert_ids fill + ntp on thread 0 → scatter (shared running-counter atomics).
Shared `cnt[E]+off[E]`, E up to 4096 (32 KB). No WMMA, no Triton.

## Status / wiring
GPU-validated by `moe_hip_parity.py` — ALL PASS equivalence vs vLLM (same P/ntp/expert_ids, same
per-expert token sets, correct sentinel padding) across decode/prefill/collision/large-E shapes; and
end-to-end `tools/moe_parity.py` is bit-identical with native align ON vs the vLLM-align baseline.
Wired in `quant/kernels.py` (`w4a8_moe`), on by default, `MINISGL_MOE_ALIGN=0` reverts to vLLM.
AOT build: `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace` (.so gitignored, rebuild per checkout).
