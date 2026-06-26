# attn_hip — native flash-attention for gfx1201 (RDNA4), Triton-free

Part of the "Triton-free or bust" push. Sibling of `gdn_hip/` (GDN already HIP) and
`zaya/cca_hip/`. Goal: replace the Triton attention kernel with a pure-HIP/rocwmma op so the
serve path has no Triton attention compile + we own the kernel.

## What's here (v0)
- `attn_kernels.hip` — `flash_prefill_kernel<HEAD_DIM>`: dense bf16 prefill, tiled flash-attention,
  causal + optional sliding window, GQA. rocwmma 16×16×16 WMMA.
- `bindings.cpp` / `op.py` — `torch.ops.attn_hip.flash_prefill(q,k,v,scale,causal,sliding_window)`,
  framework-agnostic (minisgl OR vLLM), opaque to torch.compile (fake registered).
- `attn_hip_parity.py` — numeric parity vs an fp32 SDPA reference (the build/run gate).
- `setup.py` — `GPU_ARCHS=gfx1201 python setup.py build_ext --inplace` (one AOT .so).

## The design decision that matters (read before touching the WMMA)
We **steal Atlas's algorithm, not its WMMA.** Atlas's AMD port
(`/home/pat/code/atlas-ref/kernels/strix-hip/common/inferspark_prefill.cu`, AGPL — reference only,
not copied) targets **gfx1151 (RDNA3.5)** with raw `__builtin_amdgcn_wmma_*_w32` intrinsics and a
hand-derived accumulator map `row = 2*e + (lane>>4)`. **That map is wrong on gfx1201 (gfx12).** Our
own W4A8 kernel (`../w4a8_fp8_wmma/gemm_tiled.h`) shows the gfx12 accumulator is `row=(lane>>4)*8+e`
with the `_w32_gfx12` builtins, and gfx12 WMMA takes 8 K-elems/lane (split across wave halves), not
16. Porting Atlas's raw intrinsics verbatim is the documented "naive port = garbage" trap.

So we use **rocwmma** (`load/store/mma_sync`), which encapsulates the gfx12 fragment layout, and we
**route every per-row quantity through fp32 smem**: QK^T scores are `store_matrix_sync`'d to
`smem_S` and the softmax is a plain one-thread-per-row loop; the PV output is `store_matrix_sync`'d
to `smem_PV` and folded into a fp32 `smem_O` accumulator (`O = O*resc + PV`). No WMMA
fragment-layout assumption survives into our indexing → portable + provably correct. This mirrors
Atlas's own "decouple softmax from the fragment layout" tradeoff (extra smem round-trip, correct
first).

## Status
Written, **not yet GPU-validated.** Gate = `attn_hip_parity.py` green under a 1-card lease in
`vllm22-w4a8:combined`. Expect first failures in: tail/ragged masking, the `col_major` K^T fragment
orientation (QK^T), or `store_matrix_sync` leading-dim. Debug each against the fp32 reference.

## Perf passes (after parity is green) — in order
1. **Widen warps per m-tile** — v0 uses WARPS=M_TILES=2 (one warp/m-tile), heavy underutilization.
   Split the K/d loops across more warps.
2. **Double-buffer K/V smem** (Atlas uses sync uint4 loads; overlap next-block load with compute).
3. **Bigger BR/BC** within the 64 KB LDS cap (RDNA4 did NOT raise it — budget carefully; v0 smem ≈
   Q/K/V[32×(D+8)]·2B + S/PV/O[32×D]·4B).
4. **fp8-KV** — load fp8 K/V, dequant in smem (LUT), keep WMMA bf16; or native gfx12 fp8 WMMA for
   the QK^T (our W8A8 home turf). See Atlas `paged_decode_attn_turbo*` for the LUT-dequant pattern.
5. **Paged-KV + decode kernel** — v0 is contiguous prefill only; serve needs paged + a decode path.

## Wiring (later)
Swap the engine's Triton attention call for `torch.ops.attn_hip.flash_prefill` behind a flag
(mirror how `gdn_hip` routes via `VLLM_GDN_HIP` / minisgl's kernel-provider). Validate with a
token-diff (Triton vs attn_hip), same as `gdn_hip/token_diff_qwen35.py`.
