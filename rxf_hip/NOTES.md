# rxf_hip — native RXF W4A8 for gfx1201 (RDNA4)

Migrated from the Triton RXF kernels in
`vllm-gfx1201/paroquant_rotation/rxf_kernels.py` into native HIP (Triton-free), to match this
repo's native-HIP direction. **RXF = W4(NL-codebook) · A8(int8) with a fixed block-diagonal
Hadamard rotation.** Scope of this migration: **dense linear + grouped MoE, Hadamard-32 only**
(no learned Givens — RXF's own measurements found Givens null/negative on current models).

## What RXF is (vs this repo's other "W4A8")

This repo already had a `W4A8` path (`w4a8_fp8_wmma`): int4 weights but **fp8 (e4m3)**
activations on the fp8 WMMA matrix core, no outlier handling. RXF is a *different* W4A8:

| axis | `w4a8_fp8_wmma` (existing) | `rxf_hip` (this) |
|---|---|---|
| weights | uniform int4 (AWQ/GPTQ) | **NL codebook** (IQ4-NL, non-uniform integer table) |
| activations | fp8 e4m3 | **int8** (per-token absmax) |
| matmul | fp8 WMMA | **int8 WMMA** (`i32_16x16x16_iu8`) |
| outliers | none | **Hadamard FWHT** pre-pass (orthonormal → cancels in the dot) |
| checkpoint | AWQ/GPTQ | bespoke RXF (rotated weights + NL) |

The rotation is the point: an orthonormal FWHT over each size-32 group spreads activation
outliers across channels, so `(H·a)·(H·w)ᵀ = a·wᵀ` exactly while the per-group 4-bit weight
scale tightens. Weights are rotated **offline**; activations are rotated **at runtime** by the
same H — so this only works on RXF-format checkpoints, not on the existing AWQ weights.

## Pivotal fact: int8 WMMA *does* exist on RDNA4

gfx12 retains the RDNA3 integer WMMA. `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`
compiles for gfx1201 and is wrapped by rocwmma's `int8_t` fragments
(`rocwmma/internal/wmma_impl.hpp:981`). The existing fp8 path never used it (it only needed
fp8), which is why it looked absent. There is **no K=32 int8 variant** (only iu4 has one) — but
RXF's group size 32 == exactly **two** 16-wide int8 K-steps, so the per-group fp16 weight scale
lands cleanly on a 2-kstep boundary.

## Ops (`torch.ops.rxf_hip.*`)

Built out to MIRROR the production fp8 W4A8 kernels (`w4a8_fp8_wmma/{gemm_tiled.h,
moe_gemm_tiled.h,moe_kernel.hip}`). The fp8 lesson that holds for int8: the gfx12 WMMA fragment
layout is IDENTICAL (same `v2i` operand packing, same `row=(lane>>4)*8+e, col=lane&15`
accumulator), so we reuse the validated fp8 tiling verbatim with the int8 builtin — raw
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`, register-resident int32 accumulation, no smem
round-trip. Only fp8→int8 staging (NL unpack vs e4m3 decode) and acc dtype differ.

- `rotate_quant_int8(x, span) -> (q_int8, scale_fp32)` — fused FWHT-span + per-token int8 quant.
- `linear(q, a_scale, w_packed, w_scale, nl, bias?) -> bf16` — **WMMA** (M>2): BM=64×BN=128 tile,
  4 warps × 8 N-frags, `BK_TILE=128` LDS staging chunk decoupled from the 32-wide scale group
  (== fp8's `group_size` staging — one sync per 128, not per 32), vectorized uint32 weight
  staging (8 NL codes/iter). **GEMV** (M≤2): scalar int8 dot.
- `moe_gemm` / `moe_gemm_scatter` — BN=128 tiled WMMA grouped GEMM, `SCATTER` template fuses
  gemm2's topk-weighted atomic scatter into the (M,K) fp32 acc (== `mmq_fp8_moe_gemm_scatter`).
  `BK_TILE` is scatter-conditional: decode-scatter keeps the low-LDS 32 (occupancy-bound, few real
  rows); prefill non-scatter uses 128 (many token-blocks → sync reduction wins). **`WARPS_N`
  warps split the BN columns** (each owns `NFRAG/WARPS_N`=2 fragments) so `block_m` stays 16
  (minimal padding) while the block runs 4 warps → full WGP occupancy + 4× less register pressure
  than the fp8 v5 design's 1-warp-does-all-N. This is what made RXF MoE *faster* than fp8.
- `moe_gemv` — per-token GEMV for **gemm1 at decode (M≤2)**, == fp8's per-GEMM kernel selection.
  One warp/output-column, activation row staged in LDS once; with GROUP==32==warp width, lane l
  owns k=g*32+l so a single warp-reduce is the group dot. Padding rows early-exit the whole block
  → none of the ~16× WMMA-on-padding waste. This is what closed the 3× decode-MoE gap.
- `moe_gather_reduce` — prefill gemm2 epilogue (gather by sorted_ids, topk-weight, atomic reduce).

`kernels.rxf_moe` mirrors `w4a8_moe`'s dispatch: gemm1 GEMV at decode / WMMA at prefill; gemm2
fused scatter at decode (M≤2, `MINISGL_MOE_SCATTER` gate) / gemm2 + gather-reduce at prefill.
Weight layout (op-layout, no conversion): `weight_packed` uint8 [N, K/2] (low nibble = even
channel), `weight_scale` fp16 [N, K/32], NL int8[16].

## Status

- **Parity GREEN** (`rxf_hip_parity.py` raw ops; `tools/rxf_parity.py` engine dispatch):
  rotate_quant int8 **bit-exact**; dense GEMV/WMMA/e2e cos-sim 1.00000; MoE scatter+gather 0.9999x.
- **Perf** (`tools/rxf_bench.py`, RX 9070 XT, vs the validated fp8 W4A8 kernel, µs/call; MoE has
  ~20% run-to-run variance from shared-card clock scaling):
  | shape | fp8 | RXF |
  |---|---|---|
  | dense M=1 N=K=4096 | 42 | **29** |
  | dense M=64 N=K=4096 | 707 | **177** |
  | dense M=256 N=K=4096 | 1049 | **325** |
  | dense M=64 N=11008 K=4096 | 1557 | **279** |
  | MoE M=1 (decode) | 384 | 396 |
  | MoE M=16 (small prefill) | 3561 | **1491** |
  | MoE M=128 (prefill) | 4613 | **1949** |
  Both dense and MoE now beat fp8. The `WARPS_N` split was the lever for **both**: dense
  `linear_tiled_kernel<BM,BN,BK_TILE,WARPS_N=2>` (4 M-warps × 2 N-warps, `running[4][8]`=32 regs)
  went from 1.4× slower to ~4× faster; MoE (1-warp blocks → 4 warps) ~2.4× faster. MoE M=128 sub-op
  breakdown (µs): rotate1 11, gemm1 1398, silu 17, rotate2 23, gemm2 548, gather 277. (fp8 column
  carries run-to-run clock variance on the shared card; the RXF deltas are from the kernel change.)

## Perf follow-ups (optional)

- **Dense 2-deep K-pipeline**: the fp8 `gemm_tiled_kernel` prefetches the next K-step's frags;
  RXF dropped it in the rewrite. Could add more at large M.
- Wire RXF-format checkpoint loading in the model + a real serve/PPL run (no RXF checkpoint is in
  this repo yet; parity/bench use synthetic op-layout weights).

## Build

```
GPU_ARCHS=gfx1201 python setup.py build_ext --inplace   # in the vllm22-w4a8:combined image
```

## Cross-learnings (RXF ↔ existing W4A8)

**RXF → this repo:** the Hadamard activation-outlier pre-pass and the NL codebook are the
accuracy levers the existing fp8 path lacks (fp8's 3-bit mantissa eats heavy-tailed
activations). **This repo → RXF:** native int8/fp8 WMMA matrix-core GEMMs (vs RXF's Triton int8
dot), the fused MoE scatter/gather decode kernels, and AWQ/GPTQ checkpoint compatibility.
