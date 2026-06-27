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

- `rotate_quant_int8(x, span) -> (q_int8, scale_fp32)` — fused block-diagonal FWHT-span +
  per-token symmetric int8 quant. One block/row; each thread owns whole 32-groups in registers
  (no intra-group sync), staged in smem for the row absmax + quantize. span=32 only.
- `linear(q, a_scale, w_packed, w_scale, nl, bias?) -> bf16` — int8·NL-int4 GEMM. **WMMA** path
  (M>2): 16×16 tile/warp, int32 partial per 32-group scaled by the group's fp16 weight scale
  into an fp32 smem accumulator (store-to-smem epilogue, attn_hip-style → no WMMA fragment-layout
  assumption escapes). **GEMV** path (M≤2): scalar int8 dot.
- `moe_gemm(...)` — grouped per-expert int8 GEMM (scalar v0), mirrors the fp8 MoE dispatch
  (`sorted_token_ids`/`expert_ids`/`num_tokens_post_padded`).

Weight layout (op-layout, no conversion): `weight_packed` uint8 [N, K/2] (low nibble = even
channel), `weight_scale` fp16 [N, K/32], NL codebook int8[16] (`NL_DEFAULT`, model-wide).

## Status

- **Parity GREEN** (`rxf_hip_parity.py` raw ops; `tools/rxf_parity.py` engine dispatch):
  rotate_quant int8 **bit-exact** + scale exact; dense GEMV/WMMA/e2e cos-sim 1.00000;
  grouped MoE 0.9999x. (rel-err ~0.0016 is bf16 output rounding vs the fp32 reference.)
- **Numerics matched** to the Triton reference: FWHT butterfly + norm `0.1767766953` (=1/√32),
  per-token `scale=absmax/127`, `q=round(x·127/absmax)` clamped [-127,127], low-nibble=even.

## Perf follow-ups (correctness-first v0, like the fp8 path's v0)

- MoE GEMM is scalar v0 — port the int8 WMMA tiling from `linear` into `moe_gemm`.
- Dense WMMA tile is a single 16×16 warp tile — widen to multi-warp / larger BN for occupancy.
- Gather-reduce is torch `index_add_` in `kernels.rxf_moe` — a fused HIP scatter (cf.
  `mmq_fp8_moe_gemm_scatter`) is the decode-path win.
- Wire RXF-format checkpoint loading in the model + a real serve/PPL validation (no RXF
  checkpoint is in this repo yet; parity uses synthetic op-layout weights).

## Build

```
GPU_ARCHS=gfx1201 python setup.py build_ext --inplace   # in the vllm22-w4a8:combined image
```

## Cross-learnings (RXF ↔ existing W4A8)

**RXF → this repo:** the Hadamard activation-outlier pre-pass and the NL codebook are the
accuracy levers the existing fp8 path lacks (fp8's 3-bit mantissa eats heavy-tailed
activations). **This repo → RXF:** native int8/fp8 WMMA matrix-core GEMMs (vs RXF's Triton int8
dot), the fused MoE scatter/gather decode kernels, and AWQ/GPTQ checkpoint compatibility.
