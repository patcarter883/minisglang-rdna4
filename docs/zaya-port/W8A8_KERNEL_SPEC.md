# W8A8-fp8 grouped-MoE kernel — implementation spec (full parity with W4A8)

Goal: a native **W8A8-fp8** grouped-MoE HIP kernel for gfx1201, replacing ZAYA's current
fp8→bf16-dequant→Triton MoE path, at **full feature parity** with the W4A8 kernel
(`w4a8_fp8_wmma`): WMMA prefill, GEMV decode, fused gemm1+silu, scatter (gemm2+topk+reduce),
gather_reduce. W8A8 = fp8 (e4m3) weights with a **per-output-channel** fp32 scale + fp8 activations.

Source of truth (READ THESE): the W4A8 package `/home/pat/code/vllm-gfx1201/w4a8_fp8_wmma/`
(`moe_kernel.hip`, `moe_gemm_tiled.h`, `tile_config.h`, `kernel_names.h`, `bindings.cpp`,
`__init__.py`, `setup.py`, `moe_experts.py`). Vendoring template: `minisgl-rdna4/rxf_hip/` and
`cca_hip/`. Integration: `python/minisgl/quant/kernels.py` (`w4a8_moe`), `python/minisgl/layers/moe.py`
(`_GroupedFP8Experts`, `MoELayer.forward`), `python/minisgl/models/zaya.py` (`ZayaMoEBlock`),
`python/minisgl/models/weight.py` (`_store_expert`).

## The W8A8 ≠ W4A8 delta (everything else is copied verbatim)

| Concern | W4A8 location | W8A8 change |
|---|---|---|
| B HBM layout | `w_packed (E,N,K/8) int32` | `w_fp8 (E,N,K) uint8` (e4m3), plain row-major, 1 byte/weight |
| Scale layout | `scales (E,N,K/group) fp16` | `w_scale (E,N) f32` per-output-channel |
| Zeros | `w_zeros (E,N/8,K/group)` | **delete** (fp8 float-quant is symmetric) |
| `weight_is_e2m1`/mxfp4 | present | **delete** |
| B stage+unpack | `moe_gemm_tiled.h:104-122` & `:261-283`; helper `tile_config.h:69-71` (`decode_w4_to_e4m3`) | replace nibble loop with byte copy: `B_tile[n*LDSBK+k] = (an<N)? wq_e[(long)an*K + k0 + k] : 0;` (wq_e = w_fp8 + e*N*K). Vectorize as uint/uint2 (K%16==0). Drop `ppr`,`PACK_FACTOR`,`wz_e`,`decode_w4_to_e4m3`. |
| Weight-scale fold (per-group, in K-loop) | `moe_gemm_tiled.h:144-151` | **delete**; accumulate raw acc: `running[f][ee] += acc[f][ee];` |
| Epilogue scale | `:163-180` (act scale only) | add per-N weight scale once: `out[...] = __float2half(running[f][ee] * asc * w_scale_e[abs_n]);` (w_scale_e = w_scale + e*N; hoist wsc_n out of the ee loop). Scatter: `atomicAdd(..., w * running * asc * wsc_n)`. |
| Activation fp8 quant | `moe_kernel.hip:85-127` `moe_compute_act_fp8_kernel` (token-wise dynamic e4m3, E4M3_MAX=448) | **UNCHANGED** — keep verbatim |
| WMMA intrinsic | `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` (`tile_config.h:93`) | **UNCHANGED** |
| WMMA K-loop / A-shuffle / accumulators | `moe_gemm_tiled.h` v6 | **UNCHANGED** (B is fp8 in LDS by the time the loop runs) |
| Fused silu (gate|up two-slab) | `moe_gemm1_silu_v6_kernel` `moe_kernel.hip:716-870` | same two-slab; drop per-group fold; epilogue `moe_silu_and_mul_h(run_g*asc*wsc_g_n, run_u*asc*wsc_u_n)`, wsc_g_n=w_scale_e[abs_n], wsc_u_n=w_scale_e[inter+abs_n] |
| GEMV decode | `moe_gemv_v7_kernel` `moe_kernel.hip:884-1062` | read fp8 weights directly from HBM (`v4i_t`), no `decode_w4_to_f32`/zp; per-N scale once in epilogue |
| gather_reduce | `moe_gather_reduce_kernel` `moe_kernel.hip:1241-1258` | **verbatim reuse** (no weights/scales) |
| Tiling/autotune | `tile_config.h:116` TileConfig, `make_moe_tile_config`, `autotune_grouped.hip` | LDS budget math unchanged (B-LDS still 1 byte/weight); drop `group_size` from the key |

Served default kernel to port = `moe_gemm_tiled_v6_kernel` (A-shuffle / B-only-LDS, SCATTER templated).
The 2:4-sparse SWMMAC path (`moe_gemm_swmmac_int4.h`) is OUT OF SCOPE.

## Package to create: `/home/pat/code/minisgl-rdna4/w8a8_fp8_wmma/` (flat, mirror rxf_hip)
```
setup.py          # CUDAExtension name="w8a8_fp8_wmma_C", sources=[bindings.cpp, <kernel>.hip],
                  #   include_dirs=["/opt/rocm-7.2.1/include"] if rocwmma; GPU_ARCHS env (gfx1201)
bindings.cpp      # TORCH_LIBRARY(w8a8_fp8_wmma) + TORCH_LIBRARY_IMPL(CUDA); NO PYBIND11_MODULE.
<kernel>.hip      # copied+edited from moe_kernel.hip (+inline the needed .h, or copy headers too)
op.py             # torch.ops.load_library(glob *_C*.so) + @torch.library.register_fake + aliases
__init__.py       # from .op import (...)
.gitignore        # build/  *.so  *_hip.cpp  *_hip.hip  __pycache__/
```
Loaded via `torch.ops.load_library`, NOT `import _C`. AOT build is CPU-only (NO gpu-lease):
```
docker run --rm -v "$PWD":/engine --entrypoint bash vllm22-w4a8:combined -lc \
 'source /app/.venv/bin/activate && cd /engine/w8a8_fp8_wmma && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace'
```

### torch.ops surface (drop w_zeros, weight_is_e2m1; scale per-channel)
```
mmq_w8a8_moe_gemm(Tensor x, Tensor w_fp8, Tensor scales, Tensor sorted_token_ids,
   Tensor expert_ids, Tensor num_tokens_post_padded, int top_k, int block_m, int kernel) -> Tensor
mmq_w8a8_moe_gemm1_silu(... same args ...) -> Tensor                  # returns (P, inter)
mmq_w8a8_moe_gemm_scatter(..., Tensor topk_weights, Tensor(a!) output, int top_k, int block_m, int kernel) -> ()
mmq_w8a8_moe_gather_reduce(Tensor out2, Tensor sorted_token_ids, Tensor topk_weights,
   Tensor num_tokens_post_padded, int top_k) -> Tensor               # verbatim from w4a8
```
Every `m.def` MUST carry `{at::Tag::pt2_compliant_tag}` + a `register_fake` (else cudagraph→eager).
x is fp16 (M,K)/(T,K); w_fp8 (E,N,K) e4m3; scales (E,N) f32; routing int32; scatter output fp32 (M,N) pre-zeroed.
kernel ids: 6=wmma (prefill/gemm2), 7=gemv (decode gemm1). fp16 I/O is the staging boundary; compute e4m3.

## minisgl integration
1. `python/minisgl/quant/kernels.py`: add `w8a8_moe(x, w13, w13_scales, w2, w2_scales, gating_output,
   top_k, renormalize, *, topk_weights=None, topk_ids=None, kernel="wmma", block_m=16)` mirroring
   `w4a8_moe` body verbatim (kernels.py:170-311): per-GEMM kernel pick (gemm1 gemv if M<=2 else wmma;
   gemm2 wmma), precomputed-route branch, `moe_align`, gemm1→`tail_hip.silu_and_mul`→ decode scatter
   (M<=2, MINISGL_MOE_SCATTER) OR prefill gather_reduce. Drop zeros/group args. Reuse all env gates.
2. `python/minisgl/layers/moe.py`:
   - Add `_GroupedFP8Experts.post_load(self)`: build `self._w_op` (E,N,K e4m3 contiguous — op layout is
     the natural layout, likely just `.contiguous()`; if kernel needs a transpose, mirror per-expert+stack),
     `self._scales_op` (E,N f32 = `weight_scale.squeeze(-1).contiguous().float()`); `del self.weight,
     self.weight_scale`. Underscore-prefixed so BaseOP state walk skips them.
   - Swap the `if self.fp8_experts:` forward branch (moe.py:234-252) from `dequant()`+`fused_experts_impl`
     to `kernels.w8a8_moe(hidden_states, w13._w_op, w13._scales_op, w2._w_op, w2._scales_op, None,
     self.top_k, self.renormalize, topk_weights=topk_weights, topk_ids=topk_ids)`.
   - post_load dispatch already works via BaseOP recursion (engine.py:108 → ZayaMoEBlock.post_load →
     experts(MoELayer) default recurse → _GroupedFP8Experts.post_load). No engine/model change needed.
3. Constraints the kernel must honor (ZAYA): top_k==1, renormalize==False, activation silu, gate|up merged
   (gate=first half), apply_router_weight_on_input==False, precomputed route always (topk_ids/weights given),
   MOD handled model-side (kernel sees clamped legal expert ids; discarded rows are fine), TP=1.

## Validation gates
- Build clean (AOT, gfx1201).
- **Parity**: `w8a8_moe` output vs the fp8→bf16 `dequant()`+`fused_experts_impl` reference on random
  experts/routing — rel-err within fp8 tolerance (the dequant ref is itself bf16, so compare to an fp32
  torch reference of `(x_fp8·w_fp8)*a_scale*w_scale` too). Cover prefill (M large, wmma) AND decode (M<=2,
  gemv+scatter), top_k=1.
- **Coherence**: ZAYA1-8B-fp8 eager + graph (`tools/zaya_graph_smoke.py`) still coherent with the new path.
- **Perf**: decode TPOT A/B — w8a8 kernel vs the old dequant→Triton path (expect faster: no bf16 dequant
  spike, WMMA compute). Report numbers.
