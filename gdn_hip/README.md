# gdn_hip — native HIP kernels for Gated Delta Net (gfx1201)

A **standalone, framework-agnostic** `torch.ops` extension that replaces the flash-linear-attention
**Triton** GDN kernels with **AOT-compiled HIP** — so the Qwen3-Next / Qwen3.5 / Qwen3.6 GDN
linear-attention path **never JIT/autotunes a Triton kernel again** (compile once, run any shape).
Shared by `minisgl-rdna4` and `vllm-gfx1201` (both route GDN through the same fla recurrence): build
one `.so`, call `torch.ops.gdn_hip.*` from either.

## Ops (replace ~15 Triton kernels)
| op | replaces | notes |
|---|---|---|
| `gdn_decode` | fused_sigmoid_gating decode SSM | 1 token/seq; g,β computed inline; state updated in place |
| `gdn_prefill` | `chunk_gated_delta_rule` | recurrent fp32 reference oracle; robust, slow |
| `gdn_prefill_wmma` | `chunk_gated_delta_rule` | **matrix-core chunked — the serve prefill path; 4.8-5.9× faster than recurrent** |
| `causal_conv1d_update` | mamba conv decode | depthwise causal conv + state roll + SiLU |
| `causal_conv1d_fwd` | mamba conv prefill | varlen depthwise causal conv + state write |
| `rmsnorm_gated` | RMSNormGated | norm-before-gate, SiLU(z) |

Math is lifted verbatim from `fla/ops/fused_recurrent.py` (the recurrence) — the chunked Triton path
is just a throughput optimization of this exact rank-1 update:
`S*=exp(g); v-=S@k; v*=β; S+=outer(v,k); o=S@q` (q,k l2-normed, q scaled 1/√K).

## Build (AOT — no GPU needed to compile)
```bash
# in the combined ROCm image (hipcc + PYTORCH_ROCM_ARCH=gfx1201)
cd gdn_hip && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
```
Produces `gdn_hip_C*.so`; `import gdn_hip` loads it and registers `torch.ops.gdn_hip.*` (+ fake/meta
impls for torch.compile safety).

## Status
- [x] **Builds AOT for gfx1201** (hipify clean, links torch_hip) — CPU-only build verified.
- [x] **All 6 ops load + register** with valid schemas (CPU).
- [x] **Numeric parity** vs torch reference (`tools/gdn_hip_parity.py`): ALL PASS, max|Δ|~1e-7.
- [x] **Wired into `gdn/layer.py`** (recurrent path) — **serves 4B TP1/TP2 + 35B TP2 coherent** on
      2× gfx1201 (2026-06-24). The Triton GDN compile cliff is gone.
- [x] **Chunked prefill (`gdn_prefill_chunked`)** — numerically correct vs recurrent under MILD
      decay (max|Δ|~1e-7), but **~4× SLOWER** as a scalar per-row kernel, and it NaNs under strong
      decay (`gam[j]/gam[i]`=0/0 in fp32). Kept only as a mild-decay parity oracle.
- [x] **WMMA chunked prefill (`gdn_prefill_wmma`)** — the long-context speedup. Matrix-core
      (rocWMMA) reformulation of the intra-chunk Grams/state-reads/carry; **4.8–5.9× FASTER than
      recurrent** at T=256..16384 (`tools/gdn_hip_bench.py`). Decay is applied as bounded log-space
      fp32 scalings (NOT k/γ absorption, which overflows fp16 → NaN), so it is robust to any decay —
      strictly more so than the scalar chunked op. Validated vs the recurrent oracle on short+long
      varlen seqs × strong/mild decay (`tools/gdn_hip_parity.py`), and coherent on the real 4B GDN
      serve (`tools/gdn_wmma_serve_smoke.py`: 5/6 prompts token-identical to recurrent). **Now the
      default serve prefill** (`gdn/layer.py`; `GDN_HIP_WMMA_PREFILL=0` reverts to recurrent).
- [ ] Delete the (now unused) Triton GDN tree (`gdn/{fla,mamba}`); bf16-native state (v1 = fp32).
