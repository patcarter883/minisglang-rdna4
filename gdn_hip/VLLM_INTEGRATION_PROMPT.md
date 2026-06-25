# Agent task: plug `gdn_hip` into vLLM's Qwen GDN linear-attention path

You are working in `~/code/vllm-gfx1201` (the ROCm/gfx1201 vLLM fork). Your job: **replace the
flash-linear-attention (fla) Triton kernels in the Qwen3-Next/Qwen3.5/Qwen3.6 Gated-Delta-Net path
with the native HIP `gdn_hip` ops**, so the GDN linear-attention layer **never JIT/autotunes a Triton
kernel again** (compile once, run any shape). This is the exact same motivation, on the same box,
as the existing `zaya_cca` kernel (`vllm/model_executor/layers/mamba/cca_hip/`) — which kills the
Triton/graph-break decode path for the *CCA* (DFlash) architecture. `gdn_hip` does it for **GDN**.

## What `gdn_hip` is (already built + validated)

A standalone, framework-agnostic `torch.ops` HIP extension at
`~/code/minisgl-rdna4/gdn_hip/` (sources: `gdn_kernels.hip`, `bindings.cpp`, `setup.py`, `op.py`).

- **AOT-compiled once** for gfx1201 (`GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`) → one
  `gdn_hip_C*.so`. No per-shape Triton JIT/autotune, ever.
- **Numerically validated** (`max|Δ| ~1e-7` vs a pure-torch reference of the exact fla recurrence, on
  real GDN geometry H=16 HV=32 K=V=128) — see `~/code/minisgl-rdna4/tools/gdn_hip_parity.py`.
- `import gdn_hip` (with the package dir on `sys.path`) loads the `.so` and registers
  `torch.ops.gdn_hip.*` **plus `register_fake` meta impls** so `torch.compile`/Inductor step over the
  ops without graph-breaking (same pattern as `zaya_cca`).
- The math is lifted **verbatim** from `vllm/.../fla/ops/fused_recurrent.py` — so it matches what
  vLLM's GDN already computes. The kernels just run that rank-1 recurrence natively in HIP.

## The 6 ops (schemas + semantics)

All tensors **fp32** except where noted; state tensors are mutated **in place** (`Tensor(a!)`).
GQA mapping: q/k head `h = hv // (HV // H)`. `scale = head_k_dim ** -0.5`. `use_l2norm=1` (q,k are
L2-normalized inside the kernel). `activation=1` = SiLU. NULL slot (`state_indices <= 0`) → output
zeroed, state untouched (matches the fla NULL_BLOCK_ID convention).

```
gdn_hip::gdn_decode(q[B,H,K], k[B,H,K], v[B,HV,V], a[B,HV], b[B,HV], A_log[HV], dt_bias[HV],
                    ssm_state[slots,HV,V,K](a!), state_indices[B] int64, scale, use_l2norm) -> out[B,HV,V]
   # one token/seq. g = -exp(A_log)*softplus(a+dt_bias); beta = sigmoid(b)  computed INLINE.
   # == fused_recurrent_gated_delta_rule_packed_decode (but takes split q,k,v, not packed mixed_qkv).

gdn_hip::gdn_prefill_wmma(q[T,H,K], k[T,H,K], v[T,HV,V], a[T,HV], b[T,HV], A_log[HV], dt_bias[HV],
                     cu_seqlens[N+1] int32, state_indices[N] int64, has_initial_state[N] uint8,
                     ssm_state(a!), scale, use_l2norm) -> out[T,HV,V]
   # *** DEFAULT PREFILL *** matrix-core (WMMA) chunked gated-delta-rule. Same schema/semantics as
   # gdn_prefill (below); 5-6.7x FASTER than the recurrent form at T=256..16384. fp16 matmul
   # operands => max|Δ|~1e-3 vs the recurrent oracle (well within greedy-token tolerance).

gdn_hip::gdn_prefill(q[T,H,K], k[T,H,K], v[T,HV,V], a[T,HV], b[T,HV], A_log[HV], dt_bias[HV],
                     cu_seqlens[N+1] int32, state_indices[N] int64, has_initial_state[N] uint8,
                     ssm_state(a!), scale, use_l2norm) -> out[T,HV,V]
   # RECURRENT fallback/oracle: varlen recurrent gated-delta-rule == chunk_gated_delta_rule
   # (numerically; per-token fp32 reference form). Exact but slow. reads initial state from slot iff
   # has_initial_state, writes final state back to slot. (gdn_prefill_chunked, a scalar chunked op,
   # exists too but is parity-oracle only — ~4x slower than recurrent; that's why WMMA was written.)

gdn_hip::causal_conv1d_update(x[B,C], weight[C,W], bias?[C], conv_state[slots,C,W-1](a!),
                              state_indices[B] int64, activation) -> out[B,C]
gdn_hip::causal_conv1d_fwd(x[T,C], weight[C,W], bias?, cu_seqlens[N+1] int32, state_indices[N] int64,
                           has_initial_state[N] uint8, conv_state(a!), activation) -> out[T,C]
   # depthwise causal conv (+SiLU). x is TOKEN-major [.,C] (NOT the Triton path's transposed [C,.]).
   # weight = conv1d_weight.view(conv_dim, kernel). conv_dim = 2*key_dim+value_dim, order [q|k|v].

gdn_hip::rmsnorm_gated(x[M,D], z[M,D], weight[D], eps) -> out[M,D]
   # out = x * rsqrt(mean(x^2)+eps) * weight * silu(z)  == RMSNormGated(norm_before_gate=True).
```

## What to change in vLLM

Target: `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (it imports
`chunk_gated_delta_rule`, `fused_recurrent_gated_delta_rule_packed_decode`, `l2norm_fwd` from
`fla.ops`, `causal_conv1d_*` from `mamba.ops`, and `RMSNormGated`). Replace the GDN compute with
`torch.ops.gdn_hip.*`, mirroring the reference integration already done in
`~/code/minisgl-rdna4/python/minisgl/gdn/layer.py` (read it — it's the same architecture and the
cleanest worked example: all-HIP compute, the WMMA-prefill default + recurrent fallback switch, and
the plain `GatedRMSNormWeight` holder; no Triton import at all):

1. **Conv** → `gdn_hip.causal_conv1d_fwd` (prefill, varlen) / `causal_conv1d_update` (decode). Pass
   the conv input **token-major + contiguous** (`mixed_qkv.float().contiguous()`), `bias=None`,
   `activation=1`. The conv kernel already does SiLU.
2. **Split** the conv output `[., conv_dim]` into `q,k [., H, head_k_dim]`, `v [., HV, head_v_dim]`
   (it's `[q|k|v]` along the channel dim). **Skip `fused_post_conv_prep` and `l2norm_fwd`** — the
   recurrent kernel folds L2-norm and the `g/beta` computation in; it takes raw `a,b`.
3. **SSM** → `gdn_hip.gdn_prefill_wmma` (varlen prefill, the DEFAULT) / `gdn_hip.gdn_decode` (T=1).
   Pass raw `a,b` (per token per v-head), `A_log,dt_bias`, the fp32 `ssm_state`,
   `state_indices.long()`, `scale=head_k_dim**-0.5`, `use_l2norm=1`. The op updates `ssm_state` in
   place (drop the manual `ssm_state[idx]=last_state`). Keep `gdn_hip.gdn_prefill` (recurrent) wired
   behind an env flag as the exact fallback — see `layer.py`'s `GDN_HIP_WMMA_PREFILL=0` switch.
4. **Output norm** → `gdn_hip.rmsnorm_gated(core_flat, z_flat, norm_weight, eps)` then `out_proj`.
   The norm is a **pure weight holder** — its forward is never called. Replace vLLM's `RMSNormGated`
   with a plain weight param (as `layer.py` does with `GatedRMSNormWeight`: just `.weight` (init ones)
   + `.eps`), preserving the `…norm.weight` state_dict key. This also drops the Triton dependency that
   importing `RMSNormGated` would otherwise drag in.
5. **Make the GDN recurrent state (conv_state + ssm_state) fp32** wherever vLLM allocates it (the
   kernels read/write it in place in fp32; also more accurate than per-step bf16). Cast the bf16
   activations (`mixed_qkv, q,k,v, a,b, weight, z`) to `.float()` at the op boundary; cast the final
   norm output back to the layer dtype before `out_proj`. (`A_log/dt_bias` are already fp32.)

## Build + make importable

`gdn_hip` is a shared package — choose one (mirror `zaya_cca`, which lives in the vllm tree):
- copy `~/code/minisgl-rdna4/gdn_hip/` into `vllm/model_executor/layers/mamba/gdn_hip/` and build
  in place, OR `pip install -e ~/code/minisgl-rdna4/gdn_hip` (after adding a minimal `pyproject`), OR
  add its dir to `sys.path`. Build in the combined ROCm image:
  `cd <gdn_hip dir> && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`. Import it once at layer
  import time so the ops + fakes register (`import gdn_hip`).

## Validate (do not skip)

1. **Unit parity**: port `~/code/minisgl-rdna4/tools/gdn_hip_parity.py` (it's framework-agnostic) and
   confirm `ALL PASS` on a leased gfx1201 card — it covers decode, both prefill kernels (recurrent +
   WMMA), conv, and rmsnorm_gated against the pure-torch fla recurrence oracle. Book GPUs ONLY via
   `~/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 -- <cmd>` (the shared arbiter — never hand-set
   devices). For a real-model sanity check, `tools/gdn_wmma_serve_smoke.py` greedy-diffs the WMMA vs
   recurrent prefill on the loaded 4B (they agree token-for-token on most prompts).
2. **End-to-end token-diff**: serve a real GDN model (`Qwen/Qwen3.5-4B`, or the 35B) with the HIP
   path and greedy-diff its output against the **Triton** path on the same prompts. They are fp32
   vs fp32 so they should agree to floating-point precision; a divergence means a mapping bug
   (most likely: the conv `[q|k|v]` split order, the GQA head map, or a stride/contiguity issue).

## Caveats (be honest about these)

- **Decode is a pure win. Prefill is now the WMMA chunked kernel** (`gdn_prefill_wmma`, the default) —
  matrix-core, intra-chunk parallel, validated against the recurrent kernel as oracle (`max|Δ|~1e-3`,
  greedy-token-identical on the real 4B serve). It is **5-6.7x FASTER than the recurrent form** at
  T=256..16384, so prefill no longer trades long-context throughput for zero Triton compiles — you get
  both. The recurrent `gdn_prefill` stays as the exact fp32 fallback (`GDN_HIP_WMMA_PREFILL=0`) in
  case a real-decay regime stresses the fp16 absorption. (This is what the "chunked-HIP gdn_prefill
  planned next" note in earlier drafts referred to — it's done.)
- **v1 is fp32 at the op boundary** (the kernels are fp32, except WMMA's fp16 matmul operands).
  bf16-native state/inputs is a follow-up
  (templatize the kernels on dtype). The per-call `.float()` casts are cheap but not free.
- The recurrent kernel holds one state row (`head_k_dim` floats) per thread in registers; it assumes
  `head_k_dim <= GDN_MAX_K` (128, the Qwen value). Bump the `#define` for larger head dims.

Deliverable: a PR that routes vLLM's Qwen GDN through `torch.ops.gdn_hip.*`, with the parity test
passing and a token-diff-vs-Triton result in the description.
