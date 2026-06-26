# tail_hip — native HIP elementwise "tail" kernels for gfx1201

The last small kernels of the Triton-free serve path: RMSNorm (+ fused residual-add), gated
SiLU-mul, and NeoX partial RoPE. Companion to `gdn_hip`, `attn_hip` (prefill), `attn_decode`.

## What this is (and isn't)
These ops are **already Triton-free** — minisgl does them in plain torch (`layers/norm.py`,
`activation.py`, `rotary.py`); vLLM does them as C++/HIP csrc custom ops. So this is **not** about
removing Triton. It's about:
1. **Completing the native-HIP set** so the whole decode/prefill path is one consistent toolchain
   (Atlas hand-wrote exactly these — `rms_norm`, `silu_mul`, `rope`).
2. **Cutting host dispatch + intermediate allocs.** The torch impls are several launches + temp
   tensors each (e.g. `_rms_norm` = pow/mean/rsqrt/mul; `silu_and_mul` = silu + mul + cat view).
   On the launch-sensitive decode path (rank0 dispatch overhead is a measured wall) a single fused
   HIP kernel per op removes that overhead. Each is one launch, fp32-internal, bf16-out, no temps.

## Ops (torch.ops.tail_hip.*)
- `rms_norm(x, w, eps, plus_one)` — fp32 RMSNorm; `plus_one` → gain `(w+1)` (Qwen3.5/3.6/Gemma).
- `rms_norm_add(x, residual, w, eps, plus_one)` — fused: `residual <- x+residual` (in place),
  returns `rmsnorm(residual)`. The pre-norm residual pattern in one launch.
- `silu_and_mul(x)` — gated: `silu(x[...,:D]) * x[...,D:]`, `x:[...,2D] -> [...,D]`.
- `rope(x, pos, cache, head_size, rotary_dim)` — NeoX rotate-half, partial rotary; `cache` is
  minisgl's `cat(cos[rd/2], sin[rd/2])` `[max_pos, rd]` (fp32). Tail dims `[rd:head_size]` pass through.

Conventions are matched 1:1 to minisgl-rdna4 so the torch refs are exact parity oracles.

## Status
Written, **GPU-validated** via `tail_hip_parity.py` (vs the minisgl torch refs, bf16-rounded).
Pure FMA/reduction — no WMMA, no Triton. AOT build: `GPU_ARCHS=gfx1201 python setup.py build_ext
--inplace`.

## Wiring (later, together with attn/gdn)
Swap the torch calls in minisgl `layers/{norm,activation,rotary}.py` for `torch.ops.tail_hip.*`
behind a flag; for vLLM, these already map to csrc ops so wiring is optional (perf only). The win is
launch-count reduction on the decode step, measured against the torch baseline.
