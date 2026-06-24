# Continuation prompt — finish the WMMA chunked GDN prefill (gdn_hip #22)

> **STATUS: DONE (2026-06-24).** `gdn_hip::gdn_prefill_wmma` is built, validated, benchmarked, and is
> now the DEFAULT serve prefill (`gdn/layer.py`; `GDN_HIP_WMMA_PREFILL=0` reverts to recurrent).
> Steps 1–4 below all complete. Key result: **4.8–5.9× faster than recurrent** (T=256..16384), parity
> vs the recurrent oracle ≤1.6e-3 across strong+mild decay × short+long seqs, real 4B serve coherent
> (5/6 prompts token-identical). **Critical lesson:** the textbook decay absorption `k̃=k/γ` OVERFLOWS
> fp16 (γ underflows over a 16-tok chunk) → NaN in the real serve though it passes mild-random op
> parity; the shipped kernel keeps raw operands + bounded log-space decay scalings instead. The scalar
> `gdn_prefill_chunked` is now only a mild-decay oracle (it also NaNs under strong decay). The rest of
> this brief is the original plan, kept for context.

You are in `/home/pat/code/minisgl-rdna4` (branch `rdna4`). Read `CLAUDE.md` first — the GPU lease
protocol is MANDATORY. Continue task #22: a native-HIP **WMMA (matrix-core) chunked gated-delta-rule
prefill** for gfx1201 — the genuine long-context throughput win for the GDN linear-attention path.

## Context — what already works (don't break it)

- **`gdn_hip/`** is a standalone `torch.ops.gdn_hip.*` HIP extension (AOT-compiled, no Triton) that
  replaced the fla-Triton GDN kernels. It **serves Qwen3.6-35B-A3B at TP=2** on 2× gfx1201
  (recurrent prefill + decode + causal_conv1d + rmsnorm_gated). The serve path uses the **recurrent**
  `gdn_prefill` and must keep working — do NOT wire anything unvalidated into `gdn/layer.py`.
- **Two oracles** for the chunked kernel, both validated and in `tools/gdn_hip_parity.py`:
  `gdn_prefill` (recurrent, the reference) and `gdn_prefill_chunked` (a SCALAR chunked kernel —
  numerically correct, max|Δ|~1e-7, but **~4× slower**; `tools/gdn_hip_bench.py` shows 0.23–0.29×).
  The scalar version is why #22 exists: a chunked reformulation only pays off with matrix-core matmuls.
- **WMMA foundation DONE + validated** (`gdn_hip/wmma_probe/`, commits `017c9f4`, `fca8b2e`):
  rocWMMA GEMM toolkit `gemm` (A@B / NN), `gemm_nt` (A@Bᵀ), `gemm_tn` (Aᵀ@B), fp16-in/fp32-acc,
  16×16×16 tiles, validated vs torch at the chunked shapes (16×128×16, 16×128×128), rel ~1e-7.

## The math (decay absorption → pure matmuls)

GDN geometry: `H=16` k-heads, `HV=32` v-heads, `head_k_dim=head_v_dim=128`. State `S` is `[Dv×Dk]`
per (seq, v-head). Recurrence per token (validated in the recurrent kernel):
`S*=exp(g); v-=S@k; v*=β; S+=outer(v,k); o=S@q` with q,k l2-normed, q scaled 1/√Dk,
`g = -exp(A_log)·softplus(a+dt_bias)`, `β = sigmoid(b)`.

Chunk of `C` tokens (use **C=16**, one WMMA tile dim), carried state `S0`. Let `γ_t = ∏_{i≤t}exp(g_i)`
(cumulative decay within chunk). Absorb decay into scaled copies:
`k̃_j = k_j/γ_j`, `k̄_j = γ_j k_j`, `q̃_t = γ_t q_t`, `k̆_j = (γ_C/γ_j) k_j`. Then everything is matmuls:

```
M   = K̄ @ K̃ᵀ                 # [C×C]   (gemm_nt)   M[j][i] = (γ_j/γ_i)(k_j·k_i)
Ã   = Q̃ @ K̃ᵀ                 # [C×C]   (gemm_nt)   Ã[t][j] = (γ_t/γ_j)(q_t·k_j)
KS  = K̄ @ S0ᵀ                # [C×Dv]  (gemm_nt)
QS  = Q̃ @ S0ᵀ                # [C×Dv]  (gemm_nt)
B   = β ⊙ (V − KS)            # [C×Dv]  (elementwise)
U   = (I + tril(β⊙M, −1))⁻¹ B # [C×Dv]  ← THE TRIANGULAR SOLVE (only non-matmul piece)
O   = QS + tril(Ã, 0) @ U     # [C×Dv]  (gemm, with causal mask incl. diagonal)
S_C = γ_C·S0 + Uᵀ @ K̆        # [Dv×Dk] (gemm_tn)
```

(Verify these against the scalar `gdn_prefill_chunked` in `gdn_kernels.hip`, which implements the
same recurrence as explicit per-row loops — that is your ground truth for the formulas.)

## Remaining steps (each validatable on a card)

1. **Triangular-solve kernel** — `U = (I + L)⁻¹ B`, `L = tril(β⊙M, −1)` unit-lower-triangular `C×C`,
   `B` is `C×Dv`. For C=16 a blocked forward-substitution (or compute `T=(I+L)⁻¹` then `T@B`) is fine.
   Build it standalone, validate vs `torch.linalg.solve_triangular(I+L, B, upper=False, unitriangular=True)`.
2. **Assemble** the chunked-WMMA kernel: one block per (seq, v-head), chunks sequential carrying `S`
   in LDS; per chunk run NT matmuls → build B → solve → NN masked matmul → TN carry. Reuse the
   `wmma_probe` GEMM device code (lift the kernels into `gdn_hip/gdn_kernels.hip` as device helpers,
   or keep a block-level WMMA GEMM). Inputs are fp32 at the boundary — cast the matmul operands to
   fp16 for WMMA (fp32 accumulate; matches fla precision). Expose as `gdn_hip::gdn_prefill_wmma`.
3. **Validate** vs both oracles (`tools/gdn_hip_parity.py`, add a `check_prefill_wmma`): long varlen
   multi-chunk sequences, mild decay (A_log~−2 to avoid γ underflow), tol ~5e-3.
4. **Benchmark** vs recurrent (`tools/gdn_hip_bench.py`, add the wmma op): T=256..16384. Only wire
   `gdn_prefill_wmma` into `gdn/layer.py:forward_prefill` (swap from `gdn_prefill`) **if it's faster**.
   The scalar one was 4× slower and was reverted — same bar here.

## Build / run mechanics (proven)

- Build (CPU, no GPU needed): in the combined image,
  `cd gdn_hip && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace`. For WMMA you MUST
  `#undef __HIP_NO_HALF_CONVERSIONS__` / `__HIP_NO_HALF_OPERATORS__` at the top of the .hip (the
  torch cpp_extension passes `-D…=1`, which breaks `float16_t`), and add `/opt/rocm-7.2.1/include`
  to `include_dirs` (rocwmma headers). Reference: `gdn_hip/wmma_probe/{wmma_gemm.hip,setup.py}` and
  the migrated W4A8 kernel `~/code/vllm-gfx1201/w4a8_fp8_wmma/wmma_peak_fp16.hip` (rocwmma for fp16;
  `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` for the fp8 path if you ever go fp8).
- GPU runs: ONLY via `/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 -- <cmd>` (one card is
  enough for parity/bench). Inside the lease, `docker run … vllm22-w4a8:combined` with the full ROCm
  passthrough and `-e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES`,
  `-v "$PWD":/engine -e PYTHONPATH=/engine`. Copy the exact recipe from the lease commands in the
  session (e.g. the gdn_hip_parity / wmma_probe runs). Logs land in `tools/tp2_results/` (gitignored).
- Parity/bench tests already import `gdn_hip` from `/engine` and call `torch.ops.gdn_hip.*`.

## Also open (lower priority, after #22 or independently)

- **#20 cleanup**: delete the now-unused Triton GDN tree `python/minisgl/gdn/{fla,mamba}` and replace
  the `RMSNormGated` import (used only as a norm-weight container) with a plain weight holder that
  keeps the `linear_attn.norm.weight` state_dict key; re-run `tools/qwen3_5_*build_smoke.py` +
  `qwen3_5_tp_weight_map_test.py` + a 4B serve.
- **35B-vs-vLLM token parity**: deferred oracle check for the 35B TP=2 serve.
- **`gdn_hip/VLLM_INTEGRATION_PROMPT.md`**: hand-off to plug gdn_hip into vllm-gfx1201's GDN path.

Commit each validated step with the `Co-Authored-By: Claude Opus 4.8 (1M context)` trailer. Keep the
recurrent serve path intact until the WMMA kernel is benchmarked-faster AND parity-clean.
