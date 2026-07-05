# Phase C — Fused single-forward TiDAR (the true ~2× lever)

## Why (the A/B that motivates this)

Measured 2026-07-03 in minisgl on one card, original reasoning ZAYA1-8B (AR) vs the TiDAR diffusion
model (two-forward spec):

| serve | tok/s | vs AR |
|---|---|---|
| original ZAYA (AR) | 25.2 | 1.0× |
| TiDAR diffusion, **two-forward** spec | 16.9 | **0.67× (slower)** |

The two-forward path is *structurally* losing: each step = **2 forwards** (block_predict + verify)
emitting ~1.74 tokens (accept ≈ 0.18). Per-step wall ≈ 2 × 51 ms ≈ 103 ms → 16.9 tok/s. It only
breaks even once emitted/step > ~2.6 — a hard training ask.

**Fused collapses the 2 forwards into 1.** At *today's* acceptance: 1.74 tok / **1** forward (~51 ms)
≈ **34 tok/s ≈ 1.35× AR**, no training. With higher acceptance it goes to ~2×. Fused is the
structural fix; training-for-acceptance (see `TRAINING_PLAN_ACCEPTANCE.md`) then multiplies it.

## The fused algorithm (canonical reference)

From `vllm-gfx1201-tidar-fused/zaya/tidar/single_forward_ours.py` (the P1 linchpin, measured on real
weights). ONE forward per step over a flat sequence:

```
[ committed | S = prev_block_drafts (B) | R_0 | R_1 | ... | R_{B-1} ]
```
- **S rows** (the previous block's B drafts) → the AR next-token predictions at those positions →
  `beta_verify(S_drafts, p_ar, β=1)` gives the accepted count `k` + bonus. (verify half)
- **R_r** = a replica mask-block of B masks, pre-drafting the NEXT block *conditioned on
  `committed + drafts[:r]`*. After verify yields `k`, pick `R_k` as the next block's drafts.
  (draft half) — we pre-draft ALL B possible continuations because `k` isn't known until verify, so
  it stays ONE forward. B replicas × B masks = B² mask tokens.

Emits `k+1` tokens AND pre-drafts the next block in ONE forward → speedup ≈ `avg_accept + 1`
(vs `(avg_accept+1)/2` for two-forward). Bootstrap: one `block_predict` for the first block.

### The custom attention mask (`tidar_mask.square_additive_bias`)
`[kv,kv]` additive bias over `[committed | S | R_0..R_{B-1}]`:
- committed: causal.
- S[i] (verify row i): attends `committed + S[:i+1]` (causal within S — AR verification).
- R_r[m]: attends `committed + first r drafts of S + its own R_r block` (block-bidir within R_r).
- everything else: −inf (no cross-replica / no future leakage).

### The CCA conv subtlety (flat vs segmented) — the draft-quality gate
The CCA `conv_qk` front-end is a causal conv that **ignores the attention mask** (it's a conv over
packed positions, not attention). In the FLAT layout each replica R_r's leading `total_padding` (=2)
tokens would read the wrong left-context through the conv (they'd see S's *draft* tokens, not
`committed+drafts[:r]`), capping draft quality (~1.52× "flat" number, replicas' drafts wrong).

**Fix = SEGMENTED conv** (reference `build_segmented`): immediately before each R_r, insert its
correct `total_padding` conv-context tokens (`committed+drafts[:r]` tail), MASKED from attention so
they ONLY feed the conv. This mirrors the vLLM `cca.py::_decode_verify_spec` per-segment conv. In
minisgl the CCA conv already keys on `seg_pos`/`req_id` (no kernel change) — we just build the fused
token sequence with the right `seg_pos` so each replica's conv window is correct.

## minisgl integration plan (the seams)

Mapped 2026-07-03. Live paged serve uses `attn_prefill_paged` for the multi-query verify (Q = new
tokens vs paged committed prefix), NOT the dense `attn_hip` cold-prefill.

| # | change | where | effort |
|---|---|---|---|
| C.1 | **add `mask_bias` to `attn_prefill_paged`** kernel + bindings + op.py (the live paged path has NO mask arg; only `attn_hip` does). Additive float32, pre-softmax, indexed by (global q pos, global k pos) with `cu_seqlens_q` + paged offsets. | `attn_prefill_paged/{cca? no}` kernel.hip, bindings.cpp, op.py | **MED (kernel)** |
| C.2 | `RDNA4Metadata.custom_mask` field + build it in `prepare_metadata` when TiDAR-fused batch | `python/minisgl/attention/triton_rdna4.py` (RDNA4Metadata ~L17-28, prepare_metadata ~L223) | LOW |
| C.3 | port `tidar_mask.py` (`MaskDescriptor`, `square_additive_bias`, segmented builder) into minisgl | new `python/minisgl/spec/tidar_mask.py` | LOW |
| C.4 | scheduler **fused step**: build `[S | R_0..R_{B-1}]` query batch (+ segmented conv-context via `seg_pos`) over the paged committed prefix, ONE `forward_verify`, `beta_verify` the S rows, pick `R_k`, accept + carry drafts. Reuses the B.0 CCA verify-state capture/install for the committed advance. | `scheduler/scheduler.py` (new `_spec_decode_step_tidar_fused` or a fused branch) | **MED** |
| — | CCA kernel | none — causal by `seg_pos`; segmented context handled in C.4 batch build | — |
| C.5 | forward the `custom_mask` through `HIPAttnBackend._hip_prefill_paged` → the op | `python/minisgl/attention/hip.py` | LOW |

`attn_hip` already applies exactly this additive-mask contract (`attn_kernels.hip:179-196`:
`if (mask_bias && !masked) s += mask_bias[qr*seq_len + kpos]`) — C.1 ports that same 2-line softmax
addition into `attn_prefill_paged`, plus the mask-layout bookkeeping for varlen+paged (the harder
part: the paged kernel's key index is `kv_start + c` into the paged cache, so the mask must be
addressed in the seq's local frame, sliced per `cu_seqlens_q`).

## Cost / risk

- **Query tokens/step** balloons from 2×(B+1)=10 (two-forward) to B+B²=20 (B=4) in ONE forward. Still
  ~memory-bound at 20 tokens (weight read dominates) → ~1 decode-forward wall. Validate the crossover
  with the C.0 offline number before committing; if B² is too heavy, use fewer replicas (pre-draft
  only the top-few k) or smaller B.
- **C.1 paged-kernel mask** is the main risk (varlen + paged offset indexing). Mitigation: first
  prototype on the `attn_hip` dense cold-prefill path (already has `mask_bias`) to validate the
  mask/algorithm end-to-end at bs=1, THEN port to `attn_prefill_paged` for the live paged serve.
- **Lossless gate** identical to B: fused committed stream == AR greedy (S verify rows in fp32).
- Reuses B.0 CCA verify-state (already lossless) for the committed advance.

## Sequencing
C.0 (offline ceiling — validate payoff) → C.3 (mask builder) → prototype on attn_hip dense (bs=1
lossless + accept) → C.1 (paged kernel mask) → C.2/C.5 (metadata plumb) → C.4 (scheduler fused step)
→ lossless + throughput gate vs the two-forward + AR baselines. Then the acceptance-training round.
