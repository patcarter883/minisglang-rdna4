# Qwen3.6-35B-A3B-AWQ load bring-up (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`) — DONE (2026-06-28)

**STATUS: implemented + GPU-validated.** Boots TP=2, coherent, MTP spec works (accept 0.60–0.94).
MTP spec-length sweep: optimal K≈4, **+65% tok/s** (25.45→41.89) — see tools/spec_len_sweep_results.md.
What landed (matches the plan below):
- `layers/moe.py` `_GroupedCompressedTensorsExperts` — signed int4 -> the op's `(q-zero)` convention by
  XOR 0x88 per byte + constant zero-point 8 (whole-tensor, no per-expert loop); wired the MoELayer branch.
- `models/weight.py` — skip `.weight_shape`; TP-shard weight_packed/weight_scale experts (gate/up dim 0,
  down dim 1). (Gate/up merge cat-dim 0 already correct for N-major compressed-tensors.)
- `models/qwen3_5.py` — MTP `fc` `LinearColParallelMerged` -> `LinearReplicated` (latent TP>1 bug: the
  Qwen MTP was only ever validated at TP=1; the fc must produce the FULL hidden seed) + threaded
  `mlp_factory` so the 35B's MoE MTP block builds.
- `layers/base.py` — the weight-mismatch assert now names the key/shapes (found `mtp.fc.weight` instantly).
Found via a CPU-only meta-build + real `load_state_dict` probe at simulated TP=2
(`tools/qwen35b_loadstate_probe.py`) — no GPU needed to iterate the loader/shard fixes.

The original scoping write-up follows (kept for reference).

---

## Scope: Qwen3.6-35B-A3B-AWQ load bring-up (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`)

Goal: make this checkpoint load + serve in minisgl so it can run the spec-decode length sweep
(`tools/run_spec_len_sweep.sh`). Investigated 2026-06-28 with `tools/qwen35b_load_scope.py`
(CPU-only meta-build + shape diff) — no GPU needed for the scoping.

## TL;DR
It is **NOT** a from-scratch bring-up — config parsing, the `qwen3_5_moe` model, the GDN-hybrid
path, the VL `language_model.*`/`visual.*`/`mtp.*` weight remap, and TP sharding **already exist and
work**. The single blocker is a **quant-format gap**: the checkpoint is **compressed-tensors
int4 weight-only (group-32, symmetric)**, but minisgl's MoE expert path only implements **GPTQ / AWQ /
RXF**. Despite "AWQ" in the repo name, the file has **zero** `qweight/qzeros/scales` — it is 100%
`weight_packed` / `weight_scale` / `weight_shape` (31488 of each = the routed experts only; the whole
backbone is bf16, kept in the checkpoint's `ignore` list). Estimate: **~1–2 focused sessions + GPU
validation**, medium risk (the one sharp edge is symmetric-int4 zero-point handling in the kernel).

## What already works (verified)
- **Config**: `ModelConfig.from_hf` unwraps the VL `text_config` (config.py:155) → 40 layers, hidden
  2048, 256 experts, group-32, `mtp_num_hidden_layers=1`, is_moe=True. ✓
- **Quant detect**: `QuantConfig.from_hf` recognizes compressed-tensors (config.py:91). ✓
- **Weight remap**: `weight.py::qwen3_5_remap` strips `model.language_model.`, skips `model.visual.*`,
  handles `mtp.*`, concats the GDN `in_proj_*`, renames `conv1d`. ✓
- **Backbone is bf16**: attention / GDN `linear_attn` / shared-expert / router / norms are all in the
  checkpoint `ignore` list (`.weight`, 737 of them) — no quant work needed there. ✓

## The blocker (exact failure)
`create_model` raises at **`layers/moe.py:228`**:
`AssertionError: MoE W4A8 unsupported quant method: compressed-tensors`.
`MoELayer.__init__` dispatches expert storage by quant method:
`is_gptq → _GroupedGPTQExperts`, `is_awq → _GroupedAWQExperts`, `is_rxf → _GroupedRXFExperts`,
**else raise**. There is no compressed-tensors branch. (This is why the earlier serve boot died in
`load_state_dict` — actually it dies even earlier, at model construction, once you get past the cached
build.)

## Format delta (compressed-tensors W4A16 vs the existing AWQ/GPTQ W4A8 experts)
| | this checkpoint (compressed-tensors) | existing AWQ path |
|---|---|---|
| packed weights | `weight_packed` uint8 `[N, K/2]` (2×int4) | `qweight` int32, N-major interleaved |
| scales | `weight_scale` `[N, K/32]` | `scales` `[K/g, N]` |
| zero-point | **none (symmetric)** | `qzeros` (asymmetric) |
| group size | 32 | 128 (kernel op layout is 32) |
| activations | **none (W4A16)** | runtime int8 (W4A8 kernel) |
| per-expert key | `experts.<e>.{gate,up,down}_proj.weight_{packed,scale,shape}` | `…{qweight,qzeros,scales}` |

Both ultimately target the **same int4 grouped W4A8 WMMA MoE kernel** (each `_Grouped*Experts`
converts to the op's grouped layout in `post_load`, then `kernels.*_moe` runs). So the work is a
**layout adapter**, not a new kernel — *provided* the kernel can run symmetric int4 (zero-point ≡ 8 /
no qzeros). That is the main risk to retire first.

## Work plan
1. **`QuantConfig`** (`quant/config.py`): add an `is_compressed_tensors_w4a16` discriminator (group-32,
   symmetric, weight-only) vs the existing rxf W4A8 compressed-tensors. (Detection already lands as
   `method="compressed-tensors"`; just expose the sub-variant.)
2. **`layers/moe.py`**: add `_GroupedCompressedTensorsExperts` (mirror `_GroupedAWQExperts`): hold
   stacked `weight_packed [E,N,K/2]` + `weight_scale [E,N,K/32]`; in `post_load` unpack the int4 +
   build the op grouped layout the AWQ/GPTQ experts already produce (symmetric → zero-point 8). Wire
   the `is_compressed_tensors` branch in `MoELayer.__init__` (the line that currently raises) and the
   matching `forward` kernel call (reuse the existing int4 grouped MoE kernel).
3. **`models/weight.py`** (`_load_qwen3_5_weight` + `_shard_qwen3_5`): handle the compressed-tensors
   expert suffixes — stack `weight_packed`/`weight_scale` over E (skip `weight_shape`, it's metadata);
   gate/up merge on the N axis (dim 0 of `[N,K/2]`); TP-shard gate/up by N (dim 0), down by K (dim 1).
   Add the suffixes to the AWQ-only shard table (weight.py:245-250) and the cat-dim logic
   (weight.py:295). Same for the `mtp.*` experts (MTP also has 256 compressed-tensors experts).
4. **Attention sanity**: full-attn layers use `head_dim=256` (16 q-heads × 256 ≠ hidden 2048) — confirm
   `Qwen3_5Attn` reads `head_dim` (not hidden/num_heads). Likely already correct; verify on boot.
5. **Validate**: TP=2 coherence smoke (it's ~17.5 GB int4 → fits 2×16 GB), then losslessness vs plain
   decode at K=1, then run the sweep (`ALGO=mtp` configs).

## De-risk first (1 cheap GPU probe before writing the adapter)
Confirm the int4 grouped MoE kernel produces correct output for **symmetric** weights (zero-point 8,
no qzeros) at **group-32**. If the kernel hard-assumes asymmetric AWQ qzeros, step 2 grows (either
synthesize qzeros≡8, or take the GPTQ-symmetric route). Quickest check: convert ONE expert's
`weight_packed`/`weight_scale` to the op layout with zp=8 and diff a GEMM against a bf16 dequant
reference (extend `tools/qwen35b_load_scope.py`).

## Out of scope
Vision tower (`model.visual.*`, skipped — text-only serving), and the W4A16→W4A8 activation
quantization quality question (the AWQ path already runs W4A16 weights through the W4A8 kernel, so this
inherits the same accept/quality behavior — and spec decode is lossless regardless, the verify
corrects drafts).
