# Nemotron-3.5-Lightning-30B-A3B-NVFP4 + DSpark — bring-up plan

**Status:** Phase 0 COMPLETE, Phase 1 reference COMPLETE. See the Phase 0 addendum at the end of this
file — reading the checkpoints falsified two of the plan's calls below and found two gaps it did not
name. The body of the plan is left AS WRITTEN so the corrections stay visible; where the addendum and
the body disagree, the addendum wins. No GPU work yet, and no throughput number anywhere.
Written 2026-09-10, work on branch `feat/nemotron-h`.

**Verdict.** Feasible, and cheaper than it looks — but exactly one thing dominates the schedule:
**minisgl has no Mamba-2 (SSD) recurrence.** It has a GDN (gated-delta-net) recurrence, which is a
*different* recurrence wearing very similar clothes. Everything around the recurrence — the conv1d
front end, the gated RMSNorm, the per-slot recurrent state cache, radix snapshotting, graph capture,
the spec-decode replay/rollback machinery, NVFP4 group-16 MoE, and the entire DSpark drafter
architecture — already exists and is already shipped on other arms. The job is one new kernel family
plus a lot of assembly.

---

## 1. What the model actually is (verified from the published configs)

| | |
|---|---|
| Architecture | `NemotronHForCausalLM`, `model_type: nemotron_h` |
| Layers | **52**: 23 `mamba` · 23 `moe` · 6 `attention` (`layers_block_type`, explicit list) |
| Attention layer ids | 5, 12, 19, 26, 33, 42 |
| Hidden | 2688 |
| Attention | GQA 32 q-heads / **2** kv-heads / head_dim 128, `sliding_window: null` (**global**), RoPE θ=10000, `partial_rotary_factor: 1.0`, no bias |
| Mamba-2 | 64 heads × head_dim 64 (inner 4096), `ssm_state_size` 128, `n_groups` 8, `conv_kernel` 4, `chunk_size` 128, `use_conv_bias: true`, `mamba_proj_bias: false` |
| MoE | 128 routed + **1 shared**, top-6, `moe_intermediate_size` 1856, `routed_scaling_factor` 2.5, `norm_topk_prob: true`, `gate.e_score_correction_bias` present |
| Vocab / context | 131,072 / **1,048,576** |
| Weight prefix | `backbone.layers.N.mixer.*` — **not** `model.layers.N.*` |
| Quantization | NVFP4 **W4A16, group_size 16** on routed + shared experts; **FP8 per-tensor static** (`dynamic: false`, weights *and* input_activations) on mamba `in_proj`/`out_proj`; FP8 KV cache |
| Checkpoint size | **21,559,589,596 B = 20.08 GiB**, 52 shards |

The "30B" is marketing rounding; what matters is the 20.08 GiB on disk.

### DSpark drafter (separate checkpoint, `…-NVFP4-DSpark`)

| | |
|---|---|
| Architecture | **`Qwen3DSparkModel`**, `model_type: qwen3` |
| Layers | 6, all `sliding_attention`, window **1024** |
| Shape | hidden 2688, 32 q / 2 kv heads, head_dim 128, intermediate 6144, θ=10000 |
| Block draft | `block_size: 8` (model card pairs it with `num_speculative_tokens 3`) |
| Fixup head | `dspark_fixup_head_type: markov`, `markov_head_type: vanilla`, `markov_rank: 512` |
| Aux capture | `eagle_aux_hidden_state_layer_ids: [2, 6, 20, 30, 42, 52]` — 6 taps → `fc` input 16,128 |
| Vocab | ships `embed_tokens`, **no** `lm_head` (borrows the target's; vocab identical) |
| Quantization | NVFP4 W4A16 group 16, **excluding** `*embed_tokens*`, `*markov_w1*`, `*self_attn*` — i.e. only the MLPs are quantized |

---

## 2. What we already have (the reason this is affordable)

| Capability | Where | Reusable as-is? |
|---|---|---|
| `causal_conv1d_fwd` / `_update` / `_fwd_verify` | `rdna4-hip-kernels/gdn` | **Yes.** Mamba-2's conv1d is the same op; only `conv_dim` changes (here 4096+2048=6144) |
| `rmsnorm_gated` (+ fused epilogue) | same package | **Yes.** Mamba-2 uses the identical gated RMSNorm |
| Per-slot recurrent state cache, slot alloc, eviction | `kvcache/gdn_state.py`, `scheduler/gdn_slots.py` | **Pattern**, not code. Shapes differ (`heads × hdim × state`), the machinery is right |
| Recurrent-state radix snapshot / prefix caching | `kvcache/`, `CompositeRecurrentState` | **Yes**, once a Mamba2 state object implements `clone_slot`/`load_slot` |
| Recurrent graph capture | `gdn/graph_capture.py` | **Pattern** |
| Spec-decode replay / rollback on recurrent state | `gdn_decode_replay`, `gdn_verify_replay`, `gdn_replay_rollback`, `gdn_replay_flush` | **Pattern** — and this is the part everyone forgets; a recurrent model under spec decode must be able to *un-advance* its state |
| NVFP4 **group-16** W4A16 MoE | `quant/nvfp4.py`, `_NvFp4MoEMethod`; kernels assert `group_size == 16 \|\| group_size % 32 == 0` | **Yes — already shipped on the qwen4_exp arm** |
| W4A16 e2m1 MoE prefill + decode GEMV, with SiLU fusion and scatter | `mmq_regdirect_w4a16_moe[_scatter]`, `Int4A16GemvLoader` | **Yes** (`FORMAT_MATRIX.md` §3–4) |
| FP8 W8A8 linear | `quant/method.py:696 Fp8W8A8LinearMethod` | **Probably** — verify it accepts *static* per-tensor activation scales |
| Expert parallelism (EP), TP, DP | engine | **Yes** |
| Sigmoid routing + `e_score_correction_bias` + shared expert | `glm4_moe_lite.py`, `qwen3_moe.py` | **Yes**, copy the routing policy |
| **DSpark drafter architecture** | `models/dflash.py` + `spec/dflash.py` | **Yes.** Per-layer `sliding_attention` windows, `markov_w1`/`markov_w2` rank latent, confidence head `_conf_w`, aux-hidden `fc` fusion, block drafting, `MINISGL_DSPARK_CONF_TAU` adaptive draft length. `SPEC=dspark` is already a shipped mode on the Qwen3.8-27B arms |
| Aux-hidden capture contract | `models/base.py: set_capture_layers(ids)` → `forward(return_hidden=True)` | **Yes** — the target just has to implement it |

The DSpark half is the headline: NVIDIA's DSpark drafter is a Qwen3-style sliding-attention block
drafter with a Markov fixup head, and that is precisely the class this repo already ships.

---

## 3. The gaps, in order of cost

**N1 — Mamba-2 (SSD) recurrence kernels. This is the project.**
GDN and Mamba-2 are both chunked linear recurrences behind a causal conv1d, and they share a shape,
a state layout, and an epilogue — but not their math. GDN is a delta-rule update; Mamba-2 is a
selective SSM: `S ← S·exp(dt·A) + dt·B xᵀ`, `y = C·S + D·x`, with `n_groups=8` sharing B/C across the
64 heads and a `chunk_size=128` SSD scan. Needed:
  * `mamba2_prefill_chunked` — the SSD chunk scan (the hard one)
  * `mamba2_decode` — single-token state update, ideally fused with `causal_conv1d_update` and the
    gated RMSNorm the way `gdn_decode_conv_gated` already is
  * `mamba2_prefill_verify` / `mamba2_decode_replay` / rollback / flush — required for DSpark, not
    optional; without them spec decode silently corrupts recurrent state
  * dtype: state is fp32 (as GDN's is), activations bf16
Per `KERNEL_CORE_POLICY.md` this is a genuinely different *algorithm*, so a new kernel is justified —
but the tiling, the launch geometry, and the chunked-prefill harness must be lifted from the GDN core
rather than re-invented.

**N2 — `NemotronHForCausalLM` model class.** New `models/nemotron_h.py` + registry entry. Three-way
layer dispatch off `layers_block_type`; `backbone.` weight-prefix mapping; MoE routing policy
(sigmoid + correction bias + `routed_scaling_factor` 2.5 + shared expert); `set_capture_layers` for
the DSpark taps.

**N3 — `ModelConfig` vocabulary extension.** `is_gdn_hybrid` keys on `layer_types[i] ==
"linear_attention"`; Nemotron ships `layers_block_type` with `mamba`/`moe`/`attention`. Either
normalise Nemotron's list into the existing vocabulary at `from_hf` time (cheap, preferred) or add a
third hybrid flag. Note `moe` is a *layer kind* here, not an attention kind — the existing list
conflates mixer type and MLP type, and Nemotron separates them.

**N4 — Mixed-precision loading in one checkpoint.** `config_groups` puts FP8 static on mamba
projections and NVFP4 on experts, selected by module pattern. The repo's quant config already parses
compressed-tensors-style groups; confirm it applies *two* schemes in one model rather than one global
scheme.

**N5 — Mamba-2 state cache + slot manager.** New `kvcache/mamba2_state.py` mirroring
`gdn_state.py`: `conv_state (slots, 6144, 3)` fp32 + `ssm_state (slots, heads, 64, 128)` fp32, plus
`clone_slot`/`load_slot` so prefix caching works, and a `CompositeRecurrentState` entry if attention
KV and mamba state need to snapshot together.

**N6 — DSpark wiring.** Small, but real: aux tap ids `[2, 6, 20, 30, 42, 52]` — **52 is out of range
for a 0-indexed 52-layer stack**, so it almost certainly means the post-final-norm hidden state and
the capture contract needs to express that. Drafter loading is NVFP4-with-exclusions (attention and
`markov_w1` stay bf16), which the drafter's `DraftLinear` format policy should already cover.

**N7 — serve.sh arm + Hermes/panel registration.** Alias, `min_tp=2`, tuned defaults, `SPEC=dspark`
default, `served_name`. Cheap, and per the ZAYA lesson (`e0335cc`) the arm must match **every** quant
path from day one.

---

## 4. VRAM, computed

Weights 20.08 GiB. Free per card at boot: ~15.68 GiB.

| | TP=1 | TP=2 |
|---|---|---|
| Weights / rank | 20.08 GiB | **10.04 GiB** |
| Headroom / rank | **−4.40 GiB → impossible** | **+5.64 GiB** |

**TP=2 is forced, and it is also the only TP that works**: 2 kv heads cannot shard 4 ways.
`min_tp=2` belongs in the arm.

Mamba state is per *sequence*, not per token — 23 layers × 32 heads/rank × 64 × 128 × 4 B:

| Concurrency | Mamba state / rank (TP=2) |
|---|---|
| 8 | 0.19 GiB |
| 16 | 0.37 GiB |
| 32 | 0.74 GiB |
| 64 | 1.49 GiB |

KV is tiny because only **6** layers are attention — 1 kv head/rank at TP=2:

| KV dtype | B / token / rank | 1 GiB pool | 3 GiB pool |
|---|---|---|---|
| bf16 | 3,072 | 350k tokens | 1.05M tokens |
| fp8 | 1,536 | 699k tokens | **2.10M tokens** |

**So the advertised 1M context is genuinely reachable on this box** — 1M tokens of KV costs 1.5 GiB
per rank at fp8, and the checkpoint already ships an FP8 KV scheme. Long context is the natural
strength of this architecture on 16 GB cards, and is the thing worth demonstrating.

---

## 5. Phases

Each phase has an exit criterion that is a *measurement or a passing test*, not "code written".

**Phase 0 — paper feasibility (no GPU).** Download the two checkpoints; dump the real tensor
inventory and dtypes; confirm the FP8/NVFP4 module split matches the config groups; confirm the
DSpark tensor names against `models/dflash.py`'s expectations; resolve the `aux id 52` question.
*Exit:* an inventory doc and a go/no-go on N4 and N6. **Half a day.**

**Phase 1 — Mamba-2 numerics, off the engine.** Reference implementation in torch, then the HIP
chunk-scan and decode kernels, validated against it on real checkpoint tensors at fp32 state.
*Exit:* bit-comparable prefill and decode vs the torch reference on a real layer's weights, plus a
roofline number. **This is the long pole.**

**Phase 2 — model class, single layer, no engine.** `nemotron_h.py` builds and runs one of each layer
kind on real weights; MoE routing matches a reference on the checkpoint's own gate.
*Exit:* per-layer parity test in `tests/`.

**Phase 3 — state cache + scheduler.** `mamba2_state.py`, slot alloc, chunked prefill, graph capture.
*Exit:* a coherent TP=2 serve at short context, greedy, graphs live. **First tok/s number.**

**Phase 4 — long context + prefix caching.** `clone_slot`/`load_slot`, fp8 KV, push context up.
*Exit:* a measured context ceiling and a prefix-cache hit rate, sampled (never greedy — see the
project rule).

**Phase 5 — DSpark.** Aux capture in the target, drafter load, block verify, replay/rollback on
mamba state.
*Exit:* accept-length and tok/s **measured sampled**, A/B against `SPEC=none`, with the `engaged()`
ledger diffed per leg.

**Phase 6 — production arm.** serve.sh arm, panel entry, Hermes provider model, tuned operating
point recorded in the table.

Phases 1 and 2 are independent and can run in parallel. Phase 5 depends on 3, not on 4.

---

## 6. Risks and open questions

1. **The SSD chunk scan is the whole schedule.** If it lands slow, the model still serves — a naive
   sequential scan is correct and merely slow — but the arm is not shippable until it is fast. Budget
   accordingly; do not plan around the optimistic case.
2. **`aux_hidden_state_layer_ids` contains 52** on a 52-layer model. Resolve in Phase 0.
3. **FP8 static activation scales.** `Fp8W8A8LinearMethod` was built for dynamic per-token scales on
   other arms. `dynamic: false` means a calibrated per-tensor input scale ships in the checkpoint and
   must be *used*; ignoring it is the silent-quality failure this repo has already been bitten by
   twice (`fp8-kv-was-serving-uncalibrated`).
4. **Two quant schemes in one checkpoint** may not be expressible in the current quant config path.
5. **DSpark on a recurrent target.** Every rejected draft token must un-advance 23 layers of mamba
   state. GDN's replay machinery is the proof this is solvable and the template for how; it is also
   the single most likely source of a subtle correctness bug. Gate it with a verbatim-echo probe.
6. **No fallback path.** Unlike the GGUF arms there is no llama.cpp/Lemonade escape hatch for a
   `nemotron_h` NVFP4 checkpoint on gfx1201 — if the kernels do not land, nothing serves.
7. **Disk.** 20.08 GiB target + ~1.3 GiB drafter, plus the BF16 variant if a reference is wanted.

## 7. What this plan does NOT claim

No throughput estimate. There is no measurement on this box, no comparable Mamba-2 number on RDNA4,
and the closest analogue (GDN) has a different arithmetic intensity. Any tok/s figure quoted before
Phase 3 would be invented.

---

Sources: [target](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) ·
[DSpark drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark) ·
[DFlash drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash) ·
[BF16 reference](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)

---

# Phase 0 — DONE, 2026-09-10. What reading the checkpoints changed.

Method: `tools/nemotron/probe_checkpoint.py` reads a safetensors inventory from the JSON header at
byte 0 of each shard — two HTTP range requests per shard, no weights downloaded. Full inventories in
`docs/measurements/NEMOTRON_INVENTORY/` (18,487 tensors target, 118 drafter). Configs are pinned as
test fixtures under `tests/fixtures/nemotron/`.

## Findings that change the plan

**F1 — The experts are `relu2` and NOT gated. NEW GAP, and it is load-bearing.**
`mlp_hidden_act: "relu2"`. Routed experts ship one `up_proj` (2688→1856) and one `down_proj`
(1856→2688); there is no gate half. The shared expert is 2688→3712→2688, also ungated. Every MoE in
this repo before now is SwiGLU with a fused gate+up and a SiLU epilogue — `mmq_fp8_moe_gemm1_silu`,
`moe_w8a16_gemv_silu`, all of it. A gated path applied here silently halves the intermediate and
multiplies by the wrong thing. Per `KERNEL_CORE_POLICY.md` this is an **epilogue policy on the
existing MoE core** (ReLU² instead of SiLU-and-mul), not a new kernel — but it is real work that the
original plan did not name.

**F2 — One mixer per layer. N3 was wrong.**
52 layers, 52 `backbone.layers.N.norm.weight`, and each layer carries exactly one of
`mixer.in_proj`+`A_log`+`conv1d` (mamba), `mixer.q_proj`… (attention), or `mixer.experts.*` (MoE).
There is no per-layer mixer+MLP pair. The plan said "normalise `layers_block_type` into the existing
`layer_types` vocabulary (cheap, preferred)"; that is not possible — "moe" is a peer of "attention"
here, and mapping "mamba"→"linear_attention" would flip `is_gdn_hybrid` and route the model into the
GDN state cache, slot manager and kernels, which are a **different recurrence**. It would build, run,
and be wrong. **Implemented instead:** a separate `block_types` field with its own vocabulary, with
`layer_types` left None so every GDN and SWA branch stays dead.

**F3 — A latent KV over-allocation, found and fixed.**
`full_attn_layer_ids` returns `range(num_layers)` when `layer_types is None`. For Nemotron that sizes
the paged KV pool for **52 layers instead of 6** — an 8.7× stranding with no error anywhere to
attribute it to. It is now `block_types`-aware. This is the exact failure mode the separate
vocabulary exists to prevent, and it showed up within an hour of having a config to parse.

**F4 — The target ships an MTP module.** `mtp.layers.0.{eh_proj (2688, 5376), enorm, hnorm,
final_layernorm}` plus a full attention+MoE mixer, all **BF16 and unquantized**. So there are three
speculative options, one of them already in the box: MTP (bundled), DSpark (separate checkpoint),
DFlash (separate checkpoint). `SPEC=mtp` already exists in this engine, which makes MTP the cheapest
first spec arm and a useful control for the DSpark A/B.

**F5 — N4 is precisely specified now.** `quant_algo: "MIXED_PRECISION"`, `quant_method: modelopt`,
with **explicit per-module target lists**: `group_0` (FP8 W8A8, `dynamic: false` on both weights and
input activations) names 46 targets — the 23 `in_proj` and 23 `out_proj`; `group_1` (NVFP4 W4, group
16) names 5,935. Plus a `quantized_layers` map and a 2.4 KB `ignore` list. Not a global scheme with
exceptions — an explicit assignment, which is easier to honour than feared.

**F6 — The static activation scale really does ship.** `mixer.in_proj.input_scale` F32 `(1,)` and
`out_proj.input_scale` F32 `(1,)`, alongside per-tensor `weight_scale` F32 scalars. Risk 3 is
confirmed live, not hypothetical: these must be *used*, and `Fp8W8A8LinearMethod` was built for
dynamic per-token scales.

**F7 — fp8 KV descales ship as `k_proj.k_scale` / `v_proj.v_scale`** F32 `(1,)`, which is exactly the
shape this repo already consumes.

**F8 — The DSpark drafter has attention sinks. NEW GAP.**
`layers.N.self_attn.attention_sink_bias` BF16 `(32,)` — a per-head learned sink, plus Qwen3-style
`q_norm`/`k_norm` `(128,)`. `models/dflash.py` has neither. Small, but it is a numerics change, not a
plumbing one.

**F9 — Six aux taps, confirmed by shape.** `fc.weight` is NVFP4-packed `(2688, 8064)` → K = 16,128 =
6 × 2688. The drafter fuses exactly six captured target hidden states.

**F10 — `markov_w2` IS quantized.** Only `markov_w1` is in the exclude list. `markov_w1` BF16
`(131072, 512)`, `markov_w2` NVFP4 `(131072, 256)` → K=512 = `markov_rank`. Both map onto the repo's
existing `_markov_w1`/`_markov_w2` `[vocab, rank]` contract.

**F11 — No confidence-head tensor in the DSpark checkpoint.** The repo's `_conf_w` (and therefore
`MINISGL_DSPARK_CONF_TAU` adaptive draft length) has nothing to load. Fixed-length block drafting
only, unless a later revision ships one.

**F12 — Aux tap 52 is still open,** but the evidence now favours "post-final-norm": the checkpoint has
`backbone.layers.{0..51}.norm.weight` and one `backbone.norm_f.weight`, so tap 52 addressing the
final norm's output is the only consistent reading. Not yet confirmed against a reference forward.

## Also worth knowing

`gate.weight` is **F32** (23 of them) with `gate.e_score_correction_bias` F32 `(128,)`. `lm_head` is
NVFP4-quantized — and since the DSpark drafter ships no `lm_head` of its own, drafted logits go
through the target's quantized head.

## Built in this pass

| | |
|---|---|
| `tools/nemotron/probe_checkpoint.py` | inventory-without-download probe (stdlib only) |
| `docs/measurements/NEMOTRON_INVENTORY/` | target + drafter inventories |
| `tests/fixtures/nemotron/` | the two real configs, pinned |
| `python/minisgl/mamba2/reference.py` | sequential / chunked / decode SSD reference |
| `tests/mamba2_reference_test.py` | 22 tests — chunked ≡ sequential in float64 |
| `ModelConfig` Nemotron-H support | `block_types`, mamba geometry, `relu2`, derived widths |
| `tests/nemotron_config_test.py` | 12 tests against the real config |

**Phase 1 status:** the reference is done and the chunked form — the one the kernel implements — is
proven equal to the definition in float64 at every chunk size, with ragged tails, carried state,
split prefills, and decode-vs-prefill agreement. A later kernel mismatch now means the kernel is
wrong, which is the whole point of doing this before writing HIP.
