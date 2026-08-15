# Plan: serve poolside/Laguna-XS.2-INT4 in minisglang

> Goal: `LagunaForCausalLM` served coherently on the lean gfx1201 stack. Laguna is a novel hybrid-MoE
> (NOT plain Mistral, NOT the DFlash draft). Phased for consecutive execution in fresh contexts.
> Authored from three fact-gathering passes over the repo + the actual checkpoint (2026-07-07).

## Scope reality check (read first)
- **~31B-param MoE** (256 experts × moe_intermediate 512 × 39 MoE layers). It fits ~16 GB **only because
  it is INT4** — dequant-to-bf16 (~62 GB) is INFEASIBLE. The quantized expert path is mandatory.
- The port is an **assembly + hybrid-wiring + quant-gap** job, not a config tweak. Expect TP=2/EP for
  memory headroom during bring-up.
- Two hard unknowns are front-loaded (Phase 2) because they gate correctness/feasibility: the Hadamard
  `transform_config`, and the INT8 expert layers (31–39) which have no serving path today.

---

## Phase 0 — Consolidated facts (Allowed APIs / weight schema / anti-patterns)

### Confirmed architecture (checkpoint config.json)
- `model_type=laguna`, `LagunaForCausalLM`; 40 layers, hidden 2048, vocab 100352, head_dim 128,
  rms_norm_eps 1e-6, `tie_word_embeddings=false` (separate `lm_head.weight`).
- **Hybrid attention**: `layer_types[40]` = `full, sliding, sliding, sliding` repeating → **10 full + 30
  sliding**; `sliding_window=512`. Per-layer heads: **full=48, sliding=64** (`num_attention_heads_per_layer`);
  `num_key_value_heads=8` uniform. **Gated attention**: every layer has `self_attn.g_proj` (per-head
  output gate). QK-norm present (`q_norm/k_norm [128]`).
- **Two RoPE schemes in one model** (`rope_parameters`): full → yarn, theta 500000, factor **VERIFY
  (config dump=64.0 vs agent read=32)**, orig_max_pos 4096, beta_fast/slow 64/1, partial_rotary 0.5
  (rotary_dim=64); sliding → default, theta 10000, partial_rotary 1.0 (rotary_dim=128).
- **MoE**: 256 experts, top-8, moe_intermediate 512, shared_expert_intermediate 512,
  `moe_routed_scaling_factor=2.5`, `moe_apply_router_weight_on_input=false`, `gating=true`.
  `mlp_layer_types[40]`: **index 0 = dense** SwiGLU (inter 8192), indices 1–39 = sparse (MoE).
  Shared expert is **ungated / added directly** (no `shared_expert_gate` weight).
- **Quant** (`compressed-tensors`, `pack-quantized`): TWO config_groups — **layers 1–30 INT4**
  (group 128, sym, `.weight_packed`+`.weight_scale`, pf=8), **layers 31–39 INT8** (pf=4). `ignore`:
  lm_head, layer-0 dense MLP, ALL attention proj (q/k/v/o/g), all `mlp.gate` routers, all
  `mlp.shared_expert.*`. → **only routed experts quantized; backbone + shared + L0 dense are bf16.**
- **`transform_config` (R1 Hadamard)** present — inverse-hadamard on q/k/v/gate/up/g_proj/lm_head
  inputs, forward on embed/o_proj/down_proj outputs. **Not present in the prior Qwen35B ct bring-up.**
- fp8 KV scales (`k_scale/v_scale [1]`, `kv_cache_scheme` float8) exist — skippable for a bf16-KV v1.
  **CORRECTION (2026-07-16, supersedes any earlier "fp8 descale needed for attention" gap entry):** the
  fp8-KV descale ALREADY EXISTS end-to-end — the canonical kernels (`attn_decode`/`attn_prefill_paged`/
  `mla` all take per-tensor `k_descale`/`v_descale`) and minisgl's plumbing (per-layer
  `kvcache.k_scale`/`v_scale` → `attention/hip.py:134`/`rdna4.py:226`, PERF_NOTES A5). Serving fp8 KV
  with the CHECKPOINT's declared scales is a load-time plumbing task (fill the existing per-layer
  scale buffers from `k_scale`/`v_scale` instead of calibrating), NOT kernel work.

### Weight-name schema (verified from index.json — do NOT invent)
- `model.embed_tokens.weight [100352,2048]`, `model.norm.weight`, `lm_head.weight` (untied).
- `model.layers.N.self_attn.{q_proj,k_proj,v_proj,o_proj,g_proj}.weight` + `{q_norm,k_norm} [128]` +
  `{k_scale,v_scale} [1]`. Full L0: q_proj[6144,2048], g_proj[48,2048]. Sliding L1+: q_proj[8192,2048],
  g_proj[64,2048].
- L0 dense: `model.layers.0.mlp.{gate_proj,up_proj[8192,2048],down_proj[2048,8192]}.weight` (bf16).
- L1–39 MoE: `mlp.gate.weight [256,2048]` (router, bf16); `mlp.shared_expert.{gate,up,down}_proj.weight`
  (bf16); `mlp.experts.E.{gate_proj,up_proj}.weight_packed`+`.weight_scale`, `down_proj.weight_packed`+
  `.weight_scale` (INT4 L1–30 / INT8 L31–39, symmetric, no zero_point).

### Allowed APIs (cite these; don't invent)
- `MoELayer(num_experts, top_k, hidden_size, intermediate_size, renormalize, activation,
  apply_router_weight_on_input, quant, fp8_experts)` — `layers/moe.py:228`. Route via raw
  `router_logits` OR precomputed `topk_weights/topk_ids` (`moe.py:324`). **No shared expert inside** —
  model builds it. Unquantized fused path exists (`fused_experts_impl`) but is memory-infeasible here.
- `_GroupedCompressedTensorsExperts` (INT4/pf=8) — `layers/moe.py:129–173` (XOR-0x88, zp=8, `w4a8_moe`).
- Router+scaling+added-shared template: `GLMTopkGate`/`_noaux_tc`/`GLMSparseBlock`/`GLMSharedExpert` —
  `glm4_moe_lite.py:178–265`. Selective-quant (backbone bf16 / experts quantized) via
  `dataclasses.replace(config, quant=None)` — `qwen3_5_moe.py:90–100`.
- Per-layer `layer_types[i]` dispatch template: `qwen3_5.py:281–290`; `from_hf` layer_types parse
  `config.py:257–271` (**GDN-gated — must be un-gated for Laguna**).
- Attention: `RopeAttn` `models/utils.py:83–130`; `AttentionLayer` `layers/attention.py:18–57`
  (backend forward `(q,k,v,layer_id,batch)` — `layer_id` already passed → per-layer sliding lookup is
  feasible with NO signature change). YaRN rope builder `layers/rotary.py:117–136`; `get_rope`
  `rotary.py:149`. `ForCausalLM` skeletons: minimal `qwen3_moe.py:66–80`, rich `qwen3_5.py:447–491`.
- Generic streaming weight loader that already stacks 256 experts + merges gate/up: `weight.py:501–569`
  (`_MERGE_GROUPS`, `_get_expert_stack_info`).

### Anti-patterns (do NOT)
- Do NOT route Laguna through `.mistral` — the MoE/hybrid/gated-attn/per-layer-heads break it.
- Do NOT reuse `ModelConfig.layer_types` as-is (its `from_hf` populate is gated on `linear_num_key_heads`
  and its vocab is `linear_attention`, not `sliding_attention`).
- Do NOT dequant experts to bf16 (memory-infeasible).
- Do NOT set `apply_router_weight_on_input=True` into a quantized `MoELayer` — asserts False at
  `moe.py:291` (Laguna's is False anyway; keep it False).
- Do NOT trust ANY generated text until Phase 2's Hadamard question is resolved.

---

## Phase 1 — Unblock model load: config parsing + registry (no serving yet)
**What:**
1. `models/config.py from_hf`: harden the rope parse — replace the hard subscript
   `rope_theta = rope_scaling["rope_theta"]` (line ~236) with `.get(...)` + a clear error; add a branch
   that reads Laguna's **nested per-attention-type** `rope_parameters` (`full_attention`/`sliding_attention`
   sub-dicts). Store BOTH rope schemes (see Phase 3 — likely two `RotaryConfig`s).
2. Add `ModelConfig` fields: `sliding_window:int|None`, an **un-gated** per-layer attention schedule
   (`attn_layer_types: tuple[str,...]` of `full`/`sliding`), `mlp_layer_types`/`first_k_dense_replace`
   (L0 dense), per-layer head counts (`num_attention_heads_per_layer`), and **fix the key miss**: read
   `moe_routed_scaling_factor` (2.5) not just `routed_scaling_factor`.
3. `models/register.py`: add `"LagunaForCausalLM": (".laguna", "LagunaForCausalLM")` + a stub `laguna.py`.
**Doc refs:** config field decl `config.py:59–83`; GDN layer_types parse `config.py:257–271`; props
`config.py:96–137`; registry `register.py:5–17`.
**Verify (gate):** `python -c "from minisgl.models.config import ModelConfig; ModelConfig.from_hf(<laguna hf config>)"`
runs with NO `KeyError`; asserts: sliding_window==512, 10 full + 30 sliding, routed_scaling==2.5,
L0 dense. (CPU-only, no GPU.)
**Anti-pattern guard:** don't silently default routed_scaling to 1.0; don't reuse the GDN `layer_types`.

## Phase 2 — Risk resolution gates (BEFORE writing the model) — INVESTIGATION
**2a. Hadamard `transform_config` (correctness gate).** Determine whether R1 is fused offline into the
stored weights (→ inference no-op) or requires an ONLINE activation rotation (at o_proj/down_proj input,
etc.). Method: read compressed-tensors `transform_config` semantics for `quantization_status:compressed`;
compare a stored weight vs an un-transformed reference if obtainable; inspect whether apply points are
weight-fused. **Gate:** a written decision — "no-op" or "implement online rotation at sites X" — with
evidence. NOTHING downstream can be trusted until this is answered.
**2b. INT8 expert layers (31–39) feasibility.** minisgl's grouped-expert path is INT4/pf=8 only
(`moe.py:129–173`, `w4a8_moe`). Decide the INT8 path: (i) new INT8 compressed-tensors grouped-expert
container + W8A16 MoE kernel (note: Zaya's `moe_w8a16` is **fp8 F8_E4M3, not symmetric int8** — likely
NOT directly reusable; verify), or (ii) dequant just those 9 layers to bf16 (~14.5 GB — only viable at
TP=2/EP), or (iii) offline requant int8→int4 (lossy). **Gate:** chosen path + memory budget that fits
2×16 GB with TP=2/EP.
**2c. Memory-fit / parallelism.** Confirm INT4(1–30)+INT8(31–39)+bf16 backbone fits with `--tp 2
--enable-ep` (mirror the base-Zaya serve). **Gate:** an estimated VRAM budget per card < 16 GB.
**Verify:** a short written decision doc appended here for 2a/2b/2c. No code yet.

## Phase 3 — `laguna.py` model class (hybrid attention + MoE + dense L0)
**What (copy, don't transform):**
1. Per-layer decoder dispatch on the un-gated attn schedule + `mlp_layer_types` — copy the pattern from
   `qwen3_5.py:281–290`.
2. Attention: fork `RopeAttn` (`utils.py:83–130`) to add (a) **per-layer head count** (48 full / 64
   sliding — `LinearQKVMerged` sized per layer), (b) **gated output** `g_proj` (per-head gate; no existing
   module has this — new), (c) the **layer's own RoPE** (full→yarn/partial-0.5 via `get_rope`
   `rotary.py:117–136`; sliding→default/partial-1.0), (d) **sliding window**: thread `sliding_window` into
   the backend call keyed on `layer_id` (kernels already accept it — `attn_hip` bindings:47,
   `attn_decode` bindings:82; replace the hardcoded `0` at `rdna4.py:213/233`, `hip.py:107/126`).
3. MoE block: fork `GLMSparseBlock` (`glm4_moe_lite.py:209–265`) — **ungated added shared expert**
   (drop the sigmoid gate), `routed_scaling_factor=2.5`, router type per Phase-0 (`gating=true` →
   sigmoid-score; confirm vs softmax from config). Dense L0 = `GatedMLP` (`utils.py:26–54`).
4. `LagunaForCausalLM` skeleton from `qwen3_moe.py:66–80` (+ `qwen3_5.py:447–491` for capture/tie);
   untied `lm_head`.
**Verify (gate):** module imports; a meta-device instantiation builds all 40 layers with correct
per-layer shapes (assert q_proj [6144|8192,2048], g_proj [48|64,2048] per layer type). No GPU.
**Anti-pattern guard:** don't assume uniform heads; don't forget the L0 dense branch; don't apply
sliding to full layers.

## Phase 4 — Weight loading (INT4 + INT8 experts + bf16 backbone)
**What:** extend the generic streaming loader (`weight.py:501–569`) for Laguna:
- INT4 experts L1–30 → `_GroupedCompressedTensorsExperts` (already lands via the merge+stack path).
- INT8 experts L31–39 → per Phase-2b decision.
- Backbone/shared/L0-dense/embed/lm_head bf16 direct; router `mlp.gate` bf16.
- Skip/park `k_scale/v_scale` (bf16-KV v1). Apply the Phase-2a Hadamard decision. (For an fp8-KV
  follow-up: do NOT build anything — load the checkpoint `k_scale/v_scale` into the existing per-layer
  `kvcache.k_scale/v_scale` buffers; the kernel descale is already in canonical kernels. See the
  Phase-0 correction note.)
- **Per-layer QuantConfig** by config_group regex (INT4 vs INT8) — `QuantConfig.from_hf`
  (`quant/config.py:98–108`) currently returns a single bits=4; needs per-layer bit selection.
**Verify (gate):** load completes with zero shape mismatches / zero unexpected-key warnings for a
dummy-weight-disabled boot up to "weights loaded"; log any skipped keys explicitly.

## Phase 5 — GPU coherence validation + iterate (FINAL)
**What:** serve on gfx1201 per CLAUDE.md — **isolated worktree**, lean `serve` compose profile mounting
the worktree, `gpu-lease -n 2`, `MINISGL_MODEL=poolside/Laguna-XS.2-INT4 MINISGL_TP=2 --enable-ep`,
`--attention-backend hip`, `fix_mistral_regex=True` tokenizer. Boot-watch fail-fast (per the
crash-loop lesson). Then a coherence gate.
**Verify (gate):** greedy completion of "The capital of France is" → coherent; a short factual +
a short code prompt read sanely. If garbled → bisect: (a) Hadamard (Phase 2a) wrong, (b) rope
scheme/partial-rotary mismatch, (c) sliding-window masking wrong, (d) router scaling/type, (e) INT8
path. Iterate. **Definition of done:** coherent multi-prompt generation at TP=2+EP.

---

## Final verification
- grep: no `.mistral` route for Laguna; `moe_routed_scaling_factor` read; sliding_window threaded (not
  literal 0 for sliding layers); untied lm_head.
- Re-read config vs ModelConfig fields (per-layer heads, two rope, schedules) match the checkpoint.
- Coherence gate (Phase 5) passes on ≥3 prompts. Then commit on a feature branch + validate before merge.

## Open decisions to make during execution (don't guess — resolve with evidence)
1. YaRN `factor` 32 vs 64 (config dump disagreement) — read the checkpoint config.json directly.
2. Hadamard: no-op vs online (Phase 2a).
3. INT8 experts: kernel vs dequant-9-layers vs requant (Phase 2b).
4. Router: sigmoid-score (noaux/gating) vs softmax-topk (Phase 3.3).
