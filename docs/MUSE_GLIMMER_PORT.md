# Muse-Glimmer 30B (NVFP4) — serving port

Target: `RedHatAI/Muse-Glimmer-30B-NVFP4` (base `meta-models/Muse-Glimmer-30B`, Meta, 2026-08-10,
Apache-2.0). Dense multimodal decoder; **this port is text-only**, matching every other multimodal
checkpoint this engine serves (Gemma4, Mistral3, Qwen3.5 all skip their vision towers).

Reference implementation is stock `transformers >= 5.15.0`
(`src/transformers/models/muse_glimmer/{modular,modeling,configuration}_muse_glimmer.py`) — no
remote code. Every claim below was read off that source or off the checkpoint's own tensor headers,
not inferred.

## 1. Shapes

| | |
|---|---|
| layers | 52 |
| hidden | 6656 |
| heads | 32 q / 2 kv (GQA 16:1), head_dim 128 |
| MLP | SwiGLU, intermediate 19968 |
| vocab | 202048, `tie_word_embeddings: false` |
| ctx | 131072 |

**`num_heads * head_dim = 4096 != hidden_size = 6656`.** `q_proj` and `gate_proj` are 6656→4096,
`o_proj` is 4096→6656. Nothing may assume the usual square identity.

## 2. The five things that are easy to get silently wrong

These are the whole risk surface of the port. Each produces plausible-looking output when wrong.

### 2.1 Two different RMSNorm conventions in one model
- The **four per-layer norms** are `CenteredRMSNorm`: `normed * (1.0 + weight)` — minisgl's
  `RMSNorm(..., plus_one=True)`.
- The **final `model.language_model.norm`** is plain `normed * weight` — `plus_one=False`.

Mixing them is a pure quality bug with no crash and no shape error.

### 2.2 Two different epsilons in one layer
```
input_layernorm            eps = rms_norm_eps  = 1e-5
post_attention_layernorm   eps = post_norm_eps = 1e-8
pre_feedforward_layernorm  eps = rms_norm_eps  = 1e-5
post_feedforward_layernorm eps = post_norm_eps = 1e-8
```
Sandwich (Gemma-2/3) order — the **post-norms sit on the sublayer OUTPUT, before the residual add**,
so the fused rmsnorm+residual-add op does *not* apply to them:
```python
residual = x
x = input_layernorm(x); x = self_attn(x); x = post_attention_layernorm(x); x = residual + x
residual = x
x = pre_feedforward_layernorm(x); x = mlp(x); x = post_feedforward_layernorm(x); x = residual + x
```

### 2.3 QK-norm is WEIGHTLESS, pre-RoPE, and Q carries a 3.87 factor
```python
qk_norm = RMSNorm(eps=rms_norm_eps, with_scale=False)   # no parameter -> NOT in the checkpoint
q = qk_norm(q) * qk_scale_factor                        # 3.87, Q ONLY
k = qk_norm(k)
```
Applied per-head over `head_dim`, **before** RoPE. There are no `q_norm.weight`/`k_norm.weight`
tensors — nothing in the state dict hints these norms exist, exactly the `RMSNormNoScale` trap
Gemma4's `v_norm` documents.

`3.87` is **on top of** the standard `head_dim**-0.5`, not a replacement. Since Q and K are both
RMS-normalised to unit RMS, this factor is what sets the softmax temperature. Effective scale:
`3.87 / sqrt(128) = 0.3420629053989924`. **Folded into `ModelConfig.attn_softmax_scale`** — exact
(scaling Q by c then dotting == scaling the logits by c) and saves a full-tensor multiply per layer.

### 2.4 `self_attn.gate_proj` — sigmoid output gate, and a name collision
```python
attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))   # BEFORE o_proj
attn_output = self.o_proj(attn_output)
```
- Driven by **`hidden_states`** (the `input_layernorm` output), not by the attention result.
- **Per-channel** over the full 4096 (contrast Laguna's `g_proj`, which is per-HEAD `[nqo]` and uses
  **softplus**, not sigmoid — do not copy that gate verbatim).
- `self_attn.gate_proj` (6656→4096) and `mlp.gate_proj` (6656→19968) share a leaf name and are
  different modules. A loader that keys on `gate_proj` alone will mis-map one into the other.
- TP: `colwise`, sharded exactly like q_proj.

### 2.5 NoPE on the full-attention layers
`layer_rope_theta[i] == 0` means that layer applies **no positional encoding at all**. The reference
reads the list only as a boolean and builds ONE rotary from the global `rope_theta = 500000.0`, so a
per-layer *nonzero* theta would be ignored upstream too; every nonzero entry here is 500000.0.

For this checkpoint the NoPE layers coincide **exactly** with the full-attention layers —
`3, 7, 11, …, 51` (13 of 52). Both fall out of the same rule `(num_layers - 1 - i) % 4 == 0`. The
port derives `nope_layer_ids` independently from `layer_rope_theta` rather than aliasing it to
`full_attn_layer_ids`, because the coincidence is a property of this config, not an invariant.

So: 39 sliding layers (window **2048**) carry RoPE θ=500k; 13 global layers carry none.

## 3. Embedding and logits

**Embedding** passes through a *weightless* RMSNorm after lookup (`embed_norm`, eps=`rms_norm_eps`).
This is **not** Gemma's `sqrt(hidden)` multiplier — do not reach for `embed_scale`. The reference
keeps it unfused from the embedding matrix deliberately, so the DFlash drafter can embed without it.

**Logits** — `output_multiplier` pre-scale, then the Gemma tanh softcap:
```
final = 20.0 * tanh(lm_head(h) * 0.19611613513818404 / 20.0)
```
`output_multiplier == 1/sqrt(26)`. It lands on the **returned** logits, so it changes sampling (not
just a training loss scale) and any spec-decode verify path must reproduce it.

## 4. Quantization — NVFP4, already supported

`nvfp4-pack-quantized`, 4-bit, **group_size 16**, `tensor_group`, symmetric, scales
`float8_e4m3fn`, activations dynamic `local`. `kv_cache_scheme: null` (KV not quantized).

This is the scheme `python/minisgl/quant/nvfp4.py` already implements and Laguna already serves:
`fold_nvfp4_scale` at the **leaf** (before any merge/stack) folds the per-tensor `weight_global_scale`
into the per-group `weight_scale`, after which NVFP4 is structurally MXFP4-at-16 and rides the same
e2m1 W4A8 kernel. `input_global_scale` is dropped (the kernel quantizes activations to fp8).

Quantized: `self_attn.{q,k,v,o,gate}_proj` and `mlp.{gate,up,down}_proj`, 8 linears x 52 layers.
Unquantized (in `ignore`): all norms, `embed_tokens`, final `norm`, **`lm_head`**, and the **entire**
vision tower / adapter / projection.

## 5. Checkpoint key scheme

Prefix is **`model.language_model.`** (not `model.text_model.`); `lm_head.weight` is top-level.
Per decoder layer, 12 leaves: 8 linears (each NVFP4 → `weight_packed`, `weight_scale`,
`weight_global_scale`, `input_global_scale`) + 4 norms. No biases anywhere in the text stack. No
`q_norm`/`k_norm`.

Loader must: strip `model.language_model.` → `model.`, skip `model.vision_tower.` /
`model.vision_adapter.` / `model.vision_projection.`, fold NVFP4 scales at the leaf, TP-shard, and
merge `mlp.gate_proj`+`mlp.up_proj` → `gate_up_proj` **without** touching `self_attn.gate_proj`.

## 6. VRAM — TP=2 is mandatory

~23.3 GB of weights on two 16 GB cards:

| | |
|---|---|
| 52 layers quantized (25.2B params @ 4bit + e4m3 group scales) | ~14.2 GB |
| `embed_tokens` bf16 | 2.69 GB |
| `lm_head` bf16 (untied, ignore-listed) | 2.69 GB |
| vision tower bf16 (**skipped** by this port) | ~3.7 GB |

Text-only at TP=2 ≈ **9.8 GB/card**, leaving room for KV + activations. KV is cheap — 2 KV heads x
128 x 2 x 2 B = 1 KB/token/layer, and only the 13 global layers need full-context pages; the other 39
are ring-buffered at 2048.

## 7. Serving format — harmony-style channels + ATEM tool calls

`chat_template.jinja` is **not** a `<think>`/`</think>` family template:

```
<|start|>ROLE[ to=RECIPIENT]<|message|>CONTENT<|eot|>
```
- **Reasoning is a separate TURN**, not a delimited span:
  `<|start|>assistant to=self<|message|>…<|eom|>`
- **Tool calls are a separate TURN** addressed to the tool:
  `<|start|>assistant to=<toolname><|message|><atem:function_calls>…</atem:function_calls><|eot|>`
- The generation prompt is a bare `<|start|>assistant` — the model itself emits ` to=…` or
  `<|message|>`.

Token ids: `<|start|>` 200022, `<|message|>` 200023, `<|eom|>` 200007, `<|eot|>` 200008.
`generation_config.eos_token_id = [200001, 200008]` — **`<|eom|>` is deliberately NOT an EOS**: it
ends a *turn* and generation continues with a new `<|start|>assistant` header. A stop-string
implementation truncates every reasoning reply at the end of its thinking.

The existing `ReasoningParser` (open/close token pair) does map onto this cleanly:
`start_token=" to=self<|message|>"`, `end_token="<|eom|><|start|>assistant<|message|>"`.

Tool-call payload is Anthropic-style XML:
```
<atem:function_calls>
<atem:invoke name="NAME">
<atem:parameter name="KEY">VALUE</atem:parameter>
</atem:invoke>
</atem:function_calls>
```
Structurally close to ZAYA's native `<zyphra_tool_call><function=…><parameter=…>` — the EBNF grammar
in `_zaya_xml_grammar` and its parser are the template to follow, registered through
`_derive_tool_format` (which probes the rendered template) rather than a named special case.

## 8. Status

**Landed (validated, CPU-only — the two cards were held by the live serve throughout):**
- `layers/attention.py` — `rotary_config: RotaryConfig | None`; `None` == NoPE, the rope is never
  built and never called (not a zero-frequency identity rope, which still costs a launch per layer).
- `layers/norm.py` — `RMSNormNoScale.forward_inplace`, so a weightless norm can stand in for the
  weighted `q_norm`/`k_norm` an `AttentionLayer` applies to its split q/k views.
- `models/config.py` — `output_multiplier`, `post_norm_eps`, `layer_rope_theta` fields;
  `is_muse_glimmer` and `nope_layer_ids` predicates; `qk_scale_factor` folded into
  `attn_softmax_scale`; and a fix so a multimodal wrapper whose `model_type` the installed
  transformers does not register (the generic `PretrainedConfig` fallback) gets its **dict**
  `text_config` promoted to a config object with its `model_type` preserved.

The last one is a general fix, not a Muse-Glimmer special case: it is what any brand-new
architecture hits first, and it failed as `'dict' object has no attribute 'architectures'` — which
reads as "unsupported model" rather than "your transformers is old". The image ships transformers
**5.14.1**, which does not know `muse_glimmer`; with this fix a version bump is **not** required,
since minisgl only ever reads config fields via `getattr` and implements its own modeling.

Verified parse of the real checkpoint config: 52 layers, hidden 6656, 32/2 heads, head_dim 128,
sliding_window 2048, `attn_softmax_scale = 0.3420629053989924`, `final_logit_softcapping = 20.0`,
`output_multiplier = 0.196116…`, `post_norm_eps = 1e-8`, `is_swa_hybrid = True`,
`nope_layer_ids == full_attn_layer_ids == (3, 7, …, 51)`, quant NVFP4 group-16.

**Not yet written:**
1. `models/muse_glimmer.py` — the decoder (§2 is the spec).
2. `models/weight.py::_load_muse_glimmer_weight` + the `load_weight` dispatch line, and the
   `models/register.py` entry for `MuseGlimmerForConditionalGeneration`.
3. Serve-side chat format (§7).
4. Numeric parity vs the reference, then a TP=2 serve bring-up. Both need a GPU lease.

**Deferred:** the vision tower. It is greenfield for this engine — there is no image path in
`message/`, `scheduler/`, or `core.Batch` at all, and image parts are currently dropped silently at
the API boundary (`Message._flatten_content_parts`). Serving a *vision* model text-only should be a
stated limitation, not a surprise.
