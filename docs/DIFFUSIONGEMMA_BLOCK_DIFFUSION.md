# DiffusionGemma — block-diffusion serving in minisgl

**Scope.** `cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4` (`DiffusionGemmaForBlockDiffusion`). This is
the sibling of the already-ported `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`: **same 30-layer
backbone, byte-for-byte the same weight VALUES in the text stack**, but the head is a discrete
diffusion denoiser over a fixed 256-token canvas instead of an autoregressive LM head.

**Provenance of citations.** `dg_modeling_diffusion_gemma.py`, `dg_modular_diffusion_gemma.py`,
`dg_configuration_diffusion_gemma.py`, `dg_generation_diffusion_gemma.py` are the reference copies in
the task scratchpad. The serve image (`minisgl-rdna4:lean`) ships transformers **5.14.1**, whose
`transformers/models/diffusion_gemma/*` differ from the scratchpad copies **only** in RoPE-buffer
plumbing and config field declaration (verified by diff); every passage cited below is semantically
identical, at a ≤7-line offset (e.g. `DiffusionGemmaDecoderTextAttention` is at `:369` in the
scratchpad copy, `:376` in the image). `dg_generation_diffusion_gemma.py` was copied *out of* the
image, so its line numbers are exact for the shipped version.

Everything asserted as "verified" below was read out of the checkpoint's own tensors or executed
inside the image on CPU. Everything I could not settle is in **§11 Flagged unknowns** — nowhere else.

---

## Part A — exact semantics

### A0. Checkpoint facts (verified, not inferred)

Config (`config.json`, `text_config`): `hidden_size 2816`, `intermediate_size 2112`,
`moe_intermediate_size 704`, `num_experts 128`, `top_k_experts 8`, `num_attention_heads 16`,
`num_key_value_heads 8`, `head_dim 256`, `num_global_key_value_heads 2`, `global_head_dim 512`,
`sliding_window 1024`, `final_logit_softcapping 30.0`, `use_bidirectional_attention "vision"`,
`layer_types` = 25 `sliding_attention` / 5 `full_attention` at indices **5, 11, 17, 23, 29**.
Top level: `canvas_length 256`, `eos_token_id [1, 106]`, `tie_word_embeddings true`.
This is **identical geometry to the AR sibling** — `ModelConfig.from_hf` already accepts
`model_type == "diffusion_gemma"` (`models/config.py:563-566`, `:590-596`) and already produces
`swa_head_dim=256 / head_dim=512`, `swa_num_kv_heads=8 / num_kv_heads=2`,
`attn_softmax_scale=1.0`, `embed_scale=sqrt(2816)`.

Tensor census of the 35 777-tensor index (verified by reading `model.safetensors.index.json`):

| namespace | count | content |
|---|---|---|
| `model.decoder.*` | 35 391 | the entire text stack: `embed_tokens`, 30× layers, `norm`, `self_conditioning` |
| `model.encoder.language_model.layers.N.layer_scalar` | 30 | the **only** per-layer encoder text tensors |
| `model.encoder.embed_vision.embedding_projection.weight` | 1 | vision projector |
| `model.encoder.vision_tower.*` | 355 | 27-layer SigLIP-ish tower (skipped; this engine is text-only) |
| `lm_head` | **0** | tied to `model.decoder.embed_tokens.weight` (`dg_modeling:1603`) |

`self_attn.v_proj.*` exists on 25 layers; **absent on exactly 5, 11, 17, 23, 29** — i.e. the
full-attention layers, confirming `attention_k_eq_v` behaviour. Note DiffusionGemma does not carry the
config flag at all: `DiffusionGemma{Encoder,Decoder}TextAttention.__init__` hard-codes
`v_proj = Linear(...) if self.is_sliding else None` (`dg_modeling:300-304`, `:402-406`). The existing
loader already decides this from the tensors (`models/config.py:574-589`) — correct here.

**Verified: encoder `layer_scalar` == decoder `layer_scalar`, all 30 layers, to fp16 precision**
(0.068848, 0.204102, …, 0.183594). So in *this* checkpoint the encoder text stack is numerically
identical to the decoder text stack in every parameter. See §A1 for why that is a load-time
assertion, not an assumption.

`model.decoder.self_conditioning`: `pre_norm.weight [2816]`, `gate_proj [2112,2816]`,
`up_proj [2112,2816]`, `down_proj [2816,2112]` — all **fp16, unquantized** (they are in the
compressed-tensors `ignore` list). No `post_norm` tensor: it is `with_scale=False`
(`dg_modeling:805`), i.e. our `RMSNormNoScale`.

`generation_config.json`: `canvas_length` implied 256, `max_denoising_steps 48`, `t_max 0.8`,
`t_min 0.4`, `confidence_threshold 0.005`, `stability_threshold 1`, `max_new_tokens 256`,
`pad_token_id 0`, `eos_token_id [1, 106, 50]` (note: **three** ids, one more than `config.json`),
`sampler_config = EntropyBoundSamplerConfig(entropy_bound 0.1)`.

---

### A1. The encoder/decoder split, and what "weight-tied" actually means

`DiffusionGemmaModel` holds two sub-models (`dg_modeling:1493-1498`):
`self.encoder = DiffusionGemmaEncoderModel(config)` and
`self.decoder = DiffusionGemmaDecoderModel(config)`.

The class docstring is the load-bearing sentence (`dg_modeling:1471-1477`):

> *NOTE: contrarily to most encoder-decoder models, where the encoder feeds its hidden states to the
> decoder, here the encoder only feeds its **KV cache** to the decoder. From the decoder's
> perspective, the KV cache is read-only.*

**There is no cross-attention and no encoder hidden state consumed by the decoder.** The encoder's
`last_hidden_state` is returned (`dg_modeling:1557`, `:1591`) but is used only for an optional
auxiliary AR training loss; `generate` never reads it (`dg_generation:734-742` keeps only
`encoder_outputs.past_key_values`).

**What the encoder is:** a bog-standard **autoregressive Gemma4 stack**. `DiffusionGemmaEncoderTextModel.forward`
builds `create_causal_mask` / `create_sliding_window_causal_mask` (`dg_modeling:956-959`) and
`DiffusionGemmaEncoderTextAttention.is_causal = config.use_bidirectional_attention != "all"`
(`dg_modeling:281`). This checkpoint sets `"vision"`, **not** `"all"`, so:
* text tokens are **causal**;
* the `sliding_window //= 2` rewrite in `DiffusionGemmaTextConfig.__post_init__`
  (`dg_configuration:106-108`) **does not fire** — the window stays 1024;
* `create_masks_for_generate` passes `block_sequence_ids = -1` everywhere when there are no images
  (`dg_modeling:1183-1188`), degenerating to a plain causal mask.

So **the encoder forward is exactly the AR `gemma4` forward the port already runs.** It writes the KV
cache (`past_key_values.update(...)`, `dg_modeling:344-345`).

**What "tied" covers** (`dg_modeling:1481-1491`):

```
"encoder.language_model.norm.weight":            "decoder.norm.weight"
r"encoder.language_model.layers\.(?:[^.]+\.)*weight":            r"decoder.layers\....weight"
r"encoder.language_model.layers\.(?:[^.]+\.)*scale":             ...
r"encoder.language_model.layers\.(?:[^.]+\.)*per_expert_scale":  ...
r"encoder.language_model.layers\.(?:[^.]+\.)*gate_up_proj":      ...
r"encoder.language_model.layers\.(?:[^.]+\.)*down_proj":         ...
"encoder.language_model.embed_tokens.weight":    "decoder.embed_tokens.weight"
```

The comment above it says *"All weights in the text part of the encoder are present in the decoder.
However, only the decoder has the self-conditioning layers. At the time of writing, HF code assumes
only weights can be tied."* The regexes deliberately match `weight`/`scale`/`per_expert_scale`/
`gate_up_proj`/`down_proj` and **not** `layer_scalar`, because `layer_scalar` is a **buffer**
(`nn.Buffer` / `register_buffer`, `dg_modeling:614`, `:692`) and HF's tying machinery only ties
Parameters. That is why — and the only reason why — the checkpoint ships 30 encoder tensors.

Therefore, precisely:

* **Shared (tied):** embeddings, all 30 layers' attention projections + q/k norms, dense MLP, router
  (`proj.weight`, `scale`, `per_expert_scale`), all 128×30 experts, every LayerNorm gain, final `norm`.
* **NOT tied, shipped separately:** `layer_scalar` × 30. **Verified numerically equal in this
  checkpoint** (§A0).
* **Decoder-only, no encoder counterpart:** `self_conditioning.{pre_norm,gate_proj,up_proj,down_proj}`.
* **Encoder-only:** `embed_vision.embedding_projection` + the vision tower (text-only serve: skip).

**Engine consequence.** One instantiated 30-layer stack serves both roles. The current loader's
`_GEMMA4_SKIP_PREFIXES` (`models/weight.py:847-851`) skips **all** of `model.encoder.`, with the
comment *"diffusion_gemma nests the whole vision encoder here"* — which is **incomplete**: it also
nests the 30 encoder `layer_scalar`s. Today that is harmless (they are equal), but it is a silent
assumption. **Do not just keep skipping it — load them and `assert torch.equal` against the decoder's,
failing loudly if a future checkpoint diverges.** If one ever does, the single-stack design breaks and
you need two `layer_scalar` vectors selected by execution mode.

**How often does the encoder run?** Once per **canvas block**, not per denoising step
(`dg_generation:720-742`: the encoder call is step 1.a of the *outer* loop; the inner denoising loop
at `:758-790` never touches it). The first call is the prompt prefill
(`unprocessed_input_ids = input_ids`), every later call re-encodes **only the last 256 committed
tokens** (`dg_generation:949`: `input_ids[:, -canvas_length:]`) as an append to the same cache.

---

### A2. Attention — the KV-cache question ★

This is the question that decides the port, so it is answered exhaustively.

#### A2.1 What the decoder attends over

`DiffusionGemmaDecoderTextAttention` (`dg_modeling:369-497`). Its own docstring (`:370-379`):

> *1. Removes shared KV cache logic … 2. **It doesn't update the KV cache in the forward pass. The KV
> cache here corresponds to the encoder's KV cache, which is passed in via `past_key_values` — from
> the decoder's perspective, it can be seen as a read-only encoder KV cache.** 3. `self.is_causal` is
> set to `False`.*

`self.is_causal = False  # In the decoder, attention is bidirectional!` (`dg_modeling:383`).

The forward projects Q/K/V from the **canvas** hidden states only (`:430-444`), then:

```python
if past_key_values is not None:
    key_states, value_states = self.append_to_cache(past_key_values, key_states, value_states)
```
(`dg_modeling:446-447`)

So the answer is: **a single concatenated self-attention over `[encoder KV] ++ [canvas KV]`.**
Not cross-attention, not two attention calls. The canvas provides the queries (256 of them); the keys
and values are the encoder's cached prefix followed by the canvas's own freshly computed K/V.

#### A2.2 `append_to_cache` — what it does and does not do

```python
def append_to_cache(self, past_key_values, key_states, value_states):
    """Append the key and value caches and return the full key-values. It doesn't modify
    anything in-place in contrast to `past_key_values.update`, the tensors are concatenated
    and returned"""
    cache_layer = past_key_values.layers[self.layer_idx]
    if not past_key_values.is_compileable:
        keys   = torch.cat([cache_layer.keys,   key_states],   dim=-2)
        values = torch.cat([cache_layer.values, value_states], dim=-2)
    else:
        ... index_copy_ into a fresh new_zeros(max_len + new_length) ...
    return keys, values
```
(`dg_modeling:471-497`)

**It is a pure function. It writes nothing back.** `cache_layer.keys` is untouched;
`cache_layer.cumulative_length` is untouched. The concatenated tensor is a *transient* consumed by
this one attention call and freed.

#### A2.3 Therefore: cached vs recomputed, per denoising step

| tensor | who writes it | lifetime | recomputed per denoising step? |
|---|---|---|---|
| encoder K/V for prefix positions `[0, cur_len)` | the encoder, once per block | whole block (and across blocks) | **no** |
| canvas K/V for positions `[cur_len, cur_len+256)` | the decoder, every step | one step | **yes, all 30 layers, every step** |
| canvas hidden states / logits | the decoder | one step | yes |

**Definitive answer to the design question:** the engine's paged KV cache is reusable **unchanged for
the encoder**, which is the only thing that persists — it is append-only, causal, absolute-positioned
and radix-cacheable exactly like today's prefill. The canvas needs **256 scratch slots per request
that are overwritten in place on every denoising step and never committed to the radix cache.**

The cheap way to get those scratch slots is *not* a separate cache. Because the canvas occupies the
contiguous absolute positions immediately after the prefix, you can allocate 256 real slots from the
same pool at the sequence tail, let the ordinary `store_kv` scatter rewrite them each step, and run
the existing paged-extend kernel with `causal=0` and `context_len = cur_len + 256`. That is bit-for-bit
the reference's `[prefix | canvas]` bidirectional attention. See §B8.

Two supporting facts that make this exact rather than approximate:

* `attn_prefill_paged_kernels.hip:189-191` — `unsigned int kv_limit = context_len; if (causal)
  kv_limit = min(context_len, prefix_len + q_start + q_rows);`. With `causal=0` the only mask left in
  the softmax loop is the tile-tail bound `c >= kv_len` (`:231`). **Non-causal full attention over
  `context_len` keys is already implemented and already shipped** (it is the fused-TiDAR path,
  `attention/rdna4.py:401-429`).
* The canvas positions are ordinary ascending absolute positions
  (`decoder_position_ids = arange(cache_seq_length, cache_seq_length + canvas_length)`,
  `dg_modeling:1284-1293`), and the next block's encoder pass re-encodes the committed canvas at
  **exactly those same positions** (`dg_generation:1123`: `encoder_position_ids = decoder_position_ids`).
  So RoPE, page-table addressing and the radix cache all see a perfectly ordinary monotone sequence.

#### A2.4 The mask the reference actually builds

`create_diffusion_decoder_attention_mask` (`dg_modeling:1325-1438`). Two paths:

1. **No padding** (the single-request serve case): returns `{"full_attention": None,
   "sliding_attention": None}` (`:1374-1380`) — no mask at all. With `is_causal=False` this reaches
   SDPA as `attn_mask=None, is_causal=False` (transformers `sdpa_attention_forward`:
   `is_causal = q_length > 1 and attention_mask is None and is_causal`, which stays `False`), i.e.
   **dense bidirectional attention over every key**.
2. **With padding**: it builds a 4D mask via
   `mask_interface(..., mask_function=bidirectional_mask_function, allow_is_causal_skip=False,
   local_size=text_config.sliding_window, ...)` (`dg_modeling:1420-1435`). I executed
   `transformers.masking_utils.bidirectional_mask_function` in the image: its body is
   `return q_idx >= 0` — i.e. **always True**. `local_size` is consumed only by the
   `allow_is_causal_skip` shortcut logic, never to carve a window (verified from `sdpa_mask`'s
   docstring/signature in the image). So the produced mask is a **pure padding mask** with no
   causality and no window, on **both** layer types.

The in-source justification (`dg_modeling:1399-1401`):

> *DiT module doesn't need a sliding mask and has to attend fully to prev context and itself. To
> enforce a full mask we pass `or_mask_function`, while keeping the functionality of
> `create_bidirectional_sliding_window_mask` to get correct the mask shape and offsets*

The intent is unambiguous: **the decoder attends fully.**

#### A2.5 Both facts verified empirically (CPU, real transformers 5.14.1)

Built a 2-layer / hidden-64 / `sliding_window=8` DiffusionGemma with the same layer schedule
(1 sliding + 1 full), ran a 40-token encoder prefill, then one decoder forward over a 6-token canvas:

* **Bidirectionality.** Perturbing only the **last** canvas token changes the hidden state at
  **canvas position 0** by `3.4e-3` (positions 1–4 by ~3e-3, position 5 by 3.2e-1). A causal decoder
  would leave positions 0–4 bit-identical. → **bidirectional, confirmed.**
* **Backend agreement.** `attn_implementation="sdpa"` vs `"eager"` on the same weights: max abs diff
  on the decoder's final hidden = `1.5e-8` (fp32 round-off). Since `eager_attention_forward`
  (`dg_modeling:238-269`) ignores `sliding_window` outright, agreement proves SDPA ignores it too. →
  **the `sliding_window=` kwarg at `dg_modeling:462` is inert on both reference-quality backends;
  the decoder is full bidirectional on the 25 sliding layers as well as the 5 full ones.**
* **Cache truncation.** With `sliding_window=8` and a 40-token prompt, the sliding layer's cache holds
  **7** keys and the full layer holds **40**. → confirms `sliding_window - 1`, i.e. **1023** for the
  real checkpoint.

(FA2 remains untested — no CPU build — but see U1: it is the one path that would apply a window, and
the reference's own mask comment says it should not.)

---

### A3. Layer schedule, split head_dim, and where the sliding window actually lives

**The 256/512 split and the 25/5 kv-head split apply verbatim inside the decoder.** Both attention
classes read `layer_config = config.per_layer_config[layer_idx]` for `head_dim` and
`num_key_value_heads` (`dg_modeling:289-292` encoder, `:391-394` decoder) — the *same* config object,
so the geometry is identical. `self.scaling = 1.0` on both (`:293`, `:395`). V-norm is
`with_scale=False` on both (`:312`, `:414`). The full layers alias `value_states = key_states` (the
**pre-k_norm, pre-RoPE** `k_proj` output) on both (`:335`, `:437`). Every trap in
`models/gemma4.py`'s module docstring carries over unchanged.

**The sliding window in the decoder is enforced by CACHE TRUNCATION, not by a mask.** This is the
subtle part. `DynamicCache(config=text_config)` instantiates `DynamicSlidingWindowLayer` for the 25
sliding layers and `DynamicLayer` for the 5 full ones — I verified this in the image
(`is_sliding = [True]*5 + [False] + ...`). `DynamicSlidingWindowLayer.update` (source read in image):

```python
full_key_states = torch.cat([self.keys, key_states], dim=-2)
self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]   # keeps 1023, not 1024
return full_key_states, full_value_states
```

So on a sliding layer the **encoder cache physically holds only the last `sliding_window - 1 = 1023`
prefix positions**. The decoder then attends *fully* over `1023 + 256 = 1279` keys. On a full layer it
attends fully over `cur_len + 256`.

Net effect for a canvas query at position `p ∈ [cur_len, cur_len+256)`:

| layer type | keys visible | mask |
|---|---|---|
| sliding (25) | prefix `[cur_len-1023, cur_len)` ∪ the whole 256-canvas | none (bidirectional) |
| full (5) | prefix `[0, cur_len)` ∪ the whole 256-canvas | none (bidirectional) |

Note the asymmetry this creates and **do not "fix" it**: every canvas position sees the *same* 1023
prefix keys — the window does **not** slide within the canvas. A naive "bidirectional sliding window
of ±1024 centred on each query" is a *different* model. The reference's window is a property of what
the cache retained at the moment the encoder stopped writing.

Both rows of that table are empirically confirmed in §A2.5 (`sliding_window=8`, 40-token prompt →
sliding cache holds 7 keys, full cache holds 40; and canvas position 0 demonstrably reads canvas
position 5).

Off-by-one: `1023`, not `1024`. minisgl's SWA ring is sized `W = config.sliding_window = 1024`
(`attention/rdna4.py:123`, `engine/engine.py:55-69`). For canvas reads the block table must expose
`Wp = min(cur_len, 1023)` prefix slots — see §B8.

---

### A4. The self-conditioning MLP

`DiffusionGemmaSelfConditioning` (`dg_modeling:790-823`). One instance on the decoder only
(`dg_modeling:1229`), applied once at the **input embedding**, before layer 0:

```python
def forward(self, inputs_embeds, self_conditioning_signal):
    normed    = self.pre_norm(self_conditioning_signal)              # RMSNorm, learned gain
    sc_signal = self.down_proj(self.act_fn(self.gate_proj(normed)) * self.up_proj(normed))
    combined  = inputs_embeds + sc_signal
    return self.post_norm(combined)                                  # RMSNorm, NO gain
```
(`dg_modeling:811-823`)

`act_fn = ACT2FN["gelu_pytorch_tanh"]`, `intermediate_size = 2112` (the dense-MLP width, not the MoE
width). **`post_norm` is applied to the SUM, on every step, including step 1 when the signal is
zero** — so it is not a no-op branch; it renormalizes the embedding unconditionally.

**What feeds it** (`dg_modeling:1267-1280`):

```python
inputs_embeds = self.embed_tokens(decoder_input_ids)                 # scaled by sqrt(2816)
if self_conditioning_logits is not None:
    soft_embeddings = torch.matmul(
        self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(embed_dtype),
        self.embed_tokens.weight,
    ) * self.embed_tokens.embed_scale.to(inputs_embeds.dtype)
else:
    soft_embeddings = torch.zeros_like(inputs_embeds)                # <- step 1 of each block
inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)
```

The signal is the **previous denoising step's temperature-scaled logits**, cast to the embedding dtype
(fp16) *before* the fp32 softmax — `self_conditioning_logits = processed_logits.to(embeddings_dtype)`
(`dg_generation:1070-1071`), where `processed_logits = raw_logits / T` (`dg_generation:1042`, the
`LinearTemperatureScheduleLogitsProcessor`). The resulting probability-weighted average of the
embedding table is then re-scaled by `sqrt(hidden)` exactly as a hard embedding lookup would be.

**Which steps.** Zero on the first step of every canvas block (`self_conditioning_logits` is reset to
`None` in `_prepare_denoiser_inputs`, `dg_generation:991`), the previous step's logits on every
subsequent step. A row that has hit the diffusion stopping criterion freezes its signal
(`dg_generation:1063-1065`). `self_conditioning_mask` is a training-time per-example dropout and is
never set by `generate`.

**Cost note, because it dominates.** `softmax([256, 262144]) @ [262144, 2816]` is
**≈ 378 GFLOP per denoising step per request**, i.e. as expensive as the LM head, plus a **268 MiB
fp32** softmax transient and a **134 MiB fp16** logits tensor held across the step boundary. See §B10.

---

### A5. The denoising loop

**There is no mask token.** The canvas is initialised with **uniform random token ids over the whole
262 144-entry vocabulary** and un-accepted positions are **re-randomised** every step. This is a
uniform-state discrete diffusion, not an absorbing-mask one:

```python
def initialize_canvas(self, batch_size, device):
    return torch.randint(low=0, high=self.vocab_size, size=(batch_size, self.canvas_length), device=device)
```
(`dg_generation:394-404`; `renoise_canvas` calls it again at `:467`)

**`t_max`/`t_min` are a temperature schedule on logits, not a noise level.**
`temperature = t_min + (t_max - t_min) * (cur_step / max_denoising_steps)` with `cur_step` counting
**down** N..1 (`dg_generation:315`, loop at `:758`). With the shipped config it runs
`T = 0.4 + 0.4*(n/48)`, i.e. **0.8 → 0.408**.

**How many tokens are unmasked per step: data-dependent, via the entropy bound** — not a schedule.
`EntropyBoundSampler.accept_canvas` (`dg_generation:406-448`) sorts per-position entropies ascending,
takes the longest prefix `k` with `sum_{i<=k} H_i - max_{i<=k} H_i <= entropy_bound`, i.e. the largest
approximately-independent set. Since the first element always gives `0 <= bound`, **at least one token
is accepted per step**.

**The critical non-obvious fact: acceptance is NOT monotone and the current canvas is NOT carried
forward.** Compose `accept_canvas` (`:447`) with `renoise_canvas` (`:466-468`):

```
accepted = where(mask, denoiser_sample, current)
next     = where(~mask, fresh_random, accepted)
         = where(mask, denoiser_sample, fresh_random)      # `current` cancels out entirely
```

Every position is either this step's freshly *sampled* token (if accepted) or brand-new uniform noise.
`accepted_token_mask` is recomputed from scratch each step; a token accepted at step *n* can be
re-noised at step *n+1*. Convergence comes from self-conditioning and the model's own stability, not
from an accumulating commit. **And the emitted output is `argmax_canvas`, not the canvas** —
`input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)` (`dg_generation:793`).

**`confidence_threshold` / `stability_threshold`** drive `StableAndConfidentStoppingCriteria`
(`dg_generation:484-540`) — an *early exit*, not an acceptance rule:
* stable ⇔ the argmax canvas equals it did on each of the last `stability_threshold` steps
  (`:521-530`; history is initialised to `-1` so step 1 is never stable; with `=1` it is simply
  "same argmax canvas as last step");
* confident ⇔ `mean(per-position entropy of the temperature-scaled logits) < confidence_threshold`
  (`:533-535`).

Full pseudocode (single request; batch dims dropped; every line cited):

```python
V, L, N = 262144, 256, 48                    # vocab, canvas_length, max_denoising_steps
T_MIN, T_MAX, EB   = 0.4, 0.8, 0.1
CONF, STAB         = 0.005, 1
EOS                = {1, 106, 50}            # generation_config.json
cache = DynamicCache(config)                 # 25 sliding layers keep last 1023; 5 full keep all
out, cur_len, is_prefill = prompt_ids, len(prompt_ids), True

for block in range(ceil(max_new_tokens / L)):                       # gen:720
    # ---- 1.a ENCODER: causal AR pass, WRITES the cache -------------------------
    enc_in = out if is_prefill else out[-L:]                        # gen:949
    encoder(enc_in, positions=range(cur_len - len(enc_in), cur_len),
            cache=cache, causal=True, sliding_window=1024)          # gen:734-740
    is_prefill = False

    # ---- 1.b init the denoising state ------------------------------------------
    x       = randint(0, V, (L,))            # UNIFORM NOISE, no MASK id           gen:987-989
    sc      = None                           # self-conditioning logits            gen:991
    a_prev  = None
    a       = x

    # ---- 1.c denoise ------------------------------------------------------------
    for n in range(N, 0, -1):                                       # gen:758  (N..1)
        # decoder: 256 queries at absolute positions [cur_len, cur_len+L),
        # BIDIRECTIONAL over [cache | own 256 canvas keys]; writes nothing back.   model:446-447
        h      = decoder(x, self_conditioning=sc, cache=cache,
                         positions=range(cur_len, cur_len + L))
        logits = 30.0 * tanh(lm_head(h).float() / 30.0)             # model:1666-1670 (fp32)

        T = T_MIN + (T_MAX - T_MIN) * (n / N)                       # gen:315   0.8 -> 0.408
        l = logits / T                                              # gen:1042

        y = multinomial(softmax(l, dim=-1, dtype=fp32))             # gen:1043-1048  per position
        a = argmax(l)                                               # gen:1049

        H     = categorical_entropy(l)                              # [L]        gen:437-438
        order = argsort(H)                                          # ascending  gen:439
        sel   = cumsum(H[order]) - H[order] <= EB                   # gen:440-443
        acc   = scatter(sel, order)                                 # [L] bool   gen:444-446

        x = where(acc, y, randint(0, V, (L,)))                      # gen:447 ∘ gen:466-468
                                                                    # NB: previous x is discarded

        stable    = (a_prev is not None) and all(a == a_prev)       # gen:521-530 (STAB == 1)
        confident = mean(H) < CONF                                  # gen:533-535
        a_prev    = a
        if stable and confident:                                    # gen:1067, gen:789-790
            break

        sc = l.to(fp16)                                             # gen:1070-1071
        # consumed next step as softmax(sc) @ embed_tokens.weight * sqrt(hidden)  model:1272-1275

    # ---- 1.d commit the ARGMAX canvas (not x) ----------------------------------
    out = concat(out, a)                                            # gen:793

    # ---- 1.e stop ---------------------------------------------------------------
    if any(t in EOS for t in out[-L:]):                             # gen:1090-1095 EosTokenCriteria
        # everything strictly after the FIRST eos becomes pad_token_id (0)
        out = pad_after_first_eos(out, L, EOS, pad=0)               # gen:1103-1108
        break
    if len(out) >= max_length: break                                # MaxLengthCriteria, gen:1202-1203

    # ---- 1.f next block ---------------------------------------------------------
    cur_len += L                                                    # gen:1120-1126
    # next encoder pass re-encodes out[-L:] at positions [cur_len-L, cur_len)
```

**Two consequences worth stating explicitly.**
1. `tokens_per_forward` (`dg_generation:836-856`) is the model's own efficiency metric: valid tokens
   emitted ÷ decoder forwards. Best case 256/1 (unreachable); the config's early exit makes a
   realistic block cost somewhere between a handful and 48 forwards of **256 tokens each**. Compare
   the AR sibling: 48 forwards of 1 token. Even at the worst case this is a very different
   compute/bandwidth mix — see §B10.
2. There is **no incremental streaming of committed tokens.** A block is opaque until it finishes.
   The reference offers `streamer.put_draft(argmax_canvas)` (`dg_generation:782-786`) to stream the
   *evolving draft*, which is not the same thing as an OpenAI-style token stream.

---

### A6. Termination, and output longer than 256

**Inner (per-canvas):** `StableAndConfidentStoppingCriteria`, §A5. Purely an early exit; the argmax
canvas is committed either way.

**Outer (per-block):** `_finalize_canvas` (`dg_generation:1080-1109`) runs the ordinary
`StoppingCriteriaList` = `[MaxLengthCriteria(max_length), EosTokenCriteria(eos_token_id)]`
(`dg_generation:1202-1205`) over the whole `input_ids` with `new_token_length=canvas_length`, i.e.
**"does any EOS appear anywhere in the 256 just-committed tokens?"** If so:

```python
is_eos     = torch.isin(new_tokens, eos_tensor)
eos_cumsum = is_eos.cumsum(dim=-1)
pad_mask   = (eos_cumsum > 0) & ~((eos_cumsum == 1) & is_eos)   # everything AFTER the first eos
new_tokens[pad_mask] = pad_token_id                             # 0
```
(`dg_generation:1103-1108`)

The EOS itself is kept; everything after it in the canvas becomes pad. A row that finished on an
*earlier* block has its whole new canvas overwritten with pad (`:1101`).

**Multi-block.** `max_new_canvases = ceil(max_new_tokens / canvas_length)` (`dg_generation:650`).
Each iteration of the outer loop: encode the previous block causally into the cache, denoise a fresh
canvas, commit, check EOS. `_prepare_kwargs_for_next_canvas` (`:1111-1127`) advances `cur_len += 256`,
sets `encoder_position_ids = decoder_position_ids` (the block is re-encoded at the positions it was
denoised at) and moves `decoder_position_ids` forward by 256.

**The important structural point for the engine:** the canvas's own K/V is *thrown away* at the block
boundary and the committed tokens are re-derived through the **causal** encoder. The model therefore
never serves from decoder-produced KV. Generation length is unbounded in blocks of 256, and each block
boundary is an ordinary 256-token extend-prefill.

---

## Part B — engine fit

### B7. Which autoregressive assumptions break

Grounded in `core.py`, `scheduler/scheduler.py`, `engine/engine.py`, `kvcache/`, `attention/rdna4.py`.

**B7.1 One-new-token-per-step decode — breaks, but the seam already exists.**
The 1-token assumption is not encoded in a batch shape; it is encoded in two integers.
`Req.complete_one()` does `cached_len = device_len; device_len += 1` (`core.py:158-160`), applied to
every req at `engine/engine.py:1041-1042`. Everything downstream —
`_make_positions` (`scheduler.py:4533-4554`), `_make_input_tuple` (`:4557-4564`),
`_make_write_tuple` (`:4567-4572`), `CacheManager.allocate_paged` (`scheduler/cache.py:61-72`),
`RDNA4Backend.prepare_metadata` (`attention/rdna4.py:637-692`) — derives purely from
`extend_len = device_len - cached_len`.

The spec-decode verify path already generalises this: `req.device_len = c0 + len(d) + 1`
(`scheduler.py:3895`) followed by `req.cached_len = c0 + len(keep); req.device_len = cached_len + 1`
(`:4263-4264`) and a page rollback (`:4339-4359`). **A canvas step is the same manoeuvre with
`extend_len = 256` and a rollback that is a no-op (the slots are reused, not freed, until block end).**

**B7.2 Append-only paged KV — breaks for the canvas, holds for the encoder.**
`MHAKVCache.store_kv(k, v, out_loc, layer_id)` is a scatter into `out_loc`
(`kvcache/mha_pool.py`), so **overwriting** the same 256 slots on every step already works — nothing
in the pool assumes monotonicity. What breaks is the *scheduler's* accounting: `allocate_paged`
(`cache.py:61-72`) computes pages from `div_ceil(cached_len)` → `div_ceil(device_len)` and expects the
range to be freshly needed. A canvas step must allocate the 256 slots **once per block** and then
leave `cached_len`/`device_len` alone across all ≤48 denoising steps.

**B7.3 `out_loc` / page-table addressing — holds, unchanged.**
The canvas occupies contiguous absolute positions `[cur_len, cur_len+256)` in the request's own
page-table row. `batch.out_loc = page_table[input_mapping]` (`scheduler.py:1584`, `:3941`) works
verbatim. The full-attention layers need no new addressing at all.
**The sliding layers do:** `_build_swa_metadata` (`attention/rdna4.py:694-732`) maps position `p` to
ring slot `table_idx*R + p%R` with `R = swa_ring_stride`. With `R = W = 1024`, canvas position
`cur_len + j` aliases prefix position `cur_len + j - 1024` — it would **overwrite the very window it
must read**. The fix is already in the codebase: `engine.py` widens `R = window + num_draft + 1` for
spec verify so the speculative block lands in disjoint slots. Here `R = 1024 + 256 = 1280`.

**B7.4 Radix prefix cache — holds, with one rule.**
`CacheManager.cache_req` inserts a finished request's tokens into the radix trie
(`scheduler/cache.py:94-97`). The canvas KV must **never** be inserted: it is scratch and it is
re-derived causally by the next encoder pass. But the *encoder* side is fully radix-compatible — the
committed block is a normal 256-token extend-prefill over a monotone prefix. `SamplingParams.cache_output`
(`core.py:76`) already exists as the "don't insert the generated tail" lever if needed.
**Non-obvious win:** because the encoder re-encodes each committed 256-block, a multi-turn conversation
gets a genuine prefix-cache hit on the whole prior transcript — the same as today.

**B7.5 The sampler — breaks entirely; it is a different object.**
`engine/sample.py` samples **one row per request** (`engine.py:1044`:
`self.sampler.sample(logits[:batch.size], args)`), applies temperature/top-k/top-p/penalties/grammar.
The canvas needs, per request per step: full-vocab entropy over 256 positions, a per-position
multinomial, a per-position argmax, an entropy-ordered acceptance set, and a re-noise. None of that is
top-k/top-p sampling. It must be a **separate diffusion sampler module**, and the existing
`BatchSamplingArgs` path should be bypassed (not extended) for canvas batches. Penalties, grammar
(`engine/grammar.py`) and the reasoning gate are all per-token-position autoregressive concepts and do
**not** apply — they should be rejected at admission with a clear error rather than silently ignored.

**B7.6 The scheduler's request lifecycle — needs a third phase.**
Today: `PrefillManager.pending_list` → `DecodeManager.running_reqs` → `finished_reqs`
(`scheduler/prefill.py:132`, `scheduler/decode.py:18`, `scheduler.py:297`). A diffusion request
alternates `ENCODE (1 forward) → DENOISE (≤48 forwards) → COMMIT → ENCODE …`. `DecodeManager` has no
notion of a request that needs *k* forwards before it emits anything, and `_process_last_data`
(`scheduler.py:1139-1251`) assumes exactly one token per req per forward
(`req.append_host(next_token.unsqueeze(0))`, `:1184`; `eos_hit = next_token in eos_token_ids`, `:1194`).
`DetokenizeMsg` already carries `extra_tokens: List[int]` (`message/tokenizer.py:36`) for the spec
path, so the *output* side is fine — a block emits one message with 1 + 255 extra tokens.

**B7.7 cudagraph capture — deferred, but the shape is ideal.**
`RDNA4Backend.init_capture_graph` raises (`attention/rdna4.py:735-738`); `HIPAttnBackend`
(`attention/hip.py:159`) adds decode capture, plus four multi-token capture families
(`_fill_verify_static`, `_fill_swa_verify_static`, `_fill_fused_verify_static`,
`_fill_ddtree_verify_static`, `attention/hip.py:371-599`). A canvas step is a **fixed 256-query,
fixed-batch-size forward** — a better capture target than any spec shape, since the only dynamic
quantity is the prefix length (handled the same way `_fill_swa_verify_static` handles it: a static
padded block table plus a per-seq `context_len`). **But per the repo's graph-capture rule, eager-only
is not "done"** — plan for it, do not ship without it.

**B7.8 LM head reduction — already solved by `phase="decode"`.**
`ParallelLMHead` reduces to last-token only when `batch.is_prefill`
(`layers/embedding.py:159-160`). A canvas batch is `phase="decode"` with `extend_len=256`, which
returns logits for **all** 256 positions — exactly the spec-verify contract documented at
`core.py:210-215` and `engine/engine.py:1085-1089`.

**B7.9 Weight loader — two concrete breaks.**
(a) `_GEMMA4_SKIP_PREFIXES` skips `model.encoder.` wholesale (`models/weight.py:847-851`) — see §A1;
(b) `model.decoder.self_conditioning.*` remaps cleanly to `model.self_conditioning.*` via
`_gemma4_remap` (`:855-877`) but `Gemma4Model` declares no such module, so the loader's exact-key
check fails at boot. Both are small, both must be fixed deliberately.

**B7.10 `head_dim 512` — the pre-existing blocker, unchanged.**
`_HIP_HEAD_DIMS = (64, 128, 256)` (`attention/rdna4.py:26`); the dispatch macros in
`attn_kernels_hip.hip:287-295` and `attn_prefill_paged_kernels.hip:771-773/824-826/1327-1329/1375-1377`
instantiate only 64/128/256. The 5 full-attention layers at `head_dim 512` have **no HIP kernel** —
this already blocks the AR gemma4 port (`_no_hip_kernel`'s own message says so,
`attention/rdna4.py:214-220`) and is not made worse by block diffusion. `MINISGL_ATTN_HIP=0` routes to
Triton `unified_attention`, but that path hard-codes `causal=True, window_size=(-1,-1)`
(`attention/rdna4.py:308-309`) and so is **useless for a canvas**. See §B10.

---

### B8. Implementation proposal

**Verdict on the model:** *the decoder stack is reusable as-is.* `Gemma4Model` (`models/gemma4.py:298-324`)
already implements every layer of DiffusionGemma's decoder, correctly, including the five traps. The
new code is a **head + an execution mode**, not a new backbone.

**Verdict on the shape:** a canvas denoising step is a **new `Batch` phase**, sitting exactly where
`spec_verify` sits — `phase="decode"` (all-position logits, paged-extend attention branch) plus a new
flag that means "non-causal, and do not advance the request".

#### B8.1 New/changed files and the seam at each

| file | change | seam |
|---|---|---|
| `python/minisgl/models/diffusion_gemma.py` **(new)** | `DiffusionGemmaForBlockDiffusion(BaseLLMModel)`: reuses `Gemma4Model` unchanged as the shared stack; adds `SelfConditioning` (`RMSNorm` + gate/up/down + `RMSNormNoScale`) and the tied `ParallelLMHead`. `forward()` branches on `get_global_ctx().batch.canvas is not None`: canvas → apply self-conditioning to the embedding, return all-position softcapped fp32 logits; else → plain AR encoder forward, identical to `Gemma4ForConditionalGeneration.forward`. | `get_global_ctx().batch` (`core.py:264-267`) |
| `python/minisgl/models/register.py` | `"DiffusionGemmaForBlockDiffusion": (".diffusion_gemma", "DiffusionGemmaForBlockDiffusion")` | `models/register.py:18-21` |
| `python/minisgl/models/config.py` | carry `canvas_length`, `max_denoising_steps`, `t_min`, `t_max`, `entropy_bound`, `confidence_threshold`, `stability_threshold` off `generation_config.json`. Everything else already parses. | `ModelConfig.from_hf`, `:550-700` |
| `python/minisgl/models/weight.py` | (a) narrow `_GEMMA4_SKIP_PREFIXES` to `model.encoder.vision_tower.` + `model.encoder.embed_vision.`, load the 30 encoder `layer_scalar`s and **assert** they equal the decoder's; (b) accept `model.decoder.self_conditioning.*`. | `:847-877` |
| `python/minisgl/core.py` | `Batch.canvas: CanvasMeta \| None = field(default=None, init=False)` — one new field, mirroring `spec_verify` (`core.py:215`). `CanvasMeta` carries `step`, `temperature`, `prefix_len`, `canvas_len`. | `core.py:193-231` |
| `python/minisgl/attention/rdna4.py` | in `forward()`: `causal = 0 if batch.canvas else 1`, threaded into `_hip_prefill_paged` (already parameterised, `:423-429`) and a new `_swa_prefill_canvas` cloned from `_swa_prefill_paged` (`:561-592`) with `causal=0, sliding_window=0`. In `prepare_metadata`, a canvas branch that builds the ring block table as `[Wp prefix slots | 256 canvas slots]`, `context_len = Wp + 256`. | `:261-322`, `:637-732` |
| `python/minisgl/attention/hip.py` | `init_canvas_capture` / `_fill_canvas_static` / `prepare_canvas_for_{capture,replay}`, cloned from the SWA-verify family. **Stage 5, not stage 1.** | `:295-462` |
| `python/minisgl/engine/engine.py` | `swa_ring_stride = window + canvas_length` when the model is block-diffusion (the spec widening already exists); `forward_canvas(batch)` alongside `forward_verify` (`:1080-1122`) — no sampling, no `complete_one`. KV byte reservation via `_swa_kv_geometry` (`:55-69`) must account for the widened ring. | |
| `python/minisgl/diffusion/` **(new package)** | `sampler.py` — `EntropyBoundSampler` (`initialize_canvas`/`accept_canvas`/`renoise_canvas`, a direct port of `dg_generation:394-469`); `stopping.py` — `StableAndConfident`; `state.py` — the per-request `CanvasState` (`x`, `argmax_prev`, `sc_soft`, `step`, `finished_denoise`). | |
| `python/minisgl/scheduler/canvas_slots.py` **(new)** | per-request canvas slot bookkeeping: 256 main-pool slots + 256 ring slots, allocated at block start, held across all denoising steps, released/committed at block end. Model it on `RecurrentSlotManager` (`scheduler/recurrent_slots.py:48-94`) — same idempotent-free discipline. | |
| `python/minisgl/scheduler/scheduler.py` | `_diffusion_loop()` as a sixth entry in `run_forever`'s selector (`:976-1058`); `_canvas_block_step(reqs)` alongside `_spec_decode_step` (`:3778`); a `CanvasManager` alongside `DecodeManager`. | |

#### B8.2 The canvas step, concretely

```
_canvas_block_step(reqs):                            # one denoising step for a batch of reqs
  # --- pre: every req in `reqs` is mid-block; its 256 canvas slots are already allocated,
  #          and req.cached_len == cur_len (the prefix), req.device_len == cur_len + 256.
  #          These two integers do NOT change across the ≤48 steps of the block.

  1. write the current canvas ids into the token pool at columns [cur_len, cur_len+256)
     -> exactly the spec-decode staging scatter, scheduler.py:3896-3904

  2. batch = Batch(reqs, phase="decode"); batch.canvas = CanvasMeta(step=n, temperature=T, ...)
     # phase="decode"  -> LM head returns all 256 rows            (layers/embedding.py:159-160)
     # extend_len=256  -> paged-extend attention branch           (attention/rdna4.py:641,658-660)
     # batch.canvas    -> causal=0 in the kernel; skip complete_one; skip the AR sampler
     NO allocate_paged here — the slots were allocated at block start.
     batch.positions   = arange(cur_len, cur_len+256)             (scheduler.py:4544-4553, unchanged)
     batch.out_loc     = page_table[input_mapping]                (scheduler.py:1584, unchanged)
     attn_backend.prepare_metadata(batch)                          # canvas branch, see below

  3. logits = engine.forward_canvas(batch)          # [bs*256, vocab] fp32, softcapped

  4. diffusion sampler (per req, on device):
       l   = logits / T
       H   = entropy(l); a = argmax(l); y = multinomial(softmax(l))
       acc = entropy_bound_select(H, EB)
       x   = where(acc, y, randint(V))
       done = (a == a_prev).all() and H.mean() < CONF
       sc_soft = softmax(l.to(fp16)) @ embed_tokens.weight * sqrt(hidden)   # [256, 2816]

  5. if done or n == 1: commit the block  (see below); else keep the req in the canvas set
```

**Attention metadata for a canvas step.**
* Full layers (5): `page_table` row is the request's ordinary page-table row truncated to
  `cur_len + 256`; `context_len = cur_len + 256`; `cu_seqlens_q = [0, 256, 512, ...]`;
  `causal = 0`. `flash_prefill_paged` then attends every query over every key —
  `kv_limit = context_len` (`attn_prefill_paged_kernels.hip:189`).
* Sliding layers (25): block table `[Wp prefix ring slots (ascending abs position) | 256 canvas ring
  slots]` with `Wp = min(cur_len, 1023)`, `context_len = Wp + 256`, `causal = 0`, `sliding_window = 0`.
  This is `_swa_prefill_paged` (`attention/rdna4.py:561-592`) with the causal flag flipped and the
  ring stride widened to `1024 + 256`. Note **1023, not 1024** — §A3.

**Block commit.**
```
- emit `a` (the argmax canvas), truncated at the first EOS
- req.append_host(a_kept)                                   # core.py:162-176
- req.cached_len = cur_len + 256 ; req.device_len = cached_len   # the block is now prefix
- send DetokenizeMsg(uid, a_kept[0], finished, extra_tokens=a_kept[1:])   # tokenizer.py:36
- if not finished: schedule an ENCODER pass over the 256 committed tokens
     -> an ordinary extend-prefill batch (cached_len=cur_len, device_len=cur_len+256, causal=1)
     -> this OVERWRITES the same 256 KV slots with the causal K/V. No reallocation.
     -> then allocate the NEXT block's 256 slots and start a fresh canvas.
```

That last line is the elegant part: the canvas slots and the encoder slots for a block are **the same
slots**. The decoder writes bidirectional K/V into them ≤48 times; the encoder then overwrites them
once with the causal K/V that the *next* block will read. No separate scratch pool, no extra pages
beyond one block's worth of look-ahead.

#### B8.3 Why a new phase and not a `Proposer`

The spec framework's contract (`spec/base.py:72-73`) is *"the verify / accept / commit / KV-rollback
machinery is proposer-agnostic; only `propose` varies."* Block diffusion changes the **verify** side,
not the propose side: there is no target/draft pair, no acceptance against a reference distribution,
and the forward is non-causal. Wiring it as a `Proposer` would mean overriding every piece of the
machinery the abstraction exists to share. It is a peer of `_spec_decode_step`, not a client of it.

---

### B9. Staged plan

Each stage names what runs, what is verified, and how. Stages 1–3 are **CPU-only** and can be done
today with both cards busy.

**Stage 1 — reference oracle (CPU, no minisgl code).**
Run `DiffusionGemmaForBlockDiffusion` from the image's transformers on CPU with a tiny random-init
config (30 → 2 layers, hidden 64, vocab 512) and dump, for one 8-token canvas over a 16-token prompt:
per-layer decoder attention inputs/outputs, the self-conditioning output, and the full 48-step
trajectory (canvas ids, entropies, acceptance masks) at a fixed seed.
*Verified:* the fixtures exist and are reproducible. Store them under `tools/fixtures/diffusiongemma/`
— durable, in-repo, never tmpfs.

**Stage 2 — model + loader parity (CPU).**
Add `models/diffusion_gemma.py`, `register.py`, the loader fixes. Load the **real** checkpoint on CPU
(meta/CPU device, `int4` dequantised on the fly or just the fp16 tensors) and assert:
(a) every checkpoint key is consumed, none dropped;
(b) `encoder.layer_scalar == decoder.layer_scalar` for all 30;
(c) a single decoder forward over a fixed canvas with a fixed prefix, `causal=0`, matches HF's
`DiffusionGemmaDecoderModel` output to fp16 tolerance. Use a torch-eager non-causal attention so the
comparison isolates the model, not the kernel.
*Verified:* max abs diff on the final hidden and on the logits, per layer, printed.

**Stage 3 — the diffusion sampler in isolation (CPU).**
Port `EntropyBoundSampler` + `StableAndConfident` into `minisgl/diffusion/`, and replay Stage 1's
recorded logits through them. **Bit-exact** match on the acceptance mask and the argmax canvas at
every one of the 48 steps (the sampler is pure tensor arithmetic; there is no excuse for a tolerance).
*Verified:* `pytest`, no GPU.

**Stage 4 — one denoising step on GPU (eager, bs=1, sliding layers only).**
Wire `Batch.canvas`, the `causal=0` kernel dispatch, the widened ring, `forward_canvas`. Run a single
canvas step and compare against a torch-eager `[prefix|canvas]` bidirectional reference computed from
the same K/V. Restrict to a prompt short enough that the 5 full layers can be stubbed, or run with a
2-layer synthetic config. *Verified:* per-layer attention output diff.
**This is the first stage that needs a card.** It is also the stage where the `head_dim 512` gap
becomes blocking for the real checkpoint — see §B10.

**Stage 5 — full block, eager, bs=1.**
Scheduler `_diffusion_loop`, canvas slot manager, block commit, encoder re-pass. Generate 256 tokens
from a real prompt and compare the emitted text against HF's `model.generate` on the same checkpoint
at the same seed. Greedy-ish comparison is impossible (multinomial + randint); compare instead:
(a) `tokens_per_forward` distribution over ~20 prompts, (b) mean per-step entropy trajectory,
(c) human read of coherence. *Verified:* the acceptance-count-per-step curve tracks HF's within noise.

**Stage 6 — concurrency, then capture.**
Multi-request batching (each request at a different denoising step and a different prefix length —
this is the hard scheduling case), then cudagraph capture of the fixed 256-query step. *Verified:*
throughput panel, plus a capture-vs-eager output equivalence check at fixed seed.

---

### B10. Honest risk list

**R1 — `head_dim 512` on the 5 full-attention layers. The single biggest engineering risk, and it is
pre-existing.** No HIP kernel exists (`attention/rdna4.py:26`, dispatch macros at
`attn_kernels_hip.hip:287-295`), and the Triton fallback hard-codes `causal=True, window_size=(-1,-1)`
(`attention/rdna4.py:308-309`) so it **cannot serve a canvas at all**. Options, in order of
preference: (a) instantiate `HEAD_DIM=512` in the existing `attn_prefill_paged` template — the LDS
arithmetic looks feasible (`kBR/kBC = 16` for `HEAD_DIM >= 256`, `attn_kernels_hip.hip:56-57`, giving
`16 × (512+PAD) × 2 B ≈ 16.6 KiB` each for K and V inside the 64 KiB gfx1201 LDS) but this is an
estimate I have **not** validated by compiling; (b) split the 512-wide head into two 256-wide halves
and combine with a two-pass online softmax — correct but a real kernel change; (c) plumb a non-causal
mode into the Triton path. **Whatever the AR gemma4 port does about this, block diffusion inherits.**

**R2 — the self-conditioning matmul is a second LM head, per step.**
`softmax([256, 262144]) @ [262144, 2816]` ≈ 378 GFLOP/step/request, plus a 268 MiB fp32 softmax
transient and a 134 MiB fp16 logits tensor. At 48 steps that is ~18 TFLOP per 256-token block *on top
of* the ~2 TFLOP/step of actual model. **Mitigation that is mathematically exact:** carry the
self-conditioning state as the **soft embedding** `[256, 2816]` (1.4 MiB) instead of the logits
`[256, 262144]` (134 MiB), computing it immediately after the logits are produced, and fuse
`softmax(l) @ E` into a single streaming kernel that never materialises the probability matrix. The
freeze-on-finished semantics (`dg_generation:1063-1065`) commute with this change. **Do this from the
start; retrofitting it means re-plumbing the whole state carry.**

**R3 — the entropy is computed twice per step over the full vocabulary.**
`torch.distributions.Categorical(logits=l).entropy()` in `accept_canvas` (`dg_generation:437-438`) and
again in the stopping criterion (`:533-534`). Each is a full `[256, 262144]` fp32 softmax +
`x·log x` reduction ≈ another 268 MiB transient. Compute once, share. Then fuse it with the argmax and
the multinomial into one pass over the logits — an obvious and self-contained HIP kernel
(`sampler/` in `rdna4-hip-kernels` is the right home).

**R4 — non-causal attention over 256 queries: does the existing kernel actually go fast?**
`flash_prefill_paged` with `causal=0` computes `256 × (cur_len + 256)` score cells instead of the
causal half. That is correct (§A2.4, verified in the kernel source) but it has **only ever been
exercised on the fused-TiDAR path with a `mask_bias`**, at small query counts. I have not measured it
at `q=256, causal=0`. Risk is performance and occupancy, not correctness.

**R5 — VRAM at 2×16 GB, TP=2.** Checkpoint is 16.05 GiB on disk (≈1.2 GiB of that is the vision tower
we skip; `embed_tokens` alone is 262144×2816 fp16 = 1.38 GiB and is **not** quantized). Call it
~7.5 GiB of weights per card at TP=2. The SWA ring is the dominant KV term: 25 layers × 8 kv heads ×
256 × 2 B × 2(K,V) = **200 KiB per ring slot**, TP=2 → 100 KiB/card. Widening the ring from 1024 to
1280 slots takes it from **~100 MiB to ~125 MiB per running request per card** — a 25 % increase on
the AR sibling's already-large ring. Plus 5 full layers × 2 kv heads × 512 × 2 B × 2 = 20 KiB/token
(10 KiB/card at TP=2) of main pool for the full context. My estimate is **max_running_req in the
high single digits to low teens**, which is a real concurrency ceiling but not a blocker for bs=1
bring-up. I have **not** run `_determine_num_pages` (`engine/engine.py:875-961`) against this config —
that number should be produced before Stage 5, not guessed.

**R6 — scheduling heterogeneous denoising steps.** In a batch, request A may be at step 40 with a
1 200-token prefix while B is at step 3 with a 200-token prefix. The forward shape (`256 queries`) is
uniform, which is good, but the *temperature* differs per request (it is a function of `cur_step`),
the KV lengths differ, and requests exit the block at different steps. The temperature can be folded
into a per-request scale on the logits; the varying exit is the same "some rows finished" problem
`_spec_decode_step` already solves with `finished_denoising`-style freezing. I flag it because it is
the part of the design I have thought least hard about.

**R7 — latency shape is completely different and may disappoint.** A 256-token block costs *k*
forwards of 256 tokens each, where `k ∈ [1, 48]`. The AR sibling costs 256 forwards of 1 token each.
Block diffusion wins only if `k` is small and the 256-token forward is not 256× the cost of a 1-token
forward. Given this repo's standing finding that **serving is overhead-bound, not bandwidth-bound**
(`umc ≤ 27 % at every batch size`), a 256-token forward should be *much* cheaper than 256 decode
steps — which is exactly why this is worth doing. But `k` is entirely empirical and `confidence_threshold
= 0.005` is a *tight* bar (mean entropy over 256 positions below 0.005 nats). **Measure `k` on real
prompts at Stage 5 before believing any speedup projection.**

**R8 — no incremental streaming.** §A5. A 256-token block is opaque until it completes. If the serve
front-end promises OpenAI-style streaming, the options are (a) stream the whole block at once,
(b) stream the evolving argmax draft (`dg_generation:782-786`) which is *not* monotone and will
visibly rewrite itself, or (c) don't offer streaming for this model. This is a product decision, not
an engineering one, and it should be made before the API surface is written.

---

### B11. Flagged unknowns — things I could not settle

**U1 (mostly RESOLVED — see §A2.5) — the FA2 path only.** SDPA and eager provably agree
(`1.5e-8`), so the `sliding_window=` kwarg at `dg_modeling:462` is inert on both and the decoder is
full bidirectional. The residual unknown is **flash-attention-2**, where `sliding_window` *is*
forwarded to the kernel (`_flash_attention_utils._flash_attention_forward`, arg passed through at its
line 58 in the image) and, with `is_causal=False`, would apply a symmetric ±window on the 25 sliding
decoder layers. Given (a) the "attend fully" comment at `dg_modeling:1399-1401`, (b) the fact that the
`sliding_window = (sliding_window//2)+1` *"due to fa we set exclusive bounds"* correction at
`dg_configuration:106-108` is gated on `use_bidirectional_attention == "all"` and so does **not** fire
here, and (c) the two-backend agreement above, I read the FA2 behaviour as a **latent bug in the
reference**, not the intended semantics. The port should implement full bidirectional. Confidence:
high, not certainty — an FA2 A/B on a card would close it, and is worth 20 minutes at Stage 4.

**U2 — I have not verified that the port's Stage-2 numbers actually match.** Everything in Part A is
read from source and from the checkpoint's tensors; nothing in Part A has been executed end-to-end
against the real weights. In particular I have not confirmed that `Gemma4Model` reproduces
`DiffusionGemmaDecoderModel` numerically — only that the two are structurally identical line by line.
That is what Stage 2 exists to prove.

**U3 — `head_dim=512` LDS feasibility (R1 option a) is an arithmetic estimate, not a build.**

**U4 (RESOLVED, recorded as a trap) — `per_layer_config` is a derived, lazily-materialised field.**
`AutoConfig.from_pretrained(...).get_text_config().per_layer_config` reads as `None`, yet both
attention classes index it (`dg_modeling:289`, `:391`). It is populated by
`DiffusionGemmaTextConfig.__post_init__` only from the `global_head_dim` / `num_global_key_value_heads`
kwargs; constructing a text config **without** them yields `per_layer_config[i].num_key_value_heads is
None` and the model raises `TypeError: unsupported operand type(s) for //: 'int' and 'NoneType'` at
`dg_modeling:298` (hit and confirmed while building the §A2.5 fixture). Relevant only to anyone
building a synthetic DiffusionGemma for the Stage-1/2 oracles: **always pass both kwargs.** The port
itself is unaffected — minisgl derives the same geometry from the same two config keys
(`models/config.py:590-596`), cross-checked against the 25-of-30 `v_proj` census.

**U5 — the third EOS id.** `generation_config.json` lists `[1, 106, 50]`; `config.json` lists
`[1, 106]`. Token 50 is presumably a second turn terminator. `resolve_stop_token_ids`
(`scheduler.py:318`) reads the generation config, so this should come out right, but it is worth an
eyeball at Stage 5 — an unmatched terminator on a canvas model doesn't truncate one token late, it
emits a full extra 256-token block.

---

## Part C — what was built, what the doc got wrong, and what is left

Everything below was executed. Parts A and B above were written from source; this part is the
measurement, and it corrects Part A/B where the two disagree.

### C1. Landed

| stage | commit | evidence |
|---|---|---|
| model + loader + registration | `feat(diffusiongemma): register the block-diffusion head…` | `tests/diffusiongemma_build_test.py` PASS at TP=1 and TP=2: declared 805 == emitted 805, 0 missing / 0 extra / 0 mis-shaped / 0 mis-typed / 0 duplicated |
| numerical parity, both roles | `test(diffusiongemma): numerical parity for BOTH execution roles…` | `tests/diffusiongemma_parity_test.py` PASS: canvas logits rel_fro 4.3e-07 / 5.9e-07, encoder-role final hidden rel_fro 2.4e-07, self-conditioning and `soft_embedding` BIT-EXACT |
| entropy-bound sampler | `feat(diffusion): the entropy-bound sampler…` | `tests/diffusiongemma_sampler_test.py` PASS: 39-step replay against the reference with 0 mismatches on the accepted mask, sampled canvas, argmax canvas, entropies and the early-exit flag |
| canvas attention feasibility | `probe(diffusiongemma): the paged kernel does serve a non-causal canvas` | `tools/canvas_attention_probe.py` PASS on a 9070 — see §C3 |

§B8's central verdict is confirmed: **the decoder stack is reusable as is.** DiffusionGemma's
parameter set is the autoregressive sibling's 802 keys plus exactly three
(`model.self_conditioning.{pre_norm,gate_up_proj,down_proj}`), differenced by building both models
side by side, and one instantiated stack reproduces both the causal encoder and the bidirectional
decoder to fp32 round-off.

### C2. Where Part A/B is wrong

1. **§U4 is INVERTED for the shipped transformers.** The doc says to always pass `global_head_dim`
   and `num_global_key_value_heads` or `per_layer_config[i].num_key_value_heads` reads `None`. In
   transformers **5.14.1** — the version in the serve image — `per_layer_config` is already `None`
   on the real config and both attention classes read `config.global_head_dim` / `config.head_dim`
   directly. POPULATING `per_layer_config` is the trap in this version: it marks the config
   heterogeneous and every `config.head_dim` read raises
   `AmbiguousGlobalPerLayerAttributeError`. The doc's advice comes from the older scratchpad copy.

2. **Every line-number citation in Parts A/B is stale by roughly 60–70 lines** against the current
   tree (`scheduler.py:3895` is now `:3962`; `rdna4.py:637-692` is now `:643-698`). The seam table in
   §C4 carries current numbers.

3. **§B7.9(a) understates the loader break.** Narrowing `_GEMMA4_SKIP_PREFIXES` was necessary but
   not sufficient: the compressed-tensors `ignore` list is stored in the CHECKPOINT's key space
   (`model.decoder.layers.0.mlp.gate_proj`) and `_norm_ignore` stripped only the `language_model.`
   infix, so on this checkpoint **nothing in the ignore list matched at all** — the dense MLP, all
   30 routers and the self-conditioning block would have been built int4 against fp16 tensors. The
   symptom is ~90 modules declaring `weight_packed` where the loader emits `weight`, with nothing
   pointing at a namespace mismatch.

4. **§R1 (`head_dim 512`) is resolved in SOURCE but not in every IMAGE.** The 512 instantiations
   exist across `attn_decode` / `attn_hip` / `attn_prefill_paged` in the
   `rdna4-hip-kernels-gemma4serve` worktree. They do NOT exist in `rdna4-hip-kernels` main, and the
   `/opt/kernels` baked into `minisgl-rdna4:lean` refuses them
   (`attn_prefill_paged split: head_dim 512 unsupported (64/128/256)`). `_HIP_HEAD_DIMS` already
   lists 512, so `_no_hip_kernel`'s "the loaded package is older than the engine" note is exactly
   right — but only a canvas/verify path reaches it, which is why a bs=1 autoregressive serve with a
   cold prefill never noticed. **Any canvas work must run on an image built from the gemma4serve
   kernels.**

5. **§B7.3's fix is understated.** The ring aliasing is not a partial overlap: at `R = W` with a
   256-token canvas, **256 of 256** canvas slots land on a slot inside the window they must read.

6. **§B8.2's "a canvas needs a `custom_mask`" is wrong, and the correction saves memory.** The full
   layers reach `causal=0` today only when `metadata.custom_mask` is set, but the KERNEL takes
   `causal` independently: `causal=0, mask_bias=None` is dense bidirectional attention, measured
   bit-identical to passing an all-zero mask. A canvas therefore needs no `[256, cur_len+256]` fp32
   bias at all — 2.2 MiB per layer per step at a 2000-token prefix.

### C3. Measured on the card (`tools/canvas_attention_probe.py`, RX 9070, `minisgl-rdna4:gemma4`)

Geometry is DiffusionGemma at TP=2: sliding 8 q / 4 kv at head_dim 256, full 8 q / 1 kv at head_dim
512, window 1024, canvas 256, prefix 2000, softmax scale 1.0.

```
[A] sliding layers, causal=0 + sliding_window=0 over a [Wp | canvas] RING block table
      vs dense bidirectional            rel_fro 1.46e-03   (bf16 accumulation noise)
      the SHIPPED causal=1 ring path    rel_fro 6.02e-01   -- a different model
      canvas query 0 sees canvas key 255 (max|abs| 0.0 between two identical queries)
[B] full layers, causal=0, mask_bias=None, head_dim 512
      vs dense bidirectional            rel_fro 1.34e-03
      an all-zero mask_bias is redundant (bit-identical)
[C] ring stride   R = 1024 -> 256/256 canvas slots alias a window slot
                  R = 1280 -> 0
[D] cost at q=256   sliding (1280 keys)  causal=0  428.0 us   causal=1  401.7 us   1.07x
                    full    (2256 keys)  causal=0 1759.1 us   causal=1 1699.8 us   1.03x
                    per denoising step (25 sliding + 5 full) = 19.5 ms
                    48 steps = 936 ms for a 256-token block = 3.66 ms/token of attention
```

**§R4 is answered: non-causal costs essentially nothing extra** (1.03–1.07x, not the 2x the doubled
score-cell count would suggest — the kernel is not score-bound at this shape).

**§R7 now has a number, and it is the thing to watch.** 3.66 ms/token of attention *alone* at the
48-step worst case, against the autoregressive sibling's 22.8 ms/token *total* (43.9 tok/s). There is
headroom, but only because 48 steps is the ceiling; the 5 full-attention layers cost 8.8 ms of the
19.5 ms step at 1 kv head per rank, which is ~5.4 TFLOP/s of a card that does far more, so that
kernel shape is the first thing to look at if the step cost matters. **This is an isolated kernel
bench, not a serve measurement** — per this repo's standing rule it bounds feasibility and predicts
nothing about end-to-end tok/s. `k`, the realised steps per block, is still entirely unmeasured and
is what decides whether block diffusion wins.

### C4. What is left — the execution mode, with current line numbers

| # | seam | file:line | state |
|---|---|---|---|
| 1 | `Req.complete_one()` hard-codes `+1` | `core.py:158-160`, applied at `engine/engine.py:1087-1088` | a canvas step must not go through `Engine.forward_batch` |
| 2 | multi-query staging | `scheduler.py:3962` (`req.device_len = c0 + len(d) + 1`) | template exists; canvas wants `c0 + 256`, held across all <=48 steps |
| 3 | non-causal on the 5 full layers | `attention/rdna4.py:409, 429` | **works today** — pass `causal=0` with `mask_bias=None` (§C3 B) |
| 4 | non-causal on the 25 sliding layers | `attention/rdna4.py:590-598` | `_swa_prefill_paged` hardcodes `causal=1, sliding_window=W, mask_bias=None`; the kernel accepts the other combination and it is correct (§C3 A), so this is ~20 lines of Python, not a kernel change |
| 5 | ring stride widening | `engine/engine.py:210-215` AND `engine/engine.py:792-796` | two independent `spec_block` computations that must move in lockstep; both need `+ canvas_length` |
| 6 | SWA read window capped at `W` | `attention/rdna4.py:722, 729` (`cnt = min(S, W)`) | cannot express `Wp + 256`; `attention/hip.py:414-426` (`_fill_swa_verify_static`) already builds exactly the `[window | new]` row shape needed |
| 7 | forward entry point | `engine/engine.py:1153-1168` | `forward_verify` calls `model.forward()`, not `forward_canvas`; needs a sibling |
| 8 | LM-head reduction | `layers/embedding.py:134` | done — `forward_canvas` uses `logits_all_rows` |
| 9 | slot lifecycle | `scheduler/recurrent_slots.py:55-90` | pattern to copy: uid-keyed, allocate-once, idempotent free, `KeyError` on a miss |
| 10 | page rollback | `scheduler.py:4406-4415` | must become a no-op until block end (the 256 slots are reused across all steps) |
| 11 | **no GPU harness exists** | — | nothing in `tools/` or `tests/` boots a model + paged KV pool outside the scheduler; the closest are `tools/swa_ring_roundtrip.py` (kernels only, no `Context`) and `tools/kv_fp8_perhead_check.py` (pool only, no model) |

**Seam 11 is the real blocker, and it is a design decision, not a missing capability.** Seams 1–10 are
each small and each has a working template. But there is nowhere to *drive* them from: a canvas
generation needs an encoder prefill (pages, radix, positions), then <=48 canvas steps that reuse the
same slots, then a commit and a re-encode — and the only thing in the tree that can sequence that is
`Scheduler.run_forever`. The two options are

  (a) **a bespoke boot** that reproduces `Engine.__init__`'s ordering by hand (`Context(page_size)`
      -> `set_global_ctx` -> `ctx.page_table` -> main `MHAKVCache` -> `ctx.swa_ring_stride` -> SWA
      `MHAKVCache` -> backend -> model; the backend snapshots the stride at `rdna4.py:131`, so the
      order is load-bearing). Fast to a first generation, but it is a second copy of the engine's
      sizing logic that will drift; and

  (b) **`Scheduler._diffusion_loop` + `CanvasManager`** as a peer of `_spec_decode_step` — the shape
      §B8.3 argues for and the one that ends up in production.

(b) is right and (a) is a trap: the doc's own §B7.6 lifecycle work (`ENCODE -> DENOISE xk -> COMMIT`)
has to be written either way, and writing it against a throwaway harness means writing it twice.
Recommendation: go straight to (b), in the order 4 -> 6 -> 5 -> 3 -> 7 (all mechanical, all with
templates, verifiable by a single-step canvas forward against the CPU parity fixture), then 1/2/9/10
as one `CanvasManager` change. Nothing about the autoregressive path needs to move.

---

## Part D — the execution mode, built and measured

Part C ended at a blocker: seams 1-10 were each small, but there was nowhere to drive them from.
That was resolved by taking option (b) — a scheduler loop, not a bespoke harness. This part records
what the working system measures and what Parts A/B still get wrong.

### D1. It generates, and it is faster than the autoregressive sibling

`tools/diffusiongemma_generate.sh`, TP=2, bs=1, greedy, `minisgl-rdna4:gemma4`, the same harness and
worktree for both models:

| | 256-token completions | short completion |
|---|---|---|
| block diffusion | **72.3 / 81.1 tok/s** (13.8 / 12.3 ms/tok) | 23.3 tok/s (99 tok) |
| gemma-4 (AR) | 44.1 / 44.2 tok/s (22.7 / 22.6 ms/tok) | 30.4 tok/s (117 tok) |

**1.64-1.84x on full blocks.** The autoregressive sibling lands exactly on its documented 43.9
tok/s, which is the guard that mattered most. Output is fully coherent — see the results file the
harness writes.

**The partial block is the honest cost shape.** 99 tokens still pays for a full 256-token canvas
over 12 steps, so a short answer is ~2x WORSE per token than the AR sibling. Block diffusion wins on
long outputs and loses on short ones; nothing in §B or §C predicted the sign of that.

### D2. §R7 answered: k is 12-19, not 48

```
blocks=3   k: min=12  median=17  max=19  mean=16.0   of a possible 48
forwards per emitted token = 0.078          (the autoregressive sibling is exactly 1.000)
mean entropy at commit = 0.0007 / 0.0009 / 0.0049   against the 0.005 threshold
```

**12.8x fewer forwards**, at ~256x the query count each. Every block exited on the CONFIDENCE
criterion, not on the step cap — so §R7's fear ("`confidence_threshold = 0.005` is a *tight* bar")
is unfounded on real prompts, and §C3's 48-step worst case (3.66 ms/token of attention alone) does
not occur. k is scraped from a per-commit `[canvas]` log line the scheduler emits, because a block
emits all its tokens at once and tok/s therefore cannot show k at all.

### D3. Further corrections to Parts A/B/C

7. **§B8.1's file plan is wrong in two places.** There is no `scheduler/canvas_slots.py` and no
   `diffusion/{stopping,state}.py`: the stopping criteria and the per-request state are 40 lines
   that belong with the sampler they are a property of, and the canvas *slots* are ordinary
   page-table slots owned by the CacheManager — giving them their own manager would have made a
   committed block a special region instead of the ordinary radix-cacheable prefix it is. What is
   real is `scheduler/diffusion.py` (the loop + a `CanvasManager` that owns only DENOISING state)
   and `diffusion/sampler.py`.

8. **§B8.2's `_canvas_block_step` is missing its most important line.** It ends at "commit the
   block" and never re-encodes it. The KV left in the canvas slots is the decoder's BIDIRECTIONAL
   K/V, which this model never serves from; the committed block must be re-run through the CAUSAL
   encoder, in place, before the next block reads it. §B8.2's own prose has this ("that last line is
   the elegant part") but the pseudocode does not.

9. **§B7.6 understates the encoder problem.** The issue is not only that `_process_last_data`
   assumes one token per request — it is that the encoder pass must not SAMPLE AT ALL. Its logits
   are discarded by the reference. Routed through `forward_batch` it samples a token, appends it to
   the request and emits it, and the result is a serve that looks like it works and prepends one
   junk token to every generation.

10. **`attention/hip.py` needs no canvas change**, despite constructing `RDNA4Metadata` at four
    sites. All four are CAPTURE-path static builders (`_decode_metadata_static`,
    `_verify_metadata_static`, `_fused_verify_metadata_static`, `_ddtree_verify_metadata_static`);
    `HIPAttnBackend` does not override `prepare_metadata`, so the eager canvas path runs through
    `RDNA4Backend`'s. It becomes a real gap only when the canvas step is captured (§B7.7).

### D4. What is still open

* **cudagraph capture (§B7.7).** Eager only. Per this repo's standing rule that is not "done", and
  the canvas is an unusually good capture target: fixed query count, fixed batch size, only the
  prefix length varies — the same shape `_fill_swa_verify_static` already handles.
* **Concurrency (§R6).** Measured at bs=1 only. The forward shape is uniform, but the per-request
  TEMPERATURE differs (it is a function of that request's own step index), so a batched step needs a
  per-row scale; the code currently applies it per request in the sampler, which is correct but
  serialises the fp32 softmax over `[canvas, vocab]` per request.
* **SWA-radix.** Turned OFF for the canvas phase. Its window snapshot is taken at autoregressive
  commit points and addresses the ring at the pre-canvas stride; re-validating it against a canvas
  is its own piece of work, and leaving it on would silently seed a stale window.
* **Chunked prefill and structured output** are REFUSED with a reason rather than silently ignored.
* **§U5 (the third EOS id, 50)** is now moot in practice — every test block terminated correctly on
  the resolved EOS set — but has not been isolated.
