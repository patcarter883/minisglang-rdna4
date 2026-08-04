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

* ~~**cudagraph capture (§B7.7).** Eager only.~~ **DONE, and it buys nothing — see §D6.**
* ~~**Concurrency (§R6).** Measured at bs=1 only.~~ **Measured 1/2/4; it saturates — see §D7.**
* **SWA-radix.** ~~Turned OFF for the canvas phase.~~ **Now ON, the stated reason was wrong, and FULL
  prefix reuse is proven lossless (§D5). PARTIAL prefix reuse is a KNOWN DEFECT: narrowed to the 23
  tokens computed after a restore, with the snapshot, the reused pages and the extend kernel each
  EXCLUDED by measurement (§D5.1). Mechanism not yet established; not fixed;
  `MINISGL_SWA_RADIX=0` is the workaround.**
* **Chunked prefill** ~~and~~ is now wired for the encoder pass (§D5), and it had to be: with the
  prefix cache on, 15 out of 16 prompts arrive chunked. **Structured output** is still REFUSED with a
  reason rather than silently ignored.
* **§U5 (the third EOS id, 50)** is now moot in practice — every test block terminated correctly on
  the resolved EOS set — but has not been isolated.

### D5. SWA-radix on the canvas: the stride argument was wrong, and the real bug was worse

§D4 left SWA-radix off for the canvas phase on this reasoning: *"its window snapshot is taken at
autoregressive commit points and addresses the ring at the pre-canvas stride."* **There is no
pre-canvas stride.** `Engine.__init__` computes the ring stride exactly once, at boot, as

```
swa_ring_stride = sliding_window + _swa_ring_block(model_config, spec_config)
                = sliding_window + max(num_draft + 1, canvas_length)
```

publishes it on `ctx.swa_ring_stride`, and every reader takes that one number —
`rdna4.py`'s store/gather/decode, `hip.py`'s captured builders, and `SWAWindowSnapshotter`. On
DiffusionGemma that is **1024 + 256 = 1280 in every phase**: the prompt encoder pass, every denoising
step, the block re-encode, and the finish commit. The boot log says so in one line
(`SWA ring KV: 25 layers x 3840 slots (window=1024, stride=1280, ...)`). Folding the two drifting
stride computations into `_swa_ring_block` was ccd6b198 — the very commit §D4 cites — so the hazard
it describes had already been removed when it was written. A snapshot cloned in one phase reads back
byte-identically in another, and the snapshotter needed no stride change at all.

The instinct was right and the mechanism was wrong, which made the risk assessment wrong in both
directions. Three separate findings:

**1. The real gap was a missing RESTORE, and it was live.** The canvas loop does not run through
`_finish_prepare` (so `_restore_swa_states` never fired) or `_process_last_data` (so
`_maybe_capture_swa_state` never fired at the prompt commit) — but it DOES reach
`_free_req_resources`, which attaches a window snapshot to a finished request's radix node. So with
SWA-radix at its default (ON, 12e18adb), a canvas serve produced snapshots, matched against them
(`match_prefix` caps to a snapshotted node), reported `cached_len > 0` — and never seeded the ring.
That is precisely the stale window §D4 feared, arrived at by a different route, and it is the one
thing the `MINISGL_SWA_RADIX=0` in the harness was actually protecting against. Fixed by two calls
in `scheduler/diffusion.py`: `_restore_swa_states` in `_canvas_forward` (prefill batches only) and
`_maybe_capture_swa_state` on the handle `_canvas_encode` was already discarding.

**2. Chunked prefill was not a "not yet wired" nicety — it was a hard blocker.** The
snapshot-capable radix splits EVERY prefill at its last page boundary
(`PrefillAdder._add_one_req`, `is_recurrent_radix`) so a window snapshot lands page-aligned. So a
prompt whose length is not a multiple of `page_size` — 15 out of 16 prompts — arrives as
`[aligned body][sub-page tail]`, the body is a `ChunkedReq`, and `_canvas_encode` raised. **Turning
SWA-radix on without this would have killed the scheduler on the first real request**, which is
exactly what the first validation run did. The refusal's own suggested remedy ("raise
--max-extend-tokens") could not have helped: the split is page alignment, not budget. A chunk of the
encoder pass needs three lines — advance it to its own chunk end (the AR path gets this free from
`forward_batch`'s `complete_one`, which `forward_verify` deliberately does not do), stash the
page-aligned window, and skip the cache/decode handoff. The canvas still starts only from a complete
prefix, which is what the refusal was protecting.

**3. The finishing block was published to the prefix cache as BIDIRECTIONAL K/V.** `_canvas_commit`
re-encoded a committed block causally only when the request CONTINUED; on the finishing block it
returned early, and `_free_req_resources` then inserted the whole sequence into the radix tree. Both
the main-pool full-attention KV and the SWA window snapshot for that tail therefore held the
decoder's bidirectional K/V — which this model never serves from. **This is a prefix-cache
correctness bug independent of SWA-radix**: it poisons plain radix reuse of a completed conversation,
i.e. the multi-turn case prefix caching exists for. The fix is to re-encode the finishing block too,
one causal forward per request at the end.

**Block diffusion has no greedy mode, so the gate needed a seed.** The AR losslessness gate (041036ed)
turns on `temperature 0 / top_p 1 / top_k 1`. That pins nothing here: a canvas starts as uniform noise
over the whole 262144-token vocabulary and every denoising step draws a multinomial, so two identical
requests to one serve return different text and byte-identity is vacuous by construction.
`SamplingParams.seed` (new, `None` = the previous behaviour on every path) installs a per-request
generator, making a block's whole trajectory a deterministic function of (prompt, seed). The floor is
also a different shape from the AR one: a block commits all 256 positions at once, so one flipped
denoising step rewrites the whole answer — there is no "stable for the first k tokens" regime.

**The partial-hit cell needs two serves.** Inside one serve the same prompt cannot be measured cold
and then partially-hit: measuring it cold inserts it, so the second request is a FULL hit. The AR
gate's workaround — a cold reference on a different salt — rests on the answer being
salt-independent, which is true of an AR recall tail and false of a canvas. So the gate runs the same
`--run-id` twice at `MINISGL_SWA_RADIX=1`, once with `--no-warm` (the `partial` step is then a genuine
cold measurement of that exact prompt, hit=0) and once without, and the `cold` step — identical in
both legs — is the cross-serve determinism control that decides whether the diff is admissible at
all. `tools/swa_radix_canvas_test.sh` + `tools/swa_radix_canvas_verdict.py`.

#### Measured — DiffusionGemma-26B-A4B-INT4, TP=2, bf16 KV, seed 20260804

The floor first, because nothing below means anything without it. A SEEDED canvas is reproducible
against itself, 4/4 cells, 3/3 requests each:

```
REPRO case=short  max_tokens=16  distinct=1 STABLE      REPRO case=xlong  max_tokens=16  distinct=1 STABLE
REPRO case=short  max_tokens=32  distinct=1 STABLE      REPRO case=xlong  max_tokens=32  distinct=1 STABLE
```

That is a stronger floor than the AR path's (which was 2 distinct at 16 tokens and 5 at 32 on an
open-ended probe), and it is a property of the seed, not of the model: the same cells UNSEEDED return
a different answer every time.

**FULL prefix hit — LOSSLESS, 4/4 byte-identical, with the reuse demonstrated from /metrics:**

| case | max_tokens | cold sha | full-hit sha | hit tokens | prompt | verdict |
|---|---|---|---|---|---|---|
| short | 16 | d9bcfb25465a378b | d9bcfb25465a378b | **192** | 204 | IDENTICAL |
| short | 32 | fdb8ef084105b614 | fdb8ef084105b614 | **192** | 204 | IDENTICAL |
| xlong | 16 | f0b591bcab4424f6 | f0b591bcab4424f6 | **3184** | 3190 | IDENTICAL |
| xlong | 32 | 03e7fd94dcb7eb35 | 03e7fd94dcb7eb35 | **3184** | 3190 | IDENTICAL |

`cold` reads hit=0 and `full`/`full2` read hit=192/3184 on the same prompt in the same process, so the
identity is the cache and not the harness; the serve log carries 20 `SWA-radix HIT: ... seeded ring
window` lines over the run. The repeat control (`full` vs `full2`) is IDENTICAL 4/4. The probe is not
degenerate — all four `cold` shas differ from each other and from their `partial` counterparts.

**PARTIAL prefix hit — OPEN, and it DIVERGED on the long prefix. Not yet a losslessness verdict.**

```
case=short max_tokens=16  INADMISSIBLE  A.partial hit=192  B.partial hit=192   (same measurement twice)
case=short max_tokens=32  INADMISSIBLE  A.partial hit=192  B.partial hit=192   (same measurement twice)
case=xlong max_tokens=16  DIVERGED      A.partial hit=3168 B.partial hit=0     [cold control HELD]
case=xlong max_tokens=32  DIVERGED      A.partial hit=3168 B.partial hit=0     [cold control HELD]
```

The two `short` cells are inadmissible for a reason worth writing down: `align_down(len(P))` and
`align_down(len(B1))` are both 192 there, so B1's own snapshot already sits at the shared boundary and
`--no-warm` does not produce a cold reference — the leg B request hit too. The construction needs
`len(P)` and `len(B1)` to straddle different page boundaries.

The two `xlong` cells are admissible (the `cold` control sha is identical across the two serves, so
cross-serve output IS reproducible here) and they DIVERGED. **It is a DEFECT, not noise, and the
first explanation offered here was wrong** — see §D5.1.

#### TTFT and tok/s — the win is real but SMALL, and the reason is structural

| case | cold TTFT | full-hit TTFT | saving |
|---|---|---|---|
| xlong (3190 tok prompt) | 10.14 s / 10.13 s | 9.38 s / 9.38 s | **0.76 s (7.5%)** |
| short (204 tok prompt) | 8.48 s / 8.84 s | 8.37 s / 8.75 s | 0.11 s (1.2%) |

The AR sibling's 12.4x TTFT on the same prefix does NOT transfer, and it never could have: a
block-diffusion request is opaque until its block commits, so its "TTFT" is `prefill + k denoising
steps`, and at k≈16 the denoise is ~9 s against a ~0.8 s prefill. **Prefix reuse removes 100% of the
prefill and 0% of the denoise**, so the ceiling on this win is the prefill's share of the block —
~8% at a 3.2k prompt, ~1% at 200 tokens. It grows with prompt length and shrinks with k; it is not
the lever block diffusion needs, which is the ~85%-non-compute canvas step.

**AR guard, re-run with SWA-radix ON: `256 tokens in 5.79s = 44.2 tok/s = 22.6 ms/token`** — exactly
the §D1 baseline (44.1 / 44.2). The autoregressive sibling is unaffected. (The same leg's short
completion reads `117 tokens in 5.67s = 20.6 tok/s`; that is the first request after boot and is not
the guard — §D1's own 117-token figure, 30.4 tok/s, was also a warmed one.)

### D5.1 The partial-hit divergence is a DEFECT — located to the state, then MIS-located to the kernel

§D5 offered two causes for the partial-hit divergence and leaned on the wrong one. **"Prefill-shape
rocBLAS M-dependence" is not available as an explanation on this engine at all:** the dense path does
not use rocBLAS, and its kernels are M-invariant by construction. `layers/minv.py::minv_linear`
exists precisely so a chunked prefill is bit-identical to a single pass — verified 0.0 across all 40
CCA layers by `tools/cca_chunk_bisect.py` — and Gemma4's router and lm_head both go through it. So a
chunked-vs-single difference cannot come from numerics, and the divergence is a bug.

**The instrument.** Text is the worst possible place to debug this: it has been through 30 layers, a
16-step denoising trajectory and an entropy bound that SORTS 256 values, so one flipped bit anywhere
rewrites the whole answer and no amount of reading it says where. `kvcache/state_digest.py` +
`tools/canvas_state_bisect.py` are the `cca_chunk_bisect` move applied one level down, to STATE
instead of activations, on the observation that

> a prefill is correct iff the KV it leaves behind is bit-identical to a cold prefill's,

because decode, the canvas and the sampler are all pure functions of it. `MINISGL_STATE_DIGEST=1`
makes `_canvas_encode` sha256 the KV it just produced, per layer, per pool, per 256-position segment;
two serves (partial hit / same prompt cold) are then diffed cell by cell.

**Measured.** Two controls first, and both hold: the full-hit request's state is byte-identical to
its cold reference at all 190 cells (`boundary 3189 — IDENTICAL`), and the cold request's state is
byte-identical ACROSS the two serves — so cross-serve KV determinism is not in question. Then the
partial hit (match at 3168), at boundary 3191, segmented by absolute position:

```
seg     0 ..  2816   0/5 .. 0/30 cells differ      <- the whole REUSED span: IDENTICAL
seg  3072:3191      29/30 cells differ             <- the only segment that moves
  the one cell that still MATCHES inside it:  pool=swa layer=0
```

**That pair of facts is the whole finding.**

*The cache is EXACT.* Thirteen of fourteen segments — every position below 3072, in both pools, at
every layer — are byte-identical. The reuse boundary is 3168, so the entire restored window and the
entire reused page span are proven correct. Segment `3072:3191` straddles the boundary, but positions
`[3072,3168)` have exactly the same provenance as `[2816,3072)` (restored window / reused pages), and
those match — so the divergence lives in `[3168,3191)`: the 23 tokens the forward COMPUTED after the
restore. **Not the snapshot, not the radix pages, not the tokens.**

*And it is the SLIDING layer that goes first.* `pool=swa layer=0` matches even inside the differing
segment. The first sliding layer's stored K/V is a pure function of the token embeddings and
positions — no attention output feeds it — so its matching proves tokens, positions and embeddings
are right for the new span too. Layer 1 differing means layer 0's ATTENTION OUTPUT differs. Same
queries, same keys, same values, different result.

That also explains the shape of the symptom §D5 misread as a numerics tell: the partial-hit text
being an exact PREFIX of the cold text is what a wrong attention result over a handful of positions
looks like after an argmax canvas, not what a uniform 1-ULP perturbation looks like.

**The obvious suspect was the BC front-pad, and it is NOT guilty.** A full hit extends 5-7 tokens
directly from the restored boundary in ONE forward; a partial hit page-splits into a middle chunk
`[3168,3184)` and a tail `[3184,3191)`, so the first forward after the restore is a 16-token extend at
front-pad `(3168-1024) % 32 == 0` against the full hit's `(3184-1024) % 32 == 16`. Every BC-aligned
case in `tools/swa_prefix_extend_validate.py` happened to land on pad ∈ {2, 16, 18} — the residue is a
property of whichever `(L, W)` pair the case picked — so **pad 0 had never been exercised**, and pad 0
is exactly what a page-aligned radix boundary produces.

That gate now covers it. `case_production` reproduces `_swa_prefill_extend` line for line (zero-KEY
pad, zero-QUERY front, drop `front` rows — the BC-aligned cases above front-pad with REAL keys, which
is a different buffer and not the one that ships), plus a sweep of all 32 BC residues and 11 chunk
lengths, run at the true Gemma4 sliding geometry (`HQ=8 HK=4 D=256 W=1024` per rank at TP=2):

```
PROD full-hit tail             (L=3184,M=5, pad=16)   bit-identical=True   max|cold-ext|=0.000e+00
PROD partial-hit chunk1        (L=3168,M=16,pad=0)    bit-identical=True   max|cold-ext|=0.000e+00
PROD partial-hit chunk2        (L=3184,M=7, pad=16)   bit-identical=True   max|cold-ext|=0.000e+00
pad sweep    r = 0..31 at M=16     pads that are NOT bit-identical: none
chunk sweep  M = 1..128 at pad=0   chunk lengths that are NOT bit-identical: none
```

**46/46 bit-identical, including the exact failing shape.** So the extend kernel reproduces a cold
prefill byte for byte at the shape the serve runs, and the §D5.1 conclusion above — "located to the
post-restore extend" — is WRONG. Recorded rather than quietly amended, because the gate that would
have caught the over-claim is the same gate that had the pad-0 hole in it.

**What is now excluded, each by measurement rather than argument:**

| candidate | excluded by |
|---|---|
| snapshot clone/restore round-trip | `tests/swa_window_canvas_stride_test.py`, max abs delta 0.0 |
| restored window CONTENT over `[2167,3168)`, all 25 sliding layers | state digest, 13/14 segments identical |
| reused radix PAGES over `[0,3072)`, all 5 full layers | state digest, same |
| tokens / positions / embeddings for the new span | `pool=swa layer=0` matches inside the differing segment |
| the sliding extend kernel, every BC residue and chunk length | this gate, 46/46 at production geometry |
| chunked-vs-single-pass numerics | the dense path is M-invariant by construction, not rocBLAS |

**What remains.** The divergence is in `[3168,3191)` — the tokens computed after the restore — with
correct inputs and a correct kernel, which is a contradiction, so one of the "correct"s is measured
over the wrong span. The leading gap is structural in the instrument: the digest is taken at the END
of the prefill, where `cached_len` has advanced, so a window-relative span **excludes the first
`device_len - cached_len` positions the earlier chunk actually attended** — 23 positions here,
`[2144,2167)`. `kvcache/state_digest.py` now digests `UNDER=64` positions past the window edge and
emits per CHUNK as well as per commit, which closes that blind spot; the run that would use it has
not completed (see below). The other live candidate is the FULL-attention paged extend over the
reused pages, which the sliding-only gate above does not model at all.

**STATUS — NOT FIXED, and deliberately not patched blind.** Full-prefix reuse on the canvas is proven
lossless. Partial-prefix reuse is a KNOWN DEFECT whose mechanism is NOT yet established: the previous
address was falsified, and reshaping the extend path on a hypothesis the gate contradicts would be
worse than leaving it described — the autoregressive models ride that same path. A block-diffusion
deployment that shares long prefixes across requests should set `MINISGL_SWA_RADIX=0`; full-prefix
reuse (the same prompt twice) is unaffected and remains lossless.

**A note on how the last run died, because it is the standing rule and it still bit.** The follow-up
per-chunk bisect crashed with `NameError: _SOFT_EMBED_CHUNK` — not a defect in any of this, but
another agent's mid-edit `models/diffusion_gemma.py` being read by a serve that mounted the SHARED
worktree. Any re-run of `tools/swa_radix_canvas_locate.sh` must mount an isolated `git worktree`,
exactly as CLAUDE.md's source-isolation rule says and as the earlier legs of this investigation did.

---

## D6. Cudagraph capture: byte-identical, and worth 0.3%

The canvas is captured (`GraphRunner.capture_canvas_graphs`, `HIPAttnBackend.init_canvas_capture`,
`Engine.forward_canvas`). §D4 called it "an unusually good capture target" and it is — the graph does
exactly what a graph is supposed to do. It just turns out that what a graph does is not what this
model needs.

### D6.1 Correctness first: it is bit-identical, at every captured batch size

Every serve verifies its own capture. The first two replays **at each captured batch size** also run
the eager forward on the same inputs and compare the whole `[bs*256, 2816]` backbone hidden state:

```
[canvas-graph] REPLAY #1 engaged, bs=1 check 1/2 (qlen=256, T=256):
               graph vs eager max|delta|=0.000e+00 over (256, 2816) BIT-IDENTICAL
```

`0.000e+00`, not "close" — the captured step is the same computation, not an approximation of it.
Two replays rather than one because the first step of a block carries a ZERO self-conditioning
signal and would not exercise that input; per BATCH SIZE rather than per serve because every
captured bs has its own `cu_seqlens_q`, its own `[bs, W+256]` ring block table and its own `bs*256`
store-slot vector. The first cut of this gate checked bs=1 four times and bs=2/3/4 not at all — a
concurrent serve's first canvas steps happen while the other requests are still prefilling — which
is precisely the shape of bug it exists to catch, so the gate is now keyed on bs.

This matters more than usual here because **block diffusion has no greedy mode**: the canvas is drawn
from noise and every step draws a multinomial, so cross-boot text identity is unobtainable in
principle (and capture itself perturbs the process RNG offset by reserving a generator state). The
in-process comparison is the only byte-identity claim available, and it is a stronger one than
matching text — it compares the computation rather than a sample from it.

### D6.2 It removes 97% of the launch cost and 0.3% of the step

`MINISGL_CANVAS_TIMING=1` splits each denoising step. Marginal cost per step (differenced between
report points, so the cumulative-average smear is removed), bs=1, TP=2, same session and harness for
both legs:

| | fwd_issue | fwd_tail | sampler | soft_embed | **step** |
|---|---|---|---|---|---|
| eager (`GRAPH_BS=0`) | 30.8 ms | 123.3 ms | 11.5 ms | 16.2 ms | **181.6 ms** |
| captured | **0.8 ms** | 152.9 ms | 11.6 ms | 15.9 ms | **181.1 ms** |

`fwd_issue` is the host wall spent inside `forward_canvas` issuing work; `fwd_tail` is the GPU work
still outstanding when the host stops issuing. Capture collapses the launch loop **30.8 → 0.8 ms, a
97% reduction**, and the step does not move (−0.3%, i.e. a wash; three consecutive 10-step windows in
the captured leg read 181.2 / 181.1 / 181.1 ms, so this is not noise-limited).

**That is the finding.** Those 30.8 ms of host launch were entirely overlapped with GPU work. There
was no launch gap to reclaim, because the GPU is saturated — which is what the independent rocprofv3
measurement says from the other side (gfx activity 100% median; bridging fp16→bf16 to engage the
native tail kernels removed 38% of all dispatches for +3.2%). Two instruments, two methods, same
conclusion. Anyone who re-derives "85% of the step is not compute" from a roofline estimate should
read this table before acting on it.

### D6.3 The step attribution, cross-checked

The same table, against the profiler's phase breakdown of the same workload:

| | this timer | rocprofv3 |
|---|---|---|
| forward (backbone + lm_head + softcap) | `issue+tail` = **154.1 ms** | 127.7 + 23.2 + 3.1 = **154.0 ms** |
| soft_embedding | **16.2 ms** | **15.3 ms** |
| sampler | 11.5 ms | (inside "sampler+host" ~17 ms) |
| step | 181.6 ms | ~186 ms |

0.1% apart on the forward, from a `torch.cuda.synchronize()`-bracketed host timer and a kernel
tracer respectively. The step is **GEMM-efficiency bound inside the backbone**, and neither dispatch
count nor bandwidth (umc ~24%) nor power (234 W median of a 320 W cap) is the constraint.

### D6.4 What capture is still for

It stays, for three reasons that are not tok/s. It is required by this repo's standing rule that
eager-only is not "done". It is proven byte-identical, so it costs nothing to carry. And its value is
*conditional on the rest of the stack*: the 30.8 ms of launch it removes is currently hidden behind
127.7 ms of backbone GEMM, but the same graph removes the same 30 ms from whatever the residual
becomes — so as the backbone shrinks, capture's share grows. It is item 4 of a stack, banked early.

Two implementation notes worth keeping:

* **The LM head is deliberately OUTSIDE the graph** (`forward_canvas_hidden` / `canvas_logits`). It
  is ~4 launches of ~5000, but its output is `[256, 262144]`; capturing it would pin ~1 GiB of
  vocab-wide fp16+fp32 transients in the graph's private pool **per captured batch size** on a 16 GB
  card. The captured region ends at the final norm and hands back `[bs*256, 2816]` = 1.4 MiB.
* **Exact bs match, no dummy padding**, unlike the decode and K+1-verify families. Padding a bs=3
  canvas step up to a captured bs=4 would push an extra 256-token canvas through 30 layers to avoid
  ~1 ms of launch — the wrong trade by two orders of magnitude. Uncaptured sizes fall back to the
  eager forward, which is lossless.

## D7. Concurrency: it saturates at bs≈2, and it is not a route to 300 tok/s

§D4 recorded a suspicion that per-request temperature "serialises the fp32 softmax". **That is not a
blocker and never was.** The forward was already batched — one canvas step concatenates every
in-flight block into a single forward — and only the per-request sampler tail is serial, which is
inherent work (it scales with the batch either way), not a serialisation defect. bs>1 needed no new
machinery; it needed measuring.

Marginal step cost against the batch size actually present in the step (captured leg, `CONC=4`):

| bs | step | throughput index (bs/step) | vs bs=1 |
|---|---|---|---|
| 1.0 | 181.1 ms | 5.52 | 1.00x |
| 2.0 | 268.6 ms | 7.45 | 1.35x |
| 2.4 | 312.6 ms | 7.68 | 1.39x |
| 3.5 | 433.7 ms | 8.07 | 1.46x |

and end to end, aggregate over concurrent 256-token requests:

```
conc=1    ~35 tok/s          (99-106 token partial blocks)
conc=2    348 tok in 3.87s =  89.9 tok/s
conc=4    751 tok in 8.07s =  93.0 tok/s      <- +3.4% over conc=2
```

**Batching buys ~1.35x by bs=2 and then flattens: bs=2 → bs=4 is +3.4%.** The step cost grows very
nearly linearly with the batch (181 → 269 → 434 ms), which is what a GPU that is *already saturated
at 256 rows* does when you give it more rows. The 8.1 GB/card expert-weight stream does amortize —
that is the 1.35x — but past bs≈2 the GEMM work dominates and scales with the row count, so
aggregate throughput is flat.

So concurrency is not the missing multiple either. The measured ceiling on this box is **~93 tok/s
aggregate**, against a 300 tok/s target and a backbone that alone costs 127.7 ms of a ~50 ms budget.
Nothing in the serving layer closes that gap; it is a kernel-efficiency problem.

### D7.1 Further corrections to Part D

11. **§D3 item 10 is now wrong, exactly as it predicted.** `attention/hip.py` needed no canvas change
    only while the canvas ran eager. Capturing it required a fifth static-metadata family
    (`init_canvas_capture` / `_canvas_metadata_static` / `prepare_canvas_for_{capture,replay}`).
    The rows are the SAME arithmetic as `_fill_swa_verify_static` at a different qlen, so the two now
    share `_fill_swa_multiquery_static`, and the four byte-identical main-pool fills across the
    decode / verify / fused-verify / canvas families share `_fill_paged_static`.

12. **The step is 186 ms, not 208 ms.** The 208 came from pairing k=17 (the median) with the 72.3
    tok/s leg; the correct pairing for that leg is k=19. Measured 181-187 ms across five legs.

13. **`_moe_block_m` already picks the right tile, and the reference does not.** At 256 canvas rows
    with 128 experts the average is ~16 rows/expert, and `_moe_block_m` picks the 16-row minimum —
    zero padding waste — as a pure function of static shapes, which is also what makes it
    capture-safe (a captured graph bakes the tile the eager step would have chosen). vLLM's selector
    picks 32 on the identical shape, padding 16 real rows into a 32-row tile.

14. **`soft_embedding` saves storage, not arithmetic, and pays 8x for it in traffic.** §A/§C's
    framing ("~100x smaller than carrying the logits") is a claim about the carried STATE and it is
    true. It is not a claim about cost, and the cost is the same second-LM-head vLLM pays: 378 GFLOP
    per step in total, TP-sharded to 189 GFLOP/card — sharding is not a reduction. Worse, chunking
    over 32 canvas ROWS to bound the fp32 softmax transient (268 MiB → 33 MiB) makes each of the 8
    chunks re-stream the whole 738 MB embedding shard: ~5.9 GB/card/step against a 0.74 GB floor,
    measured at 15-16 ms/step by both instruments. Chunking over VOCAB instead keeps the transient
    bound and streams the shard once; the sampler's `probs` is also already exactly `softmax(scaled)`,
    so the softmax inside it is recomputed. Not fixed here — recorded so it is not re-derived.

