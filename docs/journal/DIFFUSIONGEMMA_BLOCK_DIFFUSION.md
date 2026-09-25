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


## D8. The per-step TAIL: 58.6 ms → 12.8 ms, and where the step actually went

§D7.1 item 14 recorded that `soft_embedding` "chunks over the wrong axis" and left it. This section
is that fix plus the two it exposed. The canvas step is **190.3 → 150.5 ms (−21%)** and end-to-end
throughput on the harness prompts is **49.9 → 82.1 tok/s**, on a serve whose backbone was not touched.

All four legs below are `MINISGL_CANVAS_TIMING=1` marginal splits (consecutive report points
differenced, so the boot is not smeared into the steady state), bs=1, TP=2, CONC=1, bf16 KV,
`minisgl-rdna4:gemma4`, each served from its **own isolated git worktree** — four agents were editing
the shared tree concurrently and a serve re-reads Python lazily for the whole run, so a shared-tree
mount is how you get plausible numbers off code nobody wrote.

| leg | fwd_issue | fwd_tail | sampler | soft_embed | **step** | e2e mean |
|---|---|---|---|---|---|---|
| `9c12a6c3` baseline | 0.8 | 159.8 | 11.3 | 16.5 | **190.3** | 49.9 tok/s |
| + soft_embedding over VOCAB | 0.8 | 159.7 | 11.5 | 3.9 | **178.2** | 61.6 tok/s |
| + vocab-parallel tail | 0.6 | 151.2 | 4.9 | 3.8 | **162.9** | 76.4 tok/s |
| + `_PIPE_M` 512 → 256 | 0.7 | 139.0 | 4.7 | 3.8 | **150.5** | 82.1 tok/s |

Run-to-run, measured by repeating two of the legs in a separate lease and a separate boot: **0.2–0.5%
on `step`**. Every delta above is an order of magnitude clear of that. `e2e mean` is far noisier than
`step` and must be read as a consequence, not as the measurement: it multiplies the step by *k*, the
realised denoising steps, which is data-dependent and moves with any trajectory perturbation.

### D8.1 What each fix was

1. **`soft_embedding` chunked over canvas ROWS.** 8 chunks × the whole 738 MB embedding shard =
   5.9 GB/rank/step against a 0.74 GB floor, exactly as §D7.1-14 predicted. It also built a SECOND
   full-vocab fp32 softmax of a distribution `CanvasState.step` had already computed for the entropy
   bound and the multinomial. Chunk axis moved to vocab (and at this shape one GEMM suffices);
   `DiffusionStep.probs` carries the sampler's softmax to its last consumer. **16.5 → 3.9 ms.**
2. **The LM head all_gathered `[256, 262144]` that no rank ever indexed.** Every consumer of canvas
   logits is a *reduction* over the vocabulary, and a reduction decomposes over disjoint column
   blocks. The gather moved 67 MB/rank and was followed by a `permute().contiguous()` over the whole
   result purely to undo its own rank-major interleave. Now `ParallelLMHead.logits_local_shard`
   returns this rank's columns and `sharded_canvas_tail` reduces with four `[k, canvas]` messages per
   request per step. **−9.3 ms in `fwd_tail`, −6.4 ms in `sampler`** (the sampler's elementwise
   passes halve, net of the new collectives — so those cost under 6.4 ms and, at a few KiB, almost
   certainly under 1).
3. **`minv`'s rd→pipe threshold was 512.** `dense_gemm_rd` bypasses LDS, so every M-tile streams the
   whole B matrix from HBM and its cost grows as `ceil(M/64)·N·K`. On the canvas LM-head shard
   (`[256,2816]×[2816,131072]`) that is **16.4 ms at 11.5 TFLOP/s**; `dense_gemm_pipe` does it in
   **2.7 ms**. The two are bit-identical — `max|rd − pipe| = 0.000e+00` in all 27 cells of a
   three-shape × nine-M sweep — so this is a pure selector fix with no invariant traded. **−12.9 ms.**

### D8.2 A live correctness bug fell out of (2)

The schedulers are `mp.set_start_method("spawn")` processes, so their default torch generators are
seeded non-deterministically and **independently**. An **unseeded** canvas request therefore drew a
different `torch.multinomial` and a different `_noise()` on each TP rank. Both feed the next
forward's `input_ids`, and `VocabParallelEmbedding` all-reduces a *masked* gather — so each rank
contributed rows only for the ids **it** drew, and with a 2-way split roughly a quarter of every
canvas got the sum of two unrelated embeddings and another quarter got zero. It produced fluent text
throughout, which is precisely why nothing caught it.

A vocab-parallel multinomial has to make the draw lockstep regardless, so it now is — **by
construction**, not by trusting two RNGs: rank 0's uniform deviates and renoise ids come off
collectives A and D, which had to be sent anyway. `SamplingParams.seed` requests already agreed and
are byte-unaffected. The engine reports the state of the ranks' generators once per process:

```
[canvas] TP rank 1: RNG deviates DIVERGED from rank 0 on the first sharded step
         (rank 0's are used either way — the draw is lockstep by construction, not by luck)
```

Every canvas number in Parts D1–D7 was taken on unseeded requests and is therefore a measurement of
the *diverged* engine. The timings stand (the work per step is unchanged by which ids the canvas
holds); **the quality and *k* results should be re-read as lower bounds.**

### D8.3 Correctness, and what is NOT claimed

Block diffusion has no greedy mode — uniform-noise canvas, multinomial every step — so cross-boot
byte identity is unobtainable in principle and `temperature 0` pins nothing. The gates used:

* **Seeded self-reproducibility on the serve.** `seed=20260804`, same prompt ×3 → `DISTINCT=1` on
  every leg. This is also the gate that would fail loudly if the ranks ever disagreed again.
* **Fixes 1 and 2 are NOT bit-preserving vs the baseline, and say so.** fp32 GEMM reduction order
  changes when you stop splitting a contraction 8 ways; the softmax handed to `soft_embedding` is the
  Categorical-shifted one; and an inverse-CDF draw is not `torch.multinomial`. All are the same
  functions in exact arithmetic. A 1-ULP move in the carried state re-sorts 256 entropies through a
  thresholded cumulative sum, so the block's text changes — that is the architecture, not a defect.
* **Fix 3 IS bit-preserving end to end**, which is the strongest available statement: the seeded
  request returns sha `a11ffd81ad883e04` before *and* after, same 103 tokens, same per-block step
  counts (17/17/17/17/14/15/16).
* **The sharded tail is unit-tested against the whole-vocabulary tail**
  (`tests/diffusiongemma_sampler_test.py` [5]). The served TP path is no longer the one the 48-step
  bit-exact reference replay exercises (that can only be `tp_size == 1`), so two **real threads** run
  `sharded_canvas_tail` SPMD over the halves of one logits block through a barrier — threads, not an
  in-call stub, because the *order* of the four messages is under test and only a real barrier can
  deadlock on a mis-ordered reduction. probs shard matches the global softmax's columns to 1.19e-07
  (1 fp32 ULP at p=1), entropy to 1.0e-06, argmax exactly, and the draw is identical on both ranks
  *with deliberately different per-rank generators*.
* **AR guard**: gemma-4-26B-A4B TP=2, 256-token completion, **44.2 → 44.3 tok/s**. It matters for fix
  3, which is the only one of the three that touches a shared engine path.
* 11/11 CPU tests, plus the new [5] block (the sampler test goes 7 → 16 checks).

### D8.4 What is left, measured — and why the fused sampler kernel is NOT next

Per-stage, at the served shape, one rank (`[256,2816]×[2816,131072]`, gfx1201):

| stage | now | was |
|---|---|---|
| LM-head GEMM (`dense_gemm_pipe`) | 2.73 ms | 16.4 ms + a 67 MB/rank all_gather + a full-tensor permute |
| softcap (`tanh(x/30)·30`, fp32) | 1.51 ms | 3.1 ms (it capped both shards) |
| sampler (temperature 0.40 + exp 1.02 + cumsum 0.59 + entropy + argsort + 4 collectives) | 4.8 ms | 16.7 ms |
| `soft_embedding` (GEMM 3.07 of it) | 3.8 ms | 15.3 ms |
| **tail total** | **≈12.8 ms** | **58.6 ms** |

A fused one-launch canvas sampler (softcap + temperature + logsumexp + softmax + entropy + argmax +
multinomial + accept + renoise) was the planned third fix. On these numbers it is now worth **~4 ms
of a 150 ms step (≈2.7%)** — and it would have to fuse *across a TP shard boundary*, i.e. carry the
four collectives inside or around the launch. That is a large kernel with real distributed-correctness
risk for 2.7%. **Shelved on evidence, not skipped**: revisit only if the backbone shrinks enough to
make 4 ms matter.

The step is now **150.5 ms, of which ~137 ms is the backbone**. Every remaining lever is in there.
§D6.4's argument still holds and is now sharper: cudagraph capture removes 30 ms of launch that is
currently hidden behind the backbone, and its share grows as everything around it shrinks.

---

## Part D9 — the tail kernels were bf16-only, this checkpoint is fp16, and nothing said so

### D9.1 The defect

`rdna4-hip-kernels/tail` ships six elementwise ops. Two of them — `silu_and_mul`, `gelu_and_mul` —
were `template <typename scalar_t>`. The other four, sitting in the *same file*, were hard-typed
`__hip_bfloat16` in both the kernel bodies and the binding guards (`"bf16 only"`): `rms_norm`,
`rms_norm_add`, `rope`, `store_kv`. The engine's `layers/_tail_hip.active()` therefore asked
`dtype == bfloat16`, because that was all the kernels accepted.

Gemma4 and DiffusionGemma are **fp16**. So on every call, on every layer, for the whole serve, the
gate answered NO and the engine ran the eager torch decomposition instead — 7 launches per RMSNorm,
~6 per RoPE, an `index_put_` per KV store. This does not fail. It produces plausible output at
plausible speed. The **only** observable tell was that a serve log for these two models contained
**zero `[hip-engage] tail_hip.*` lines** while every other kernel family reported in:

```
leg A (baseline, engine 9fc1c892, minisgl-rdna4:gemma4):
  [hip-engage] attn_decode.flash_decode_paged        [hip-engage] fp8_wmma.mmq_fp8_gemm(decode_gemv)
  [hip-engage] attn_hip.flash_prefill                [hip-engage] fp8_wmma.mmq_fp8_moe_gemm(wmma)
  [hip-engage] attn_prefill_paged.flash_prefill_paged(canvas)   ... 14 lines, and
  tail_hip.*  ->  COUNT 0     (canvas phase AND autoregressive phase)
```

Two further gates were hiding behind the first, and neither is fp16-specific:

* `RMSNormNoScale` (transformers `with_scale=False`) had **no native path at all**, because
  `tail_hip.rms_norm` required a weight tensor and the checkpoint ships none for those norms. Gemma4
  runs two per layer — `self_attn.v_norm` and `router.norm` — i.e. 60 of the ~331 norms in a step.
* `gelu_tanh_and_mul` (HF `gelu_pytorch_tanh`, Gemma4's activation for **both** the dense MLP and the
  routed experts) had no kernel because adding a third activation looked like a third copy of
  `silu_mul_kernel`. It is now a policy (`GeluTanhAct`) on one `gated_mul_kernel<scalar_t, Act>`.

### D9.2 What was measured

Same harness (`tools/diffusiongemma_generate.sh`), same image (`minisgl-rdna4:gemma4`), TP=2, bs=1,
`MINISGL_CANVAS_TIMING=1`, each leg from its own isolated git worktree, back to back on the same box.
`[canvas-timing]` reports CUMULATIVE averages, so every figure below is **differenced between report
points** over the *same* window (n=30→40) in both legs.

| marginal, per canvas step | A: baseline | C: templated tail + tanh-gelu | Δ |
|---|---|---|---|
| `fwd_issue` | 0.60 ms | 0.50 ms | −0.10 |
| **`fwd_tail` (the backbone)** | **137.90 ms** | **122.00 ms** | **−15.90 (−11.5%)** |
| `sampler` | 4.80 ms | 4.80 ms | **0.00** |
| `soft_embed` | 3.50 ms | 3.60 ms | +0.10 |
| **step** | **147.50 ms** | **133.40 ms** | **−14.10 (−9.6%)** |

The two untouched phases landing on 4.80/4.80 and 3.50/3.60 is the provenance check: the differencing
method and the two legs agree to 0.1 ms on everything this change does not touch, so the −15.9 ms in
`fwd_tail` is the change and not the weather. (C's steadiest window, n=40→50, reads 130.60 ms — i.e.
−11.5% — but n=30→40 is the honest same-window comparison.)

**The autoregressive sibling is where this lands hardest.**

| gemma-4-26B-A4B AR, 256-token greedy | A | C |
|---|---|---|
| prompt 2 | 254 tok in 5.81 s = **43.7 tok/s** (22.9 ms/tok) | 254 tok in 3.41 s = **74.4 tok/s** (13.4 ms/tok) |
| prompt 3 | 256 tok in 5.84 s = **43.8 tok/s** (22.8 ms/tok) | 256 tok in 3.42 s = **74.8 tok/s** (13.4 ms/tok) |

**+70%, at identical emitted-token counts.** That is not a bandwidth result and it is not luck — it
is the reason the canvas number is the *smaller* of the two.

### D9.3 Dispatches removed, measured

`rocprofv3 --kernel-trace`, 5 s window at t=180 s (past boot and graph capture), TP=2, per rank,
**both legs on engine 9fc1c892** so the counts are same-engine. Steps are calibrated off the MoE
grouped GEMM, which fires exactly 60× per canvas step per rank — a structural constant, so the
window calibrates without trusting wall-clock. (Kernel-trace inflates *small* kernels more than
large ones, so the elementwise device times below are upper bounds and the GEMM ones lower bounds;
the dispatch COUNTS are exact.)

| per canvas step, per rank | A: baseline | C: templated tail + tanh-gelu | Δ |
|---|---|---|---|
| **total dispatches** | **5 365** | **2 195** | **−3 170 (−59%)** |
| torch elementwise/copy | 4 580.8 · 21.96 ms | 928.4 · 8.36 ms | −3 652 disp · −13.60 ms |
| native `tail_hip` | **0** · 0.00 ms | 499.4 · 2.20 ms | +499 disp · +2.20 ms |
| collective (`one_shot_ar`) | 99.2 · 51.19 ms | 96.8 · 44.71 ms | −6.48 ms |
| our kernels + misc | 652.2 · 81.57 ms | 641.2 · 73.45 ms | −8.12 ms |
| profiled busy | 154.80 ms | 128.77 ms | −26.03 ms |

Net: **−3 153 elementwise dispatches/step/rank**, and the whole eager norm/rope/activation workload
(21.96 ms of profiled device time) collapses into 499 native dispatches costing 2.20 ms. The
`one_shot_ar` row falling 6.5 ms is a consequence, not a separate win: a one-shot all-reduce *spins*
until its peer arrives, so removing work symmetrically from both ranks shortens the barrier too.

The four native kernels appear in the trace exactly where the model says they should:
`rms_norm_kernel<__half>` 343.3 calls/step, `gated_mul_kernel<__half, GeluTanhAct>` 62.8,
`rope_kernel<__half>` 61.9, `store_kv_kernel<__half, __half>` 30.9.

### D9.4 The traffic model was WRONG; the dispatch model was right

The prior session's hypothesis was **7.8 GB/step of avoidable HBM traffic** from the eager RMSNorm
decomposition. The *arithmetic* reproduces almost exactly — 36 B/element eager (`.float()`,
`.pow(2)`, `.mean(-1)`, `+eps`, `rsqrt`, `*`, `.to(fp16)`, `*gain`) versus 4 B/element native, over
206.5 M norm elements/step/rank = **6.6 GB/step**, or **7.6 GB** including RoPE and the gated
activation. The *roofline attribution* does not.

The largest intermediate in one eager norm is the fp32 copy of a `[256, 2816]` activation: **2.88 MB**
— inside the 8 MB L2, far inside the 64 MB Infinity Cache. And the baseline trace measures it
directly: `pow_tensor` moves 8 B/element over 720 896 elements in **3.09 µs = 1 867 GB/s**, which is
**264% of the 706.6 GB/s HBM roofline**. Traffic that runs at 2.6× the HBM roofline was never in HBM.

So "remove 7.6 GB of HBM traffic" was never the mechanism, and the GB figure — though correct as
arithmetic — has no predictive value here. What the change actually removes is **3 153 dispatches per
step per rank**, each with a fixed per-dispatch cost that dwarfs its own work at these sizes. That
reframes both earlier negative results correctly:

* the fp16→bf16 **bridge** removed launches but *added* two casts per call, and left the
  `with_scale=False` norms, the KV store and the tanh-gelu in torch — hence +3.2%;
* **cudagraph capture** cut the *host* launch loop 30.8 → 0.8 ms for −0.3%, because the host was
  never the bottleneck. The **device-side** per-dispatch cost was, and a captured graph still issues
  every one of those 5 365 dispatches — it only stops the CPU from having to ask for them.

The canvas step (N=256 rows) is the *unfavourable* case for this fix: the kernels are big enough that
per-dispatch overhead is a minority of their cost, so it buys 9.6%. AR decode (N=1 row) is the
favourable case: every one of those ~3 100 dispatches is almost pure overhead, which is why the same
patch is worth +70% there. **Dispatch count matters in inverse proportion to rows per step** — the
single most useful thing this measurement establishes, and the reason "5 070 dispatches/step" was
worth chasing on the canvas but was *undersold* as a lever for the autoregressive sibling.

### D9.5 What is still elementwise, and who owns it

928 torch dispatches/step survive. From the trace, the fusable chains left in the backbone are:

1. `h = post_feedforward_layernorm(dense + moe); h = residual + h; return h * layer_scalar` — an
   add, a norm, an add and a scalar multiply, 4 ops × 30 layers. One `rms_norm` variant taking a
   second addend and an output scale would fold all four.
2. `Gemma4Router.forward`: `RMSNormNoScale(x) * self.scale * hidden**-0.5` is **already** expressible
   as one call — `rms_norm(x, scale * hidden**-0.5, eps, plus_one=0)` — because a weighted RMSNorm is
   exactly "normalize, then multiply by a per-channel vector". 3 launches → 1, no new kernel, and the
   constant folds into the weight at load time.
3. `torch.cat([q, k, v])` in `Gemma4Attention.forward` — 2 `CatArrayBatchedCopy` per layer.

All three live in `models/gemma4.py`, which another agent owns; they are recorded here rather than
taken. Item 2 is the cheapest real win left in the backbone.

### D9.6 Numerics

Not bit-identical to the torch fallback, and strictly closer to the fp32 reference. The eager path
rounds *mid-chain* — `normed = (xf * rsqrt(var+eps)).to(fp16)` and only then multiplies by
`(weight + 1)`. The kernel carries fp32 through the gain and rounds once, at the store. Parity vs the
fp32 reference rounded to fp16: `max|Δ| ≤ 2.0e-3` across bf16 and fp16 at every shape in
`tail/tests/test_tail.py`, `store_kv` bit-exact, e4m3 saturation unchanged. This is the same
convention every bf16 model in this repo has been served under since the tail kernels landed.

---

## Part D10 — the step is 97.7 ms, and the collective is no longer the story

First profiling pass on this workload since the gfx1201 hardware counters became usable, and the
first one at all with the canvas loop instrumented: `_diffusion_loop` is a peer of the plain-decode
and spec loops but was the only one of the three with **no ROCTx markers**, so
`rocprofv3 --selected-regions` never opened a window and a block-diffusion serve traced *empty*. The
markers (`canvas_step#N` / `canvas_fwd` / `canvas_sampler` / `canvas_soft_embed` / `canvas_encode`)
and the two-leg harness `tools/canvas_rocprof.sh` land with this section.

Fixtures: `tools/_fixtures/canvas_diffusiongemma/` (breakdown, roofline, tile selection, both raw
logs, and PROVENANCE). Engine `4d719ad6`, kernels `8a8bca6`, image `minisgl-rdna4:dgprof` built from
clean worktrees of both, TP=2, CONC=1, bs=1, `MINISGL_SWA_RADIX=0` pinned, auto perf level.

### D10.0 Three things that had to be fixed before a number existed

1. **`minisgl-rdna4:post-tile-prof` cannot run this engine.** Its `mmq_fp8_moe_gemm` predates the
   `x_fp8` producer-quant kwarg, and the serve dies inside graph capture with
   `TypeError: ... unexpected keyword argument 'x_fp8'`. The image had to be rebuilt from both
   worktrees. This is the ABI-split rule with a new symptom: not a wedge, a `TypeError`.
2. **The marker window is in STEP space.** `_rtx_step` counts canvas steps, and a 256-token block is
   only k≈9–26 of them, so a window at step 60 off one request never opens.
3. **`MINISGL_EXIT_AFTER_STEPS` must be reachable.** It counts loop iterations, and the canvas loop
   *blocks* when idle, so the count is ≈ steps + prefills. A bound of 250 was never reached, the
   engine never returned, rocprofv3 never ran its destructor, and the trap tore the container down —
   the abort-without-writing failure `propose_rocprof.sh` documents. The trace came back empty three
   times, from three different causes, before it came back at all.

### D10.1 The baseline, today

`[canvas-timing]` differenced over the same window twice (n=60→70 and n=70→80) — the cumulative
average makes the raw report points useless on their own, and the two windows agree to 1.8%:

| marginal, per canvas step | n=60→70 | n=70→80 |
|---|---|---|
| `fwd_issue` | 0.3 | 0.9 |
| `fwd_tail` | 88.5 | 88.5 |
| `sampler` | 4.7 | 4.9 |
| `soft_embed` | 3.5 | 3.7 |
| **`step`** | **99.5** | **97.7** |

**97.7 ms**, against **150.5 ms** in §D8 and **133.4 ms** in §D9. The AR vectorisation (`74a01eb6`)
landed *after* §D9 was written, so neither recorded figure contains it — and neither does the 82
tok/s in flight. Note `fwd_tail` is 88.5 of the 97.7: the backbone is 91% of the step.

End-to-end tok/s stays as noisy as §D8 warned, and for the reason §D8 gives — it multiplies the step
by *k*, which is data-dependent. Four measured requests: **75.8 / 49.0 / 106.5 / 53.0 tok/s**, with
k = 12/19/9/19 at 95–103 emitted tokens. Reading tok/s as the measurement here is a mistake; the step
is the measurement, and *k* is a separate (and larger) lever.

### D10.2 Where the step goes

Trace, one rank, 40 canvas steps. rocprofv3 stretches the step (139.15 ms traced vs 97.70
un-profiled = **1.424x**), so shares come from the trace and the `real` column divides by that.

| phase | real ms/step | share |
|---|---|---|
| `canvas_fwd` (backbone) | 52.8 | 72.4% |
| `canvas_sampler` | 13.0 | 17.8% |
| gap between phases | 4.7 | 6.4% |
| `canvas_encode` (amortised) | 2.2 | 3.0% |
| `canvas_soft_embed` | **0.2** | 0.3% |

| kernel family | real ms/step | share | disp/step |
|---|---|---|---|
| MoE grouped GEMM | 22.4 | 31.5% | 66.8 |
| **collective** | **13.3** | **18.3%** | 107.6 |
| dense GEMM W4A8 | 10.3 | 15.8% | 400.2 |
| attention | 8.5 | 11.8% | 64.5 |
| torch elementwise/copy | 7.2 | 9.9% | 1065.8 |
| dense GEMM fp16 | 5.8 | 7.9% | 109.9 |
| tail (native) | 3.2 | 4.4% | 556.1 |

**Summed kernel busy is 74.6% of the marker wall** — a quarter of the canvas step is inter-kernel
gap, *under graph capture*.

### D10.3 The "collective-bound 33%" claim does not survive

§D9.3 measured `one_shot_ar` at 44.71 ms of 128.77 ms profiled busy = **34.7%**, on engine
`9fc1c892` — i.e. **pre-vectorisation**. Today, on `custom_ar::one_shot_ar_vec_kernel`: 107.6
calls/step at 176 µs traced (124 µs deflated) = **13.3 ms/step = 18.3%**. The collective is no longer
the largest item, and optimising on the basis that it is would be optimising a stale trace.

**And what is left is not payload-bound.** The kernel launches **8 workgroups** on a 64-CU card and
moves 1.44 MB in 124 µs = 11.6 GB/s, which is nowhere near any link. A one-shot all-reduce *spins*
until its peer arrives, so this number is mostly **rank skew**, exactly as §D9.3 said when the same
row fell 6.5 ms because work was removed *symmetrically from both ranks*. Shrinking the payload will
not move it; making the two ranks arrive together, or issuing fewer of them (107.6 per step is ~3.6
per layer), will.

### D10.4 The dense W4A8 path IS live here — unlike Qwen — and it is under-occupied

The check that had to happen first. `Qwen3.6-35B-A3B-AWQ` leaves **every** dense linear in bf16, so
its dense-tile surface is never dispatched. This checkpoint's compressed-tensors `ignore` list holds
`mlp.{gate,up,down}_proj`, `router.proj` and `self_conditioning` — but **not**
`self_attn.{q,k,v,o}_proj`. (`quant/config.py::_norm_ignore` rewrites the checkpoint's
`model.decoder.` namespace to the loader's `model.`, which is what makes those entries match at all.)

Confirmed at runtime, not inferred — the ledger reports `[hip-engage]
fp8_wmma.mmq_fp8_gemm(wmma_tiled_tuned)`, and the trace counts **132.2 dispatches/step** of it. So
**115 dense W4A8 GEMMs per canvas step per rank, all at M=256**, squarely inside the chooser's M≥64
band that `tile_select.h` notes "no tile-selection change here moves a decode step". Here it moves
every step.

What the chooser picks, and what the hardware then does (grid/VGPR straight from the kernel trace):

| shape | tile | wgs | waves/SIMD | runner-up |
|---|---|---|---|---|
| SWA q_proj | 128x64 | 64 | 8/32 model, **12/16 measured** | 192x64 at 1.100x |
| SWA k/v_proj | 64x32 | 128 | 8/32 | 128x32 at **1.000x** |
| SWA o_proj | 192x96 | **60** | 12/32 | 128x64 at 1.009x |
| FULL q_proj | 128x64 | 128 | 16/32 | 256x64 at **1.000x** |
| FULL k_proj | 64x32 | 64 | 4/32 | 80x32 at 1.028x |
| FULL o_proj | 192x96 | **60** | 12/32 | 128x64 at 1.009x |

Three of six launch ≤64 workgroups in ONE round — one block per CU, no co-residency — and the two
`o_proj` shapes launch **60 workgroups on 64 CUs**, leaving four CUs with no work at all. The
measured kernel carries **VGPR=112 → 12 of 16 waves/SIMD**.

The recently-merged near-tie band is **inert here**: it exists to keep the 256x128 incumbent when the
argmin is within 1.10x, and 256x128 is not close at any of these shapes. Meanwhile the actual
runner-ups sit at 1.000x and 1.009x — margins the model explicitly cannot resolve — and are being
broken arbitrarily.

The result is a GEMM at **neither** roofline: 0.312 GB and 284 GFLOP in 9.26 ms = **4.8% of the
706.6 GB/s HBM ceiling and ~8% of fp8 WMMA peak**. Being bound by neither is what 64 workgroups at
12/16 waves means.

Beside it, `compute_act_fp8_and_scales_kernel` fires **132.2 times per step** for 1.0 ms. The
producer-side activation quant that deletes exactly this dispatch is built and bit-exact, and the
ledger shows no `+prequant` tag — **it is not engaged on this path.**

### D10.5 The five full-attention layers cost more than the twenty-five sliding ones

| kernel | disp/step | real ms/step |
|---|---|---|
| `flash_prefill_paged_fp8_split_kernel<__half, 512, 8>` | 5.0 | 4.74 |
| `flash_prefill_reduce_kernel<__half, 512>` | 5.0 | 2.17 |
| `flash_prefill_paged_fp8_split_kernel<__half, 256, 4>` | 25.0 | 1.35 |
| `flash_prefill_reduce_kernel<__half, 256>` | 25.0 | 0.21 |

**5 full layers = 6.91 ms; 25 sliding layers = 1.56 ms.** Per layer that is 1.38 ms against 0.062 ms
— **22x**. Two compounding causes: `global_head_dim` is 512 against the sliding layers' 256, and the
full layers attend the whole `[encoder KV ++ canvas]` while the sliding ones see a 1024 window. The
split kernel measures **VGPR=248 → 5 of 16 waves/SIMD**, i.e. 31% occupancy and register-bound; and
the fp32-partial `reduce` pass costs another 2.17 ms on its own for five layers.

### D10.6 Two open items from §D8 are CONFIRMED FIXED; one regression is NOT

* **`soft_embedding` re-streaming the 738 MB shard 8x**: fixed and confirmed from the trace —
  `canvas_soft_embed` is **0.2 ms/step**, against 15.3 ms in §D7.1-14. The 8 chunks are gone.
* **The LM-head `all_gather` of `[256, 262144]`**: gone. No gather of the vocab-parallel logits
  appears in the trace.
* **The fused canvas sampler**: still shelved, and the trace supports that. `canvas_sampler` is
  13.0 ms/step, but only **0.2 ms** of it is in the SAMPLER kernel family — the rest is elementwise,
  a share of the collectives, and one 1.86 ms rocBLAS `Cijk_...MT64x64x32` dispatch. A fused sampler
  would not touch most of it.
* **REGRESSION — cudagraph capture is no longer bit-identical. ROOT-CAUSED; see §D11.**
  §D6.1 asserts "bit-identical, at every captured batch size". Today the engine's own in-process
  check says otherwise, on both replays of every boot:

  ```
  [canvas-graph] REPLAY #1 engaged, bs=1 check 1/2 (qlen=256, T=256):
        graph vs eager max|delta|=1.575e+01 over (256, 2816) *** NOT bit-identical ***
  [canvas-graph] REPLAY #2 ... max|delta|=8.957e+00 ... *** NOT bit-identical ***
  ```

  A max|Δ| of 15.75 on a backbone hidden state is not a rounding difference. The served canvas is
  replaying a graph that computes something materially different from the eager forward it was
  captured from, and every number in D10 is taken on that graph. **This is the first thing to chase,
  ahead of any performance work** — it is a correctness claim the doc currently makes and the engine
  currently denies, and the likely suspects are the kernels that changed underneath it (the MoE
  producer act-quant path, or the templated tail).

### D10.7 The ranked gap to 200 tok/s

At 97.7 ms/step and the observed k≈9–26 over ~100-token blocks, the realised rate is 49–107 tok/s.
Reaching 200 needs roughly a **2.5–3x** step reduction (or the same factor out of *k*, which is a
different and unexplored lever — §D7 never established what sets it).

1. **MoE grouped GEMM — 22.4 ms/step (23% of the step).** *Mechanism: it is a pure weight stream at
   40.6% of roofline.* 256 canvas rows x top-8 over 128 experts is ~16 rows/expert, so **every**
   expert is touched and the whole 6.42 GB/rank expert set is read every step. That part is the
   architecture. What is not: 287 GB/s against 706.6. The kernel is already at **full occupancy**
   (VGPR=40 → 16/16 waves, 2728 workgroups), so this is not an occupancy fix — it is the
   reduction/arithmetic-intensity floor at block_m=16, and the lever is the same split-K/gemm2 grid
   work already in flight on the Qwen path.
2. **The collective — 13.3 ms/step (14%).** *Mechanism: spin-barrier, i.e. rank skew, not payload.*
   8 workgroups moving 1.44 MB in 124 µs is not a bandwidth number. 107.6 all-reduces per step is
   ~3.6 per layer; the levers are **fewer** of them (fuse the attention-`o_proj` and MoE reductions)
   and **symmetric** rank work, not a faster kernel.
3. **Dense W4A8 — 10.3 ms/step (11%).** *Mechanism: under-occupancy the tile chooser is choosing.*
   64 workgroups on 64 CUs at 12/16 waves, 4.8% of HBM and 8% of fp8 peak. Two concrete moves, both
   cheap: (a) the chooser is picking among 1.000x/1.009x ties at these shapes with a band that only
   protects a 256x128 incumbent that is never in contention — extend the tie-break to prefer the tile
   that **launches more workgroups**; (b) engage the producer act-quant that already exists and
   deletes 132 dispatches/step.

Below those, in order: the **25% inter-kernel gap** (24.8 ms/step idle under capture); **attention**
at 8.5 ms with the 5 full layers taking 80% of it at 5/16 waves; and the **810 torch elementwise
dispatches/step** costing 5.2 ms, 37% of which launch ≤64 workgroups — the fusion targets §D9.5
already named and left to `models/gemma4.py`.

**Not measured here:** the VALU/VMEM/LDS issue split, which still needs `--pmc` under
`profile_standard` in the ROCm 7.14 image against replayed shapes. Everything above comes from the
kernel trace, which already carries VGPR/scratch/LDS/grid per dispatch — so occupancy needed no
counter run at all.


## Part D11 — the capture ran a DIFFERENT attention kernel, §D10's step was 12% too long, and the gate only printed

§D10.6's `max|delta|=1.575e+01` is not a rounding difference, was not the MoE producer act-quant,
and was not the templated tail. It is one line of kernel-selection arithmetic reading one number
that is not the same on the two paths.

### D11.1 The mechanism

`attn_prefill_paged`'s split-K planner is keyed on the block-table ROW WIDTH:

```c
prefill_split_policy(max_blocks * block_size, align, base_grid, &num_splits, &chunk);
```

with the stated justification that a shape-derived number is "constant across capture/replay". It
is constant — and it is a **different constant on each path**:

| | where the width comes from | width | `prefill_split_policy` |
|---|---|---|---|
| eager (`prepare_metadata`) | `page_table[idx, :max_seqlen_k:page_size]` | 18 pages = **288** | 288 ≤ `MIN_CTX` (1024) → **num_splits = 1, single-pass** |
| graph (`_canvas_metadata_static`) | the static capture buffer, sized from `max_seq_len` | 16384 pages = **262144** | → **num_splits = 64, split-K + a reduce pass** |
| graph, sliding layers | `_ccap_swa_page_table`, `swa_window + canvas_len` | **1280** (sched: 276) | 1280 > 1024 → **split too** |

So the captured canvas ran `flash_prefill_paged_fp8_split_kernel` + `flash_prefill_reduce_kernel`
on all 30 layers while the eager forward it was captured from ran the single-pass kernel. Two
different kernels, same inputs — against the split path's own claim of "bit-identical (mod fp32
accum order) to the single-pass kernel".

### D11.2 Isolated, in one boot, by making the gate say which variable moved

The old gate compared the captured region against an eager forward that differed from it in **two**
ways at once — the graph mechanism, and the program, because `tp_overlap` is capture-transparent
(under capture every all_reduce is inline and Gemma4's FFN row split is off). A gate that does not
hold the second one fixed cannot attribute its own number, which is why 1.575e+01 sat in the doc
with four plausible suspects and no way to choose. The gate now takes its reference inside
`inline_collectives()` and reports four controls plus a free host-side metadata diff:

```
METADATA DIFF max_seqlen_k: sched=276 static=262144
metadata page_table: values agree over [:1,:18], row width differs sched=(1,18) static=(1,16384)
graph vs eager[matched regime]                   = 8.324e+00   *** NOT bit-identical ***
eager self-consistency                           = 0.000e+00
replay self-consistency                          = 0.000e+00
eager-fallback regime (side-stream AR + 2 chunks) = 0.000e+00
LOCALISE: eager[static replay metadata] vs eager[scheduler metadata] = 8.324e+00
          graph vs eager[static replay metadata]                     = 0.000e+00
```

Read the last two lines together: **the graph reproduces its own capture exactly**, and the whole
divergence is that the static metadata selects a different kernel. `ref max|x|` is 41.44, so 8.324
is **20% of full scale** — the split path is not merely reassociating, it is wrong in this corner.

Then the proof, with **one env var and no code change** — `MINISGL_ATTN_MAX_SPLITS=1` forces
single-pass on both paths, the page-table width diff still present:

```
graph vs eager[matched regime] = 0.000e+00  BIT-IDENTICAL
LOCALISE: eager[static] vs eager[scheduler] = 0.000e+00
```

Two things fall out on the way. **The row split is acquitted**: the `eager-fallback regime` line
reads 0.000e+00 at both `MINISGL_TP_AR_CHUNKS=2` and `=1`, so the side-stream all_reduce and the
two-chunk FFN split are bit-exact *in serve*, which `tp_overlap.py` had only ever claimed op-level.
And **there is no bisect answer**: the divergence is a property of the shape the capture allocates,
so it has been latent since capture landed (`d3f42c6e`) for any config whose static page table
exceeds `MIN_CTX`. §D6.1's `0.000e+00` predates the split-K path. Bisecting engine commits would
have found the split-K merge and named the wrong thing.

### D11.3 The fix is in the kernel, and "cap the capture" is not it

The obvious engine patch — bound `_ccap_max_pages` the way the DDTree family bounds `_dcap_max_kv`
— **does not work**, and the reason is worth stating so it is not tried again. No static width can
equal a dynamic context in general; capping only moves the crossover. Capping *below* `MIN_CTX`
would make the graph bit-identical and then drop the canvas to the eager fallback after ~3 blocks
(context = prompt + 256·blocks passes 1024 almost immediately), which is a worse trade than the bug.

The split decision has to be a **caller-supplied constant that both paths pass** — the engine
already knows one number that bounds every replay of a given graph. That is a kernel signature
change plus engine plumbing, and it is the open item. Until it lands the gate **raises** rather than
serving a graph that computes something else; `MINISGL_ATTN_MAX_SPLITS=1` is the operator workaround
and is what the numbers below were taken under.

The same defect is latent in **every capture family that sizes its page table from `max_seq_len`**
(`_vcap_max_pages` does); only the DDTree family caps. Their gates have simply not been exercised on
a context long enough to cross `MIN_CTX`.

### D11.4 §D10 re-taken: 97.7 → 86.9 ms/step, and every attention row is void

On a graph the gate certifies BIT-IDENTICAL at both replays, six consecutive 10-step windows
differenced out of the cumulative average (TP=2, bs=1, CONC=1, `MINISGL_SWA_RADIX=0`):

| window | `fwd_issue` | `fwd_tail` | `sampler` | `soft_embed` | **step** |
|---|---|---|---|---|---|
| n=10→20 | 0.6 | 77.2 | 4.7 | 3.5 | **86.0** |
| n=20→30 | 0.4 | 76.9 | 4.8 | 3.4 | **88.0** |
| n=30→40 | 0.6 | 77.5 | 4.6 | 3.4 | **86.2** |
| n=40→50 | 0.5 | 77.1 | 5.1 | 3.6 | **89.6** |
| n=50→60 | 0.7 | 77.3 | 4.7 | 3.2 | **88.2** |
| n=60→70 | 0.8 | 77.6 | 4.7 | 3.6 | **85.7** |

**~86.9 ms/step**, spread 4.5%, against §D10.1's **97.7 ms** — **11% faster**, because the old
number was paying for 30 layers of 64-way split-K attention plus a reduce pass that should never
have been dispatched. Two changes are in that figure and this run does not separate them (the other
is §D11.5's producer act-quant); they cannot be separated any more, because the gate now refuses to
serve the broken graph. tok/s over four requests: 92.1 / 54.9 / 73.1 / 58.8 — the same 49–107 spread
§D10.1 reports, for the same reason.

**What this voids.** Every attention row in §D10.2 and the whole of §D10.5 was measured on the
mis-selected path: `flash_prefill_paged_fp8_split_kernel` at 5.0 and 25.0 dispatches/step,
`flash_prefill_reduce_kernel` at the same counts, and the 8.5 ms/step attention total describe a
kernel selection the served canvas should never have made. §D10.5's "the five full-attention layers
cost more than the twenty-five sliding ones" is not a statement about this architecture; it is a
statement about a 64-way split of a 276-token context. The MoE, collective, dense-GEMM and
elementwise rows are unaffected in *mechanism*, but every one of their SHARES is a share of a step
that was 12% too long.

### D11.5 The two §D10.7 items, landed and falsified

**Producer act-quant (item 3(b)) — LANDED.** It was never wired on the dense path at all: only
`layers/moe.py` took the pair. Gemma4 does not merge QKV, so `input_layernorm`'s output feeds THREE
separate w4a8 linears that each launched their own `compute_act_fp8_and_scales_kernel` over the same
rows. `RMSNorm.forward_quant` + a `supports_producer_actquant` declaration on the linear method
closes it. Verified BY PRESENCE, not by absence of an error:

```
[hip-engage] tail_hip.rms_norm_quant
[hip-engage] fp8_wmma.mmq_fp8_gemm(wmma_tiled_tuned+prequant)
[hip-engage] fp8_wmma.mmq_fp8_gemm(decode_gemv+prequant)
```

Bit-exactness was *measured*, not asserted: the canvas gate read 8.324e+00 before and 8.324e+00
after, with eager self-consistency 0.000e+00 on both — engaging the fusion moved the number by
exactly zero while an unrelated defect held it constant.

**The near-tie tie-break (item 3(a)) — the complaint is right, the proposed cure is FALSIFIED.**
The band *is* inert at these shapes and the tie *was* broken by `BM_SET`'s declaration order, which
is not a decision. `tile_better` now states the order — cost, then MORE WORKGROUPS, then a stable
lattice key — applied where the model expresses **no preference at all** (three of the seven canvas
shapes are exact ties: SWA k/v_proj 64x32 == 128x32, FULL q_proj 128x64 == 256x64). It moves **zero
picks** on both scorers, which is the point: the accidental order already agreed with the only prior
that survives scoring.

Extending it into the band, as §D10.7 asks, does not survive:

| gband | restricted-argmin geomean | moved | better | worse | lattice-wide: scored-on-a-measured-tile | moved | both measured |
|---|---|---|---|---|---|---|---|
| 1.000 | 1.0333 | 0 | 0 | 0 | 205/300 | 0 | 0 |
| 1.020 | 1.0328 | 3 | 3 | 0 | 205/300 | 11 | 3 |
| 1.060 | 1.0309 | 11 | 8 | 2 | 197/300 | 33 | 3 |
| 1.100 | **1.0396** | 32 | 9 | **22** | **139/300** | 134 | **0** |

At the full band it is a net loss on the restricted scorer, and the lattice-wide geomean's apparent
improvement (1.2525 → 1.1313) is an artefact — it picks unmeasured tiles on 66 more cells, so it is
scored on an easier subset. The honest column is *both measured*, and there it is **zero**: 134
picks move and not one is verifiable. Five relatives (max occupancy, min ragged tail, min rounds,
largest BM, smallest tile) are all worse than doing nothing.

**What is actually left on the dense path is the occupancy TERM, not a tie-break.**
`occ = min(blocks_per_cu, rounds) * nwarps` prices what *could* be resident; at `rounds == 1` with
`wgs < cu` what actually launched is smaller — the two `o_proj` shapes launch 60 workgroups on 64
CUs at 12/32 waves. Fixing that RE-RANKS the lattice, which a tie-break deliberately does not, so it
needs the surface re-swept rather than a prior re-argued.

---

## Part D12 — §D11.3's open item is CLOSED: the caller supplies the split decision

§D11.3 left one thing open — "the split decision has to be a caller-supplied constant that both
paths pass" — and said the gate would keep raising until it landed. It has landed.

### D12.1 The change

**Kernel** (`attn_prefill_paged`, `8644f41`). `prefill_split_policy` is keyed on a new **required**
op argument `split_ctx`; `max_blocks * block_size` is **deleted**, not demoted to a fallback, and the
argument carries no default. A silent fallback is precisely what let this ship since capture landed.

**Engine** (`77498b4e`). Both paths pass the **capacity of the pool being read**, which is a
serve-lifetime constant and therefore trivially the same at capture, at replay, and eagerly:

| call site | `split_ctx` |
|---|---|
| main paged pool (`_hip_prefill_paged`) | the global page table's token width (`aligned_max_seq_len`) |
| SWA ring pool (`_swa_prefill_paged`) | `window + max_seqlen_q` — the ring row's own capacity |

That is four call sites and therefore **every** family that reaches the op: K+1 spec verify, fused
TiDAR verify, DDTree tree verify, the canvas, and eager chunked/radix-hit prefill.

**DDTree was NOT exempt.** §D11.3 said "only the DDTree family caps". It does — at
`MINISGL_DDTREE_MAXCTX`, whose default is **2048**, which is still above the kernel's `MIN_CTX` of
1024. The cap moved the crossover; it did not remove it.

**Eager prefill was exposed too, with no graph involved.** The eager row width is the **batch max**
context, so the same request got a different split depending on which requests it was batched with.

### D12.2 The regression test the old tests could not be

`attn_prefill_paged/tests/test_graph_capture.py` captures and replays with the **same** block table,
so the width never varies and the divergence is invisible to it. The new
`tests/test_split_width_invariance.py` states the invariant directly — same inputs, two row widths,
require bit-identical — with **no graph in sight**. Pre-fix it fails 6 of 12 shapes (verify K+1,
long-context verify, chunked prefill; bf16 and fp8, 2.4e-4 to 1.9e-3 absolute). Post-fix all 18,
including the production Qwen3.6-35B-A3B verify geometry and the DDTree capped-table shape, are
`0.000e+00`.

### D12.3 The canvas gate, with `MINISGL_ATTN_MAX_SPLITS` UNSET

```
METADATA DIFF max_seqlen_k: sched=276 static=262144
metadata page_table: values agree over [:1,:18], row width differs sched=(1,18) static=(1,16384)
metadata swa_verify_page_table: values agree over [:1,:276], row width differs sched=(1,276) static=(1,1280)
graph vs eager[matched regime]                    = 0.000e+00 BIT-IDENTICAL
eager self-consistency                            = 0.000e+00
replay self-consistency                           = 0.000e+00
eager-fallback regime (side-stream AR + 2 chunks) = 0.000e+00
LOCALISE: eager[static replay metadata] vs eager[scheduler metadata] = 0.000e+00
          graph vs eager[static replay metadata]                     = 0.000e+00
```

The point is the first three lines together with the fourth: **the width divergence is still there
and is now inert.** The LOCALISE line that read `8.324e+00` reads `0.000e+00`, without the env knob.

### D12.4 What it costs, and the design consequence that is not going away

`tools/canvas_step_time.sh` + `tools/canvas_timing_windows.py` (differenced 10-step windows, TP=2,
bs=1, `MINISGL_SWA_RADIX=0`, `MINISGL_CANVAS_TIMING=1`, seven windows each):

| arm | median step | spread |
|---|---|---|
| `fixed` — split-K selected, `MINISGL_ATTN_MAX_SPLITS` unset | **101.2 ms** | 6.2% |
| `nosplit` — `MINISGL_ATTN_MAX_SPLITS=1` | **91.4 ms** | 10.2% |

Both gate at `0.000e+00`. The ratio, **1.107**, is what split-K costs this canvas, and it agrees with
§D11.4's 97.7 / 86.9 = 1.124 — the ~4 ms level offset against §D11.4 is the timing syncs and a longer
generation (`fwd_tail` climbs 77.9 → 84.7 ms across the windows as context grows). **The fix does not
add a cost; it makes the eager path pay the one the captured graph was already paying**, and
`MINISGL_ATTN_MAX_SPLITS=1` is now a consistent operator lever rather than a workaround for a bug.

**The consequence worth stating.** A captured graph bakes its launch configuration, so the split
decision cannot depend on the running context length — only on numbers fixed for the serve. Context
length therefore no longer informs it and the only discriminator left is SHAPE (`base_grid` vs
`MINISGL_ATTN_PREFILL_FILL_CTAS`). A thin-grid call over a short context now splits where the old
eager path went single-pass; the canvas is exactly that shape (`base_grid` = Hq × 16 q-tiles, under
the 512-CTA fill threshold) over a 276-token context. Making split-K context-aware again needs the
work distribution to move device-side (num_splits is the grid and must stay host-constant, so only
`chunk` can), which is a separate change with its own measurement.

### D12.5 Served regression: none

Interleaved median-of-2, `tools/split_ctx_serve_ab.sh`, base = engine `fdc482a8` + kernels `a45a14d`
in a purpose-built image, provenance asserted per leg on the kernel file that changed:

| model | phase | base | cand | delta |
|---|---|---|---|---|
| GLM-4.7-Flash-AWQ, SPEC=none | bs=1 / 5 / 6 | 61.34 / 195.91 / 237.11 | 61.45 / 196.18 / 238.26 | +0.18% / +0.14% / +0.49% — all NOISE |
| Qwen3.6-35B-A3B-AWQ, SPEC=mtp | bs=1 / 5 / 6 | 43.76 / 129.84 / 157.23 | 43.60 / 129.51 / 157.26 | −0.37% / −0.25% / +0.01% |

GLM reproduces its standing 61.53 / 196.44 / 237.55. Qwen's −0.37% at bs=1 sits just outside a
0.14–0.26% repeat spread and is flat for practical purposes; the captured verify graph's split
decision is **unchanged** by the fix (the static verify row width already equalled the page-table
width), so this is expected — what moved is the eager path, onto the same kernel.

**GLM has no exposure at all**, and the engage ledger says so rather than an argument: it is MLA, its
decode and verify run `mla_hip.mla_decode_fp8` / `mla_verify`, and `attn_prefill_paged` never appears.
Qwen's ledger carries both `gdn_hip.gdn_verify_replay` (the verify graph IS replaying) and
`attn_prefill_paged.flash_prefill_paged_fp8`, which is the exposure, live, on its default config.


---

## Part D13 — the step is NOT host-bound: the sampler's D2H syncs are FALSIFIED as a lever, and §D10's "25% inter-kernel gap" is void

### D13.1 The hypothesis, and where it came from

Upstream review (SGLang `srt/dllm/`, vLLM `models/diffusion_gemma.py`, both at 2026-08-06 HEAD) found
that vLLM's entire denoising step is one `@torch.compile` region with **zero GPU→CPU syncs**, while
this engine's `CanvasState.step` did two per request per step:

```python
stable = all(bool(torch.equal(h, argmax)) for h in self._history)   # D2H
mean_entropy = float(entropy.mean())                                 # D2H
```

Against §D10.2's **"25% inter-kernel gap (24.8 ms/step idle under capture)"** the inference was
obvious: the host drains the pipe every step, the GPU idles until the host comes back and issues the
next forward, and the fix is to make the stopping criterion device-resident and issue the next
forward BEFORE reading it back.

**It is wrong, and the reasoning had a hole that §D11.4 had already opened.** The 25% gap comes from
the same D10 profile whose attention rows §D11.4 voided — taken while the captured path ran
`flash_prefill_paged_fp8_split_kernel` + `flash_prefill_reduce_kernel` on all 30 layers instead of
the single-pass kernel. That is 60 extra dispatches per step, and an inter-kernel GAP is a property
of the dispatch sequence above all else. §D11.4 said the *shares* were void; the gap row is not a
share, it is the dispatch sequence itself, and it should have been voided first.

### D13.2 Three arms, matched k

`tools/diffusiongemma_generate.sh`, `AR=0`, TP=2, bs=1, image `minisgl-rdna4:splitctx`, one boot per
arm. The run is deterministic — every arm realised the SAME k (17 / 23 / 19), so wall time is
directly comparable and none of this is a convergence-luck artifact.

| arm | change | 106 tok (k=17) | 256 tok (k=23) | 256 tok (k=19) |
|---|---|---|---|---|
| 0 | unmodified (`356bfdaa`) | 4.66 s | **2.36 s** | **1.97 s** |
| A | sampler de-sync: `done` device-resident, ONE batched D2H per step instead of two per request | — | 2.37 s (+0.4%) | 1.98 s (+0.5%) |
| B | A + speculative issue of the next forward before the readback | 4.83 s | 2.46 s (+4.2%) | 2.07 s (+5.1%) |

And the measurement that closes it — **ms per model forward**, where arm B does `k+2` per block
(k denoise + 1 causal re-encode + 1 speculative) against arm 0's `k+1`:

| arm | fwd/block | 256 tok (k=23) | 256 tok (k=19) |
|---|---|---|---|
| 0 | k+1 | 98.3 ms | 98.5 ms |
| B | k+2 | 98.4 ms | 98.6 ms |

**Per-forward cost is identical to within 0.2%.** The pipeline made no forward cheaper; it added one.
Arm B's regression is exactly the extra forward (+1 on k+1 ≈ +5.6% predicted, +4.2/+5.1% measured)
with no offsetting gain, which is only possible if the GPU was never waiting on that readback.

### D13.3 What this establishes

* **The canvas step is ~98 ms of GPU work at bs=1 and is NOT host-bound.** The host is off the
  critical path; `fwd_issue` at 0.6–0.8 ms under capture was already telling us this and the gap row
  was the only thing arguing otherwise.
* **§D10.2's 25% inter-kernel gap is VOID**, on the same grounds as its attention rows. Nothing
  should be planned against it until the profile is re-taken on the fixed (§D12) path.
* **Neither change is merged.** Arm A is +0.4/+0.5% — noise, no measured benefit — and shipping
  `resolve()`, tensor-backed `DiffusionStep` properties and a `MINISGL_CANVAS_PIPELINE` env knob for
  that is exactly the env-gated dead weight this repo does not carry. The branch
  `feat/canvas-desync` is retained as the record, not as a candidate.
* **What upstream does here does not transfer.** vLLM needs its sync-free sampler because its
  denoise loop IS the scheduler loop — one full engine step (scheduler → prepare_inputs → forward →
  sampler → post_update) per denoising iteration. This engine already runs the loop worker-resident,
  which is the thing vLLM's own source names as the way to beat it. Same for SGLang's
  `mark_forward_metadata_ready` (plan attention once per block, NPU-only there): it targets a
  per-step FlashInfer `plan()` this engine does not have, and the equivalent host work here is
  inside the 0.6–0.8 ms `fwd_issue`.

### D13.4 Where the levers actually are

Everything left is inside the kernels, and the ranking in §D10.7 survives in MECHANISM only — every
share is a share of a step that was 12% too long, so the first task is re-taking that profile. The
mechanisms that do survive: the MoE grouped GEMM at 40.6% of roofline with every expert touched
every step (256 rows × top-8 over 128 experts), 107.6 all-reduces per step where the lever is FEWER
collectives rather than a faster one, and the dense W4A8 under-occupancy §D11.5 traced to the
occupancy TERM rather than the tie-break.

One transferable kernel observation from the vLLM read, still unmeasured here:
`vllm/v1/attention/ops/triton_unified_attention.py:941-955` retunes for exactly this shape —
*"head_size 256 with many query rows per sequence (diffusion-gemma bidirectional canvas passes) is
prefill-shaped, but the decode-oriented defaults under-tile it … ~2x faster"*. The canvas batch here
is built `phase="decode"` at M=256, and §D10.7 already reports the dense path picking a tile that
launches 64 workgroups on 64 CUs at 12/16 waves. Whether the dispatch keys off phase rather than
actual M is the open question, and it is a measurement, not an argument.

---

## Part D14 — 2026-09-25: it no longer booted, it had regressed 51%, and then 148.6 -> 76.7 ms/step

All numbers: `tools/canvas_step_time.sh` (differenced 10-step windows, TP=2, bs=1, `MINISGL_SWA_RADIX=0`),
graph gate `0.000e+00` on every arm. Fixtures: `/home/pat/fixtures/minisgl-dgopt-20260925/`.

### D14.1 Two breakages found before any optimisation

* **It did not boot.** transformers 5.17 (in the serve image) marks per-layer-overridden attributes
  (`head_dim`, `num_key_value_heads` on the five full layers) and RAISES on a global read, and it
  DROPS `global_head_dim` / `num_global_key_value_heads` in favour of those overrides. Opting into
  global reads alone would have built the full layers at the sliding geometry (256/8, not 512/2).
  Fixed in `cached_load_hf_config` + `ModelConfig._full_layer_override` (8ca22e35).
* **It had regressed 98.4 -> 148.6 ms/step** — the 08-06 engine re-run on today's box gave 98.4, so the
  box was fine. The trace put 51.5 ms of a 184 ms traced step on one fp32 rocBLAS GEMM (`MT16x16x16`,
  16 dispatches/step): the fp32-logits store (the 0.125-grid fix) sent every LM-head call over 16 rows
  to a fallback that cast each 8192-row weight chunk to fp32 — documented as "rare, once per request".
  The canvas scores 256 rows every step. `torch.mm(..., out_dtype=float32)` keeps the fp32 accumulate
  and store with no cast: 53.5 -> 1.71 ms/call (757fcad1). 148.6 -> **90.2 ms**.

### D14.2 What moved the step (each arm measured on its own boot)

| change | step (ms) | where |
|---|---|---|
| baseline, today (boots with D14.1's config fix) | 148.6 | |
| LM-head fallback: one half GEMM with fp32 out | 90.2 | engine 757fcad1 |
| dense W4A8, group 32: 4 groups per barrier (bit-identical) | 87.8 | kernels 0a56376 |
| W4A8 MoE `block_m` rule (canvas 16 -> 32) | 81.4 | engine 8107f884 |
| fused native canvas tail (sampler 4.8 -> ~1.4 ms) | **76.7** | kernels 99b1405 + engine bded6711 |

`MINISGL_ATTN_MAX_SPLITS=1` no longer buys anything (148.6 vs 148.7): the 09-24 per-tile slab split
policy made split-K free at the canvas shape, which closes D12.4's standing 10%.

### D14.3 Findings that outlive the canvas

* **Dense W4A8 at group 32 was a chain of serial memory latencies.** One group (two WMMA k-steps) per
  `__syncthreads` pair: `k_proj` (N=512) and `q_proj` (N=2048) both cost ~87-98 us at M=256. Staging
  four groups per barrier, chosen jointly with the tile by the existing cost model at span 128, is
  1.11-1.40x on the canvas projections, 1.50-2.34x at M=17, bit-identical in every cell (int4 and
  MXFP4). Two M=1024 `o_proj` cells lose 8-9% where the model moves to a 256x96 tile it misprices.
* **The W4A8 MoE `block_m` rule was wrong for 15 of 18 measured cells.** For silu/group-capable
  experts, `block_m=64` engages the register-tiled `gemm1_silu_flag`: Qwen3.6-35B M=64..512 2.2-3.0x
  per MoE layer; 35B cold-prefill TTFT -27..-44% (277->174, 426->237, 624->454, 1186->851 ms), with the
  DFlash CONC=4 verify graphs capturing at the same free memory. The shared analytic chooser picks the
  measured block_m in 3 of the 18 cells, so it does not decide this.
* **Side-stream collectives DO capture** (event fork/join, gate stays 0.000e+00) — `tp_overlap`'s
  "cannot be recorded" was wrong — but the captured step got SLOWER, 87.8 -> 99.3 ms: the spinning
  one-shot all-reduce competes with the grouped GEMM for CUs. Left inline, docstring corrected.
* **Not host-bound, re-confirmed.** py-spy in steady state: 82% of the canvas thread is in
  `synchronize()` waiting on the GPU. The trace's multi-ms inter-phase gaps are the profiler's cost on
  eager launches.

### D14.4 Measured and dropped

* Dense **W4A16** instead of W4A8 at the canvas shapes: 1.4-3.0x SLOWER.
* MoE **GTILE sized in K** (16 groups at g=32): ~2x slower (LDS per block kills occupancy).
* MoE **A-operand / group-scale prefetch** in the ashuffle core: within +-3% of the control.
* **hipBLASLt** for the LM head / soft embedding: identical to rocBLAS (1.73 / 2.53 ms).

### D14.5 Where the 76.7 ms is now (card-1 trace, shares only)

MoE grouped GEMM ~38% (at ~35-39% of HBM whatever the tile — the core's limit needs counters, not
guesses), one-shot all-reduce ~20% (1.44 MB over card 1's Gen4 x8 link, ~99 us of the ~130 us is the
link), dense W4A8 ~14%, soft-embed GEMM (2.7 ms, rocBLAS at ~287 GB/s against the 738 MB shard) and
LM head (1.7 ms), then norms/elementwise, attention ~3.6%.
