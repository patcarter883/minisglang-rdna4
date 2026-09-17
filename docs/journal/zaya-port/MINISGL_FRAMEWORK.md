# minisglang model framework map (for the ZAYA1-8B port)

> Precise, code-quoting map of minisglang's model/cache/engine framework, written for the Zaya CCA
> port. Every claim carries `file:line` + a verbatim snippet. The **GDN hybrid path** (Qwen3.5) is
> the analog for CCA's conv/recurrent state and is covered in depth (§5).
>
> Paths are relative to `/home/pat/code/minisgl-rdna4/python/minisgl/` unless noted.
> Verified against source on 2026-06-27.

---

## 0. The single most important wiring fact (read this first)

A **CCA layer needs BOTH state stores, addressed by two independent index spaces**:

1. A **per-sequence recurrent/conv state cache** (constant size per sequence, NOT prefix-cacheable),
   exactly like `GDNStateCache` (`kvcache/gdn_state.py`). Indexed by a **slot id** owned by a
   uid-keyed slot manager (`scheduler/gdn_slots.py`), reached at forward time via
   `get_global_ctx().gdn_state` + `ctx.batch.gdn_metadata.state_indices`.
2. A **normal paged attention KV cache** for the softmax attention that follows the conv (CCA
   produces normalized q/k/v then does standard partial-RoPE attention). This is `MHAKVCache`
   (`kvcache/mha_pool.py`), indexed by `layer_id` (the cache buffer is `[2, num_layers, …]`) and by
   **out_loc** (a flat token→slot map from the global `page_table`). The GDN *linear*-attention
   layers do NOT use this — but CCA layers DO, because they end in real attention.

This is the structural difference from the GDN hybrid: GDN's linear layers replace attention
entirely, so a GDN layer touches only the recurrent state cache. A Zaya CCA layer is a *conv
front-end feeding a real attention back-end*, so it consumes **a conv-state slot AND a KV-cache
`layer_id`/`out_loc`**. See §6 for the addressing of each.

---

## 1. The model base contract — `models/base.py`

Every `*ForCausalLM` subclasses `BaseLLMModel(ABC, BaseOP)` (`base.py:12`). `BaseOP` is the
minisgl state-dict/meta-load base (NOT `nn.Module`; it walks `__dict__` for sub-ops/tensors —
see §5 for the GDN bridge that this matters for).

**No `__init__` is declared on the base** — concrete models define their own and call
`super().__init__()` at the END (after building sub-ops), e.g. `qwen3_5.py:427`. The constructor
receives exactly one positional arg, the `ModelConfig`:

```python
# register.py:25
return model_cls(model_config)
```

`forward()` is the sole abstract method (`base.py:50`):

```python
@abstractmethod
def forward(
    self, return_hidden: bool = False
) -> Union["torch.Tensor", "Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]"]:
    """Return lm_head logits; or (logits, last_hidden, aux_hidden) if return_hidden."""
```

**`forward()` takes NO batch argument.** All per-step state is pulled from the global context
inside the method (`qwen3_5.py:429-435`):

```python
def forward(self, return_hidden: bool = False):
    input_ids = get_global_ctx().batch.input_ids
    ...
    return self.lm_head.forward(self.model.forward(input_ids))
```

So `forward()` returns `[num_tokens, vocab]` logits for the active batch. `return_hidden=True`
additionally returns `(logits, last_hidden, aux_hidden)` for draft-head spec decode (`base.py:24-28`).
The Zaya port can ignore `return_hidden` for v0 (n-gram spec works without it).

Spec-decode aux capture seam (optional, OFF by default): `set_capture_layers(ids)` (`base.py:34`),
which the default impl forwards to `self.model.set_capture_layers` (`base.py:42-44`).

---

## 2. Registration — `models/register.py`

`_MODEL_REGISTRY` maps an HF `architectures[0]` string to `(module_path, class_name)`
(`register.py:5-16`):

```python
_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    ...
    "Qwen3_5MoeForConditionalGeneration": (".qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
    "Glm4MoeLiteForCausalLM": (".glm4_moe_lite", "Glm4MoeLiteForCausalLM"),
    ...
}
```

`get_model_class` looks up the arch, imports the module relative to `minisgl.models`, gets the
class, and instantiates it with the config (`register.py:19-25`). The arch string comes from
`model_config.architectures[0]` (`models/__init__.py:8`):

```python
def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)
```

**Precise edit to add Zaya:** insert one line into `_MODEL_REGISTRY` —
```python
"ZayaForCausalLM": (".zaya", "ZayaForCausalLM"),
```
(use the exact arch string from `~/models/ZAYA1-8B-fp8/config.json`'s `architectures[0]` — the
ZAYA_REFERENCE doc names `ZayaForCausalLM`; VERIFY). Then create `models/zaya.py` exporting that
class. No other registration step exists.

---

## 3. ModelConfig — `models/config.py`

`ModelConfig` is a frozen dataclass (`config.py:18`) built from an HF `PretrainedConfig` by the
classmethod `from_hf` (`config.py:103`). Core fields the model reads: `num_layers`, `num_qo_heads`,
`num_kv_heads`, `head_dim`, `hidden_size`, `vocab_size`, `intermediate_size`, `rms_norm_eps`,
`rotary_config` (a `RotaryConfig`, `config.py:9`), `hidden_act`, `tie_word_embeddings`,
`num_experts`, `num_experts_per_tok`, `moe_intermediate_size`, `norm_topk_prob`,
`shared_expert_intermediate_size`, `model_type`, `architectures`, `quant` (`config.py:19-40`).

`from_hf` handles a multimodal wrapper by descending into `config.text_config` if present
(`config.py:106-111`) — relevant if Zaya's HF config wraps the text decoder.

**Partial rotary** (Zaya uses `partial_rotary_factor 0.5`) is already handled (`config.py:160-165`):
```python
partial = getattr(config, "partial_rotary_factor", None)
...
rotary_dim = int(head_dim * partial) if partial is not None else head_dim
```
So `rotary_config.rotary_dim < head_dim` flows through `AttentionLayer` → `get_rope` for free
(see §6/§5's `Qwen3_5Attn`, which relies on exactly this for its `rotary_dim=64`).

**MoE knobs** are surfaced: `num_experts` is read from `num_local_experts | num_experts |
n_routed_experts` (`config.py:121-125`). Zaya is `num_experts 16, moe_router_topk 1` →
`num_experts_per_tok` would need to come from `num_experts_per_tok` (`config.py:126`); if Zaya's
config names it `moe_router_topk`, add that fallback in `from_hf`.

`quant` is built by `QuantConfig.from_hf(config)` from the top-level `quantization_config`
(`config.py:105`). Zaya's fp8 experts (`compressed-tensors` float-quant) will surface here.

**`is_*` properties** (`config.py:69-93`) gate framework behavior:
- `is_moe` → `"moe" in model_type` (`config.py:71`).
- `is_mla` → `kv_lora_rank is not None` (`config.py:76`) — Zaya is NOT MLA.
- `is_gdn_hybrid` → `layer_types is not None` (`config.py:81`). **For Zaya you must decide
  whether to reuse this flag or add an analogous `is_cca_hybrid`** (see §8).
- `gdn_layer_ids` (`config.py:84-89`) returns the global indices of linear-attention layers; the
  recurrent state cache is indexed by position in THIS list (`gdn_layer_id`), NOT the global
  `layer_id`. Zaya's CCA layers (even index per ZAYA_REFERENCE §1) need the analogous
  "cca_layer_id" enumeration.

**To add Zaya config fields** (CCA dims: `cca_time0/1`, latent_q/latent_k, conv groups; MoE EDA/MOD
flags): extend the dataclass + populate them in `from_hf`, exactly as the GDN block does
(`config.py:59-67` for the dataclass fields, `config.py:180-194` for the `from_hf` population, which
ONLY populates them when the config carries the linear-attention dims so dense models stay untouched).

---

## 4. Weight loading — `models/weight.py` + `models/utils.py`

The engine loads via `load_weight(model_path, device)` (`weight.py:310`), a **streaming generator**
yielding `(native_key, tensor)` pairs already sharded/merged/on-device. It dispatches on the config:

```python
# weight.py:317-319
if config.is_gdn_hybrid:
    yield from _load_qwen3_5_weight(model_folder, device, config)
    return
```

The dense/default path (`weight.py:320-388`) does, per checkpoint tensor:
1. skip vision/MTP keys (`weight.py:332-343`),
2. strip `language_model.` prefix (`weight.py:350`),
3. **shard** via `_shard_tensor(name, raw, rank, size, num_kv_heads)` (`weight.py:358`) — col-parallel
   (`_SPLIT_DIM_0`, `weight.py:13`) shards dim 0, row-parallel (`_SPLIT_DIM_1`) shards dim 1,
   embed/lm_head are vocab-parallel; AWQ packed tensors flip the axis (`weight.py:16-18, 60-85`),
4. **merge** q/k/v→qkv_proj and gate/up→gate_up_proj via `_MERGE_GROUPS` (`weight.py:21-27, 361-373`),
5. **stack** per-expert tensors over E via `_get_expert_stack_info` (`weight.py:96-105, 375-383`).

**The engine, not the loader, casts dtypes** (`engine/engine.py:251-258`): bf16 weights → model
dtype, `.scales` and quant int-packs preserved, and `.A_log`/`.dt_bias` forced to fp32. A Zaya CCA
layer with fp32 `temp`/conv params will need analogous suffix exceptions here.

**The model declares the native key layout** that the loader targets — i.e. the model's
`state_dict()` keys (from BaseOP's `__dict__` walk) are the contract; the loader's job is to remap
checkpoint names onto them. For a custom name layout (Zaya's `self_attn.cca.*`, `moe.router.*`,
`experts.local_experts.{e}.linear_fc1/2`), you write a dedicated streaming loader + remap mirroring
`_load_qwen3_5_weight` (`weight.py:246-307`) and `qwen3_5_remap` (`weight.py:146-168`), and branch
to it in `load_weight` on a new `config.is_cca_hybrid`-style flag.

The fused MoE expert storage classes the loader fills are in `layers/moe.py`
(`_GroupedGPTQExperts`/`_GroupedAWQExperts`/`_GroupedRXFExperts`, §7). Zaya's fp8 experts
(compressed-tensors float-quant) are NOT one of these three int4 paths — see §7 for what exists.

`models/utils.py` provides the shared building blocks `GatedMLP` (dense SwiGLU, `utils.py:26`),
`MoEMLP` (router gate + `MoELayer`, `utils.py:57`), and `RopeAttn` (standard fused-QKV GQA,
`utils.py:83`). These are the copy-from primitives.

---

## 5. THE GDN HYBRID PATH (the analog for CCA conv/recurrent state)

Files: `models/qwen3_5.py` (4B), `models/qwen3_5_moe.py` (35B MoE), `gdn/layer.py` (compute),
`gdn/metadata.py` (per-batch metadata), `kvcache/gdn_state.py` (state cache),
`scheduler/gdn_slots.py` (slot lifecycle).

### 5.1 How layers are built + scheduled

`Qwen3_5Model.__init__` (`qwen3_5.py:273`) builds `num_layers` decoder blocks, computing each
layer's `gdn_layer_id` from its position in `config.gdn_layer_ids` (`qwen3_5.py:280-289`):

```python
gdn_pos = {gid: pos for pos, gid in enumerate(config.gdn_layer_ids)}
self.layers = OPList([
    Qwen3_5DecoderLayer(config, lid, is_gdn=lid in gdn_pos, gdn_layer_id=gdn_pos.get(lid),
                        mlp_factory=mlp_factory)
    for lid in range(config.num_layers)
])
```

`Qwen3_5DecoderLayer.__init__` (`qwen3_5.py:220`) picks the mixer: a `GDNLinearAttn` bridge for
linear layers, a `Qwen3_5Attn` (gated partial-rotary GQA) for full-attention layers
(`qwen3_5.py:230-248`). The MLP is injected via a `mlp_factory` seam (`qwen3_5.py:251`) — the ONLY
structural difference between the dense 4B and the MoE 35B (`qwen3_5_moe.py:97-100`). The decoder
forward is the standard pre-norm residual pair (`qwen3_5.py:261-270`):

```python
x, residual = self.input_layernorm.forward(x, residual)
x = self._attn_op.forward(x)                 # GDN bridge OR gated attention
x, residual = self.post_attention_layernorm.forward(x, residual)
x = self.mlp.forward(x)
return x, residual
```

For Zaya, a `ZayaDecoderLayer` would similarly branch on layer index (even=CCA mixer, odd=MoE per
ZAYA_REFERENCE §1) — but the CCA mixer is `conv_front_end + standard_attention`, so it wraps BOTH a
conv-state read AND a `AttentionLayer` (§6), unlike the GDN bridge which has no attention.

### 5.2 The state cache — `kvcache/gdn_state.py`

`GDNStateCache` (`gdn_state.py:6`) is a **simple slot allocator (free-list), NOT a paged/radix
structure** — fixed size per sequence, not prefix-cacheable (`gdn_state.py:10-13`). Two buffers,
both indexed `[gdn_layer_id, slot]` (`gdn_state.py:59-66`):

```python
self.conv_state = torch.zeros(
    (num_gdn_layers, num_slots, conv_dim, conv_kernel - 1), dtype=dtype, device=device)
self.ssm_state = torch.zeros(
    (num_gdn_layers, num_slots, num_v_heads, head_v_dim, head_k_dim), dtype=ssm_dtype, device=device)
```

★ **Slot 0 is the reserved NULL block, never handed out** (`gdn_state.py:18-27, 69-70`); the
free-list starts at slot 1: `self._free = list(range(num_slots - 1, 0, -1))`. The kernels treat
`slot==0` as padding (output left unwritten). Allocation: `alloc_many(n)` returns an int32 device
tensor of slot ids (`gdn_state.py:72-77`); `free`, `reset_slots`, `snapshot`/`restore`,
`install_verify_state` (spec) round out the API. Per-layer views: `conv(gdn_layer_id)` /
`ssm(gdn_layer_id)` (`gdn_state.py:132-138`).

**Zaya CCA analog:** the reference (ZAYA_REFERENCE §2) needs `conv_states [NB, C=1280, TP=2]`
(fp32) and `prev_hs [NB, hidden=2048]`. So a `CCAStateCache` would carry two buffers shaped
`(num_cca_layers, num_slots, 1280, 2)` and `(num_cca_layers, num_slots, 2048)`, same slot-0-NULL
free-list discipline. The kernel arg order (`cca_hip/cca_op.py`) takes `conv_states` + `slot` +
`is_pad`, mirroring the GDN `state_indices` + null-slot convention exactly.

### 5.3 Sizing + allocation — `engine/engine.py`

The engine constructs the state cache ONLY for GDN-hybrid models (`engine.py:128-152`):

```python
if mc.is_gdn_hybrid:
    tp = config.tp_info.size
    self.ctx.gdn_state = self.gdn_state = GDNStateCache(
        num_gdn_layers=mc.num_gdn_layers,
        num_slots=config.max_running_req + 2,   # +1 NULL block, +1 dummy
        conv_dim=div_even(mc.gdn_conv_dim, tp),
        conv_kernel=mc.linear_conv_kernel_dim,
        num_v_heads=div_even(mc.linear_num_value_heads, tp),
        head_v_dim=mc.linear_value_head_dim,
        head_k_dim=mc.linear_key_head_dim,
        dtype=torch.float32,                    # conv state always fp32
        ssm_dtype=(torch.bfloat16 if MINISGL_SSM_BF16 else torch.float32),
        device=self.device,
    )
    for gdn in self.model.iter_gdn_layers():
        gdn.warmup_conv(_GDN_WARMUP_TOKENS)
```

`num_slots = max_running_req + 2` (`engine.py:137`). It is exposed to layers via
`self.ctx.gdn_state` (a field on `Context`, `core.py:118-120`). The model must expose
`iter_gdn_layers()` so the engine can size + warm the layers (`qwen3_5.py:440-447`). **Zaya needs an
`is_cca_hybrid` branch here building a `CCAStateCache`** (and CCA touches the KV cache too, which is
already allocated unconditionally just above at `engine.py:110-118`).

### 5.4 Slot lifecycle — `scheduler/gdn_slots.py`

`GDNSlotManager` (`gdn_slots.py:40`) keys the slot by **uid** (the only identity stable across fresh
prefill / chunked continuation / prefill→decode / finish — `gdn_slots.py:9-27`):

```python
def state_indices(self, batch: Batch) -> torch.Tensor:
    reqs = batch.reqs
    if batch.is_prefill:
        self._ensure_slots(reqs)          # alloc+zero fresh uids only
    idx = [self._slot_of[req.uid] for req in reqs]
    return torch.tensor(idx, dtype=torch.int32, device=self.device)
```

`_ensure_slots` (`gdn_slots.py:61-70`) allocates only NEW uids and zeroes ONLY those slots
(continuations keep accumulated state). `free(uid)` is idempotent (`gdn_slots.py:72-77`). The
scheduler wires it in `__init__` (`scheduler.py:82-87`), per-batch in `_prepare_batch`
(`scheduler.py:271-275`), and on finish in `_free_req_resources` (`scheduler.py:258`). A Zaya
`CCASlotManager` is a near-verbatim copy.

### 5.5 Per-batch metadata — `gdn/metadata.py`

`build_gdn_metadata(batch, state_indices, device)` (`metadata.py:66`) packages exactly what the
layer consumes (`metadata.py:42-47`): `is_prefill`, `num_seqs`, `query_start_loc` (cu_seqlens int32
`(num_seqs+1,)`), `state_indices` (slot per seq, all ≥1), `has_initial_state` (bool, prefill only).
Decode builds `query_start_loc = arange(num_seqs+1)` (`metadata.py:89`); prefill builds a cumsum of
`req.extend_len` (`metadata.py:102-107`) and `has_initial_state = [req.cached_len > 0 …]`
(`metadata.py:111-113`). Stashed on `batch.gdn_metadata` (`core.py:85`).

### 5.6 The bridge + compute — `gdn/layer.py`

`GDNLinearAttn` (`qwen3_5.py:150`) is a BaseOP bridge wrapping the `nn.Module` `QwenGatedDeltaNet`
(its params live in `nn._parameters`, invisible to BaseOP's `__dict__` walk, so `state_dict`/`load`
are delegated — `qwen3_5.py:202-217`). Its `forward` pulls state + metadata from the global ctx and
**dispatches prefill vs decode** (`qwen3_5.py:168-197`):

```python
ctx = get_global_ctx()
state = ctx.gdn_state
md = ctx.batch.gdn_metadata
conv = state.conv(self._gdn_layer_id)
ssm = state.ssm(self._gdn_layer_id)
if ... spec verify with capture ...:
    out, conv_scr, ssm_scr = self._gdn.forward_prefill_verify(...)
elif ctx.batch.is_prefill or ctx.batch.spec_verify:
    out = self._gdn.forward_prefill(x, conv, ssm, md.query_start_loc, md.state_indices, md.has_initial_state)
else:
    out = self._gdn.forward_decode(x, conv, ssm, md.query_start_loc, md.state_indices)
if self._tp_size > 1:
    out = self._comm.all_reduce(out)
```

The compute module (`gdn/layer.py:52`) calls **native HIP ops** (`torch.ops.gdn_hip.*`, imported
lazily as `from gdn_hip import op as gdn`, `layer.py:185`). Prefill (`layer.py:175-227`):
`gdn.causal_conv1d_fwd(...)` updates `conv_state` in place per slot, then `gdn.gdn_prefill_wmma(...)`
updates `ssm_state` in place. Decode (`layer.py:282-313`): `gdn.causal_conv1d_update(...)` +
`gdn.gdn_decode(...)`, both in-place per slot. **State is updated IN PLACE, keyed by `state_idx =
state_indices.long()`** (`layer.py:191, 296`). This is the exact pattern Zaya's `cca_decode_qk` /
`cca_prefill_qk` follow (the vendored `cca_op.py` shows `conv_states` declared mutable and `slot`
passed in — §0).

---

## 6. Engine forward / metadata / what a layer learns each step

### 6.1 Batch, Req, ForwardMode — `core.py`

There is **no `ForwardMode` enum**; the mode is `Batch.phase: Literal["prefill","decode"]`
(`core.py:73-76`) with `is_prefill`/`is_decode` properties (`core.py:93-99`). minisgl schedules
**homogeneous batches** (all-prefill XOR all-decode), so a layer never sees a mixed batch — the
vLLM prefill/decode split is unnecessary (`gdn/metadata.py:3-7`).

`Req` (`core.py:29-69`) carries `input_ids` (cpu), `table_idx` (its row in the global page table),
`cached_len`, `output_len`, `uid`, `sampling_params`, `cache_handle`. Derived:
`extend_len = device_len - cached_len` (`core.py:50-51`) = tokens processed THIS pass;
`device_len`/`max_device_len` set in `__post_init__` (`core.py:39-43`).

`Batch` (`core.py:72-108`) — scheduler-set fields: `input_ids`, `positions`, `out_loc`,
`padded_reqs`; backend-set: `attn_metadata`; GDN-only: `gdn_metadata` (`core.py:85`); spec-only:
`spec_verify` (`core.py:91`).

### 6.2 Context + the forward-batch contract — `core.py`

`Context` (`core.py:110`) holds `page_size`, `page_table`, `attn_backend`, `moe_backend`,
`kv_cache`, and the GDN-only `gdn_state` (`core.py:118-120`). The active batch is set for the
duration of a forward via the `forward_batch` context manager (`core.py:128-135`), reached anywhere
through `get_global_ctx().batch` (`core.py:123-126, 147`). This is HOW `forward()` gets its batch
without an argument (§1).

### 6.3 What the scheduler computes per step — `scheduler.py:_prepare_batch`

`_prepare_batch` (`scheduler.py:260-281`) sets, in order: `positions` (`_make_positions`,
`scheduler.py:583`), the input/write token maps, `out_loc = engine.page_table[input_mapping]`
(`scheduler.py:266`) — the **flat token→KV-slot map** the attention store/load uses — then
`attn_backend.prepare_metadata(batch)` (`scheduler.py:267`), then (GDN-only) `state_indices` +
`build_gdn_metadata` (`scheduler.py:271-275`). So a layer can learn, per step: which reqs
(`batch.reqs`), their seq positions (`batch.positions`), the per-token KV slot (`batch.out_loc`),
the page table (`ctx.page_table[req.table_idx]`), the conv-state slot per seq
(`gdn_metadata.state_indices`), and prefill-vs-decode (`batch.phase`).

### 6.4 The normal attention KV cache — `kvcache/mha_pool.py` + `layers/attention.py`

`MHAKVCache` (`mha_pool.py:10`) allocates one big buffer (`mha_pool.py:28-32`):

```python
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim), ...)
```

So it is indexed by `[k_or_v, layer_id, page, slot, head, dim]`. `store_kv(k, v, out_loc, layer_id)`
(`mha_pool.py:45-55`) scatters new K/V at `out_loc` for that layer. The pool is created
unconditionally in the engine (`engine.py:112-118`) with `num_pages = self._determine_num_pages(...)`
sized from `head_dim * num_kv_heads * page_size * num_layers` (`engine.py:274-282`).

`AttentionLayer` (`layers/attention.py:18`) is the per-layer attention op every model attention site
builds. Its `forward(qkv)` (`attention.py:47-57`): split q/k/v, optional q/k RMSNorm, rotary over
`ctx.batch.positions`, then `ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)`. The
backend (`attention/hip.py:49-72`) calls `self.kvcache.store_kv(k, v, batch.out_loc, layer_id)`
then dispatches prefill/extend/decode on `metadata.cold_prefill` / `metadata.max_seqlen_q`.

`RDNA4Metadata` (`attention/triton_rdna4.py:18-27`): `cache_seqlens` (per-seq total KV len),
`cu_seqlens_q` `[bs+1]`, `max_seqlen_q`, `page_table` `[bs, max_pages]`, `cold_prefill`. Built by the
inherited `prepare_metadata` (`triton_rdna4.py` ~`:225-240`).

**CONFIRMED for Zaya:** a CCA layer that ends in standard softmax attention reuses this whole
mechanism verbatim — build an `AttentionLayer(layer_id=…)` for the attention back-end, and it gets
its own `layer_id` slice of `_kv_buffer` and writes via `out_loc`. The `layer_id` it passes must be
unique per attention site across the model (the buffer is sized `num_layers`); for Zaya, where CCA
layers are interleaved, the attention-bearing layers each need a distinct contiguous attention
`layer_id` (decide whether `num_layers` for the KV pool = number of CCA layers, since only CCA
layers do attention — VERIFY against `_determine_num_pages` sizing, which uses `mc.num_layers`).

---

## 7. MoE building blocks — `layers/moe.py`, `qwen3_5_moe.py`, `models/utils.py`

The reusable MoE op is `MoELayer` (`moe.py:123`). Construction
(`MoELayer(num_experts, top_k, hidden_size, intermediate_size, renormalize, activation,
apply_router_weight_on_input, quant)`). With `quant=None` it allocates plain stacked expert weights
(`moe.py:171-181`) and routes through `ctx.moe_backend.forward(...)` (a fused softmax+topk kernel,
`moe.py:226-238`). With an int4 quant it uses `_GroupedGPTQExperts`/`_GroupedAWQExperts`/
`_GroupedRXFExperts` (`moe.py:16-121`) and `kernels.w4a8_moe` / `kernels.rxf_moe` (`moe.py:194-225`).

`forward` accepts EITHER raw `router_logits` (fused softmax+topk inside the kernel) OR precomputed
`topk_weights`/`topk_ids` (the GLM/DeepSeek `noaux_tc` path computed in the model) — `moe.py:183-193`.
**Zaya's router (ZayaRouter: down_proj→RMSNorm→GELU MLP→logits, plus balancing_biases + top-1 +
MOD skip-expert) is a custom router computed IN THE MODEL** → it should produce `topk_weights`/
`topk_ids` and pass them to `MoELayer.forward` via the precomputed path (like GLM noaux_tc), since
the fused-kernel softmax+topk can't express EDA/MOD/balancing-bias.

**Shared expert** pattern (qwen3_5_moe) — `Qwen3_5MoeSparseBlock` (`qwen3_5_moe.py:57`):
```python
shared_out = self.shared_expert.forward(hidden_states)
shared_out = torch.sigmoid(self.shared_expert_gate.forward(hidden_states)) * shared_out
router_logits = self.gate.forward(hidden_states)
routed_out = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
return (routed_out + shared_out).view(num_tokens, hidden_dim)
```
The shared expert is an UNQUANTIZED SwiGLU (`qwen3_5_moe.py:40-54`). Zaya has NO Qwen-style shared
expert; its MOD "skip expert" is different (output = input * route_prob) and is model-side logic.

**fp8 experts:** ⚠️ Zaya's experts are `compressed-tensors` **float (fp8) quant**, which is NOT one
of the three int4 grouped paths in `moe.py` (GPTQ/AWQ/RXF are all W4). There is **no existing fp8
fused-MoE expert path** in `layers/moe.py`. Options for the port: (a) run experts unquantized (bf16,
`quant=None`) for v0 correctness, or (b) add an fp8 grouped-expert storage class + kernel. Check
`minisgl/quant/` for any fp8 dense kernel to reuse — memory notes a `w4a8_fp8_wmma` and the serve
image has no `sgl_kernel`; treat fp8 MoE as a real port sub-task, not a reuse. (Per the memory
"serve-image-fused-op-availability", `vllm._custom_ops.topk_softmax` is the route-fuse used in
production; relevant for the router top-k.)

For non-MoE models the dense MLP primitive is `GatedMLP` (`utils.py:26-54`).

---

## 8. Serve/run entry, `--attn hip`, graph capture, per-model state hooks

### 8.1 Attention backend selection — `engine/engine.py`

`--attn` maps to `config.attention_backend`; `_adjust_config` (`engine.py:367-419`) resolves
`"auto"` → `"triton_rdna4"` on ROCm (`engine.py:383-390`). `--attn hip` selects `HIPAttnBackend`
(`attention/hip.py:37`) — the Triton-free native-HIP backend, **the only capture-capable production
path** (per the production-serve-config memory). The backend is created at `engine.py:176-178`. The
`triton_rdna4` backend forces `page_size` to a multiple of 16 (`engine.py:398-400`); the hip backend
inherits that.

### 8.2 Graph capture — `engine/graph.py`, `gdn/graph_capture.py`

`GraphRunner` is built at `engine.py:201-213` and given `gdn_state=self.gdn_state`. `forward_batch`
(`engine.py:316-332`) replays a captured graph when `graph_runner.can_use_cuda_graph(batch)`, else
runs eager `self.model.forward()`. The attention backend exposes the capture hooks
(`prepare_for_capture`/`prepare_for_replay`, `attention/hip.py:163-169`) that refresh STATIC decode
buffers (`cache_seqlens`, `page_table`) the captured kernel reads. **GDN graph capture exists**
(`gdn/graph_capture.py`, threading recurrent-state slots through static buffers — `engine.py:199`).
**Spec decode disables capture** (eager verify, `engine.py:412-418`).

For the Zaya port v0, run **eager** (the GDN MVP did — `gdn/metadata.py:26`), and add capture later;
CCA's in-place conv-state update is graph-capturable the same way GDN's is (static slot buffers).

### 8.3 Per-model state-cache hook

The "hook" is the trio: `config.is_gdn_hybrid` (gates the engine branch), the engine's
`GDNStateCache` construction (`engine.py:129-161`), and the model's `iter_gdn_layers()`
(`qwen3_5.py:440`) for sizing + warmup. The state cache is forced **non-radix ("naive") prefix
cache** in the scheduler (`scheduler.py:62-73`) because recurrent state is not prefix-cacheable —
Zaya's CCA conv state has the SAME constraint and MUST force naive too. There is no generic
"register a state-cache shape" API; you add an `is_cca_hybrid` branch alongside the GDN one in
`engine.py` + `scheduler.py` + a `iter_cca_layers()` on the model.

---

## Summary — the 10 load-bearing wiring decisions for the Zaya CCA port

1. **A CCA layer uses BOTH caches.** Conv/`prev_hs` state via a slot-allocator `CCAStateCache`
   (analog of `GDNStateCache`), AND the normal paged `MHAKVCache` for the softmax attention that
   follows the conv — GDN layers use only the first; CCA adds the second.
2. **Conv state is addressed by uid→slot.** Copy `GDNSlotManager` (`gdn_slots.py`): uid-keyed,
   slot-0 reserved NULL, idempotent free, zero-only-fresh. Cache buffer indexed `[cca_layer_id, slot]`.
3. **Attention KV is addressed by `layer_id` + `out_loc`.** Build a standard `AttentionLayer(layer_id)`
   inside each CCA mixer; it writes/reads `_kv_buffer[layer_id]` via `batch.out_loc` and the page
   table automatically (§6.4). Assign each attention-bearing CCA layer a distinct `layer_id`.
4. **Force the naive prefix cache + eager** for Zaya (`scheduler.py:62-73`, `engine.py:412`), exactly
   as GDN does — conv/recurrent state is not prefix-cacheable.
5. **Register one line** in `_MODEL_REGISTRY` (`"ZayaForCausalLM": (".zaya", "ZayaForCausalLM")`) and
   add an `is_cca_hybrid` property + CCA config fields in `config.py` (`from_hf` populates them only
   when present — keep dense models untouched).
6. **Per-batch metadata mirrors `GDNMetadata`**: `query_start_loc` (cu_seqlens), `state_indices`
   (conv slot per seq), `has_initial_state` (prefill); build it in `_prepare_batch` and stash on
   `batch.cca_metadata`. The vendored CCA kernels take `slot`/`is_pad`/`seg_pos`/`req_id`/`is_last`
   in the order `cca_hip/cca_op.py` declares — trust that file.
7. **In-place conv-state update keyed by slot**, prefill vs decode dispatched on `batch.phase`
   (`zaya_cca.cca_prefill_qk` for prefill, `cca_decode_qk` for decode), exactly like the GDN bridge
   (`qwen3_5.py:189-194`).
8. **The custom ZayaRouter (EDA/MOD/balancing-bias, top-1) computes route in the model** and feeds
   `MoELayer.forward(..., topk_weights=, topk_ids=)` via the precomputed path (`moe.py:183-193`),
   like GLM noaux_tc — the fused softmax+topk kernel can't express EDA/MOD.
9. **fp8 experts have NO existing reuse path.** `layers/moe.py` only has W4 (GPTQ/AWQ/RXF) grouped
   experts. Start unquantized (bf16) for correctness, then add an fp8 grouped-expert kernel as a
   distinct sub-task.
10. **Engine + scheduler wiring is explicit, not pluggable.** Add `is_cca_hybrid` branches in
    `engine.py` (build `CCAStateCache`, set `ctx.cca_state`, warm conv) and `scheduler.py` (slot
    manager + per-batch metadata + free-on-finish), plus `iter_cca_layers()` + `set_capture_layers`
    on the model. Run eager v0; graph capture follows the GDN static-slot pattern later.
