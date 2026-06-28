# ZAYA1-8B port plan (minisglang / RDNA4)

> Decision-complete implementation plan for `python/minisgl/models/zaya.py` and the engine/scheduler
> wiring. Architecture calls are MADE here; implementers should not need to invent structure. Every
> non-obvious claim is grounded in a source file read on 2026-06-27:
> - Reference model math: `/home/pat/code/vllm-gfx1201/zaya/overlay/vllm/model_executor/models/zaya.py`
> - CCA layer math: `/home/pat/code/vllm-gfx1201/zaya/overlay/vllm/model_executor/layers/mamba/cca.py`
> - Kernel arg order (AUTHORITATIVE): `cca_hip/cca_op.py` + `cca_hip/bindings.cpp`
> - Live config: `/home/pat/models/ZAYA1-8B-fp8/config.json`
> - **Live checkpoint key names + shapes/dtypes inspected from the safetensors index/headers on
>   2026-06-27 (authoritative over the reference doc's guessed names).**
> - minisgl analogs: `models/qwen3_5.py`, `gdn/*`, `kvcache/*`, `layers/{attention,moe}.py`, `models/config.py`
>
> Companion docs: `MINISGL_FRAMEWORK.md` (framework map), `ZAYA_REFERENCE.md` (reference map).

---

## 0. Confirmed facts (verified against source — supersede the reference doc where they differ)

| Fact | Source | Notes |
|---|---|---|
| `architectures[0] == "ZayaForCausalLM"` | config.json | registry key |
| `model_type == "zaya"` (NOT "...moe") | config.json | ⚠ `ModelConfig.is_moe` is `"moe" in model_type` → **False for Zaya**. Do NOT rely on `is_moe`. See §8. |
| **Layer schedule: `layer_n % 2 == 0` → CCA attention; `% 2 == 1` → MoE** | zaya.py:644-671 | 80 layers → 40 CCA, 40 MoE. **Even=CCA confirmed.** |
| CCA: q_heads=8, k_heads=2 (`num_query_groups`), head_dim=128, gqa=4 | zaya.py:134-138, config | latent_q=1024, latent_k=256, C=1280 |
| conv: K0=cca_time0=2, K1=cca_time1=2, total_padding TP=2 | cca.py:108-110 | conv_states width = TP = 2 |
| RoPE partial 0.5, theta 5e6, neox style, applied AFTER CCA op | zaya.py:187-216 | rotary_dim = 64 |
| MoE: 16 experts, top-1, ffn 4096, SwiGLU/silu; experts fp8 compressed-tensors float-quant | config, zaya.py:502-512 | intermediate per expert = `ffn_hidden_size//2 = 2048` (see §7 ⚠) |
| Router (`ZayaRouter`): down_proj 2048→256(+bias), EDA add, RMSNorm(256), GELU MLP `router_mlp.{0,2,4}` 256→256→256→17, balancing_biases[17], top-1 | zaya.py:290-447 + ckpt | router stays bf16/fp32, NEVER quantized. router_mlp final dim = 17 (16+MOD skip), verified |
| **Checkpoint prefixes (VERIFIED): CCA = `self_attn.qkv.*`; MoE = `zaya_block.*`; res_scale = layer-level `layers.{i}.res_scale.*`** | safetensors index | reference doc's `self_attn.cca.*`/`moe.*` names are WRONG — see §9 |
| Experts fp8: `linear_fc1.weight` `[4096,2048]` F8_E4M3 + `weight_scale` `[4096,1]` F32; `linear_fc2` `[2048,2048]` + `[2048,1]` | safetensors headers | per-output-channel (dim-0) scale; dequant = `w.float()*scale` |
| EDA skipped on `layer_number == 1` (the FIRST MoE layer); active on all later MoE layers | zaya.py:344-350 | `zaya_first_layer = 1`; threads `prev_router_hidden_states` |
| MOD: extra "skip" expert at index `num_experts` (=16); balancing_biases[-1]=-1.0; if skip → out = input * route_prob | zaya.py:381-382, 524-534 | balancing_biases length = num_experts+1 = 17 |
| `tie_word_embeddings: true`; lm_head tied to embed_tokens | config, zaya.py:844-845 | |
| `zaya_high_prec: true` (config default) → router softmax fp32 + fp32 lm_head logits | config py:37, zaya.py:418-421, 851-852 | v0 may keep logits bf16; router softmax SHOULD be fp32 |
| `residual_in_fp32: true`, `scale_residual_merge: true` | config | **Non-standard residual scheme — see §3.1. This is the trickiest math to port.** |
| State dtype = fp32 (`mamba_cache_dtype: float32`) | config | conv_states + prev_hs fp32 |
| CCA has NO TP split (runs replicated) | zaya.py:807-813 | port targets **TP=1** (8B fits 16GB); replicate if TP>1 later |

**Kernel signatures (verbatim from `bindings.cpp` / `cca_op.py` — AUTHORITATIVE arg order):**
```
cca_decode_qk (Tensor qk_new, Tensor(a!) conv_states, Tensor slot, Tensor is_pad,
               Tensor w0, Tensor b0, Tensor w1, Tensor b1,
               Tensor temp_eff, int num_q, int gqa, int latent_q, float sqrt_d) -> qk_out
cca_prefill_qk(Tensor qk_new, Tensor(a!) conv_states, Tensor init_states, Tensor seg_pos,
               Tensor req_id, Tensor slot, Tensor is_last,
               Tensor w0, Tensor b0, Tensor w1, Tensor b1,
               Tensor temp_eff, int num_q, int gqa, int latent_q, float sqrt_d) -> qk_out
conv_state_decode(...) -> qk_out  # NOT USED by the port (subsumed by cca_decode_qk)
```
- `qk_new`/`qk_out`: `[N, C=1280]` fp32. `conv_states`: `[NB, 1280, 2]` fp32, mutated in place.
- `qk_out` is the **normalized q|k concatenation** (means + per-head RMS-norm + temp already baked in):
  columns `[0:1024]` = q, `[1024:1280]` = k. Value (`v`) is computed in the MODEL, not the kernel.
- `num_q=8, gqa=4, latent_q=1024, sqrt_d=sqrt(128)`. `temp_eff` = `exp(clamp(temp,1e-7,2.0))` if
  `clamp_temp` else `temp` (config `clamp_temp=False` → `temp_eff = temp`), fp32 `[2]`.
- `w0`:[1280,2] (squeeze of conv_qk.0.weight), `b0`:[1280], `w1`: **pre-transposed** `[H=10,d=128,d=128,K1=2]`
  (from conv_qk.1.weight `[1280,128,2]` → view `[10,128,128,2]` → permute `[10,128(out),128(in),2]`),
  `b1`:[1280]. H here = num_k_heads+num_q_heads = 10 (the grouped-conv groups). See cca.py:1046-1074.

---

## 1. Module breakdown — `python/minisgl/models/zaya.py`

All classes subclass `BaseOP` (the minisgl state-dict base) except where noted. Mirror `qwen3_5.py`
structure. The CCA conv front-end is wrapped in a `nn.Module` bridge exactly like `GDNLinearAttn`
wraps `QwenGatedDeltaNet`, because the conv weights (`conv_qk.0/1`, `temp`, the projections) are
cleanest to hold as plain `nn.Parameter`s loaded via `assign=True` (fp32-preserving) the same way GDN does.

```
ZayaForCausalLM(BaseLLMModel)          # registry target; forward() pulls batch from global ctx
  ├─ model: ZayaModel(BaseOP)
  │    ├─ embed_tokens: VocabParallelEmbedding(vocab=262272, hidden=2048)
  │    ├─ layers: OPList[ ZayaDecoderLayer × 80 ]   # branch on layer_n%2
  │    ├─ res_scale_final: ResidualScaling(layer_n=80)   # scale_residual_merge final merge
  │    └─ final_norm: RMSNorm(2048, eps=1e-5)            # PLAIN weight (init 1), NOT plus_one
  └─ lm_head: ParallelLMHead(tie_word_embeddings=True, tied_embedding=embed_tokens)

ZayaDecoderLayer(BaseOP)               # one class, branches by is_cca. res_scale is LAYER-level
   self.input_norm: RMSNorm(2048, eps=1e-5)            # plain weight; on EVERY layer
   self.res_scale:  ResidualScaling(layer_n)           # ckpt key layers.{i}.res_scale.* (layer 0: no residual_*)
   if even (CCA):
     self.self_attn:  ZayaCCAAttn(BaseOP)              # ckpt self_attn.{qkv.*, o_proj}
   else (MoE):
     self.zaya_block: ZayaMoEBlock(BaseOP)             # ckpt zaya_block.{router.*, experts.*}

ZayaCCAAttn(BaseOP)                    # conv front-end + standard partial-RoPE attention
   ├─ _cca:  CCAConv(nn.Module)        # bridge holding conv weights/projections (see §1.1)
   ├─ attn:  AttentionLayer(layer_id=<cca attn id>, nqo=8, nkv=2, head_dim=128,
   │                        rotary_config=partial-0.5, q_norm=None, k_norm=None)
   └─ o_proj: LinearOProj(1024 → 2048, no bias)

ZayaMoEBlock(BaseOP)
   ├─ router:  ZayaRouter(nn.Module)   # bridge: down_proj, rmsnorm_eda, router_mlp, biases, EDA scale
   └─ experts: MoELayer(num_experts=16, top_k=1, hidden=2048, intermediate=2048,
                        renormalize=False, activation="silu", quant=<fp8 or None>)

CCAConv(nn.Module)                     # nn params for assign=True fp32-preserving load
   ├─ linear_q  [2048→1024], linear_k [2048→256], val_proj1 [2048→128], val_proj2 [2048→128]
   ├─ conv_qk.0 (depthwise Conv1d 1280,k=2,bias), conv_qk.1 (grouped Conv1d 1280,g=10,k=2,bias)
   └─ temp [2]    # all of these are NOT quantized (config ignore list: re:.*self_attn.*)

ZayaRouter(nn.Module)                  # nn params (bf16/fp32), assign=True load
   ├─ down_proj [2048→256, bias], rmsnorm_eda [256], router_states_scale [256] (EDA layers only)
   ├─ router_mlp: Seq( Linear256→256+b, GELU, Linear256→256+b, GELU, Linear256→16 )
   └─ balancing_biases buffer [17] (fp32; [-1]=-1.0)
```

### 1.1 Why bridge `CCAConv` and `ZayaRouter` as `nn.Module` (not pure BaseOP)

The CCA conv weights and the router are **never quantized** (config `ignore` list:
`re:.*self_attn.*`, `re:.*router.*`) and several must stay non-bf16 (`temp` and the conv weights are
consumed fp32 by the kernel; the engine casts bf16 by default — `engine.py:251-258`). Holding them as
`nn.Parameter`s loaded with `assign=True` (the GDN bridge pattern, `qwen3_5.py:202-217`) preserves
dtype and keeps the load path simple. Pre-compute the fp32 kernel weights (`w0,b0,w1,b1` transposed,
`temp_eff`) ONCE in `CCAConv.post_load()` and cache them (mirror cca.py `_conv_weights_fp32` /
`_temp_eff`), so the hot path never re-derives them.

`linear_q/k`, `val_proj1/2`, `o_proj` are bf16 dense — they can be plain BaseOP `LinearColParallel`/
`LinearOProj` OR live inside the `CCAConv` nn bridge. **Decision: put the four input projections
(`linear_q/k/val_proj1/2`) and `conv_qk`+`temp` inside `CCAConv`** (one load surface for the
checkpoint's `self_attn.qkv.*` prefix — the CCA module is literally named `qkv` in the checkpoint);
keep `o_proj` as a BaseOP `LinearOProj` on `ZayaCCAAttn` (matches checkpoint `self_attn.o_proj.*`,
contracts 1024→2048). The loader (§9) remaps `self_attn.qkv.*` → the `CCAConv` (`_cca`) params.

---

## 2. Layer schedule — how to express it

`layer_n % 2 == 0` is CCA, odd is MoE (zaya.py:644-671). Compute two enumerations in `ZayaModel.__init__`
(mirroring `gdn_pos` in `qwen3_5.py:280`):

```python
cca_layer_ids = [lid for lid in range(num_layers) if lid % 2 == 0]   # 40 entries
cca_attn_id   = {lid: pos for pos, lid in enumerate(cca_layer_ids)}  # global -> [0..39]
self.layers = OPList([
    ZayaDecoderLayer(config, lid, is_cca=(lid % 2 == 0),
                     cca_layer_id=cca_attn_id.get(lid),     # index into CCA state cache AND KV pool
                     mlp_layer_n=lid)
    for lid in range(num_layers)
])
```

`cca_layer_id` (0..39) is BOTH:
- the index into the `CCAStateCache` (conv_states/prev_hs are `[num_cca_layers, slot, ...]`), and
- the `layer_id` passed to `AttentionLayer` (only CCA layers do attention → the KV pool needs only
  40 layer slices, see §4.2). Use the SAME id for both; it is contiguous over attention-bearing layers.

Add to `ModelConfig` (§8): `is_cca_hybrid` property, `cca_layer_ids` property, `num_cca_layers`,
plus CCA dims (`cca_time0/1`, `cca_num_k_heads`, `cca_num_q_heads`, `cca_head_dim`) and MoE/router
dims (`zaya_mlp_expansion`, `zaya_use_eda`, `zaya_use_mod`, `scale_residual_merge`).

---

## 3. Residual scheme (the #1 correctness risk — port EXACTLY)

Zaya does **NOT** use minisgl's `RMSNormFused(x, residual)` pre-norm pattern. The reference does, per
layer (zaya.py:272-280 / 593-601), in fp32:

```
# inputs: hidden_states (this layer's pre-input), residual (fp32 running stream, None at layer 0)
if scale_residual_merge:
    residual, hidden_states = res_scale(residual, hidden_states)   # affine, see §3.1
residual = (residual.float() if residual is not None else 0) + hidden_states.float()  # fp32 merge
hidden_states = input_norm(residual).to(layer_dtype)               # RMSNorm over the MERGED fp32 residual
... mixer (CCA or MoE) consumes hidden_states ...
return hidden_states, residual    # residual carries forward (fp32); mixer output becomes next hs
```

Final (zaya.py:723-736): one more `res_scale` merge, `hidden_states = hs.float()+residual.float()`,
then `final_norm`, then lm_head.

**Key differences from minisgl's standard decoder:** (1) the residual is the FULL fp32 stream and the
norm is applied to `residual` (the merged stream), not to a separately-tracked `x`; (2) the layer
RETURNS `(mixer_output, residual)` where the mixer output becomes the next layer's `hidden_states`
input (NOT added back in-layer — the add happens at the TOP of the next layer). So do **not** reuse
`RMSNormFused`; implement the merge+norm explicitly with a plain `RMSNorm` (init-1 weight) and an
fp32 residual accumulator carried through the loop. Plan: implement a small helper
`merge_and_norm(input_norm, res_scale, residual, hidden_states) -> (hidden_states, residual)` and
call it at the top of every `ZayaDecoderLayer.forward`.

### 3.1 `ResidualScaling` (`scale_residual_merge`)
Affine on the fp32 stream (zaya.py:61-98). Per layer it owns `hidden_states_scale[H]`,
`hidden_states_bias[H]`, and (for `layer_n != 0`) `residual_scale[H]`, `residual_bias[H]`:
```
hidden_states = (hidden_states.float() + hs_bias) * hs_scale
if layer_n != 0 and residual is not None:
    residual = (residual.float() + res_bias) * res_scale
```
Layer 0 has only the hidden_states affine (no residual params). The FINAL `res_scale`
(`layer_n = num_hidden_layers = 80`) is a full one (has residual params). Checkpoint keys:
`model.layers.{i}.{self_attn|moe}.res_scale.*` and `model.res_scale.*` — VERIFY exact prefixes (§9).

**v0 simplification allowed?** No. `scale_residual_merge=True` and `residual_in_fp32=True` are both
on in the live config and materially change numerics; port them faithfully from step 1.

---

## 4. State-cache plan (EXACT)

### 4.1 CCA conv/recurrent state — `kvcache/cca_state.py` (new; near-verbatim copy of `gdn_state.py`)

Two fp32 buffers, both indexed `[cca_layer_id, slot]`, slot-0 reserved NULL (free-list starts at 1):
```python
self.conv_states = torch.zeros((num_cca_layers, num_slots, C=1280, TP=2), dtype=float32, device=device)
self.prev_hs     = torch.zeros((num_cca_layers, num_slots, hidden=2048), dtype=float32, device=device)
```
- `C = (num_q_heads+num_k_heads)*head_dim = 10*128 = 1280`; `TP = (cca_time0-1)+(cca_time1-1) = 2`.
- `num_slots = max_running_req + 2` (NULL + dummy), exactly like GDN (`engine.py:137`).
- Per-layer views `conv(cca_layer_id)` → `[num_slots,1280,2]`, `prev(cca_layer_id)` → `[num_slots,2048]`.
- Reuse the GDN free-list / `alloc_many` / `free` / `reset_slots` / snapshot API verbatim
  (`gdn_state.py:18-138`). Spec-decode snapshot/install can be stubbed for v0 (run eager, no spec).

### 4.2 Post-conv paged attention KV cache — reuse `MHAKVCache` unchanged

CCA ends in standard softmax GQA attention, so it uses the normal paged KV pool
(`kvcache/mha_pool.py`), already allocated unconditionally in the engine (`engine.py:110-118`). Each
CCA layer builds an `AttentionLayer(layer_id=cca_layer_id)` → it writes/reads `_kv_buffer[cca_layer_id]`
via `batch.out_loc` + page table automatically. Only the 40 CCA layers do attention.

⚠ **KV pool sizing.** `_determine_num_pages` sizes from `mc.num_layers` (=80) but only 40 layers
attend. Two options: (a) **simplest/safe v0** — leave `num_layers=80`; the KV buffer over-allocates
(40 unused layer slices) but is correct. (b) size the pool by `num_cca_layers`. **Decision: v0 uses
option (a)** (correctness first; the unused slices waste ~half the KV pool but 8B at TP=1 has room).
Optimize to (b) in a follow-up after parity is green. The `layer_id` passed to `AttentionLayer` is the
CCA-attn id (0..39), which is `< num_layers=80`, so it indexes a valid slice either way.

### 4.3 Engine wiring (`engine/engine.py`)
Add an `is_cca_hybrid` branch next to the GDN one (`engine.py:128-152`):
```python
if mc.is_cca_hybrid:
    self.ctx.cca_state = self.cca_state = CCAStateCache(
        num_cca_layers=mc.num_cca_layers,
        num_slots=config.max_running_req + 2,
        conv_dim=1280, conv_kernel=2, hidden_size=mc.hidden_size,
        dtype=torch.float32, device=self.device)
    for cca in self.model.iter_cca_layers():
        cca.warmup_conv(_GDN_WARMUP_TOKENS)   # optional; CCA kernel has no autotune, can no-op
```
Add `cca_state: CCAStateCache | None = None` to `Context` (`core.py`, beside `gdn_state`).

### 4.4 Scheduler wiring (`scheduler/cca_slots.py` + `scheduler.py`)
`CCASlotManager` = verbatim copy of `GDNSlotManager` (`gdn_slots.py`): uid-keyed, slot-0 NULL,
idempotent free, zero-only-fresh. Wire in `scheduler.py.__init__`, `_prepare_batch`
(build `state_indices` + `build_cca_metadata`), and `_free_req_resources` (free on finish). **Force
the naive (non-radix) prefix cache** for `is_cca_hybrid` (`scheduler.py:62-73`) — conv state is not
prefix-cacheable, identical to GDN.

---

## 5. Per-batch metadata — `cca/metadata.py` (new; mirror `gdn/metadata.py`)

`build_cca_metadata(batch, state_indices, device)` → object stashed on `batch.cca_metadata`
(add the field to `Batch`, `core.py:85`). Fields (mirror `GDNMetadata`):
- `is_prefill: bool`, `num_seqs: int`
- `query_start_loc: int32 [num_seqs+1]` — decode: `arange(num_seqs+1)`; prefill: `cumsum([0,extend_len...])`
- `state_indices: int32 [num_seqs]` — conv slot per seq, all ≥1 (from `CCASlotManager.state_indices`)
- `has_initial_state: bool [num_seqs]` — prefill only; `[req.cached_len > 0 for req in reqs]`

minisgl batches are HOMOGENEOUS (all-prefill XOR all-decode, `core.py:73-76`), so — unlike the vLLM
reference which splits decode-first/prefill — **there is no mixed batch and no decode/prefill split
inside the layer**. Dispatch purely on `batch.phase`.

---

## 6. Forward paths — `ZayaCCAAttn.forward`

`forward(self, hidden_states) -> attn_output`. Pulls `ctx = get_global_ctx()`, `state = ctx.cca_state`,
`md = ctx.batch.cca_metadata`, `conv = state.conv(cca_layer_id)`, `prev = state.prev(cca_layer_id)`.

### 6.1 Common pre-conv compute (both phases)
```
hs   = hidden_states                      # [N, 2048]
q    = linear_q(hs)                        # [N, 1024]
k    = linear_k(hs)                        # [N, 256]
qk_new = cat([q, k], -1).float().contiguous()   # [N, 1280] fp32 -> kernel input
```
`val` is computed AFTER the conv kernel (it needs `prev_hs` for `val_proj2`):
```
v1 = val_proj1(hs)                         # [N, 128]
# v2 = val_proj2(hs2) where hs2 = previous-token hidden (per-seq shift; seeded by prev_hs)
```

### 6.2 Decode path (`batch.phase == "decode"`)
```
slot   = md.state_indices.to(int64)        # [N]; all >= 1 (no PAD in homogeneous minisgl decode...
is_pad = (slot == 0)                        # ...but pass is_pad for graph-capture padding rows)
qk_out = torch.ops.zaya_cca.cca_decode_qk(
            qk_new, conv,                    # conv mutated IN PLACE (left-roll, append new at tail)
            slot, is_pad,
            w0, b0, w1, b1,                  # cached fp32 conv weights (CCAConv.post_load)
            temp_eff, num_q=8, gqa=4, latent_q=1024, sqrt_d=sqrt(128))   # [N, 1280] normalized q|k
# previous-hidden for val_proj2: decode = exactly the cached prev_hs of each slot, THEN store current
hs2 = prev[slot]                            # [N, 2048]  (old prev_hs)
prev[slot] = hs.float()                     # store this token's hidden for next step
```

### 6.3 Prefill path (`batch.phase == "prefill"`)
Build the flat per-token metadata the kernel expects (mirror cca.py:528-578; minisgl is pure-prefill
so no decode offset):
```
qsl   = md.query_start_loc                  # [S+1] cu_seqlens of extend_len
req_id= repeat_interleave(arange(num_seqs), seq_lens)            # int32 [num_prefill_tokens]
seg_pos = (arange(num_tokens) - qsl[:-1][req_id]).int()          # position within its sequence
slot_p  = md.state_indices[req_id].to(int64)                     # conv slot per token
is_last = arange(num_tokens) == (qsl[1:]-1)[req_id]              # last token of each seq
# init_states: gather cached conv state, zero where no initial state (continuation vs fresh)
init_states = conv[md.state_indices].float()
init_states = where(md.has_initial_state[:,None,None], init_states, 0).contiguous()
qk_out = torch.ops.zaya_cca.cca_prefill_qk(
            qk_new, conv, init_states, seg_pos, req_id.int(), slot_p, is_last,
            w0, b0, w1, b1, temp_eff, 8, 4, 1024, sqrt(128))    # conv updated only at is_last token
# hs2 = previous-token hidden, per-seq shift; first token seeded by prev[slot] (zero if fresh)
hs2 = empty_like(hs); hs2[1:] = hs[:-1]
init_hs = where(md.has_initial_state[:,None], prev[md.state_indices], 0)
hs2[qsl[:-1]] = init_hs                      # seed each sequence's first token
prev[md.state_indices] = hs[qsl[1:]-1].float()   # store each seq's LAST hidden
```

### 6.4 Values, attention, output (both phases)
```
v2  = val_proj2(hs2)                         # [N, 128]
v   = cat([v1, v2], -1)                      # [N, 256]  (latent_k = 2 heads * 128)
q   = qk_out[:, :1024]                       # normalized, means+RMSnorm+temp baked in
k   = qk_out[:, 1024:1280]
qkv = cat([q.to(model_dtype), k.to(model_dtype), v], -1)   # [N, 1024+256+256]
o   = self.attn.forward(qkv)                 # AttentionLayer: partial RoPE(0.5) + paged GQA attn
return self.o_proj.forward(o)                # [N, 2048]
```
`AttentionLayer` (`layers/attention.py:18`) splits `qkv` as `[qo_dim=1024, kv_dim=256, kv_dim=256]`,
applies partial RoPE over `ctx.batch.positions` (rotary_dim=64 from partial 0.5), stores K/V to the
paged pool, and runs the backend GQA attention — exactly the layout CCA produces. **q_norm/k_norm =
None** (the CCA kernel already RMS-normed q/k). Build it with `nqo=8, nkv=2, head_dim=128`.

> The reference's `ZayaAttention.forward` (zaya.py:203-219) does rotary then `attn(q,k,v)` then
> `o_proj` — identical ordering. minisgl's `AttentionLayer` folds rotary+store+attn into one call.

---

## 7. MoE plan — `ZayaMoEBlock` + `ZayaRouter`

### 7.1 Router (computed IN THE MODEL → precomputed top-k path)
`ZayaRouter.forward(hs, prev_router_hidden_states) -> (route_prob[N,1], expert_idx[N,1], router_hs_next[N,256])`
(mirror zaya.py:384-447):
```
hs256 = down_proj(hs)                                    # [N,256] (+bias)
if use_eda and prev is not None: hs256 = hs256 + prev * router_states_scale
router_hs_next = hs256.clone()                           # PRE-norm (this is what threads forward)
hs_n  = rmsnorm_eda(hs256)
logits= router_mlp(hs_n)                                 # [N,17]  (16 experts + MOD skip)
probs = softmax(logits, dim=-1, dtype=fp32)              # zaya_high_prec -> fp32
biased= probs.detach().float() + balancing_biases        # [N,17]; biases affect CHOICE only
idx   = topk(biased, 1).indices                          # [N,1]  may select skip == 16
route_prob = gather(probs, 1, idx).to(model_dtype)       # [N,1]  the chosen prob
return route_prob, idx, router_hs_next
```
EDA is OFF for the first MoE layer (`layer_number == 1`); plumb `use_eda` per layer in `__init__`.

### 7.2 Expert dispatch + MOD (mirror zaya.py:519-539)
The reference packs `[probs, idx]` into a fake gating tensor and a `custom_routing_function` unpacks
it. **minisgl is cleaner**: pass the precomputed route directly to `MoELayer.forward(hs,
topk_weights=route_prob, topk_ids=clamped_idx)` (the GLM noaux_tc path, `moe.py:61-71, 183-193`).
```
clamped_idx = clamp(idx, 0, 15)                          # skip(16) -> a real expert id for the kernel
experts_out = self.experts.forward(hidden_states=hs, topk_weights=route_prob, topk_ids=clamped_idx)
if use_mod:
    mod_out  = hs * route_prob                            # skip-expert output = scaled residual
    mask     = (idx != 16)                                # [N,1] True where a real expert ran
    out      = mask*experts_out + (~mask)*mod_out
else:
    out      = experts_out
return out, router_hs_next
```
`renormalize=False`, `activation="silu"`. `apply_router_weight_on_input=False` (the route_prob is the
top-1 gate applied to the expert output, standard). The model threads `router_hs_next` to the NEXT MoE
layer via the decoder loop (see §7.4).

### 7.3 fp8 experts — ⚠ NO existing reuse path
`layers/moe.py` has only W4 grouped experts (GPTQ/AWQ/RXF). Zaya's experts are **compressed-tensors
float (fp8) per-channel-weight + token-dynamic-act**. Decision:
- **v0 (correctness):** load experts UNQUANTIZED. Either dequantize the fp8 weights to bf16 at load
  (multiply by per-channel `weight_scale`) into a `quant=None` `MoELayer` (plain stacked experts,
  `moe.py:171-181`, routed via `ctx.moe_backend` or the precomputed kernel path), OR keep them bf16
  if a bf16 checkpoint is available. This proves model math + router + CCA without an fp8 kernel.
- **v1 (perf):** add an fp8 grouped-expert storage class + W8A8-fp8 grouped-MoE kernel as a SEPARATE
  sub-task (out of scope for the 3 steps below; the `w4a8_fp8_wmma` notes in memory are a starting
  point but it is W4 not fp8-W8). Treat as a real port, not reuse.

`intermediate_size` per expert: reference passes `intermediate_size = ffn_hidden_size // 2 = 2048`
to `FusedMoE` (zaya.py:506-507), because `linear_fc1` is the MERGED gate+up (4096) and FusedMoE wants
the per-gate width. So minisgl `MoELayer(intermediate_size=2048)`; the `_GroupedExperts` allocate
`2*intermediate = 4096` for w13. **VERIFY** against checkpoint `linear_fc1` shape (§9).

### 7.4 EDA threading in `ZayaModel.forward`
```
residual = None; prev_router_hs = None
for lid, layer in enumerate(self.layers):
    if even(CCA):  hs, residual = layer.forward(hs, residual)        # no router state touched
    else (MoE):    hs, residual, prev_router_hs = layer.forward(hs, residual, prev_router_hs)
```
`prev_router_hs` is produced ONLY by MoE layers and consumed by the NEXT MoE layer's EDA. CCA layers
pass it through untouched. (Implement `ZayaDecoderLayer.forward` to accept+return `prev_router_hs`;
CCA layers return it unchanged.)

---

## 8. ModelConfig + registry + engine `is_moe` handling

1. **Registry** (`models/register.py`): add `"ZayaForCausalLM": (".zaya", "ZayaForCausalLM")`.
2. **ModelConfig** (`models/config.py`): add fields + properties (only populated in `from_hf` when
   the config carries CCA dims, so dense models untouched — mirror the GDN block at config.py:59-67,
   180-194):
   - fields: `cca_time0`, `cca_time1`, `cca_num_k_heads`, `cca_num_q_heads`, `cca_head_dim`,
     `zaya_mlp_expansion`, `zaya_use_eda`, `zaya_use_mod`, `scale_residual_merge`, `is_cca` flag.
   - properties: `is_cca_hybrid` (`getattr(config,'cca',False) is True` OR `model_type=='zaya'`),
     `cca_layer_ids` (`[i for i in range(num_layers) if i%2==0]`), `num_cca_layers`.
   - `from_hf`: map `num_query_groups → num_kv_heads` (=2), `num_attention_heads → num_qo_heads` (=8),
     `head_dim=128`, `partial_rotary_factor 0.5` is already handled (config.py:160-165),
     `moe_router_topk → num_experts_per_tok` (add the fallback), `ffn_hidden_size → intermediate_size`,
     `norm_epsilon → rms_norm_eps`. Build `quant` from `quantization_config` (compressed-tensors fp8).
3. **`is_moe` is False for Zaya** (`model_type=="zaya"`). Two consequences in `engine.py`:
   - `engine.py:179` (`if is_moe: build moe_backend`) WON'T fire. The precomputed-route `MoELayer`
     path with `quant=None` still uses `ctx.moe_backend` for the fused experts (`moe.py:226-238`).
     **Decision: make `is_moe` return True for Zaya** (extend the property:
     `"moe" in model_type or is_cca_hybrid`) so the moe_backend is built — simplest. Verify this
     doesn't trigger MLA/other moe-gated paths incorrectly (it gates only moe_backend + page_size).
   - `engine.py:413` (`if not is_mla and page_size != 1`) is fine (Zaya is not MLA).
4. **Weight loader** (`models/weight.py`): branch `if config.is_cca_hybrid: yield from
   _load_zaya_weight(...)` BEFORE the dense path (mirror the GDN branch, weight.py:317-319). The
   engine dtype-cast exceptions (`engine.py:251-258`) must skip CCA `temp`, conv weights, and the
   fp32 router pieces — add suffix guards like the GDN `.A_log`/`.dt_bias` rule.

---

## 9. Weight-loading plan — `_load_zaya_weight` (new in `models/weight.py`)

Checkpoint name → minisgl native key (model's `state_dict()` keys are the contract). Streaming
generator yielding `(native_key, tensor)`, TP=1 (no shard for v0). **The checkpoint prefixes below
are VERIFIED from the live safetensors index — they differ from the ZAYA_REFERENCE doc's guesses:
the CCA module is named `qkv` (`self_attn.qkv.*`, NOT `self_attn.cca.*`), the MoE block is
`zaya_block.*` (NOT `moe.*`), and `res_scale` is a LAYER-level module (`layers.{i}.res_scale.*`),
not nested under the mixer.** Schedule confirmed from keys: layers 0,2,4,… = `self_attn` (CCA, 40);
layers 1,3,5,… = `zaya_block` (MoE, 40).

| Checkpoint key | Native module | Notes |
|---|---|---|
| `model.embed_tokens.weight` `[262272,2048]` BF16 | `model.embed_tokens.weight` | tied → lm_head (no separate `lm_head.weight` in ckpt) |
| `model.layers.{i}.self_attn.qkv.linear_q.weight` (i even) | `...self_attn._cca.linear_q.weight` | bf16 |
| `...self_attn.qkv.{linear_k,val_proj1,val_proj2}.weight` | same under `_cca` | bf16 |
| `...self_attn.qkv.conv_qk.0.{weight,bias}` | `_cca.conv_qk.0.*` | weight `[1280,1,2]` → squeeze in post_load |
| `...self_attn.qkv.conv_qk.1.{weight,bias}` | `_cca.conv_qk.1.*` | weight **`[1280,128,2]` BF16 (verified)**; transposed→`[10,128,128,2]` fp32 in post_load |
| `...self_attn.qkv.temp` | `_cca.temp` | **`[2]` BF16 (verified)** → cast/store fp32 (engine cast exception) |
| `model.layers.{i}.self_attn.o_proj.weight` | `...self_attn.o_proj.weight` | 1024→2048 |
| `model.layers.{i}.input_norm.weight` | `...input_norm.weight` | plain RMSNorm; on BOTH CCA & MoE layers |
| `model.layers.{i}.res_scale.{hidden_states_scale,hidden_states_bias}` | `...res_scale.*` | **EVERY layer (verified)**; layer 0 has ONLY the hidden_states pair |
| `model.layers.{i}.res_scale.{residual_scale,residual_bias}` | `...res_scale.*` | present on layers ≥1 only (layer 0 omits — `not_first_layer=False`) |
| `model.layers.{i}.zaya_block.router.down_proj.{weight,bias}` (i odd) | `...moe.router.down_proj.*` | `[256,2048]` BF16, +bias |
| `...zaya_block.router.rmsnorm_eda.weight` | `...router.rmsnorm_eda.weight` | `[256]` |
| `...zaya_block.router.router_states_scale` | `...router.router_states_scale` | EDA layers only — **ABSENT on layer 1 (verified)** |
| `...zaya_block.router.router_mlp.{0,2,4}.{weight,bias}` | `...router.router_mlp.{0,2,4}.*` | idx 1,3 = GELU (no params); **idx 4 `[17,256]` has NO bias (verified)** |
| `...zaya_block.router.balancing_biases` | buffer | **`[17]` BF16 (verified)** → store fp32; [-1]=-1.0 |
| `...zaya_block.experts.local_experts.{e}.linear_fc1.weight` | `...experts` w13 | **`[4096,2048]` F8_E4M3**; SPLIT dim-0 in half → w1 (gate `[:2048]`) / w3 (up `[2048:]`); stack over e (e=0..15) |
| `...zaya_block.experts.local_experts.{e}.linear_fc1.weight_scale` | w13 scale | **`[4096,1]` F32**; splits dim-0 same as weight |
| `...zaya_block.experts.local_experts.{e}.linear_fc2.weight` | `...experts` w2 | **`[2048,2048]` F8_E4M3**; stack over e |
| `...zaya_block.experts.local_experts.{e}.linear_fc2.weight_scale` | w2 scale | **`[2048,1]` F32** |
| `model.final_norm.weight` | `model.final_norm.weight` | plain RMSNorm |
| `model.res_scale.{hidden_states_*,residual_*}` | `model.res_scale_final.*` | **TOP-LEVEL full res_scale (verified)**; final merge (layer_n=80) |
| (no `lm_head.weight` in ckpt) | tied to embed | `tie_word_embeddings=True` |

**The fc1 split** (verified): `linear_fc1.weight` is `[4096, 2048]` fp8 = merged
`[gate(2048) | up(2048)]` along dim 0; `weight_scale` is `[4096,1]` fp32 (per-OUTPUT-channel, dim 0).
Split: `gate = w[:2048]`, `up = w[2048:]`; the scale splits the same way. Dequant for v0 unquantized:
`w_bf16 = (w_fp8.float() * weight_scale).to(bf16)` — scale is along the output dim, so split-then-scale
== scale-then-split. Risk #1 (scale axis) is **RESOLVED** by inspection: dim-0 per-output-channel.

---

## 10. Risk list (where the reference doc / my reading is uncertain — CONFIRM during impl)

1. **fp8 expert dequant correctness** (layout/scale axis RESOLVED by inspection — `[4096,2048]` fp8,
   `[4096,1]` fp32 per-output-channel; dequant `w.float()*scale`). Remaining unknown: whether
   activation also needs scaling for the v1 fp8 kernel (input_activations are token-dynamic per the
   config group). For the v0 bf16-dequant path this is moot. Confirm dequant matches the reference's
   fp8 compute when building the v1 kernel.
2. **`res_scale` semantics** (presence RESOLVED: layer-level on every layer; layer 0 lacks
   `residual_*`; top-level `model.res_scale` is full). Remaining: confirm the fp32 affine ordering in
   §3.1 is bit-faithful (the merge happens BEFORE input_norm, on the fp32 stream).
3. **EDA first-layer index** (RESOLVED: `router_states_scale` is ABSENT on layer 1's checkpoint —
   verified — so EDA is genuinely off there; build `router_states_scale` only on EDA layers). Keep as
   a load-time assertion: every odd layer ≥3 HAS `router_states_scale`, layer 1 does NOT.
4. **Residual/norm exact ordering & fp32 boundaries.** §3 is reconstructed from zaya.py:259-287 /
   542-607 / 701-738. The merge-then-norm-on-residual ordering and the fp32 cast points must be
   bit-faithful; a single misplaced `.float()` shifts logits. Parity-test against the reference
   forward on a fixed prompt.
5. **`is_moe`/moe_backend wiring.** Forcing `is_moe=True` for Zaya (§8.3) to build the moe_backend may
   touch other moe-gated paths (page_size auto, moe_backend="auto" resolution at engine.py:402).
   Confirm no unintended MLA/shared-expert assumptions fire. Alternative: build the moe_backend
   explicitly under `is_cca_hybrid` without flipping `is_moe`.
6. **CCA `qk_out` semantics** (lower risk — kernel is vendored & validated upstream): confirm
   `cca_decode_qk`/`cca_prefill_qk` return the FULL normalized q|k (means+RMSnorm+temp baked) and that
   `v` is model-side (it is — cca.py builds v from val_proj1/2 after the kernel). The kernel does NOT
   touch `prev_hs`; the model manages it (decode: read-then-write; prefill: shift+seed+store-last).
7. **KV pool over-allocation** (§4.2): correctness-safe but wasteful at `num_layers=80`. Confirm 8B
   TP=1 fits before optimizing to `num_cca_layers`.

---

## 11. Implementation checklist — 3 implementer steps

### Step 1 — Config, registry, state cache, scheduler/engine wiring (no model math yet)
- [ ] `models/register.py`: add the `ZayaForCausalLM` line.
- [ ] `models/config.py`: add CCA/MoE/router fields + `is_cca_hybrid`/`cca_layer_ids`/`num_cca_layers`
      properties; populate in `from_hf` from `~/models/ZAYA1-8B-fp8/config.json`; extend `is_moe`
      (or add `is_cca_hybrid` moe_backend branch). Map heads/eps/intermediate.
- [ ] `kvcache/cca_state.py`: `CCAStateCache` (copy `gdn_state.py`; conv_states `[L,slot,1280,2]` fp32,
      prev_hs `[L,slot,2048]` fp32, slot-0 NULL).
- [ ] `core.py`: add `cca_state` to `Context`, `cca_metadata` to `Batch`.
- [ ] `cca/metadata.py`: `build_cca_metadata` (copy `gdn/metadata.py`).
- [ ] `scheduler/cca_slots.py`: `CCASlotManager` (copy `gdn_slots.py`); wire into `scheduler.py`
      (init / `_prepare_batch` / `_free_req_resources`); force naive prefix cache for `is_cca_hybrid`.
- [ ] `engine/engine.py`: `is_cca_hybrid` branch building `CCAStateCache` + `iter_cca_layers()` warm;
      dtype-cast exceptions for `temp`/conv/router fp32 (engine.py:251-258); eager-only (no graph v0).
- [ ] Smoke: model loads on meta, engine constructs both caches, no forward yet.

### Step 2 — Model forward (CCA + attention + residual scheme; experts UNQUANTIZED)
- [ ] `models/zaya.py`: `CCAConv` nn bridge (projections, conv_qk, temp; `post_load` caches
      `w0,b0,w1(transposed),b1,temp_eff`). `ZayaCCAAttn` (decode/prefill dispatch on `batch.phase`,
      the §6 kernel calls, val_proj1/2 + prev_hs management, `AttentionLayer` + `o_proj`).
- [ ] `ResidualScaling` + the §3 merge-and-norm helper; `ZayaDecoderLayer` (CCA branch),
      `ZayaModel` (residual + final merge + final_norm), `ZayaForCausalLM` (tied lm_head, `forward`
      pulling batch from ctx). `iter_cca_layers()`.
- [ ] `models/weight.py`: `_load_zaya_weight` (§9) — UNQUANTIZED experts (dequant fp8→bf16, gate/up
      split, stack over e), tied embed, res_scale, router, conv (fp32-preserving).
- [ ] MoE: `ZayaRouter` (down_proj/EDA/rmsnorm/router_mlp/biases/top-1) + `ZayaMoEBlock` (MOD mask,
      `MoELayer(quant=None)` precomputed-route); EDA threading in `ZayaModel.forward`.
- [ ] **Parity gate:** compare logits on a fixed prompt vs the vLLM reference forward (or vs a
      golden capture); chase the residual/EDA/MOD math (risks 2-4) until coherent.

### Step 3 — fp8 experts + perf (after parity green)
- [ ] fp8 grouped-expert storage class in `layers/moe.py` + W8A8-fp8 grouped-MoE HIP kernel (new
      sub-task; not a reuse of the W4 paths). Wire `quant`=fp8 path in `MoELayer` for Zaya.
- [ ] KV pool sizing by `num_cca_layers` (§4.2 option b).
- [ ] Graph capture (CCA in-place conv update is capturable via static slot buffers, GDN pattern).
- [ ] Serve via `--attn hip --graph N` (production config); measure TPOT; spec-decode optional later.
```
