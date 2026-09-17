# ZAYA HF-format loader — mapping OEM (HF) checkpoints onto minisgl's Megatron layout

**Why.** Zyphra now publishes ZAYA in **HF format** (`model_type: "zaya"`, natively in transformers —
no `auto_map`/remote code), incl. the official quants `Zyphra/ZAYA1-8B-{FP8,MXFP4}-Experts`. Our port
(`zaya.py` + our hand-rolled fp8/RXF) is **Megatron-layout**. Loading an OEM checkpoint fails:
`KeyError: 'model.layers.0.input_norm.weight'`. Path chosen: **update the loader to consume HF-format**
(Zyphra's HF releases are the maintained canonical source; our Megatron checkpoints become legacy).

## The core structural difference

| | Megatron (ours / `zaya.py`) | HF (OEM) |
|---|---|---|
| layers | **80, alternating**: even `lid%2==0`=CCA-attn, odd=MoE | **40 fused**: each has `self_attn` + `mlp` |
| ⇒ layer map | — | **HF layer `L` → minisgl `2L` (attn) + `2L+1` (MoE)** |
| experts | per-expert unbundled `experts.local_experts.{e}.linear_fc1/fc2` | **stacked** `mlp.experts.gate_up_proj`/`down_proj` `(E,·,·)` |
| norms | `input_norm` (per minisgl layer) | `input_layernorm` + `post_attention_layernorm` |
| res-scale | per-layer `res_scale` (attn:2 / moe:4 fields) + top `model.res_scale` | `post_attention_residual_scale`(4) + `post_mlp_residual_scale`(4) + top `model.input_hidden_states_*`(2) |
| MoE block | `zaya_block.router` | `mlp.gate` |
| final norm | `model.final_norm` | `model.norm` |

## Key map (shape-verified on FP8-Experts L0)

### Attention: HF `layers.L.*` → minisgl `layers.{2L}.*`
```
input_layernorm.weight                     -> input_norm.weight                  (2048,)
self_attn.o_proj.weight                    -> self_attn.o_proj.weight            (2048,1024)
self_attn.qkv_proj.q_proj.weight           -> self_attn.qkv.linear_q.weight      (1024,2048)
self_attn.qkv_proj.k_proj.weight           -> self_attn.qkv.linear_k.weight      (256,2048)
self_attn.qkv_proj.v_proj_current.weight   -> self_attn.qkv.val_proj1.weight     (128,2048)
self_attn.qkv_proj.v_proj_delayed.weight   -> self_attn.qkv.val_proj2.weight     (128,2048)
self_attn.qkv_proj.conv_qk_depthwise.{w,b} -> self_attn.qkv.conv_qk.0.{w,b}      (1280,1,2)/(1280,)
self_attn.qkv_proj.conv_qk_grouped.{w,b}   -> self_attn.qkv.conv_qk.1.{w,b}      (1280,128,2)/(1280,)
self_attn.qk_norm.temp                     -> self_attn.qkv.temp                 (2,)
post_attention_residual_scale.*            -> layers.{2L}.res_scale.*   (RECONCILE field count: HF 4, attn 2)
```

### MoE: HF `layers.L.*` → minisgl `layers.{2L+1}.*`  (UNSTACK experts)
```
post_attention_layernorm.weight            -> input_norm.weight                  (2048,)
mlp.experts.gate_up_proj[e]                -> experts.local_experts.{e}.linear_fc1.weight       (4096,2048)
mlp.experts.gate_up_proj.weight_scale[e]   -> experts.local_experts.{e}.linear_fc1.weight_scale (4096,1)
mlp.experts.down_proj[e]                   -> experts.local_experts.{e}.linear_fc2.weight        (2048,2048)
mlp.experts.down_proj.weight_scale[e]      -> experts.local_experts.{e}.linear_fc2.weight_scale  (2048,1)
mlp.gate.balancing_biases                  -> zaya_block.router.balancing_biases                 (17,)
mlp.gate.down_proj.{weight,bias}           -> zaya_block.router.down_proj.{weight,bias}          (256,2048)/(256,)
mlp.gate.router_mlp.fc1.{weight,bias}      -> zaya_block.router.router_mlp.0.{weight,bias}       (256,256)/(256,)
mlp.gate.router_mlp.fc2.{weight,bias}      -> zaya_block.router.router_mlp.1.{weight,bias}       (256,256)/(256,)
mlp.gate.router_mlp.norm.weight            -> zaya_block.router.rmsnorm_eda.weight               (256,)
mlp.gate.router_mlp.out_proj.weight        -> zaya_block.router.???   (CONFIRM: (17,256) — where in ZayaRouter)
mlp.gate.router_states_scale               -> zaya_block.router.router_states_scale
post_mlp_residual_scale.*                  -> layers.{2L+1}.res_scale.*   (4 fields — matches moe layer)
```

### Top-level
```
model.embed_tokens.weight        -> model.embed_tokens.weight   (tied lm_head)
model.norm.weight                -> model.final_norm.weight
model.input_hidden_states_{bias,scale} -> model.res_scale.hidden_states_{bias,scale}
  (RECONCILE: top model.res_scale has 4 fields; HF top has only 2 -> residual_{bias,scale} default?)
```

## RESOLVED — residual-scale factoring (was the risky one)
minisgl indexes `res_scale` by the **consuming** layer (applied at TOP of layer N to prior mixer
output; `ResidualScaling`: layer 0 = hidden-only 2 fields, layers≥1 = 4 fields; top-level
`res_scale_final` at layer_n=80). HF indexes by the **producing** sublayer. Verified map:
```
HF model.input_hidden_states_{bias,scale}     -> layers.0.res_scale.hidden_states_{bias,scale}  (2 fields)
HF layers.L.post_attention_residual_scale.*   -> layers.{2L+1}.res_scale.*        (all L, 4 fields)
HF layers.L.post_mlp_residual_scale.*         -> layers.{2L+2}.res_scale.*        (L<39, 4 fields)
                                              -> model.res_scale_final.*          (L==39)
```

## RESOLVED — router (minisgl router_mlp = Linear[0]->GELU->Linear[2]->GELU->Linear[4], + rmsnorm_eda)
```
HF mlp.gate.down_proj.{weight,bias}    -> zaya_block.router.down_proj_{weight,bias}
HF mlp.gate.router_mlp.norm.weight     -> zaya_block.router.rmsnorm_eda_weight
HF mlp.gate.router_mlp.fc1.{w,b}       -> zaya_block.router.router_mlp_0_{weight,bias}
HF mlp.gate.router_mlp.fc2.{w,b}       -> zaya_block.router.router_mlp_2_{weight,bias}
HF mlp.gate.router_mlp.out_proj.weight -> zaya_block.router.router_mlp_4_weight   (17,256; the ne+1 MOD head)
HF mlp.gate.router_states_scale        -> zaya_block.router.router_states_scale   (EDA layers only)
HF mlp.gate.balancing_biases           -> zaya_block.router.balancing_biases
```

## RESOLVED — experts (HF already stacked = closer to internal than our per-expert ckpt)
HF `mlp.experts.{gate_up_proj,down_proj}[.weight_scale]` are `[E,·,·]` stacked -> assign directly to
native `layers.{2L+1}.zaya_block.experts.{gate_up_proj,down_proj}[.weight_scale]`, EP-SLICE
`[ep_offset:ep_offset+ep_local]` (vs our per-expert path which accumulates+stacks).

## RESOLVED — HF detector + config translation (ModelConfig.from_hf)
Detect HF-format ZAYA iff config has **`layer_types`** (HF-only; Megatron config lacks it).
When HF: `num_layers = 2 * num_hidden_layers` (40->80). Translate config fields (Megatron key -> HF key):
`ffn_hidden_size`->`moe_intermediate_size`, `moe_router_topk`->`num_experts_per_tok`,
`zaya_mlp_expansion`->`router_hidden_size`, `norm_epsilon`->`rms_norm_eps`,
`num_query_groups`->`num_key_value_heads`, `rope_theta`->`rope_parameters.rope_theta`. Megatron-only
flags with no HF key -> defaults: `scale_residual_merge=True`, `zaya_use_eda=True`, `zaya_use_mod=True`
(ZAYA1-8B ships these on; confirm against transformers ZayaConfig defaults).

## (historical) Two spots needing module semantics (NOW RESOLVED above)
1. **Residual-scale factoring** — verify against `ResidualScaling.forward` (zaya.py ~L100). HF factors
   the residual affine per-sublayer (`post_attention_*` before MoE-merge on the attn layer,
   `post_mlp_*` before next-attn-merge on the MoE layer) + a top-level `input_hidden_states_*`. Map so
   the applied scale/bias at each merge point is numerically identical.
2. **Router `out_proj` (17,256)** — HF's `router_mlp.out_proj` projects 256→17 (16 experts + bias/aux?).
   Confirm minisgl's `ZayaRouter` equivalent (it may fold this into `down_proj`/scoring). Match it.

## Implementation plan
- Add an HF-format detector in the `zaya.py` weight load path (e.g. any key ends `input_layernorm.weight`
  OR `num_hidden_layers==40` with fused layers) → route through a `_hf_to_minisgl_keys(state_dict)` remap
  that: doubles layers (`L→2L/2L+1`), unstacks experts (`[E,·,·]→E×`), renames per the table, and
  reconciles res-scale/router. Megatron checkpoints keep the current direct path.
- **Validate (closely monitored GPU):** load `Zyphra/ZAYA1-8B-FP8-Experts` DP=2, generate a few tokens,
  confirm coherence; then MXFP4-Experts. THEN the 3-way quant comparison + fidelity vs bf16
  `Zyphra/ZAYA1-8B` (TP=2). Kill the process TREE on teardown ([[gpu-lease-kill-process-tree-flock-fd]]).
- Config note: OEM `num_hidden_layers=40`; ours 80. The loader/config for HF-format must set the
  minisgl internal 80 (2×) and the even/odd split accordingly.
