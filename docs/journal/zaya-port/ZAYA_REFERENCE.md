# ZAYA1-8B reference architecture map (for the minisglang port)

> AI-synthesized from the vLLM-gfx1201 reference implementation. **Treat every claim
> as a hypothesis to verify against the actual source before relying on it.** Source of
> truth files:
> - Model: `/home/pat/code/vllm-gfx1201/zaya/overlay/vllm/model_executor/models/zaya.py`
> - Config: `/home/pat/code/vllm-gfx1201/zaya/overlay/vllm/transformers_utils/configs/zaya.py`
> - CCA layer math: `/home/pat/code/vllm-gfx1201/zaya/overlay/vllm/model_executor/layers/mamba/cca.py`
> - Kernel: `/home/pat/code/vllm-gfx1201/zaya/cca_hip/{cca_kernel.hip,cca_op.py,bindings.cpp}`
> - Kernel tests (concrete dims): `/home/pat/code/vllm-gfx1201/zaya/cca_hip/test_cca_*.py`
> - Live config: `/home/pat/models/ZAYA1-8B-fp8/config.json`
> - Vendored kernel in THIS repo: `/home/pat/code/minisgl-rdna4/cca_hip/` (registers
>   `torch.ops.zaya_cca.{conv_state_decode,cca_decode_qk,cca_prefill_qk}`)

## 1. Overall structure (config.json)
- vocab_size 262,272 · hidden_size 2,048 · ffn_hidden_size 4,096 · num_hidden_layers 80
- num_attention_heads 8 · num_key_value_heads 2 · head_dim 128 · kv_channels 128
- num_experts 16 · moe_router_topk 1 · zaya_mlp_expansion 256
- max_position_embeddings 131,072 · rope_theta 5,000,000 · partial_rotary_factor 0.5
- norm RMSNorm eps 1e-5 · gated_linear_unit (SwiGLU) · tie_word_embeddings true
- residual_in_fp32 true · scale_residual_merge true · zaya_use_mod true · zaya_use_eda true
- mamba_cache_dtype float32 · attention_bias/lm_head_bias/add_bias_linear false
- **Layer schedule:** alternating — even layer index = CCA, odd = MoE (VERIFY against source).

## 2. CCA (Compressed Convolutional Attention) layer
Head geometry: num_q_heads 8, num_kv_heads 2 (gqa=4), head_dim 128.
latent_q = 8*128 = 1024, latent_k = 2*128 = 256, C = latent_q+latent_k = 1280.

Projections (ReplicatedLinear, no bias):
- linear_q: 2048 -> 1024
- linear_k: 2048 -> 256
- val_proj1: 2048 -> 128 (latent_k/2)
- val_proj2: 2048 -> 128
- o_proj: 1024 -> 2048

Two-stage conv (conv_qk), kernel sizes K0=cca_time0=2, K1=cca_time1=2, TP=(K0-1)+(K1-1)=2:
- conv_qk.0: depthwise Conv1d(1280,1280,k=2,groups=1280,bias=True)
- conv_qk.1: grouped  Conv1d(1280,1280,k=2,groups=10,bias=True)
Per-k-head learnable `temp` shape [2]; applied to normed keys (exp(clamp(temp,1e-7,2.0)) unless clamp_temp).
Per-head RMS-norm over the 128 channels of each head: x/sqrt(mean_sq+eps)*sqrt(128), eps 1e-12.
Grouped-mean injection: query += 0.5*query_pre + 0.5*key_base; key += 0.5*mean(query_pre)+0.5*key_base.

After CCA produces normalized q[.,1024], k[.,256], v[.,256]: apply partial RoPE (50%) then
**standard softmax attention** (this DOES use a paged KV cache for q·k·v) then o_proj 1024->2048.
So a CCA layer needs BOTH a conv/temporal state cache AND a normal attention KV cache. VERIFY this —
it's the single most important architectural question for the minisgl cache wiring.

State tensors per CCA layer (mamba_cache_dtype=float32):
- conv_states: [num_blocks, C=1280, TP=2] — last TP cols of depthwise conv input per seq
- prev_hs:     [num_blocks, hidden_size=2048] — previous-token hidden for the hs2/val_proj2 path

## 3. CCA kernel call signatures (verbatim intent — confirm against cca_op.py)
```
zaya_cca.conv_state_decode(qk_new, conv_states, slot, is_pad, w0, b0, w1, b1) -> qk_out
zaya_cca.cca_decode_qk(qk_new, conv_states, slot, is_pad, w0, b0, w1, b1,
                       temp_eff, num_q, gqa, latent_q, sqrt_d) -> qk_out   # rolls conv_states in place
zaya_cca.cca_prefill_qk(qk_new, conv_states, init_states, seg_pos, req_id, slot, is_last,
                        w0, b0, w1, b1, temp_eff, num_q, gqa, latent_q, sqrt_d) -> qk_out
```
- qk_new/qk_out: [N, C=1280]; conv_states: [NB, 1280, 2]; w0:[C,K0] b0:[C]; w1:[H,d,d,K1] (transposed) b1:[C]
- decode: per seq, builds window=[cached TP | new token], depthwise then grouped conv, grouped-mean,
  per-head RMS-norm, temp on k; then LEFT-rolls conv_states (drop col0, append new at tail).
- prefill: per-token seg_pos/req_id/slot/is_last; causal conv over [init_state | tokens]; writes new
  conv_state only at is_last token. init_states pre-gathered for has_initial_state requests.
- is_pad masks padded decode slots (no state update, zeroed output).
- The vendored `cca_op.py` register_fake shows the EXACT current arg order — trust that file.

## 4. MoE layer
num_experts 16, top_k 1, intermediate (ffn) 4096, SwiGLU. Experts FP8 (compressed-tensors
float-quantized, per-channel weight scales, token-wise dynamic act quant). lm_head/embed/router/
self_attn NOT quantized.
Router (ZayaRouter): down_proj 2048->256, RMSNorm(256), router_mlp 256->256(GELU)->256(GELU)->16,
learnable router_states_scale[256] and balancing_biases[num_experts(+1 for MOD skip)].
- EDA (zaya_use_eda): hs += prev_layer_router_hidden * router_states_scale (not layer 0); stash hidden for next.
- MOD (zaya_use_mod): extra "skip" expert; if skip chosen output = input * route_prob (scaled residual).
Top-1 choice on (softmax(logits)+balancing_biases).

## 5. Weight names -> modules
- embed_tokens.weight (tied to lm_head)
- layers.{i}.self_attn.cca.{linear_q,linear_k,val_proj1,val_proj2,o_proj}.weight
- layers.{i}.self_attn.cca.conv_qk.0.{weight[C,1,K0],bias[C]}, conv_qk.1.{weight[C,1,K1],bias[C]}
- layers.{i}.self_attn.cca.temp[num_k]
- layers.{i}.self_attn.attn.* (standard attention o_proj etc. — verify exact names)
- layers.{i}.moe.input_norm.weight
- layers.{i}.moe.router.{down_proj,rmsnorm_eda,router_states_scale,balancing_biases}.* and router_mlp.{0,2,4}.*
- layers.{i}.moe.experts.local_experts.{e}.linear_fc1.{weight,weight_scale} (split half->gate w1/up w3)
- layers.{i}.moe.experts.local_experts.{e}.linear_fc2.{weight,weight_scale} (-> w2)
- final norm + lm_head (tied).

## 6. Key invariants
- CCA conv-state roll is LEFT shift (append new token at tail col TP-1).
- Grouped-mean injection is post-conv (not part of conv output).
- Per-head RMS-norm is over the 128 channels within a head.
- w1 stored transposed [H,d_in,d_out,K1] for coalesced access.
- CCA has NO TP>1 (per-head norm + grouped-mean state). Multi-card = DP + EP only.
- Residual in fp32; optional scale_residual_merge affine on residual/output before add.
