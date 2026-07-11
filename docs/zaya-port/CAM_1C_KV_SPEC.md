# 1c — KV injection (memory the frozen base *reads*, at the mid-depth full-attention layer)

**Why:** residual-tap injection is capped below RAG and degrades when sustained (single-layer 0.31–0.41,
multi-layer worse). RAG wins because attention *reads* the fact. KV injection gives the frozen base that
same readable signal from **memory** — the attention decides, per-query/per-head, how much to attend to
it (self-dosing, no residual over-steer / degeneration). This is `boltA`'s own "multi-layer KV" escalation.

**Architecture fact (decisive):** Qwen3.5-4B = 32 layers, hybrid: 24 linear-attention (GDN) + **8
full-attention at [3,7,11,15,19,23,27,31]**. KV injection needs *softmax* attention → only the 8 full
layers qualify. Our residual sweet spot was **L12** (mid-depth); the nearest full-attention layer is
**L11 (34% depth)** — same depth. **1c targets L11** (band {7,11,15} available later).

## Mechanism
At full-attention layer L11, the HF attention (`transformers/models/qwen3_5/modeling_qwen3_5.py`,
`Qwen3_5Attention.forward`) computes `key_states`,`value_states` `[B, n_kv_heads, T, head_dim]`, applies
RoPE, then calls `attention_interface(q, k, v, mask, ...)`. Inject **after RoPE, before the attention
call**:
1. From the staged memory bank `[B, K, mem_dim]` compute **K_mem = W_km(bank)**, **V_mem = W_vm(bank)**,
   each reshaped `[B, n_kv_heads, K, head_dim]`. `W_km`,`W_vm`: `mem_dim → n_kv_heads*head_dim` — the
   TRAINED bolt-on params (the 1c analog of the tap's projection).
2. **Position-free memory**: do NOT apply RoPE to K_mem (append after the base's RoPE) — memory tokens
   are timeless (mirrors Titans persistent-memory prepend), not tied to a sequence position.
3. **Append along the key/value sequence axis**: `k = cat([key_states, K_mem], dim=2)`,
   `v = cat([value_states, V_mem], dim=2)` → T_kv grows by K.
4. **Mask**: extend the additive attention mask with K zero columns (all queries may attend to all
   memory tokens — non-causal). If mask is None (pure causal SDPA), build one that's causal over the
   sequence and open over the K memory columns.
5. **Optional conf-gate**: scale the memory columns' contribution by the store's retrieval-strength
   scalar (reuse `_last_conf`), so a weak read stays quiet — the KV analog of the tap's conf-gate.
6. GQA: append at the `n_kv_heads` level (repeat_kv runs inside the attention interface, unchanged).

## Training / eval (reuse the pipeline)
- **Stage 1 (bind):** UNCHANGED — the pk-store binds facts; the bank is read via the episodic
  `memory_bank` path (the one that delivers 0.6–0.8, the same the residual tap used).
- **Stage 2:** train `W_km`,`W_vm` by LM loss through the FROZEN base (base + store frozen), exactly like
  `train_taps` trains the residual tap — only the injection changes (KV-append at L11 vs residual-add at
  L24). New injector class `KVInjector` (analog of `MAGInjector` in `cam/gated_tap.py`): monkey-patches
  L11's `self_attn.forward` with a copy of the original that inserts steps 1–5; holds `W_km`,`W_vm` as
  trainable params; `set_bank()` stages the bank; `detach()` restores the original forward.
- **Eval:** the SAME in-distribution ripple (`eval_indist_ripple`) + single-hop delivery GATE. Report
  head-to-head vs the **L12 residual baseline (~0.37)** and **RAG (~0.5)** on the full 101-set. Keep the
  gate: if single-hop delivery <0.4, withhold.

## Gating / flags
Behind `--inject-mode kv` (default `residual` = unchanged) with `--kv-layer 11`. Everything else
(bind, store, eval, few-shot, matching, confound control) identical to the residual runs so it's a clean
A/B vs L12 residual.

## Success criterion & risks
- **Success:** KV-injection ripple ≥ L12 residual (0.37), ideally **≥ RAG (~0.5)** — that would be the
  first "frozen-base bolt-on matches/beats retrieval on reasoning" result (the Titans end-goal).
- **Risks:** (a) mask construction across SDPA/eager paths (validate memory columns are attendable +
  don't leak causality); (b) the `q_proj` doubled-head_dim/gating in this model — memory K must match
  `key_states` head_dim (it does; k/v are standard head_dim); (c) noisy metric (n≈35–54, free-gen) — the
  A/B vs the SAME-run residual baseline is the clean read; (d) memory tokens with no RoPE vs base keys
  with RoPE — intended (position-free), but verify the attention interface handles the concatenated
  positions (it should; RoPE is already baked into key_states before concat).
- Untouched base guarantee: byte-identical when no bank staged (the patched forward no-ops the append).
