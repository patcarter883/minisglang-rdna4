# `minisgl.cam` — CAM editable-memory serving path (WS-A)

Folds the memory-organ **CAM** editing mechanism (base-uncertainty **write gate** + product-key
**store** + trained **tap at layer 24** + per-token **gate router**) into the minisgl serve engine for
Qwen3.5-4B. Implements the shared contract in `docs/zaya-port/CAM_SERVE_CONTRACT.md`.

Ownership (WS-A): this package (`memory.py`, `__init__.py`, this README) **and** the ~5-line data-plane
hook in `python/minisgl/models/qwen3_5.py`. It does **not** touch `api_server.py` (WS-C) or the
memory-organ export (WS-B).

The serving pieces are ported self-contained from memory-organ so the module is **CPU-importable** with
no memory-organ dependency: `_ProductKeyStore` (← `cam/pk_store.py`), `_PKAdapter` (← `cam/pk_store_adapter.py`,
persistent path only), `_GatedMemoryTap` (← `cam/gated_tap.py`), `_GateRouter` + `signal_features` +
`_inj_pertoken` (← `cam/gate_router.py`).

---

## Public surface — `CAMMemory`

```python
CAMMemory(checkpoint_dir, base_embed, lm_head_weight)
  # base_embed: the serve model's embedding module (uses .weight) or the weight tensor directly.
  # lm_head_weight: [vocab, hidden] tensor for the router's store-push logits.
  # checkpoint_dir=None or missing  ->  constructs DISABLED (enabled=False): all methods no-op,
  #                                     read() returns (None, None). Lets a server boot without a ckpt.

remember(subject_ids, prompt_last_logits[, object_ids]) -> bool   # WRITE GATE
read(subject_ids) -> (bank [1,K,mem], conf [1])                   # once per request, at prefill
apply_tap(h, bank, conf) -> h'                                    # residual-stream tap at tap_layer
router_delta(base_last_logits, bank, conf) -> logit delta        # per-token, at the lm_head

# control-plane conveniences (WS-C's /cam API builds on these):
set_pending_object(object_ids); store_token_of(bank); facts(); reset(); forget(subject_ids)
```

### Write gate — how the object ids are supplied (contract resolution)
`remember` implements the base-uncertainty gate: `p = softmax(prompt_last_logits)[object_first_token]`;
if `p < remember_tau` it writes `subject -> object` and returns `True`, else it skips (the base already
recalls it) and returns `False`. `prompt_last_logits` is the base's **memory-OFF** last-position logits
for the fact probe (e.g. "The capital of France is"), computed by the caller.

The **object token-ids** are supplied **either** as the explicit third arg `object_ids`, **or** (to keep
the contract's 2-arg `remember(subject_ids, prompt_last_logits)` callable) via a prior
`set_pending_object(object_ids)`. WS-C's `/cam/remember` tokenizes the object string (same tokenizer,
`add_special_tokens=False`) and passes the ids.

The stored **value** is the object's **first token** embedding by default (byte-identical to the
memory-organ reference `_persistent_write_val`); set `obj_latent: true` in `meta.json` to instead store
the pooled mean of the whole object phrase (the `CAM_OBJ_LATENT` path).

---

## Load / stage / forward flow

1. **Load (once, at engine start).** Construct `CAMMemory(ckpt_dir, model.model.embed_tokens,
   model.lm_head.weight)`. It reads `meta.json`, `tap.pt`, `adapter.pt`, `router.pt`; rebuilds the tap,
   the PK adapter (its frozen embed table is a **reference** to the live serve embedding — no 3 GB
   reload), and the router; freezes them; and allocates `n_banks` empty product-key value banks. TP=1
   for the MVP (the embed must be the full, un-sharded table).

2. **Write (edit).** `remember(subject_ids, prompt_last_logits, object_ids)` routes the subject to its
   disjoint bank by a stable md5 hash of the token-ids and does the error-correcting delta write. Edits
   persist across requests until `reset()`/`forget()`.

3. **Stage (per request, before `model.forward`).** Compute `bank, conf = cam.read(subject_ids)` at
   prefill, then stage it on the model:
   ```python
   model.model.stage_cam(cam, bank, conf)   # bank=None -> tap skipped (normal serving unperturbed)
   ...forward...
   model.model.clear_cam()
   ```
   `stage_cam` sets `_cam / _cam_bank / _cam_conf / _cam_tap_layer` (leading-`_` fields, hidden from the
   BaseOP state-dict walk, like `_capture_layer_ids`). Reuse the same `(bank, conf)` for every decode
   step of that request — the store read is **never** on the decode hot path.

4. **Forward hook (data plane).** In `Qwen3_5Model.forward`, after decoder layer `tap_layer`:
   ```python
   if cam_bank is not None and lid == self._cam_tap_layer:
       h = x + residual
       residual = residual + (self._cam.apply_tap(h, cam_bank, self._cam_conf) - h)
   ```
   The tap sees the full post-layer hidden `h = x + residual` and returns `h + upd`; folding the additive
   `upd = apply_tap(h) - h` into `residual` mirrors the HF `output[0]` hook so the next layer's
   `input_layernorm(x, residual)` sums the injected hidden. **Byte-exact no-op when unstaged** (branch
   skipped) and at `gamma=0`, where `apply_tap(h) == h` so `upd == 0` exactly — validated on CPU.

5. **Router (at the lm_head).** For router-gated generation, add `cam.router_delta(base_last_logits,
   bank, conf)` to the memory-OFF last-token logits. The caller owns **seed-once** (stop adding the delta
   once the store's object token — `cam.store_token_of(bank)` — has been emitted). This is applied
   outside any captured decoder graph, so it is capture-safe.

Eager-first: run `--graph 0` for the MVP. Decode-tap graph capture (a `CAMGraphCapture` static bank
buffer, GDN pattern) is a later phase and out of WS-A's scope.

---

## Checkpoint format WS-B must produce

A directory with four files:

### `meta.json` (scalars/knobs — bake the training env-knobs here, not `os.environ`)
Required: `tap_layer`, `n_banks`, `remember_tau`, `router_alpha`, `topk`, `mem_dim`, `hidden_size`,
`signal_names` (8), `base_model`.

Also read (defaults in parentheses — most shapes are otherwise **inferred from the tensors**, so these
matter mainly when a tensor is absent or a knob changes addressing/injection math):

| key | meaning | default |
|---|---|---|
| `topk` | **router** multigate top-k for `_inj_pertoken` (NOT the store top-k) | 16 |
| `store_topk` | product-key global top-k per read | 8 |
| `store_sub_topk` | product-key candidates per half (`sub_topk`) | 4 |
| `n_sub` | codebook size per half (N = n_sub²) | inferred from `store.codebook1` |
| `k` | readout K slots (bank slot count) | inferred from `readout_q` |
| `read_heads` | store multi-head read count | inferred from `store.read_q.*` |
| `tap_heads` | tap cross-attention heads | inferred from `null_key` |
| `n_key_heads` | learned key-pool heads (multi-vector keys) | inferred from `subj_pool_q` |
| `n_rel` | conf-gate per-relation EMA count | inferred from `conf_ema` |
| `conf_gate` | tap store-confidence gate on | false |
| `learned_key_pool` | attention key pool (`CAM_LEARNED_KEY_POOL`) | false |
| `pooled_subj_key` | pool subject span for the write key (`CAM_POOLED_SUBJ_KEY`) | true |
| `key_maxsim` / `key_maxsim_temp` | multi-vector MaxSim reduce (`CAM_KEY_MAXSIM`) | false / 0.1 |
| `write_at_read` | write at the read slot (`CAM_WRITE_AT_READ`, K1) | true |
| `norm_gate` | norm-relative injection (`CAM_NORM_GATE`) | false |
| `twosided` | two-sided suppression (`CAM_TWOSIDED`) | false |
| `obj_latent` | store object-phrase latent vs first token (`CAM_OBJ_LATENT`) | false |

> **Correctness note:** the write and read paths use these knobs **symmetrically**. If the tap/store was
> trained with e.g. `CAM_LEARNED_KEY_POOL=1 CAM_POOLED_SUBJ_KEY=1 CAM_WRITE_AT_READ=1 conf_gate`, those
> must be recorded in `meta.json` — a mismatch silently mis-addresses. Not-yet-supported training knobs:
> `CAM_QUERY_BATCHNORM`, `CAM_GTE_KEYS`, `perpos`/`disjoint` value modes, `decoder` readout (persistent
> single-token path only).

### `tap.pt` — `GatedMemoryTap.state_dict()` (the layer-24 tap, fp32)
Keys: `to_q.weight`, `to_k.weight`, `to_v.weight`, `to_o.weight`, `gamma`, `gate_alpha`, `supp`,
`null_key`, `conf_scale`, `conf_bias`, `conf_ema`. Save the **single tap's** state_dict — if you save the
`MAGInjector.taps` ModuleDict instead (keys like `"24.to_q.weight"`), the loader strips the leading
numeric component automatically.

### `adapter.pt` — the `PKStoreAdapter` learned tensors, **embed/unembed dropped**
Keys (as in `PKStoreAdapter.state_dict()` minus `embed.*` and `unembed`):
`in_proj.weight`, `norm.weight`, `norm.bias`, `subj_pool_q`, `readout_q`, `out_proj.weight`,
`store.codebook1`, `store.codebook2`, `store.to_wkey.weight`, `store.to_wval.weight`,
`store.read_q.{h}.weight`, `store.read_o.{h}.weight`, `store.read_norm.{h}.weight`,
`store.read_out_norm.weight`, `store.head_bias`. Extra training-only tensors (e.g. `pos_tag`, `pos_proj`,
`dec_*`) are ignored. **Do not** include `embed.*`/`unembed` — they are rebuilt from the live serve
embedding (the `embed_shape` donor-match check from `save_ckpt` is enforced upstream by WS-B/coordinator).

### `router.pt` — `GateRouter.state_dict()` (n_out=2 per-token: g_top, g_rest)
Keys: `net.0.weight/bias`, `net.2.weight/bias`, `net.4.weight/bias`. `hidden` and `n_out` are inferred
from the shapes.

---

## Validation done (CPU, in `vllm22-w4a8:combined`)
- `import minisgl.cam.memory` succeeds; disabled construction (no ckpt) no-ops correctly.
- Full synthetic round-trip: load → write gate (skips base-known p≥τ, remembers unknowable p<τ) →
  `read` returns `(bank [1,K,mem], conf [1])` → `apply_tap` is **byte-exact no-op at gamma=0** and injects
  when the gate opens → `router_delta` (1-D and batched) → `store_token_of` / `forget` / `reset`.
- Model hook AST-valid; no-op path is the untouched `residual` (branch skipped when no bank staged).

**Not run:** GPU serving / graph capture (out of WS-A scope; needs WS-B's real checkpoint + WS-C's API).
