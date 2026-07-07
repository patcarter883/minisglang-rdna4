# CAM → minisgl serving integration — SHARED CONTRACT (interfaces the 3 workstreams implement to)

Goal: fold the memory-organ CAM serving path (base-uncertainty WRITE GATE + product-key store + trained
tap at layer 24 + per-token gate router) into the minisgl-rdna4 serve engine (Qwen3.5-4B, port 1919).
Read first: `memory-organ .../docs/serving/integration_design.md` and `online_api.md` (deep detail), and the
memory-organ code `cam/gated_tap.py` (GatedMemoryTap), `cam/pk_store_adapter.py` (PKStoreAdapter read),
`cam/gate_router.py` (GateRouter/per-token), `cam/recall_mag.py::eval_serve` (the offline reference loop).

All three workstreams implement to THESE interfaces so the pieces interlock.

## CHECKPOINT (dir) — produced by WS-B (memory-organ), loaded by WS-A (minisgl)
```
meta.json : {base_model, mem_dim, tap_layer(=24), n_banks(=32), n_sub(=32), signal_names[8],
             router_alpha, topk, remember_tau, hidden_size(=2560)}
tap.pt    : GatedMemoryTap.state_dict()  (to_q/to_k/to_v/to_o, gamma, conf-gate params; fp32)
adapter.pt: dict of PKStoreAdapter learned tensors — in_proj, norm, subj_pool_q, store.to_wkey,
            store.to_wval, store.codebook1, store.codebook2, store.read_q[h], store.read_o[h],
            store.read_norm[h], store.read_out_norm, store.head_bias, readout_q, out_proj
router.pt : GateRouter.state_dict()  (n_out=2 per-token: g_top, g_rest)
```

## SERVE CLASS — `python/minisgl/cam/memory.py::CAMMemory` (WS-A owns)
```python
class CAMMemory:
    def __init__(self, checkpoint_dir, base_embed, lm_head_weight): ...   # loads weights, B empty banks
    # WRITE GATE (base-uncertainty): store iff base can't recall the object.
    def remember(self, subject_ids: list[int], prompt_last_logits: Tensor) -> bool:
        # p = softmax(prompt_last_logits)[object first token]; if p < remember_tau: write subject->object; return stored?
        # (object ids passed via set_pending_object or a param — WS-A picks; document it)
    def read(self, subject_ids: list[int]) -> tuple[Tensor, Tensor]:      # (bank [1,K,mem], conf [1]) for the tap
    def apply_tap(self, h: Tensor, bank, conf) -> Tensor:                 # GatedMemoryTap on the residual stream
    def router_delta(self, base_last_logits: Tensor, bank, conf) -> Tensor:  # per-token router-gated logit injection
```
Keep it eager-first; note (do not yet solve) graph-capture: tap must be a byte-exact no-op when no bank is set
(gamma=0 path), and the router_delta is applied at the lm_head (outside the captured decoder graph is fine).

## DATA-PLANE HOOK — `qwen3_5.py::Qwen3_5Model.forward` (WS-A owns, ~5 lines)
After decoder layer `tap_layer` in the layer loop (line ~308), if a CAM bank is staged for this forward:
`h = cam.apply_tap(h, bank, conf)`. Bank/conf are staged per-request BEFORE forward (a thread-local or a field
on the model). No-op (skip) when nothing staged — must not perturb normal serving.

## EDIT-PLANE API — `python/minisgl/server/cam_api.py` (WS-C owns) + ~5-line wire into `api_server.py`
```
POST /cam/remember {subject:str, prompt:str, object:str} -> {stored:bool, base_p:float}
POST /cam/ask      {prompt:str, subject:str}             -> {text:str}   # router-gated seed-once generation
GET  /cam/facts                                          -> [{subject, object}...]
DELETE /cam/facts/{subject}                              -> {deleted:bool}
```
`/cam/ask` = the eval_serve serve_gen loop: read(subject) -> bank; per step base logits + router_delta;
seed-once (stop injecting once the object's first token lands). Reuse CAMMemory.

## Constraints
- Do NOT break the existing `/generate` and `/v1/*` paths — CAM is additive, gated off when no bank staged.
- minisgl conventions: models are BaseOP; the layer loop is the integration point; keep dtype (bf16) + the
  tap params fp32 with the additive cast back (byte-exact no-op at gamma=0), per integration_design.md §0.
- Qwen3.5-4B: hidden_size=2560, num_layers=32, tap_layer=24 valid. Base is dense (no MoE).
- Do NOT commit; create/edit files and report. The coordinator reviews + commits + GPU-validates.
```
