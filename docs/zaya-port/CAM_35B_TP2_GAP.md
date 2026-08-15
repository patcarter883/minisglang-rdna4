# CAM transparent/ambient path on Qwen3.6-35B-A3B (W4A16 MoE, TP=2) — gap analysis

READ-ONLY code-grounded gap list. Nothing was run on a GPU. Goal: what must be true for
`MINISGL_CAM_AUTO` (auto-read) + `MINISGL_CAM_AUTO_WRITE` + #100 pointer delivery + the residual tap
to work on the real production target (Qwen3.6-35B-A3B-AWQ, compressed-tensors W4A16 MoE, served TP=2),
given it is validated ONLY on Qwen3.5-4B single-card.

## What the target actually is (confirmed from code)
- Arch `Qwen3MoeForCausalLM` → `models/qwen3_moe.py::Qwen3Model` (register.py:10). (The memory note pins
  cyankiwi/Qwen3.6-35B-A3B-AWQ = compressed-tensors W4A16; served via the qwen3_moe class, TP=2 MTP.)
- Launch is **SPMD**: `server/launch.py:66` spawns `tp_size` separate scheduler processes
  (`for tp_rank in range(tp_size): mp.Process(... Scheduler ...)`). At TP=2 there are **two** processes,
  each building its OWN `engine.cam` from its OWN sharded weights and each running `_prepare_cam` /
  `_stage_cam` / `_process_last_data` independently. This is central to gaps 2 and 4.
- dp_size=1 (TP=2 is a single DP replica). So the doc's "DP multi-replica pinning" caveat does **not**
  apply — but "one consistent engine.cam" is still false at the *process* level (two stores, one per rank).

---

## Ranked gaps

### GAP 1 — BLOCKER: the MoE model class has no `stage_cam` tap seam → CAM never builds
- **Today:** the engine only builds CAM when the model exposes the seam:
  `engine/engine.py:286-287` gates on `hasattr(inner, "stage_cam")`. The seam
  (`stage_cam` / `clear_cam` / `stage_cam_rows` / `stage_cam_buf` + the in-loop tap fold at `tap_layer`)
  exists **only** in `models/qwen3_5.py` (lines 377-464, the dense 4B). `Qwen3Model` in
  `models/qwen3_moe.py:44-63` has a plain decoder loop and **no** `stage_cam`, `clear_cam`, or tap fold.
  `qwen3_5_moe.py` also lacks it (grep: `stage_cam` matches only qwen3_5.py).
- **Why it breaks 35B:** `getattr(self.model, "model")` → `Qwen3Model`, `hasattr(..., "stage_cam")` is
  False → the whole CAM block is skipped, `self.cam = None`, no `/cam/*` behaviour, no tap, no
  `_prepare_cam` (scheduler.py:994 `if self.engine.cam is not None`). CAM is simply off.
- **file:line:** `engine/engine.py:285-287`; seam absent in `models/qwen3_moe.py:44-63`.
- **Severity: BLOCKER.** Nothing CAM runs until the seam (or a seam-less pointer build path) exists on
  the MoE model. Porting the seam is mechanical (copy the 4 methods + the `lid == tap_layer` residual
  fold into `Qwen3Model.forward`), but see GAP 3: the *tap itself* additionally needs a 35B checkpoint.

### GAP 2 — BLOCKER: vocab-parallel sharded embed + tied lm_head; CAM assumes the FULL table
- **Today:** `VocabParallelEmbedding` shards the vocab across TP ranks —
  `layers/embedding.py:25-29`: `num_embeddings_tp = div_ceil(vocab, tp_size)`, each rank's `.weight` is
  only `[vocab/tp, hidden]` (its slice `vocab_range`). `lm_head` is `ParallelLMHead(tied_embedding=...)`
  (qwen3_moe.py:69-73), so under tying its live weight **is** the sharded embedding weight. The engine
  hands CAM exactly these: `CAMMemory(_cam_ckpt, inner.embed_tokens, _lm_w)` where
  `_lm_w = lmh.tied_embedding.weight` (engine.py:294-296) — i.e. the **half-vocab shard**.
- CAMMemory treats both as full, dense, unsharded tables:
  - `_subj_key` / `_e`: `F.embedding(ids, self._embed_w/self._embed_weight)` (memory.py:246, 604) with
    `ids` in `[0, vocab)` but weight rows only `[0, vocab/2)`. On the shard a rank doesn't own → CUDA
    index-out-of-bounds (device-side assert); on the half it does own the rows are mis-indexed (rank 1's
    row 0 is global row vocab/2). Result on BOTH ranks: crash or silently-wrong subject key.
  - `router_delta` / `store_token_of`: `... @ lm.t()` (memory.py:680, 692) with `lm = [vocab/2, hidden]`
    → `[1, vocab/2]` logits → argmax over half the vocab → wrong token.
  - SPMD makes this worse: each of the 2 processes builds a store over a *different* half → the two
    stores are not even the same object.
- **Why it breaks 35B/TP=2:** CAM needs the FULL embedding to (a) compute the cosine subject key that
  drives pointer delivery/retrieve and (b) compute router/seed logits. Under TP=2 it only ever gets a
  half. This is the #1 core issue the brief predicted — confirmed.
- **file:line:** `layers/embedding.py:25-29`; `engine/engine.py:294-296`; `cam/memory.py:246,604,680,692`.
- **Severity: BLOCKER.** Fix = give CAM the full-vocab embedding (and lm_head): all-gather the shard
  once at build (`vocab/tp × tp` rows reassembled in `vocab_range` order — note `div_ceil` padding on the
  last rank), or reconstruct a full CPU copy for the (tiny, prefill-only) CAM reads. This is single-card
  today only because tp_size=1 leaves the table whole.

### GAP 3 — BLOCKER (for tap/value-store/router): checkpoint is dimensioned to the 4B base
- **Today:** the trained tap/adapter/router are built with shapes read from the **checkpoint** tensors,
  then applied to the *served* model's hidden width:
  - `base_hidden = tap_sd["to_q.weight"].shape[0]`, `mem_dim = tap_sd["to_k.weight"].shape[1]`
    (memory.py:468-469); the tap's `to_q/to_k/to_v/to_o` (memory.py:309-312) are sized to the 4B hidden.
  - `_PKAdapter.in_proj = Linear(base_hidden, mem_dim)` (memory.py:232) — again 4B hidden.
- The 35B residual/embed width differs from Qwen3.5-4B's (4B hidden 2560 vs 35B ~4096 — differ either
  way; the exact numbers don't matter, only that they differ). So:
  - `apply_tap`: `self.to_q(h32)` with `h` = 35B residual `[N, 4096]` × `to_q [2560,2560]` → **matmul
    shape mismatch** (memory.py:339, called from qwen3_5.py:456 fold if the seam were ported).
  - `remember`/`_write`: calls `self.adapter._e(tids)` → `in_proj(F.embedding(...) [.,4096])` where
    `in_proj` expects 2560 → **matmul mismatch** (memory.py:246, 570). And `_write` is invoked directly
    from `_prepare_cam` (scheduler.py:1331) and from `cam.remember` — no try/except → it throws inside
    the scheduler step.
  - `router_delta` `out_proj(bank)` back to `base_hidden=2560` then `@ lm.t()` also mismatches the 35B.
- **Key nuance — the pointer path is the only dimension-independent one.** `deliver_object_ids`
  (memory.py:608-621) and `_subj_key` (memory.py:599-605) use `self._embed_w` (the served model's own
  embed) directly — mean-pool + cosine over `_subj_keys`, never through the adapter — so they self-adapt
  to any hidden width. BUT `_write` (memory.py:567-597) **entangles** the pointer index update with the
  value-store `adapter._e`/`persistent_write` (memory.py:570, 585), so `remember`/auto-write still
  crashes on the hidden mismatch even when you only want pointer delivery.
- **Why it breaks 35B:** the tap, the product-key value store, and the router-gated logit injection all
  require a checkpoint **trained on the 35B base**. No such checkpoint exists (the validated one is 4B).
- **file:line:** `cam/memory.py:232,246,309-312,339,468-469,570,585,680`; `scheduler.py:1331`.
- **Severity: BLOCKER for tap+router+value-store; needs-work to isolate pointer-only.** Two sub-outcomes:
  (a) residual tap / router injection = needs a **35B-trained CAM checkpoint** (a training project, out of
  scope for a delivery proof); (b) pointer delivery + retrieve = achievable **without** the checkpoint's
  tap/adapter dims IF `_write`'s pointer-index branch is decoupled from the adapter value-store call.

### GAP 4 — needs-work: per-rank divergence of the delivery decision / forced token
- **Today:** `_prepare_cam` computes `deliver_object_ids` (scheduler.py:1340-1344) and `_cam_retrieve`
  (1370-) on the local store; `_process_last_data` (738-799, forced-token block ~759-780) overrides the
  sampled token with the forced object token for the first N steps. All of this runs **independently in
  each of the 2 SPMD scheduler processes**.
- **Why it breaks TP=2:** the forced-token *mechanism* is scheduler-level (append_host + emit) and
  TP-agnostic, BUT the *decision* (which object ids, computed from the local, per-rank, currently-broken
  store) is made per-rank. Even once GAP 2 gives each rank a full embed, two ranks must force the SAME
  token stream or the sampler outputs diverge and the verify/collective batch desyncs — exactly the class
  of bug already fixed for structured decoding by broadcasting rank0's token (see memory note
  `structured-spec-tp2-deadlock`: "broadcast rank0's constrained token"). CAM has no such broadcast.
- **file:line:** `scheduler.py:1296-1348` (`_prepare_cam`), `738-799` (`_process_last_data` forced path).
- **Severity: needs-work.** Fix = pin the CAM decision to rank0 and broadcast the forced object ids (or
  make delivery deterministic and identical across ranks once GAP 2 is fixed). Retrieve/auto-read is less
  fragile (it only prepends an in-context system note, no forced token) but still needs both ranks to
  agree on the augmented prompt — that already happens because the prompt is augmented in the FastAPI
  frontend (`api_server._cam_auto_augment`, api_server.py:464/1044/1214), upstream of both ranks.

### GAP 5 — probably-fine: TP=2 is a single DP replica (not the DP-pinning case)
- dp_size=1, so the doc's "CAM state is per-scheduler-replica; multi-replica (DP) needs pinning" and the
  `MINISGL_CAM_DP_RANK` follow-up do **not** apply. The two TP processes both run `_prepare_cam`, so an
  auto-write's `_write` lands in *both* per-rank stores (redundantly consistent), and reads hit the same
  local store — no cross-replica skew. The only consistency risk is the READ/deliver *decision* diverging
  (GAP 4), not store contents. **Severity: probably-fine** (once GAP 2/4 are addressed).

### GAP 6 — probably-fine: quantization does NOT touch CAM's tensors
- `embed_tokens` is `VocabParallelEmbedding` (dense `torch.empty` weight, full precision) and the tied
  `lm_head` is not quantized (qwen3_5_moe.py header: "the … lm_head" are not among the quantized routed
  experts). CAM's `F.embedding` / `@ lm.t()` therefore never hit a 4-bit packed/compressed-tensors weight.
  Q1's worry ("does CAM assume a dense unquantized weight it won't get?") — it assumes dense, and it DOES
  get dense; the W4A16 quant is confined to the routed-expert GEMMs. **The 35B problem is sharding +
  dimensions, not quantization.** **Severity: probably-fine.**

### GAP 7 — needs-work: graph capture (`--attn hip --graph N`, the prod config)
- **Tap under capture:** `cam/graph_capture.py::CAMGraphCapture` threads a static `[max_bs,K,mem]` bank
  buffer so the residual tap runs inside the decode graph. That requires the seam + `stage_cam_buf` on the
  model (GAP 1) AND a working tap (GAP 3). Also its zero-bank-is-no-op invariant assumes bias-free
  to_k/to_v/to_o and warns for `twosided` taps — carries over unchanged to 35B, but moot until GAP 1/3.
- **Pointer forced-token under capture:** `_process_last_data` overrides the *sampled* token AFTER the
  forward — it is a scheduler post-step edit, not part of the captured graph, so pointer delivery is
  graph-capture-safe (the first N forced steps don't depend on graph internals). Auto-read (retrieve) is
  entirely in the frontend + a normal generate, also capture-safe.
- **Severity: needs-work for the tap; probably-fine for pointer/retrieve.** A pointer+retrieve-only proof
  can run under `--graph N`; the residual tap cannot until GAP 1/3 land.

### GAP 8 — probably-fine: the auto read/write generation calls
- `_cam_auto_write` → `FrontendCAMRuntime.extract_facts` (runtime.py:227) is a plain generate — works at
  35B/TP=2 like any other request. `_cam_auto_augment` → backend `mem_op="retrieve"` → `_cam_retrieve` →
  `deliver_object_ids` is blocked only by GAP 2 (full-embed cosine), not by anything TP/quant-specific.
  The subsequent `remember` write is blocked by GAP 3's `_write` entanglement. **Severity: probably-fine
  once GAP 2/3 are addressed** (no CAM-specific generate assumption breaks at 35B/TP=2).

---

## Minimal validation plan (smallest proof on the local box, one lease)

**Must-fix before any run is even possible:**
1. **GAP 2 — full embedding for CAM.** All-gather the vocab shard (and the tied lm_head) once at
   `CAMMemory` build so `_subj_key`/`_e`/router index the full vocab. Without this every CAM op crashes
   or is wrong under TP=2. (Highest priority; unblocks retrieve + pointer.)
2. **GAP 1 — let CAM build on the MoE model.** For a **pointer/retrieve-only** proof (no residual tap) you
   do NOT need the tap fold — relax the `hasattr(inner,"stage_cam")` gate (engine.py:286) so CAM builds
   for pointer/retrieve, OR port the full seam if you also want the tap. Simplest first proof: pointer +
   retrieve, no tap → no seam port, no 35B checkpoint needed for the *tap*.
3. **GAP 3 — decouple the pointer index from the adapter value-store in `_write`** (memory.py:567-597) so
   `remember`/auto-write populates only the cosine subject index (`_subj_keys`/`_subj_objs`) and does not
   call `adapter._e`/`persistent_write` (which crash on the 4B-vs-35B hidden mismatch). This gives a
   working pointer store on 35B with the existing 4B checkpoint's *addressing-independent* path.
4. **GAP 4 — pin the CAM decision to rank0 + broadcast forced object ids** (mirror the existing structured
   -decode rank0 broadcast) so the two SPMD processes force identical tokens. For a retrieve-only
   (in-context, no forced token) first proof this is not strictly required (augmentation is frontend-side),
   so it can be deferred to the pointer-delivery step.

**Smallest run (one lease, `gpu-lease -n 2`):**
- Serve: `MINISGL_CAM=1 MINISGL_CAM_CHECKPOINT=<4B-ckpt, pointer-only> MINISGL_CAM_AUTO=1
  MINISGL_CAM_AUTO_WRITE=1` + `--tensor-parallel-size 2 --attn hip --graph 0` on the 35B-AWQ, following
  this repo's container recipe (mount source, forward the lease's HIP/ROCR pair). Start eager (`--graph 0`)
  to isolate CAM logic from capture.
- Prove **auto-read + auto-write (retrieve, in-context)**: plain `/v1/chat` that STATES a novel fact
  ("the mother tongue of <NovelName> is Dothraki"), then a later plain chat "what language does he speak?"
  → expect **Dothraki**; `/cam/facts` shows the auto-learned fact. (No `/cam/*` params on either turn.)
- Prove **pointer delivery** (GAP 4 required here): `/cam/ask` with `mem_subject` → forced exact object
  tokens + coherent base continuation. Then re-run the whole thing under `--graph N` to confirm the
  pointer/retrieve path survives capture (GAP 7).

**Explicitly out of scope for a delivery proof (separate projects):**
- The **residual tap** and **router-gated injection** need a CAM checkpoint **trained on the 35B base**
  (GAP 3a) plus the ported tap seam (GAP 1) plus the graph-capture buffer (GAP 7). That is a training +
  integration effort, not a delivery-only validation.

## Confirmed vs unknown
- **Confirmed from code:** SPMD 2-process launch; vocab-parallel sharding of embed+tied lm_head; the
  `hasattr(stage_cam)` build gate and the seam living only in qwen3_5.py; CAM's full-table assumption in
  `_e`/`_subj_key`/`router_delta`; the checkpoint-dimensioned tap/adapter; `_write` entangling pointer
  index with the value store; embed/lm_head are NOT quantized; pointer/retrieve is post-sample /
  frontend-side (capture-safe).
- **Unknown / not verified (no GPU, no model files read):** the exact 35B `hidden_size`/`num_layers` and
  vocab (only that hidden differs from 4B, which is all the argument needs); whether any 35B-trained CAM
  checkpoint exists (assumed not); the runtime behavior of an all-gathered embed under the prod image
  (would need the live serve). These are the items a first eager run would settle.
