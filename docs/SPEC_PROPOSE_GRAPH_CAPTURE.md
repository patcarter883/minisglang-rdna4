# Spec-decode PROPOSE-path graph capture — shipped design + per-proposer status

**Rule:** eager is unfinished work; a spec path only counts as complete when it runs under CUDA graph
(`graph-capture-required-for-complete`). VERIFY has been captured for a long time. PROPOSE was the
remaining eager hole. It is now closed by **one shared mechanism**, `python/minisgl/spec/capture.py`.

> **This document previously described a plan, and two of its load-bearing claims were already false
> when it was read.** It said "PROPOSE is EAGER for **all four** model-based proposers. No
> `capture_propose`/`replay_propose` exists" — but MTP propose had in fact been captured (privately,
> inside `MTPProposer._run_chain`) and default-ON, and TiDAR's propose is the K+1 verify graph, which
> was also already captured. It also said the new path would ship behind `MINISGL_SPEC_PROPOSE_GRAPH`
> "default OFF", which contradicts the repo's no-env-gating-on-merge rule and was not what shipped.
> The corrected state is below. Planning from the old text cost a scouting pass; that is why this
> file now records what EXISTS rather than what is intended.

---

## 1. The mechanism — one path, four hooks

`spec/capture.py` defines `CapturableProposer(Proposer)`. A proposer supplies only what is genuinely
drafter-specific:

| hook | runs | responsibility |
|---|---|---|
| `init_propose_capture(engine)` | once, from `__init__` | allocate EVERY persistent tensor: draft-KV pool, per-slot cursors, static propose I/O |
| `stage_propose(reqs, K, ctx)` | host, per step, outside the graph | pick the rows that draft, reset reused slots, refresh the static inputs in place; return `StagedPropose` or `None` |
| `propose_body(bs)` | **the captured region** | static buffers in, static buffers out. No sync, no dynamic shape, no host branch |
| `read_drafts(reqs, staged)` | host, after replay | the ONE device→host sync of the step |

`propose()` is final on the base: `stage → replay-or-eager → read`. The body is **the same callable**
in both arms, which is what makes "replay == eager" a testable claim rather than an assertion — and
is exactly what `tools/propose_capture_ab.sh` tests.

Everything else exists once, on the base: bucket selection, NULL-slot padding, side-stream warmup,
shared graph pool, the DP+EP gate, the replay/eager counters, and teardown.

### What the shared path fixed that MTP's private capture had wrong
* **Lazy, mid-serve, per-EXACT-batch-size capture** → boot-time capture over the engine's existing
  bucket grid (`[1,2,4,8,12,16,24,32]…`, capped at `max_running_req`). The old form could capture up
  to `max_running_req` distinct graphs at arbitrary moments during serving. Padding is cheap because
  the low buckets are fine-grained: bs=1 pads to 1.
* **Invisible to teardown.** `GraphRunner.destroy_cuda_graphs` knew nothing about `MTPProposer._graphs`
  — a live graph past NCCL teardown hangs shutdown. `Engine.shutdown` now calls
  `proposer.destroy_propose_graphs()` first.
* **Two live implementations of propose** selected by env (`MINISGL_SPEC_PROPOSE_GRAPH`). The eager
  Python-list path was dead for both shipped MTP heads. Deleted, along with the flag.
* **No engagement evidence.** The `[spec-timing]` line now carries
  `propose-graph replay=N eager=M buckets=[...]`, because a captured path that silently degrades to
  eager is the failure mode, and a speedup number cannot distinguish the two.

### The rules the body obeys (all enforced by construction, all documented in `capture.py`)
no host sync · no data-dependent shape · no per-step host control flow · persistent index tensors
refreshed in place · small in-body transients (the graph pool keeps them forever) · `inference_mode`
around warmup AND capture · side-stream warmup ×2 then shared-pool capture · collectives in lockstep
· padded rows land on a reserved NULL slot.

---

## 2. Per-proposer status

| proposer | propose | status |
|---|---|---|
| **MTP** | K-step chain over the target's MTP head | **CAPTURED**, default ON. Moved off its private mechanism onto the shared one. |
| **DFlash (Laguna)** | one batched block-diffusion denoise + head | **CAPTURED**, default ON. Required a new fixed-capacity prefix ring and a batched trunk. |
| **DFlash (z-lab / CCA)** | same entry point, different drafter | **EAGER by construction** — see §4. |
| **EAGLE3** | K-step chain over the draft layer | **CAPTURED**, default ON. Required new `step_masked`/`seed_buffered`/`draft_buffer_dims` on the draft model. |
| **TiDAR** | `_tidar_block_predict` = the K+1 verify graph | Already captured before this work. Unchanged. |
| **DDTree** | has no `propose` — it is a scheduler MODE | See §5. Its three forwards are all captured; its tree ASSEMBLY cannot be. |

### 2.1 MTP
Slot = `req.table_idx`; `_cur[slot]` = committed draft-KV length; the KV lives in one global
`[max_slots, max_ctx, nkv, hd]` buffer with an additive `-inf` mask beyond the cursor. `on_accept` is
a cursor advance (capped at K — `d_{K-1}`'s KV is never written), `free` releases the slot. The
masked attention primitive (`forward_draft_masked`) was already present on both shipped heads and is
byte-exact vs the sliced stack (`tools/mtp_forward_draft_parity.py`).

### 2.2 DFlash
The hard one, and the only proposer whose propose is a single block forward rather than a K-step
chain. Three things had to change:

* **Prefix K/V: per-uid compacting buffers → one MODULO RING pool** `[slots, C, Hkv, hd]` per layer,
  `C = window + slack`. Compaction is a conditional memmove, i.e. host control flow. Going modulo
  costs the concatenation-index mask, so `_ppos[slot, col]` now carries each column's **absolute
  position** and the mask is `pos <= qpos and (qpos - pos) < window`. That is strictly more general —
  and it incidentally removes the latent bug where a nonzero `MINISGL_DFLASH_POS_OFF` made RoPE
  positions non-contiguous while `_block_mask` still assumed they were.
* **The variable-length prefix update became fixed-length.** How many positions were newly committed
  varies per request per step (1..block). The body instead re-projects a FIXED block-sized tail every
  step and routes the overhang to the NULL slot. This is sound because it is **idempotent**: the
  scheduler only ever APPENDS accepted positions to the aux buffer, so re-projecting an
  already-projected position reproduces the identical K/V bit for bit. It is also nearly free — at
  these shapes the projection is bandwidth-bound on the weights, not on the 4-vs-16 rows.
  A cold start (or a gap larger than one block) still needs a variable-length rebuild; that runs
  **eagerly, once per request**, and the count is reported rather than hidden.
* **Batched + grouped-query trunk.** The per-request Python loop is host control flow; and
  `repeat_interleave(group)` inside a graph would materialise an 8×-larger K/V into the graph's
  permanent private pool. `attend_block_batched` carries the group axis in the einsum instead.

The eager per-request `attend_block` deliberately kept `repeat_interleave` for bit-identity at a 3.8%
cost (measured, `tools/dflash_gqa_formulation_probe.py`). That trade does not survive multiplication
by the batch and the graph pool, so the batched body makes the opposite call — and says so in place.

### 2.3 EAGLE3
Mechanically MTP's twin (one draft layer, K-step chain, per-uid growing list → global buffer +
cursor), so it rides the identical hooks. Two EAGLE3-specific facts that matter for how a result is
read:

* **EAGLE3 is NOT a sliding-window drafter.** `step` attends every cached key with no mask. Capping
  its draft KV at `max_ctx` is therefore *not* a pure traffic reduction — past the window the drafts
  genuinely differ. It stays end-to-end LOSSLESS because verify gates every emitted token, but an
  EAGLE3 comparison must be read as **accept-len at a stated window**, never as byte-equality of
  drafts. (The replay-vs-eager gate in §3 is unaffected: both legs use the same window.)
* Its prompt seed was **O(prompt) in interpreted Python** — `seed_kv` returned `P-1` one-row tuples
  into a list that every later draft step re-`stack`ed. The buffered seed writes the global buffer
  directly.

---

## 3. The gate, and how it is run

`tools/propose_capture_ab.sh` — one worktree, one image, two boots. The legs differ only in whether
`propose_body` is replayed or run eagerly (`MINISGL_SPEC_PROPOSE_NOCAPTURE=1`, a **temporary A/B
override that is deleted before merge**).

* **identity** — `MINISGL_SPEC_DEBUG=2` prints every drafted chain; greedy, fixed prompt, fixed seed,
  NREQ=1; the legs' `draft=[...]` lines are diffed VERBATIM. Capture must be bit-identical to eager
  for the same inputs, so **one differing chain is a failure, not noise**.
* **timing** — `MINISGL_SPEC_TIMING=1` gives a cuda-synchronized per-phase running mean plus the
  replay/eager split. A leg whose `replay=` stays 0 is not testing what it claims to test.
* **provenance** — the env witness is read with `docker exec <c> sh -c "... < /proc/1/environ"` (an
  unquoted redirect is evaluated by the HOST shell and reads the host's pid 1), and the boot log must
  show either `PROPOSE graphs CAPTURED buckets=[...]` or `propose capture disabled by A/B override`.

Measured results are in `tools/propose_capture_ab_*.txt`.

---

## 4. Why the z-lab and CCA DFlash drafters stay eager

This is a property of those checkpoints, not a switch.

* **z-lab (`z-lab/Qwen3.6-35B-A3B-DFlash`, `z-lab/Qwen3.5-4B-DFlash`)**: `spec/dflash.py` builds them
  with neither `causal` nor `sliding_window`, so `DFlashDraftModel._block_mask` returns `None` and the
  block attends the **entire prefix bidirectionally, unmasked**. There is no fixed capacity that can
  hold an unbounded prefix, and imposing one would change the drafter's output rather than skip work
  the mask already discards. (Those checkpoints' configs *do* declare `sliding_window: 4096` and a
  `layer_types` list with one `full_attention` layer, which minisgl ignores. Honouring that is a
  separate change with its own accept-len consequences — and one layer would remain O(P) regardless.)
* **CCA-recurrent (`DFlashCCADraftModel`)**: a different architecture with no per-layer prefix at all;
  its whole context is a single fused seed. Nothing here applies to it.

`DFlashProposer.propose` dispatches on the drafter, and `propose_capturable` is set by
`_build_laguna` only when `causal and sliding_window > 0`.

---

## 5. DDTree — a reasoned NO for the tree assembly, and what was done instead

**DDTree is not a proposer.** There is no `ddtree` branch in `make_proposer` and no `propose` method.
It is a scheduler MODE (`_spec_decode_step_dflash_ddtree` / `_spec_decode_step_ddtree`) layered over
DFlash or TiDAR, made of three forwards:

1. **F1, the marginals** — `DFlashProposer.propose(..., topk=K)`, or TiDAR's `block_predict`.
   → **now captured** by this work (DFlash) / already captured (TiDAR).
2. **F2, the tree-verify** — one ancestor-masked target forward. → **already captured**
   (`GraphRunner.capture_ddtree_verify_graphs`).
3. **F3, the commit** — the ordinary linear verify. → **already captured**.

So there was never a "DDTree propose" to capture, and after F1 there is no eager *forward* left in
the path at all.

### The premise "capture requires a fixed topology" is false here — do not act on it
The ancestor mask enters the captured graph as a **runtime tensor copied into a static buffer**
(`RDNA4Backend.prepare_ddtree_verify_for_replay` does `_dcap_custom_mask[:tq,:kv].copy_(src_mask)`),
and `can_use_ddtree_verify` gates only on `extend_len == tree_qlen` and `device_len <= max_kv`.
**Topology is never consulted.** The DDTree tree-verify already runs under CUDA graph with a fully
data-dependent topology, today.

Conversely, forcing a fixed topology is not free: the existing `MINISGL_DDTREE_STATIC` mode freezes
the best-first shape for a *synthetic geometric* marginal (`build_static_template`, `logp[i][k] =
-decay*k`, identical at every depth), so it **changes which nodes get proposed** whenever the real
marginals' cross-depth score ordering differs from that canonical one — which is essentially always,
since the heap's choice between breadth at depth 1 and depth at rank 0 depends entirely on the actual
log-prob gaps. The module concedes the trade itself. So "fix the topology without changing which
tokens get proposed" is **not achievable**, and it is **not needed**.

### What genuinely cannot be captured, quoting the step
`build_draft_tree` is a Python `heapq` over **host floats**:

```python
heap = [(-topk_logp[0][0], 0, (0,))]
while heap and tree.n_draft < budget:
    neg_score, _, rt = heapq.heappop(heap)      # <-- host control flow on a data-dependent value
    ...
    heapq.heappush(heap, (-sib_score, tie, sib))
```

The loop's trip count, the pop order and the resulting `parent[]`/`depth[]` arrays are all functions
of values that only exist on the host, having arrived there through a `.cpu().tolist()`. A CUDA graph
records a fixed sequence of kernel launches; it cannot contain a heap, a `while` on a Python float
comparison, or a dict. `ddtree_walk` is the same shape (greedy host descent over `argmax_per_node`).
**This is the evidenced NO.** It is a host-cost problem, not a launch-storm problem.

### What was fixed instead
The dynamic path was materialising its ancestor mask the worst possible way: for each node, a deny
row followed by one **individual device store per ancestor** — `n + n*avg_depth` single-element GPU
writes per request per step (~163 at the default budget of 32). That is pure launch overhead, and the
data (`tree.parent`) is already on the host. `ddtree.ancestor_block_host` now assembles the whole
`[tree_qlen, tree_qlen]` block in Python for ONE H2D copy. **Byte-identical content — no proposed or
accepted token changes.**

### Remaining DDTree gaps (recorded, not fixed here)
* `docker-compose.yml` forwards only `MINISGL_DFLASH_DDTREE`, `MINISGL_DDTREE_TOPK` and
  `MINISGL_DDTREE_BUDGET`. `MINISGL_DDTREE_STATIC`/`_SEG`/`_FUSE`/`_MAXCTX`/`_MAXBS` and
  `MINISGL_TIDAR_DDTREE` are not forwarded, so a compose serve cannot reach the baked-mask path at
  all. Any DDTree number taken through compose to date is of the dynamic-heap variant.
* `tree_qlen = budget + 1` = **33** (DFlash default), which is on the wrong side of the M≤16
  decode-GEMV boundary. Any DDTree-vs-linear-verify throughput comparison at the default budget is
  confounded by a kernel-family switch rather than by the tree. `budget <= 15` is the only
  cliff-comparable configuration.

---

## 6. Known limits of the shipped path

* **DP+EP keeps propose eager**, deliberately and permanently: an in-graph MoE `all_gather` pins a
  fixed N while an idle replica self-agrees its own N eagerly. EP-over-TP is fine (the draft head is
  built replicated). Logged at init.
* **Propose graphs are not in `Engine._graph_capture_bytes`.** They are captured after the KV pool is
  sized, so they come out of the `(1 - memory_ratio)` slack — the same position MTP's capture was
  always in. Boot-time capture at least makes an over-subscription loud and early instead of a
  mid-serve stall; OOM at capture is caught, logged at ERROR, and degrades to eager with the counters
  showing it.
* **A request whose slot falls outside DFlash's memory-capped prefix ring skips propose** and decodes
  plain. Lossless; warned once.
* **Reduced-vocab drafting is still unbuilt** and is now the largest remaining propose lever: after
  the window fix and capture, DFlash propose is LM-head-bound (a 100352-wide borrowed target head,
  ~205 MB/rank at TP=2, streamed per step), and the sibling `Laguna-XS.2-speculator.dflash`
  checkpoint already ships `draft_vocab_size: 32000` with the d2t machinery present but hard-disabled
  for the Laguna build. That is a 3.1× head reduction sitting unused.
