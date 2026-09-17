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
| **DDTree** | has no `propose` — it is a scheduler MODE | See §5. Its tree ASSEMBLY cannot be captured. Its F1 propose is DFlash's, and that is now **observed** engaged (`propose-graph replay=148 eager=0`) rather than asserted — DDTree could not boot on Laguna at all until the fix in §4/§5. |

The `propose-graph` field of the `[spec-timing]` line is the readout for this table, and it now
distinguishes `captured` from `ALWAYS-EAGER(never: …)` from `ALWAYS-EAGER(failed: …)`. A row that is
eager by construction no longer prints as `eager=0`.

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

### 3.1 Results

**Engagement.** Every captured leg logs `replay=N eager=0` for the whole run, at every batch size,
for all three proposers. No leg silently fell back.

**Bit-identity, replay vs eager (same body, same inputs):**

| proposer / model | verdict |
|---|---|
| DFlash / Laguna, NREQ=1 | **PASS** — 178/178 drafted chains identical verbatim, same completion md5 |
| DFlash / Laguna, NREQ=8 | **PASS** — 1342/1342 chains identical as a multiset (60 lines differ in log ORDER only; see below) |
| EAGLE3 / GLM-4.7-Flash, NREQ=1 | **PASS** — 354/354 chains identical verbatim |
| MTP / Qwen3.6-35B-AWQ, NREQ=1 | **NOT RUNNABLE** — see §3.2 |

The NREQ=8 comparison is a multiset, deliberately: each leg is an independent boot, so admission
order and per-step batch composition permute the interleaving of per-request debug lines without
changing a single chain. Calling that a failure would be an artefact of the instrument.

**A CONTROL run establishes the noise floor** — two boots of the *identical captured* configuration
(`CTL=1`), which is what makes the passes above mean anything:

* DFlash / Laguna: **144/144 identical.** Noise floor is ZERO. The DFlash and EAGLE3 passes are real.
* MTP / Qwen3.6-35B-AWQ: **208 of 280 chains differ.** See §3.2.

**Timing** (cuda-synchronized `[spec-timing]`, per-50-step running means differenced to steady state;
first window discarded as warmup):

| config | propose ms/step | tok/s |
|---|---|---|
| DFlash NREQ=1, captured vs eager-same-body | 5.7 vs 6.1 | 78.3 vs 75.4 |
| DFlash NREQ=1, **vs the parent commit** | **5.7 vs 6.0** | **79.6 vs 74.6** |
| DFlash NREQ=8, captured vs eager-same-body | 13.7 vs 13.8 | 154.2 vs 152.2 |
| DFlash NREQ=8, **vs the parent commit** | **13.7 vs 44.4 (3.2x)** | **154.2 vs 135.9 (+13.5%)** |
| MTP NREQ=1, captured vs eager-same-body | 5.4 vs 5.7 | (confounded, §3.2) |
| EAGLE3 NREQ=1, captured vs eager-same-body | 6.6 vs 6.7 | 58.4 vs 56.2 |

**Read this honestly: the graph replay itself is the small half.** At NREQ=1 it buys 1–8% of propose.
The large win — 3.2x at NREQ=8 — comes from the *batched, fixed-shape rewrite that capture required*,
because the old per-request loop issued ~150 launches PER REQUEST and the new body issues ~180 for
the whole batch. Capture and that rewrite are not separable as a deliverable (the old body could not
be captured), but they are separable as a measurement, and the A/B legs above measure only the
replay. At NREQ=8 replay-alone is ~1%, exactly as expected once the launch count is amortised over
8x the work.

**And the propose floor is NOT launch-bound**, which contradicts what the O(window) work inferred
from a floor that did not move. At bs=1 the captured DFlash propose is 5.7 ms against a ~1.6 ms
weight-bandwidth roofline (drafter trunk ~875 MB + LM head ~205 MB/rank at 700 GB/s). Removing every
launch moved it 0.4 ms. The remaining 3.5x is small-M GEMM efficiency: the drafter's `_PlainLinear`
calls `F.linear` (rocBLAS) at M=16, bypassing the tuned `dense_bf16_gemv` that
`layers/minv.py` already dispatches below M=16 and that measured 5.4x over rocBLAS on the LM head.
Routing the drafter's linears through the shared primitive is the next lever, and it is a
"share the compute primitive" fix, not a new kernel.

Raw logs: `tools/propose_capture_ab_*.txt`, `tools/propose_capture_vs_base_*.txt`.

### 3.2 MTP: the gate is not runnable on this model, and that is a finding

MTP propose on Qwen3.6-35B-A3B-AWQ is **not bit-reproducible against itself**. The control — two
boots of the identical captured configuration, same prompt, same seed, greedy — differs in 208 of 280
drafted chains. The replay-vs-eager difference (80 of 276) is *smaller than that noise floor*, so it
cannot be attributed to capture, and no bit-identity claim about capture can be made on this model
either way.

The obvious suspect is named by the kernel itself — `quant/kernels.py` on the fused MoE gemm2 taken
below M=2, which is exactly what an MTP draft head hits at bs=1:

> "NOT bit-exact vs gather_reduce: the atomic reduction order varies, so this is a tolerance-gated
> path, never a bit-exact one."

But that is **not the whole cause**: re-running the control with `MINISGL_MOE_G2FUSE=0` (the
documented bit-exact gemm2) still gives 74 of 264 chains differing. Something else in this model's
decode is non-reproducible too. Laguna+DFlash reproduces exactly under the same harness, so it is
specific to this stack (GDN-hybrid + AWQ MoE), it is upstream of propose, and it predates this work.
**Not chased — recorded.** A greedy serve that does not reproduce itself is worth its own
investigation.

**Independently confirmed at the SERVED-TEXT level** (review follow-up, `tools/fixgate_neutral.txt`),
which matters because a later review reported that fp8 KV alone explained it. It does not. With
**both** `MINISGL_KV_FP8=0` **and** `MINISGL_MOE_G2FUSE=0`, one greedy prompt at fixed seed, issued
twice inside a single boot:

* `753f08d2` returned **two different completions** at max_tokens 128;
* the fixed tree returned **the same two, in the opposite order**, and two more at 256.

So Qwen3.6-35B-A3B-AWQ is nondeterministic at the emitted-text level from ~128 tokens, on **both**
trees, with every known nondeterminism source disabled. Laguna + DFlash under the identical harness
is `repeat=SAME` at all four lengths. It is intermittent (some boots reproduce cleanly at all
lengths), which is why a single clean run must not be read as "the floor is zero" on this model.

What IS established for MTP: capture engages 100% (`replay=N eager=0`), propose drops 5.7 → 5.4
ms/step, and losslessness is structural — verify gates every emitted token regardless of what the
drafter proposes.

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

### These configurations used to CRASH the engagement readout, and used to READ GREEN

Two defects found in review, both fixed, both re-gated by `tools/fixgate_capture_crash.sh`:

1. **`propose_capture_stats` assumed the capture state existed.** Inheriting `CapturableProposer`
   declares a proposer *can* be captured; whether it *is* depends on the checkpoint. DFlash skips
   `init_propose_capture` for every drafter above (and for `MINISGL_DFLASH_PERSIST_KV=0`), so
   `self._pc_replays` did not exist — and the scheduler's timing line dereferenced it every 50
   steps. With `MINISGL_SPEC_TIMING=1` (both compose-forwarded) the worker died mid-serve:
   `AttributeError: 'DFlashProposer' object has no attribute '_pc_replays'`. It killed exactly the
   configurations the capture work claimed to have "verified at runtime", under exactly the
   diagnostic added to prove those claims. Now read through `getattr`, as the base stub always was.
2. **100% eager rendered as `eager=0`.** Any proposer with no captured propose (n-gram, TiDAR, the
   drafters above, DP+EP) fell through to the base stub's `(0, 0, [])`, so the line read
   `propose-graph replay=0 eager=0` — the exact "green number over a silent eager fallback" this
   whole exercise exists to prevent. `propose_capture_stats` now returns a `ProposeCaptureStats`
   carrying a **mode** (`captured` / `never` / `failed`) plus a reason, and the line renders
   `propose-graph ALWAYS-EAGER(never: <why>)` instead. `never` is by construction; `failed` is a
   degradation (graphs off, capture OOM'd) and is named as one.

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

### DDTree could not BOOT on the one model where it mattered — now it can

Review finding, and it invalidated the sentence "its F1 propose is DFlash's, so it is captured by
this work": `MINISGL_DFLASH_DDTREE=1` on **Laguna** — the only model whose DFlash drafter has a
capturable propose — died at boot with

```
AssertionError: SWA metadata missing (is_swa_hybrid not wired?)      attention/rdna4.py
  <- GraphRunner.capture_ddtree_verify_graphs                        engine/graph.py
```

because `RDNA4Backend._ddtree_verify_metadata_static` (`attention/hip.py`) populates **no** `swa_*`
fields, while the K+1 verify capture allocates a per-qlen ring block table and `out_loc`. So the
first sliding layer of the capture *warmup* asserted, and no DDTree serve could ever exercise the
captured propose. The crash is **pre-existing**: reproduced here on `753f08d2`, and the review
reports the identical assert on the phase parent `d276137c` with `git diff` showing this phase never
touched that metadata function. "Pre-existing" does not make "unverifiable" acceptable, though, and
it was not disclosed.

**Fix:** `capture_ddtree_verify_graphs` now **declines** on an SWA-hybrid model with a warning
instead of asserting. The tree-verify runs eager there — the scheduler's own `prepare_metadata` does
build the SWA fields, so the eager path is complete — which is slower but **runnable, and therefore
gateable**. Wiring a sliding-window ring at `tree_qlen` is the real fix and is a separate change
(and is of doubtful value while `tree_qlen = 33` sits past the M≤16 boundary anyway).

Gate: `tools/fixgate_ddtree.sh` — pre-fix must crash, fixed must serve, must show DFlash PROPOSE
capture engaged *under DDTree*, and must be reproducible boot-to-boot. **Result**
(`tools/fixgate_ddtree.txt`, Laguna TP=2, `MINISGL_KV_FP8=0`):

* pre-fix `753f08d2`: `NEVER BECAME READY`, `AssertionError: SWA metadata missing` out of
  `Capturing ddtree-verify graphs` — and note it dies *after* `DFlash PROPOSE graphs CAPTURED
  buckets=[1]`, which is precisely why the capture claim looked fine and was untestable.
* fixed: `ddtree-verify CUDA graph: SKIPPED on an SWA-hybrid model … runs EAGER`, then coherent text,
  and — the point of the exercise — **`propose-graph replay=148 eager=0 buckets=[1]`**. The DFlash
  captured propose is now *observed* engaged under DDTree instead of asserted.
* reproducibility: two independent boots agree byte-for-byte at 64 and 128 tokens, and **disagree at
  256 — including between the two requests of the SAME boot**. That is the fused MoE gemm2's
  documented run-to-run atomic reduction order (Laguna-XS.2 is a 256-expert MoE), not anything in
  this phase; it is why the determinism gates elsewhere force `MINISGL_MOE_G2FUSE=0`. DDTree's noise
  floor on this stack is therefore **not zero past ~128 tokens**, and no DDTree text comparison
  should be run without pinning that kernel.

### Remaining DDTree gaps (recorded, not fixed here)
* The tree ASSEMBLY remains uncapturable for the reason quoted above (a host heap over host floats).
  That verdict is unchanged; what changed is that it can now be run and measured.
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
