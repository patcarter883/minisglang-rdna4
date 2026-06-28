# ZAYA1-8B-fp8 — DP launcher + Expert-Parallel (EP) results

Measured on 2× gfx1201 (RX 9070 XT / RX 9070, 16 GB each), native Triton-free path: `--attn hip` +
W8A8-fp8 MoE kernel + **CUDA graphs (bs 1..16)**, `MINISGL_MOE_SCATTER=0`, `memory_ratio=0.90`.
**Validated in an isolated git worktree off `4986cea`** (source-isolation rule, CLAUDE.md) — every
container mounted the worktree, never the shared `$PWD`. Harness: `tools/ep_validate.sh` (one 2-card
lease, each phase hard-timeout-wrapped, fail-fast). All phases under graph capture (graph is the gate).

## Headline — EP's payoff is KV headroom, not throughput

| Config | experts/card | KV pool / card | sat. throughput (conc=64) | coherent | graph capture |
|---|---|---|---|---|---|
| **DP-only** (`--data-parallel-size 2`) | 16 (replicated) | **65,615 tok / 5.01 GiB** | 706 tok/s | ✓ | ✓ [1,2,4,8,16] |
| **DP + EP** (`… --enable-ep`)          | 8 (sharded)     | **114,869 tok / 8.76 GiB** | 737 tok/s | ✓ | ✓ [1,2,4,8,16] |

- **EP frees ~3.75 GiB of expert weights per card** (16→8 experts) → KV pool **+75% (65,615 → 114,869
  tok)**. That is the reason to enable EP for ZAYA: more KV → more RSA rollout concurrency / longer
  context per card (see SERVING_MATRIX.md frontier `B×L ≤ KV`).
- **Throughput is ~neutral** (EP 737 vs DP 706 tok/s, both saturated at conc=64). The per-step EP
  all_gather/all_reduce run **inside the captured decode graph**, so the collective launch cost is
  hidden; the extra KV lets slightly more requests co-batch. EP is not a throughput play — it is a
  capacity play that costs nothing in tok/s here.
- **Greedy parity:** EP and DP both answer the coherence probes identically (single-prompt "What is
  the capital of France?" → `Paris.` byte-for-byte under both). Concurrent probes (2+2, hot↔cold,
  H₂O) coherent on both.

## What was validated (graph is the gate — "not done until graph-capturable")

- **DP launcher (EP off):** one OpenAI endpoint; the tokenizer round-robins each `UserMsg` to exactly
  ONE replica's rank-0 ingress (per-replica `zmq_backend_addr` keyed by `dp_rank`); replies funnel
  through one detokenizer demuxed by globally-unique `uid`. Removes the need for a multi-backend RSA
  shim router. Each replica captures + replays its own decode graphs independently (no cross-replica
  coupling). Coherent single + concurrent, 0 errors, rc=0.
- **EP (`--enable-ep`, dp_size=2):** experts sharded 8/card (`weight.py:_store_expert` loads only
  `[dp_rank*E/dp : +E/dp]`; `MoELayer` sized to the local count). Dispatch/combine = `all_gather`
  token rows + top-1 route → masked local-expert `w8a8_moe` → `all_reduce(SUM)` → slice OUR rows,
  **all INSIDE the captured decode graph** at fixed (agreed-bs) shapes. The only per-step host sync is
  one gloo `all_reduce(MAX)` of `[prefill_tokens, decode_bs]` OUTSIDE the graph (it selects which
  graph to replay). Graphs captured at [1,2,4,8,16] including the in-graph EP collectives — no eager
  fallback. Coherent single + concurrent (conc=64, 0 errors), rc=0.

## Bugs found + fixed during isolated re-validation

The prior session's EP had only ever reached "boots + one fix"; re-validation in isolation surfaced
three real bugs (all under graph capture / concurrency, none visible in a single-request eager probe):

1. **Prefill lockstep agreed the request COUNT, not the token COUNT.** `ep_loop` used
   `prefill_batch.size` (= `len(reqs)`) for the common-size `all_reduce(MAX)`, but the MoE all_gather
   pads to `hidden_states.shape[0]` = total query *tokens*. A 31-token real prefill on one replica
   all_gathered N=31 while the idle replica padded its dummy to N=1 → mismatched RCCL shapes → the
   collective wedged at the next CUDA sync. **Fix:** agree on `sum(r.extend_len for r in reqs)`.
2. **The idle replica's dummy prefill leaked a KV page + mutated the shared dummy_req.** It ran
   `reqs=[dummy_req]`, so `allocate_paged` allocated a real KV page for the dummy (never freed) and
   `forward_batch`'s `complete_one` advanced the shared `dummy_req`'s lengths every step. The leak
   tripped `CacheManager.check_integrity()` (`free_pages+cache_pages != num_pages`, off by one) the
   moment a replica went idle → it crashed → its gloo peer's `_ep_agree` failed (`Connection closed`).
   This only bit when **both** replicas had real work (the single-request path returned before the
   idle check fired). **Fix:** `_finish_prepare(..., skip_alloc=True)` for the dummy prefill (the
   dummy_req already points at the reserved null page, so no allocation is needed) + snapshot/restore
   the dummy_req's length state around the forward. CCA prefill metadata is built from `batch.reqs`
   and needs one real seq, so the decode-style `reqs=[]` all-dummy trick is NOT usable for prefill —
   hence `reqs=[dummy_req]` + `skip_alloc`.
3. **Eager-EP idle decode replays a non-existent graph (EAGER-ONLY, non-production).** With
   `--graph 0` (`max_graph_bs=0`, no capture, no `GraphRunner.buffer`), an idle replica's all-dummy
   decode batch has `batch.size==0`, so `can_use_cuda_graph = is_decode and 0<=0` → True → `replay()`
   → `AttributeError: no 'buffer'`. In **graph mode (production)** `0<=16` → replay the captured
   `graph_map[common_bs]` (buffer exists) → returns `logits[:0]` correctly, so this never bites. Eager
   EP is "bring-up isolation" only (the spec's `EP2NG` phase); production EP is graph-captured and is
   the validated gate. Left as a documented eager-only limitation (not fixed — eager EP is not a
   shipping path).

## Risks / notes (carried forward)

- **top-1 expert skew across only 2 EP ranks.** ZAYA routes top-1 over 16 experts; sharded 8/8, a hot
  expert lands entirely on one rank → that rank does more MoE work while the other waits at the
  `all_reduce`. Not load-balanced; fine at dp=2 with the masked-sum (correctness is exact), but the
  throughput ceiling is the busier rank. Revisit if skew is measured to matter.
- **`all_gather` ships ALL tokens** (every rank sees every token) — more bandwidth than a true
  `all_to_all`, but `all_to_all` has data-dependent sizes and is NOT CUDA-graph-capturable. The
  fixed-shape all_gather is mandatory for the in-graph path. Fine for dp=2; for larger dp bind
  `ncclAlltoAll` (header already present in `csrc/include/minisgl/nccl227.h`) outside the graph.
- **Idle busy-spin.** When no replica has work, both spin `ep_loop` issuing the gloo `all_reduce(MAX)`
  every iteration (a coordinated idle). Correct + deadlock-free, but burns CPU + gloo traffic when
  fully idle. A coarser idle heartbeat is a future optimization (does not affect correctness).
