# QSA indexer — sparse attention for qwen4_exp (Qwen3.8-Flash-Next)

## 0. What this removes

`models/qwen4exp.py` used to call `QSAIndexer.assert_dense_is_exact(max_ctx_len)` on every
full-attention forward and **raise** the moment any request crossed `indexer_budget` = 2048, against
a checkpoint whose native context is 262,144 — a 128x shortfall. The refusal was correct: below the
budget the sparse selection provably degenerates to dense causal attention, so dense is exact there;
above it, serving dense would be serving *different attention from the model's own definition*, and
the difference is undetectable from the output text.

The consequence, which is the reason this was the priority: **every measurement this model had ever
produced — coherence, greedy-id parity, capture identity, 14.7 tok/s — was taken in the one regime
where the sparse path is bypassed.**

The refusal is now the **no-QSA path's guard only**. It still fires when the sparse path is
unavailable (no `qsa_index` kernels, a KV page size the DSV4 addressing cannot use, `MINISGL_QSA=0`),
because a build that cannot select must not quietly serve dense above the budget.

## 1. The math

Five numbers, parsed once by `attention/qsa/config.py::parse_qsa_profile`, from the checkpoint:

| field | value | meaning |
|---|---|---|
| `indexer_n_heads` | 4 | index **query** heads, `Hi` |
| `indexer_kv_heads` | 1 | index key heads — MQA; the scoring op requires exactly 1 |
| `indexer_head_dim` | 128 | per-head index width, `di` |
| `indexer_budget` | 2048 | tokens each query row may attend, `T` |
| `indexer_compress_ratio` | 4 | tokens averaged into one compressed key, `r` |

Derived: `block_topk = T/r = 512` compressed blocks per query row, and
`index_width = T + r - 1 = 2051` — the expanded token-index row is the 2048 selected tokens plus the
query's own still-incomplete trailing group (at most `r-1` more).

Per full-attention layer (12 of 48; `idx % 4 == 3`), per query row `m`:

1. **compress** — group `g` = tokens `[g*r, g*r+r)`; its key is
   `rope( layernorm_gemma( mean_fp32( k_tok[g*r .. g*r+r-1] ) ), pos = g*r )`.
   Order is load-bearing and invisible when wrong: **fp32 mean THEN norm THEN rope at the group's
   OLDEST member**. Mean-then-norm is not interchangeable with norm-then-mean, and roping at the
   newest member instead of the oldest does not raise — selection just quietly degrades.
2. **score** — `logits[m, g] = ( SUM_h relu( q[m,h] · c_g ) ) / sqrt(di)` over the blocks FULLY
   visible to `m`, i.e. `g ∈ [0, (pos_m + 1) // r)`. No softmax, no V, no output projection. ReLU is
   **per head**, the sum is **over heads**, and the surviving axis is the KEY axis.
3. **top-k** — exact top-512 under a **total order**: value descending, block index ascending.
4. **expand** — block `g` → tokens `g*r .. g*r+r-1`, truncated to `T`, then the query's own partial
   group `[tail_start, pos_m]` appended; fixed width 2051, `-1` padded, valid entries contiguous.
5. **attend** — ordinary paged attention restricted to those tokens' physical KV slots.

Compression must precede scoring in the same forward, or the group a query just completed is
invisible to the query that completed it.

## 2. The HIP decomposition — what was EXTENDED and what is NEW

The reference (`sglang-upstream/.../attention/qsa`) is 3,176 lines, of which 1,051 are Triton and
422 are NV-only tilelang. Neither is servable here: `attn=hip` is this repo's canonical backend and
the `rdna4` Triton fallback cannot even capture a graph. So the selection was **written in HIP** and
the attention half was **not written at all**.

### 2a. The attention half — ZERO new kernel lines

`HIPAttnBackend.forward_sparse` calls `attn_decode.flash_decode_paged` with the **selected physical
slots as its block table at `page_size = 1`**. Per `KERNEL_CORE_POLICY.md`, sparse attention computes
the same shape as paged attention and differs only in WHICH rows it visits, so it is a **caller-side
selection policy on the existing core**, not a fork. Copy-pasting `attn_decode` and editing the key
loop is exactly the debt that fragmented the decode-GEMV family into five copies.

Two consequences that are the design, not side effects:

* **Bit-exactness at/below the budget is STRUCTURAL.** Below the budget the selection is every
  visible token in ascending order, so slot `j` of the sparse table IS the row the dense call's
  `bt[j/page]*stride + j%page` names, and the kernel's key loop visits them in the same order with
  the same online-softmax accumulation. Same kernel, same order, same bits.
* **The width-inference landmine (61d96cf / 0972e387) is DISSOLVED, not added to.** That bug class is
  a kernel policy inferred from a shape that differs between capture and replay. `sel_slots` has a
  FIXED width (`index_width` = 2051) at every context length, so the split-K policy `attn_decode`
  infers from `max_blocks * block_size` is a constant 2051 on the sparse path — it cannot change with
  sequence length. The one place it bites is the dense-vs-sparse A/B, where the two *widths* differ
  (max_seq vs 2051) and therefore pick different policies; `--pin-split`
  (`MINISGL_ATTN_SPLIT_MIN_CTX` above both) forces single-pass on both sides.

### 2b. The selection half — ONE new package, `qsa_index`, one core, two policies

`rdna4-hip-kernels/qsa_index` (`qsa_index_kernels.hip`, 682 lines) — three kernels:

* `qsa_score_kernel<T, HEAD_DIM, HEADS, ROWS, KLoad>` — **one `__global__`, two KLoad policies.**
  Packed (prefill, contiguous keys) and paged (decode, page-table indirection) differ ONLY in how a
  key row is addressed, so they are a `KLoad` **policy** on that core — the same axis
  `gemv_decode_core` calls GATHER. `parity_qsa_index.py` asserts the two produce bit-identical logits
  on identical keys, which is the property a fork would silently lose.
* `qsa_topk_kernel` — exact 4-round 8-bit radix select + ascending compaction, 1024 threads.
* `qsa_expand_kernel` — blocks → member tokens + the partial trailing group.

**Why the scorer is a NEW kernel and not an extension of `attn_decode`** (checklist item 3 — the math
genuinely differs): `flash_decode_paged` reduces OVER keys (online softmax) and emits one vector per
head; the scorer reduces over HEADS and **materialises a value per key**. Bending the flash core into
that shape is an `if constexpr` fork of its whole loop body plus deletion of its entire PV half — a
fork wearing a template parameter. What IS shared is the paged *addressing*, and that is isolated in
`PagedKLoad`.

**Why a new PACKAGE and not a format in an old one**: `qsa_index` is an algorithm family
(weight-free scoring, radix selection, index expansion), matching this repo's per-algorithm package
convention (`attn_decode` / `gdn` / `cca` / `mla`). It adds no weight format and forks no core.

**Nothing infers a policy from a tensor shape.** `rows_per_cta` (1/4/16) and `num_key_cols` are
EXPLICIT op arguments; `page_size` is address arithmetic only and selects nothing. `rows_per_cta` is
a pure launch policy that touches no reduction order, so all three values give bit-identical output —
asserted, not asserted-in-prose.

**Determinism / the tie rule.** The upstream `fast_topk` collects survivors with an atomic counter
and its own header says the within-row order is unspecified; at k-th place it keeps whichever tied
entry arrives first. That is a cross-rank divergence waiting to happen (cf. `_ep_route`'s
`torch.topk` breaking k-th-place ties opposite to the served `moe_route_align`). This kernel defines
a **total order** — `ordered_key(logit)` descending, block index ascending — using the standard
order-preserving fp32→uint32 map, and emits in ascending index order. Every step is
order-independent: integer histograms (commutative and exact), a deterministic block scan, no
floating-point atomic.

### 2c. The engine half — `python/minisgl/attention/qsa/`

761 lines across five files (`__init__.py` is a 25-line re-export):

* `config.py` — the five numbers, once. `require_page_size` **raises** rather than falling back: the
  compressed key is addressed by `physical_kv_slot // r` with no allocator and no ownership state,
  and that identity holds only when a group cannot straddle a page (`page_size % r == 0`). At
  `page_size = 1` it would silently address another request's compressed key — right shapes,
  plausible text, wrong model.
* `cache.py` — the memory economy. **Raw index keys are never a per-token cache**: a token's raw
  `k_tok` is dead once its group of `r` is averaged, so the raw keys live in a ring of exactly `r`
  slots per REQUEST ROW (`table_idx*r + pos%r`) — 786 KiB for the whole engine at 64 rows. The
  compressed cache is one row per group, `num_kv_slots / r` rows per index layer. One extra row at
  the end is the graph-capture scratch (the upstream reference writes slot ZERO for non-boundaries,
  which is only safe if slot 0 is a reserved null; this engine's allocator hands slot 0 to a real
  request).
* `runtime.py` — `QSAPlan.build` derives everything batch-shaped ONCE per forward (positions, the
  request→row map, the visible-block window, the compressed page table, the ring slots, the
  compression plan), so 12 index layers pay ONE index-arithmetic bill. `QSARuntime.select` runs the
  four stages per layer. The `[rows, blocks]` fp32 scoring workspace is row-tiled under a 128 MiB
  budget, and the sparse attention is row-tiled at 256 (its split-KV partials are
  `[rows, q_heads, splits, head_dim]` fp32).
* `ops.py` — the HIP ops plus a **torch reference** (`MINISGL_QSA_OPS=torch`). It is not a serving
  fallback; it exists so the two backends can be diffed.

**Sparsity is a MEASURED output, not an assumption.** `QSARuntime.total_visited` is the number of KV
slots the attention kernel actually reads; `total_dense` is the causal count `min(pos+1, seq_len)`.
A "sparse" path that quietly selected everything passes every coherence and retrieval test; the ratio
is the only thing that catches it.

## 3. The dense-equivalence gate (≤ indexer_budget) — `tests/qwen4exp_qsa_gate_test.py`

Below the budget the reference clamps `row_topk = min(topk, visible)`, so sparse selection is
EXACTLY dense causal attention. That makes a free, immediately-runnable, exact gate.

* **G1 — same-code floor.** `eps0 = 0.000e+00` with greedy ids identical, bs=1, dense path run twice
  in separate builds (`MINISGL_MOE_G2FUSE=0`, which bypasses the non-bit-exact decode gemm2 atomic
  scatter). Established BEFORE any comparison, and it immediately paid: the first harness (4-layer
  GDN-hybrid subset, random weights) returned `eps0 = 3.4` and NaN from the first GDN block — i.e.
  the initial QSA "result" of `max|d| = 3.2` was measuring harness noise. Two harness defects were
  found by that floor and fixed: random-weight GDN is NaN-unstable (the subset is now
  all-full-attention, which also puts an indexer on every layer instead of one in four), and a
  randomly-filled NVFP4 build reaches the MoE kernel with fp8 scales the real loader would have cast
  to fp16 (the subset builds unquantized).
* **G2 — decode is BIT-EXACT, and it is tested IN PLACE.** `MINISGL_QSA_DENSE_CHECK=1` runs the same
  `q` and the same KV cache through both the sparse call and the ordinary dense paged call inside one
  forward. **16/16 decode attention calls bit-identical, worst |d| = 0.000e+00**, at prompt_len 1024
  + 4 decodes; greedy ids identical on both legs. An end-to-end logit comparison CANNOT test this —
  the prefill kernels genuinely differ, so by the first decode step the two legs' caches have already
  diverged.
* **G2 — prefill is NOT bit-exact and is not claimed to be.** Dense prefill runs
  `attn_hip.flash_prefill` (WMMA tiles); sparse runs the paged decode core per query row →
  `max|d| = 1.42e-2` on logits (5.99e-3 relative to `|logit|max = 2.37`). The first explanation
  ("reordering noise") was REFUTED by its own control: the two dense prefill kernels
  (`flash_prefill` vs `flash_prefill_paged`) agree BIT-FOR-BIT (`eps_kernel = 0.0`). What settles it
  is a float64 reference over the same stored paged KV: **the sparse prefill attention is CLOSER to
  exact than the dense one at every sampled row** (mean rel 1.65e-3 vs 2.03e-3).
* **Caveat, recorded.** The split-KV policy is keyed on the block-table row width, which differs
  (max_seq vs 2051). Without `--pin-split` the decode check is 9/12 exact, worst 1.5e-5; with it,
  16/16.
* **G3 — above the budget, vs a float64 reference.** The selection is re-derived on the HOST in
  float64 from the tapped `q` + compressed keys, scored as relu-per-head-then-sum / sqrt(128),
  top-512 under the kernel's own total tie order, expanded, and compared as an index SET:
  **4/4 sampled rows exact at a 7000-token prompt.** Separately, the engine's torch ops were
  cross-checked on CPU against the kernels package's own float64 spec
  (`qsa_index_reference.py`): top-k index sets identical on 4 shapes incl. (3, 2048, 512), expand
  byte-identical on 3 shapes, `score_paged` max_abs 8.9e-7 with matching `-inf` masks. And the two op
  backends agree end to end: `MINISGL_QSA_OPS=torch` and `=hip` give the identical sparsity ratio
  0.4377 at ctx=8192.

Artifacts: `docs/measurements/QSA_2026-09-06/{qsa_gate_final,qsa_gate_pinned,QSA_REACH}.json`
(the `out/` copies are gitignored — `*.json` is in `.gitignore`, which is why the stage-4
artifacts were never actually committed and are re-landed here).

## 4. Long-context results

48-layer checkpoint, TP=2 (RX 9070 XT + RX 9070), graphs OFF, `--page-size 16`,
`--weight-offload-device-gb 7.4 --host-gb 26 --max-extend-tokens 1024 --memory-ratio 0.90`.
Sampled at the checkpoint's own `generation_config`: **temperature 1.0, top_k 20, top_p 0.95** —
never greedy. Harness `tests/qwen4exp_qsa_longctx_test.py`, artifact
`docs/measurements/QSA_2026-09-06/r2b.json`.

**Two different ceilings, and they are not the same number.** The engine ACCEPTS up to
`engine_max_seq_len = 105,104` tokens (§5); the longest prompt actually driven end to end with a
coherence and retrieval verdict is **65,530**. The row above it in the ladder (131,056) is past the
accepted ceiling and was correctly refused. Nothing here claims 105,104 was generated at — only that
it is what the engine admits.

**The prompt is a needle-in-a-haystack, because that is the failure mode a wrong selection actually
has.** A bad sparse selection does not emit noise — it emits fluent text that has FORGOTTEN the
early part of the prompt. So the haystack is numbered, varied prose (a haystack of one repeated
sentence makes every compressed block score identically and measures the tie rule, not retrieval)
with two facts planted at known depths: an EARLY needle at 2% and a LATE needle at 90%. The late one
is the control — if it also fails, the failure is the model or the harness, not the selection.

| ctx (actual) | batch | prefill s | prefill tok/s | decode ms/step (fwd) | decode tok/s (fwd) | decode tok/s (wall) | sparsity visited/dense | early needle (2%) | late needle (90%) | gen tok |
|---|---|---|---|---|---|---|---|---|---|---|
| 4087 | 1 | 67.084 | 60.9 | 60.8 | 16.45 | 10.53 | 0.746899 | True | True | 37 |
| 16382 | 1 | 161.653 | 101.3 | 61.18 | 16.35 | 6.05 | 0.234062 | True | True | 37 |
| 32761 | 1 | 416.011 | 78.8 | 61.59 | 16.24 | 3.39 | 0.121088 | True | True | 32 |
| 65530 | 1 | 519.601 | 126.1 | 61.94 | 16.14 | 2.32 | 0.061539 | True | True | 37 |
| 131056 | 1 | **REFUSED — Rejected** | | | | | | | | |
| 16382 | 2 | **REFUSED — ValueError** | | | | | | | | |

**Coherence verdict: PASS, at every length that ran.** Judged against the four degeneration
signatures (`degeneration-signatures-triage-table`) with the mechanical part computed per run and
kept in the artifact: no loop (longest immediate repeat run 0 at every length), no letter-spell
(longest single-character run 0), no token noise (non-ASCII fraction 0.0), distinct-word ratio
0.85-0.87. The model answers in ONE correct sentence and stops on EOS at 32-37 tokens rather than
running out the 96-token budget — which is itself part of the verdict: a model that has lost its
context does not stop, it keeps going.

**And the retrieval result is the one that could not be faked.** The early needle sits at 2.0% depth
in every prompt (line 4 of 201 at 4k, 16 of 801 at 16k, 31 of 1,572 at 32k, 63 of 3,146 at 64k),
i.e. deep inside the region a broken selection drops first, and it is recovered VERBATIM including
the digits at every length while the selection is discarding 25%, 77%, 88% and **94%** of the keys
respectively.

**ctx=4087** — `{'chars': 125, 'words': 20, 'distinct_word_ratio': 0.85, 'longest_immediate_repeat_run': 0, 'longest_single_char_run': 0, 'non_ascii_char_frac': 0.0, 'tokens': 37}`

```
The Meridian vault access code is 47-ZEBRA-9183, and the night supervisor on the eastern depot roster is Dr. Imogen Halloway.
```

**ctx=16382** — `{'chars': 125, 'words': 20, 'distinct_word_ratio': 0.85, 'longest_immediate_repeat_run': 0, 'longest_single_char_run': 0, 'non_ascii_char_frac': 0.0, 'tokens': 37}`

```
The Meridian vault access code is 47-ZEBRA-9183, and the night supervisor on the eastern depot roster is Dr. Imogen Halloway.
```

**ctx=32761** — `{'chars': 97, 'words': 15, 'distinct_word_ratio': 0.8667, 'longest_immediate_repeat_run': 0, 'longest_single_char_run': 0, 'non_ascii_char_frac': 0.0, 'tokens': 32}`

```
The Meridian vault access code is 47-ZEBRA-9183, and the night supervisor is Dr. Imogen Halloway.
```

**ctx=65530** — `{'chars': 125, 'words': 20, 'distinct_word_ratio': 0.85, 'longest_immediate_repeat_run': 0, 'longest_single_char_run': 0, 'non_ascii_char_frac': 0.0, 'tokens': 37}`

```
The Meridian vault access code is 47-ZEBRA-9183, and the night supervisor on the eastern depot roster is Dr. Imogen Halloway.
```

**ctx=131056 REFUSED**

```
input sequence length 131056 exceeds the servable maximum 105104 (KV pool); request rejected  [relabelled post hoc — see post_hoc_correction]
```

**ctx=16382 REFUSED**

```
QSA prefill chunk is not group-aligned: cached_len=2 is not a multiple of compress_ratio=4. The KV page size must be a multiple of r so a group cannot straddle a chunk.
```


### 4a. One non-reproducing GPU hang, recorded

The first attempt at this operating point (`r2`, identical flags) died on the FIRST ladder rung with
`HW Exception by GPU node-1 ... reason :GPU Hang` on both ranks, immediately after
`fp8_wmma.mmq_fp8_moe_gemm1_silu(wmma+e2m1)` first engaged at the prefill batch width. Per
CLAUDE.md's rule for exactly this signature the run was repeated, and `r2b` — same flags, same
image, same kernels, same box — completed the same rung and every rung after it. **It did not
reproduce, so it is recorded as a wedge, not as a defect in the code it launched.** Both cards were
health-checked (a 20-iteration 4096^2 bf16 matmul each) between the two runs and were clean. The
`r2` log is kept at `docs/measurements/QSA_2026-09-06/r2.log`; if this signature returns at the same
engage point it is evidence, and the next step would be a 4-layer-subset bisect (which boots in
~30 s instead of ~5 min) over the prefill chunk width.

### 4b. A DEFECT THE LADDER FOUND: an over-length request is answered, not refused (offline path)

The 131,072-token rung is the one that was supposed to produce a clean refusal, and instead it
produced **`'!'`** — a one-token completion, in 15 ms, with zero prefill steps and zero decode steps
and no error anywhere the caller could see.

The scheduler is not at fault: `Scheduler._handle_msg` checks `max_seq_len - input_len <= 0` and
replies with `DetokenizeMsg(next_token=0, finished=True, error="input sequence length ... exceeds
the servable maximum ...")`. `DetokenizeMsg.error` exists precisely for this and its own comment says
the detokenizer "must NOT decode" the filler token. The HTTP path honours it.

**`LLM.offline_send_result` does not.** It reads only `finished` and `next_token`, so filler token 0
is appended to `output_ids` like any other token and decodes to `'!'`, and `error` is dropped on the
floor. Every offline consumer — this harness, `tools/kv_fp8_calibrate.py`, every bench in `tools/` —
therefore reads a REJECTED request as a successful one-token generation. That is the silent-wrong
class: no exception, no log the caller sees, a plausible-looking result.

FIXED here: `RequestStatus` carries `error`, `offline_send_result` skips token accumulation and
records it, `generate` returns it, and this harness now reports such a row as a REFUSAL rather than
as `ok` (which is how it slipped through on the first pass — the harness itself scored the rejection
as a successful run with 1 token and no needles).

### 4c. A SECOND DEFECT, FIXED — and the first diagnosis of it was WRONG

**Status 2026-09-06: FIXED and re-measured green. `tools/serve.sh`'s `qwen4exp` arm is back to
CONC=2.** What follows keeps the original wrong diagnosis visible on purpose, because the fix it
proposed would have been a silent no-op and someone would have shipped it.

The CONC=2 leg — two 16k prompts submitted together, deliberately renumbered so they are NOT copies
— used to fail outright:

```
QSA prefill chunk is not group-aligned: cached_len=2 is not a multiple of compress_ratio=4.
The KV page size must be a multiple of r so a group cannot straddle a chunk.
```

That is `QSAPlan.build`'s own guard doing exactly what it was written to do — it RAISES rather than
falling back to a second addressing scheme, which is the right behaviour and the reason this was a
failed request rather than silently wrong attention.

**THE ORIGINAL DIAGNOSIS WAS PREFIX REUSE. IT IS WRONG.** This document previously said the two
prompts "share a leading token run, the request resumes at the matched length, and that length is a
token count, not a page count", and prescribed rounding the prefix match down to a multiple of `r`.
That fix cannot do anything here: `qwen4_exp` forces the **naive** prefix cache
(`engine/config.py::resolve_prefix_cache` — the PLE recurrent state is not covered by the
recurrent-radix snapshot store), so `handle.cached_len` is **always 0** and there is no prefix match
to round. And a radix match could not have produced it either: every radix `cached_len` is
`align_down(..., page_size)`, and QSA already requires `page_size % r == 0`.

**The real source is chunk PACKING.** `token_budget` is per STEP, not per request. A request's FINAL
chunk is `remain_len` — an arbitrary number — so the leftover budget handed to the next request in
the same batch is arbitrary too, and becomes that request's first chunk; the chunk after it then
starts at a `cached_len` that is that arbitrary leftover. Both recorded reproductions are exactly
this arithmetic and nothing else:

    16382 = 15*1024 + 1022  ->  leftover 2  ->  cached_len=2   (48 layers, TP=2, r2b.log)
     4087 =  3*1024 + 1015  ->  leftover 9  ->  cached_len=9   (4 layers,  TP=1, r3.log)

Two different non-multiples of 4, same guard, same cause — and it is NOT a TP, depth or scale
effect, which is why both are kept.

**The fix** is `PrefillAdder.chunk_gran` (`python/minisgl/scheduler/prefill.py`), set by the
scheduler from `indexer_compress_ratio`: a NON-FINAL prefill chunk's end is rounded DOWN to a
multiple of `r`, and a request for which no aligned chunk fits this step's remaining budget is
handed its resources back and admitted on the next step with a full budget. The final chunk is
exempt — it ends at the prompt's own length, which nothing requires to be a multiple of anything,
because the decode path sources a group's members from the per-request raw-key RING and not from the
chunk. Cost: at most `r-1` tokens of one step's token budget.

**MEASURED GREEN at BOTH recorded shapes**, 48 layers TP=2 on cards 0 (RX 9070 XT) + 1 (RX 9070),
sampled at temp 1.0 / top_k 20 / top_p 0.95 (never greedy), `--graph-bs 0`, 0 harness failures on
both ranks (`QSA_2026-09-06/CAPTURE_CONC/eagS.json`):

| leg | requests completed | gen tokens | early needle | late needle | decode ms/step |
|---|---|---|---|---|---|
| CONC=2 @ ctx=4087 | 2 / 2 | [32, 37] | recovered | recovered | 95.87 |
| CONC=2 @ ctx=16382 | 2 / 2 | [37, 37] | recovered | recovered | 92.08 |

The concurrency gate is asserted OUTSIDE the `ok`-filtered loop on purpose: a leg that RAISED would
otherwise drop silently out of every check above it and the run would still report PASS.

## 5. KV and memory — where the reach ceiling actually is

**The ceiling is not the model and it is not the selection. It is the KV pool.**
`Engine.__init__` sets `max_seq_len = min(config.max_seq_len, num_tokens)`, and `num_tokens` is
`num_pages * page_size` — what is left of the card after the weights.

Per-token KV cost, per rank, derived and then confirmed against the live pool:

    12 full-attention layers x 1 kv head/rank (TP=2) x 256 head_dim x 2 B (bf16) x 2 (K and V)
      = 12,288 B/token/rank   ->   196,608 B per 16-token page      [MEASURED: the engine's own
                                                                     "@ 196608 B/page" sizing line]

Two operating points, same boot procedure, same day, both TP=2 / memory-ratio 0.90 / graphs OFF:

| device tier | device MoE layers | model (KV-sizing line) | available | KV pages | **max context** |
|---|---|---|---|---|---|
| `--weight-offload-device-gb 8.1` | 12 / 48 | 13.45 GiB | 0.51 GiB | 2,810 | **44,960 tok** |
| `--weight-offload-device-gb 7.4` | 11 / 48 | 12.79 GiB | 1.20 GiB | 6,569 | **105,104 tok** |

So the exchange rate is explicit and linear: **one MoE layer moved off the card (0.668 GiB/rank at
the e4m3 two-level scale) buys 3,759 KV pages = 60,144 tokens of context.** Nothing about QSA is
what bounds reach; the weight tier is.

Extrapolating the same arithmetic (and it is an EXTRAPOLATION, labelled as one — only the two rows
above were booted): the checkpoint's full 262,144 needs `262144 * 12288 = 3.00 GiB` of pool, i.e.
`model <= 14.13 - 3.00 - 0.15 = 10.98 GiB`, i.e. **8 device MoE layers (`--weight-offload-device-gb
5.4`)** — three more layers over PCIe. `MINISGL_KV_FP8=1` would halve the per-token bill and get
there at 10-11 device layers instead, but it is UNCALIBRATED on this checkpoint and this repo's own
record is that an uncalibrated fp8 KV serves quietly wrong.

**What QSA itself costs is small and is the point of the design.** The index cache is one compressed
key per group of `r=4` tokens per index layer:

    12 index layers x 128 dims x 2 B / 4 tokens = 768 B/token/rank = 6.25% of the 12,288 B KV bill

MEASURED, both points: `QSAIndexCache(... comp_slots=11244, MiB=33.0)` at 44,976 slots and
`comp_slots=26280, MiB=77.0` at 105,120 slots — 768 B/token in both cases. The raw-key ring does not
scale with context AT ALL, because a raw index key is dead once its group of 4 is averaged: it is
`12 layers x (max_running_req+1) x r x 128 x 2 B` = **36 KiB** at this operating point's 3 request
rows (786 KiB even at 64 rows). That is the whole economy of the design — the thing that grows with
context is 1/16th of the KV, and the thing that would have grown per token does not exist.

**At CONC=2 the pool is SHARED**, so `max_seq_len` is a single-request bound and the aggregate bound
is the same 105,104 tokens across all running requests.

## 6. Throughput

Two numbers are reported for decode and they are not interchangeable, because on this model the gap
between them is enormous:

* **forward-only** — the median of `STEP_LOG`'s per-step wall time with the prefill steps dropped.
  This is the model's compute.
* **wall** — generated tokens over the wall clock of the whole `generate`. On this serve 36-37 of 48
  layers read their experts over PCIe **outside** the timed forward, worth ~42-66 ms/token, so the
  wall tok/s is roughly a third of the forward-only tok/s. Every published qwen4_exp figure
  (14.55 captured / 13.93 eager) is a WALL figure. Where a prompt is long the whole-`generate` wall
  figure is dominated by the PREFILL and is reported separately as "e2e"; "decode tok/s (wall)"
  below is decode tokens over (wall - prefill), which is the quantity comparable to 13.93.

The comparison baseline must be the EAGER leg, not the 14.7 headline: QSA cannot capture (§2c, §7),
so a QSA-vs-captured comparison would be measuring capture, not sparsity. Short-context reference,
same 48-layer TP=2 boot procedure, `docs/measurements/QWEN4EXP_L48_E4M3_HCFUSE_2026-09-05.json`:

| leg | wall tok/s (median of 5) | forward-only ms/step | forward-only tok/s |
|---|---|---|---|
| captured, 12 device layers | 14.55 (max 14.70) | 0.504 | 1983 |
| **eager, 12 device layers** | **13.93** | **27.05** | **36.96** |

### 6a. What QSA costs, and the thing it buys

| context (tokens) | batch | prefill s | prefill tok/s | decode ms/step (fwd) | decode tok/s (fwd) | decode tok/s (wall) | e2e tok/s incl. prefill |
|---|---|---|---|---|---|---|---|
| ~15 (eager baseline, no QSA) | 1 | 0.274 | 55 | **27.05** | 36.96 | **13.93** | 13.93 |
| 4087 | 1 | 67.084 | 60.9 | **60.8** | 16.45 | **10.53** | 0.524 |
| 16382 | 1 | 161.653 | 101.3 | **61.18** | 16.35 | **6.05** | 0.221 |
| 32761 | 1 | 416.011 | 78.8 | **61.59** | 16.24 | **3.39** | 0.075 |
| 65530 | 1 | 519.601 | 126.1 | **61.94** | 16.14 | **2.32** | 0.069 |

Three readings, in order of how much they matter:

1. **The decode step is FLAT in context above the budget.** That is the whole point of sparse
   attention and it is the one result here that could not have been predicted from the ≤2048 gate.
   A dense decode step grows with `seq_len` because the key loop grows; QSA's does not, because the
   visited set is capped at `index_width = 2051` at every length. The measured forward-only step
   times bear that out.
1b. **But the decode WALL time per token is NOT flat, and the growth is not in the forward.**
   Forward 60.80 -> 61.18 -> 61.59 ms while wall-per-token goes 95 -> 165 -> 295 ms across
   4k/16k/32k. The difference is host-side per-step work that sits OUTSIDE `STEP_LOG`'s window —
   `Scheduler` builds the batch, allocates pages, and calls `attn_backend.prepare_metadata` before
   `_forward` starts the clock. This measurement does not attribute it and nothing here should be
   read as saying the serve is flat in context: **the model's compute is flat; the step is not**
   (§8.6).

2. **The step is ~2.2x the short-context eager step, and that is the price of the selection.**
   27.05 ms (eager, ~15-token context) -> ~61 ms with QSA live. Twelve index layers each run four
   stages (ring store, compress, score, top-k/expand) plus a sparse attention over 2051 slots
   instead of a dense one over `seq_len`. Below ~2051 tokens that trade is pure loss — which is
   exactly why `MINISGL_QSA=0` remains a legal configuration for a short-context serve, at the cost
   of re-arming the 2048 refusal.
3. **PREFILL IS THE WEAK POINT AND IT HAS NEVER BEEN MEASURED ON THIS MODEL AT ANY LENGTH BEFORE.**
   The sparse form has no notion of "a sequence's page table" — selection is per QUERY ROW, so every
   row is its own sequence to `flash_decode_paged` and the prefill runs a DECODE kernel once per
   prompt token. Two terms pull opposite ways:
   * the ATTENTION per row saturates at 2051 visited slots, so it stops growing above the budget,
     which is why per-token prefill cost generally FALLS with length;
   * the SCORE stage does not saturate — row `m` scores all `(pos_m+1)/r` visible compressed blocks,
     so the scorer is `O(n)` per row and `O(n^2/r)` per prompt.

   Measured per-token prefill cost: **16.4 / 9.9 / 12.7 / 7.9 ms at 4k / 16k / 32k / 64k.** The
   curve is NOT monotone and **this measurement does not explain the 32k outlier** — it is one
   sample per length on one boot, and reading a model into it would be exactly the kind of
   just-so story this repo's rules exist to prevent. Attributing prefill between the scorer and the
   attention it gates is §8.7 and has not been done.

## 7. The operating point

`tools/serve.sh`'s `qwen4exp` arm, `[QSA-2026-09-06]`. Four terms moved and none of them is a
preference:

* **`GRAPH_BS=0`** — still 0, but the reason on file has EXPIRED. `QSARuntime.prepare` no longer
  raises inside a capture: plan T5.1 is implemented (`init_capture` / `_fill_decode_plan` /
  `prepare_for_replay`, a fixed `max_blocks = ceil(max_seq_len/r)` so every downstream grid is a
  build-time constant), and a captured QSA decode is CORRECT — 37/37 replays with 0 eager forwards,
  both needles recovered, sparsity 0.746899 matching the eager leg to six decimals. It is now a
  measured TRADE, and it loses. See §7a.
* **`--page-size 16`** — a hard requirement, not a default. `physical_kv_slot // r` is the whole
  compressed-key addressing scheme, and it is legal only when a group of 4 cannot straddle a page.
  `QSAProfile.require_page_size` raises rather than falling back.
* **`--max-extend-tokens 1024`** (was 2048) — a FEASIBILITY term, not a latency knob. The selection's
  stage-4b workspace is full CHUNK width, so the prefill activation peak scales with the chunk; at
  the old 8.1/2048 point the FIRST 4k prefill OOMs (§8.1). It goes back to 2048 the moment 4b is
  row-tiled.
* **`CONC=2`** — RESTORED (was clamped to 1 while QSA was live). §4c: the group-alignment refusal
  was chunk PACKING, not prefix reuse, and `PrefillAdder.chunk_gran` closes it at the source.
  Measured green at both recorded leftovers (ctx=4087 and ctx=16382), 2/2 requests completing on
  each, needles recovered, 0 failures.
* **`--weight-offload-device-gb 7.4`** (was 8.1, so 11 device MoE layers instead of 12) — a REACH
  decision. It buys 60,144 tokens of context (§5) and costs one layer's worth of PCIe streaming.
  `WOFF_DEVICE_GB=8.1` is left as an override, and 8.1 WITH the 1024 chunk is explicitly UNTESTED —
  it may well work and would trade those 60,144 tokens back.

The launch line this produces, which is the thing to diff before believing any of the above:

```
TP=2 CONC=2 MODEL=qwen4exp tools/serve.sh
  -> --page-size 16 --cuda-graph-max-bs 0 --max-running-requests 2 --memory-ratio 0.90
     --max-prefill-length 1024 --weight-offload-device-gb 7.4 --weight-offload-gb 26
     (MINISGL_WEIGHT_ARENA_CHUNK_MIB=1372, MINISGL_WEIGHT_ARENA_FLOOR_GIB=9)
```

### 7a. Capture: IMPLEMENTED, MEASURED, and REFUTED as a lever for this arm

Two legs of the same shape, same boot procedure, 48 layers TP=2, cards 0 (RX 9070 XT) + 1
(RX 9070), sampled temp 1.0 / top_k 20 / top_p 0.95. **Both legs took `decode_ms_per_step_median`
with the DEVICE-SYNCED bracket (`MINISGL_STEP_LOG_SYNC=1`)** — without it a captured step logs its
`hipGraphLaunch` return (0.54 ms against the eager leg's 60.5) and the comparison is 112x of pure
instrument. The harness GATES on the bracket for any `--graph-bs > 0` leg.

| ctx (tokens) | eager ms/step (r0 / r1) | captured ms/step (r0 / r1) | recovered |
|---|---|---|---|
| 4,087 | 60.52 / 60.55 | **57.07 / 57.06** | 3.45 ms, **-5.7%** |
| 16,382 | 60.33 / 60.34 | **56.67 / 56.70** | 3.66 ms, **-6.1%** |
| 32,761 | 60.12 / 60.16 | — (see the fault below) | — |
| 65,530 | 82.75 / 82.87 | — (see the fault below) | — |

Raw: `QSA_2026-09-06/CAPTURE_CONC/{eagS,capC,cap16kmr}*.json`. The 16,382 captured row was taken at
`--memory-ratio 0.84` and is therefore an INDICATIVE comparison, not an iso-config one — see the
fault. Its sparsity ratio (0.234462) matches the eager leg's (0.234062), so it is the same work.

**Read against the 27.05 ms short-context eager reference, capture recovers about a TENTH of what
QSA costs**: QSA takes the step from 27.05 to 60.52 ms (+33.5 ms) and capture gives back 3.45. This
is the SAME 2.80-3.78 ms/step capture was worth on this model BEFORE QSA existed
(`QWEN4EXP_L48_E4M3_HCFUSE_2026-09-05.json`), which is the honest reading: a decode where 37 of 48
layers read their experts over PCIe has little launch overhead to remove, and QSA neither helps nor
hurts that. **This is a wash-adjacent 6%, not a throughput lever.**

**AND AT THE SHIPPED MEMORY RATIO IT FAULTS.** Capture allocates its graphs out of the same headroom
the KV pool is sized from. At `mem_default=0.90` that leaves **0.42 GiB** free after capture — not
enough for the selection's stage-4b prefill workspace, which is full CHUNK width (§8.1). The failure
is NOT a clean OOM:

```
Memory access fault by GPU node-1 ... Reason: Page not present or supervisor privilege.
:0:rocdevice.cpp :3586: Callback: Queue ... aborting with error :
  HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory beyond the
  largest legal address. code: 0x29
[parent] rank exit codes [-6, -6]
```

about **one second into the FIRST prefill chunk** of a 16,382-token request. What is established
about it, and what is not:

* **Deterministic.** Three independent 48-layer TP=2 boots (`capB`, `capC`, `cap16k`), plus a fourth
  under `AMD_SERIALIZE_KERNEL=3` (`cap16kdbg`) which did not name the faulting dispatch.
* **Length-gated, not rung-order-gated.** `ctx=4087` captured completes clean. With `--lens 16384`
  as the ONLY rung it faults on the first rung, so it is not a request-teardown or table-slot-reuse
  effect.
* **Not reproducible at 4 layers TP=1** (`rep4l`), which ran 4,087 and 16,382 captured back to back.
  So it needs the 48-layer footprint — the weight-offload arena, 12 index layers, or TP=2 — and
  cannot be bisected on the cheap subset.
* **It is HEADROOM.** At `--memory-ratio 0.84` (1.41 GiB free after capture) the captured 16,382-
  token run PASSES clean, 0 failures on both ranks. But the KV pool falls **104,912 -> 22,592
  tokens**.
* **NOT root-caused:** why an exhausted-headroom prefill produces an aperture violation instead of a
  torch OOM is unexplained. That is a real defect independent of capture — an allocation failure
  should raise, not fault — and it is §8.7 below.

So the trade is: **78% of the arm's context reach for 6% of its decode step.** Reach is the entire
point of QSA, so `GRAPH_BS=0` stands. The thing that would change the answer is row-tiling stage 4b
(§8.1): it bounds the prefill peak, and a bounded peak is what would let capture and a 0.90 pool
coexist. `GRAPH_BS=2` is left as an override for a short-context serve — measured good to ctx=4087
at 0.90, and to ctx=16382 at 0.84 — and it no longer requires `MINISGL_QSA=0`.

## 8. What remains

0. ~~**BLOCKING: prefix reuse breaks the group alignment.**~~ **DONE 2026-09-06, and the diagnosis
   in this line was WRONG** — it is chunk PACKING, not prefix reuse, and the fix it prescribed
   (rounding the prefix match down) would have been a silent no-op because `qwen4_exp` forces the
   naive prefix cache and `cached_len` is always 0. Fixed in `PrefillAdder.chunk_gran`; CONC=2 is
   restored and measured green at both recorded leftovers. See §4c.

1. **The selection is not row-tiled, and that is what bounds the operating point.**
   `QSARuntime.select` row-tiles stages 3-4a (the `[rows, blocks]` fp32 logits, under a 128 MiB
   budget) but stage 4b runs at FULL chunk width: `sel_tokens` `[chunk, 2051]` int32, `flat`
   `[chunk, 2051]` **int64**, and `sel_slots` `[chunk, 2051]` int32 — ~67 MB of transient per index
   layer at a 2048-token chunk, of which `flat` alone is 33.6 MB purely because `index_select`
   demands int64 indices. Tiling 4b at `_ATTN_ROW_TILE` (already the tile the sparse attention uses,
   and numerically a no-op because the mapping is per row) would cut that by 8x. MEASURED
   motivation: at `--weight-offload-device-gb 8.1` / chunk 2048 the FIRST 4096-token prefill OOMs
   asking for 92 MB with 138 MB free (`docs/measurements/QSA_2026-09-06/r1.log`).

2. ~~**cudagraph capture of the SELECTION (plan T5.1).**~~ **IMPLEMENTED 2026-09-06 — and then
   MEASURED AND REFUTED as a lever for this arm.** The static-buffer treatment landed exactly as
   described (fixed `max_blocks = ceil(max_seq_len/r)`, per-bs preallocated
   logits/blocks/tokens/slots, `prepare_for_replay` refreshing contents in place), the capture is
   bit-identical at 4 layers (`eps_capture = 0.0`, `bit_equal: true`, `topk_num_splits` 16 on both
   legs) and correct at 48 layers TP=2. It buys 3.45-3.66 ms of a ~60 ms step (-5.7 to -6.1%) and
   costs 78% of the context reach at the memory ratio where it is stable. `GRAPH_BS=0` stands. Full
   numbers, the aperture-violation fault, and what is NOT root-caused: §7a.

3. ~~**The `qsa_index` ops are NOT in the `engaged()` ledger.**~~ **DONE 2026-09-06.** All three
   arms are marked, and both the HIP arm and its `(torch)` fallback are marked separately so a
   silent fallback is visible rather than merely absent. `_hip_engage` also grew per-arm COUNTS
   (`counts()` / `counts_delta()`) because the ledger as a SET saturates and cannot answer an A/B at
   all. The counter diff of the eager vs captured 48-layer legs is the worked example: all seven
   PREFILL arms identical count-for-count (prefill is eager on both legs), and every decode arm's
   delta exactly `decode_steps x per-step` — e.g. `qsa_index.score_paged` 492 -> 48, and
   492 - 48 = 444 = 37 decode steps x 12 index layers. **Under graph REPLAY these counts do not
   move at all** (`engaged()` is host Python; a replay re-executes recorded launches), so frozen
   counts on a captured leg are the positive signal that it replayed, not evidence an arm died.

4. **fp8 KV is the untried 2x on reach.** `MINISGL_KV_FP8=1` halves the 12,288 B/token bill and would
   double the context at any device tier, but it is UNCALIBRATED on this checkpoint and this repo's
   own record (`fp8-kv-was-serving-uncalibrated`) is that an uncalibrated fp8 KV serves quietly
   wrong.

5. **The 262,144 GPU fault, bracketed but not root-caused.** `QSA_REACH.json`: ctx 196,608 runs (max
   compressed-block window 49,152); ctx 262,144 dies with `Memory access fault ... Page not present`.
   The boundary sits exactly at `max_blocks = 65536 = 2^16`, and nothing on the engine side changes
   shape across it (the fp32 logits workspace is 2^25 elements on BOTH sides and the score grid is
   131,072 CTAs on both). It does not bite at the shipped operating point — 105,104 tokens puts
   `max_blocks` at 26,276 — but it is NOT harmlessly out of range: §5's own arithmetic says an
   8-device-layer tier reaches 279,088 tokens, and a request past 262,144 there crosses exactly this
   boundary. **So it has to be root-caused BEFORE the device tier is lowered for reach**, not after.

6. **NEW, OPEN: an exhausted prefill headroom FAULTS instead of raising.** §7a: with 0.42 GiB free
   the 16,382-token prefill produces `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` and aborts both
   ranks (-6) rather than a torch OOM. Deterministic on 3 boots, length-gated, gone at 1.41 GiB
   free, not reproducible at 4 layers TP=1, and `AMD_SERIALIZE_KERNEL=3` did not name the dispatch.
   This is independent of capture — capture only makes the headroom small enough to reach it — and
   it matters beyond QSA, because it means a memory-pressure regression on this model presents as a
   GPU fault rather than an allocator error. Bisecting it needs a per-dispatch name
   (`AMD_LOG_LEVEL=3`, or a rocgdb attach), which was not attempted.

6. **The decode step's HOST half grows with context and is unattributed.** Forward-only is flat at
   ~61 ms from 4k to 32k, but wall-per-token triples over the same range (95 -> 295 ms). The gap is
   per-step work outside `STEP_LOG` — batch construction, `cache_manager.allocate_paged`,
   `attn_backend.prepare_metadata`, the radix-cache walk — none of which the sparse path made
   flat. On a serve whose forward is now context-independent this is the NEXT bottleneck, and it is
   a scheduler/metadata question, not an attention one.

7. **The indexer's own cost has never been separated from the attention's.** Every throughput number
   here is end-to-end. `qsa_index`'s three kernels run 12 times per forward and nothing has profiled
   them against the attention they gate.

## 9. Reproducing

```bash
# 1. Build BOTH kernel packages CLEAN, in the image that will run them. `--clean` is not optional:
#    an incremental build finishes in 4 s by relinking stale objects and ships a different binary.
docker run --rm -v /home/pat/code/rdna4-hip-kernels-qsa:/kern --entrypoint bash \
  minisgl-rdna4:m1b-20260903 -lc \
  'cd /kern/fp8_wmma && bash local/build_local.sh --clean && \
   cd /kern/qsa_index && bash local/build_local.sh --clean'

# 2. The <=budget dense-equivalence gate (one card, 4-layer subset, ~2 min)
gpu-lease -n 1 -- docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES -e HIP_VISIBLE_DEVICES \
  -v /home/pat/code/minisgl-rdna4-qsa:/engine -v /home/pat/code/rdna4-hip-kernels-qsa:/kern:ro \
  -v /home/pat/.cache/hf-q4e:/model:ro --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
  'PYTHONPATH=/engine/python:/kern/qsa_index/torch-ext:/opt/kernels \
   python /engine/tests/qwen4exp_qsa_gate_test.py --gate all --pin-split'

# 3. The long-context ladder (both cards, 48 layers, ~6 min boot + the ladder)
OUTDIR=/home/pat/code/minisgl-rdna4-qsa/out/qsa_longctx TAG=r2b RUN_TIMEOUT=9000 FLOOR_GIB=6 \
gpu-lease -n 2 -- tools/qsa/run_qsa_longctx.sh \
  --tp 2 --layers 48 --experts 512 --device-gb 7.4 --host-gb 26 --memory-ratio 0.90 \
  --max-extend-tokens 1024 --lens 4096,16384,32768,65536,131072 --gen-tokens 96 \
  --conc-len 16384 --conc-width 2 --json /out/r2b.json --rank-json /out/r2b

# 4. The tables in this document
python3 tools/qsa/qsa_longctx_report.py out/qsa_longctx/r2b.json
```

`run_qsa_longctx.sh` asserts BOTH fresh kernel packages resolve out of `/kern` and prints their
sha256 before the harness starts, because a missing `qsa_index` does not crash: `build_qsa_runtime`
catches the ImportError, logs a warning, and the full-attention layers run DENSE. The harness then
gates on `ctx.qsa is not None` for the same reason.
