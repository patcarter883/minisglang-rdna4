# ZAYA + Markovian RSA — performance continuance (2026-07-11)

**Goal:** make ZAYA+RSA usable at ~10 concurrent RSA requests, some at 50k context, ~30 tok/s/request.
RSA is a ZAYA-only feature (tuned prompts). Served **DP=2 + EP**, image `minisgl-rdna4:lean-zaya`,
model `/models/ZAYA1-8B-fp8`, compose `zaya` profile. Triggered per-request via the `rsa` field.

## Hardware (corrected)
- GPU0 = RX 9070 XT, **board power cap 375 W**. GPU1 = RX 9070, **265 W**. (rocm-smi "Graphics
  Package Power" reads a 250 W *chip* cap — the board caps are 375/265.) Both 16 GB gfx1201.

## Measured findings (instrumented: MINISGL_BATCH_DEBUG=1 in scheduler.py `_flush_stats`, logs
`[batch] running=N waiting=W kv=used/total (%) kv_per_seq=` @0.2s; probes in scratchpad/rsa_*_probe.sh)

1. **Aggregation is NOT KV-crowded.** Single RSA req: KV peak 27% of a 114k-token pool. The "single-
   request-speed" aggregation was serial *selection* + GPU-idle inter-round CPU orchestration (tail
   tokenization + message assembly on the event loop) — NOT memory. See [[rsa-aggregation-bottleneck-not-kv]].
2. **At concurrency (12 short reqs) the GPU is SATURATED, not idle.** run pinned at the cuda-graph-max-bs
   cap (16), util ~100%, but **power only ~120 W of 265-375 W caps** → memory-bandwidth-bound with
   2-3x POWER HEADROOM. The single-request serial sections are hidden by cross-request overlap. So the
   fix is NOT idle/orchestration at concurrency — it's raising arithmetic intensity (bigger batches).
3. **fp8 KV cache (MINISGL_KV_FP8=1) DOUBLES the pool: 114k → 228k tokens/replica** (~458k total DP=2).
   So KV capacity is NOT the wall for ~10×50k in *decode* (228k/50k ≈ 4-5/card × 2 ≈ 9-10 fit).
4. **The 50k wall is the PREFILL RAMP, not KV.** 10 distinct-50k reqs: `running=0-1, waiting=20`, KV
   only 11-32% used, both cards 100% util but **~55-88 W** (compute-underutilized). Prefill admission
   is budget-capped: `prefill_budget = max_extend_tokens = 8192` (scheduler.py:143; PrefillAdder
   greedily fills up to the budget, prefill.py:126). A 50k req's first 8192-chunk eats the whole
   budget → one chunk/step → slow ramp (2/10 decoding after 3 min).

## Improvement roadmap (by impact/effort)

### Cheap config wins (no code)
- **[DONE] fp8 KV** (`MINISGL_KV_FP8=1`) → 2x KV pool → 2x concurrent-50k capacity in decode.
- **[DONE] cuda-graph-max-bs 32** (was 16) → decode batches >16 captured, not eager.
- **Raise `--max-prefill-length`** (the prefill budget): uses the idle prefill compute (100% util but
  ~80 W). BUT bigger prefill activations OOM at mem-ratio 0.9 (32768 OOM'd GPU0: pool fills to 14.9 GB,
  no activation room). FIX: `--max-prefill-length 16384 --memory-ratio 0.85` (KV has headroom to give
  up — only 32% used at 50k). Test 16384 first, then 24576. Expect ~2x faster ramp.

### Medium (scheduler / RSA code)
- **Distribute the prefill budget across requests** (cap per-req chunk to budget/N) so multiple 50k
  prefills progress together and reach decode staggered — complements the bigger budget.
- **RSA orchestration off the event loop**: `_tails_for` tokenization + `build_aggregation_messages`
  run on the FastAPI event loop (core.py); thread-pool them so the loop isn't the coordination
  bottleneck at high concurrency and the GPU never waits on host work between rounds.
- **Chunked-prefill activation reuse / smaller peak** so a bigger budget doesn't OOM at high mem-ratio.

### Large — TP=2 CCA (the per-request-latency lever; user requested)
**Why:** DP=2 puts each 50k request on ONE card → its prefill serializes and uses one card's compute.
TP=2 puts BOTH cards on each request → ~2x faster prefill/decode per request + a single ~458k-token KV
pool for one request's context. **Tradeoff: ~half aggregate throughput vs DP=2 (two independent
streams), plus all_reduce overhead.** Best for the 50k-heavy tail; worst for many short reqs. Could be
offered as a serve MODE (DP=2 for throughput vs TP=2 for big-context latency).

**Why it's not already done:** `zaya.py` has NO `tp_size` handling — CCA is TP=1-only, which is why
ZAYA runs DP=2. The **MoE already shards via EP** (shared `MoELayer`, zaya.py:447), so **only the CCA
attention needs the TP work.**

**Design (head-parallel, analog of [[glm-tp2-mla-sharding]]):** ZAYA geometry hidden=2048, 80 layers,
16 experts, CCA o_proj 1024→2048 (nqo*hd=1024). Shard the CCA heads (`cca_num_q_heads`/`cca_num_k_heads`)
across the 2 ranks:
- `CCAConv.linear_q / linear_k / val_proj1 / val_proj2` (plain nn.Parameter today, zaya.py:113-116):
  COLUMN-shard → each rank loads its head-slice of the latents.
- Per-head **temperature** (`_temp_eff`) + conv weights: shard by head.
- CCA conv kernel + `AttentionLayer` (zaya.py:255): run on LOCAL head counts (pass nqo/tp, nkv/tp).
- `o_proj` (LinearOProj 1024→2048, zaya.py:265): ROW-parallel (input = local heads) + **all_reduce**.
- CCA state (`cca_state.py`): `conv [C, conv_width]` shards by head-channel; `prev [hidden]` replicated.
- Weight loader (`weight.py`, zaya `self_attn.qkv.*` / `o_proj.*`): head-sliced load per rank.
- Validate: TP=2 output coherent vs DP=2/TP=1; then bench 50k prefill ramp (expect ~2x) + tok/s.

## Gotchas
- fp8 KV pool + mem-ratio 0.9 fills the card (~14.9 GB) → no room for big prefill activations → lower
  mem-ratio when raising max-prefill-length.
- Serve flag is `--max-prefill-length` (dest `max_extend_tokens`), NOT `--max-extend-tokens`.
- ZAYA DFlash has an accept-len 0.26 fidelity bug ([[zaya-cca-dflash-port]]) — separate from RSA perf.

See [[rsa-aggregation-bottleneck-not-kv]] [[zaya-dp-ep-validated]] [[zaya-port-into-minisgl]]
[[rsa-in-engine]] [[glm-tp2-mla-sharding]].
