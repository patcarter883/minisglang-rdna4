# Continuation prompt — ZAYA + Markovian RSA performance

Paste this into a fresh session to resume.

---

We're optimizing **ZAYA + Markovian RSA** serving performance on the two gfx1201 cards (GPU0 RX 9070 XT
375 W board cap, GPU1 RX 9070 265 W; both 16 GB). RSA is a **ZAYA-only** feature. Read
**`docs/CONTINUANCE_zaya_rsa_perf.md`** first — it has all measured findings, the improvement roadmap,
and the TP=2 CCA design. Also recall memories `rsa-aggregation-bottleneck-not-kv`,
`zaya-dp-ep-validated`, `glm-tp2-mla-sharding`, `zaya-cca-dflash-port`.

**Target:** ~10 concurrent RSA requests, some at 50k context, ~30 tok/s/request.

**What's established (do NOT re-derive):**
- Served **DP=2 + EP**, image `minisgl-rdna4:lean-zaya`, model `/models/ZAYA1-8B-fp8`, compose `zaya`
  profile / hand-rolled `python -m minisgl --data-parallel-size 2 --enable-ep`. RSA per-request via the
  `rsa` field.
- At concurrency the GPUs are **memory-bandwidth-bound with 2-3x POWER HEADROOM** (~120 W of 265-375 W).
  Not idle, not KV-bound. Every win = use the idle compute (bigger prefill/decode batches, or TP=2).
- **fp8 KV (`MINISGL_KV_FP8=1`) doubles the pool** (114k→228k tok/replica). KV is NOT the 50k wall.
- **The 50k wall is the PREFILL RAMP** — prefill_budget = `max_extend_tokens` = 8192 (CLI flag is
  `--max-prefill-length`, NOT `--max-extend-tokens`); a 50k req's first 8192-chunk eats the whole budget
  → 1 chunk/step → slow (2/10 decoding after 3 min).
- Instrumentation IN PLACE: `MINISGL_BATCH_DEBUG=1` in `scheduler.py _flush_stats` logs
  `[batch] running/waiting/kv/kv_per_seq` @0.2s. Probes: `scratchpad/rsa_50k_probe.sh`,
  `rsa_batch_probe.sh`; host power sampler pattern using `rocm-smi --showpower --showuse`.
- Memory-planner bug FIXED in `engine.py` (duplicate `_draft_model_bytes`/`_graph_capture_bytes`;
  restored the dropped 0.7 GB draft working-set margin) — uncommitted, verify + commit.

**In flight when this was written:** a 50k-context probe at `--max-prefill-length 16384
--memory-ratio 0.85` (16384 to avoid the 32768-OOM on GPU0). Check its result first:
`tools/kperf_results/ab_capture/rsa_50k_16k.driver.log` — did the ramp speed up (how fast running climbs
to ~9-10), and per-request tok/s vs the 30 target. If 16384 still OOMs, drop mem-ratio further or try
12288.

**Next actions (priority order):**
1. Confirm the `16384/0.85` prefill-ramp win (above). If good, bake into the zaya serve defaults.
2. **Build TP=2 CCA** (the main ask). `zaya.py` has no `tp_size` → CCA is TP=1-only (forces DP=2); MoE
   already EP-shards, so only CCA needs work. Head-parallel design (GLM-MLA analog) in the CONTINUANCE
   doc §"Large — TP=2 CCA": column-shard `CCAConv.linear_q/k`+`val_proj1/2`+per-head temperature, local
   head counts to the CCA conv kernel + `AttentionLayer`, ROW-parallel `o_proj` + all_reduce, shard the
   CCA conv-state by head (`cca_state.py`), head-sliced weight load (`weight.py`, `self_attn.qkv.*` /
   `o_proj.*`). Validate coherence vs DP=2, then bench 50k prefill ramp (~2x) + tok/s. Consider a serve
   MODE flag (DP=2 throughput vs TP=2 big-context latency).
3. RSA orchestration off the event loop (`core.py` `_tails_for` + `build_aggregation_messages` →
   thread pool) so the host loop isn't the coordination bottleneck at high concurrency.

**Protocol reminders:** always `gpu-lease -n 2 -- ...`; ZAYA needs both cards. Verify `docker -e` env
forwarding actually lands (silent `sed` failures wasted windows). Kill+free cards after each run
(`docker kill` + `gpu-status`). Don't thrash the shared cards — measure, then build.
