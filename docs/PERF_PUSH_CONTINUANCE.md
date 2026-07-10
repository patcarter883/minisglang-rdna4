# minisgl RDNA4 perf push — continuance (spec-overlap / FutureMap + review findings)

**Mission:** push minisgl (gfx1201, RDNA4) serving perf to beat vllm on throughput. Two threads:
(1) land a batch of verified review findings (Tier 1–3), (2) build the **FutureMap-style overlap**
for the spec-decode loop (the deferred MVP tradeoff — the real concurrency-throughput lever).

**Standing rule (do NOT relax):** every overhead removal must stay **lossless**. Gate each change
with fuzz/numpy byte-equivalence AND a GPU coherence pass. We reverted #12 because GPU validation
caught an OOM the numpy check missed — that discipline is the whole point. Prefer "skipped, here's
why" over a silent correctness break.

---

## Repo / worktree layout
- `~/code/minisgl-rdna4` — **main checkout, branch `rdna4`** (default branch is `rdna4`, not main). All
  engine commits land here.
- `~/code/minisgl-rdna4-mxfp4` — **serve worktree** (mounted into the container as `/engine`). Kept in
  sync via `git reset --hard rdna4`. Serve port **1919**. Has an untracked `docker-compose.override.yml`
  mounting the rebuilt dtype-generic kernels (survives `reset --hard`).
- `~/code/minisgl-rdna4-specoverlap` — **`spec-overlap` branch worktree** for the FutureMap work
  (Phase 1 done, Phase 2a in flight). Isolated so prod stays stable.
- `~/code/rdna4-hip-kernels` — **canonical HIP kernels, branch `main`**. Attention + W4A8 made
  dtype-generic here.

## How to run
- **Prod serve (35B, TP=2):** `IMG=minisgl-rdna4:lean MTP=1 SPEC_K=4 GRAPH_BS=8 MEM=0.85 MAXREQ=6 bash
  <scratchpad>/launch_mxfp4_dflash.sh` (model `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`, bf16 native,
  graph-captured, `--gdn-radix` auto-off under spec). 35B needs TP=2 (>16 GB/card). The launch script
  supports `MTP=1` / `NOSPEC=1` / (default DFlash) / `DTYPE=float16` / `SPEC_K` / `GRAPH_BS` / `MEM` / `MAXREQ`.
- **One-card validation (model-agnostic scheduler/kvcache/graph changes):** Qwen3.5-4B is GDN-hybrid and
  fits one 16 GB card. Boot TP=1 + ngram spec to exercise the GDN verify-state path (#12) and everything
  scheduler-side, leaving the other GPU free:
  ```
  export LEASE_NAME=val MINISGL_IMAGE=minisgl-rdna4:lean MINISGL_ATTN_BACKEND=hip MINISGL_HOST_PORT=1920
  export MINISGL_MODEL=Qwen/Qwen3.5-4B MINISGL_TP=1 MINISGL_CUDA_GRAPH_MAX_BS=8 MINISGL_MEM_RATIO=0.80
  export MINISGL_EXTRA_ARGS="--spec-algorithm ngram --spec-num-draft 4 --max-running-requests 6 --reasoning-parser auto"
  gpu-lease -n 1 --detach --name val -- docker compose -p lease-val --profile serve up -d
  ```
- **GPU protocol:** always `gpu-lease -n 1 -- ...` (blocking queue IS the coordination; never poll/kill
  holders). `gpu-status` shows holders. **Gotcha:** inside a `docker run`, `-e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES`
  must expand in the **lease shell** → wrap the whole run in `bash -c '…'`, else torch sees no GPU (silent CPU fallback).
- **Kernel rebuild (CPU, no GPU):** `docker run --rm -v ~/code/rdna4-hip-kernels:/work -w /work/<pkg>
  --entrypoint bash minisgl-rdna4:lean -lc 'source /opt/venv/bin/activate && GPU_ARCHS=gfx1201 bash local/build_local.sh'`
  → produces `torch-ext/<pkg>/<pkg>_C*.so`. Mount over `/opt/kernels/<pkg>` to test (see override file).
- **Eager profiler (cast/launch attribution):** boot with `GRAPH_BS=0` + `MINISGL_PROFILE=/engine/tools/traces/x.pt.trace.json.gz
  MINISGL_PROFILE_SKIP=30 MINISGL_PROFILE_STEPS=60` (compose forwards these; hook is in engine.py `forward_batch`
  AND `forward_verify`). Parse `cpu_op` events (ROCm kineto emits no device-kernel durations — use op/launch counts).

---

## DONE this session — committed + validated

### minisgl `rdna4` (in commit order; all GPU-validated unless noted)
- `af8f844` grammar: align padded bitmask to actual logits rows (multi-req constrained crash)
- `ed9848d` reasoning-token budget backstop (reason-then-JSON)
- `d3097d1` accept OpenAI array-of-parts message content (was 422)
- `5401853` bf16 `fused_topk` falls back to vllm/pure-torch when sgl_kernel absent (MTP-head MoE crash)
- `258722d` remove `MINISGL_AUX_POSTMLP` footgun — always fold MLP into aux capture
- `e053dd6` **review #1**: vectorize verify page-table fill (both variants) + **fuse MoE shared/routed all-reduce** (−40 collectives/step)
- `ed2abc3` **review #6**: `--gdn-radix` launch flag (default on), remove `MINISGL_GDN_RADIX` env gate
- `e248780` **review #2/#3/#5**: cold-prefill per-layer `.tolist()` memoized once/forward; metadata builders (decode fast-path + `repeat_interleave`, keeps C-speed prefill — a per-token Python list REGRESSES 60k ctx, don't); batched tokenizer encode. (#4/#7 assessed.)
- `df3be1d` **W4A8 native-dtype wiring**: pass bf16 activations straight to the fp8 GEMM (drop `x.to(fp16)`); **casts −49% (449→229 _to_copy/step)**, coherent, tok/s-neutral.
- `393b745` **Tier-1 #1–4**: graph-coverage cliff closed (`cuda_graph_max_bs` derived from `max_running_req`; `_graph_capture_bytes` reproduces the bs-list); `running_reqs`→uid-keyed dict + lazy-sorted view; `filter_reqs` incremental; **`Req.input_ids`→preallocated O(1) host buffer** (was O(seq²) torch.cat). GPU-validated: 800-tok gen START+END coherent, 6-way concurrent, reason-then-JSON=391, multi-turn recall.
- `f0e8b90` **API/IPC #6–8**: non-streaming chat list+join (O(n²)→O(n)); streaming disconnect probe every 0.5s (was per-token); per-token IPC (DetokenizeMsg/UserReply/StatsMsg) whitelisted fast-lane serialize, **wire byte-identical** to recursive path.
- `eff0eef` **#9/#13**: finer decode graph buckets `[1,2,4,8,12,16,24,32]`+step16 (fewer graphs AND less padding; covers `max` exactly); hoist `from minisgl.quant import kernels` to module scope (proven cycle-free).
- `6cbdcbe` **REVERT #12** — see "Reverted/open" below.
- `01808d5` **#5** persistent radix evict leaf min-heap (incremental at 4 structural transitions, lazy stale-skip; fuzz 200×4000 byte-identical evicted set). *(Note: this commit ALSO originally carried #12; #12 was reverted by 6cbdcbe, #5 stands.)* GPU-validated under 6-way concurrent load.

### rdna4-hip-kernels `main`
- `bd010fb` attention kernels **dtype-generic fp16+bf16** (`attn_decode`, `attn_hip`, `attn_prefill_paged`); rocwmma f16/bf16 fragments both work on gfx1201; bf16 path bit-unchanged.
- `aaad74d` + `7016733` **W4A8 fp8 GEMM/MoE dtype-generic activations** (fp16+bf16, output follows input); `7016733` fixed the missed `gather_reduce` `out2` fp16 check.

### Spec-overlap branch (committed `d34508c`, isolated worktree)
- **Phase 1** `spec/accept_gpu.py::accept_greedy_ondevice` — on-device greedy acceptance (segmented
  mismatch-cumsum → leading-run), `(num_accepted, committed_flat, committed_offsets, committed_lens)`,
  zero host syncs. **Byte-lossless 4008 cases** vs host `verify_greedy`.
- **Phase 2a** `spec/accept_gpu.py::truncate_at_eos_ondevice` — per-req EOS truncation respecting
  `ignore_eos` → `(kept_lens, kept_finished_eos, kept ids)`. **Byte-lossless 6011 cases** vs the host
  `keep`/EOS loop. Shared `_leading_zero_run_per_segment` helper used by both.
- **Phase 2b** `spec/accept_gpu.py::build_commit_ondevice` (`d34508c`) — on-device commit layout:
  token-pool scatter `(scatter_rows, scatter_cols, scatter_vals)` in the SAME req-then-token order
  the host `c_rows/c_cols/c_vals` loop made + advanced `new_cached_len`/`new_device_len` + GDN install
  `gdn_t_index (=kept_lens-1)` + draft-head `seed_rows (=target_offset+kept_lens-1)`, all from GPU
  `kept_lens`, zero host syncs. **Byte-lossless 5007 cases** (chains accept→truncate→commit) vs the
  host loop. Also fixed a latent Phase-1 crash: `_leading_zero_run_per_segment` indexed an empty
  `excl` when the flat buffer is length 0 (all-K=0 / ngram all-miss batch) → guard `total==0`.
  Validators: `tools/validate_ondevice_accept.py`, `tools/validate_eos_trunc.py`,
  `tools/validate_commit_layout.py`.

---

## KEY MEASURED FINDINGS (so we don't re-derive)
- **fp16 vs bf16 is a trap for THIS model.** Qwen3.6-35B-A3B-MXFP4 is **bf16-native** (store_kv is bf16-
  gated at `mha_pool.py:100`; GDN state bf16 across 30 layers; attention was bf16-only). Running fp16
  end-to-end ADDS casts (+292 `_to_copy`/step) because most ops are bf16-native — only the GEMM was fp16.
  **The fix was the opposite of "make everything fp16": make the W4A8 GEMM accept bf16** (done), so the
  bf16 model runs cast-free. Attention kernels were made dtype-generic anyway (correct architecture, and
  fp16 IS coherent — no overflow on the 35B — so fp16-native models can use them).
- **Casts are cheap under graph capture** (they're captured HBM traffic, launches already gone), so the
  −49% cast cut is a correctness/architecture win, **tok/s-neutral at bs=1**. The structural fixes
  (allreduce fusion, verify vectorize, graph-cliff) similarly scale with concurrency, not bs=1 latency.
- **Spec-decode on 35B (GDN-hybrid + 128-expert MoE) is break-even at bs=1.** DFlash K=15 accept-len 4.67
  (reference band, drafter faithfully ported) but 25 tok/s (half of no-spec 49) because verify of 16
  positions scatters ~1 token/expert across the fine-grained MoE → WMMA-tile-starved + all-expert weight
  load. K=4 → 49 (cheap verify, low accept). MTP K=4 ≈ 50 ≈ no-spec. **The overlap's win is concurrency
  throughput, NOT bs=1** (verify compute ~28-30ms is the irreducible floor). See memory
  `spec-decode-breakeven-gdn-moe`.

---

## IN FLIGHT (as of session end)
- **Nothing running.** Phases 1, 2a, 2b done, validated, committed to `spec-overlap` (`d34508c`).
  Next is **Phase 2c** (next-batch layout from GPU lengths — the crux) — start there. NB: all three
  primitives so far are STANDALONE (validated in isolation); NONE is wired into `scheduler.py` yet.
  The wiring + GPU coherence pass happens once 2c gives a layout the scheduler can consume without a
  count sync (2b's scatter/lengths land at the same wiring point).

---

## FutureMap overlap roadmap (Phases; each gated byte-lossless)
The serializing sync is `preds = logits.argmax(dim=-1).cpu()` at **scheduler.py:1877** — the host needs
the acceptance count to lay out the next batch, blocking CPU-run-ahead. Target = the accept+commit loop
at **scheduler.py:1897-~2050**. Constrained-decode reqs (`_verify_greedy_constrained`, xgrammar matcher
is host-side) **stay on the sync path** — overlap targets unconstrained greedy spec-decode.
- **1 ✅** on-device accept (`accept_greedy_ondevice`) — done, lossless (4008).
- **2a ✅** on-device EOS truncation (`truncate_at_eos_ondevice`) → `kept_lens` + `kept_finished_eos` per
  req on GPU, respects per-req `ignore_eos`. Done, lossless (6011). Committed `f94cdc0`.
- **2b ✅** on-device **commit** (`build_commit_ondevice`, `d34508c`): scatter committed ids into the
  GPU token-pool + advance `cached_len`/`device_len` from GPU lengths (replaces host `c_rows/cols/vals`
  + `append_host`). GDN install `t_index = kept_lens-1` and draft-head seed row `= target_offset +
  kept_lens-1` from GPU lengths. Done, lossless (5007). Still a standalone primitive — not yet wired.
- **2c** **next-batch layout from GPU lengths** (the crux): build next positions/page_table without
  syncing counts, so the CPU schedules step N+1 while N's copy is in flight.
- **2d** **lazy detok (FutureMap)**: committed ids copied host-async (a per-req "future"); detok resolves
  lazily; scheduler never blocks. This is where the throughput win actually lands.
- Validate each: byte-lossless vs the sync path (GRAPH==EAGER, accept/emit identical) + coherence + a
  **concurrency** throughput measurement (the payoff is multi-request, not bs=1).

---

## Reverted / open / queued
- **#12 GDN/CCA `install_verify_state` batching — REVERTED (`6cbdcbe`).** The `torch.stack` over ~30
  per-layer scratch tensors made an ~864 MiB transient that OOMs 16 GB (worse on 35B TP=2). **Redo needs
  a PRE-STACKED scratch buffer allocated ONCE in `gdn/metadata.py`** (`[L, ...]`, reused — not stacked per
  step); then the gather/scatter batch for free with no transient. Numpy byte-identical is necessary but
  NOT sufficient — must GPU-validate memory too.
- **#11** constrained-bitmask `[bs,vocab]` unpack every step (`engine/grammar.py:72-76`) — wants a fused
  HIP mask kernel (not yet built).
- **Bake dtype-generic kernels into `:lean` image** — currently mounted via `docker-compose.override.yml`.
  Rebuild the image so the cast-free path is default and the override retires.
- **#10** per-admit `pin_memory` — deliberately SKIPPED (shared buffer corrupts under no-sync admits;
  caching host allocator already pools). Do not "fix" without a ring+event scheme; not worth it.

## Gotchas (bit us this session)
- `HIP_VISIBLE_DEVICES` expansion (wrap docker run in `bash -c`) — else silent CPU fallback.
- 35B DFlash K=15 is memory-razor-thin at TP=2/16GB (GRAPH_BS=4 MEM=0.80 boots; MEM=0.74 starves KV;
  MEM=0.82/GRAPH_BS=8 runtime-OOMs GDN verify). MTP K=4 GRAPH_BS=8 MEM=0.85 is the stable prod config.
- W8A8 path (`w8a8_fp8_wmma`, kernels.py:382/402) is a DIFFERENT kernel, still fp16-only — left its cast.
- The suggested "batch the TiDAR .item()/.tolist() at scheduler.py:1722-1753" is a red herring — that
  range is `_tidar_dump`, a diagnostic (debug-flag) function, not the hot path.

## Resume checklist
1. Phases 1+2a+2b are committed to `spec-overlap` (`d34508c`), byte-lossless. **Start Phase 2c**
   (next-batch layout from GPU lengths — positions/page_table built without syncing counts, so the CPU
   schedules step N+1 while N's copy is in flight). This is the crux that actually unblocks run-ahead.
2. After 2c, WIRE the standalone primitives (1/2a/2b/2c) into `scheduler._spec_decode_step` behind an
   env gate (unconstrained greedy path only) and run the FIRST GPU coherence + concurrency-throughput
   pass — everything so far is validated in isolation, never end-to-end in the engine.
3. Keep constrained reqs (`_verify_greedy_constrained`) on the host sync path; measure the overlap win
   at **concurrency**, not bs=1 (verify compute ~28-30ms is the bs=1 floor — see KEY FINDINGS).
4. Boot a **one-card** validation (recipe above) before trusting anything unvalidated in prod.
5. When ready: redo #12 with a pre-stacked scratch buffer in gdn/metadata.py; bake dtype-generic kernels
   into the `:lean` image (retire the override).
