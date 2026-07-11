# Spec-decode PROPOSE-path graph capture — design + validation plan

**Rule:** eager is unfinished work; a spec path only counts as complete when it runs under CUDA graph
(`graph-capture-required-for-complete`). VERIFY is fully captured; PROPOSE is the remaining eager hole.
This doc is the actionable plan so the GPU window is pure validation, not greenfield writing.

## State of play (as of 2026-07-10)

- VERIFY is captured for every backend: `capture_verify_graphs`/`replay_verify` (`engine/graph.py:309,436`),
  `capture_fused_verify_graphs` (TiDAR), `capture_ddtree_verify_graphs`, `can_use_verify_graph:413`.
  Recurrent state threaded through static buffers: `GDNVerifyGraphCapture` (`gdn/graph_capture.py:53`),
  `CCAVerifyGraphCapture`.
- PROPOSE is EAGER for **all four** model-based proposers. No `capture_propose`/`replay_propose` exists.

| Proposer | propose runs a forward? | eager/captured | where |
|---|---|---|---|
| ngram | no (torch `unfold` match) | N/A | `spec/proposer.py:15-72` |
| **MTP** | yes — per-req K-step `head.step` chain | **EAGER** | `spec/mtp.py:72,101-108`; head `models/qwen3_5.py:559` |
| DFlash | yes — per-req `draft.denoise` block | EAGER | `spec/dflash.py:253,319` |
| EAGLE3 | yes — per-req K-step `draft.step` | EAGER | `spec/draft_model.py:179,213` |
| TiDAR | yes — full-target `forward_verify` (verify graphs off in CCA cfg) | EAGER | `spec/tidar.py:83`→`scheduler.py:1400`; note `:1348` |
| DDTree | no (heap/tree over another drafter's marginals) | N/A | `spec/ddtree.py:58` |

MTP is the right FIRST target: it's the one we're actively debugging, its eager cost (~8.8 ms/step, a full
MoE decoder layer run K× at TP=2, two all-reduces each) is why a low accept-len is catastrophic, and its
draft-KV rewrite is the exact analog of the already-shipped `GDNVerifyGraphCapture`.

## Why MTP propose is uncapturable today — the one hard piece

`Qwen3_5MTPAttn.forward_draft` (`models/qwen3_5.py:489-516`):
```python
cache.append((k, v))                          # per-uid Python list, mutated
Ks = torch.stack([c[0] for c in cache], 0)    # [S,T,nkv,hd], S = len(cache) is DYNAMIC + grows/step
scores = einsum("thd,sthd->ths", q, Ks) * scale
```
Dynamic contraction dim `S`, Python-list mutation, and a per-req Python loop (`mtp.py:72`) are all
uncapturable. Everything else in the head (`fuse`/`embed`/`step`, the MoE MLP, RMSNorms, tied `lm_head`)
is shape-generic on a leading `T` dim and batches cleanly with `T = bs`.

## The rewrite (the crux — needs on-GPU iteration)

Re-express the persistent per-uid draft KV as a **fixed-shape device buffer + per-req cursor + masked
attention**, mirroring `GDNVerifyGraphCapture`'s in-place recurrent-state contract
(`gdn/graph_capture.py:75-100`):

- Persistent `draft_kv_k`,`draft_kv_v` `[max_bs, max_ctx, nkv, hd]`, written in place at row = `cursor[b]`.
- `cursor` `int32[max_bs]` = each req's draft-context length (NULL slot 0 for padding, like
  `GDNGraphCapture._state_indices` `gdn/graph_capture.py:27,41`).
- `forward_draft` becomes fixed-width masked attention: additive `-inf` mask for columns `>= cursor[b]`
  (same mask-bias idea as `ddtree_paged_layout` `ddtree.py:133-141`) instead of `stack` over a list.
- `on_accept` (`mtp.py:145`, already patched for the full-accept cap) becomes a **cursor rewind** instead
  of `del cache[committed:]`; absolute RoPE positions stay via `positions_all` (`mtp.py:91`).
- `seed_prefill` (`mtp.py:117`) pre-seeds P entries → a captured path must accept `cursor > 0` at entry
  (natural with the cursor buffer).

**De-risk independently:** first make `forward_draft` use the buffer+cursor+mask path in EAGER mode and
prove it is **byte-exact** vs the current `stack` version (a plain forward diff, no graph needed). Only
then wrap the K-step chain in a graph. This splits the correctness risk (masked attn) from the capture
risk (aliasing/pool).

## The capture (mechanical — clone of verify)

One graph per `bs`, unrolling all K iterations of `fuse→step→argmax→embed` (the argmax→embed feedback is
already in-graph/on-device, `mtp.py:107-108`), writing draft KV in place at `cursor+step`. Direct clone of
`capture_verify_graphs` staging `bs*(K+1)` tokens (`graph.py:321,390`).

**`MTPProposeCaptureBuffer`** (clone `VerifyCaptureBuffer` `graph.py:359`), static buffers:
- `seed_hidden [max_bs, hidden]` — step-0 `prev_hidden`, refreshed from `ctx.last_hidden` per replay.
- `cur_tok [max_bs] int64` — step-0 confirmed token; overwritten in-graph by each step's argmax.
- `positions [max_bs, K] int32` — `arange(base_pos, base_pos+K)` + `pos_shift` (`mtp.py:61,91`).
- `draft_kv_k/v [max_bs, max_ctx, nkv, hd]` + `cursor [max_bs] int32` — written in place.
- `attn_mask_bias [max_bs, max_ctx] fp32` — causal+cursor mask, refreshed per replay.
- `out_draft_ids [max_bs, K] int64` — sliced host-side after replay (like `replay_verify` `graph.py:452`).

**Runner methods** (one-to-one with the verify pair):
- `capture_propose_graphs(model, num_draft, bs_list, hidden_size, dtype)` ← `capture_verify_graphs:309`.
- `prepare_propose_for_capture/replay` on the KV threader ← `GDNVerifyGraphCapture.*:102,109`.
- `can_use_propose_graph(batch)` ← `can_use_verify_graph:413`: only when every req drafts exactly
  `num_draft` (uniform K, `mtp.py:73`) and `bs` fits; partial-K / first-cold-step (`seed is None`,
  `mtp.py:75`) fall back to eager.
- `replay_propose(batch, ctx)` ← `replay_verify:436`: `copy_from` seeds/tokens/positions/cursor, refresh
  mask, `.replay()`, slice `out_draft_ids[:bs,:K]` to host.

Gate captured propose `and not enable_ep` — EP pins a fixed MoE N (`moe.py:674`), same constraint that
keeps verify eager under EP (`scheduler.py:2074-2081`).

## MoE safety — already OK

Under `MINISGL_MOE_SCATTER=0` (default) the head's MoE runs unfused gemm2 + `scatter_add_` gather_reduce
(graph-safe; the fused scatter's `atomicAdd` is not — `quant/kernels.py:31-35`, `moe/fused.py:142-144`).
`MoELayer.forward` non-EP is self-contained (`layers/moe.py:735`). Nothing to change; just don't capture
under EP.

## Ship gating

New captured path behind `MINISGL_SPEC_PROPOSE_GRAPH` (default OFF) + `can_use_propose_graph` returning
False until validated, so the eager path is the untouched default (same pattern as `MINISGL_SPEC_ONDEVICE`).
Flip default ON only after byte-exact losslessness is proven.

## GPU-window validation sequence (when the kperf matrix frees the cards)

1. **Diagnostics first (cheap, decides headroom):** 35B MTP TP=2, real prompt:
   - `MINISGL_SPEC_DEBUG=1` → mean accept-len + accept_rate (the number everything hinges on).
   - `MINISGL_SPEC_FORCE_N0=1` → isolate verify-forward correctness from accept/commit.
   - Validate the 3 staged CPU fixes (mtp full-accept cap, engine docstring, bench fail-fast) are green.
2. **forward_draft masked rewrite** → byte-exact vs eager `stack` version (eager mode, no graph).
3. **MTP propose capture** → wrap K-step chain; byte-exact vs eager; measure propose-overhead delta
   (expect the ~8.8 ms/step to collapse). Validate with `MINISGL_SPEC_PREFILL_SEED=1` too.
4. **Extend** to EAGLE3/DFlash (same per-req-list→cursor rewrite) and TiDAR (reuse `fused_verify` capture,
   turn it on for the CCA cfg + drop the snapshot/rollback).

## Already staged (CPU, unvalidated pending step 1)

- `spec/mtp.py` `on_accept`: cap `committed` at `len(cache)` — fixes the full-accept 1-key hole.
- `engine/engine.py` docstring: `last_hidden` is PRE-final-norm residual (not post-norm) — a future edit
  trusting the old text would double-norm the MTP seed and collapse acceptance.
- `tools/_bench_inner.sh` `launch()`: fail-fast on a worker-subprocess crash (was wedging the lease 20 min).
- `tools/kernel_perf_matrix.sh`: dropped the Qwen3.5-4B-AWQ rows (cached checkpoints are Qwen3.5-VL
  multimodal, unloadable by the text loader) + the ZAYA rows (weights absent offline).
