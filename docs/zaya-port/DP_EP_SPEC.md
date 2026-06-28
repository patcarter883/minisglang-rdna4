# DP + EP serving for minisglang (ZAYA) — implementation spec

Goal: native **data-parallel (DP)** launcher for minisglang (presents ONE OpenAI endpoint, internally
routes requests across `dp_size` full-model replicas — removes the need for a multi-backend RSA shim
router), plus **expert-parallel (EP)** as a toggle that shards the MoE experts across the DP ranks.
Test **with and without EP**. Target: ZAYA1-8B-fp8 on 2× gfx1201. CCA cannot TP, so the attention/CCA
backbone is always replicated (DP); EP only shards the MoE experts.

Architecture map (source of truth for seams): see the agent report in this session / re-read the files
it cites. Key files: `python/minisgl/server/{launch.py,args.py,api_server.py}`,
`python/minisgl/scheduler/io.py`, `python/minisgl/distributed/{impl.py,info.py}`,
`python/minisgl/engine/engine.py`, `python/minisgl/layers/moe.py`, `python/minisgl/models/weight.py`,
`python/minisgl/quant/kernels.py` (w8a8_moe).

## Current model (TP) — what we build on
- `launch.py:54-69` spawns one process per rank: `for i in range(world_size): mp.Process(_run_scheduler,
  replace(server_args, tp_info=DistributedInfo(i, world_size)))`. `spawn` start method.
- Ingress: front-end → tokenizer → **rank 0** reads, then PUB-broadcasts every msg to all ranks +
  broadcasts the COUNT over the gloo CPU group (`io.py:88-122`). So all ranks process the SAME stream
  (TP lockstep is emergent from this fan-out, not enforced in the loop).
- Replies only from rank 0, demuxed by `uid` in the front-end (`api_server.py:116-123`); `uid` is
  globally unique (single front-end counter) → multi-replica replies merge transparently.
- Collectives: `DistributedCommunicator` exposes ONLY `all_reduce` + `all_gather` (`impl.py:15-70`).
  On ROCm the active backend is `TorchDistributedImpl` (RCCL via torch.distributed); pynccl is OFF.
  **No `all_to_all`.** `MoELayer.forward` ends with `if tp_size>1: all_reduce` (`moe.py:404-405`).
- CCA recurrent state + KV are per-rank already (`engine.py:174-191`, `_determine_num_pages`).
- ZAYA is eager-only (recurrent/MLA graph capture out of scope); MoE collectives need not be
  graph-capturable.

## DP launcher (EP off) — design
1. **CLI**: add `--data-parallel-size` / `--dp-size` (default 1) in `args.py`. Total processes =
   `dp_size * tp_size`. For ZAYA tp_size=1, so dp_size replicas.
2. **Parallel coords**: extend `DistributedInfo` (or add a sibling `DpInfo(dp_rank, dp_size)` + global
   `_DP_INFO` like `_TP_INFO` in `info.py`). Each spawned scheduler gets `(dp_rank, tp_rank)`.
3. **Spawn**: `launch.py` loops `dp_rank × tp_rank`, tagging each process; each builds its own engine
   (full model replica) on its own card (the gpu-lease/HIP_VISIBLE_DEVICES assigns the device; for an
   in-process multi-card launch, set per-replica `CUDA_VISIBLE_DEVICES`/device index = dp_rank).
4. **Request routing (the core change)**: replace the rank-0 PUB-broadcast-to-all in `io.py` with
   **per-replica delivery** — each `UserMsg` goes to exactly ONE replica (round-robin or least-pending).
   Cleanest: the tokenizer/front-end round-robins `UserMsg` across `dp_size` backend ZMQ addresses
   (one per replica's rank 0). Within a replica (tp_size>1) keep the existing broadcast.
5. **Replies**: unchanged — each replica's rank 0 replies; front-end demuxes by uid. Track which
   replica owns each uid for aborts (`api_server.py:202-209`).
6. **EP off → replicas are independent**: NO per-step cross-replica collective. Each replica schedules
   + forwards on its own queue. Per-replica KV sizing unchanged. The world memory-imbalance guard
   (`engine.py:358`) must be scoped within a replica (relaxed across replicas).

## EP (toggle: `--enable-ep`) — design (all_gather + all_reduce, zero new collectives)
EP requires the DP ranks to be ONE communication group with per-step lockstep.
1. **Expert shard load** (`weight.py:_store_expert`, ~439-454): load only experts
   `[dp_rank*E/dp : (dp_rank+1)*E/dp]`; stack only the local count. `_GroupedFP8Experts` sized to the
   LOCAL expert count (`moe.py:185-188`). E=16, dp=2 → 8 experts/rank.
2. **MoE dispatch/combine** in `MoELayer.forward` fp8 branch (`moe.py:309-344`), EP path:
   - `all_gather` the token rows + their top-1 `topk_ids`/weights so every rank sees ALL tokens.
   - Remap global expert id → local: `local_id = gid - dp_rank*(E/dp)`; mask tokens whose expert is
     NOT local (set to a skip / zero-contribution).
   - Run `kernels.w8a8_moe` over the LOCAL expert tensors with the remapped local ids (kernel already
     handles an arbitrary expert subset — `E = w13.shape[0]`).
   - `all_reduce(SUM)` the per-rank partial outputs: each token's top-1 expert lives on exactly one
     rank, so the sum reconstructs the full result on every rank; each rank slices out ITS tokens.
   - This REPLACES the TP all-reduce branch (`moe.py:404-405`) when EP is on.
   - MOD skip-expert (`zaya.py:453-469`) operates per-token on the home rank — keep it on the
     pre-dispatch (home) side so masked rows never enter dispatch.
3. **Lockstep via a common per-step batch size (GRAPH-COMPATIBLE — replaces any host handshake).**
   EP must NOT use an in-hot-path "does anyone have work" host handshake or a conditional dummy forward
   — that is host control flow and breaks CUDA-graph capture. Instead:
   - The EP MoE collectives (all_gather + all_reduce) run **INSIDE the captured decode graph** with
     FIXED shapes (RCCL collectives are CUDA-graph-capturable when shapes are fixed).
   - For the collective shapes to MATCH across ranks, all EP replicas must replay the **same bs graph**
     each step. Before replay, agree a common bs = the smallest captured graph bs >= max real batch
     across replicas — one tiny `all_reduce(MAX)` of the per-replica batch size on the gloo CPU group.
     This agreement is the ONLY per-step host sync and it is **OUTSIDE** the captured graph (it just
     SELECTS which graph to replay), so capture is preserved.
   - Every replica then replays that same-bs graph. A replica with no/less work pads to the agreed bs
     with all-padding rows (the existing `is_pad`/slot-0 mechanism from CCA graph capture, `engine.py`
     dummy_req) — it participates in the in-graph all_gather/all_reduce, its padding outputs discarded.
     Lockstep is now IMPLICIT (everyone replays the same graph every step) → no deadlock, no host
     handshake, fully capturable.
   - This is also why the **all_gather+all_reduce** approach (fixed shapes) is mandatory over a variable
     `all_to_all` (data-dependent sizes → NOT graph-capturable).
4. **EP group = DP group**: `torch.distributed.new_group` over the dp ranks; the all_gather/all_reduce
   in MoE use that group. Everything else stays replica-local.

## Graph capture (REQUIRED — "not done until it's graph-capturable")
The production ZAYA path is `--attn hip` + CUDA graphs (graph decode ~1.8x eager). DP and EP MUST both
serve under graph capture, not just eager. Concretely:
- **DP-only**: each replica captures + replays its own decode graphs at its own bs — it's exactly the
  single-card capturable path per replica (CCA conv-state capture + W8A8 MoE via the capture-safe
  gather_reduce, NOT the atomic scatter). No cross-replica coupling. Graph capture is essentially free
  for DP-only; just confirm replicas capture and replay.
- **EP**: the all_gather + masked-local-expert w8a8_moe + all_reduce all live INSIDE the captured decode
  graph at fixed (agreed-bs) shapes; the per-step common-bs `all_reduce(MAX)` is the only host sync and
  is outside the graph. Use the gather_reduce MoE epilogue (capture-safe), keep `MINISGL_MOE_SCATTER=0`.
- A non-capturable EP (variable all_to_all, in-graph host handshake, or `.item()`/data-dependent shapes
  inside the forward) is a FAIL. Validation must run WITH graph capture on and confirm coherence +
  greedy-parity UNDER capture (eager is necessary but not sufficient).

## Validation (with AND without EP) — ALL under CUDA graph capture (graph is the gate)
- **DP-only coherent + throughput, GRAPH ON**: dp_size=2 on both cards (one endpoint), graph capture
  enabled (cuda_graph_max_bs>0), drive concurrent requests, confirm coherent output, that graphs
  captured + replayed (not silent eager fallback), and ~2× single-replica throughput.
- **DP+EP coherent, GRAPH ON**: same, `--enable-ep`, graph capture enabled; confirm graphs capture
  (incl. the in-graph EP collectives), coherent, and greedy-parity vs DP-only. Eager may be checked
  first for bring-up, but EP is NOT done until it serves under graph capture.
- **A/B (the headline)**: with vs without EP, report (a) KV pool size per card (EP frees ~4 GB of
  expert weights → bigger KV → more concurrency), (b) decode throughput, (c) top-1 expert-skew impact
  (16 experts / 2 ranks, top-1 → load imbalance risk). Use `tools/zaya_serving_matrix.py` per replica.
- **GPU tests MUST use timeouts** (a deadlocked lockstep test must time out and report FAIL, not hang).
  Lease 2 cards via gpu-lease `-n 2`. Keep iterations short.

## EP — as implemented (2026-06-28)
Seams landed (EP off => byte-for-byte the phase-1 DP path; gated on `--enable-ep` + dp_size>1):
- **Toggle / group**: `--enable-ep` already in `args.py`; `distributed/info.py` grows a process-global
  `is_ep_enabled()` (set by `Engine` alongside `set_dp_info`). `engine.py:_init_dp_communication`
  builds, in addition to the gloo DP subgroup (the lockstep channel), an **nccl/RCCL** DP subgroup and
  stores an `EPCommunicator` (group + dp coords + global E) on `ctx.ep`. The nccl group is mandatory:
  gloo is NOT CUDA-graph-capturable, so the in-graph all_gather/all_reduce must use RCCL.
- **Expert shard load**: `weight.py:_store_expert` loads ONLY experts `[dp_rank*E/dp : +E/dp]` and
  stacks them under LOCAL ids 0..E/dp-1; `MoELayer.__init__` sizes `_GroupedFP8Experts` to the local
  count (E=16, dp=2 -> 8/rank, ~4 GB freed per card). EP sharding is fp8-only (ZAYA).
- **Dispatch/combine** (`MoELayer.forward` fp8 branch, `EPCommunicator`): all_gather token rows +
  top-1 ids/weights over the EP group; remap `gid -> gid-offset`, **mask non-local tokens by ZEROING
  their route weight** (id clamped to a valid local 0 -> the gather-reduce contributes nothing — this
  is graph-safe, no out-of-range expert id ever reaches `moe_align`, which has no bounds check); run
  `w8a8_moe` over the local experts; `all_reduce(SUM)`; slice `[dp_rank*N : +real_n]`. Verified on CPU:
  the masked-sum reconstructs the full top-1 result with **0.0 error** on every rank. Replaces the TP
  all_reduce epilogue when EP is on.
- **Graph-capturable lockstep** (`scheduler/ep.py:SchedulerEPMixin.ep_loop`): a dedicated synchronous
  loop (like the spec loop). Each step does ONE gloo `all_reduce(MAX)` of `[prefill_tokens,decode_bs]`
  (OUTSIDE the graph) to agree the per-step phase+size, then:
  * DECODE: common bs = smallest captured graph bs >= agreed max; every replica replays the SAME
    graph (under-full replica = all-dummy padded_reqs, zero real reqs -> no KV writes, logits[:0]
    discarded). The EP all_gather/all_reduce live INSIDE that captured graph at fixed N=common_bs.
  * PREFILL (eager): `ep.pad_tokens` = agreed common token count; `MoELayer` zero-pads its rows up to
    it before the all_gather (token counts differ per replica), slices its real rows back. An idle
    replica runs a 1-token dummy prefill (track_reqs=False so its dummy never enters the decode set).
  The MAX all_reduce is the ONLY per-step host sync and is outside the graph (it just SELECTS the
  graph), so capture — incl. the in-graph EP collectives — is preserved. No all_to_all, no in-graph
  host handshake, no conditional dummy forward.

## Risks (call them out honestly in results)
- Lockstep + dummy-forward correctness (deadlock if an idle replica skips the collective).
- top-1 routing skew across only 2 EP ranks (a hot expert starves a rank).
- all_gather communicates ALL tokens (more bandwidth than true all_to_all) — fine for dp=2, revisit
  for larger dp (then bind `ncclAlltoAll`, header already present in `csrc/include/minisgl/nccl227.h`).
- Independent-DP-scheduling (EP off) vs lockstep (EP on) are different loop modes — gate cleanly on
  `--enable-ep`.
