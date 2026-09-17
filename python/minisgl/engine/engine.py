from __future__ import annotations

import os
import time
from collections import deque
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, SamplingParams, set_global_ctx
from minisgl.distributed import (
    EPCommunicator,
    destroy_distributed,
    enable_custom_ar_distributed,
    enable_custom_ar_ep,
    enable_pynccl_distributed,
    set_dp_info,
    set_tp_info,
)
from minisgl.kvcache import create_kvcache_pool
from minisgl.kvcache.cca_state import CCAStateCache
from minisgl.kvcache.gdn_state import GDNStateCache
from minisgl.kvcache.host_arena import PinnedFrameArena, host_tier_enabled, stage_ring_depth
from minisgl.layers import set_rope_device
from minisgl.models import ModelConfig, cast_checkpoint_tensor, create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import (
    div_even,
    init_logger,
    is_rocm,
    is_sm90_supported,
    is_sm100_supported,
    torch_dtype,
)
from minisgl.weights.accounting import corrected_model_memory
from minisgl.weights.bake import StageASession

from .config import EngineConfig, resolve_prefix_cache, snapshot_ladder_depth
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)

# --- CUDA-graph reservation heuristics (see Engine._graph_capture_bytes) ------------------------
# Peak transient (activation-pool) memory a captured decode/verify forward folds into its graph
# memory pool, estimated as this multiple of tokens*hidden*dtype. The exact static I/O buffers
# (logits especially, for large vocabularies) dominate; this term is a modest, conservative pad for
# the per-forward activations the pool must also hold. Bumped by _GRAPH_ROUNDUP + the env margin.
_GRAPH_ACT_MULT = 8
# We can't know the proposer's aux-capture count at KV-sizing time (it's built later), so assume up
# to this many aux-hidden layers for the spec-verify buffer. Over-reserving is harmless: the aux
# buffer is tokens*hidden*dtype, tiny next to the vocab-wide logits buffer.
_GRAPH_ASSUMED_AUX = 8
# Slight round-up so the estimate errs toward preventing OOM rather than over-starving KV.
_GRAPH_ROUNDUP = 1.1
# Device bytes a BLOCK-DIFFUSION canvas graph costs per canvas token (graph pool + the warmup's
# retained allocator growth, which never returns to the device). Deliberately NOT expressed as
# `_GRAPH_ACT_MULT * hidden * dtype`: that multiplier was fitted against decode/verify graphs of one
# to a few tokens per sequence, and a canvas step is a prefill-shaped 256-token forward through 30
# layers of MoE — the same formula under-reserves it by ~50x. MEASURED on the served checkpoint
# (DiffusionGemma-26B-A4B-int4, TP=2, canvas 256, bs=1): device-free fell 2.96 -> 2.41 GiB across
# capture, i.e. ~2.2 MiB per canvas token. This is an OBSERVATION with headroom, not a derivation,
# and it is named that way so nobody mistakes it for a model of the allocator.
_CANVAS_GRAPH_BYTES_PER_TOKEN = int(2.5 * (1 << 20))


def _ple_stage_tokens(config: EngineConfig) -> int:
    """Tokens the Qwen4-Exp PLE staging buffer must hold — the largest single forward.

    Two batch shapes, and it is the MAX of both, not the prefill one: an extend batch is bounded by
    the chunked-prefill budget (`SchedulerConfig.max_extend_tokens`), but a DECODE batch carries one
    token per running sequence and those are configured independently. `--max-extend-tokens 8` with
    `--max-running-requests 256` is a legal (and, for a 16 GB card, sensible) combination that sizes
    the buffer at 8 and then hands it a 256-token decode step. `PLEEmbeddingSource.stage_rows` does
    raise on that rather than corrupt anything — but it raises on the first busy decode step of a
    serve that booted fine, which is the wrong place to find out.

    The allocation and the KV-pool reservation both call this, so they cannot disagree.
    """
    return max(config.max_forward_len, config.max_running_req + 2)


def _swa_kv_geometry(mc: ModelConfig) -> Tuple[int, int]:
    """(head_dim, num_kv_heads) of the SLIDING layers' ring KV pool.

    Gemma4's two layer types do not share a KV geometry: `head_dim`/`num_kv_heads` carry the
    FULL-attention one (512 / 2) because they size the MAIN paged pool, and the sliding layers keep
    their own (256 / 8). Every other SWA model (Laguna) leaves `swa_head_dim`/`swa_num_kv_heads`
    None and gets back exactly the values it always used, so its sizing is unchanged.

    Both the ring-pool ALLOCATION and its byte RESERVATION in _determine_num_pages go through this
    one function on purpose: if they ever disagreed the engine would reserve one pool's bytes and
    allocate another's — an OOM at boot or a silently under-allocated KV pool, neither of which
    names the mismatch."""
    head_dim = mc.swa_head_dim if mc.swa_head_dim is not None else mc.head_dim
    num_kv_heads = mc.swa_num_kv_heads if mc.swa_num_kv_heads is not None else mc.num_kv_heads
    return head_dim, num_kv_heads


def _swa_ring_block(mc: ModelConfig, spec_config) -> int:
    """Extra ring slots per sequence BEYOND the sliding window.

    A plain window-sized ring (stride == W) is correct for single-query decode/extend, but any path
    that writes a multi-token block into the ring BEFORE attending it needs those tokens in slots
    disjoint from the live window, because position p and p+W share slot p%W once the sequence is
    longer than W. Two such paths exist and they are mutually exclusive:

      * SPEC VERIFY stores anchor + drafts (num_draft + 1) and the rejected drafts must not land on
        a valid-window slot the next gather reads;
      * BLOCK DIFFUSION stores a whole canvas_length canvas and REREADS the window in the same
        forward — at stride W all 256 of 256 canvas slots alias a window slot, so the decoder would
        overwrite the very prefix it must attend to (measured, tools/canvas_attention_probe.py).

    Returns the larger of the two so one number serves both, and 0 when neither applies (stride ==
    window, byte-identical to the pre-spec path).

    Both the ring ALLOCATION and its byte RESERVATION in _determine_num_pages call this, for the same
    reason _swa_kv_geometry exists: two independent copies of the arithmetic would drift, and the
    failure is a ring sized for one stride addressed with another — a silently corrupt cache, not a
    crash."""
    spec_block = (spec_config.num_draft + 1) if spec_config is not None else 0
    canvas_block = mc.canvas_length if mc.is_block_diffusion else 0
    return max(spec_block, canvas_block)


# --- env-gated decode-loop profiler (diagnostics only) -----------------------------------------
# MINISGL_PROFILE=<trace.json> captures a window of forward steps on the primary rank into a Chrome
# trace, then no-ops. Used to split per-step wall time into GPU-active vs launch-bubble overhead.
# Inert unless the env var is set, so it costs nothing on a normal serve.
_PROF_STATE: Dict[str, Any] = {"p": None, "n": 0}


def _maybe_profile() -> None:
    spec = os.environ.get("MINISGL_PROFILE")
    if not spec:
        return
    from minisgl.distributed import get_tp_info

    if not get_tp_info().is_primary():
        return
    skip = int(os.environ.get("MINISGL_PROFILE_SKIP", "40"))
    active = int(os.environ.get("MINISGL_PROFILE_STEPS", "50"))
    st = _PROF_STATE
    st["n"] += 1
    if st["n"] == skip:
        import torch.profiler as tp

        st["p"] = tp.profile(activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA])
        st["p"].__enter__()
    elif st["p"] is not None and st["n"] == skip + active:
        torch.cuda.synchronize()
        st["p"].__exit__(None, None, None)
        # Under DP (tp_size=1) EVERY replica is its own TP-primary, so an unqualified path would have
        # both replicas write the SAME file concurrently -> corrupt gzip. Qualify by dp_rank.
        from minisgl.distributed import try_get_dp_info
        dp = try_get_dp_info()
        out = spec if (dp is None or dp.dp_size == 1) else spec.replace(
            ".pt.trace", f".dp{dp.dp_rank}.pt.trace")
        st["p"].export_chrome_trace(out)
        st["p"] = None
        logger.info_rank0(f"[profile] wrote {active}-step trace to {out}")

# Token count for the one-time GDN conv autotune warmup (3c-3). A single representative
# prefill length settles the per-process in-place batch_ptr autotune.
_GDN_WARMUP_TOKENS = 512


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event
    # Per-token logprob capture for rows that asked (SamplingParams.logprobs > 0): batch-row ->
    # (top_ids, top_logprobs, sampled_token_logprob). None on the common no-logprobs path. NOTE:
    # positional destructures of this tuple must use starred unpacking or field access.
    logprobs: dict | None = None


class Engine:
    def __init__(self, config: EngineConfig):
        # Completed-but-unread prefill timing events (see prefill_seconds_total).
        self._pf_events: deque = deque()
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        # Register this replica's DP coordinates (inert DpInfo(0,1) when dp_size=1). Done before any
        # CUDA init so EP (later) can build its dp collective group; also lets get_dp_info() resolve
        # to the real replica everywhere instead of the default.
        # EP-over-TP (vllm-style TP+EP): with dp=1 and tp>1, --enable-ep shards the experts across the
        # TP ranks (attention stays TP-sharded) instead of across DP replicas. Otherwise EP means DP+EP.
        self._ep_over_tp = (
            config.enable_ep and config.dp_info.dp_size == 1 and config.tp_info.size > 1)
        set_dp_info(
            dp_rank=config.dp_info.dp_rank,
            dp_size=config.dp_info.dp_size,
            enable_ep=config.enable_ep,
            ep_over_tp=self._ep_over_tp,
        )
        _adjust_config(config)

        # Per-replica device: dp_rank=1 (tp_size=1) lands on cuda:1, etc. dp_size=1 -> cuda:{tp_rank}
        # (unchanged historical mapping). See EngineConfig.device_index.
        self.device = torch.device(f"cuda:{config.device_index}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.tp_size = config.tp_info.size
        # Speculative decoding config (None unless --spec-algorithm enables it). The scheduler
        # routes to the synchronous spec loop when this is set; every spec path is gated on it.
        self.spec_config = config.spec_config
        # Served model dir — kept so self-draft proposers (TiDAR) can read sidecar config
        # (tidar_config.json) from the model folder without re-plumbing the path.
        self.model_path = config.model_path
        # Expert-parallel coordinates (inert when --enable-ep is off / dp_size==1). The scheduler
        # reads these to drive the per-step common-bs lockstep over self.dp_cpu_group (built in
        # _init_dp_communication). self.ctx.ep carries the in-graph collective group for MoELayer.
        self.enable_ep = config.enable_ep and (config.dp_info.dp_size > 1 or self._ep_over_tp)
        self.dp_rank = config.dp_info.dp_rank
        self.dp_size = config.dp_info.dp_size
        # fp8 (e4m3fn) KV cache — opt-in via MINISGL_KV_FP8=1 (the "no-F16" KV path:
        # store e4m3 -> cast once to bf16 -> f32 accumulate, scalar scale folded in the
        # attention kernel). Activations/weights stay bf16; only the KV buffer is fp8.
        self.kv_dtype = (
            torch.float8_e4m3fn if os.environ.get("MINISGL_KV_FP8") == "1" else self.dtype
        )
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)

        # BOOT ATTRIBUTION (`weights/boot_timeline.py`). Every `with _bt.phase(...)` below is two
        # `perf_counter()` calls plus four small reads, against phases measured in seconds — so it
        # runs unconditionally rather than behind a flag. Boot on this model takes 519-871 s and,
        # before this, only the Stage-B slice of it had ever been timed; the other 352-593 s was dark.
        from minisgl.weights import boot_timeline as _bt

        with _bt.phase("init_communication"):
            self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with _bt.phase("model_construct_meta"):
            with torch.device("meta"), torch_dtype(config.dtype):
                self.model = create_model(config.model_config)
        # Weight offload, step 1 of 4: resolve the plan from the META-BUILT model — after the build
        # (which allocates ZERO bytes on the card) and before anything is loaded, so an unsatisfiable
        # plan is still a config error in seconds rather than a 90-second OOM. Inert, at zero device
        # cost, whenever the model fits, which is why it runs unconditionally instead of behind a
        # flag — see weights/bake.py for the whole window.
        #
        # RESOLVING OFF THE MODEL, NOT OFF config, IS THE GENERALITY REQUIREMENT. Given a model
        # object `resolve_weight_plan` reads the layer set, this rank's EP/TP sharding and the
        # per-expert byte count off objects that already exist (one format-agnostic granule walker),
        # so a new model family or quant format implements NOTHING. Given only config it TRANSCRIBES
        # seven builder files and nine container `__init__`s, and that transcription is already
        # wrong here: `models/utils.py`'s `MoEMLP` builds its `MoELayer` with no `quant=` at all, so
        # its experts are bf16 whatever `config.quant` says — the config path sizes them int4 and
        # under-reserves the arena ~4x, in the direction that makes an infeasible plan look feasible.
        # The THIRD tier, built BEFORE the plan is resolved because the plan must not see the layers
        # it takes. Returns (None, ()) unless --weight-offload-stream-layers asked for it.
        with _bt.phase("weight_stream_tier_build"):
            self._woff_stream, stream_layers = self._build_weight_stream_tier(config)
        _bt_plan = _bt.phase("weight_plan_resolve")
        _bt_plan.__enter__()
        self._woff = StageASession.begin(
            config,
            probe=self._woff_mem_probe,
            log=logger.info_rank0,
            model=self.model,
            stream=self._woff_stream,
            stream_layers=stream_layers,
            # EXPLICIT, never omitted: `resolve_weight_plan` reads an absent/zero device budget as
            # "the expert tier may occupy zero VRAM", i.e. an all-host plan for every MoE model.
            device_budget_bytes=self._weight_offload_device_budget(config),
            # DERIVED, or CHOSEN? The resolver cannot tell the two apart and they do not mean the
            # same thing. With no --weight-offload-device-gb the budget above falls back to
            # `total_memory * memory_ratio` — the WHOLE KV budget — and the device tier is billed
            # inside `model` in _determine_num_pages (by design: the plan forbids a sixth
            # subtrahend). So a non-empty plan under a derived budget makes `available_memory`
            # negative before state/draft/graph/snap are even counted, and `assert num_pages > 1`
            # cannot not fire. Saying which it was lets that verdict be reached from integers here,
            # instead of after the arena has pinned tens of GiB of unevictable host RAM and the whole
            # checkpoint has been read and repacked.
            budget_is_derived=config.weight_offload_device_gb <= 0,
            # PROVE the plan is rank-identical rather than arguing that it is. It is pure only if
            # every input is, and `device_budget_bytes` is an input the resolver cannot vet: a budget
            # derived from a live per-rank free-memory reading differs by tens of MiB between the two
            # rank processes, which is enough to move one layer across the greedy fill boundary. From
            # there the ranks hold different device tiers, `_determine_num_pages` bills a different
            # `model_memory` (the offload correction is per-rank and `num_pages` is NOT cross-rank
            # reduced), and the two KV pools come out different sizes with no error anywhere. One
            # tiny gloo all_gather_object at boot, over the TP CPU group built at :213.
            agreement_group=self.tp_cpu_group,
        )
        _bt_plan.__exit__()

        def _mem_probe(tag: str) -> None:
            """Localize where DEVICE memory goes that torch does not account for. `used` is the true
            device-wide draw (what the KV sizing bills as `model`); `reserved` is everything torch
            holds. used-minus-reserved is the non-torch remainder — bisecting it across load vs
            post_load says whether it is the weight stream or the quant layout conversion."""
            torch.cuda.synchronize(self.device)
            free = torch.cuda.mem_get_info(self.device)[0]
            used, res = init_free_memory - free, torch.cuda.memory_reserved()
            logger.info_rank0(
                f"[mem] {tag}: used={mem_GB(used)} allocated={mem_GB(torch.cuda.memory_allocated())} "
                f"reserved={mem_GB(res)} non-torch={mem_GB(used - res)} "
                # PEAKS separate the two explanations for reserved >> allocated: a transient spike
                # during load (peak allocated near reserved -> reduce the loading pattern) versus
                # fragmentation (peak allocated near final -> the segments are simply unusable).
                f"| peak_alloc={mem_GB(torch.cuda.max_memory_allocated())} "
                f"peak_reserved={mem_GB(torch.cuda.max_memory_reserved())}"
            )

        _mem_probe("after build (meta)")
        # Weight offload, step 2 of 4: pin the host arena BEFORE the checkpoint is read. Pinning is
        # the slow, capacity-bound step (P3b: 4.88 GB/s, and 62 of 68 GiB was the ceiling for two
        # ranks on an IDLE box), so doing it first turns "this box does not have the RAM" into a
        # failure in seconds instead of after a full load. It takes ZERO device bytes — asserted at
        # seal(), never assumed, because Phase 0 caught the driver reporting host memory that was
        # actually VRAM.
        with _bt.phase("arena_pin_attach"):
            self._woff.attach()
        #: `stage_b.ChunkedLoadLedger` when the load was chunked, None when it was one-shot. Always
        #: present so "was this serve chunked?" is a readable fact rather than a `hasattr`.
        self.stage_b_ledger = None
        # Stage B, when the model family has a chunk enumeration AND there is a host tier to bake
        # into: read-fill-finalize-PLACE one chunk at a time, so the peak live set is one chunk
        # instead of one checkpoint. Falls back to the one-shot load for every other model, and for
        # this one when offload is not configured — the fallback is the shipped path, unchanged.
        with _bt.phase("weight_load"):
            if not self._load_weight_chunked(config):
                with _bt.phase("oneshot_read_and_fill"):
                    self.model.load_state_dict(self._load_weight_state_dict(config))
                _mem_probe("after load_state_dict")
                with _bt.phase("oneshot_post_load"):
                    self.model.post_load()  # finalize weights (quantized layout conversion)
        _mem_probe("after post_load")
        # THE ONE DECISION post_load MAKES THAT NO TENSOR RECORDS, checked across the ranks that made
        # it independently. A compressed-tensors int4 container decides its packed sign convention
        # from a SAMPLE of its own shard, and no two TP ranks hold the same bytes (plain TP splits
        # the output rows, EP-over-TP splits the experts). On a mixed-packing checkpoint the ranks
        # can land on opposite answers, and then half a TP-split GEMM is dequantized in the uint4b8
        # domain and half in two's-complement — right shapes, no kernel error, fluent wrong text.
        # The detector cannot catch it alone: its tie-refusal is evaluated per shard, and a mixed
        # stack is only a tie when you can see all of it, which no rank ever does. One
        # all_gather_object of a dict of bools on the gloo group the engine already built. Runs
        # unconditionally so a checkpoint that develops the problem cannot boot quietly; it is a
        # no-op at TP=1 and on every model with no CT containers (NVFP4/MXFP4 declare none).
        from minisgl._hip_engage import engaged
        from minisgl.quant.method import verify_ct_sign_across_ranks

        with _bt.phase("ct_sign_verify"):
            self.ct_sign_decisions = verify_ct_sign_across_ranks(
                self.model, self.tp_cpu_group, config.tp_info.size, config.tp_info.rank
            )
        if self.ct_sign_decisions:
            engaged(f"quant.ct_sign_cross_rank[tp{config.tp_info.size}]")
        # Weight offload, steps 3-4: host-placed containers are copied into the arena and their
        # device originals dropped STRICTLY between post_load() and _determine_num_pages. That
        # placement is the whole VRAM-accounting design: everything the bake moves is inside the
        # `device_used = old_free - new_free` window measured below, so the device tier is billed by
        # the existing `model_memory` term and the KV budget needs no sixth subtrahend (which, at a
        # 16 GB tier, would size the pool negative). Device-placed layers are NOT copied — they are
        # already where they belong, and reallocating them would hold two copies live at once.
        # seal() closes the mapping window (rule R1); a later mapping would be invisible both to
        # this sizing and to Scheduler._prefill_budget_now.
        with _bt.phase("weight_offload_bind_seal"):
            self._woff.note_loaded()
            self._woff.bind(self.model)
            self._woff.seal()
        # AFTER seal, which is the moment residency stops changing. The hook is a forward-path
        # wrapper, not a residency change, but installing it inside the window would put file I/O in
        # the middle of the gates that are still measuring the window's device cost.
        self._woff.install_stream_hooks()
        # ROUTING TRACE, AFTER the stream tier so this wrapper is the OUTERMOST one and the layer
        # id is in scope before the tier's staging runs. Measure-only and default OFF: returns
        # None unless MINISGL_MOE_ROUTE_TRACE names an existing writable dir, and RAISES if it
        # names one that does not exist rather than degrading to a silent no-trace. The model
        # shape is derived from the discovered ops, not transcribed from config -- a header that
        # disagrees with the trace body would mis-scale every hit rate the oracle reports.
        from minisgl.weights import route_trace as _route_trace

        # The expert-residency cache learns routes from THIS ring, so the tracer is armed whenever
        # a cache was attached even if no fixture dir was named (`observe_only`). One ring, one
        # D2H: a separate observation path for the cache would pay the same transfer twice.
        from minisgl.weights.moe_interpose import live_expert_cache

        _cache = live_expert_cache()
        _tracer = _route_trace.maybe_install(
            self.model,
            model_slug=str(config.model_path),
            tp_rank=config.tp_info.rank,
            dp_rank=self.dp_rank,
            device=self.device,
            observe_only=_cache is not None,
            # A speculative VERIFY forward carries max_running_req * (num_draft + 1) rows, and the
            # ring must hold them or the record falls to the host path and never reaches the expert
            # cache's observer. Same row count engine.py already derives for the scored-row cap.
            ring_rows=(
                config.max_running_req * (1 + config.spec_num_draft)
                if config.spec_config is not None else 1
            ),
        )
        if _cache is not None:
            if _tracer is None:
                # The cache cannot learn anything without the ring, and a cache that observes
                # nothing keeps `slot_of` at -1 forever: every expert reads the host base, i.e.
                # SLOWER than the shipped path (the slab is allocated and never used). Refuse.
                raise RuntimeError(
                    "expert cache is attached but the route tracer did not arm, so no references "
                    "would ever reach it. The cache would hold VRAM and never serve a hit."
                )
            _tracer.set_observer(_cache.observe)
            # STARTED FROM THIS THREAD ON PURPOSE: `start()` captures the compute stream, and
            # `torch.cuda.current_stream()` is per-thread — captured anywhere else every fence in
            # the promotion path would order against a stream no kernel runs on.
            _cache.start()
            print(f"[expert-cache] observing the route ring "
                  f"(drain_every={_tracer.drain_every} steps), manager thread started", flush=True)
        if self._woff.enabled:
            _mem_probe("after weight offload")

        # ======================= KV cache initialization ========================
        with _bt.phase("kv_sizing"):
            self.num_pages = self._determine_num_pages(init_free_memory, config)
        # The slab, now that the pool has been sized around it. Allocating BEFORE the pool would let
        # a sizing bug hide (the pool would simply shrink to fit); allocating after means a mismatch
        # between what was reserved and what is taken shows up as an OOM here, at boot, rather than
        # on someone's first prefill. `_moe_prefill_stage_bytes` returns 0 when it declined.
        if self._woff.enabled:
            from minisgl.weights import prefill_stage
            with _bt.phase("moe_prefill_stage_alloc"):
                prefill_stage.install(getattr(self, "_moe_stage_reserved", 0), self.device)
        num_tokens = self.num_pages * config.page_size
        with _bt.phase("kv_pool_alloc"):
            self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
                model_config=config.model_config,
                num_pages=self.num_pages + 1,  # +1 for dummy page
                page_size=config.page_size,
                device=self.device,
                dtype=self.kv_dtype,
            )

        # ======================= SWA (sliding-window) ring KV pool ========================
        # A SWA-hybrid model (Laguna) keeps its FULL-attention layers in the main pool above (sized by
        # num_kv_layers = the full-attn count) and its SLIDING layers in this SEPARATE window-bounded
        # pool: one fixed per-sequence ring of `sliding_window` slots (page_size=1), indexed like the
        # GDN/CCA state cache by table_idx (slot = table_idx*window + pos%window). This is what avoids
        # allocating full-context KV for the 30 window-512 sliding layers (~3.8x at 32k). Its bytes are
        # reserved up front in _determine_num_pages (like the recurrent-state caches), so the main pool
        # gets the remainder. None for every non-SWA model, so other paths are untouched.
        mc0 = config.model_config
        if mc0.is_swa_hybrid:
            from minisgl.kvcache.mha_pool import MHAKVCache

            # Ring STRIDE per sequence = window + the multi-token block any path writes before it
            # reads (spec verify's K+1, or block diffusion's whole canvas). See _swa_ring_block for
            # why a stride of exactly W corrupts both. No spec and no canvas -> stride == window,
            # byte-identical to Track A.
            swa_stride = mc0.sliding_window + _swa_ring_block(mc0, self.spec_config)
            self.ctx.swa_ring_stride = swa_stride
            swa_slots = (config.max_running_req + 2) * swa_stride  # +1 NULL, +1 dummy
            # The ring holds the SLIDING layers, so it takes the SLIDING geometry — which for Gemma4
            # is not the model-wide one (256/8 here vs the 512/2 that sizes the main pool). Sizing it
            # off mc0.head_dim/num_kv_heads would hand store_kv a mis-shaped buffer view: wrong bytes
            # per slot and a silently corrupt cache, not a crash. Same values as before for Laguna.
            swa_head_dim, swa_num_kv_heads = _swa_kv_geometry(mc0)
            self.ctx.swa_kv_cache = self.swa_kv_cache = MHAKVCache(
                num_kv_heads=swa_num_kv_heads,
                num_layers=mc0.num_swa_layers,
                head_dim=swa_head_dim,
                num_pages=swa_slots,
                page_size=1,  # ring is addressed by absolute slot; no page grouping
                device=self.device,
                dtype=self.kv_dtype,
            )
            logger.info_rank0(
                f"SWA ring KV: {mc0.num_swa_layers} layers x {swa_slots} slots "
                f"(window={mc0.sliding_window}, stride={swa_stride}, {config.max_running_req} seqs, "
                f"{swa_num_kv_heads} kv heads x {swa_head_dim})"
            )
        else:
            self.swa_kv_cache = None  # type: ignore[assignment]

        # ======================= fp8-KV scale install (must precede EVERY store) ================
        # Resolve k_scale/v_scale for an fp8 KV cache and freeze them HERE — after both pools exist,
        # before the warmup forward, graph capture, or any request. The ordering is a correctness
        # requirement, not a preference: one descale has to undo every store ever written under it,
        # so changing it later invalidates the whole cache (a captured graph does pick the new value
        # up — measured max|Δ|=4.85e-01 on the attention output). Sources, in order: an offline
        # calibration sidecar (the only per-head one), then the checkpoint's own
        # quantization_config.kv_cache_scheme + self_attn.{k,v}_scale, then a loud warning + the
        # defined identity fallback. No-op for a bf16/fp16 cache. See kvcache/fp8_scales.py.
        if self.kv_dtype == torch.float8_e4m3fn:
            from minisgl.kvcache.fp8_scales import install_kv_fp8_scales

            install_kv_fp8_scales(
                config.model_path, config.model_config, self.kv_cache, self.swa_kv_cache
            )

        # ======================= GDN recurrent-state cache (Phase 3c/3d) ========================
        # GDN-hybrid models keep a fixed per-sequence recurrent state (conv + ssm) alongside
        # the paged MHA KV cache. The scheduler wires GDN slot alloc/free + per-batch GDN
        # metadata ONLY when this is non-None — see Scheduler.__init__ / _prepare_batch /
        # _free_req_resources — so the dense path stays untouched (gdn_state is None).
        # ★ A GDN-hybrid engine MUST run the non-radix ("naive") prefix cache (GDN state is not
        # prefix-cacheable — see GDNSlotManager). GDN cudagraph capture IS supported (GDNGraphCapture
        # wires the recurrent-state static buffers; validated once _capture_graphs runs grad-free) —
        # the old "GDN must run eager" constraint is lifted; only the naive-prefix-cache one remains.
        mc = config.model_config
        if mc.is_gdn_hybrid:
            # GDN is head-parallel under TP: each rank's linear_attn owns conv_dim/tp channels
            # and num_v_heads/tp value heads (Phase 4-1), so its recurrent state slot must match.
            # conv_dim = 2*key_dim + value_dim is linear in the (tp-divisible) head counts, so
            # div_even(conv_dim, tp) == the layer's local conv_dim exactly. head_*_dim are per-head.
            tp = config.tp_info.size
            self.ctx.gdn_state = self.gdn_state = GDNStateCache(
                num_gdn_layers=mc.num_gdn_layers,
                num_slots=config.max_running_req + 2,  # +1 NULL block, +1 dummy
                conv_dim=div_even(mc.gdn_conv_dim, tp),
                conv_kernel=mc.linear_conv_kernel_dim,
                num_v_heads=div_even(mc.linear_num_value_heads, tp),
                head_v_dim=mc.linear_value_head_dim,
                head_k_dim=mc.linear_key_head_dim,
                # fp32 conv state: the gdn_hip HIP kernels read/update conv state in place in fp32.
                dtype=torch.float32,
                # ssm_state defaults to bf16: halves the (large) recurrent-state HBM (~2x
                # max_running_req), compute stays fp32 in-register (gdn_hip GDN_DISPATCH_SSM). The
                # bf16 storage rounding is bounded (contractive gated decay, not accumulating —
                # gdn_hip_parity.check_ssm_state_bf16 + bench_bf16_state_longdecode) so decode output
                # is fp32-equivalent. Set MINISGL_SSM_BF16=0 to force fp32 (max accuracy). conv_state
                # always stays fp32.
                ssm_dtype=(torch.float32 if os.environ.get("MINISGL_SSM_BF16", "1") == "0"
                           else torch.bfloat16),
                device=self.device,
            )
            # Settle causal_conv1d's per-process in-place batch_ptr autotune on a private
            # scratch BEFORE the first real batch (3c-3) — an unwarmed first prefill is
            # op-sequence-sensitive (NaN/0/OOM). Writes no real state.
            with _bt.phase("gdn_conv_warmup"):
                for gdn in self.model.iter_gdn_layers():
                    gdn.warmup_conv(_GDN_WARMUP_TOKENS)
            logger.info_rank0(
                f"GDN state: {mc.num_gdn_layers} layers x {config.max_running_req + 2} slots "
                f"(conv_dim={mc.gdn_conv_dim}); conv warmup done"
            )
        else:
            self.gdn_state = None  # type: ignore[var-annotated]  # GDNStateCache | None

        # ======================= ZAYA CCA recurrent-state cache (conv + prev_hs) ========================
        # CCA-hybrid (Zaya) models keep two fixed per-sequence recurrent buffers per CCA (even) layer
        # alongside the paged GQA KV cache: the causal-conv window (conv_states) and the previous
        # token's hidden (prev_hs, consumed by val_proj2). Same lifecycle as GDN — the scheduler wires
        # slot alloc/free + per-batch metadata ONLY when this is non-None (cca_state is None for every
        # non-Zaya model, so other paths are untouched). Forces naive prefix cache + eager, like GDN.
        # Guarded on getattr so the engine stays import-safe until ModelConfig grows the CCA fields
        # (PORT_PLAN §8): a model without is_cca_hybrid leaves cca_state None.
        if getattr(mc, "is_cca_hybrid", False):
            # CCA is head-parallel under TP (like GDN above): each rank's conv front-end owns
            # conv_dim/tp channels ((nq+nk)/tp heads), so its recurrent conv slot matches. prev_hs
            # stays FULL hidden_size — it feeds val_proj (replicated, full-hidden input), not the
            # sharded conv. conv_dim = (nq+nk)*hd is linear in the tp-divisible head counts, so
            # div_even(conv_dim, tp) == the layer's local conv_dim exactly.
            tp = config.tp_info.size
            self.ctx.cca_state = self.cca_state = CCAStateCache(
                num_cca_layers=mc.num_cca_layers,
                num_slots=config.max_running_req + 2,  # +1 NULL block, +1 dummy
                conv_dim=div_even(mc.cca_conv_dim, tp),
                conv_kernel=mc.cca_conv_width,
                hidden_size=mc.hidden_size,
                # fp32 recurrent state: the cca_hip conv kernels read/update conv_states in fp32.
                dtype=torch.float32,
                device=self.device,
            )
            # CCA conv kernels have no autotune; warmup is a no-op (kept for engine-loop symmetry).
            for cca in self.model.iter_cca_layers():
                cca.warmup_conv(_GDN_WARMUP_TOKENS)
            logger.info_rank0(
                f"CCA state: {mc.num_cca_layers} layers x {config.max_running_req + 2} slots "
                f"(conv_dim={div_even(mc.cca_conv_dim, tp)}, conv_width={mc.cca_conv_width})"
            )
        else:
            self.cca_state = None  # type: ignore[var-annotated]  # CCAStateCache | None

        # ======================= Qwen4-Exp PLE n-gram runtime ========================
        # The PLE block on one decoder layer reads its n-gram embeddings from `Context.ple`, staged
        # host-side before every forward (`Scheduler._stage_ple`). It is built HERE, next to the GDN
        # cache, because it is the same kind of object: a fixed per-sequence recurrent state indexed
        # by the SAME slot id the GDN slot manager hands out, plus a static staging buffer. Passing
        # the identical `max_running_req + 2` slot count is what keeps those slot ids valid on both
        # sides. `Qwen4ExpPLE.forward` REFUSES when nothing is staged rather than skipping the block,
        # so a model with `ple_layer_ids` and no runtime cannot serve quietly-degraded output — which
        # is exactly what this wiring closes: before it, every qwen4_exp forward raised.
        self.ple_runtime = None
        if getattr(mc, "ple_layer_ids", ()) and hasattr(self.model, "ple_block"):
            from minisgl.ple import build_ple_runtime

            with _bt.phase("ple_runtime_build"):
                self.ctx.ple = self.ple_runtime = build_ple_runtime(
                    model=self.model,
                    config=mc,
                    model_path=config.model_path,
                    num_slots=config.max_running_req + 2,  # == the GDNStateCache slot count above
                    max_tokens=_ple_stage_tokens(config),
                    device=self.device,
                    dtype=self.dtype,
                )
            logger.info_rank0(
                f"PLE runtime: n-gram table open, {config.max_running_req + 2} state slots, "
                f"staging {_ple_stage_tokens(config)} tokens x {mc.ple_embed_dim} "
                f"({mem_GB(self._ple_runtime_bytes(config))} device)"
            )

        # Pinned host arena for the recurrent-radix snapshot store. HERE, not in
        # _determine_num_pages: the pool is sized at line ~220, BEFORE either state cache exists
        # (which is exactly why _recurrent_state_bytes is config-derived rather than measured), so
        # attaching an arena from there would find no cache to attach it to. Ordering is fine — the
        # arena is HOST memory and does not come out of the device budget; the only device cost is
        # the staging ring, which each state cache allocates in its own ctor and which
        # _rec_snapshot_store_bytes already reserved.
        with _bt.phase("snapshot_host_arena"):
            self._build_snapshot_host_arena(config)

        # ======================= CAM editable-memory (Option B, Phase 0) ========================
        # Build the CAM store+tap+router IN THE BACKEND, reusing the SERVED model's weights (no
        # co-located 8 GB HF base — that is the whole point of the backend model-share). Gated OFF by
        # default: only when MINISGL_CAM=1 + MINISGL_CAM_CHECKPOINT is a real dir AND the model exposes
        # the L24 tap seam (stage_cam). Off → cam_state stays None → the tap hook is a byte-exact no-op,
        # so dense/GDN serving is completely unperturbed. Per-request staging + seed-once decode is
        # Phase 1 (scheduler); this block only builds the state and registers the tap layer.
        self.cam = None
        _cam_ckpt = os.environ.get("MINISGL_CAM_CHECKPOINT")
        inner = getattr(self.model, "model", None)

        def _cam_gather_full_vocab(mod):
            # CAM's cosine subject index needs the FULL vocab embedding table, but under TP the embedding
            # is vocab-parallel sharded (each rank holds vocab/tp rows). All-gather the shards once here —
            # a build-time collective every SPMD rank reaches symmetrically — so both ranks build an
            # IDENTICAL full-vocab store (this is what keeps their pointer-delivery decisions in lockstep,
            # so no per-rank broadcast of the forced tokens is needed). tp_size==1 leaves the table whole.
            w = mod.weight
            tp = getattr(mod, "tp_size", 1)
            if tp <= 1:
                return w
            g = mod._comm.all_gather(w)                          # rank-ordered concat along dim0
            g = g.view(tp, mod.num_embeddings_tp, w.shape[1]).reshape(tp * mod.num_embeddings_tp, w.shape[1])
            # Land the full-vocab table on CPU: it is allocated AFTER the KV pool is sized, so keeping
            # ~1.2 GB/card on-GPU steals the headroom runtime activations need (OOMs the first forward on
            # a 16 GB card). The pointer cosine reads (_subj_key) are tiny + prefill-only, so CPU is free.
            # NOTE: TP>1 is currently always pointer_only (dim-mismatch); a future matching-dim TP>1 tap
            # checkpoint would need this back on-device (the adapter runs on GPU) — gate on that then.
            return g[: mod.num_embeddings].contiguous().to("cpu")

        if (os.environ.get("MINISGL_CAM") == "1" and _cam_ckpt and os.path.isdir(_cam_ckpt)
                and inner is not None and hasattr(inner, "embed_tokens")):
            try:
                from minisgl.cam.memory import CAMMemory
                # Model-share: build the store from the SERVED model's own embedding + lm_head (all-gathered
                # to full vocab under TP). Qwen3.5 ties word embeddings, so the tied embedding weight IS the
                # lm_head table; gather the head separately only when untied.
                _full_embed = _cam_gather_full_vocab(inner.embed_tokens)
                _lmh = self.model.lm_head
                _full_lm = (_full_embed if getattr(_lmh, "tied_embedding", None) is not None
                            else _cam_gather_full_vocab(_lmh))
                # The residual tap needs the model's `stage_cam` seam (dense qwen3_5 only). MoE models (35B)
                # lack it → build in POINTER/RETRIEVE-ONLY mode: exact-object delivery + ambient retrieve
                # run from the cosine subject index alone (dimension-independent, no tap, no 35B-trained
                # checkpoint). The tap/router path stays off until both a ported seam and a 35B ckpt exist.
                _has_seam = hasattr(inner, "stage_cam")
                cam = CAMMemory(_cam_ckpt, _full_embed, _full_lm, pointer_only=not _has_seam)
                if cam.enabled:
                    self.cam = self.ctx.cam_state = cam
                    # CAMMemory may downgrade to pointer_only when the checkpoint hidden != served hidden,
                    # so gate the tap-seam registration on the RESOLVED mode, not just seam presence.
                    if _has_seam and not cam.pointer_only:
                        inner.stage_cam(cam, None, None)  # register the cam + tap_layer; no bank => no-op
                    logger.info_rank0(
                        f"CAM: backend memory built from {_cam_ckpt} "
                        f"(tap_layer={cam.tap_layer}, n_banks={cam.n_banks}, "
                        f"mode={'pointer/retrieve-only' if cam.pointer_only else 'tap+pointer'}) — model-share"
                    )
                else:
                    logger.warning_rank0(f"CAM: checkpoint {_cam_ckpt} loaded DISABLED — memory off")
            except Exception as e:  # noqa: BLE001 — CAM must never break normal serving
                logger.warning_rank0(f"CAM: backend build failed ({e}) — memory off, serving unaffected")

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        if config.spec_config is not None:
            # Verify-width padding (spec/width.pad_to_captured_width) stages up to the max captured
            # width of filler drafts with no per-req budget clamp, so a request clamped to the exact
            # context limit can stage columns past max_seq_len; give every row that much slack or the
            # staged scatter runs off it into the next request's page-table entries. token_pool
            # (zeros_like this table) inherits the slack. GraphRunner sees the widened value below,
            # so capture and replay agree on the table width.
            aligned_max_seq_len += _align_up_32(config.spec_config.num_draft + 1)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        with _bt.phase("backends_init"):
            self.ctx.attn_backend = self.attn_backend = create_attention_backend(
                config.attention_backend, config.model_config
            )
            if config.model_config.is_moe:
                self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        # real_vocab_size fences the untrained padded lm_head tail off from sampling (see Sampler).
        # Best-effort: the tokenizer is cached (the frontend already loaded it), and an unloadable
        # tokenizer must not take the engine down over a defence-in-depth mask.
        _real_vocab = None
        try:
            from minisgl.utils import load_tokenizer

            _tok_len = len(load_tokenizer(config.model_path))
            if 0 < _tok_len < config.model_config.vocab_size:
                _real_vocab = _tok_len
                logger.info_rank0(
                    f"sampler: masking padded vocab tail [{_tok_len}, "
                    f"{config.model_config.vocab_size}) — {config.model_config.vocab_size - _tok_len} "
                    "untrained ids fenced off"
                )
        except Exception:
            pass
        self.sampler = Sampler(self.device, config.model_config.vocab_size,
                               real_vocab_size=_real_vocab,
                               logit_softcap=getattr(config.model_config, "final_logit_softcapping", None))

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            # Greedy (default) params, NOT None: the EP lockstep path builds all-dummy batches for idle
            # replicas (ep.py) that flow through Sampler.prepare, which reads r.sampling_params.is_greedy
            # on every req. None crashed it (AttributeError). Graph capture + decode padding only use
            # dummy_req for padded_reqs (sampler reads real batch.reqs), so this is otherwise inert.
            sampling_params=SamplingParams(),
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        # GDN-hybrid models capture too: per-seq recurrent-state slots are threaded through static
        # buffers (GDNGraphCapture), so the decode graph replays against the live conv/ssm state.
        _bt_cap = _bt.phase("graph_capture")
        _bt_cap.__enter__()
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
            gdn_state=self.gdn_state,
            cca_state=self.cca_state,
            max_running_req=config.max_running_req,
            cam=self.cam,
            # Qwen4-Exp PLE. Without it the capture-time warmup forward raises inside
            # `Qwen4ExpPLE.forward` ("no staged batch") before the first graph is recorded: the block
            # reads `Context.ple` and REFUSES to run without one, while `_capture_graphs` stages
            # attn/GDN/CCA/CAM metadata and nothing else. None for every other model.
            ple=self.ple_runtime,
        )
        _bt_cap.__exit__()
        # Block-diffusion canvas graphs. Unlike the spec-verify families this needs nothing from the
        # scheduler (no proposer, no aux-capture layers) — the shape is fixed by the checkpoint's
        # canvas_length — so it is captured here, beside the decode graphs, on the engine stream.
        with _bt.phase("canvas_capture"):
            self._capture_canvas_graphs(config)
        # Weight offload, POST-capture gate. `seal()` above ran at :284, before a single graph was
        # captured, so the one thing `ArenaMemPool.assert_clean()` cannot have seen there is its own
        # `alloc_during_capture` branch — the counter only moves while a HIP capture is in flight,
        # which is exactly the window between there and here. An arena allocation served mid-capture
        # bakes a bump-allocated address (never freed, never reclaimable) into a graph that replays
        # for the life of the process; one that MISSES the arena mid-capture cannot fall back at all,
        # because hipMalloc is illegal during capture, and returns NULL for torch to dereference.
        # Re-gating here is what turns both into a boot error. Inert on every serve that does not
        # offload. NOT the last capture in the process — see the method's docstring.
        with _bt.phase("verify_arena_after_capture"):
            self.verify_weight_arena_after_capture()
        # THE REPORT. Emitted at the end of every boot, not behind a flag: it is ~30 log lines once
        # per process, and an instrument that has to be switched on is never on for the run that
        # turned out to matter. `MINISGL_BOOT_TIMELINE_JSON` additionally writes the machine-readable
        # form (rank-suffixed) for a harness to diff across configurations.
        for _line in _bt.timeline().describe().splitlines():
            logger.info_rank0(_line)
        if (_p := _bt.timeline().dump()) is not None:
            logger.info_rank0(f"[boot-timeline] wrote {_p}")

    def verify_weight_arena_after_capture(self) -> None:
        """Re-gate the weight arena on the far side of graph capture. Idempotent, any rank.

        Called TWICE on purpose, and the second call is the load-bearing one. `Engine.__init__` ends
        after the decode and canvas graphs, but three of the five capture families — spec-verify,
        propose, fused-TiDAR and DDTree — are captured from `Scheduler.__init__` (`scheduler.py`
        :636/:666/:684/:793), because they need the proposer, which does not exist here yet. So the
        scheduler calls this again as its last capture-time act. Without that call, the three
        families that dominate a spec serve would run with no arena gate behind them at all, which
        is the same hole `seal()` had — a check placed where the thing it detects cannot occur.
        """
        self._woff.verify_after_capture()

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        # self.dp_cpu_group is the cross-replica gloo group over the dp ranks (one member per replica,
        # this replica's tp-primary). It is built ONLY when dp_size>1 and reserved for EP's
        # all_gather/all_reduce dispatch (the DP-launcher / EP-off path never touches it in the
        # forward loop, so replicas stay independent per-step). None when dp_size=1.
        self.dp_cpu_group = None
        if config.dp_info.dp_size == 1:
            return self._init_single_replica_communication(config)
        return self._init_dp_communication(config)

    def _init_single_replica_communication(
        self, config: EngineConfig
    ) -> torch.distributed.ProcessGroup:
        # Historical single-replica path — UNCHANGED. The whole TP group IS the world.
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            # LONG timeout (not the default 30-min gloo timeout, and NOT distributed_timeout which is a
            # short active-work hang detector): this group carries the per-step control-message sync
            # (scheduler/io.py), where rank1's `broadcast(count, root=0).wait()` LEGITIMATELY blocks
            # indefinitely while the serve is IDLE (rank0 blocks on the ZMQ tokenizer recv and only
            # broadcasts when a request arrives). With the default 30-min timeout a parked serve dies
            # after 30 min of no traffic. A crashed rank0 is caught by the parent's worker-death watch,
            # not this timeout, so a long value is safe.
            tp_cpu_group = torch.distributed.new_group(backend="gloo", timeout=timedelta(days=7))
            assert tp_cpu_group is not None
            if self._ep_over_tp:
                # EP-over-TP: the expert-sharding group IS the TP group (all ranks, since dp=1). Build the
                # nccl EP collective group MoELayer._ep_dispatch reduces over, and reuse the TP gloo group
                # as dp_cpu_group so the scheduler's per-step ep lockstep (all_reduce MAX) has a valid
                # group (trivially agrees — the TP ranks run the same batch in lockstep).
                ep_grp = torch.distributed.new_group(
                    ranks=list(range(config.tp_info.size)), backend="nccl")
                self.ctx.ep = EPCommunicator(
                    group=ep_grp,
                    dp_rank=config.tp_info.rank,
                    dp_size=config.tp_info.size,
                    num_experts=config.model_config.num_experts,
                )
                self.dp_cpu_group = tp_cpu_group
            # Install the custom_ar one-shot all-reduce as the default all_reduce (TP==2 + P2P only;
            # falls back to RCCL otherwise). Graph-safe, and now faster than RCCL at EVERY payload, not
            # just the small decode ones.
            #
            # The slot used to be capped at 8 MB with the note that large prefill all_reduces "exceed
            # the slot and self-fall-back to RCCL". That cap was written when custom_ar read the peer one
            # ELEMENT at a time and so was 2.7x SLOWER than RCCL on big tensors — falling back was the
            # right call. The vectorized (16-byte) peer read inverted that. Measured on this box, TP=2,
            # bf16, hidden 2816 (tools/tp_collective_regime_sweep.py, custom_ar vs RCCL, us/call):
            #
            #     1024 rows ( 5.8 MB)   492.8 vs  581.7   1.18x       <- already under the old cap
            #     2048 rows (11.5 MB)   963.6 vs 1127.6   1.17x       <- fell back to RCCL, needlessly
            #     3200 rows (18.0 MB)  1494.5 vs 1713.7   1.15x
            #     8192 rows (46.1 MB)  3777.8 vs 4361.6   1.15x
            #
            # So the cap was costing ~15% on every prefill collective. Size the slot for the largest
            # forward the engine can actually issue, bounded by MINISGL_CAR_MAX_MIB (default 64) so a
            # very wide model cannot silently reserve an unbounded amount of fine-grained IPC memory.
            # Cost is 2 slots (double-buffered) at that size, per rank.
            car_cap_mib = int(os.environ.get("MINISGL_CAR_MAX_MIB", "").strip() or 64)
            car_max_bytes = min(
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize,
                car_cap_mib * 1024 * 1024,
            )
            # all_gather slot. The gather that matters is LMHead's: it gathers the vocab-SHARDED
            # logits on every forward, so a row is ceil(vocab/tp) elements wide, and the row count is
            # the scored-row count — max_running_req for plain decode, and (num_draft+1) rows per
            # request for a spec VERIFY batch, which is the widest the decode path can issue. A full
            # prefill logprob gather is orders of magnitude larger and deliberately NOT covered: it
            # self-falls-back to RCCL rather than reserving hundreds of MB of fine-grained IPC.
            # Bounded by the same MINISGL_CAR_MAX_MIB cap as the all_reduce slot, for the same reason.
            _vocab_shard = -(-config.model_config.vocab_size // config.tp_info.size)
            _rows = config.max_running_req * (1 + (config.spec_num_draft if config.spec_config else 0))
            car_ag_bytes = min(_rows * _vocab_shard * self.dtype.itemsize, car_cap_mib * 1024 * 1024)
            enable_custom_ar_distributed(
                config.tp_info, tp_cpu_group, car_max_bytes, ag_max_bytes=car_ag_bytes)
        return tp_cpu_group

    def _init_dp_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        # dp_size>1: one GLOBAL world spans every (dp_rank, tp_rank) process so we can carve out both
        # the per-replica TP subgroup (used by this replica's collectives) and the cross-replica DP
        # subgroup (reserved for EP). The single boot rendezvous is the only cross-replica sync when
        # EP is off; after init each replica runs its own queue/forward loop independently.
        dp_size = config.dp_info.dp_size
        tp_size = config.tp_info.size
        global_rank = config.device_index  # dp_rank*tp_size + tp_rank
        world_size = dp_size * tp_size
        # pynccl is CUDA-only and assumes the world IS the TP group; with a global DP world it cannot
        # build the per-replica TP comm, so use torch.distributed (gloo here for CPU control msgs).
        torch.distributed.init_process_group(
            backend="gloo",
            rank=global_rank,
            world_size=world_size,
            timeout=timedelta(seconds=config.distributed_timeout),
            init_method=config.distributed_addr,
        )
        # TP subgroup for THIS replica: the tp_size contiguous global ranks owning the same dp_rank.
        tp_cpu_group = None
        for dp in range(dp_size):
            ranks = list(range(dp * tp_size, (dp + 1) * tp_size))
            grp = torch.distributed.new_group(ranks=ranks, backend="gloo")
            if dp == config.dp_info.dp_rank:
                tp_cpu_group = grp
        assert tp_cpu_group is not None
        # DP subgroup: one tp-rank slice across replicas (for tp_size>1 there are tp_size such groups;
        # each process joins the one matching its tp_rank). The gloo copy is the per-step common-bs
        # lockstep channel (all_reduce(MAX) of real batch size, OUTSIDE the graph). EP additionally
        # builds an nccl (RCCL) copy for the in-graph all_gather/all_reduce — gloo is NOT CUDA-graph-
        # capturable, so the dispatch/combine collectives must use the nccl group.
        for tr in range(tp_size):
            ranks = list(range(tr, world_size, tp_size))
            grp = torch.distributed.new_group(ranks=ranks, backend="gloo")
            if tr == config.tp_info.rank:
                self.dp_cpu_group = grp
        # (EP-over-TP builds its ctx.ep in the single-replica path; this dp>1 method is DP+EP only.)
        if config.enable_ep:
            # DP+EP: new_group must be called on EVERY process for each group (collective construction),
            # so build all tp_size nccl dp-subgroups and keep the one this process belongs to. An nccl
            # group off a gloo world is supported by torch.distributed (the reverse of the single-
            # replica path, which builds a gloo group off an nccl world).
            for tr in range(tp_size):
                ranks = list(range(tr, world_size, tp_size))
                grp = torch.distributed.new_group(ranks=ranks, backend="nccl")
                if tr == config.tp_info.rank:
                    self.ctx.ep = EPCommunicator(
                        group=grp,
                        dp_rank=config.dp_info.dp_rank,
                        dp_size=dp_size,
                        num_experts=config.model_config.num_experts,
                    )
            # Replace the in-graph EP MoE all_reduce (RCCL, ~23.7% of ZAYA decode GPU time) with the
            # custom_ar one-shot P2P all-reduce when there are exactly 2 DP replicas with working P2P.
            # The decode all_reduce tensor is (dp_size*bs, hidden) at a captured bs (<= cuda_graph_max_bs);
            # cap the IPC slot at 8 MB (long eager prefills exceed it and self-fall-back to RCCL on BOTH
            # replicas, staying lockstep). The IPC handshake rides dp_cpu_group (the 2 DP replicas). Only
            # tp_size==1 DP+EP is wired here (ep.dp_size gate); enable_custom_ar_ep no-ops for dp_size!=2.
            # MINISGL_CUSTOM_AG_EP (DEFAULT ON) additionally moves the fused EP dispatch all_gather onto
            # the custom one-shot P2P all_gather (all_gather_p2p) — moving the residual EP RCCL off the
            # decode graph. It is DECOUPLED from AR: the IPC infra is set up if EITHER is requested, and
            # each collective uses custom vs RCCL per its own flag. Both DEFAULT ON: a safe no-op on non-EP
            # serves (dp_size!=2 or ctx.ep is None) and falls back to RCCL if the baked custom_ar lacks the
            # op or P2P is unavailable. With both on, the EP decode graph carries ZERO ncclDevKernel.
            want_ar = os.environ.get("MINISGL_CUSTOM_AR_EP", "1") != "0"
            want_ag = os.environ.get("MINISGL_CUSTOM_AG_EP", "1") != "0"
            if (want_ar or want_ag) \
                    and config.tp_info.size == 1 and self.ctx.ep is not None \
                    and self.dp_cpu_group is not None:
                car_max_bytes = min(
                    dp_size * config.max_forward_len * config.model_config.hidden_size
                    * self.dtype.itemsize,
                    8 * 1024 * 1024,
                )
                enable_custom_ar_ep(
                    self.ctx.ep, self.dp_cpu_group, car_max_bytes,
                    enable_ar=want_ar, enable_ag=want_ag, ag_max_bytes=car_max_bytes,
                )
        return tp_cpu_group

    @staticmethod
    def _dummy_like(t: torch.Tensor, device: torch.device) -> torch.Tensor:
        """A plausible dummy value for one parameter, in ITS OWN dtype.

        `--use-dummy-weight` boots without a checkpoint, and it used to be a bare `randn_like`, which
        raises `NotImplementedError: "normal_kernel_cuda" not implemented for 'Byte'` on any model
        with a packed-integer parameter — so the one path designed to work WITHOUT a checkpoint was
        the one path a quantized checkpoint could not use. Only the non-float arms are new; float
        parameters keep the exact `randn_like` every existing dummy boot has always had.

        The integer split follows `tests/qwen4exp_gpu_forward_test.py::_fill_random`, which was
        validated against real NVFP4 kernels: a uint8 blob is packed E2M1 pairs and ANY byte is a
        legal pair, so randomise it; wider integer tensors are index / id / hash constants
        (`layer_multipliers`, expert maps) where a random value is not "dummy", it is wrong — and
        wrong in the out-of-bounds-index way, not the plausible-noise way.
        """
        if t.dtype == torch.uint8:
            return torch.randint(0, 256, t.shape, dtype=torch.uint8, device=device)
        if not t.is_floating_point():
            return torch.zeros_like(t, device=device)
        return torch.randn_like(t, device=device)

    def _build_weight_stream_tier(self, config: EngineConfig):
        """The THIRD weight tier, or `(None, ())`. See `weights/stream_tier.py`.

        Which layers: the LAST `--weight-offload-stream-layers` MoE layers of the decoder. Last,
        because the plan's greedy device fill walks `(-priority, declaration index)` — it takes the
        FIRST layers onto the card — so streaming the tail keeps the two orders from fighting over
        the same layers. The MTP draft head owns a `MoELayer` too and is never a candidate: it is
        re-read every draft step, its granule shapes differ from a decoder layer's, and aliasing it
        onto the tier's shared buffers would read the wrong bytes.

        Every refusal here is loud. A stream tier that silently does not engage is the worst outcome
        available: the boot then tries to keep 70 GiB of experts resident and dies in
        `load_state_dict` with an OOM that names nothing.
        """
        n = int(getattr(config, "weight_offload_stream_layers", 0) or 0)
        if n <= 0:
            return None, ()
        if config.use_dummy_weight:
            raise ValueError(
                "--weight-offload-stream-layers with --dummy-weight: there is no checkpoint to "
                "re-read, so a streamed layer would serve whatever the donor layer left behind."
            )
        if config.cuda_graph_max_bs is None or config.cuda_graph_max_bs > 0:
            raise ValueError(
                "--weight-offload-stream-layers requires --cuda-graph-max-bs 0. The per-forward "
                "expert gather does file I/O, allocates and synchronizes inside MoELayer.forward, "
                "and all three are illegal under HIP graph capture. Capturing it would bake one "
                "batch's expert rows into every replay -- plausible logits, wrong weights, no error."
            )
        from minisgl.models import expert_row_source
        from minisgl.weights.plan import _path_layer_index, is_mtp_path
        from minisgl.weights.moe_interpose import discover_moe_layers
        from minisgl.weights.stream_tier import ExpertStreamTier

        source = expert_row_source(config.model_path, self.device, config.spec_algorithm)
        if source is None:
            raise ValueError(
                f"--weight-offload-stream-layers {n} was asked for, but this checkpoint family has "
                f"no per-expert row source (models/weight.py::expert_row_source). Streaming a whole "
                f"1.4 GiB stack per layer per token is not a serve, so this refuses rather than "
                f"falling back to it."
            )
        candidates = [
            i
            for p, _ in discover_moe_layers(self.model)
            if not is_mtp_path(p)
            for i in (_path_layer_index(p),)
            if i is not None
        ]
        if n > len(candidates):
            raise ValueError(
                f"--weight-offload-stream-layers {n} exceeds the {len(candidates)} offloadable MoE "
                f"layers this model has."
            )
        ids = tuple(sorted(candidates)[-n:])
        logger.info_rank0(
            f"weight offload: stream tier will take the last {n} MoE layer(s) {ids[0]}..{ids[-1]}; "
            f"they are excluded from the placement plan."
        )
        return ExpertStreamTier(source, device=self.device, log=logger.info_rank0), ids

    def _load_weight_chunked(self, config: EngineConfig) -> bool:
        """Stage B: load + finalize + PLACE the checkpoint one chunk at a time. True if it ran.

        WHY THE ENGINE AND NOT A HARNESS. `Engine.__init__`'s load is
        `load_state_dict(dict(load_weight(...)))`: the generator is drained into one dict, so every
        tensor of the checkpoint is live on the card simultaneously and the peak is the whole
        checkpoint. That ceiling is what stops `RadixArk/Qwen3.8-Flash-Next-NVFP4` at 4 layers of 48
        — it OOMs INSIDE `load_state_dict`, before the Stage-A bake that exists to make it fit has
        run at all. No arena size fixes that, because the bake is downstream of the OOM.

        The chunked driver interleaves the two: each chunk is read, filled, `post_load`ed and handed
        to the sink, which bakes the host-placed layers into the arena and drops their device
        originals before the next chunk is read. The steady state is one chunk (1.465 GiB here), and
        a host-placed layer costs zero VRAM the moment its chunk closes.

        THREE CONDITIONS, ALL REQUIRED, AND EACH IS A DIFFERENT KIND OF "NO":
          * `use_dummy_weight` — there is no checkpoint to chunk. The dummy path fabricates from the
            model's own state_dict and is unaffected by any of this.
          * no chunk enumeration for this family (`chunked_weight_source` -> None) — the shipped
            one-shot load is correct and is what every other model here uses.
          * the Stage-A session is disabled — no host tier, so the sink would have nothing to bake.
            Chunking alone would still lower the load peak, but it would do so on a path no test
            covers for models that have never needed it; turning it on for them is a separate change
            with its own A/B, not a side effect of this one.

        The trailing `model.post_load()` is NOT redundant with the per-chunk ones: the chunks
        finalize only the ops they complete (the MoE stacks), and the non-expert body's containers
        are finalized here. `BaseOP._post_load_done` is what keeps that call from re-entering the
        layers the chunks already did — a second `post_load` on a quantized MoE container reads
        buffers its first one deleted.
        """
        if config.use_dummy_weight or not self._woff.enabled:
            return False
        from minisgl.models import chunked_weight_source
        from minisgl.weights.stage_b import ChunkedWeightLoader

        from minisgl.weights import boot_timeline as _bt

        # SEPARATE from `stage_b_run`: this enumerates the 196 shards and runs the checkpoint-wide
        # NVFP4 header pre-pass (196 `safe_open`s). That is real seconds before a single tensor is
        # read, and it is charged to the wrong phase if it is folded into the chunk loop.
        with _bt.phase("stage_b_enumerate_prepass"):
            source = chunked_weight_source(
                config.model_path, self.device, spec_algorithm=config.spec_algorithm
            )
        if source is None:
            return False
        chunks, stream = source
        loader = ChunkedWeightLoader(
            self.model,
            cast=lambda k, v: cast_checkpoint_tensor(k, v, self.dtype),
            sink=self._woff.chunked_sink(),
            log=logger.debug_rank0,
        )
        with _bt.phase("stage_b_run"):
            ledger = loader.run(chunks, stream)
        with _bt.phase("stage_b_trailing_post_load"):
            self.model.post_load()
        # KEPT, and MARKED. A boot log proves nothing an A/B can diff, and this repo's rule is that a
        # vanished arm shows up as a missing `engaged()` line rather than as a number nobody notices.
        # The ledger is what a harness asserts on: chunks, keys filled, and the peak device
        # allocation that is the entire claim being made here.
        self.stage_b_ledger = ledger
        from minisgl._hip_engage import engaged

        engaged("weight_offload.stage_b_chunked_load")
        logger.info_rank0(f"weight offload: {ledger.describe()}")
        return True

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: self._dummy_like(v, self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: cast_checkpoint_tensor(k, v, self.dtype)
                for k, v in load_weight(
                    config.model_path, self.device, spec_algorithm=config.spec_algorithm
                )
            }

    def _rec_snapshot_store_bytes(self, config: EngineConfig) -> int:
        """Bytes the recurrent-radix SNAPSHOT store will consume, reserved up front.

        This store holds cloned recurrent state on radix nodes so a request that branches inside a
        shared prefix can resume instead of re-prefilling from zero. It is filled DURING serving, long
        after the KV pool is sized — exactly like the recurrent-state slots, draft model and graph
        buffers above, all of which are already subtracted here for the same reason.

        It was missing from that list, and the omission is why --memory-ratio 0.85 over-committed: the
        pool claimed memory the snapshots then took, PyTorch's reserved climbed to 93% of the card,
        device-free hit 0 during the very first prefill, and any allocation the caching allocator
        could not serve from a cached block OOM'd. Measured at 0.85: alloc 13.3 GiB / reserved 15.1 GiB
        / devfree 0 with a 16.3 GiB card. Concurrency did not consume this memory — it only changed
        the allocation shapes, which is why 4x57k died where a single 123k prompt did not.

        One snapshot is one slot's worth of recurrent state, and the cap is a VRAM budget shared with
        the scheduler (MINISGL_GDN_RADIX_SNAP_BUDGET_GIB), so both agree on the number. 0 when the
        model has no recurrent state or the recurrent radix is off."""
        # THE gate: ask the shared resolver whether a snapshot store will exist at all, instead of
        # re-deriving it here from gdn_radix + "per-slot bytes > 0". That local guess was WRONG for a
        # SWA hybrid — per_slot counts the SWA ring, so it is > 0 with zero recurrent state, while
        # gdn_radix (a GDN flag) says nothing about the SWA path, which the scheduler gates on
        # MINISGL_SWA_RADIX. Result on gemma-4-26B-A4B TP=2: 0.98 GiB reserved on a 16 GB card for a
        # store the scheduler had already decided not to build — ~100k KV tokens bought and thrown
        # away. It also could not see `--cache-type naive`, which forces the same outcome.
        # resolve_prefix_cache() is the single source of truth for both sides now (engine/config.py):
        # same answer as before for GDN/CCA, zero for dense/MHA/MLA and for any hybrid whose snapshot
        # path is switched off, and — with SWA-radix now default ON — a reservation for SWA hybrids
        # that is matched by a store the scheduler really does build and fill.
        if not resolve_prefix_cache(config).snapshot_kind:
            return 0
        # WITHOUT the ReplaySSM ring: a snapshot is what clone_slot() copies, which is conv+ssm only.
        # The ring is per LIVE slot, never cloned onto a radix node, so folding it into per_slot would
        # inflate this reservation by the ring's whole 25% for state that is never stored here.
        # For snapshot_kind == "swa" this same expression yields the SWA ring's per-sequence bytes,
        # which is what SWAWindowSnapshotter.clone() copies (all sliding layers x min(boundary, W)
        # positions of K+V). Under spec the ring stride is window + num_draft + 1 while a snapshot is
        # only `window` wide, so this over-reserves by the spec block — deliberately, on the side that
        # cannot OOM. A model is GDN xor CCA xor SWA, so exactly one family contributes here.
        per_slot = (self._recurrent_state_bytes(config, replay_ring=False)
                    // max(1, config.max_running_req + 2))
        if per_slot <= 0:
            return 0
        # HOST TIER: the snapshots themselves now live in pinned host RAM, so the only DEVICE
        # memory this feature needs is the staging ring — one contiguous frame that the gather
        # writes and the D2H reads. At depth 1 that is 16.4 MiB instead of 0.32 GiB, handing
        # ~64k KV tokens back to the pool (141,408 -> ~205,248 on the 35B at TP=2/CONC=4). The
        # snapshot is a fixed size regardless of context length and is touched once, off the decode
        # path, so PCIe-vs-recompute is the relevant comparison and it is not close.
        if host_tier_enabled():
            return stage_ring_depth() * per_slot
        # DERIVED, not a magic GiB: the store's job is to hold every live sequence's ladder plus its
        # end snapshot, which is exactly (ladder + 1) * max_running_req entries.
        #
        # A NOTE ON THE OLD JUSTIFICATION HERE: this comment used to claim "cap 12 and cap 23
        # produced the SAME hit count and the same TTFT ... while the difference cost 36,704 KV pool
        # tokens", citing tools/rec_radix_ab.sh. The archived fixtures for that script do not
        # contain that comparison: both store legs booted at the SAME cap 23 and varied only the
        # LADDER DEPTH (0 vs 4), which is what came back identical. The "cap 12" leg is the `tuned`
        # arm, which also changes ladder AND --max-prefill-length, so it is not a cap-only A/B; the
        # 36,704 figure traces to commit 209e4aaa as 11 x 16.4 MiB arithmetic, not a measurement.
        # What IS supported: interior resume points buy nothing. Cross-request CAP depth was never
        # measured — and with the store in host RAM it stops costing pool tokens, so it is now worth
        # measuring properly.
        live = self._rec_snap_live_snapshots(config)
        env_gib = os.environ.get("MINISGL_GDN_RADIX_SNAP_BUDGET_GIB")
        budget = int(float(env_gib) * (1 << 30)) if env_gib else 0
        return max(budget, live * per_slot)

    def _build_snapshot_host_arena(self, config: EngineConfig) -> None:
        """Allocate the pinned host arena and point the live state cache at it.

        Best-effort by design. A multi-GiB `cudaHostRegister` can be slow or can fail outright under
        a cgroup memory limit, and this is a CACHE — on any failure we log loudly and leave the
        state cache on its legacy device-clone path, which is byte-identical behaviour, just with
        the old VRAM cost. Never crash a serve for a cache.
        """
        self.snapshot_host_arena = None
        if not host_tier_enabled():
            return
        if resolve_prefix_cache(config).snapshot_kind != "recurrent":
            return  # SWA frames are 100 MiB and its clone is a gather; deferred (Stage 1c/2).
        cache = self.gdn_state if self.gdn_state is not None else self.cca_state
        if cache is None or getattr(cache, "_stage", None) is None:
            return
        frames = self._rec_snap_host_arena_frames(config)
        layout = cache.snapshot_layout
        try:
            t0 = time.perf_counter()
            arena = PinnedFrameArena(layout, frames)
            cache.attach_host_arena(arena)
            self.snapshot_host_arena = arena
            logger.info(
                f"recurrent-radix snapshot store: HOST tier — {arena.stats()} "
                f"({layout.nbytes / (1 << 20):.2f} MiB/frame, pinned in "
                f"{time.perf_counter() - t0:.2f}s); device staging "
                f"{stage_ring_depth()} x {layout.nbytes / (1 << 20):.2f} MiB"
            )
        except (RuntimeError, MemoryError) as e:
            logger.warning(
                f"recurrent-radix snapshot store: pinning {frames} x "
                f"{layout.nbytes / (1 << 20):.1f} MiB FAILED ({e}); falling back to device clones. "
                "The KV pool was already sized for the host tier, so it is now oversized relative "
                "to the store — restart with MINISGL_REC_SNAP_HOST=0 for a consistent boot."
            )
            self.snapshot_host_arena = None

    def _rec_snap_host_arena_frames(self, config: EngineConfig) -> int:
        """Pinned host frames to allocate for the snapshot store.

        `live` is the working set that must never be dropped. The extra `ladder * max_running` term
        covers snapshots held in `_pending_rec_snap` / `_rec_snap_ladder` that have NOT yet reached
        the radix store — today those are device clones sitting OUTSIDE both `_enforce_rec_cap` and
        this reservation, i.e. ~0.256 GiB of unreserved VRAM. They move to host too, which is why
        the measured free-VRAM improvement will exceed the reservation delta, and why the reclaim
        must be validated on POOL TOKENS from the boot log rather than a mem_get_info delta.

        The 1.25 is headroom for the attach transition, where a snapshot is momentarily referenced
        by both the ladder list and the radix node.
        """
        from .config import snapshot_ladder_depth

        live = self._rec_snap_live_snapshots(config)
        inflight = snapshot_ladder_depth(config) * max(1, config.max_running_req)
        env = os.environ.get("MINISGL_REC_SNAP_HOST_GIB")
        frames = -(-int(1.25 * (live + inflight)) // 1)
        if env:
            per_slot = (self._recurrent_state_bytes(config, replay_ring=False)
                        // max(1, config.max_running_req + 2))
            if per_slot > 0:
                frames = max(1, int(float(env) * (1 << 30)) // per_slot)
        return max(1, frames)

    @staticmethod
    def _rec_snap_live_snapshots(config) -> int:
        """Snapshots the LIVE working set needs: each concurrent sequence's ladder plus its end
        boundary. The one number both the engine's reservation and the scheduler's LRU cap derive
        from, so they cannot disagree about how much VRAM this store is allowed.

        The ladder term means different things to the two snapshot kinds, and defaults accordingly:

        * RECURRENT (GDN/CCA): interior resume points. A sequence really does hold `ladder` of them
          plus its end boundary, so the default 4 is the working set and dropping it makes the ladder
          thrash against itself.
        * SWA: there IS no interior ladder — a window snapshot is taken at the page-aligned prefix
          boundary and nowhere else, so the live set is one per concurrent sequence and the default
          is 0. Anything above that buys purely CROSS-REQUEST reuse depth, which is expensive here in
          a way it is not for GDN: a Gemma4 window snapshot is 100 MiB (W=1024 x 25 sliding layers x
          4 local kv heads x 256 head_dim x K+V x 2 B) against 16.4 MiB for the 35B's recurrent
          state, so inheriting GDN's ladder of 4 cost 1.95 GiB — 204,800 KV-pool tokens, 59% of
          Gemma4's whole pool — to buy reuse depth the GDN A/B measured as worth nothing on this box
          (cap 12 and cap 23 produced the same hits and the same TTFT).

        MINISGL_GDN_RADIX_SNAP_LADDER overrides either, so a prefix-sharing-heavy deployment can buy
        the depth back explicitly and pay the KV tokens knowingly."""
        return (snapshot_ladder_depth(config) + 1) * max(1, config.max_running_req)

    @staticmethod
    def _replay_ring_bytes(mc, num_slots: int, num_v_heads: int, ssm_itemsize: int) -> int:
        """Bytes of the ReplaySSM ring GDNStateCache allocates next to ssm_state (0 when the baked
        kernel package has no replay op, i.e. nothing will be allocated)."""
        try:
            import gdn_hip as gdn

            from minisgl.kvcache.gdn_state import resolve_ring_len
            if not hasattr(gdn, "gdn_decode_conv_gated_replay"):
                return 0
            return mc.num_gdn_layers * gdn.replay_ring_bytes(
                num_slots, num_v_heads, mc.linear_value_head_dim, mc.linear_key_head_dim,
                resolve_ring_len(gdn), itemsize=ssm_itemsize)
        except Exception:   # see GDNStateCache: the extension may fail to load, not just to import
            return 0

    def _recurrent_state_bytes(self, config: EngineConfig, replay_ring: bool = True) -> int:
        """Bytes the fixed GDN/CCA recurrent-state caches will consume (they are allocated AFTER the
        KV pool). Mirrors GDNStateCache / CCAStateCache buffer shapes so _determine_num_pages can
        reserve them up front. Returns 0 for models with no recurrent state (dense / MHA / MLA)."""
        mc = config.model_config
        tp = config.tp_info.size
        num_slots = config.max_running_req + 2  # +1 NULL + 1 dummy, matches the cache ctors
        total = 0
        if getattr(mc, "is_gdn_hybrid", False):
            # conv_state (num_gdn_layers, num_slots, conv_dim, conv_kernel-1) fp32 +
            # ssm_state  (num_gdn_layers, num_slots, num_v_heads, head_v_dim, head_k_dim) ssm_dtype
            conv_dim = div_even(mc.gdn_conv_dim, tp)
            conv = mc.num_gdn_layers * num_slots * conv_dim * (mc.linear_conv_kernel_dim - 1) * 4
            ssm_itemsize = 4 if os.environ.get("MINISGL_SSM_BF16", "1") == "0" else 2
            num_v_heads = div_even(mc.linear_num_value_heads, tp)
            ssm = (
                mc.num_gdn_layers * num_slots * num_v_heads
                * mc.linear_value_head_dim * mc.linear_key_head_dim * ssm_itemsize
            )
            # ReplaySSM ring — a per-slot store allocated by GDNStateCache alongside ssm_state, so it
            # comes out of the same budget. Sized by the kernel package's OWN estimator, which is
            # derived from the shapes its allocator uses, so this cannot drift from what gets
            # allocated (the failure mode is silent VRAM over-commit; cf. the recurrent-radix
            # snapshot store, 71e322bf). L*(K+V)/(V*K) of ssm_state = +25% at the default L=16.
            total += conv + ssm
            if replay_ring:
                total += self._replay_ring_bytes(mc, num_slots, num_v_heads, ssm_itemsize)
        if getattr(mc, "is_cca_hybrid", False):
            # conv_states (num_cca_layers, num_slots, conv_dim/tp, conv_kernel) fp32 +
            # prev_hs     (num_cca_layers, num_slots, hidden_size) fp32  (prev_hs stays FULL hidden)
            conv = mc.num_cca_layers * num_slots * div_even(mc.cca_conv_dim, tp) * mc.cca_conv_width * 4
            prev = mc.num_cca_layers * num_slots * mc.hidden_size * 4
            total += conv + prev
        if getattr(mc, "is_swa_hybrid", False):
            # SWA ring KV pool (allocated AFTER the main pool): 2 (K+V) * num_swa_layers * num_slots *
            # STRIDE * local_kv_heads * head_dim * kv_dtype.itemsize. num_slots as above. Stride is the
            # per-seq ring stride = window + spec block (must match the pool sizing above); no spec => W.
            # The ring's geometry is the SLIDING one, which need not be the model-wide head_dim /
            # num_kv_heads (Gemma4: 256/8 ring vs 512/2 main pool) — the same _swa_kv_geometry the
            # allocation uses, so the reserve can never describe a different pool than the one built.
            swa_head_dim, swa_num_kv_heads = _swa_kv_geometry(mc)
            local_kv = div_even(swa_num_kv_heads, tp, allow_replicate=True)
            swa_stride = mc.sliding_window + _swa_ring_block(mc, config.spec_config)
            total += (
                2 * mc.num_swa_layers * num_slots * swa_stride
                * local_kv * swa_head_dim * self.kv_dtype.itemsize
            )
        return total

    def _moe_prefill_stage_bytes(self, budget_left: int, cache_per_page: int,
                                 page_size: int, max_running_req: int) -> int:
        """Bytes to reserve for the MoE prefill expert-staging slab (`weights/prefill_stage.py`).

        Sized from the LIVE seams — the largest single host-resident MoE layer — because that is
        exactly what one staged launch copies, and the slab is reused by every layer in turn.

        RESERVED HERE, ALLOCATED LATER, which is the whole reason this is a method and not a
        `torch.empty` at the use site. A device buffer this size taken out of the (1-memory_ratio)
        slack after the pool is sized is how the 2026-09-15 21:01 boot died: 48 MiB unavailable on a
        16 GiB card, both ranks, mid-prefill, with four requests in flight.

        THE GUARD IS A POLICY AND IS STATED AS ONE. The slab competes directly with the KV pool, so
        this is a capacity-for-throughput trade and the floor has to be something operational rather
        than a ratio. The first cut required the pool to stay larger than the slab, and on the
        served arm that declined it by arithmetic — slab 0.665 GiB against a 1.05 GiB pool — which
        would have made the whole mechanism inert with nothing in the log but one DECLINED line.

        The floor used instead is FULL CONCURRENCY AT A USABLE CONTEXT: whatever remains must still
        hold `max_running_req` requests of `_STAGE_MIN_CTX` tokens each. A pool that cannot do that
        is genuinely too small to spend on a slab; one that can is better spent on it, because a
        large context you cannot afford to PREFILL is worth less than a smaller one you can — at
        4.0 tok/s a 2k prompt costs nine minutes, so the context ceiling was never the binding
        constraint on this arm.

        Deliberately NOT keyed on `max_seq_len`: the engine derives that from the pool it is sizing
        here (`min(checkpoint, num_pages*page_size)`), so a floor expressed in it is circular.

        Both outcomes print the token counts on either side of the trade, because "the pool shrank"
        is the one consequence an operator must never have to infer from a byte figure.
        """
        _STAGE_MIN_CTX = 4096  # tokens per concurrent request the pool must still afford
        # MEMOISED, and the allocation site reads this value back rather than recomputing. The
        # guard below depends on `budget_left`, which the allocation site does not have; a second
        # independent derivation there is exactly how a reserve/allocate pair drifts into
        # allocating a slab the KV pool never paid for.
        if (prev := getattr(self, "_moe_stage_reserved", None)) is not None:
            return prev

        def _decide() -> int:
            if not self._woff.enabled:
                return 0
            # DEFAULT OFF, and the reason is a measurement, not caution. Wired, fired and measured
            # end-to-end on the served Qwen4-Exp arm 2026-09-15:
            #   staging ON : 1820 prompt tokens / 511.2 s = 3.56 tok/s, moe_stage[prefill] engaged
            #   staging OFF: 2243 prompt tokens / 558.6 s = 4.02 tok/s
            # i.e. no improvement, while the slab costs 0.67 GiB and takes the KV pool from 207,456
            # to 91,184 tokens. (The two runs used different prompts, so 0.89x is NOT a controlled
            # A/B and this does not claim staging is slower -- only that the benefit is
            # unmeasurable while the capacity cost is certain.)
            #
            # WHY it does nothing, confirmed by the same boot's hostprof: prefill is
            # fwd_launch=83% / gpu_wait=1%, so the forward is not waiting on the GPU or the PCIe
            # link at all. Expert reads were ~0.7% of prefill by P1's own measured bandwidth
            # (28.93 GB/s card 0), and removing 0.7% is not observable. The mechanism is correct and
            # kept for a serve whose prefill is actually link-bound; this one is not.
            # `or "0"` after `.strip()`, NOT a get() default: compose passes every knob as
            # "${VAR:-}", so the variable IS SET to the empty string when the operator did not set
            # it, and `get(name, "0")` returns "" rather than the default. This exact trap is
            # already recorded in this repo; the first cut of this line fell into it and shipped a
            # boot that took the 0.67 GiB slab while claiming the feature was off.
            if (os.environ.get("MINISGL_MOE_PREFILL_STAGE", "").strip() or "0") == "0":
                return 0
            from minisgl.weights import prefill_stage
            try:
                need = prefill_stage.per_layer_host_bytes(self.model)
            except Exception as e:  # noqa: BLE001
                logger.info_rank0(
                    f"MoE prefill staging: not sized ({e!r}); prefill keeps the host read.")
                return 0
            if need <= 0:
                return 0  # no host-resident MoE layer on this rank -> nothing to stage
            def _tokens(nbytes: int) -> int:
                return max(0, nbytes) // max(1, cache_per_page) * max(1, page_size)

            before, after = _tokens(budget_left), _tokens(budget_left - need)
            floor = max_running_req * _STAGE_MIN_CTX
            if after < floor:
                logger.info_rank0(
                    f"MoE prefill staging DECLINED: the slab needs {mem_GB(need)} and would leave "
                    f"the KV pool at {after:,} tokens, under the {floor:,} needed for "
                    f"{max_running_req} concurrent requests of {_STAGE_MIN_CTX:,}. Prefill keeps "
                    f"the in-place host read (measured 4.0 tok/s on the Qwen4-Exp arm); shrink the "
                    f"device tier or raise --memory-ratio to afford it."
                )
                return 0
            logger.info_rank0(
                f"MoE prefill staging: taking {mem_GB(need)} for the slab; KV pool "
                f"{before:,} -> {after:,} tokens ({max_running_req} x "
                f"{after // max(1, max_running_req):,}). This is the capacity-for-prefill trade — "
                f"the pool shrinks so a prefill chunk stops re-reading the host expert set."
            )
            return need

        self._moe_stage_reserved = _decide()
        return self._moe_stage_reserved

    def _ple_runtime_bytes(self, config: EngineConfig) -> int:
        """Bytes the Qwen4-Exp PLE runtime will consume on the DEVICE. Zero for every other model.

        Same reason as `_recurrent_state_bytes`: the runtime is built after the KV pool is sized, so
        without this its staging buffers come out of the (1-memory_ratio) slack. They are not small
        — two `max_extend_tokens x ple_embed_dim` buffers, ~126 MiB at the 8192-token default —
        which on a 16 GB card is real KV pool. The formula lives in `ple/runtime.py` next to the
        allocation it mirrors."""
        mc = config.model_config
        if not getattr(mc, "ple_layer_ids", ()):
            return 0
        from minisgl.ple import ple_device_bytes

        return ple_device_bytes(
            mc,
            num_slots=config.max_running_req + 2,
            max_tokens=_ple_stage_tokens(config),
            dtype=self.dtype,
        )

    def _draft_model_bytes(self, config: EngineConfig) -> int:
        """Bytes the speculative DRAFT model (EAGLE3 / DFlash) will consume. The proposer loads it in
        ``Scheduler.__init__`` — AFTER the engine has built and the KV pool is allocated — so, like the
        recurrent state, it is NOT part of ``model_memory`` and must be reserved up front or it eats the
        ``(1-memory_ratio)`` slack and OOMs at proposer build / first request. Returns 0 for no-spec and
        for proposers with no separate checkpoint (n-gram / target-embedded MTP).

        Estimate = sum of the cached checkpoint's ``*.safetensors`` bytes (a conservative proxy: on-disk
        bf16 >= an fp8 runtime). Resolved from the HF cache without downloading under ``HF_HUB_OFFLINE``.
        Override with ``MINISGL_DRAFT_RESERVE_GB=<float>`` (e.g. to correct an fp8-runtime vs bf16-on-disk
        mismatch, or to reserve when the checkpoint can't be sized here)."""
        # Truthiness, NOT `is not None`: compose forwards an unset variable as the empty string, so
        # `is not None` would accept "" and crash in float(""). Empty means "not set".
        override = (os.environ.get("MINISGL_DRAFT_RESERVE_GB") or "").strip()
        if override:
            return int(float(override) * (1 << 30))
        sc = config.spec_config
        draft_path = getattr(sc, "draft_model_path", None) if sc is not None else None
        if not draft_path:
            return 0
        try:
            from minisgl.utils import download_hf_weight

            folder = download_hf_weight(draft_path)
            total = 0
            for name in os.listdir(folder):
                if name.endswith(".safetensors"):
                    total += os.path.getsize(os.path.join(folder, name))  # follows symlink into blobs/
            # MINISGL_DFLASH_QUANT weight-only quantizes the draft linears. On-disk is bf16, so scale
            # by what the mode actually keeps. This MUST track the mode: a flat //2 was right while
            # fp8/int8 were the only options, but it over-reserves an nvfp4 drafter by ~0.8 GiB —
            # which on a 16 GB card is the whole margin a 4-bit drafter exists to buy, so the reserve
            # alone would leave the KV pool unsizable and the boot would fail anyway.
            _mode = (os.environ.get("MINISGL_DFLASH_QUANT", "") or "").strip().lower()
            if _mode in ("none", "bf16", "off"):
                _mode = ""   # explicit "no quant" sentinel — reserve the full bf16 size
            if _mode in ("fp8", "int8"):
                total //= 2  # 1 byte/param vs bf16's 2
            elif _mode == "nvfp4":
                # 4-bit codes + an fp16 per-16-element group scale = 4.5 bits/param, but the
                # sensitive leaves stay fp8 (see `_load_draft_weights`), so budget 6 bits = 3/8 of
                # bf16. Deliberately a slight OVER-estimate of the measured 1.61 GiB: under-reserving
                # does not fail at boot, it fails later in the drafter's eager forward.
                total = total * 3 // 8
            # TP-SHARDED drafter: the decoder-layer linears (q/k/v/o, gate/up/down) are split across
            # ranks (models/draft_linear.py), so each card holds roughly 1/tp of them. NOT everything
            # shards — `fc` (num_aux*hidden -> hidden), every norm, and any tensor whose dimension
            # does not divide tp stay REPLICATED — so scale only the sharded fraction and keep the
            # rest whole. 0.20 replicated is a deliberate OVER-estimate (Muse-Glimmer's fc is ~9% of
            # the drafter): over-reserving merely costs KV pool, while under-reserving does not fail
            # at boot, it fails later inside the drafter's eager forward. Correct it exactly with
            # MINISGL_DRAFT_RESERVE_GB if a checkpoint's split is known.
            from minisgl.distributed import get_tp_info

            _tp = get_tp_info().size
            if _tp > 1:
                _REPLICATED = 0.20
                total = int(total * (_REPLICATED + (1.0 - _REPLICATED) / _tp))
            # + working-set headroom for the drafter's EAGER forward (DFlash denoise / EAGLE3 step) and
            # its spec-verify transient. This is NOT covered by the captured-graph reserve (propose runs
            # eager for DFlash/EAGLE3), and it is the ~340 MB that OOMs a tight 27B+DFlash boot at warmup
            # (past capture). Default 0.7 GB; tune via MINISGL_DRAFT_RESERVE_MARGIN_GB (0 disables).
            # `or "0.7"` AFTER strip, not a get() default: compose forwards an unset variable as the
            # EMPTY STRING, which satisfies the default and then explodes in float("").
            _m = (os.environ.get("MINISGL_DRAFT_RESERVE_MARGIN_GB") or "").strip() or "0.7"
            margin = int(float(_m) * (1 << 30))
            return total + margin
        except Exception as e:  # noqa: BLE001 — sizing must never block boot
            logger.warning_rank0(
                f"Draft-model reserve: could not size {draft_path} ({e}); reserving 0 — set "
                "MINISGL_DRAFT_RESERVE_GB to reserve explicitly"
            )
            return 0

    def _graph_capture_bytes(self, config: EngineConfig, free_memory: int) -> int:
        """Bytes the CUDA-graph STATIC buffers will consume. Graph capture runs AFTER the KV pool is
        allocated (decode graphs at the end of ``Engine.__init__``; spec-verify graphs later, from the
        scheduler once the proposer is built), so its buffers come out of the ``(1-memory_ratio)`` slack
        unless reserved here — the capture-time / first-request OOM this prevents. Mirrors the
        ``GraphCaptureBuffer`` / ``VerifyCaptureBuffer`` shapes in ``engine/graph.py``.

        Returns 0 when capture is off (``cuda_graph_max_bs == 0`` — ``_adjust_config`` has already zeroed
        it for the non-MLA/non-CCA spec case where verify graphs aren't captured). Conservative by design
        (round-up + assume spec-verify captures last_hidden/aux); add headroom with
        ``MINISGL_GRAPH_RESERVE_MARGIN_GB`` or replace the whole estimate with ``MINISGL_GRAPH_RESERVE_GB``.
        General: works for dense / MLA / GDN / CCA, spec and non-spec."""
        full = os.environ.get("MINISGL_GRAPH_RESERVE_GB")
        if full:  # non-empty (empty env string is ignored)
            return int(float(full) * (1 << 30))
        if config.cuda_graph_max_bs == 0:
            return 0
        # Reproduce the captured batch-size set the GraphRunner will pick (same inputs it is handed).
        from .graph import _determine_cuda_graph_bs

        bs_list = _determine_cuda_graph_bs(
            config.cuda_graph_bs, config.cuda_graph_max_bs, free_memory
        )
        if not bs_list:
            return 0
        mc = config.model_config
        vocab = mc.vocab_size
        hidden = mc.hidden_size
        f32 = 4
        i32 = 4
        dt = self.dtype.itemsize
        max_bs = max(bs_list)
        # Decode graphs share ONE GraphCaptureBuffer sized at max_bs (logits [max_bs, vocab] fp32 +
        # three [max_bs] int32 buffers); the shared graph pool holds the largest forward's activations.
        total = max_bs * vocab * f32 + 3 * max_bs * i32
        total += _GRAPH_ACT_MULT * max_bs * hidden * dt
        # Spec-verify graphs (MLA / CCA today; GDN when enabled): ONE VerifyCaptureBuffer at
        # (verify_bs, qlen=K+1). verify_bs is the graph bs-set capped at max_running_req (scheduler).
        # Reaching here with spec set implies verify capture is on (the unsupported case was zeroed in
        # _adjust_config). Assume last_hidden + aux — draft-head proposers need them (n-gram over-
        # reserves harmlessly). These are the big buffers for spec (T = verify_bs*(K+1) rows of vocab).
        sc = config.spec_config
        if sc is not None:
            # ADAPTIVE VERIFY WIDTH (spec/width.py): several widths are captured, but they SHARE one
            # VerifyCaptureBuffer sized at the widest (VerifyCaptureBuffer.view), and the widest is
            # `min(num_draft, 15)` <= num_draft. So this single-width reservation still bounds the
            # static I/O buffers, and now over-reserves slightly rather than under-reserving. The
            # per-width graphs share one pool captured widest-first, so the pool high-water mark is
            # also the widest graph's — which the _GRAPH_ACT_MULT term below already approximates.
            qlen = sc.num_draft + 1
            vbs = max((b for b in bs_list if b <= config.max_running_req), default=max_bs)
            T = vbs * qlen
            total += T * vocab * f32 + 3 * T * i32
            total += (1 + _GRAPH_ASSUMED_AUX) * T * hidden * dt  # last_hidden + aux_hidden
            total += _GRAPH_ACT_MULT * T * hidden * dt
        # Block-diffusion canvas graphs (captured in Engine.__init__, i.e. AFTER the KV pool is
        # sized). Same reason every other term here exists: without it the pool takes the whole
        # budget and capture OOMs at boot, on the one path where "graphs disabled" is not a graceful
        # degradation but a 2-3x slower serve. Batch-size set must match _capture_canvas_graphs'.
        if mc.is_block_diffusion and mc.canvas_length:
            cbs = min(max_bs, config.max_running_req,
                      max(1, config.max_forward_len // int(mc.canvas_length)))
            total += cbs * int(mc.canvas_length) * _CANVAS_GRAPH_BYTES_PER_TOKEN
        total = int(total * _GRAPH_ROUNDUP)
        margin = os.environ.get("MINISGL_GRAPH_RESERVE_MARGIN_GB")
        if margin:  # non-empty (empty env string is ignored)
            total += int(float(margin) * (1 << 30))
        return total

    def _weight_offload_device_budget(self, config: EngineConfig) -> int:
        """VRAM per rank the MoE expert tier may occupy — the input `resolve_weight_plan` needs.

        WHY THIS EXISTS AT ALL. The resolver is a pure function of config and cannot ask the card
        anything. Left to its own fallback it reads `config.weight_offload_device_gb` and, at 0.0,
        plans EVERY MoE layer host-resident — a 4.5 GiB expert stack on a 16 GB card would be pinned
        to host RAM and streamed over PCIe at ~1/10 the native rate, on a serve that never asked for
        offload and with no flag to turn it off. So the engine, which is the only component that
        knows the hardware, supplies the number.

        WHY `total_memory` AND NOT `mem_get_info`. The two TP ranks must resolve the IDENTICAL plan:
        if they disagree about which layers are host-resident, one rank streams a layer the other
        holds resident and the model emits plausible wrong text with no error anywhere. A live free-
        memory delta differs between the rank processes by whatever else touched the card; total
        memory is a hardware constant. It is still MIN-reduced over the TP group so a heterogeneous
        pair (this box: RX 9070 XT + RX 9070) cannot make the ranks diverge either.

        WHY OVER-GRANTING IS THE RIGHT DIRECTION, AND WHERE IT STOPS. This budget deliberately does
        not subtract the KV pool, dense weights or graph buffers — there is no honest config-time
        number for those, and a guess would be a confident wrong number. Granting too little silently
        converts a serve that fits into a PCIe-streamed one, so the bias is towards resident.

        But the derived value is not merely generous, it is the ENTIRE KV budget, and that has a hard
        consequence rather than a soft one. `_determine_num_pages` computes
        `available = memory_ratio * old_free - model - ...` with the device tier billed inside
        `model` (the plan forbids a sixth subtrahend), so whenever the expert stack does not fit —
        i.e. exactly when the plan is non-empty — the greedy fill takes the tier up to this number
        and `available` is negative before anything else is counted. `assert num_pages > 1` cannot
        not fire. That is why `StageASession.begin` is told `budget_is_derived`: the unbootable case
        is refused from integers at plan time (`bake._refuse_derived_budget_that_leaves_no_kv`)
        instead of after the arena has pinned tens of GiB and the checkpoint has been loaded. A model
        that FITS still resolves to an empty plan and never reaches that refusal, which is what keeps
        this path on every serve.
        """
        if config.weight_offload_device_gb > 0:
            # GiB, and the UNIT IS NOT A STYLE CHOICE. This value has two readers: here, and
            # `resolve_weight_plan`'s own fallback when no budget is passed. That fallback is
            # `weight_offload_device_gb * GIB_PER_UNIT` (plan.py), the plan's `_gb()` renders every
            # figure it prints in GiB, and `WeightPlanResolution.summary_line()` — which is what the
            # `[serve]` banner and the `KV sizing:` line quote — prints the granted tier back in GiB
            # too. Reading the same field as decimal 1e9 here would grant 7.4% LESS VRAM than the
            # operator asked for while every log line echoed the request as honoured, and 7.4% of a
            # tier is enough to push a layer across the greedy fill boundary into host residency.
            # plan.py's own comment names this exact trap; one unit, one place.
            from minisgl.weights.plan import GIB_PER_UNIT

            return int(config.weight_offload_device_gb * GIB_PER_UNIT)
        total = int(torch.cuda.get_device_properties(self.device).total_memory)
        if config.tp_info.size > 1 and self.tp_cpu_group is not None:
            t = torch.tensor([total], device="cpu", dtype=torch.int64)
            torch.distributed.all_reduce(
                t, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
            )
            total = int(t[0].item())
        return int(total * config.memory_ratio)

    def _woff_mem_probe(self) -> Tuple[int, int, int]:
        """`(free, allocated, reserved)` for the weight-arena ledger (weights/accounting.py).

        Deliberately does NOT call `empty_cache()`, unlike `_sync_get_memory`: the ledger's whole job
        is to attribute changes in device-free memory to the arena, and returning segments to the
        driver mid-window would move `free` for a reason that has nothing to do with the arena and
        make the Phase-0 "is this really host memory?" assertion meaningless. It DOES synchronize,
        because `mem_get_info` is only meaningful once queued frees have retired.

        `free` is the driver's own number. That is the point: Phase 0 found `hipMemCreate(location=
        Host)` returning VRAM while the property query echoed "Host", and P5b found
        `hipPointerGetAttributes` reporting "Device" for real host pages — both queries lie, in both
        directions, so the arena is verified by asking the card how much of itself is left.
        """
        torch.cuda.synchronize(self.device)
        # Scoped to `self.device`, not the ambient current device. `free` already is (it goes
        # through `mem_get_info(device)`), and mixing a device-scoped `free` with process-current
        # allocator stats would compare two different cards' numbers the moment anything moved the
        # current device — the ledger's whole output is differences between these three, so a
        # mismatch there is a plausible wrong number rather than an error.
        return (
            get_free_memory(self.device),
            torch.cuda.memory_allocated(self.device),
            torch.cuda.memory_reserved(self.device),
        )

    def _tp_min_num_pages(self, num_pages: int, config: EngineConfig) -> int:
        """MIN-reduce the KV page count across this replica's TP ranks. See the call site.

        Scoped to `self.tp_cpu_group` — the per-replica TP subgroup under DP — for the same reason
        `_sync_get_memory` is: different DP replicas sit on different cards and are sized
        independently, so reducing across replicas would shrink every one of them to the weakest.

        Called UNCONDITIONALLY (including on the `--num-pages` override path, where it is a no-op in
        value): a collective that only some ranks enter is a worse failure than the one it fixes.
        """
        if config.tp_info.size <= 1 or self.tp_cpu_group is None:
            return num_pages
        t = torch.tensor([int(num_pages)], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            t, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        reduced = int(t[0].item())
        if reduced != num_pages:
            # Not fatal — MIN is exactly the resolution — but it means the per-rank terms of the KV
            # budget disagreed, which is worth a line because it is invisible in every other log.
            logger.warning(
                f"KV sizing disagreed across TP ranks: this rank sized {num_pages} pages, the "
                f"replica minimum is {reduced}; using the minimum so every rank's pool can hold "
                f"every page id. The per-rank terms are memory_allocated/memory_reserved and the "
                f"weight-arena correction."
            )
        return reduced

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        mc = config.model_config
        if mc.is_mla:
            # MLA stores ONE latent (kv_lora_rank + qk_rope_head_dim) per token per layer — no ×2
            # for K/V and no per-head factor (the latent is shared across heads, TP-replicated).
            cache_per_page = (
                (mc.kv_lora_rank + mc.qk_rope_head_dim)
                * config.page_size
                * self.kv_dtype.itemsize
                * mc.num_kv_layers  # == num_layers for MLA (all-attention); matches pool alloc
            )
        else:
            # head_dim / num_kv_heads are the MAIN pool's geometry, which for a SWA hybrid is the
            # FULL-attention layers' (num_kv_layers counts exactly those). A split-head_dim model
            # (Gemma4) keeps its sliding geometry in swa_head_dim/swa_num_kv_heads, charged to the
            # ring pool by _recurrent_state_bytes — the two pools have different per-token costs and
            # are billed separately.
            cache_per_page = (
                2  # key + value
                * mc.head_dim
                * div_even(mc.num_kv_heads, config.tp_info.size, allow_replicate=True)
                * config.page_size
                * self.kv_dtype.itemsize
                * mc.num_kv_layers  # only full-attn layers keep paged KV (GDN hybrid: 10, not 40)
            )
        num_pages = config.num_page_override
        if num_pages is None:
            # Bill the model for its RESIDENT tensors plus whatever is genuinely outside torch — NOT
            # the raw free-memory delta. Weight loading peaks well above the final weight size
            # (measured on Qwen3.8-27B-NVFP4, TP=2: peak allocated 14.11 GiB vs 11.67 GiB resident),
            # torch sizes its segment pool to that PEAK, and empty_cache does not hand it back to the
            # driver. So the delta charges the model ~2.7 GiB of allocator segments that are FREE —
            # and that the KV pool, which allocates through the SAME torch allocator, reuses. Charging
            # them shrank this model's pool from ~186k tokens to 25k, and at the shipped 0.80 ratio
            # sized it NEGATIVE and refused to boot. `used - reserved` keeps the part that really is
            # non-torch (HIP context, kernel code objects). Unquantized models barely notice — their
            # load has almost no transient (qwen27b: reserved 9.09 vs allocated 9.00).
            device_used = old_free_memory - new_free_memory
            # WEIGHT OFFLOAD: a MEASURED correction, not a reservation. The offload arena's DEVICE
            # tier stays fully billed — it is inside `device_used` and that is exactly where the plan
            # wants it (no sixth subtrahend). What is corrected here is the HOST arena: P5b validated
            # it through torch's `_cuda_customAllocator` + `MemPool`, and a pool-served tensor over
            # `hipHostGetDevicePointer` pages is an ordinary `cuda` tensor to
            # `memory_allocated()`/`memory_reserved()`. Left alone, ~50 GB of HOST RAM would be billed
            # as device memory, `available_memory` would go hard negative and the assert below would
            # refuse to boot. Both terms are the delta measured across arena construction, clamped to
            # the arena size, and are exactly 0 when the arena is invisible to torch — so every serve
            # that does not offload is byte-identical. `reserved` is corrected too: fixing only
            # `allocated` would zero the non-torch term and under-bill the HIP context.
            #
            # BOTH terms are CLAMPED TO THE LIVE READING before they are applied, and that clamp is
            # load-bearing rather than defensive. The corrections are deltas sampled at `seal()`,
            # but they are applied to `memory_allocated()`/`memory_reserved()` read HERE — and
            # `self._sync_get_memory()` on the first line of this method calls
            # `torch.cuda.empty_cache()` in between. `empty_cache` releases the segments the bake's
            # dropped originals left behind, so `reserved` can legitimately fall by roughly the
            # arena size between the sample and the read. Subtracting an unclamped seal-time delta
            # from a shrunken reading drives `reserved - woff_reserved` NEGATIVE, which inflates the
            # non-torch term `max(0, device_used - reserved)` by up to the whole arena and refuses
            # the boot; the mirror case on `allocated` makes `model_memory` negative, which
            # over-states `available_memory` and sizes a KV pool that OOMs on the first forward.
            # A clamp cannot mask a real fault — the ledger's own gate at `seal()` already refused
            # to reach this line if the arena and the plan disagreed — it only stops a stale delta
            # from turning into a wrong pool size in either direction.
            alloc_now, reserved_now = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
            _c_alloc, _c_reserved = self._woff.model_memory_correction()
            model_memory, woff_alloc, woff_reserved = corrected_model_memory(
                allocated=alloc_now,
                reserved=reserved_now,
                device_used=device_used,
                alloc_correction=_c_alloc,
                reserved_correction=_c_reserved,
            )
            # Reserve the fixed GDN/CCA recurrent-state cache, which is allocated AFTER the KV pool
            # and scales with max_running_req. Without this the KV pool takes the whole budget and the
            # state alloc OOMs — the reason GDN 35B on 16 GB needed a manual --max-running-requests cap
            # ([B2]). Subtracting it up front co-sizes the two caches automatically; it is 0 for dense/
            # MHA/MLA models, so their sizing is unchanged.
            state_memory = self._recurrent_state_bytes(config)
            # Reserve the spec draft model (EAGLE3/DFlash) and the CUDA-graph static buffers, both of
            # which are allocated AFTER the KV pool (proposer build / graph capture) and otherwise come
            # out of the (1-memory_ratio) slack — the cause of capture-time and first-request OOMs that
            # forced manual mem-ratio tuning. Both are 0 when not applicable (no spec / no capture).
            draft_memory = self._draft_model_bytes(config)
            graph_memory = self._graph_capture_bytes(config, old_free_memory)
            snap_memory = self._rec_snapshot_store_bytes(config)
            ple_memory = self._ple_runtime_bytes(config)
            stage_memory = self._moe_prefill_stage_bytes(
                int(config.memory_ratio * old_free_memory)
                - model_memory - state_memory - draft_memory - graph_memory - snap_memory
                - ple_memory,
                cache_per_page, config.page_size, config.max_running_req,
            )
            available_memory = (
                int(config.memory_ratio * old_free_memory)
                - model_memory
                - state_memory
                - draft_memory
                - graph_memory
                - snap_memory
                - ple_memory
                - stage_memory
            )
            # Per-term breakdown. Without it the "Not enough memory for KV cache" assert below names
            # five candidate causes and gives no way to tell which one actually ate the budget —
            # every diagnosis starts by re-deriving these numbers by hand.
            logger.info(
                f"KV sizing: free={mem_GB(old_free_memory)} x ratio={config.memory_ratio} = "
                f"{mem_GB(int(config.memory_ratio * old_free_memory))} budget; "
                f"model={mem_GB(model_memory)} state={mem_GB(state_memory)} "
                f"draft={mem_GB(draft_memory)} graph={mem_GB(graph_memory)} "
                f"snap={mem_GB(snap_memory)} ple={mem_GB(ple_memory)} "
                f"stage={mem_GB(stage_memory)} "
                f"-> available={mem_GB(available_memory)} "
                f"@ {cache_per_page} B/page; "
                # `model` is a free-memory DELTA, so it also carries allocator slack, fragmentation
                # and the HIP context. `allocated` is the exact resident tensor total — when the two
                # diverge the gap is overhead, not weights, and the fix is not a higher memory-ratio.
                f"resident tensors={mem_GB(alloc_now)} "
                # reserved-minus-allocated is torch's own segment overhead (kept after empty_cache
                # because those segments still hold a live block); whatever `model` exceeds RESERVED
                # by is not torch at all — HIP context, kernel code objects, comms buffers.
                f"torch reserved={mem_GB(reserved_now)}"
                # The two raw figures above are UNCORRECTED, and with a pool-served host arena they
                # include tens of GB of host RAM: on a 16 GB card "resident tensors=42.31 GiB" is
                # not a typo and not a bug, it is the arena. Print what was actually removed, on the
                # same line, or the only way to tell a mis-sized pool from a mis-read log is to
                # re-derive both numbers by hand. Empty when nothing was corrected, so a serve that
                # does not offload keeps this line byte-identical.
                + (
                    f" (host-arena corrected: -{mem_GB(woff_alloc)} allocated, "
                    f"-{mem_GB(woff_reserved)} reserved)"
                    if (woff_alloc or woff_reserved)
                    else ""
                )
                # One annotation on the SAME line rather than a second log line, so an operator
                # diagnosing a small pool sees the arena in the same place they see the five
                # reservations. Empty string when nothing is offloaded, so this line is
                # byte-identical on every serve that does not offload. It delegates to the resolved
                # plan's own `summary_line()` so the KV-sizing line and the `[serve]` banner can
                # never quote different numbers for the same plan — including the step floor, which
                # is derived against the SLOWEST rank's measured host bandwidth (the links are
                # independent, so at tp>=2 the slower card sets the step and quoting card 0's
                # 28.93 GB/s for the pair is wrong by ~2x).
                + self._woff.kv_annotation()
            )
            num_pages = available_memory // cache_per_page
            if snap_memory:
                # Name the snapshot KIND. "recurrent" is conv+ssm / conv+prev_hs clones; "swa" is
                # sliding-window K/V clones — the same store, but a reader who sees "recurrent" on a
                # model with no recurrent state (the old wording) reasonably concludes the accounting
                # is broken. It was.
                logger.info(
                    f"Reserved {mem_GB(snap_memory)} for the "
                    f"{resolve_prefix_cache(config).snapshot_kind}-radix snapshot store "
                    f"(filled during serving); KV pool gets the remainder"
                )
            if state_memory:
                # This line covers THREE disjoint pools and used to name only two of them, so on a SWA
                # model (Gemma4: 25 sliding layers x a 1024-token window) it reported a "GDN/CCA
                # recurrent state" reservation for a model that has neither — legitimate memory,
                # unrecognisable label. Name whichever one this model actually pays for.
                _kinds = [
                    n for n, on in (
                        ("GDN", mc.is_gdn_hybrid),
                        ("CCA", mc.is_cca_hybrid),
                        ("SWA ring KV", getattr(mc, "is_swa_hybrid", False)),
                    ) if on
                ]
                logger.info(
                    f"Reserved {mem_GB(state_memory)} for {' + '.join(_kinds) or 'recurrent'} state "
                    f"({config.max_running_req} slots); KV pool gets the remainder"
                )
            if draft_memory:
                logger.info(
                    f"Reserved {mem_GB(draft_memory)} for the spec draft model "
                    f"({config.spec_algorithm}); KV pool gets the remainder"
                )
            if graph_memory:
                spec_note = (
                    f", spec-verify qlen={config.spec_num_draft + 1}"
                    if config.spec_config is not None
                    else ""
                )
                logger.info(
                    f"Reserved {mem_GB(graph_memory)} for CUDA graph capture "
                    f"(max_bs={config.cuda_graph_max_bs}{spec_note}); KV pool gets the remainder"
                )

        # Make the pool size RANK-IDENTICAL before anything is allocated against it. Every other
        # input above is either a config constant or already cross-rank reduced (`_sync_get_memory`
        # returns the MIN/MAX all_reduce, so `old_free_memory`, `new_free_memory` and therefore
        # `device_used` are the same integer on every rank). The exceptions are
        # `torch.cuda.memory_allocated()` / `memory_reserved()` — live per-rank allocator readings —
        # and now the weight-arena correction derived from them, which is measured across the bake on
        # each rank's own card. They agree when both ranks run an identical allocation sequence on
        # identical hardware, which is an assumption, not an invariant: this box's TP pair is an
        # RX 9070 XT and an RX 9070, `empty_cache()` runs between the correction's sample and its use
        # (`corrected_model_memory`'s clamp exists precisely because that reading can move), and the
        # offload path adds a second measured term on top.
        #
        # A single page of disagreement is not a small error. `num_pages` sizes the KV pool AND
        # bounds the page indices the primary rank hands out (`weights/plan.py`'s own divergence
        # message names this: "size different KV pools (num_pages is not cross-rank reduced)"), so a
        # rank with fewer pages is written past the end of its pool by the peer's page ids — silently
        # wrong attention, not an allocation error. MIN is the safe side: every rank can hold it.
        # No-op at tp_size == 1, and a no-op in bytes whenever the ranks already agreed.
        num_pages = self._tp_min_num_pages(int(num_pages), config)

        assert num_pages > 1, (
            "Not enough memory for KV cache after reserving recurrent state / draft model / "
            "CUDA-graph buffers; reduce --max-running-requests, lower --memory-ratio, or set "
            "--num-pages"
            # WEIGHT OFFLOAD: name the device tier when there is one. `--weight-offload-device-gb`
            # (and its default, a fraction of the card's TOTAL memory) grants VRAM to the resident
            # expert tier BEFORE the KV pool is sized, and that tier lands inside `model` — by
            # design, since the plan forbids a sixth subtrahend. So the single most likely way to
            # reach this assert on an offloading serve is a device tier that left nothing for KV,
            # and the four causes listed above do not include it. Without this clause the operator
            # is sent to `--memory-ratio`, which makes the budget SMALLER and the failure worse.
            + self._woff.kv_budget_failure_hint()
        )
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across this replica's TP ranks.

        The all_reduce is over ``self.tp_cpu_group`` — the per-replica TP subgroup under DP — so the
        >2GB imbalance guard is scoped WITHIN a replica. Different DP replicas sit on different cards
        (card 0 vs card 1, with the iGPU skewing one) and are sized independently, so the guard never
        compares free memory ACROSS replicas (which would false-trip on benign per-card differences).
        """
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    # Cumulative GPU time spent in prefill forwards; exported as minisgl_prefill_seconds_total.
    #
    # Timed with CUDA EVENTS, not host wall-clock. forward_batch returns once the kernels are
    # LAUNCHED, so host timing measures launch overhead: it read 84,631 tokens / 0.384 s = 220k
    # tok/s, which is not a prefill rate. Events measure the device interval without forcing a host
    # sync; pairs are drained opportunistically once complete, so nothing blocks the scheduler.
    prefill_seconds_total: float = 0.0

    # Cumulative prompt tokens actually COMPUTED, accumulated per prefill chunk as the work happens.
    # Exported as minisgl_prefill_computed_tokens_total.
    #
    # This exists because the two pre-existing prompt-token counters are both attributed to an INSTANT
    # rather than to the interval over which the work ran, which made prefill and decode look
    # concurrent on the throughput panel:
    #   * minisgl_prompt_tokens_total (frontend, metrics.py) credits the whole prompt on the FIRST
    #     REPLY for a request — i.e. at TTFT, after prefill finished — so a 52k-token prompt landed as
    #     one spike exactly on top of the first decode token, displaced by the full prefill duration
    #     (measured: 21 s).
    #   * minisgl_prefix_cache_prompt_tokens_total (prefill.py) credits it once at ADMISSION, before
    #     the work runs, and deliberately counts the whole prompt (it is the hit-ratio denominator).
    # rate() then smears either instantaneous step across the whole rate window, drawing a flat
    # plateau that overlaps the decode ramp.
    #
    # Counted here rather than in the scheduler because forward_batch is the single choke point every
    # prefill chunk passes through (including the spec-decode seeded prefill), and because it keeps
    # this counter scope-identical to prefill_seconds_total — numerator and denominator of the prefill
    # throughput panel then refer to the same set of forwards. extend_len is per-chunk (device_len -
    # cached_len), so a chunked prefill contributes its chunks across the steps that ran them, and
    # prefix-cache hits are excluded for free (cached tokens are never re-computed). Spec VERIFY
    # batches are phase="decode", so their extend_len>1 query tokens are correctly NOT counted as
    # prompt.
    prefill_computed_tokens_total: int = 0

    # Decode steps that ran the EAGER `model.forward()` instead of replaying a captured graph. The
    # counter-part to `GraphRunner.replays`, and it exists for the same provenance reason: with only
    # a replay count, "eager leg" is a claim about a switch someone flipped, not an observation. With
    # both, a capture-vs-eager A/B leg is arithmetically pinned — the captured leg must show
    # replays>0 AND eager_decode_forwards unchanged, the eager leg the exact mirror. A leg where both
    # move is a leg where the graphs only covered part of the decode and the ratio means nothing.
    eager_decode_forwards: int = 0

    def _drain_prefill_events(self) -> None:
        """Fold completed prefill event pairs into the counter. Non-blocking: an incomplete pair is
        left for a later step, so this never syncs the host onto the GPU just to keep a metric."""
        ev = self._pf_events
        while ev and ev[0][1].query():
            start, end = ev.popleft()
            self.prefill_seconds_total += start.elapsed_time(end) / 1000.0

    def forward_batch(
        self, batch: Batch, args: BatchSamplingArgs, return_hidden: bool = False
    ):
        assert torch.cuda.current_stream() == self.stream
        _maybe_profile()
        extra = None
        # GPU time for minisgl_prefill_seconds_total — the denominator of the "prefill throughput"
        # panel, which until now divided by a series the engine never exported and so rendered blank.
        _pf_ev = None
        if batch.is_prefill:
            _pf_ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            _pf_ev[0].record(self.stream)
            # Chunk-accurate prompt-token accounting: credit only what THIS chunk computes, at the
            # step that computes it. See prefill_computed_tokens_total.
            self.prefill_computed_tokens_total += sum(r.extend_len for r in batch.reqs)
        _pf_probe = batch.is_prefill and os.environ.get("MINISGL_PREFILL_MEM_PROBE") == "1"
        if _pf_probe:
            torch.cuda.reset_peak_memory_stats()
            _pf_free0 = torch.cuda.mem_get_info()[0]
            _pf_alloc0 = torch.cuda.memory_allocated()
            _pf_tok = sum(r.extend_len for r in batch.reqs)
            _pf_ctx = max(r.device_len for r in batch.reqs)
            # SUM of context across the batch, not just the max: the single-stream probe showed peak
            # allocation flat in ctx, but a 4-way concurrent long-context batch still OOM'd, so the
            # scaling term has to be looked for across the batch, not within one request.
            _pf_ctxsum = sum(r.device_len for r in batch.reqs)
        with self.ctx.forward_batch(batch):
            if return_hidden:
                # Draft-head spec-decode PREFILL SEED path: one forward yields the bonus-token logits
                # (lm_head still does the prefill last-token reduction internally) AND the per-token
                # target hidden over the whole prompt — no extra pass. Never a CUDA graph (return_hidden
                # is spec-only, and spec disables graph capture). See scheduler._spec_prefill_seeded.
                logits, last_hidden, aux_hidden = self.model.forward(return_hidden=True)
                extra = (last_hidden, aux_hidden)
            elif self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                if batch.is_decode:
                    self.eager_decode_forwards += 1
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        if args.grammar_bitmask is not None and self.tp_size > 1:
            # Structured output at TP>1: the grammar bitmask makes per-rank token selection DIVERGE —
            # it amplifies tiny cross-rank logit FP differences over the small allowed/renormalized
            # set, so multinomial sampling (and, at a near-tie, even argmax) can pick a different token
            # on each rank. Divergent commits desync the decode managers and deadlock the collectives
            # (and KV would silently differ). Force rank0's tokens onto every rank for an identical
            # commit (host seq + KV pool). Scoped to constrained batches; they already run the
            # synchronous decode path, so the extra D2H + CPU broadcast is cheap.
            next_tokens_cpu = next_tokens_gpu.to("cpu")
            self.tp_cpu_group.broadcast(next_tokens_cpu, root=0).wait()
            next_tokens_gpu = next_tokens_cpu.to(next_tokens_gpu.device)
        else:
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        out = ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event,
                            self.sampler.pop_captured_logprobs())
        if _pf_ev is not None:
            _pf_ev[1].record(self.stream)
            self._pf_events.append(_pf_ev)
        self._drain_prefill_events()
        if _pf_probe:
            torch.cuda.synchronize()
            _peak = torch.cuda.max_memory_allocated() - _pf_alloc0
            logger.info_rank0(
                # ABSOLUTE levels, not just the delta: the delta is flat (465 MiB per 8192-token
                # step, independent of ctx/bs), yet the card still fills up under concurrency — so
                # what matters is what the BASELINE climbs to between steps, not what one step adds.
                f"[pf-probe] tokens={_pf_tok} ctx={_pf_ctx} ctxsum={_pf_ctxsum} bs={batch.size} "
                f"peak_delta_MiB={_peak/(1<<20):.1f} "
                f"alloc_MiB={torch.cuda.memory_allocated()/(1<<20):.0f} "
                f"reserved_MiB={torch.cuda.memory_reserved()/(1<<20):.0f} "
                f"devfree_MiB={torch.cuda.mem_get_info()[0]/(1<<20):.0f}"
            )
        return (out, *extra) if return_hidden else out

    def forward_verify(self, batch: Batch, return_hidden: bool = False):
        """Eager forward for a speculative-decode verify batch.

        The batch is built with ``phase='decode'`` but each req carries ``extend_len = K+1`` query
        tokens (confirmed + K drafts). Two consequences we rely on (see SPEC_DECODE.md):
          * ``prepare_metadata`` keys on ``extend_len`` (not phase), so it takes the paged-extend
            branch → the existing ``flash_prefill_paged`` kernel applies the topk=1 linear-chain
            causal mask for free (no new kernel).
          * the LM head does the per-req last-token reduction only for ``is_prefill``, so a decode
            batch returns logits for ALL tokens — exactly the K+1 per-position logits verify needs.

        Returns the full logits ``[sum(extend_len), vocab]``. With ``return_hidden=True`` (draft-head
        proposers — MTP / EAGLE3 / DFlash) returns ``(logits, last_hidden, aux_hidden)`` where
        ``last_hidden`` is the PRE-final-norm residual ``[sum(extend_len), hidden]`` (NOT post-norm:
        the model returns the pre-norm residual stream so the MTP head's own ``pre_fc_norm_hidden``
        re-normalizes it; feeding a post-norm hidden here would double-norm the seed and collapse
        acceptance — see qwen3_5.py:599-606) and
        ``aux_hidden`` is the stacked captured decoder layers ``[num_capture_layers, sum(extend_len),
        hidden]`` (or ``None`` if no capture layers are programmed). Capture layers are programmed via
        ``model.set_capture_layers`` at proposer init. No sampling and no ``complete_one``: the
        scheduler owns acceptance, commit, and req advancement.

        MLA verify is CUDA-graph captured: when the batch is graphable (uniform K+1 query tokens, bs
        fits a captured size) ``replay_verify`` runs the staged forward as one graph replay (the
        scheduler has copied input_ids/positions/out_loc into the static buffers). Otherwise — a
        partial-K step, or a non-MLA backend — it falls back to the eager forward."""
        assert torch.cuda.current_stream() == self.stream
        _maybe_profile()  # count verify steps too, so MINISGL_PROFILE can trace the spec-verify path
        # v2 S4: the FUSED-TiDAR custom-mask verify forward has its own captured graph (distinct qlen +
        # a static dense mask). Check it first — its batch carries `fused_verify=True` and fused_qlen
        # query tokens, so it never collides with the K+1 two-forward verify graph below. Logits-only.
        # Under DP+EP the in-graph MoE all_gather bakes the captured N while the idle replica's dummy
        # self-agrees N eagerly (moe.py case 3) → keep the fused verify eager there. EP-over-TP has no
        # idle replica (the TP ranks run the same bs in lockstep), so replay stays safe.
        if self.graph_runner.can_use_fused_verify(batch) and (
            not self.enable_ep or self._ep_over_tp
        ):
            return self.graph_runner.replay_fused_verify(batch)
        # DDTree draft-TREE verify: its own captured graph (fixed tree_qlen + a static ancestor mask +
        # state-neutral recurrent scratch). Batch carries `ddtree_verify=True`; logits-only. Under EP the
        # in-graph MoE all_gather uses a fixed captured N while an idle replica self-agrees N eagerly →
        # keep the tree-verify eager there too (same reasoning as the K+1 verify's EP gate).
        if self.graph_runner.can_use_ddtree_verify(batch) and not self.enable_ep:
            return self.graph_runner.replay_ddtree_verify(batch)
        if self.graph_runner.can_use_verify_graph(batch):
            return self.graph_runner.replay_verify(batch, return_hidden)
        with self.ctx.forward_batch(batch):
            return self.model.forward(return_hidden=return_hidden)

    def forward_canvas(
        self, batch: Batch, canvas_ids: torch.Tensor, self_conditioning: torch.Tensor
    ) -> torch.Tensor:
        """One block-diffusion denoising step. Returns full-vocab fp32 logits for EVERY canvas
        position, ``[sum(extend_len), vocab]``.

        The two things it deliberately does NOT do are the whole reason it is not ``forward_batch``:

          * NO ``sampler.sample``. A canvas step needs per-position entropy over the full vocabulary,
            a per-position multinomial, an entropy-ordered acceptance set and a re-noise — none of
            which is top-k/top-p sampling. Penalties, grammar and the reasoning gate are per-token
            autoregressive concepts that do not apply here at all. ``minisgl.diffusion`` owns this.
          * NO ``complete_one()``. The canvas is SCRATCH: the same ``canvas_length`` slots are
            overwritten on every one of the <=48 steps of a block, so ``cached_len``/``device_len``
            must stay exactly where the block started. Advancing them would allocate a fresh canvas
            of slots per step and leave the block attending its own denoising history.

        The step is CUDAGRAPH-CAPTURED when the batch matches a captured size (see
        ``GraphRunner.capture_canvas_graphs``); only the backbone is in the graph, the LM head runs
        eagerly either way. Note where ``prepare_metadata`` sits: building the eager metadata is
        itself a per-step O(bs*(window+canvas)) Python loop over ring slots, so it is built ONLY on
        the eager branch — the captured branch's static rows are filled by
        ``prepare_canvas_for_replay``. Doing both would leave a measurable part of the very host cost
        capture exists to remove.

        ``canvas_ids`` is passed explicitly rather than read from ``batch.input_ids`` because the
        canvas is re-sampled between steps and never enters a request's host token buffer; the token
        pool holds a copy only so the KV scatter can address the right slots. ``self_conditioning``
        is always a real ``[tokens, hidden]`` tensor (zeros on the first step of a block) rather than
        ``None``: a captured graph cannot branch on it, and zeros are bit-identical to skipping the
        block (see ``DiffusionGemmaSelfConditioning.forward``)."""
        assert torch.cuda.current_stream() == self.stream
        assert batch.canvas, "forward_canvas requires a canvas batch (Batch.canvas is False)"
        _maybe_profile()
        if self.graph_runner.can_use_canvas_graph(batch):
            hidden = self.graph_runner.replay_canvas(batch, canvas_ids, self_conditioning)
        else:
            self.attn_backend.prepare_metadata(batch)
            with self.ctx.forward_batch(batch):
                hidden = self.model.forward_canvas_hidden(canvas_ids, self_conditioning)
        return self.model.canvas_logits(hidden)

    def _capture_canvas_graphs(self, config: EngineConfig) -> None:
        """Capture the block-diffusion canvas graphs, at boot, for every batch size the serve can
        actually admit. No-op for every autoregressive model.

        Two bounds, and both are real rather than defensive:
          * ``max_graph_bs`` — the operator's own graph-coverage knob (``--cuda-graph-max-bs``, which
            ``tools/serve.sh`` pins to the concurrency). Above it, capture is off by request.
          * ``max_forward_len`` (= ``--max-extend-tokens`` on a served config) — a canvas step of
            ``bs*canvas_length`` tokens IS a prefill of that many tokens as far as activation memory
            is concerned, and the engine already refuses prefills above that budget. Capturing a
            batch the forward budget forbids would reserve graph memory for a step that can never
            run.
        Sizes are CONTIGUOUS (1..max) rather than the decode bucket ladder, because the canvas path
        matches bs EXACTLY (no dummy padding — a padded canvas row is a whole extra 256-token forward
        through 30 layers), so a bucket gap is a batch size that silently runs eager."""
        mc = config.model_config
        if not mc.is_block_diffusion or self.graph_runner.max_graph_bs == 0:
            return
        canvas_len = int(mc.canvas_length)
        by_tokens = max(1, config.max_forward_len // canvas_len)
        max_bs = min(self.graph_runner.max_graph_bs, config.max_running_req, by_tokens)
        if max_bs < 1:
            return logger.info_rank0("canvas CUDA graph: no admissible batch size — capture skipped")
        # Buffer dtype is the EMBEDDING's, not ``self.dtype``: the self-conditioning input and the
        # backbone hidden are both in the embedding table's dtype, and a static buffer one step off
        # would silently CAST on every copy_ instead of raising — a numerical difference against the
        # eager path, on the input the whole denoising loop is conditioned on.
        emb_w = self.model.model.embed_tokens.weight
        with torch.cuda.stream(self.stream):
            self.graph_runner.capture_canvas_graphs(
                model=self.model,
                canvas_len=canvas_len,
                bs_list=list(range(1, max_bs + 1)),
                hidden_size=emb_w.shape[1],
                dtype=emb_w.dtype,
            )

    def capture_spec_verify_graphs(
        self, needs_hidden: bool, num_aux: int, bs_list: "list[int]",
        widths: "list[int] | None" = None,
    ) -> None:
        """Capture the MLA spec-decode verify graphs. Called by the scheduler AFTER the proposer is
        built and the target's aux-capture layers are programmed (so the captured forward stashes the
        hidden states the draft head consumes). No-op if graphs are disabled / non-MLA backend.

        `widths` is the adaptive-verify-width ladder (spec/width.py); None captures the single fixed
        width `num_draft`."""
        if self.graph_runner.max_graph_bs == 0 or self.spec_config is None:
            return
        hidden_size = self.model.model.embed_tokens.weight.shape[1]
        # Capture on the ENGINE stream (the scheduler may have switched the current stream to its own
        # in __init__); the warmup forward + graph context must share it.
        with torch.cuda.stream(self.stream):
            self.graph_runner.capture_verify_graphs(
                model=self.model,
                num_draft=self.spec_config.num_draft,
                bs_list=bs_list,
                needs_hidden=needs_hidden,
                num_aux=num_aux,
                hidden_size=hidden_size,
                dtype=self.dtype,
                widths=widths,
            )

    def capture_spec_propose_graphs(self, proposer, bs_list: "list[int]") -> None:
        """Capture the proposer's PROPOSE graphs (spec/capture.py). Called by the scheduler right
        after the verify graphs, on the ENGINE stream — the scheduler may have switched the current
        stream in its __init__, and the warmup forward plus the graph context must share one stream.

        Capture failure is logged and survivable, NOT fatal: the proposer keeps the same body and
        runs it eagerly, its eager counter climbs, and the [spec-timing] line reports the split. A
        boot crash on a memory-tight card is a worse trade than a slower-but-correct serve — but
        "slower" has to be VISIBLE, which is what the counters are for."""
        if self.graph_runner.max_graph_bs == 0 or self.spec_config is None:
            return
        self._spec_proposer = proposer
        try:
            with torch.cuda.stream(self.stream):
                proposer.capture_propose_graphs(bs_list)
        except torch.cuda.OutOfMemoryError as e:
            proposer.destroy_propose_graphs()
            torch.cuda.empty_cache()
            logger.error(
                f"spec-decode: PROPOSE graph capture ran out of memory ({e}); propose will run "
                "EAGER. Lower --mem-fraction-static or --graph to recover the headroom.")

    def capture_spec_fused_verify_graphs(self, fused_qlen: int, bs_list: "list[int]") -> None:
        """Capture the FUSED-TiDAR custom-mask verify graphs (v2 S4). Called by the scheduler when the
        TiDAR FUSED path is enabled, after the proposer is built (it knows B → fused_qlen). No-op if
        graphs are disabled. Logits-only (TiDAR self-draft reads verify logits, not target hidden)."""
        if self.graph_runner.max_graph_bs == 0 or self.spec_config is None:
            return
        with torch.cuda.stream(self.stream):
            self.graph_runner.capture_fused_verify_graphs(
                model=self.model, fused_qlen=fused_qlen, bs_list=bs_list,
            )

    def capture_spec_ddtree_verify_graphs(
        self, tree_qlen: int, bs_list: "list[int]", max_ctx: int
    ) -> None:
        """Capture the DDTree draft-TREE verify graphs. Called by the scheduler when the DDTree path
        (DFlash/TiDAR + MINISGL_*_DDTREE=1) is enabled, after the proposer is built (it knows the node
        budget → tree_qlen = budget+1). No-op if graphs are disabled. Logits-only."""
        if self.graph_runner.max_graph_bs == 0 or self.spec_config is None:
            return
        with torch.cuda.stream(self.stream):
            self.graph_runner.capture_ddtree_verify_graphs(
                model=self.model, tree_qlen=tree_qlen, bs_list=bs_list, max_ctx=max_ctx,
            )

    def shutdown(self) -> None:
        # Propose graphs FIRST, for the same reason destroy_cuda_graphs exists at all: a live CUDA
        # graph held past NCCL teardown hangs the process. They were invisible here while capture
        # lived privately inside MTPProposer.
        if getattr(self, "_spec_proposer", None) is not None:
            self._spec_proposer.destroy_propose_graphs()
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    # MLA (DeepSeek / GLM-4.x MoE) requires the dedicated mla backend (absorbed decode + materialized
    # prefill over the latent cache); page_size 16 matches the validated mla_hip block size.
    if config.model_config.is_mla:
        if config.attention_backend not in ("auto", "mla"):
            logger.warning_rank0(
                f"MLA model: overriding attention backend {config.attention_backend!r} -> 'mla'"
            )
        override("attention_backend", "mla")
        if config.page_size % 16 != 0:
            override("page_size", 16)
            logger.warning_rank0("Page size is overridden to 16 for the mla backend")

    if config.attention_backend == "auto":
        if is_rocm():
            # gfx1201: the Triton-free native-HIP backend WITH cudagraph capture (decode + spec-verify).
            # NOT "rdna4" — that path is eager-only (capture is Phase-4 NotImplemented); select it
            # explicitly only for the Triton fallback (MINISGL_ATTN_HIP=0) or eager debugging.
            backend = "hip"
        else:
            backend = (
                "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
            )
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    # The native-HIP attention kernels require a KV block size that is a multiple of 16 — this covers
    # the "hip" backend and the "rdna4"/"triton_rdna4" base it subclasses (all share those kernels).
    if any(b in config.attention_backend for b in ("hip", "rdna4")) and config.page_size % 16 != 0:
        override("page_size", 16)
        logger.warning_rank0("Page size is overridden to 16 for the native-HIP attention backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")

    # Speculative decoding constraints — see SPEC_DECODE.md §5:
    #   * MHA verify reuses the paged-extend kernel; MLA verify uses the absorbed multi-query
    #     mla_hip.mla_verify kernel (no prefix re-materialization). Both supported.
    #   * page_size: MHA forces 1 (per-token rollback); MLA keeps 16 (the mla_hip block size) —
    #     the scheduler's rollback is page-size-aware, freeing whole pages beyond the kept run.
    #   * CUDA graph: MLA verify IS graph-captured (fixed bs*(K+1) tokens; scheduler-triggered after
    #     the proposer is built — see GraphRunner.capture_verify_graphs / forward_verify). Non-MLA
    #     (MHA/GDN) verify stays eager (page_size-1 / recurrent-state paths) — graph disabled there.
    if config.spec_config is not None:
        if not config.model_config.is_mla:
            # MHA/SWA spec historically forced page_size=1 for per-token KV rollback. That rollback is
            # now the SHARED page-aware form used by MLA spec at page_size=16: every free site rounds
            # via div_ceil(len, ps)*ps, releasing only WHOLE pages beyond the kept run and retaining the
            # partial page straddling cached_len (its rejected-draft tail is overwritten as the seq
            # regrows). Allocation (cache.allocate_paged) is likewise page-granular, the HIP verify
            # kernel reads the global page_size=1 table STRIDED by page_size (hip.py: gpt[..., ::ps]),
            # and the 30 SWA layers use an independent page_size=1 ring — so the whole MHA/SWA spec path
            # is page_size-parametrized. page_size>1 gives the 10 full-attn layers 16-token-contiguous KV
            # (better decode/prefill gather coalescing).
            #
            # THIS IS NOW THE DEFAULT and the MINISGL_SPEC_MHA_PAGED gate is gone. The gate existed
            # only because the MHA/SWA path had never been exercised at page_size>1 on a GPU; the
            # rollback itself is the same page-aware form MLA spec has shipped at page_size=16. What
            # backs the change:
            #   * tools/spec_page_rollback_check.py — the CPU proof of I1-I4 (kept slots unique, no
            #     kept KV freed, no page leak, ps=16 frees a whole-page superset of ps=1) over 1,989
            #     shapes for ps in {1,16}. Re-run it if this is ever touched.
            #   * Qwen3.8-27B-MTP-NVFP4 (MHA + MTP, TP=2) has served with it on.
            #   * Qwen3.8-Flash-Next (qwen4_exp: GDN hybrid + PLE + MTP, TP=2) booted and returned
            #     coherent greedy output at page_size=16, 2026-09-11 — the harder case, because GDN
            #     recurrent state and a PLE n-gram block both ride the same rollback.
            # Output correctness was never the exposure and that is what makes this safe to default:
            # spec decode is lossless by construction, so a rollback freeing the wrong pages shows up
            # as corrupted KV and visibly degraded text, not as a plausible-but-different answer.
            # All non-MLA backbones now cudagraph-capture the spec-VERIFY forward: the HIP attn
            # verify-capture (S1) is model-agnostic (pure MHA works by itself), and the recurrent
            # backbones thread their per-token state through static scratch buffers — CCA via
            # CCAVerifyGraphCapture, GDN via GDNVerifyGraphCapture. So keep graphs ON for MHA, GDN,
            # and CCA alike. (the verify capture reads the global table strided by
            # page_size, so it is correct at any page_size.)
            # The FUSED TiDAR
            # forward's non-K+1 qlen auto-falls-back to eager via can_use_verify_graph until S4.
            pass
        else:
            # Spec batches never exceed max_running_req (all running reqs verify together), so cap the
            # captured graph sizes there — a default 160 would capture huge unused decode graphs and
            # OOM (each verify graph also captures bs*(K+1) tokens). 0 keeps the user's eager opt-out.
            cur = config.cuda_graph_max_bs
            capped = config.max_running_req if cur is None else min(cur, config.max_running_req)
            if cur != capped:
                override("cuda_graph_max_bs", capped)
                logger.info_rank0(f"spec-decode (MLA): capping CUDA graph bs at {capped} (max_running_req)")
