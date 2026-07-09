from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, SamplingParams, set_global_ctx
from minisgl.distributed import (
    EPCommunicator,
    destroy_distributed,
    enable_pynccl_distributed,
    set_dp_info,
    set_tp_info,
)
from minisgl.kvcache import create_kvcache_pool
from minisgl.kvcache.cca_state import CCAStateCache
from minisgl.kvcache.gdn_state import GDNStateCache
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import (
    div_even,
    init_logger,
    is_rocm,
    is_sm90_supported,
    is_sm100_supported,
    torch_dtype,
)

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


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
        st["p"].export_chrome_trace(spec)
        st["p"] = None
        logger.info_rank0(f"[profile] wrote {active}-step trace to {spec}")

# Token count for the one-time GDN conv autotune warmup (3c-3). A single representative
# prefill length settles the per-process in-place batch_ptr autotune.
_GDN_WARMUP_TOKENS = 512


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        # Register this replica's DP coordinates (inert DpInfo(0,1) when dp_size=1). Done before any
        # CUDA init so EP (later) can build its dp collective group; also lets get_dp_info() resolve
        # to the real replica everywhere instead of the default.
        set_dp_info(
            dp_rank=config.dp_info.dp_rank,
            dp_size=config.dp_info.dp_size,
            enable_ep=config.enable_ep,
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
        self.enable_ep = config.enable_ep and config.dp_info.dp_size > 1
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

        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            self.model = create_model(config.model_config)
        self.model.load_state_dict(self._load_weight_state_dict(config))
        self.model.post_load()  # finalize weights (e.g. quantized layout conversion)

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        num_tokens = self.num_pages * config.page_size
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.kv_dtype,
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
                # fp32 recurrent state: the gdn_hip HIP kernels read/update conv+ssm state in place
                # in fp32 (also more accurate than the bf16 the Triton path stored each step).
                dtype=torch.float32,
                # MINISGL_SSM_BF16=1 stores the (large) ssm_state in bf16 instead — halves its HBM
                # (~2x max_running_req), compute still fp32 in-register (gdn_hip GDN_DISPATCH_SSM).
                # Default fp32 (most accurate). conv_state always stays fp32.
                ssm_dtype=(torch.bfloat16 if os.environ.get("MINISGL_SSM_BF16", "0") != "0"
                           else torch.float32),
                device=self.device,
            )
            # Settle causal_conv1d's per-process in-place batch_ptr autotune on a private
            # scratch BEFORE the first real batch (3c-3) — an unwarmed first prefill is
            # op-sequence-sensitive (NaN/0/OOM). Writes no real state.
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
            self.ctx.cca_state = self.cca_state = CCAStateCache(
                num_cca_layers=mc.num_cca_layers,
                num_slots=config.max_running_req + 2,  # +1 NULL block, +1 dummy
                conv_dim=mc.cca_conv_dim,
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
                f"(conv_dim={mc.cca_conv_dim}, conv_width={mc.cca_conv_width})"
            )
        else:
            self.cca_state = None  # type: ignore[var-annotated]  # CCAStateCache | None

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        if config.model_config.is_moe:
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

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
        )

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
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
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
        if config.enable_ep:
            # new_group must be called on EVERY process for each group (collective construction), so
            # build all tp_size nccl dp-subgroups and keep the one this process belongs to. An nccl
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
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # Cast bf16 weights/biases to the model dtype, but PRESERVE quantized tensors:
            # int packs (qweight/qzeros) and fp16 scales must keep their checkpoint dtype.
            def _cast(k: str, v: torch.Tensor) -> torch.Tensor:
                if not v.is_floating_point() or k.endswith(".scales"):
                    return v
                # ZAYA experts stay fp8: the F8_E4M3 weight must NOT be upcast (that re-inflates
                # ~8 GB fp8 -> ~16 GB bf16 and OOMs the 16 GB card), and its per-channel fp32
                # weight_scale must keep fp32. Dequant is deferred to compute (_GroupedFP8Experts).
                if v.dtype == torch.float8_e4m3fn:
                    return v
                if k.endswith(".weight_scale"):
                    # fp8 (ZAYA) per-channel scales are fp32 and MUST stay fp32; compressed-tensors
                    # int4 scales ship fp16 OR bf16 -> normalize to fp16 (the op's + the linear
                    # buffer's declared scale dtype) so a bf16-scale head and an fp16-scale backbone
                    # both load against the same float16 buffer.
                    return v if v.dtype == torch.float32 else v.to(torch.float16)
                # GDN gating params stay fp32 (A_log ships fp32; dt_bias ships bf16 -> upcast).
                # The kernels + the model's nn.Parameter dtype both require fp32 here.
                if k.endswith((".A_log", ".dt_bias")):
                    return v.to(torch.float32)
                # ZAYA router balancing_biases is an fp32 buffer (added to the fp32 router softmax;
                # ships bf16, upcast to keep the buffer fp32 and the choice numerics faithful). The
                # ResidualScaling affines run in fp32 on the fp32 residual stream (scale_residual_merge
                # ships bf16 -> upcast so the merge stays fp32). CCA conv/temp can stay bf16 (post_load
                # upcasts them to fp32 once for the kernel-weight cache).
                if k.endswith(
                    (
                        ".balancing_biases",
                        ".hidden_states_scale",
                        ".hidden_states_bias",
                        ".residual_scale",
                        ".residual_bias",
                    )
                ):
                    return v.to(torch.float32)
                return v.to(self.dtype)

            return {k: _cast(k, v) for k, v in load_weight(config.model_path, self.device)}

    def _recurrent_state_bytes(self, config: EngineConfig) -> int:
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
            ssm_itemsize = 2 if os.environ.get("MINISGL_SSM_BF16", "0") != "0" else 4
            ssm = (
                mc.num_gdn_layers * num_slots * div_even(mc.linear_num_value_heads, tp)
                * mc.linear_value_head_dim * mc.linear_key_head_dim * ssm_itemsize
            )
            total += conv + ssm
        if getattr(mc, "is_cca_hybrid", False):
            # conv_states (num_cca_layers, num_slots, conv_dim, conv_kernel) fp32 +
            # prev_hs     (num_cca_layers, num_slots, hidden_size) fp32
            conv = mc.num_cca_layers * num_slots * mc.cca_conv_dim * mc.cca_conv_width * 4
            prev = mc.num_cca_layers * num_slots * mc.hidden_size * 4
            total += conv + prev
        return total

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
                * mc.num_layers
            )
        else:
            cache_per_page = (
                2  # key + value
                * mc.head_dim
                * div_even(mc.num_kv_heads, config.tp_info.size, allow_replicate=True)
                * config.page_size
                * self.kv_dtype.itemsize
                * mc.num_layers
            )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            # Reserve the fixed GDN/CCA recurrent-state cache, which is allocated AFTER the KV pool
            # and scales with max_running_req. Without this the KV pool takes the whole budget and the
            # state alloc OOMs — the reason GDN 35B on 16 GB needed a manual --max-running-requests cap
            # ([B2]). Subtracting it up front co-sizes the two caches automatically; it is 0 for dense/
            # MHA/MLA models, so their sizing is unchanged.
            state_memory = self._recurrent_state_bytes(config)
            available_memory = (
                int(config.memory_ratio * old_free_memory) - model_memory - state_memory
            )
            num_pages = available_memory // cache_per_page
            if state_memory:
                logger.info(
                    f"Reserved {mem_GB(state_memory)} for GDN/CCA recurrent state "
                    f"({config.max_running_req} slots); KV pool gets the remainder"
                )

        assert num_pages > 1, (
            "Not enough memory for KV cache after reserving recurrent state; reduce "
            "--max-running-requests or --num-pages"
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

    def forward_batch(
        self, batch: Batch, args: BatchSamplingArgs, return_hidden: bool = False
    ):
        assert torch.cuda.current_stream() == self.stream
        _maybe_profile()
        extra = None
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
        out = ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)
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
        ``last_hidden`` is the post-final-norm hidden ``[sum(extend_len), hidden]`` (pre-lm_head) and
        ``aux_hidden`` is the stacked captured decoder layers ``[num_capture_layers, sum(extend_len),
        hidden]`` (or ``None`` if no capture layers are programmed). Capture layers are programmed via
        ``model.set_capture_layers`` at proposer init. No sampling and no ``complete_one``: the
        scheduler owns acceptance, commit, and req advancement.

        MLA verify is CUDA-graph captured: when the batch is graphable (uniform K+1 query tokens, bs
        fits a captured size) ``replay_verify`` runs the staged forward as one graph replay (the
        scheduler has copied input_ids/positions/out_loc into the static buffers). Otherwise — a
        partial-K step, or a non-MLA backend — it falls back to the eager forward."""
        assert torch.cuda.current_stream() == self.stream
        # v2 S4: the FUSED-TiDAR custom-mask verify forward has its own captured graph (distinct qlen +
        # a static dense mask). Check it first — its batch carries `fused_verify=True` and fused_qlen
        # query tokens, so it never collides with the K+1 two-forward verify graph below. Logits-only.
        if self.graph_runner.can_use_fused_verify(batch):
            return self.graph_runner.replay_fused_verify(batch)
        if self.graph_runner.can_use_verify_graph(batch):
            return self.graph_runner.replay_verify(batch, return_hidden)
        with self.ctx.forward_batch(batch):
            return self.model.forward(return_hidden=return_hidden)

    def capture_spec_verify_graphs(
        self, needs_hidden: bool, num_aux: int, bs_list: "list[int]"
    ) -> None:
        """Capture the MLA spec-decode verify graphs. Called by the scheduler AFTER the proposer is
        built and the target's aux-capture layers are programmed (so the captured forward stashes the
        hidden states the draft head consumes). No-op if graphs are disabled / non-MLA backend."""
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
            )

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

    def shutdown(self) -> None:
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
            if config.page_size != 1:
                override("page_size", 1)
                logger.warning_rank0("spec-decode (MHA): overriding page_size -> 1 (rollback)")
            # All non-MLA backbones now cudagraph-capture the spec-VERIFY forward: the HIP attn
            # verify-capture (S1) is model-agnostic (pure MHA works by itself), and the recurrent
            # backbones thread their per-token state through static scratch buffers — CCA via
            # CCAVerifyGraphCapture, GDN via GDNVerifyGraphCapture. So keep graphs ON for MHA, GDN,
            # and CCA alike. (page_size stays 1 above for per-token rollback.) The FUSED TiDAR
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
