from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
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
        _adjust_config(config)

        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
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
        # prefix-cacheable — see GDNSlotManager) AND eager (GDN cudagraph out of scope) — both
        # forced below / in Scheduler.__init__.
        mc = config.model_config
        if mc.is_gdn_hybrid:
            self.ctx.gdn_state = self.gdn_state = GDNStateCache(
                num_gdn_layers=mc.num_gdn_layers,
                num_slots=config.max_running_req + 2,  # +1 NULL block, +1 dummy
                conv_dim=mc.gdn_conv_dim,
                conv_kernel=mc.linear_conv_kernel_dim,
                num_v_heads=mc.linear_num_value_heads,
                head_v_dim=mc.linear_value_head_dim,
                head_k_dim=mc.linear_key_head_dim,
                dtype=self.dtype,
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
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        # GDN-hybrid models run eager (cudagraph out of scope): an empty bs list disables
        # capture (max_graph_bs -> 0, can_use_cuda_graph -> False). Dense path unchanged.
        cuda_graph_bs = [] if self.gdn_state is not None else config.cuda_graph_bs
        if self.gdn_state is not None:
            logger.info_rank0("GDN-hybrid model: CUDA graph disabled (eager only)")
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
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
                return v.to(self.dtype)

            return {k: _cast(k, v) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.kv_dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
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

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        if is_rocm():
            backend = "triton_rdna4"  # tuned RDNA4 unified attention (gfx1201)
        else:
            backend = (
                "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
            )
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    # The RDNA4 unified Triton kernel requires a KV block size that is a multiple of 16.
    if "triton_rdna4" in config.attention_backend and config.page_size % 16 != 0:
        override("page_size", 16)
        logger.warning_rank0("Page size is overridden to 16 for the triton_rdna4 backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
