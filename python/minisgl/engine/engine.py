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
    enable_custom_ar_distributed,
    enable_custom_ar_ep,
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


class Engine:
    def __init__(self, config: EngineConfig):
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

            # Ring STRIDE per sequence = window + spec block. A plain window-sized ring (stride == W)
            # is correct for single-query decode/extend, but a K+1 spec-VERIFY stores its whole block
            # (anchor + drafts) into the ring BEFORE attention; the rejected drafts then land in slots
            # that COLLIDE with the live window (position p and p+W share slot p%W once len >= W), so
            # the next step's window gather reads stale speculative keys. Widening the stride to
            # window + num_draft + 1 gives the speculative block its own disjoint slots, so a rejected
            # draft never overwrites a valid-window slot (position q and any window position p differ by
            # < stride => distinct mod stride). No spec -> stride == window (byte-identical to Track A).
            spec_block = (self.spec_config.num_draft + 1) if self.spec_config is not None else 0
            swa_stride = mc0.sliding_window + spec_block
            self.ctx.swa_ring_stride = swa_stride
            swa_slots = (config.max_running_req + 2) * swa_stride  # +1 NULL, +1 dummy
            self.ctx.swa_kv_cache = self.swa_kv_cache = MHAKVCache(
                num_kv_heads=mc0.num_kv_heads,
                num_layers=mc0.num_swa_layers,
                head_dim=mc0.head_dim,
                num_pages=swa_slots,
                page_size=1,  # ring is addressed by absolute slot; no page grouping
                device=self.device,
                dtype=self.kv_dtype,
            )
            logger.info_rank0(
                f"SWA ring KV: {mc0.num_swa_layers} layers x {swa_slots} slots "
                f"(window={mc0.sliding_window}, stride={swa_stride}, {config.max_running_req} seqs)"
            )
        else:
            self.swa_kv_cache = None  # type: ignore[assignment]

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
            max_running_req=config.max_running_req,
            cam=self.cam,
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
            # falls back to RCCL otherwise). Graph-safe + ~1.3x on the small decode/verify tensors; large
            # (prefill) all_reduces exceed the slot and self-fall-back to RCCL. Cap the IPC slot at 8 MB
            # (covers any decode/verify batch) so the fine-grained buffers stay small.
            car_max_bytes = min(
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize,
                8 * 1024 * 1024,
            )
            enable_custom_ar_distributed(config.tp_info, tp_cpu_group, car_max_bytes)
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

            return {
                k: _cast(k, v)
                for k, v in load_weight(
                    config.model_path, self.device, spec_algorithm=config.spec_algorithm
                )
            }

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
            ssm_itemsize = 4 if os.environ.get("MINISGL_SSM_BF16", "1") == "0" else 2
            ssm = (
                mc.num_gdn_layers * num_slots * div_even(mc.linear_num_value_heads, tp)
                * mc.linear_value_head_dim * mc.linear_key_head_dim * ssm_itemsize
            )
            total += conv + ssm
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
            local_kv = div_even(mc.num_kv_heads, tp, allow_replicate=True)
            spec_block = (config.spec_config.num_draft + 1) if config.spec_config is not None else 0
            swa_stride = mc.sliding_window + spec_block
            total += (
                2 * mc.num_swa_layers * num_slots * swa_stride
                * local_kv * mc.head_dim * self.kv_dtype.itemsize
            )
        return total

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
        override = os.environ.get("MINISGL_DRAFT_RESERVE_GB")
        if override is not None:
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
            # MINISGL_DFLASH_QUANT=fp8|int8 weight-only quantizes the draft linears to ~half memory.
            if (os.environ.get("MINISGL_DFLASH_QUANT", "") or "").strip():
                total //= 2
            # + working-set headroom for the drafter's EAGER forward (DFlash denoise / EAGLE3 step) and
            # its spec-verify transient. This is NOT covered by the captured-graph reserve (propose runs
            # eager for DFlash/EAGLE3), and it is the ~340 MB that OOMs a tight 27B+DFlash boot at warmup
            # (past capture). Default 0.7 GB; tune via MINISGL_DRAFT_RESERVE_MARGIN_GB (0 disables).
            margin = int(float(os.environ.get("MINISGL_DRAFT_RESERVE_MARGIN_GB", "0.7")) * (1 << 30))
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
            qlen = sc.num_draft + 1
            vbs = max((b for b in bs_list if b <= config.max_running_req), default=max_bs)
            T = vbs * qlen
            total += T * vocab * f32 + 3 * T * i32
            total += (1 + _GRAPH_ASSUMED_AUX) * T * hidden * dt  # last_hidden + aux_hidden
            total += _GRAPH_ACT_MULT * T * hidden * dt
        total = int(total * _GRAPH_ROUNDUP)
        margin = os.environ.get("MINISGL_GRAPH_RESERVE_MARGIN_GB")
        if margin:  # non-empty (empty env string is ignored)
            total += int(float(margin) * (1 << 30))
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
                * mc.num_kv_layers  # == num_layers for MLA (all-attention); matches pool alloc
            )
        else:
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
            model_memory = old_free_memory - new_free_memory
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
            available_memory = (
                int(config.memory_ratio * old_free_memory)
                - model_memory
                - state_memory
                - draft_memory
                - graph_memory
            )
            num_pages = available_memory // cache_per_page
            if state_memory:
                logger.info(
                    f"Reserved {mem_GB(state_memory)} for GDN/CCA recurrent state "
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

        assert num_pages > 1, (
            "Not enough memory for KV cache after reserving recurrent state / draft model / "
            "CUDA-graph buffers; reduce --max-running-requests, lower --memory-ratio, or set "
            "--num-pages"
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
        if self.graph_runner.can_use_fused_verify(batch):
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
