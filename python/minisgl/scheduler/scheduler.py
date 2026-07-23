from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
import torch.distributed as dist
import torch.profiler
from minisgl.cam.memory import _canon_subject   # #10 optional subject canonicalization (env-gated no-op)
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.spec import (
    AcceptResult,
    ProposeContext,
    make_proposer,
    probs_from_logits,
    verify_greedy,
    verify_sampled,
)
from minisgl.spec.accept_gpu import accept_greedy_ondevice, truncate_at_eos_ondevice
from minisgl.utils import div_ceil, init_logger, load_tokenizer, resolve_stop_token_ids

from .cache import CacheManager
from .cca_slots import CCASlotManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .ep import SchedulerEPMixin
from .gdn_slots import GDNSlotManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerEPMixin, SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # GDN-hybrid models MUST use the non-radix ("naive") prefix cache: GDN recurrent state
        # is not prefix-cacheable, and a radix hit would report cached_len>0 with no state behind
        # it (silent garbage). Force it here; dense models keep config.cache_type.
        cache_type = config.cache_type
        # GDN AND CCA recurrent state are both non-prefix-cacheable UNLESS we checkpoint it: a plain
        # radix hit would report cached_len>0 with no recurrent state behind it (silent garbage).
        # The --gdn-radix flag (default on) opts into the recurrent-radix cache, which snapshots the linear-attention
        # recurrent state at page-aligned prefix-commit boundaries and restores it on a hit (lossless
        # under the bit-exact recurrent kernel; see radix_cache.py). Default OFF -> force 'naive' as
        # before, so the recurrent path is byte-unchanged unless explicitly enabled.
        has_recurrent_state = (
            self.engine.gdn_state is not None or self.engine.cca_state is not None
        )
        self._rec_radix = False
        # Recurrent radix COMPOSES with expert-parallelism: EP's ep_loop runs the SAME shared prep path
        # as the normal loop — _finish_prepare (recurrent-state RESTORE) + _process_last_data /
        # _free_req_resources (SNAPSHOT) — and it is synchronous (the ordering the snapshot needs). EP
        # shards only the MoE experts; the CCA/GDN recurrent state is DP-local and untouched by EP, so
        # the per-replica snapshots/restores stay consistent (EP-over-TP ranks run lockstep-identical
        # batches). Spec-decode is DIFFERENT and stays gated: it runs its OWN decode-time snapshot/
        # restore over the same recurrent state (verify-state install) with prompt-dependent
        # losslessness, so combining two snapshot systems there is unsafe -> force naive under spec.
        _rec_radix_ok = self.engine.spec_config is None
        # CCA (ZAYA) is EXCLUDED from recurrent radix. This is NOT the recurrent state's fault: the
        # (conv_states, prev_hs) snapshot is captured/restored byte-faithfully AND the reused prefix
        # keys are bit-identical to a fresh forward (both verified: tools/cca_radix_whitebox.py and
        # tools/cca_kv_seam.py -> 0.0 diff). The real cause is that a radix HIT routes the request
        # through the CHUNKED / paged-extend prefill path (attn_prefill_paged._hip_prefill_paged, the
        # cached_len>0 branch in attention/hip.py) instead of the single-pass full-prefill kernel
        # (attn_hip.flash_prefill) a naive short-prompt request uses. The two attention kernels are only
        # ULP-equal (flash block-wise online-softmax vs single-pass), and that tiny per-token difference
        # COMPOUNDS through the 40-layer CCA network + long greedy chain-of-thought into DIFFERENT (and,
        # per GSM8K, systematically worse) final answers. Evidence it is the chunked-vs-single-pass path,
        # not fp8 / cudagraph / concurrency:
        #   * GSM8K n=200 radix-ON: bf16 KV 25.5%, fp8 KV 37.5% (both << naive 45%) -> not fp8.
        # NOTE (2026-07-14): the earlier "CCA prefix reuse is fundamentally lossy" gate was WRONG about
        # the cause. The chunked/paged-extend prefill divergence was NOT attention-softmax reassociation
        # — it was rocBLAS bf16 GEMM M-dependence in the CCA projections + router + lm_head (a chunk runs
        # those at a different M than a single pass; non-associative float => ~1 ULP, amplified by the
        # int8/fp8 downcast). Routing those dense linears through the engine's fixed-tile WMMA GEMM
        # (layers/minv.py::minv_linear) makes chunked prefill BIT-IDENTICAL to single-pass (verified 0.0
        # across all 40 CCA layers, tools/cca_chunk_bisect.py), so recurrent radix is now lossless for
        # CCA too. Both GDN and CCA recurrent state are prefix-cacheable; the only remaining gate is
        # spec-decode (both snapshot the recurrent state). See [[cca-prefix-cache-gemm-m-dependence]].
        if has_recurrent_state and cache_type != "naive":
            if config.gdn_radix and _rec_radix_ok:
                self._rec_radix = True
                cache_type = "recurrent_radix"
                logger.warning_rank0(
                    "recurrent-state hybrid model: using recurrent radix prefix cache (page-aligned "
                    "recurrent-state snapshots reused on prefix hits; --no-gdn-radix to disable)"
                )
            else:
                why = "GDN/CCA recurrent state is not prefix-cacheable (--no-gdn-radix set)"
                if config.gdn_radix and not _rec_radix_ok:
                    why = "recurrent radix is not supported with spec-decode (both snapshot the recurrent state)"
                logger.warning_rank0(
                    f"recurrent-state hybrid model: forcing prefix cache 'naive' (was {cache_type!r}); "
                    + why
                )
                cache_type = "naive"
        # SWA-radix prefix caching (Laguna / any sliding-window-attention hybrid). Its FULL-attention
        # layers are already radix-cacheable; its SLIDING layers keep a window-bounded ring whose
        # boundary is transient — so, exactly like GDN/CCA recurrent state, we SNAPSHOT the window at
        # the page-aligned prefix boundary and RESTORE it on a hit (rdna4.py::_swa_prefill_extend runs
        # the cross-boundary extend; PROVEN bit-identical to cold with a BC front-pad). We reuse the
        # snapshot-capable radix ('recurrent_radix': match-cap to a snapshotted boundary + attach), and
        # gate off spec-decode (a second snapshot system) like recurrent radix. Feature-flagged
        # (MINISGL_SWA_RADIX, default off) so the naive SWA path is byte-unchanged until proven.
        mc0 = config.model_config
        self._swa_radix = False
        self._swa_snap = None
        _swa_on = os.environ.get("MINISGL_SWA_RADIX", "0") != "0"
        # SWA-radix COMPOSES with spec-decode (unlike recurrent radix). The window snapshot/restore is
        # made ring-STRIDE-aware (SWAWindowSnapshotter takes swa_ring_stride = window + num_draft + 1
        # under spec): a snapshot reads/writes the committed window at slots table_idx*R + pos%R, exactly
        # the addressing rdna4.py's store/gather/decode use, so the reusing request's spec decode/verify
        # gathers the right window. The speculative block lives in the ring's disjoint tail slots and is
        # never in the committed window [boundary-Wp, boundary), so it never corrupts a snapshot. The
        # prefix-HIT restore runs at the reusing request's PREFILL (a non-spec_verify batch, see
        # _restore_swa_states / the _finish_prepare gate), before any spec propose/verify — no ordering
        # conflict with the widened-ring verify path Track B added.
        if getattr(mc0, "is_swa_hybrid", False) and cache_type != "naive":
            if _swa_on:
                self._swa_radix = True
                cache_type = "recurrent_radix"
                logger.warning_rank0(
                    "SWA-hybrid model: SWA-radix prefix cache ENABLED (page-aligned sliding-window "
                    "snapshots reused on prefix hits; MINISGL_SWA_RADIX=0 to disable)"
                    + ("; spec-decode active — snapshot/restore stride = window + num_draft + 1"
                       if self.engine.spec_config is not None else "")
                )
            else:
                logger.warning_rank0(
                    f"SWA-hybrid model: forcing prefix cache 'naive' (was {cache_type!r}); "
                    "SWA-radix disabled (MINISGL_SWA_RADIX=0)"
                )
                cache_type = "naive"
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        # runaway-generation-kv-guard: a single degenerate no-EOS generation can grow its context to
        # ~100% of the KV pool and then crawl (every decode step attends over the whole pool) while
        # starving every other request. This second, POOL-RELATIVE cap force-finishes any one request
        # whose KV footprint (device_len tokens ~= its allocated pages) reaches a fraction of the whole
        # pool, independent of its max_tokens. OFF by default (frac 0 -> budget 0 -> the decode-loop
        # check below is a no-op), so it never touches a serve that hasn't opted in; when set it never
        # fires below the fraction, so legitimate long contexts up to the budget are unaffected.
        _kv_frac = float(os.environ.get("MINISGL_CAM_REQ_KV_FRAC", "0") or 0)
        self._req_kv_budget_tokens = (
            int(_kv_frac * self.cache_manager.num_pages * config.page_size)
            if 0.0 < _kv_frac <= 1.0
            else 0
        )
        if self._req_kv_budget_tokens:
            logger.info_rank0(
                f"runaway-generation-kv-guard active: per-request KV budget "
                f"{self._req_kv_budget_tokens} tokens ({_kv_frac:.2f} of pool)."
            )
        # GDN recurrent-state slot lifecycle — active ONLY for GDN-hybrid models (engine
        # constructs the state cache in 3d). None (inert) for every dense model today, so
        # the dense scheduling path below is unchanged.
        self.gdn_slots = (
            GDNSlotManager(self.engine.gdn_state)
            if self.engine.gdn_state is not None
            else None
        )
        # ZAYA CCA recurrent-state slot lifecycle — active ONLY for CCA-hybrid (Zaya) models
        # (engine builds the state cache when is_cca_hybrid). None (inert) for every other model,
        # so the scheduling path below is unchanged.
        self.cca_slots = (
            CCASlotManager(self.engine.cca_state)
            if self.engine.cca_state is not None
            else None
        )
        # Recurrent-radix prefix caching binds to the single active recurrent state cache + its slot
        # manager (a model is GDN xor CCA, never both). None unless --gdn-radix enabled it above.
        if self._rec_radix and self.gdn_slots is not None:
            self._rec_cache, self._rec_slots = self.engine.gdn_state, self.gdn_slots
        elif self._rec_radix and self.cca_slots is not None:
            self._rec_cache, self._rec_slots = self.engine.cca_state, self.cca_slots
        else:
            self._rec_cache, self._rec_slots = None, None
        # SWA-radix window snapshotter (parallel to _rec_cache; a model is SWA xor GDN/CCA). It clones
        # the sliding-window ring at a page-aligned boundary and restores it on a prefix hit. The
        # snapshot is stored on the radix node's rec_state field (opaque) and stashed via the shared
        # _pending_rec_snap dict — reused verbatim since a SWA model never also has recurrent state.
        if self._swa_radix and self.engine.swa_kv_cache is not None:
            from minisgl.kvcache.swa_window import SWAWindowSnapshotter

            # Stride-aware: pass the engine's ring stride (swa_ring_stride = window + num_draft + 1 under
            # spec, else window) so the snapshot addresses the ring like store/gather/decode do. Without
            # spec this is just the window, so Track A stays byte-identical.
            self._swa_snap = SWAWindowSnapshotter(
                self.engine.swa_kv_cache,
                config.model_config.sliding_window,
                ring_stride=getattr(self.engine.ctx, "swa_ring_stride", None),
            )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        # whitened-GTE subject key (MINISGL_CAM_GTE_KEY=1): wire the served tokenizer as the ids->text
        # decoder the key needs, then reindex any store loaded at CAM build (which fell back to base-embed
        # keys because no decoder was wired yet) into the GTE key space. No-op unless GTE mode is active.
        _cam = getattr(self.engine, "cam", None)
        if _cam is not None and getattr(_cam, "_gte", None) is not None:
            _cam._decode = lambda ids: self.tokenizer.decode(list(ids)).strip()
            _cam.reindex()
        # rank0-authoritative store persistence: only the tp-primary (== the metrics emitter) writes the
        # store to disk. One writer => no TP save race; the writer reads the true primary at boot before
        # its own repair-save, so its recovered_from_backup flag/gauge is exact. Restore stays on every
        # rank (they each serve deliveries from their in-memory store).
        if _cam is not None:
            _cam._persist_owner = config.tp_info.is_primary()
        # FULL end-of-generation set: generation_config `eos_token_id` (often a LIST) ∪ tokenizer ∪
        # config. Multi-EOS models (GLM-4.x: [154820,154827,154829]) end turns on a token other than
        # the tokenizer's single EOS, so honouring only that one leaves them generating forever.
        self.eos_token_ids = set(resolve_stop_token_ids(config.model_path, self.tokenizer))
        # Hand the resolved EOS set to the sampler so it can suppress end-of-turn tokens for rows still
        # inside a <think> span (the reasoning gate — see _build_eos_suppress / BatchSamplingArgs).
        if self.eos_token_ids:
            self.engine.sampler.eos_token_ids = torch.tensor(
                sorted(self.eos_token_ids), dtype=torch.int64, device=self.device
            )
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Speculative-decode proposer (n-gram / MTP / draft-model). None unless spec is enabled.
        self._proposer = (
            make_proposer(self.engine.spec_config, self.engine)
            if self.engine.spec_config is not None
            else None
        )
        # Target hidden-state capture wiring. A draft-head proposer (MTP / EAGLE3 / DFlash) declares
        # `needs_last_hidden` and/or `capture_layer_ids`; the engine then asks forward_verify to
        # return the target's hidden states and feeds them back into ProposeContext for the next
        # propose. A pure n-gram proposer declares neither, so capture stays OFF and a normal serve
        # pays nothing. Aux-layer capture is programmed into the target model ONCE here.
        self._spec_needs_last_hidden = False
        self._spec_capture_layer_ids: List[int] | None = None
        # Prompt-prefill draft-KV seed (MINISGL_SPEC_PREFILL_SEED=1): run a hidden-capturing prefill
        # and seed the proposer's persistent draft KV over the prompt, so the FIRST draft already sees
        # full prompt context (lifts early-token acceptance). Off by default — it adds prefill work and
        # grows the per-step draft-KV stack by the prompt length; lossless either way (the verify
        # corrects every draft, so the seed can only change acceptance, never output). Only engages for
        # a proposer that owns a seedable per-req draft KV (MTP / EAGLE3 -> supports_prefill_seed).
        self._spec_seed_enabled = False
        # Under EP, an MTP draft HEAD that is a full EP-sharded MoE layer issues a data-dependent
        # number of collectives in propose, which an idle replica can't match with a fixed count. The
        # fix is to build that draft MoE REPLICATED (all experts local, no EP shard) so propose issues
        # no collectives — then MTP+EP reduces to matching only the verify forward (like EAGLE3). Detect
        # it by introspecting the MTP head's MoE. False for a dense/lookup draft (EAGLE3/DFlash/n-gram)
        # or when the MTP MoE is still EP-sharded. See _spec_num_ep_forwards.
        self._spec_draft_ep_replicated = False
        _mtp = getattr(self.engine.model, "mtp", None)
        if _mtp is not None:
            _experts = getattr(getattr(_mtp, "mlp", None), "experts", None)
            if _experts is not None and not getattr(_experts, "enable_ep", True):
                self._spec_draft_ep_replicated = True
        if self._proposer is not None:
            self._spec_needs_last_hidden = bool(self._proposer.needs_last_hidden)
            self._spec_capture_layer_ids = self._proposer.capture_layer_ids
            if self._spec_capture_layer_ids:
                self.engine.model.set_capture_layers(self._spec_capture_layer_ids)
            # TiDAR self-draft: the block_predict forward needs this scheduler's batch machinery
            # (token pool / paged-KV allocate / CCA+attn metadata), so bind it into the proposer.
            from minisgl.spec.tidar import TiDARProposer

            self._tidar_fused = False
            self._tidar_ddtree = False
            if isinstance(self._proposer, TiDARProposer):
                self._proposer.bind_block_predict(self._tidar_block_predict)
                # MINISGL_TIDAR_FUSED=1: route to the Phase-C single-forward fused step (verify+draft in
                # ONE forward) instead of the two-forward propose+verify path. Off by default.
                self._tidar_fused = os.environ.get("MINISGL_TIDAR_FUSED") == "1"
                if self._tidar_fused:
                    logger.info_rank0("spec-decode: TiDAR FUSED single-forward path ENABLED")
                # MINISGL_TIDAR_DDTREE=1: route to the DDTree draft-tree step (block_predict top-K ->
                # draft tree -> ancestor-mask verify -> walk -> linear commit). Off by default.
                self._tidar_ddtree = os.environ.get("MINISGL_TIDAR_DDTREE") == "1"
                if self._tidar_ddtree:
                    logger.info_rank0("spec-decode: TiDAR DDTree draft-tree path ENABLED")
                    self._ddtree_budget = int(os.environ.get("MINISGL_DDTREE_BUDGET") or "24")
            # MINISGL_DFLASH_DDTREE=1: route DFlash (block-diffusion drafter) through the same DDTree
            # draft-tree step — its one denoising forward emits per-position top-K marginals, from which
            # we build a B-budget tree, verify it in one ancestor-masked target forward, and commit the
            # accepted path via the linear verify. Off by default (vanilla DFlash = argmax chain).
            from minisgl.spec.dflash import DFlashProposer

            self._dflash_ddtree = False
            self._ddtree_budget = getattr(self, "_ddtree_budget", 0)
            if isinstance(self._proposer, DFlashProposer):
                self._dflash_ddtree = os.environ.get("MINISGL_DFLASH_DDTREE") == "1"
                if self._dflash_ddtree:
                    logger.info_rank0("spec-decode: DFlash DDTree draft-tree path ENABLED")
                    self._ddtree_budget = int(os.environ.get("MINISGL_DDTREE_BUDGET") or "32")
            self._spec_seed_enabled = (
                os.environ.get("MINISGL_SPEC_PREFILL_SEED") == "1"
                and bool(self._proposer.supports_prefill_seed)
                and (self._spec_needs_last_hidden or bool(self._spec_capture_layer_ids))
            )
            if self._spec_seed_enabled:
                logger.info_rank0("spec-decode: prompt-prefill draft-KV seed ENABLED")
            # Capture the verify CUDA graphs NOW — after the aux-capture layers are programmed above, so
            # the captured forward stashes the hidden/aux the draft head consumes. Supported for every
            # backbone now: MLA, pure MHA (generic attn verify-capture), and the recurrent hybrids CCA
            # (CCAVerifyGraphCapture) / GDN (GDNVerifyGraphCapture). No-op only when graphs are off
            # (--graph 0). Verify batches are all-running-reqs, so cap the captured sizes at
            # max_running_req.
            needs_hidden = self._spec_needs_last_hidden or bool(self._spec_capture_layer_ids)
            num_aux = len(self._spec_capture_layer_ids) if self._spec_capture_layer_ids else 0
            verify_bs = [b for b in self.engine.graph_runner.graph_bs_list
                         if b <= config.max_running_req]
            self.engine.capture_spec_verify_graphs(needs_hidden, num_aux, verify_bs)
            # v2 S4: ALSO capture the FUSED-TiDAR custom-mask verify graphs when the fused path is on.
            # fused_qlen is fixed per (block_size B, layout) — flat 1+B+B², segmented 1+B+B·(tp+B) — so
            # compute it once here (c0-independent) via the same layout helpers the fused step uses, and
            # capture logits-only graphs at that qlen. NOREP (qlen 1+B) auto-falls-back to eager.
            if self._tidar_fused and verify_bs:
                B = self._proposer.block_size
                seg = os.environ.get("MINISGL_TIDAR_SEG") == "1"
                from minisgl.spec.tidar_mask import fused_paged_layout
                if seg and self.engine.cca_state is not None:
                    from minisgl.spec.tidar_mask import fused_paged_layout_segmented

                    tp = int(self.engine.cca_state.conv_states.shape[-1])
                    fused_qlen = fused_paged_layout_segmented(0, B, tp, device=self.device)["n_query"]
                else:
                    _, _, fused_qlen, _ = fused_paged_layout(0, B, device=self.device)
                logger.info_rank0(
                    f"spec-decode: capturing FUSED-verify graphs (B={B} seg={seg} qlen={fused_qlen})")
                self.engine.capture_spec_fused_verify_graphs(fused_qlen, verify_bs)
            # ALSO capture the DDTree draft-TREE verify graphs when a DDTree path is on. tree_qlen =
            # budget+1 is fixed (budget read once at init above). The ancestor mask carries a per-key
            # column over the whole context, so the static mask buffer width is CAPPED at
            # MINISGL_DDTREE_MAXCTX (default 4096) — sequences past it fall back to eager (lossless).
            if (self._dflash_ddtree or self._tidar_ddtree) and verify_bs and self._ddtree_budget:
                tree_qlen = self._ddtree_budget + 1
                max_ctx = int(os.environ.get("MINISGL_DDTREE_MAXCTX") or "2048")
                # The ancestor mask keeps a per-key column over the whole context, so the static buffer
                # (max_bs*tree_qlen*max_ctx*4 B) is the dominant cost — cap the captured DDTree bs
                # (MINISGL_DDTREE_MAXBS, default 4) independently of the decode/linear-verify graphs.
                # Batches beyond the cap fall back to eager (lossless). Bench (bs=1) is unaffected.
                dmaxbs = int(os.environ.get("MINISGL_DDTREE_MAXBS") or "4")
                ddtree_bs = [b for b in verify_bs if b <= dmaxbs]
                logger.info_rank0(
                    f"spec-decode: capturing DDTREE-verify graphs "
                    f"(budget={self._ddtree_budget} qlen={tree_qlen} mask_ctx<={max_ctx} "
                    f"bs={ddtree_bs})")
                # MINISGL_DDTREE_STATIC=1: FIXED-topology draft tree (Medusa-style) instead of the
                # per-step best-first heap. Freezes the tree shape once so the ancestor mask is a
                # compile-time CONSTANT — the per-step O(n*depth) host mask rebuild becomes a slice-copy
                # of a baked [n,n] block (the brief's "bake the mask" for DDTree), and the parent indices
                # become compile-time constants (substrate for parent-indexed tree recurrence). Trades the
                # heap's per-step adaptivity for a baked mask; default OFF (dynamic heap) — A/B once the
                # retrained drafter lands. L = the drafter block width (TiDAR block_size / DFlash num_draft).
                # MINISGL_DDTREE_SEG=1 (implies STATIC): parent-indexed CCA tree recurrence (brief #3) —
                # the SEGMENTED ancestor-conv-context layout, so the tree conv reads each node's true
                # ancestor window (not packed neighbours). Fixed n_query (from the template) → capturable.
                self._ddtree_template = None
                self._ddtree_template_block = None
                self._ddtree_seg = False
                self._ddtree_seg_layout = None
                self._ddtree_seg_block = None
                # FUSE implies SEG (the fused replicas are appended to the segmented tree layout).
                _seg = (os.environ.get("MINISGL_DDTREE_SEG") == "1"
                        or os.environ.get("MINISGL_DDTREE_FUSE") == "1")
                if os.environ.get("MINISGL_DDTREE_STATIC") == "1" or _seg:
                    from minisgl.spec.ddtree import (build_static_template, template_ancestor_block,
                                                     fill_static_template, ddtree_paged_layout_segmented)
                    K = int(os.environ.get("MINISGL_DDTREE_TOPK") or "8")
                    L = int(getattr(self._proposer, "block_size", None)
                            or self.engine.spec_config.num_draft)
                    self._ddtree_template = build_static_template(self._ddtree_budget, K, L)
                    self._ddtree_template_block = template_ancestor_block(
                        self._ddtree_template, device=self.device)
                    logger.info_rank0(
                        f"spec-decode: DDTree STATIC template ENABLED (budget={self._ddtree_budget} "
                        f"K={K} L={L} n_nodes={self._ddtree_template.n_nodes}); ancestor mask BAKED")
                    self._ddtree_fuse = False
                    self._ddtree_fused_layout = None
                    self._ddtree_rank0 = None
                    self._ddtree_block_len = L
                    self._ddtree_mask_id = int(getattr(self._proposer, "mask_token_id", 0))
                    if _seg:
                        tp = int(self.engine.cca_state.conv_states.shape[-1]
                                 if self.engine.cca_state is not None else 2)
                        # Structure is topology-only (c0=0): precompute the seg layout + its tree-local
                        # [n_query,n_query] mask block ONCE; per step only fill token ids + offset by c0.
                        dummy = fill_static_template(self._ddtree_template, [[0] * K] * L, 0)
                        self._ddtree_seg_layout = ddtree_paged_layout_segmented(0, dummy, tp, device=self.device)
                        self._ddtree_seg_block = self._ddtree_seg_layout["mask"]  # [n_query, n_query]
                        self._ddtree_seg = True
                        logger.info_rank0(
                            f"spec-decode: DDTree SEGMENTED CCA recurrence ENABLED (conv_width={tp} "
                            f"n_query={self._ddtree_seg_layout['n_query']} vs n_nodes="
                            f"{self._ddtree_template.n_nodes}); ancestor conv-context inserted")
                        # MINISGL_DDTREE_FUSE=1 (implies SEG): fuse the next-block draft off the rank-0
                        # top path (brief #2) — append R_0..R_L replicas to the captured verify. On-path
                        # steps reuse R_{k+1} as the next tree's marginals (skip block_predict); off-path
                        # steps fall back to block_predict. Fixed shape (rank-0 path) -> still capturable.
                        if os.environ.get("MINISGL_DDTREE_FUSE") == "1":
                            from minisgl.spec.ddtree import (template_rank0_path,
                                                             ddtree_fused_paged_layout_segmented)
                            self._ddtree_rank0 = template_rank0_path(self._ddtree_template)
                            self._ddtree_fused_layout = ddtree_fused_paged_layout_segmented(
                                0, dummy, tp, self._ddtree_rank0, L, device=self.device)
                            self._ddtree_fused_block = self._ddtree_fused_layout["mask"]
                            self._ddtree_fuse = True
                            logger.info_rank0(
                                f"spec-decode: DDTree FUSED next-block ENABLED (rank0_depth="
                                f"{len(self._ddtree_rank0) - 1} replicas={len(self._ddtree_fused_layout['replica_rows'])} "
                                f"n_query={self._ddtree_fused_layout['n_query']}); next block off the top path")
                # Capture the ddtree verify graph at the ACTUAL query length (fused n_query if fused, seg
                # n_query if segmented, else tree_qlen=budget+1). After the template/seg/fuse setup.
                if getattr(self, "_ddtree_fuse", False):
                    tree_qlen = self._ddtree_fused_layout["n_query"]
                elif self._ddtree_seg:
                    tree_qlen = self._ddtree_seg_layout["n_query"]
                self.engine.capture_spec_ddtree_verify_graphs(tree_qlen, ddtree_bs, max_ctx)
        # uid -> last_hidden / aux_hidden of the verified position carried to the NEXT propose. Empty
        # unless a draft-head proposer requested capture (so n-gram serve allocates nothing).
        self._spec_last_hidden: dict[int, torch.Tensor] = {}
        self._spec_aux_hidden: dict[int, torch.Tensor] = {}
        # DFlash full-context conditioning (the z-lab fix): accumulate the target aux over ALL committed
        # positions and feed the whole prefix to the drafter, instead of a single last-token vector. The
        # drafter was trained/eval'd conditioned on the full context, so the 1-token feed ran it OOD
        # (~0.33 accept-len). Stored aux becomes 3D [num_aux, P, hidden] (P = committed positions); the
        # DFlash proposer branches on aux.dim(). MINISGL_DFLASH_FULLCTX=0 restores the legacy 1-token
        # feed (for A/B). MINISGL_DFLASH_CTX_WINDOW>0 caps P to the last W positions (perf/memory).
        self._dflash_fullctx = os.environ.get("MINISGL_DFLASH_FULLCTX", "1") not in ("0", "false", "no")
        self._dflash_ctx_window = int(os.environ.get("MINISGL_DFLASH_CTX_WINDOW", "0") or 0)

        # MINISGL_SPEC_ONDEVICE=1: compute greedy acceptance + EOS truncation with the on-device
        # vectorized chain (spec/accept_gpu.py) instead of the per-position argmax .cpu() + per-req
        # Python verify_greedy/keep loop. Byte-lossless (validated in tools/validate_*), and one
        # batched sync of small [num_reqs]/[sum kept] results replaces the per-req Python that scales
        # with batch — the concurrency lever, and the substrate for the future zero-sync overlap.
        # Falls back to the host path for constrained/ddtree/FORCE_N0 batches. Default OFF.
        self._spec_ondevice = os.environ.get("MINISGL_SPEC_ONDEVICE") == "1"

        # MINISGL_SPEC_SAMPLED=1: sampled (rejection-sampling) speculative verify — lets spec-decode
        # engage for NON-greedy reqs (temperature>0 / top_p<1, e.g. every RSA rollout) instead of
        # falling back to plain decode. Lossless DISTRIBUTIONALLY (the emitted tokens are drawn from
        # exactly the target's temp/top_k/top_p distribution; NOT byte-identical). v1: standard linear
        # verify only (not DDTree / fused-TiDAR), unconstrained reqs, host accept path. Default OFF.
        # See docs/SAMPLED_SPEC_VERIFY.md. Per-step generator seeded identically on every TP rank so the
        # rejection draws stay in lockstep (drafts are already broadcast; p is identical post-all_gather).
        # DDTree paths (dflash/tidar) are supported: their tree DISCOVERY stays greedy (a heuristic for
        # a good draft path) and the lossless LINEAR COMMIT routes through verify_sampled like any other
        # req — so DDTree engages under sampling (the tree's higher-acceptance benefit under sampling
        # needs SpecTr multi-candidate rejection, a follow-up). Fused-TiDAR has its own accept path
        # (_spec_decode_step_tidar_fused) that this route doesn't cover, so it stays excluded for now.
        self._spec_sampled = (
            os.environ.get("MINISGL_SPEC_SAMPLED") == "1"
            and not getattr(self, "_tidar_fused", False)
        )
        self._spec_step = 0
        self._spec_seed_base = int(os.environ.get("MINISGL_SPEC_SAMPLED_SEED", "42"))
        if self._spec_sampled:
            logger.info_rank0("spec-decode: SAMPLED (rejection-sampling) verify ENABLED")

        # DFlash-drafter training-data CAPTURE (MINISGL_ZAYA_CAPTURE_DIR=<dir>): dump the target's aux
        # taps + tokens per prefill position to seedbuf_<pid>_<n>.pt — the exact format
        # train_cca_drafter.py --seed-dir ingests. This lets the drafter be re-distilled ON minisgl's
        # OWN aux trajectory (the cross-engine fidelity fix: the drafter was OOD at 0.26 accept because
        # it trained on vLLM-ZAYA aux). Teacher-forcing capture — the driver re-feeds prompt+greedy-
        # continuation as one prefill; this hook forces the aux taps on and dumps every position. Serve
        # with --cache-type naive so the re-fed text is a full prefill (naive.match_prefix returns no
        # match -> no reuse; a radix hit would skip the forward -> no aux). Inert (a pure passthrough in
        # _forward) when the env is unset.
        self._capture_dir = os.environ.get("MINISGL_ZAYA_CAPTURE_DIR") or None
        self._capture_buf: List[dict] = []
        self._capture_n = 0
        # On-policy Draft-OPD capture (MINISGL_ZAYA_OPD_CAPTURE_DIR=<dir>): during DFlash spec decode,
        # dump the drafter's OWN trajectory per verify step — the seed aux at the last committed
        # position, the anchor token, the draft block, num_accepted, the target's correction, and the
        # target top-K logits — to opdbuf_<pid>_<n>.pt, the schema train_drafter.py --opd-dir ingests
        # (dflash-drafter/docs/CAPTURE_CONTRACT.md). Enables on-policy OPD + soft-KL ON minisgl. Inert
        # (guarded) unless the env is set; run the rollout with MINISGL_DFLASH_DDTREE=0 so the LINEAR
        # block maps 1:1 to the opdbuf schema (the tree path has no linear num_accepted).
        self._opd_dir = os.environ.get("MINISGL_ZAYA_OPD_CAPTURE_DIR") or None
        self._opd_buf: List[dict] = []
        self._opd_n = 0
        if self._opd_dir:
            os.makedirs(self._opd_dir, exist_ok=True)
            import atexit
            atexit.register(self._flush_opd)
            logger.info_rank0(f"DFlash on-policy OPD CAPTURE on -> {self._opd_dir}")
        if self._capture_dir:
            ids_env = os.environ.get("MINISGL_ZAYA_CAPTURE_LAYERS", "1,39,76")
            self._capture_layer_ids = [int(x) for x in ids_env.split(",") if x.strip()]
            if hasattr(self.engine.model, "set_capture_layers"):
                os.makedirs(self._capture_dir, exist_ok=True)
                self.engine.model.set_capture_layers(self._capture_layer_ids)
                import atexit
                atexit.register(self._flush_capture)
                logger.info_rank0(
                    f"DFlash CAPTURE on: aux layers {self._capture_layer_ids} -> {self._capture_dir}"
                )
            else:
                logger.warning_rank0(
                    "MINISGL_ZAYA_CAPTURE_DIR set but model has no set_capture_layers; capture OFF"
                )
                self._capture_dir = None

        # Structured-output (constrained decoding) state. Built lazily on the first constrained
        # request, so a plain serve never imports xgrammar. uid -> live GrammarMatcher.
        self._grammar_backend = None
        self._grammar_matchers: dict[int, object] = {}
        # Recurrent radix: in-flight recurrent-state checkpoints keyed by uid -> (page-aligned boundary,
        # cloned slot state). Stashed when a sequence's prefill crosses a page boundary (the slot then
        # holds state@boundary EXACTLY), and attached to the align_down radix node when its prefix is
        # inserted (tail/finish commit). Decouples "capture the aligned state" from "a node exists to
        # hang it on", and avoids mutating pages on a chunk commit. Popped on attach / free / abort.
        self._pending_rec_snap: dict[int, tuple[int, object]] = {}
        # Reasoning + structured output: while a constrained req is still inside its `<think>…</think>`
        # reasoning span, the grammar matcher must NOT advance or mask (else the JSON schema suppresses
        # the reasoning phase → truncated / CoT-leaked answers). uid -> think-close token id, present
        # only WHILE gated; the entry is dropped once that token is emitted, after which the matcher
        # enforces the schema on the answer. `think_close_delim` -> token id is resolved once per
        # delimiter string (generic; keyed off the reasoning parser, no model-name branch).
        self._grammar_think_gate: dict[int, int] = {}
        self._think_close_ids: dict[str, int | None] = {}
        # Reasoning BUDGET backstop: this model often reasons in long/unstructured prose and never
        # emits a clean `</think>`, so the gate above would never open → pages of CoT and NO JSON. We
        # count generated tokens WHILE a uid is gated (uid -> count) and, once it reaches the uid's
        # budget (uid -> budget; per-request override or the MINISGL_THINK_BUDGET default), FORCE the
        # think-close token into the stream and open the gate so the schema engages on the answer.
        self._grammar_think_count: dict[int, int] = {}
        self._grammar_think_budget: dict[int, int] = {}
        self._grammar_think_budget_default = int(
            os.environ.get("MINISGL_THINK_BUDGET", "1024") or 1024
        )
        # uids whose reasoning gate has already opened (</think> emitted) — so the plain-path arm
        # helper does not RE-arm a request after its reasoning phase ended. Cleared on free.
        self._think_gate_done: set[int] = set()
        # Escape hatch / A-B toggle: MINISGL_GRAMMAR_THINK_GATE=0 reverts to applying the schema from
        # token 0 even with thinking on (the pre-fix behavior — for demonstrating before/after).
        self._grammar_think_gate_enabled = (
            os.environ.get("MINISGL_GRAMMAR_THINK_GATE", "1") not in ("0", "false", "no")
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

        # ---- Prometheus metrics (backend-observed) ------------------------------------------------
        # Cumulative spec-decode counters (see server/metrics.py) + a wall-clock-throttled snapshot
        # push. Only the tp-primary emits (it owns the detokenizer link); the frontend sums replicas.
        # Cheap: the counters below are int adds at the accept site, and the gauge sample + StatsMsg
        # push happen at most once per MINISGL_METRICS_INTERVAL seconds — nothing per-token.
        self._metrics_enabled = os.environ.get("MINISGL_METRICS", "1") != "0"
        self._metrics_interval = float(os.environ.get("MINISGL_METRICS_INTERVAL", "0.5"))
        self._metrics_last_flush = 0.0
        self._m_dp_rank = config.dp_info.dp_rank
        self._m_spec_draft_tokens = 0
        self._m_spec_accepted_tokens = 0
        self._m_spec_emitted_tokens = 0
        self._m_spec_steps = 0

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        # debug, not info: this fires every idle loop and otherwise floods the log (drowning e.g. the
        # [rsa-timing] breakdown). Enable debug logging if you want the idle heartbeat back.
        logger.debug_rank0("Scheduler is idle, waiting for new reqs...")
        # DFlash capture: flush the seedbuf buffer whenever idle (the driver pauses between prompts and
        # is fully idle at the end) — atexit does NOT run on the container's SIGTERM, so this is how the
        # tail lands. Cheap no-op when not capturing / buffer empty.
        if self._capture_dir and self._capture_buf:
            self._flush_capture()
        if self._opd_dir and self._opd_buf:
            self._flush_opd()
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Grammar-constrained decode cannot overlap: the grammar matcher must be advanced by the
        # PREVIOUS batch's committed token BEFORE the next batch's bitmask is filled (in
        # _schedule_next_batch). The overlap order (schedule -> forward -> process-last) leaves the
        # mask one token stale, so a greedy model re-emits each grammar position before the matcher
        # catches up -> doubled/garbled output. When a constrained req is live, serialize: commit +
        # accept the last batch first, then schedule (like the spec-decode / EP host-sync loops).
        if last_data is not None and self._grammar_matchers:
            self._process_last_data(last_data)
            last_data = None

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # Establish the rank0->rank{1..} PUB/SUB fan-out before any request flows, so the first
        # message can't be lost to the ZMQ slow-joiner (which deadlocked the first request). No-op
        # for TP=1. See SchedulerIOMixin.establish_inter_rank_link.
        self.establish_inter_rank_link()
        # Speculative decoding runs in a dedicated synchronous loop: acceptance is a
        # data-dependent host-sync that fundamentally conflicts with the zero-sync overlap path
        # (see SPEC_DECODE.md §1). All GPU work runs on the engine stream, like the eager path.
        # Spec-decode AND expert-parallelism together: a single loop that both agrees the per-step
        # EP phase/size across replicas AND runs the spec step. Neither _spec_loop nor ep_loop alone
        # works — _spec_loop issues MoE collectives with no cross-replica agreement (deadlock), ep_loop
        # has no spec step. Must be checked BEFORE the spec-only / ep-only branches below.
        from minisgl.distributed import is_ep_over_tp
        if self.engine.spec_config is not None and self.engine.enable_ep and is_ep_over_tp():
            # EP-over-TP (dp=1): the TP ranks always run the SAME batch in lockstep, so the MoE
            # collectives are deterministic and match — the normal _spec_loop is safe (no cross-replica
            # agreement needed, unlike DP+EP). MTP works too: its draft head is built REPLICATED
            # (force_no_ep, Qwen3_5MoeSparseBlock), so propose issues plain-TP all_reduces while the
            # backbone verify does EP dispatch — all deterministic. So ALL proposers fall through to
            # _spec_loop below (no _spec_ep_loop, no MTP block).
            pass
        elif self.engine.spec_config is not None and self.engine.enable_ep:
            # DP+EP (independent replicas, different per-step batches): needs the cross-replica agreement
            # loop. Fail fast on the one unsupported combo: MTP with an EP-SHARDED draft head — its propose
            # issues a data-dependent number of MoE collectives the idle-replica lockstep can't match.
            if self.engine.spec_config.algorithm == "mtp" and not self._spec_draft_ep_replicated:
                raise RuntimeError(
                    "MTP spec-decode + DP+EP is unsupported unless the MTP draft MoE is built REPLICATED "
                    "(all experts local). Serve MTP without --enable-ep, or use EAGLE3/DFlash/TiDAR."
                )
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self._spec_ep_loop()
        if self.engine.spec_config is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self._spec_loop()
        # Expert parallelism runs a dedicated synchronous lockstep loop: every step issues the MoE
        # all_gather/all_reduce over the DP/EP group, so all replicas must agree the per-step
        # phase+size (one gloo all_reduce(MAX), OUTSIDE the graph) or the collectives deadlock. This
        # conflicts with the zero-sync overlap path, like spec decode. See SchedulerEPMixin.
        if self.engine.enable_ep:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.ep_loop()
        # Recurrent radix caches a sequence's linear-attention state at commit points by cloning its
        # live slot. That clone must be ordered AFTER the sequence's own forward and BEFORE any next
        # forward that could advance the slot — which the synchronous normal loop guarantees (schedule
        # -> forward -> process/commit, no forward launched ahead) but the overlap loop does not. So
        # run the non-overlap loop when recurrent radix is enabled. SWA-radix has the identical
        # ordering requirement — its window snapshot is cloned from the live ring at a commit point and
        # RESTORED (ring seed + metadata.swa_prefix) in _finish_prepare right before the forward, which
        # only the synchronous loop guarantees — so it forces the non-overlap loop too.
        if ENV.DISABLE_OVERLAP_SCHEDULING or self._rec_radix or self._swa_radix:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _flush_stats(self, force: bool = False) -> None:
        """Wall-clock-throttled scheduler metrics snapshot -> StatsMsg on the detokenizer link.

        Called from the I/O receive path (every loop variant funnels through receive_msg), so it fires
        during active serving without touching the per-token decode path. Only the tp-primary has the
        detokenizer socket; every other rank / offline scheduler no-ops via _emit_stats' guard. Gauges
        are sampled here (instantaneous); the spec counters are cumulative."""
        # MINISGL_BATCH_DEBUG=1: fine-grained batch-composition log on its OWN fast cadence (independent
        # of Prometheus / _metrics_enabled). For diagnosing RSA aggregation batching: align these lines
        # with the "[rsa] round N:" logs — kv_per_seq = kv_used/running exposes when long-context
        # aggregation prompts crowd the KV pool and collapse the effective batch to ~single-request.
        if os.environ.get("MINISGL_BATCH_DEBUG") == "1":
            _bd_now = time.time()
            if _bd_now - getattr(self, "_bd_last", 0.0) >= 0.2:
                self._bd_last = _bd_now
                _cm = self.cache_manager
                _run = len(self.decode_manager.running_reqs)
                _wait = len(self.prefill_manager.pending_list)
                _kvt = _cm.num_pages * _cm.page_size
                _kvu = (_cm.num_pages - len(_cm.free_slots)) * _cm.page_size
                _per = int(_kvu / _run) if _run else 0
                logger.info_rank0(
                    f"[batch] running={_run} waiting={_wait} kv={_kvu}/{_kvt} "
                    f"({100 * _kvu / max(_kvt, 1):.0f}%) kv_per_seq={_per}"
                )
        if not self._metrics_enabled:
            return
        now = time.time()
        if not force and now - self._metrics_last_flush < self._metrics_interval:
            return
        self._metrics_last_flush = now

        cm = self.cache_manager
        kv_total = cm.num_pages * cm.page_size
        kv_used = (cm.num_pages - len(cm.free_slots)) * cm.page_size
        gdn_total = gdn_used = 0
        slots = self.gdn_slots or self.cca_slots  # a model is GDN XOR CCA, never both
        if slots is not None:
            gdn_used = slots.num_active
            # slot 0 is the reserved NULL block -> subtract it from the advertised capacity.
            gdn_total = max(int(getattr(slots.state_cache, "num_slots", 0)) - 1, 0)

        from minisgl.message import StatsMsg

        _cam = self.engine.cam
        _cam_s = _cam.metrics_summary() if _cam is not None else {}
        self._emit_stats(
            StatsMsg(
                dp_rank=self._m_dp_rank,
                spec_draft_tokens=self._m_spec_draft_tokens,
                spec_accepted_tokens=self._m_spec_accepted_tokens,
                spec_emitted_tokens=self._m_spec_emitted_tokens,
                spec_steps=self._m_spec_steps,
                running_requests=len(self.decode_manager.running_reqs),
                waiting_requests=len(self.prefill_manager.pending_list),
                kv_tokens_total=int(kv_total),
                kv_tokens_used=int(kv_used),
                gdn_slots_total=int(gdn_total),
                gdn_slots_used=int(gdn_used),
                cam_facts=int(_cam_s.get("facts", 0)),
                cam_namespaces=int(_cam_s.get("namespaces", 0)),
                cam_evicted=int(_cam_s.get("evicted", 0)),
                cam_max_bank_load=int(_cam_s.get("max_bank_load", 0)),
                cam_crowded_banks=int(_cam_s.get("crowded_banks", 0)),
                cam_recovered_from_backup=int(_cam_s.get("recovered_from_backup", 0)),
                cam_index_nn_cos_max=float(_cam_s.get("index_nn_cos_max", 0.0)),
                cam_last_save_age_s=float(_cam_s.get("last_save_age_s", 0.0)),
            )
        )

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Recurrent radix: a chunk that just committed a page-aligned prefix left the slot
                    # holding state@boundary EXACTLY (complete_one advanced req.cached_len to the chunk
                    # end, which the forced page-aligned split keeps on a page multiple). Clone + stash
                    # it now; it attaches to the node@boundary when this sequence's prefix is inserted
                    # (tail/finish commit). No cache_req here → no page/handle mutation on the chunk.
                    if self._rec_cache is not None:
                        self._stash_rec_state(req)
                    elif self._swa_radix:
                        self._stash_swa_state(req)
                    continue
                next_token = next_tokens_cpu[i]
                # #100 POINTER delivery: for the first len(obj) steps, OVERRIDE the sampled token with the
                # exact object token from engine.cam (deliver_object_ids, computed in _prepare_cam). The
                # forced token is what's committed to the KV history (append_host) AND emitted, so the base
                # continuation conditions on the delivered object. After the object, sampling resumes.
                _dl = getattr(req, "_mem_deliver", None)
                if _dl is not None and req._mem_deliver_pos < len(_dl):
                    next_token = next_token.new_tensor(_dl[req._mem_deliver_pos])
                    req._mem_deliver_pos += 1
                # β (thinking budget) force-close: arm the reasoning gate for any thinking request
                # (grammar reqs are also armed in _build_grammar_bitmask — idempotent), then once the
                # reasoning budget is spent OVERRIDE the sampled token with </think> so the model stops
                # reasoning and produces its answer. This covers PLAIN / RSA rollouts, which the
                # grammar-bitmask backstop (constrained-only) never reached — the paper's bounded
                # workspace, and the fix for truncated-thinking-with-no-answer.
                self._maybe_arm_think_gate(req)
                _gate = self._grammar_think_gate.get(req.uid)
                if _gate is not None and self._think_gate_over_budget(req.uid) \
                        and int(next_token.item()) != _gate:
                    next_token = next_token.new_tensor(_gate)
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                # CAM seed-once: the object's first token has landed -> stop injecting this req's bank
                # (subsequent _stage_cam calls skip it; the base continues fluently). The
                # MINISGL_CAM_ALWAYS_INJECT debug knob keeps injecting every step (used to exercise the
                # captured decode tap, since with seed-once the object lands at prefill).
                if getattr(req, "mem_bank", None) is not None and not req._mem_placed \
                        and next_token == req._mem_seed \
                        and os.environ.get("MINISGL_CAM_ALWAYS_INJECT") != "1":
                    req._mem_placed = True
                eos_hit = (not req.sampling_params.ignore_eos) \
                    and (next_token in self.eos_token_ids)
                finished = eos_hit or (not req.can_decode)
                # runaway-generation-kv-guard (env MINISGL_CAM_REQ_KV_FRAC): force-finish a single
                # request whose KV footprint has grown past the pool-fraction budget, so one runaway
                # can't monopolize the pool. Budget 0 (default) -> disabled. device_len is prompt +
                # committed tokens ~= this req's KV token count (pages*page_size for MLA differs by
                # <page_size, immaterial at a 70%-of-pool cap).
                if not finished and self._req_kv_budget_tokens \
                        and req.device_len >= self._req_kv_budget_tokens:
                    logger.warning_rank0(
                        f"runaway-generation-kv-guard: request {req.uid} reached "
                        f"{req.device_len} KV tokens (>= budget {self._req_kv_budget_tokens}); "
                        f"force-finishing to protect the KV pool."
                    )
                    finished = True
                # Structured output: advance this req's grammar matcher with the committed token so the
                # next step's bitmask reflects the new state. Skip on finish (req is done). A terminated
                # grammar (complete JSON) is allowed to emit EOS, which the matcher won't accept — guard.
                if not finished:
                    gate = self._grammar_think_gate.get(req.uid)
                    m = self._grammar_matchers.get(req.uid)
                    if gate is not None:
                        # Reasoning phase (grammar OR plain-β): count this token toward the budget, and
                        # open the gate once </think> is emitted (the model's own close, or the budget-
                        # forced override above). For a grammar req the schema then engages on the NEXT
                        # token; for a plain/RSA req generation simply continues to the answer. Think
                        # tokens are NOT fed to the matcher (the schema starts fresh on the answer).
                        if next_token == gate:
                            self._clear_think_gate(req.uid)
                        else:
                            self._grammar_think_count[req.uid] = (
                                self._grammar_think_count.get(req.uid, 0) + 1
                            )
                    elif m is not None and not m.is_terminated():
                        m.accept_token(next_token)
                fr = ("stop" if eos_hit else "length") if finished else None
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token,
                                           finished=finished, finish_reason=fr))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    inserted = self.cache_manager.cache_req(req, finished=False)
                    # Recurrent radix: snapshot this prefix's recurrent state onto the inserted node so
                    # a future request sharing it can restore instead of re-prefilling.
                    if self._rec_cache is not None:
                        self._maybe_capture_rec_state(req, inserted)
                    elif self._swa_radix:
                        self._maybe_capture_swa_state(req, inserted)

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            # CAM control ops (mem_op: stats/facts/forget/namespaces/lookup/retrieve/...) produce a JSON
            # result computed from engine.cam — NO model generation. Answer them DIRECTLY here, before the
            # req ever enters a batch, instead of tokenising the result and force-emitting it one token per
            # 35B forward through the decode loop (O(result_tokens) forwards; a big facts/stats/audit dump
            # cost hundreds of decode steps — the /cam/stats-slow symptom). Deliveries (mem_subject) and
            # writes (mem_remember) carry NO mem_op and still take the normal generate path. TP-safe: the
            # req never joins the batch (nothing to diverge), both ranks apply side effects (forget/drop
            # stay in sync), and only rank0 emits the reply.
            if self.engine.cam is not None and getattr(msg.sampling_params, "mem_op", None):
                self._cam_direct_control(msg)
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                self._free_req_resources(req_to_free)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(req.table_idx)
        inserted = self.cache_manager.cache_req(req, finished=True)
        # Recurrent radix: snapshot the FINISHED sequence's recurrent state onto its inserted prefix
        # node BEFORE the slot is freed below, so the next turn of a multi-turn conversation (prompt =
        # this full sequence + new tokens) restores it instead of re-prefilling the whole history.
        if self._rec_cache is not None:
            self._maybe_capture_rec_state(req, inserted)
        elif self._swa_radix:
            self._maybe_capture_swa_state(req, inserted)
        # Release the GDN state slot (idempotent — overlap scheduling can free a req twice).
        # This single site covers both normal finish (via _process_last_data) and abort.
        if self.gdn_slots is not None:
            self.gdn_slots.free(req.uid)
        # Release the CCA conv-state slot (idempotent — overlap scheduling can free a req twice).
        if self.cca_slots is not None:
            self.cca_slots.free(req.uid)
        # Release any spec-decode proposer draft state (MTP persistent per-uid KV; n-gram no-op).
        if self._proposer is not None:
            self._proposer.free(req.uid)
        # Release the structured-output grammar matcher + reasoning gate/budget (idempotent).
        self._grammar_matchers.pop(req.uid, None)
        self._clear_think_gate(req.uid)
        # Drop any un-attached recurrent-state checkpoint (idempotent; frees the cloned slot state).
        self._pending_rec_snap.pop(req.uid, None)
        # Drop the reasoning-gate "done" marker (idempotent; the gate dicts are cleared via _clear_think_gate).
        self._think_gate_done.discard(req.uid)

    def _restore_rec_states(self, batch: Batch) -> None:
        """Install cached recurrent-state snapshots into the slots of prefill reqs that hit the
        recurrent radix. Only reqs with a snapshot (cache_handle.rec_state) and cached_len>0 restore;
        everyone else keeps their zeroed/continuation slot untouched."""
        for req in batch.reqs:
            handle = getattr(req, "cache_handle", None)
            rec_state = getattr(handle, "rec_state", None)
            if rec_state is None or req.cached_len == 0:
                continue
            # Restore ONLY on the initial prefix-hit pass (req.cached_len == the matched boundary). A
            # chunked continuation carries the SAME handle but a larger cached_len; re-restoring there
            # would clobber the state advanced by earlier chunks.
            if req.cached_len != handle.cached_len:
                continue
            slot = self._rec_slots.slot_for(req.uid)
            if slot is not None:
                self._rec_cache.load_slot(slot, rec_state)
                logger.info_rank0(
                    f"recurrent-radix HIT: uid={req.uid} restored recurrent state at "
                    f"cached_len={req.cached_len} (skips re-prefill of the shared prefix)"
                )

    def _stash_rec_state(self, req: Req) -> None:
        """Checkpoint a sequence's recurrent state at a page-aligned prefill boundary. Called at a
        chunk commit, when complete_one has set req.cached_len to the (forced page-aligned) chunk end
        and the slot holds state@that-boundary EXACTLY. We clone the slot and stash it by uid; it is
        attached to the align_down radix node when this sequence's prefix is later inserted (tail /
        finish commit). Overwriting a prior stash is correct — only the DEEPEST boundary matches the
        single node the eventual insert creates (align_down of the full committed length)."""
        cached_len = req.cached_len
        if cached_len == 0 or (cached_len % self.cache_manager.page_size) != 0:
            return
        slot = self._rec_slots.slot_for(req.uid)
        if slot is None:
            return
        self._pending_rec_snap[req.uid] = (cached_len, self._rec_cache.clone_slot(slot))

    def _maybe_capture_rec_state(self, req: Req, handle) -> None:
        """Attach a page-aligned recurrent-state checkpoint to the freshly-inserted radix node so a
        later request sharing this prefix restores it instead of re-prefilling. The node's boundary is
        handle.cached_len (insert_prefix already align_down's), and the losslessness precondition is
        that the attached state be exactly state@boundary. Two sources, in order:
          1. A stash from _stash_rec_state at that boundary (the common path — the forced page-aligned
             split leaves the sub-page tail unaligned, so the aligned state lives in the stash).
          2. Fallback: the req's own committed length is itself page-aligned and equals the node
             boundary, so the live slot holds state@boundary directly (e.g. a prompt whose length is a
             page multiple, no split needed)."""
        if self._rec_cache is None or handle is None:
            return
        boundary = handle.cached_len
        if boundary == 0:
            return
        stash = self._pending_rec_snap.get(req.uid)
        if stash is not None and stash[0] == boundary:
            self.cache_manager.attach_rec_state(handle, stash[1])
            self._pending_rec_snap.pop(req.uid, None)
            return
        # Fallback: no stash at this boundary, but the live slot IS at the boundary (aligned commit).
        if req.cached_len != boundary or (boundary % self.cache_manager.page_size) != 0:
            return
        slot = self._rec_slots.slot_for(req.uid)
        if slot is None:
            return
        self.cache_manager.attach_rec_state(handle, self._rec_cache.clone_slot(slot))

    # ---- SWA-radix window snapshot/restore (parallel to the recurrent path above) ------------------
    # Same three-stage lifecycle as recurrent radix, but the snapshot is the sliding-window ring's last
    # min(L,W) tokens (all sliding layers) instead of GDN/CCA state. Stored on the radix node's opaque
    # rec_state field and stashed via the shared _pending_rec_snap dict (a model is SWA xor recurrent).
    def _stash_swa_state(self, req: Req) -> None:
        """Checkpoint the sliding-window at a page-aligned chunk boundary (mirror _stash_rec_state).
        The ring holds the last W tokens at device_len==cached_len here, so clone(table_idx, cached_len)
        captures the window exactly. Attached to the align_down radix node at the tail/finish commit."""
        cached_len = req.cached_len
        if cached_len == 0 or (cached_len % self.cache_manager.page_size) != 0:
            return
        self._pending_rec_snap[req.uid] = (
            cached_len, self._swa_snap.clone(req.table_idx, cached_len)
        )

    def _maybe_capture_swa_state(self, req: Req, handle) -> None:
        """Attach a page-aligned window snapshot to the freshly-inserted radix node (mirror
        _maybe_capture_rec_state): a chunk stash at that boundary, else the live ring if the committed
        length IS the boundary (single-shot prefill — the common 'warm a prefix' path)."""
        if self._swa_snap is None or handle is None:
            return
        boundary = handle.cached_len
        if boundary == 0:
            return
        stash = self._pending_rec_snap.get(req.uid)
        if stash is not None and stash[0] == boundary:
            self.cache_manager.attach_rec_state(handle, stash[1])
            self._pending_rec_snap.pop(req.uid, None)
            return
        if req.cached_len != boundary or (boundary % self.cache_manager.page_size) != 0:
            return
        self.cache_manager.attach_rec_state(handle, self._swa_snap.clone(req.table_idx, boundary))

    def _restore_swa_states(self, batch: Batch) -> None:
        """CROSS-REQUEST reuse only: seed the reusing request's ring block from the page-aligned window
        snapshot attached to the matched radix node, BEFORE the forward. The sliding-layer extend
        (_gather_swa_windows) then reads the window straight from the ring — identical to a chunked
        continuation, whose window its own prior chunk already wrote. Only the initial prefix-hit pass
        restores (cached_len == matched boundary); a chunked continuation carries the same handle at a
        larger cached_len and is skipped (its ring already holds the window)."""
        for req in batch.reqs:
            handle = getattr(req, "cache_handle", None)
            snap = getattr(handle, "rec_state", None)
            if snap is None or req.cached_len == 0 or req.cached_len != handle.cached_len:
                continue
            self._swa_snap.restore_ring(req.table_idx, snap)
            logger.info_rank0(
                f"SWA-radix HIT: uid={req.uid} seeded ring window at cached_len={req.cached_len} "
                f"(sliding layers extend across the boundary; skips re-prefill of the shared prefix)"
            )

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        return self._finish_prepare(batch)

    def _finish_prepare(self, batch: Batch, skip_alloc: bool = False) -> ForwardInput:
        # The padding decision (batch.padded_reqs) is set by the caller — pad_batch for the DP/normal
        # path, or the EP lockstep mixin (which forces a common-bs / dummy-prefill padding so every
        # replica issues identical-shape MoE collectives). Everything below is padding-agnostic.
        # skip_alloc=True (EP idle-replica dummy prefill ONLY): the batch's lone dummy_req already
        # points its page_table at the reserved null page, so allocating a real KV page for it would
        # leak one every dummy step -> CacheManager integrity-check crash. Skip the allocation and let
        # the dummy write into the reserved page.
        if not skip_alloc:
            self.cache_manager.allocate_paged(batch.reqs)
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        self.engine.attn_backend.prepare_metadata(batch)
        # GDN-hybrid: allocate/reuse a recurrent-state slot per sequence and build the
        # per-batch GDN metadata (cu_seqlens / state_indices / has_initial_state). Inert for
        # dense models (gdn_slots is None). The GDN layers read batch.gdn_metadata in 3d.
        if self.gdn_slots is not None:
            from minisgl.gdn.metadata import build_gdn_metadata

            state_indices = self.gdn_slots.state_indices(batch)
            batch.gdn_metadata = build_gdn_metadata(batch, state_indices, self.device)
        # CCA-hybrid (Zaya): allocate/reuse a conv-state slot per sequence and build the per-batch
        # CCA metadata (query_start_loc / state_indices / has_initial_state). Inert for non-CCA
        # models (cca_slots is None). The CCA layers read batch.cca_metadata in the model forward.
        if self.cca_slots is not None:
            from minisgl.cca.metadata import build_cca_metadata

            cca_state_indices = self.cca_slots.state_indices(batch)
            batch.cca_metadata = build_cca_metadata(batch, cca_state_indices, self.device)
        # Recurrent-radix RESTORE: for any prefill req that hit a page-aligned recurrent-state snapshot
        # (cache_handle.rec_state set, cached_len>0), install that snapshot into its freshly-allocated
        # slot BEFORE the forward. state_indices() above zero-inited the new slot; we overwrite it with
        # the cached prefix state, and has_initial_state (cached_len>0) makes the GDN/CCA kernels
        # continue from it — byte-identical to prefilling the shared prefix from zero (recurrent kernel).
        if self._rec_cache is not None and batch.is_prefill and not batch.spec_verify:
            self._restore_rec_states(batch)
        # SWA-radix RESTORE: for any prefill req that hit a page-aligned window snapshot, seed its ring
        # block AND build batch.attn_metadata.swa_prefix so the sliding-layer extend attends across the
        # window boundary (rdna4.py::_swa_prefill_extend). Runs AFTER prepare_metadata (which built the
        # RDNA4Metadata); swa_prefix is a post-hoc field.
        elif self._swa_radix and batch.is_prefill and not batch.spec_verify:
            self._restore_swa_states(batch)
        # CAM editable-memory (Option B): compute each memory request's tap bank ONCE, at its prefill
        # (mem_bank starts None; product-key read is variable-shape so it must NOT run per decode step or
        # inside a graph — read here, reuse across decode). Inert when CAM is not built.
        if self.engine.cam is not None:
            self._prepare_cam(batch)
        sample_args = self.engine.sampler.prepare(batch)
        # Structured output: attach the per-row grammar bitmask (None unless a constrained req is in
        # the batch). The sampler masks disallowed tokens before argmax/sampling. Built over
        # padded_reqs so the row order matches logits[:batch.size]. No-op for a plain serve.
        sample_args.grammar_bitmask = self._build_grammar_bitmask(batch)
        # Reasoning gate: suppress EOS for rows still inside <think> so a thinking model can't stop
        # mid-reasoning and return a blank answer (bounded by the budget backstop, which forces </think>).
        sample_args.eos_suppress = self._build_eos_suppress(batch)
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _build_eos_suppress(self, batch: Batch) -> torch.Tensor | None:
        """Bool [batch.size] marking rows whose req is still inside its reasoning span (gate armed, not
        yet over budget). Their EOS logits are masked to -inf so the model can't end the turn before it
        closes </think> — the budget backstop then force-emits </think> and clears the gate. None when no
        row is gated (the common case) so the sampler skips the mask entirely."""
        if not self._grammar_think_gate_enabled or not self._grammar_think_gate:
            return None
        gate = self._grammar_think_gate
        flags = [
            (getattr(r, "uid", None) in gate) and not self._think_gate_over_budget(r.uid)
            for r in batch.reqs
        ]
        if not any(flags):
            return None
        return torch.tensor(flags, dtype=torch.bool, device=self.device)

    def _gate_mask_spec_logits(
        self, reqs: List[Req], staged_drafts: List[List[int]], logits: torch.Tensor
    ) -> bool:
        """Reasoning gate on the SPEC verify path — the analogue of _build_eos_suppress/_build_grammar_bitmask
        for the propose→verify step (which has its own accept and never touches those). For each req still
        inside its <think> span, mask its per-position verify logits IN PLACE so the accepted chain can't
        emit EOS mid-reasoning (under budget) or is forced to </think> (over budget). The accept below then
        naturally avoids EOS; the gate is COUNTED/OPENED from the committed (rank0-authoritative) tokens in
        pass 2, so all TP ranks advance it identically. Returns True if any req was gated. No-op otherwise."""
        if not self._grammar_think_gate_enabled or not self._grammar_think_gate:
            return False
        eos_ids = self.engine.sampler.eos_token_ids
        any_gated = False
        offset = 0
        for req, sd in zip(reqs, staged_drafts):
            q_len = len(sd) + 1
            gate_tid = self._grammar_think_gate.get(getattr(req, "uid", None))
            if gate_tid is not None:
                any_gated = True
                block = logits[offset:offset + q_len]
                if self._think_gate_over_budget(req.uid):
                    block.fill_(float("-inf"))       # force </think> (drafts won't match → num_accepted=0)
                    block[:, gate_tid] = 0.0
                elif eos_ids is not None:
                    block[:, eos_ids] = float("-inf")  # reasoning phase: forbid EOS until </think>/budget
            offset += q_len
        return any_gated

    def _resolve_think_close_id(self, delim: str) -> int | None:
        """Token id of the reasoning-close delimiter (e.g. "</think>"), cached per string. These tags
        are registered as single special/added tokens on every target family (qwen3/deepseek/glm), so
        `convert_tokens_to_ids` returns the id directly; fall back to encoding and take the sole (or
        trailing) id. Returns None if it can't be resolved, in which case the gate is not armed and the
        grammar applies from token 0 (safe, no worse than before)."""
        if delim in self._think_close_ids:
            return self._think_close_ids[delim]
        tid: int | None = None
        tok = self.tokenizer
        unk = getattr(tok, "unk_token_id", None)
        try:
            cand = tok.convert_tokens_to_ids(delim)
        except Exception:
            cand = None
        if isinstance(cand, int) and cand >= 0 and cand != unk:
            tid = cand
        else:
            try:
                ids = tok.encode(delim, add_special_tokens=False)
            except Exception:
                ids = []
            if ids:
                # Single-token close tag (the real case). A multi-token tag gates on its LAST id — a
                # rare, tolerable approximation; all mainstream reasoning tags are single tokens.
                tid = int(ids[-1])
        self._think_close_ids[delim] = tid
        return tid

    def _maybe_arm_think_gate(self, req: Req) -> None:
        """Arm the reasoning (β) budget gate for ANY thinking request that declared a think-close
        delimiter. Two callers: the grammar-matcher path (gate the schema until </think>) AND the
        plain-decode path (bound RSA/plain rollouts so reasoning is force-closed after β tokens and the
        model always produces an answer — the paper's bounded-workspace). Idempotent: no-op if already
        armed or if this uid's gate already opened (so we don't re-arm after reasoning ended)."""
        if not self._grammar_think_gate_enabled:
            return
        uid = req.uid
        if uid in self._grammar_think_gate or uid in self._think_gate_done:
            return
        delim = getattr(req.sampling_params, "think_close_delim", None)
        if not delim:
            return
        tid = self._resolve_think_close_id(delim)
        if tid is None:
            return
        self._grammar_think_gate[uid] = tid
        # Arm the reasoning budget: per-request override (>0) else the env default. Count starts at 0
        # and is incremented per reasoning token committed on both the plain-decode and spec paths.
        self._grammar_think_count[uid] = 0
        per_req = getattr(req.sampling_params, "think_budget", None)
        budget = per_req if (isinstance(per_req, int) and per_req > 0) else self._grammar_think_budget_default
        # Reserve room for the ANSWER: if the reasoning budget is >= this request's whole completion
        # budget, force-closing </think> would land exactly at max_tokens and truncate before any
        # answer. Cap the budget to ~3/4 of max_tokens so the answer always has space.
        mt = int(getattr(req.sampling_params, "max_tokens", 0) or 0)
        if mt > 0 and budget >= mt:
            budget = max(1, (mt * 3) // 4)
        self._grammar_think_budget[uid] = budget

    def _clear_think_gate(self, uid: int) -> None:
        """Drop all reasoning-gate state for ``uid`` (gate open / req finished). Idempotent. Marks the
        uid done so the plain-path arm helper won't re-arm it after its reasoning phase ended."""
        self._grammar_think_gate.pop(uid, None)
        self._grammar_think_count.pop(uid, None)
        self._grammar_think_budget.pop(uid, None)
        self._think_gate_done.add(uid)

    def _think_gate_over_budget(self, uid: int) -> bool:
        """True once ``uid`` has emitted >= its budget reasoning tokens — the backstop must now FORCE
        the think-close token so the gate opens and the schema engages."""
        budget = self._grammar_think_budget.get(uid)
        if budget is None:
            return False
        return self._grammar_think_count.get(uid, 0) >= budget

    def _mask_only_token(self, bitmask: torch.Tensor, row: int, tid: int) -> None:
        """Rewrite ``bitmask[row]`` to allow ONLY token ``tid`` (bit==1), matching the packing that
        ``apply_token_bitmask`` unpacks (bit i of word w is token w*32+i). Used to force-emit the
        think-close token at the reasoning budget."""
        bitmask[row].zero_()
        word, bit = divmod(int(tid), 32)
        val = 1 << bit
        if val >= 0x80000000:  # bit 31 -> wrap to signed int32
            val -= 0x100000000
        bitmask[row, word] = val

    def _build_grammar_bitmask(self, batch: Batch) -> torch.Tensor | None:
        """Packed xgrammar token bitmask [batch.size, ceil(vocab/32)] on device, or None if no req in
        the batch is constrained. Constrained rows carry the matcher's currently-allowed token set;
        every other row (unconstrained reqs + CUDA-graph dummy padding) is left all-ones (no mask).
        Matchers are created lazily here so a constrained req's FIRST sample (the prefill bonus token)
        is already masked."""
        reqs = batch.padded_reqs
        if not any(r.sampling_params.is_constrained for r in reqs):
            return None
        if self._grammar_backend is None:
            try:
                from minisgl.engine.grammar import GrammarBackend

                self._grammar_backend = GrammarBackend(self.tokenizer, self.engine.sampler.vocab_size)
            except Exception as e:  # noqa: BLE001
                # A missing/broken grammar backend (e.g. xgrammar not installed) must NOT crash the
                # serve. Latch a False sentinel (so we don't retry every batch) and fall back to
                # UNCONSTRAINED decoding for structured requests, with one clear warning.
                self._grammar_backend = False
                logger.warning_rank0(
                    "structured output requested but the grammar backend is unavailable (%r); serving "
                    "these requests UNCONSTRAINED. Install xgrammar in the serve image to enable it.", e
                )
        if self._grammar_backend is False:
            return None  # unconstrained fallback; every row stays all-ones (no mask)
        backend = self._grammar_backend
        bitmask = backend.allocate_bitmask(len(reqs))
        bitmask.fill_(-1)  # all-ones default => unconstrained / dummy rows allow every token
        for i, r in enumerate(reqs):
            if not r.sampling_params.is_constrained:
                continue
            m = self._grammar_matchers.get(r.uid)
            if m is None:
                m = self._grammar_matchers[r.uid] = backend.make_matcher(r.sampling_params.grammar)
                self._maybe_arm_think_gate(r)  # gate the schema until </think> if thinking is active
            # Reasoning gate: while still inside <think>…</think>, leave this row all-ones (free
            # reasoning) and do NOT advance the matcher — the schema starts fresh on the answer.
            # BACKSTOP: once the reasoning budget is spent, force the think-close token by masking this
            # row to allow ONLY that id. The sampler emits it, _process_last_data sees next_token==gate
            # and opens the gate, and the next step is schema-constrained.
            if r.uid in self._grammar_think_gate:
                if self._think_gate_over_budget(r.uid):
                    self._mask_only_token(bitmask, i, self._grammar_think_gate[r.uid])
                continue
            if not m.is_terminated():
                m.fill_next_token_bitmask(bitmask, i)
            # a terminated matcher leaves its row all-ones: nothing left to emit but EOS, allow it.
        return bitmask.to(self.device)

    def _verify_greedy_constrained(
        self, matcher, draft: List[int], logits_block: torch.Tensor, ignore_eos: bool, uid: int
    ) -> AcceptResult:
        """Grammar-constrained speculative acceptance for ONE request (structured output + spec decode).

        The proposer drafts UNCONSTRAINED; the grammar is enforced here, at the verify argmax. Walk the
        K+1 verify positions: at each, mask the target logits with the matcher's currently-allowed set
        (its state after the committed prefix), take the masked argmax as the grammar-valid target token
        ``t_i``, and accept ``draft[i]`` iff it equals ``t_i``. The committed run is exactly the masked
        target argmaxes (== what plain constrained decode would emit -> LOSSLESS), so a draft that
        violates the grammar simply mismatches ``t_i`` and is rejected. The matcher is advanced by every
        committed token (never by EOS, which isn't a grammar token); the walk stops at the bonus /
        first mismatch / EOS, so it never over-advances past what the caller keeps. ``logits_block`` is
        the req's [K+1, vocab] verify logits on CPU.

        Reasoning gate: while ``uid`` is still inside its `<think>…</think>` span, take the UNCONSTRAINED
        argmax (no mask, matcher not advanced) so reasoning is free and spec-decode keeps helping. The
        gate opens mid-walk the moment `</think>` is the committed token; the matcher — untouched during
        reasoning — then enforces the schema from the very next position."""
        from minisgl.engine.grammar import apply_token_bitmask

        backend = self._grammar_backend
        emitted: List[int] = []
        n_acc = 0
        K = len(draft)
        for i in range(K + 1):
            gate = self._grammar_think_gate.get(uid)
            if gate is not None:
                # BACKSTOP: reasoning budget spent -> FORCE the think-close token here (overriding the
                # unconstrained argmax) and open the gate. It's the bonus of this spec step; the next
                # step verifies schema-constrained. (The draft is unconstrained, so it won't match the
                # forced close -> the walk ends, which is exactly what we want.)
                if self._think_gate_over_budget(uid):
                    emitted.append(gate)
                    self._clear_think_gate(uid)
                    break
                # Reasoning phase: unconstrained greedy verify; matcher stays at its initial state.
                t_i = int(logits_block[i].argmax().item())
                emitted.append(t_i)
                if (not ignore_eos) and t_i in self.eos_token_ids:
                    break
                if t_i == gate:
                    self._clear_think_gate(uid)  # open gate: NEXT position is schema-constrained
                else:
                    self._grammar_think_count[uid] = self._grammar_think_count.get(uid, 0) + 1
                if i < K and draft[i] == t_i:
                    n_acc += 1
                    continue
                break
            bitmask = backend.allocate_bitmask(1)
            bitmask.fill_(-1)
            terminated = matcher.is_terminated()
            if not terminated:
                matcher.fill_next_token_bitmask(bitmask, 0)
            masked = apply_token_bitmask(logits_block[i : i + 1].float(), bitmask)
            t_i = int(masked[0].argmax().item())
            emitted.append(t_i)
            is_eos = (not ignore_eos) and t_i in self.eos_token_ids
            if not is_eos and not terminated:
                matcher.accept_token(t_i)
            if is_eos:
                break  # EOS ends the sequence; not fed to the matcher
            if i < K and draft[i] == t_i:
                n_acc += 1
                continue
            break  # bonus token (t_i != draft[i], or i == K)
        return AcceptResult(emitted=emitted, num_accepted=n_acc)

    def _verify_sampled_constrained(
        self, matcher, draft: List[int], logits_block: torch.Tensor, sp, gen, uid: int
    ) -> AcceptResult:
        """Sampled (rejection-sampling) analogue of _verify_greedy_constrained (structured output +
        SAMPLED spec). Identical think-gate / matcher-advance / EOS scaffolding; per position the target
        dist p is built from the GRAMMAR-MASKED logits with the req's temp/top_k/top_p, and the draft is
        accepted with prob min(1, p(draft)/q), q=onehot(draft) — a grammar-violating draft has masked
        p=0 so it is ALWAYS rejected, exactly as the greedy masked-argmax rejects it. On reject/bonus the
        token is sampled from the (masked) p; the matcher is advanced by every committed token. Output is
        distributed as plain constrained sampled decode (distributionally lossless). ``logits_block``
        [K+1, V] on device."""
        from minisgl.engine.grammar import apply_token_bitmask

        backend = self._grammar_backend
        emitted: List[int] = []
        n_acc = 0
        K = len(draft)

        def _reject_sample(p: torch.Tensor, di):
            # accept draft di w.p. p[di] (q=onehot); else residual = renorm(relu(p - onehot(di))).
            # di is None at the bonus position -> straight sample from p. Returns (token, accepted).
            if di is not None:
                u = float(torch.rand(1, generator=gen, device=p.device).item())
                if u < float(p[di]):
                    return di, True
                resid = p.clone()
                resid[di] = 0.0
                s = resid.sum()
                dist = resid / s if float(s) > 0.0 else p
                return int(torch.multinomial(dist, 1, generator=gen).item()), False
            return int(torch.multinomial(p, 1, generator=gen).item()), False

        for i in range(K + 1):
            di = int(draft[i]) if i < K else None
            gate = self._grammar_think_gate.get(uid)
            if gate is not None:
                if self._think_gate_over_budget(uid):
                    emitted.append(gate)
                    self._clear_think_gate(uid)
                    break
                # Reasoning phase: UNCONSTRAINED sampled rejection; matcher stays at its initial state.
                p = probs_from_logits(
                    logits_block[i : i + 1], sp.temperature, sp.top_k, sp.top_p
                )[0]
                t_i, acc = _reject_sample(p, di)
                emitted.append(t_i)
                if (not sp.ignore_eos) and t_i in self.eos_token_ids:
                    break
                if t_i == gate:
                    self._clear_think_gate(uid)  # open gate: NEXT position is schema-constrained
                else:
                    self._grammar_think_count[uid] = self._grammar_think_count.get(uid, 0) + 1
                if i < K and acc:
                    n_acc += 1
                    continue
                break
            bitmask = backend.allocate_bitmask(1)
            bitmask.fill_(-1)
            terminated = matcher.is_terminated()
            if not terminated:
                matcher.fill_next_token_bitmask(bitmask, 0)
            masked = apply_token_bitmask(logits_block[i : i + 1].float(), bitmask)  # disallowed -> -inf
            p = probs_from_logits(masked, sp.temperature, sp.top_k, sp.top_p)[0]  # grammar-masked dist
            t_i, acc = _reject_sample(p, di)
            emitted.append(t_i)
            is_eos = (not sp.ignore_eos) and t_i in self.eos_token_ids
            if not is_eos and not terminated:
                matcher.accept_token(t_i)
            if is_eos:
                break
            if i < K and acc:
                n_acc += 1
                continue
            break  # bonus token (rejected draft, or i == K)
        return AcceptResult(emitted=emitted, num_accepted=n_acc)

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        return self._prepare_batch(batch) if batch else None

    def _prepare_cam(self, batch: Batch) -> None:
        """Read each memory request's tap bank from the standing store at prefill (once per req).

        Subject rides on sampling_params.mem_subject; it is space-prefixed to match how the store was
        trained (memory-organ `_sp_tokens`). `_mem_seed` is the store's preferred first token (for the
        seed-once policy in `_stage_cam`). Requests without a subject are left untouched (tap no-op)."""
        cam = self.engine.cam
        for req in batch.reqs:
            if not hasattr(req, "_mem_placed"):   # ChunkedReq / non-Req rows carry no memory state
                continue
            sp = getattr(req, "sampling_params", None)
            # #100 CONTROL op (facts/forget/stats): compute the result from engine.cam and FORCE-EMIT it
            # (tokenised) as the reply text + EOS — the data-returning ops ride the generate path too, no new
            # message type. Handled before the subject guard (facts/stats carry no subject).
            ns = getattr(sp, "mem_namespace", None)          # #6: per-tenant/session store scope
            mem_op = getattr(sp, "mem_op", None)
            if mem_op and getattr(req, "_mem_deliver", None) is None:
                if mem_op == "retrieve":         # TRANSPARENT read: match stored subjects that appear in
                    result = self._cam_retrieve(cam, req.input_ids, ns)   # the prompt -> facts for auto-RAG
                else:
                    result = self._cam_ctrl_result(cam, mem_op, getattr(sp, "mem_subject", None), ns)
                toks = list(self.tokenizer(result, add_special_tokens=False).input_ids)
                _eos = next(iter(self.eos_token_ids), None)
                if _eos is not None:
                    toks = toks + [_eos]                             # terminate after the result string
                req._mem_deliver, req._mem_deliver_pos = toks, 0
                continue
            subj = getattr(sp, "mem_subject", None)
            if not subj or req.mem_bank is not None or getattr(req, "_mem_deliver", None) is not None:
                continue
            subj_ids = list(self.tokenizer(" " + _canon_subject(subj), add_special_tokens=False).input_ids)
            # #100 REMEMBER (multi-process write): the store lives in THIS scheduler process, so a write
            # must ride the request — mem_remember=object_token_ids writes subject->object into engine.cam.
            # Write-only: no forced tokens (the frontend sends max_tokens=1; the 1-token generation is a stub).
            mem_remember = getattr(req.sampling_params, "mem_remember", None)
            if mem_remember:
                # WRITE GATING: explicit ingest (mem_write_mode="force") always writes; ambient auto-write
                # ("auto"/None) is refused when the namespace is frozen or (no-clobber) the subject is already
                # curated — so conversational chatter can't overwrite a deliberately-ingested store.
                mode = getattr(sp, "mem_write_mode", None) or "auto"
                if cam.write_allowed(subj_ids, source=mode, ns=ns):
                    cam._write(subj_ids, list(mem_remember), ns=ns)   # records the fact in this namespace
                req._mem_deliver, req._mem_deliver_pos = [], 0    # mark processed; deliver nothing
                continue
            # #100 POINTER delivery (multi-process): force the EXACT object token sequence retrieved from
            # the cosine-NN subject index, then release to base continuation — memory supplies the
            # unknowable object tokens, the served base finishes the sentence. `_process_last_data` overrides
            # the sampled token with the forced object token for the first len(obj) steps. Falls back to the
            # residual tap (mem_bank) only when the subject addresses no stored object.
            _deliver = getattr(cam, "deliver_object_ids", None)
            obj = _deliver(subj_ids, ns) if _deliver is not None else []
            if obj:
                req._mem_deliver, req._mem_deliver_pos = list(obj), 0
                req.mem_bank = None                          # pointer forces exact tokens; no tap needed
            elif not getattr(cam, "pointer_only", False):
                # Residual-tap fallback (tap+router path only). Skipped in pointer-only mode: cam.read
                # runs the checkpoint-dimensioned adapter, which mismatches a different served base.
                bank, conf = cam.read(subj_ids, ns=ns)
                req.mem_bank, req.mem_conf = bank, conf
                req._mem_seed = int(cam.seed_token(bank, conf)) if bank is not None else None
        # TP>1: make the forced-token deliveries rank0-AUTHORITATIVE before the forward, so a per-rank
        # store drift can't desync the batch (see _bcast_cam_deliver_tp).
        if self._tp_size > 1:
            self._bcast_cam_deliver_tp(batch)
        autosave = getattr(cam, "autosave", None)   # #7 debounced persistence after any writes this batch
        if autosave is not None:
            autosave()

    def _bcast_cam_deliver_tp(self, batch: Batch) -> None:
        """TP>1 lockstep for CAM: make each req's forced-token delivery rank0-AUTHORITATIVE.

        Pointer delivery + control-op results are computed per-rank from engine.cam. The stores are
        built + written symmetrically, so they are USUALLY identical — but any drift (an FP edge at the
        deliver_tau gate, or a write applied asymmetrically under concurrency) makes the two ranks force
        a DIFFERENT number of tokens. Different generation lengths -> the ranks' decode batches diverge
        in size -> the forward's TP all_reduce/all_gather shapes mismatch and gloo-deadlock to the 1800s
        timeout (the tp2-cam-op-sched-divergence outage, 2026-07-17). Broadcasting rank0's `_mem_deliver`
        for every req makes all ranks force the IDENTICAL sequence regardless of store state — the same
        rank0-authoritative pattern the spec path already uses for drafts (_bcast_drafts_tp), and it
        removes the fragile cross-rank-determinism assumption noted at engine.py `_cam_gather_full_vocab`.
        Overrides only the VALUE, never `_mem_deliver_pos` (each rank then advances it identically in
        _process_last_data since the forced tokens now match). Scope: the pointer/control `_mem_deliver`
        path (production pointer_only mode); the tap `mem_bank` path is not broadcast (inactive there)."""
        if self._tp_size <= 1:
            return
        reqs = [r for r in batch.reqs if hasattr(r, "_mem_placed")]
        # Symmetric gate: sampling_params are broadcast with the batch, so both ranks agree whether this
        # batch carries any CAM request and thus whether to run the (collective) broadcast at all — the
        # decision itself can't drift. Skips the per-step cost on the common no-CAM batch.
        def _is_cam(sp) -> bool:
            return bool(sp is not None and (getattr(sp, "mem_op", None) or getattr(sp, "mem_subject", None)
                                            or getattr(sp, "mem_remember", None)))
        if not any(_is_cam(getattr(r, "sampling_params", None)) for r in reqs):
            return
        g = self.tp_cpu_group
        prim = self._tp_is_primary
        # per-req length: -1 = no _mem_deliver (non-CAM / unset), >=0 = forced-token count (0 = write stub).
        if prim:
            lens = torch.tensor(
                [(len(r._mem_deliver) if getattr(r, "_mem_deliver", None) is not None else -1)
                 for r in reqs], dtype=torch.int64)
        else:
            lens = torch.zeros(len(reqs), dtype=torch.int64)
        g.broadcast(lens, root=0).wait()
        lens_l = [int(x) for x in lens.tolist()]
        total = sum(L for L in lens_l if L > 0)
        if prim:
            flat = torch.tensor([t for r in reqs if getattr(r, "_mem_deliver", None)
                                 for t in r._mem_deliver], dtype=torch.int64)
        else:
            flat = torch.zeros(total, dtype=torch.int64)
        if total > 0:
            g.broadcast(flat, root=0).wait()
        if prim:
            return
        off = 0
        for r, L in zip(reqs, lens_l):
            if L < 0:
                continue                              # rank0 had no delivery here -> keep local state
            if L == 0:
                r._mem_deliver = []                   # write stub: deliver nothing (max_tokens=1 finishes it)
            else:
                r._mem_deliver = [int(x) for x in flat[off:off + L]]
                off += L
                r.mem_bank = None                     # pointer forces exact tokens; no tap (matches rank0)
            if getattr(r, "_mem_deliver_pos", None) is None:
                r._mem_deliver_pos = 0                # first sync starts at 0; hereafter each rank advances it

    def _cam_ctrl_result(self, cam, op: str, subj: str | None, ns: str | None = None) -> str:
        """#100 control op -> JSON string (force-emitted as the reply), scoped to namespace `ns` (#6).
        facts: [{subject,object}]; forget: bool; stats: dict; freeze/unfreeze: {frozen}."""
        import json
        if op == "forget":
            sids = list(self.tokenizer(" " + _canon_subject(subj), add_special_tokens=False).input_ids) if subj else []
            return json.dumps(bool(cam.forget(sids, ns=ns)) if sids else False)
        if op == "facts":
            out = []
            for f in (cam.list_facts(ns) or []):
                sids, oids = f.get("subject_ids"), f.get("object_ids")
                out.append({"subject": self.tokenizer.decode(list(sids)).strip() if sids else "",
                            "object": self.tokenizer.decode(list(oids)).strip() if oids else ""})
            return json.dumps(out)
        if op == "stats":
            # Control-op results are force-emitted ONE TOKEN PER DECODE STEP (see _process_last_data),
            # so a verbose result costs one scheduler step per token. stats()'s per-bank "banks" array
            # grows with the store (one entry per non-empty bank × every namespace) and dominated the
            # token count — /cam/stats crept to ~7-9s on a grown store while the summary is what callers
            # read. Drop the bulky per-bank enumeration from the emitted result (crowded_banks + the
            # summary counts stay); the full per-bank breakdown remains available via cam.stats() itself.
            s = cam.stats(ns)
            s.pop("banks", None)
            return json.dumps(s)
        if op in ("freeze", "unfreeze"):        # read-only toggle: protect a curated namespace from auto-write
            frozen = cam.freeze(ns) if op == "freeze" else cam.unfreeze(ns)
            return json.dumps({"frozen": bool(frozen)})
        if op == "save":                        # #7 explicit persistence flush
            return json.dumps({"saved": cam.save() if hasattr(cam, "save") else -1})
        if op == "reload":                      # #11 pull shared-store writes from another replica
            return json.dumps({"edits": cam.reload() if hasattr(cam, "reload") else -1})
        if op == "undo":                        # #12 undo the last write in this namespace
            u = cam.undo(ns) if hasattr(cam, "undo") else {}
            return json.dumps({"subject": self.tokenizer.decode(list(u["subject_ids"])).strip(),
                               "object": self.tokenizer.decode(list(u["object_ids"])).strip()} if u else {})
        if op == "rebuild":                     # #12 true-erase / compact this namespace's banks
            return json.dumps({"rebuilt": cam.rebuild(ns) if hasattr(cam, "rebuild") else 0})
        if op == "audit":                       # #12 recent write/forget/evict events for this namespace
            out = []
            for r in (cam.audit_log(ns) if hasattr(cam, "audit_log") else []):
                out.append({"op": r["op"], "ts": r["ts"],
                            "subject": self.tokenizer.decode(list(r["subject_ids"])).strip(),
                            "object": self.tokenizer.decode(list(r["object_ids"])).strip() if r["object_ids"] else ""})
            return json.dumps(out)
        if op == "lookup":                      # spine #1/#5: subject-direct dry-run — does this subject deliver?
            sids = list(self.tokenizer(" " + _canon_subject(subj),
                                       add_special_tokens=False).input_ids) if subj else []
            oids = cam.deliver_object_ids(sids, ns) if sids else []
            return json.dumps({"delivered": bool(oids), "subject": subj or "",
                               "object": self.tokenizer.decode(oids).strip() if oids else ""})
        if op == "namespaces":                  # spine #4: enumerate namespaces + fact counts
            return json.dumps(cam.list_namespaces() if hasattr(cam, "list_namespaces") else [])
        if op == "drop_ns":                     # spine #4: delete a namespace's store
            return json.dumps({"dropped": bool(cam.drop_namespace(ns)) if hasattr(cam, "drop_namespace") else False})
        return json.dumps(None)

    def _cam_retrieve(self, cam, prompt_ids, ns: str | None = None) -> str:
        """TRANSPARENT read: which stored subjects does this prompt mention? Decode the prompt, pull
        candidate proper-noun spans (capitalised word runs — the common subject shape), and query each
        against the store's cosine-NN subject index (deliver_object_ids, tau-gated). Returns the matched
        facts as JSON [{subject,object}] for the frontend to fold into context (auto-RAG). Empty when
        nothing confidently matches — the tau threshold keeps it quiet on unrelated prompts."""
        import json
        import re
        import time
        _prof = os.environ.get("MINISGL_CAM_RETRIEVE_PROF") == "1"
        _t0 = time.perf_counter()
        deliver = getattr(cam, "deliver_object_ids", None)
        if deliver is None:
            return json.dumps([])
        text = self.tokenizer.decode(list(prompt_ids))
        cands = set()
        # (a) capitalised proper-noun spans — high precision (names / places).
        for m in re.finditer(r"[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4}", text):
            cands.add(m.group(0))
        # (b) content-word n-gram WINDOWS — recall for PHRASE subjects like "capital of zorbia" that a
        # question ("What is the capital of Zorbia?") never surfaces as a proper-noun span (#10). The
        # cosine subject key is case/order-robust and tau-gated, so windows that address nothing self-
        # reject (unknown subjects max-cos ~0.5 < deliver_tau 0.7). Bound the word budget so a huge agent
        # prompt can't explode the candidate count.
        # Interrogatives (what/where/when/...) carry the RELATION signal for multi-fact retrieval: "WHERE
        # was Mozart born" addresses the birthplace fact, "WHEN was Mozart born" the birth-year fact.
        # They stay in _STOP for the TRAILING trim and the all-stop check, but are allowed at the LEADING
        # boundary (_LEAD_STOP) so "Where was Mozart born" survives as a candidate while "Mozart born where"
        # still trims to "Mozart born". Without this the disambiguator is stripped and both collapse to the
        # ambiguous "Mozart born" (measured: fixes the one transparent multi-fact miss, 6/7 -> 7/7).
        _INTERROG = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}
        _STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was",
                 "were", "be", "does", "do", "did", "that", "this", "these", "those", "it", "its", "his",
                 "her", "their", "your", "my", "you", "he", "she", "they", "we", "as", "by", "with",
                 "from", "about", "tell", "me", "please", "can", "could", "would", "will", "s"} | _INTERROG
        _LEAD_STOP = _STOP - _INTERROG
        # Split on sentence/clause punctuation FIRST so a window can't cross a boundary and swallow the
        # next clause's words (ngram-window-slop: "What is the capital of Zorbia? Just the name." must not
        # yield "capital of Zorbia Just" — the stray word dilutes the cosine key). Windows are generated
        # WITHIN each clause; the 160-word budget is shared across clauses so a huge prompt still can't
        # explode the candidate count.
        budget = 160
        for clause in re.split(r"[.?!,;:\n]+", text):
            if budget <= 0:
                break
            words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.\-]*", clause)[:budget]
            budget -= len(words)
            n = len(words)
            for i in range(n):
                for L in range(2, 7):                         # 2..6-word windows (1-word covered by (a))
                    if i + L > n:
                        break
                    span = words[i:i + L]
                    if span[0].lower() in _LEAD_STOP or span[-1].lower() in _STOP:
                        continue                              # trim stopword boundaries (leading interrogatives kept)
                    if all(w.lower() in _STOP for w in span):
                        continue
                    cands.add(" ".join(span))
        ordered = sorted(cands, key=len, reverse=True)        # prefer longer (fuller) spans first
        # Safety cap on candidate count for pathological (huge agent) prompts. Loose by default — the
        # per-candidate cost is now O(1)-batched (one tokenize call + one key build + one matmul), so this
        # only bounds the truly-degenerate case; normal prompts fall well under it (no recall change).
        _cap = int(os.environ.get("MINISGL_CAM_MAX_SPANS", "256") or 0)
        if _cap and len(ordered) > _cap:
            ordered = ordered[:_cap]
        _t_cand = time.perf_counter()
        # ONE batched tokenization over ALL candidates (was a Python loop of C tokenizer invocations).
        canon = [" " + _canon_subject(c) for c in ordered]
        cids_list = ([list(x) for x in self.tokenizer(canon, add_special_tokens=False).input_ids]
                     if canon else [])
        _t_tok = time.perf_counter()
        # Batched cosine: one [C,d]@[d,M] over all candidate keys instead of a per-candidate stack+matmul
        # (deliver_object_ids in a Python loop) — same per-row argmax/tau/LRU + same order, so byte-identical
        # results, but O(candidates×facts) collapses to one matmul (fixes the big-prompt retrieve wedge).
        # Falls back to the singular deliver if a batched method isn't present (older store build).
        batch = getattr(cam, "deliver_object_ids_batch", None)
        oids_list = (batch(cids_list, ns, texts=ordered) if batch is not None
                     else [deliver(cids, ns) for cids in cids_list])
        _t_match = time.perf_counter()
        seen_obj, out = set(), []
        for c, oids in zip(ordered, oids_list):
            if oids:
                obj = self.tokenizer.decode(oids).strip()
                if obj and obj not in seen_obj:               # dedupe by delivered fact (longest span wins)
                    seen_obj.add(obj); out.append({"subject": c, "object": obj})
        if _prof:
            line = ("[cam-prof] retrieve cands=%d cand_gen=%.1fms tokenize=%.1fms match=%.1fms "
                    "total=%.1fms hits=%d" % (
                        len(ordered), (_t_cand - _t0) * 1e3, (_t_tok - _t_cand) * 1e3,
                        (_t_match - _t_tok) * 1e3, (time.perf_counter() - _t0) * 1e3, len(out)))
            print(line, flush=True)                    # plain stdout — captured by docker logs (boot lines prove it)
            try:  # file sink — robust to per-process docker-log capture
                with open(os.environ.get("MINISGL_CAM_RETRIEVE_PROF_FILE", "/cam_store/retrieve_prof.log"),
                          "a") as _f:
                    _f.write(line + "\n")
            except Exception:  # noqa: BLE001
                pass
        return json.dumps(out)

    def _cam_direct_control(self, msg: "UserMsg") -> None:
        """Answer a CAM control op (mem_op) DIRECTLY — no forward, no decode loop. Computes the JSON
        result from engine.cam and sends it as ONE finished reply (next_token + extra_tokens, the same
        multi-token message spec-decode uses), so a control op costs O(1) instead of one 35B forward per
        result token. Both TP ranks run this (side effects like forget/drop stay in sync); only rank0
        emits the reply to the frontend. The req never enters a batch, so there is no per-rank batch to
        diverge — inherently TP-safe."""
        cam = self.engine.cam
        sp = msg.sampling_params
        op = sp.mem_op
        ns = getattr(sp, "mem_namespace", None)
        if op == "retrieve":
            result = self._cam_retrieve(cam, msg.input_ids, ns)
        else:
            result = self._cam_ctrl_result(cam, op, getattr(sp, "mem_subject", None), ns)
        autosave = getattr(cam, "autosave", None)   # #7 persist mutating ops (forget/drop/undo/...)
        if autosave is not None:
            autosave()
        if not self._tp_is_primary:
            return                                  # only rank0 emits replies to the frontend
        toks = list(self.tokenizer(result, add_special_tokens=False).input_ids)
        eos = next(iter(self.eos_token_ids), None)
        if eos is not None:
            toks = toks + [eos]                     # terminate the reply after the result string
        if not toks:
            toks = [eos if eos is not None else 0]
        self.send_result([DetokenizeMsg(uid=msg.uid, next_token=int(toks[0]),
                                        extra_tokens=[int(t) for t in toks[1:]], finished=True)])

    def _stage_cam(self, batch: Batch) -> None:
        """Build PER-TOKEN tap banks for an EAGER forward and stage them, so concurrent memory +
        non-memory requests in one batch each get the right injection. Row t (flat, per-req contiguous
        over padded_reqs, req.extend_len tokens each — matches _make_positions) carries that token's
        owning request's bank, or ZERO for a non-memory / seed-once-placed / padding row (tap no-op).
        SEED-ONCE: once `_mem_seed` has landed (`_mem_placed`, set in _process_last_data) the req's rows
        go zero. Graph-decode replay ignores these Python tensors (it reads the captured static buffer);
        this path drives eager prefill + eager decode. (Overlap loop: the placed flag lags one step.)"""
        cam, inner = self.engine.cam, self.engine.model.model
        if getattr(cam, "pointer_only", False) or not hasattr(inner, "stage_cam_rows"):
            return                                   # no tap seam / pointer-only: bank staging is a no-op
        reqs = batch.padded_reqs if batch.padded_reqs is not None else batch.reqs
        active = [(getattr(r, "mem_bank", None) is not None and not getattr(r, "_mem_placed", False))
                  for r in reqs]
        if not any(active):
            inner.clear_cam()                       # no active memory row -> tap no-op for the whole batch
            return
        dev = cam.device
        K, mem = cam.k_slots, cam.mem_dim
        bank_chunks, conf_chunks = [], []
        for r, act in zip(reqs, active):
            nt = r.extend_len                        # prefill: extend_len tokens; decode: 1
            if act:
                bank_chunks.append(r.mem_bank[0].unsqueeze(0).expand(nt, K, mem))   # [nt,K,mem]
                cv = r.mem_conf.reshape(-1)[0] if r.mem_conf is not None \
                    else torch.zeros((), device=dev)
                conf_chunks.append(cv.expand(nt))
            else:
                bank_chunks.append(torch.zeros(nt, K, mem, device=dev))
                conf_chunks.append(torch.zeros(nt, device=dev))
        inner.stage_cam_rows(cam, torch.cat(bank_chunks, 0), torch.cat(conf_chunks, 0))

    def _forward(self, forward_input: ForwardInput, track_reqs: bool = True) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.engine.cam is not None:
            self._stage_cam(batch)
        # DFlash training-data capture: run the forward WITH aux (eager, no graph) whenever the batch has
        # any EXTEND (prefill/teacher-forcing) rows — NOT only pure-prefill batches, since under
        # concurrency prefills get batched with decodes (batch.is_prefill=False) and would otherwise be
        # skipped. _dump_capture picks out the extend_len>1 rows.
        if (self._capture_dir and not getattr(batch, "spec_verify", False)
                and any(r.extend_len > 1 for r in batch.reqs)):
            # Snapshot the extend spans BEFORE the forward: forward_batch calls complete_one() which sets
            # cached_len=device_len, collapsing extend_len to 1 — so the post-forward req can't tell us
            # the prefill length. (uid, row offset, ext, cached_len) in padded_reqs order.
            spans, off = [], 0
            for r in batch.padded_reqs:
                spans.append((getattr(r, "uid", -1), off, r.extend_len, r.cached_len))
                off += r.extend_len
            out, _lh, aux = self.engine.forward_batch(batch, sample_args, return_hidden=True)
            if aux is not None:
                self._dump_capture(batch, aux, spans)
            forward_output = out
        else:
            forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # track_reqs=False for an EP lockstep DUMMY batch (no real reqs): its dummy_req must NOT be
        # promoted into the decode running set (it would pollute every subsequent decode step).
        if track_reqs:
            self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output

    def _dump_capture(self, batch: Batch, aux_hidden: torch.Tensor, spans) -> None:
        """Buffer this batch's EXTEND (prefill/teacher-forced) per-position aux + token for DFlash
        re-distillation. aux_hidden [num_aux, T, hidden] over the batch rows (padded_reqs order).
        ``spans`` = [(uid, row_offset, ext, cached_len)] snapshot BEFORE the forward (complete_one
        collapses extend_len afterward). seed_in[j] = concat of the num_aux layer taps at position j (the
        SAME order set_capture_layers used → matches the drafter's fc); bonus[j] = the token AT position
        j (teacher-forced input_ids); pos[j] = absolute position; slot = uid (separates sequences)."""
        num_aux, _T, H = aux_hidden.shape
        ids = batch.input_ids
        positions = getattr(batch, "positions", None)
        seed_l, tok_l, pos_l, slot_l = [], [], [], []
        for uid, off, ext, c0 in spans:
            # Only extend rows (ext>1). Decode rows (ext==1, mixed-batch neighbours) and dummy/padding
            # rows (uid<0) are skipped — the teacher-forcing re-feed gives the dense per-position aux.
            if uid is None or uid < 0 or ext <= 1:
                continue
            sl = aux_hidden[:, off:off + ext]                              # [num_aux, ext, H]
            seed_l.append(sl.permute(1, 0, 2).reshape(ext, num_aux * H).half().cpu())
            tok_l.append(ids[off:off + ext].to(torch.long).cpu())
            if positions is not None:
                pos_l.append(positions[off:off + ext].to(torch.long).cpu())
            else:
                pos_l.append(torch.arange(c0, c0 + ext, dtype=torch.long))
            slot_l.append(torch.full((ext,), int(uid), dtype=torch.long))
        if seed_l:
            self._capture_buf.append({
                "seed_in": torch.cat(seed_l), "bonus": torch.cat(tok_l),
                "pos": torch.cat(pos_l), "slot": torch.cat(slot_l),
            })
            if sum(int(r["bonus"].numel()) for r in self._capture_buf) >= 2048:
                self._flush_capture()

    def _flush_capture(self) -> None:
        if not self._capture_buf:
            return
        path = os.path.join(self._capture_dir, f"seedbuf_{os.getpid()}_{self._capture_n:05d}.pt")
        npos = sum(int(r["bonus"].numel()) for r in self._capture_buf)
        torch.save(self._capture_buf, path)
        logger.info_rank0(f"[capture] flushed {npos} positions -> {path}")
        self._capture_n += 1
        self._capture_buf = []

    def _flush_opd(self) -> None:
        if not self._opd_buf:
            return
        path = os.path.join(self._opd_dir, f"opdbuf_{os.getpid()}_{self._opd_n:05d}.pt")
        torch.save(self._opd_buf, path)
        logger.info_rank0(f"[opd-capture] flushed {len(self._opd_buf)} steps -> {path}")
        self._opd_n += 1
        self._opd_buf = []

    # ===================================================================================
    # Speculative decoding (synchronous loop). See SPEC_DECODE.md for the full design.
    # ===================================================================================

    def _spec_loop(self) -> None:
        """One synchronous spec-decode iteration. Prefill still goes through the normal path;
        only an all-greedy decode batch is replaced by a propose→verify→accept step."""
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Prefill takes priority and uses the normal (non-spec) synchronous path — except when the
        # prompt-prefill draft-KV seed is enabled, where it runs a hidden-capturing prefill instead.
        batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        if batch is not None:
            # A constrained (structured-output) req must take the plain prefill path so its first token
            # (the prefill bonus) is grammar-masked — the hidden-capturing seed forward bypasses the
            # sampler's bitmask. It also isn't a spec req, so seeding its draft KV is pointless.
            constrained = any(r.sampling_params.is_constrained for r in batch.reqs)
            if self._spec_seed_enabled and not constrained:
                self._spec_prefill_seeded(batch)
            else:
                forward_input = self._prepare_batch(batch)
                self._process_last_data((forward_input, self._forward(forward_input)))
            return

        if not self.decode_manager.runnable:
            return

        reqs = self.decode_manager.ordered_reqs
        # Spec-decode runs for an all-greedy decode set (lossless accept is greedy-only) UNLESS sampled
        # spec is enabled (MINISGL_SPEC_SAMPLED), which lets non-greedy unconstrained reqs spec-decode
        # via rejection sampling. Constrained (structured-output) reqs spec-decode greedily too (grammar
        # enforced at the verify argmax); a constrained-AND-sampled req falls back to plain decode.
        spec_ok = all(self._req_spec_ok(req) for req in reqs)
        if spec_ok:
            if getattr(self, "_tidar_ddtree", False):
                self._spec_decode_step_ddtree(
                    reqs, self._proposer.block_size, self._proposer.mask_token_id)
            elif getattr(self, "_dflash_ddtree", False):
                self._spec_decode_step_dflash_ddtree(reqs)
            elif getattr(self, "_tidar_fused", False):
                self._spec_decode_step_tidar_fused(
                    reqs, self._proposer.block_size, self._proposer.mask_token_id)
            else:
                self._spec_decode_step(reqs)
        else:
            batch = self.decode_manager.schedule_next_batch()
            forward_input = self._prepare_batch(batch)
            self._process_last_data((forward_input, self._forward(forward_input)))

    def _spec_num_ep_forwards(self) -> int:
        """How many FULL-TARGET forwards a spec step issues that hit the EP-sharded MoE — i.e. how many
        sets of per-layer MoE collectives an idle EP replica must mirror. This is NOT the total forward
        count: a draft head/model with a DENSE MLP issues NO EP collectives in propose, so only its
        verify forward counts. DETERMINISTIC per config (same on every replica) EXCEPT MTP.

          * TiDAR: block_predict + verify, both run the FULL target = 2 (fused: 1).
          * EAGLE3 / DFlash / n-gram: verify only — the draft is a dense 1-layer model / a table
            lookup, no EP-MoE in propose = 1.
          * MTP: verify (1) PLUS a data-dependent number of MoE draft-HEAD steps (the head is a full
            MoE decoder layer, EP-sharded). A fixed count can't match it -> return -1 (the caller
            replicates the draft MoE so propose issues no collectives; see _spec_ep_loop / MoELayer).
        """
        spec = self.engine.spec_config
        assert spec is not None
        algo = spec.algorithm
        if algo == "tidar":
            return 1 if getattr(self, "_tidar_fused", False) else 2
        if algo == "mtp":
            return 1 if self._spec_draft_ep_replicated else -1
        return 1  # eagle3, dflash, ngram: dense/lookup draft -> verify only

    def _spec_ep_loop(self) -> None:
        """Spec-decode under expert parallelism. Runs when BOTH spec_config and enable_ep are set (see
        run_forever). Combines the EP lockstep (every replica agrees phase+size each step, or the MoE
        collectives deadlock) with the synchronous spec step.

        Prefill uses the EP prefill lockstep (identical to ep_loop). A DECODE step is one of:
          * plain decode (any replica has a non-greedy req) — the EP graph decode, exactly like ep_loop.
          * spec decode (all replicas all-greedy) — the busy replica runs the real propose→verify→accept
            (forced EAGER under EP so the MoE self-coordinates its token count per layer, moe.py case 3);
            an idle replica issues _spec_num_forwards() EAGER dummy prefills so its per-layer MoE
            all_gathers line up 1:1 with the busy replica's propose+verify forwards. Dummy prefills reuse
            the proven _ep_prepare_dummy_prefill machinery (skip_alloc, allocates the dummy's recurrent
            slot on first use, snapshot/restore lengths) — the shapes differ from a verify forward but
            the MoE self-agreement equalizes N, which is the only cross-replica collective under EP
            (attention/CCA are DP-local; EP shards only the experts).
        """
        engine = self.engine
        ep = engine.ctx.ep

        for msg in self.receive_msg(blocking=False):
            self._process_one_msg(msg)

        prefill_batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        local_prefill_tokens = (
            sum(r.extend_len for r in prefill_batch.reqs) if prefill_batch is not None else 0
        )
        local_reqs: List[Req] = (
            self.decode_manager.ordered_reqs if self.decode_manager.runnable else []
        )
        local_decode_bs = len(local_reqs)
        # A replica with decode work vetoes spec iff ANY of its reqs can't spec-decode (non-greedy with
        # sampled-spec OFF, or constrained-and-sampled); a replica with NO decode work has nothing to
        # veto. Agreed via MAX over (1 - all_spec_ok): any 1 -> some replica must fall back.
        local_nongreedy = (
            1 if (local_reqs and not all(self._req_spec_ok(r) for r in local_reqs)) else 0
        )
        t = torch.tensor(
            [local_prefill_tokens, local_decode_bs, local_nongreedy], dtype=torch.int64
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=engine.dp_cpu_group)
        max_prefill, max_decode, any_nongreedy = int(t[0]), int(t[1]), int(t[2])

        if max_prefill == 0 and max_decode == 0:
            self.run_when_idle()
            return

        if max_prefill > 0:
            # ---- PREFILL lockstep (eager) — identical to ep_loop's prefill branch --------------------
            is_real = prefill_batch is not None
            ep.pad_tokens = max_prefill
            dr = engine.dummy_req
            saved_lens = None if is_real else (dr.cached_len, dr.device_len)
            try:
                forward_input = (
                    self._prepare_batch(prefill_batch)
                    if is_real
                    else self._ep_prepare_dummy_prefill()
                )
                data = (forward_input, self._forward(forward_input, track_reqs=is_real))
            finally:
                ep.pad_tokens = None
                if saved_lens is not None:
                    dr.cached_len, dr.device_len = saved_lens
            if is_real:
                self._process_last_data(data)
            return

        # ---- DECODE lockstep ----------------------------------------------------------------------
        ep.pad_tokens = None
        if any_nongreedy:
            # Some replica has a non-greedy req -> ALL fall back to the EP graph decode (one forward),
            # exactly like ep_loop's decode branch. Keeps every replica issuing one matching forward.
            graph_bs_list: List[int] = engine.graph_runner.graph_bs_list
            common_bs = max_decode
            if graph_bs_list:
                bigger = [b for b in graph_bs_list if b >= max_decode]
                if bigger:
                    common_bs = min(bigger)
            batch = Batch(reqs=local_reqs, phase="decode") if local_reqs else None
            forward_input = self._ep_prepare_decode(batch, common_bs)
            data = (forward_input, self._forward(forward_input, track_reqs=bool(local_reqs)))
            if local_reqs:
                self._process_last_data(data)
            return

        # SPEC decode lockstep: busy replica runs the real step; idle replica issues n_ep matching dummy
        # forwards. Two regimes, chosen IDENTICALLY on every replica (same config + agreed max_decode):
        #   * CAPTURED (ep_cap_bs set): the two-forward path where every EP forward is the SAME captured
        #     K+1 verify graph (busy = block_predict + verify; idle = n_ep× block_predict). All replicas
        #     agree one verify bs and replay that graph, so the fixed-N MoE all_gather matches — the same
        #     mechanism as the plain-decode EP capture (ep.py). _ep_common_bs threads the agreed bs into
        #     block_predict + verify (lifts their EP eager gate, forces pad-to-ep_cap_bs, lets the idle
        #     dummy capture). Big TPOT win: the wide draft/verify forwards stop paying eager host dispatch.
        #   * EAGER (ep_cap_bs None): fused/ddtree (distinct or own graph), MTP, or max_decode past the
        #     largest captured verify bs — the per-layer MoE self-agrees N across replicas (moe.py case 3),
        #     exactly the prior behavior.
        ep_cap_bs = self._ep_spec_common_verify_bs(max_decode)
        self._ep_common_bs = ep_cap_bs
        try:
          if local_reqs:
            if getattr(self, "_tidar_fused", False):
                self._spec_decode_step_tidar_fused(
                    local_reqs, self._proposer.block_size, self._proposer.mask_token_id
                )
            else:
                self._spec_decode_step(local_reqs)
          else:
            # Idle replica: issue n_ep dummy forwards that take the SAME spec_verify-DECODE model path
            # as the busy replica's FULL-TARGET (EP-collective-issuing) forwards. A dummy PREFILL
            # forward diverges in the CCA path (md.is_prefill branch, zaya.py) so the per-layer MoE
            # collectives desync and both ranks wedge — so we replay _tidar_block_predict on the dummy
            # instead: it IS the spec_verify-decode path, state-neutral, and works for ANY proposer
            # (the busy VERIFY forward is the same full-target spec_verify-decode shape regardless of
            # how drafts were proposed). skip_alloc uses the reserved null page (frees nothing). It reads
            # decode-phase recurrent state, so ensure the dummy has a registered slot (a prior dummy
            # prefill registers it, but don't rely on ordering).
            #   n_ep counts only the FULL-TARGET forwards: TiDAR=2 (block_predict+verify), EAGLE3/DFlash/
            #   n-gram=1 (dense draft -> verify only). MTP with an EP-sharded draft head is unsupported
            #   (n_ep<0) — it needs the draft MoE replicated (build-time), which makes it n_ep=1.
            n_ep = self._spec_num_ep_forwards()
            if n_ep < 0:
                raise RuntimeError(
                    "MTP spec-decode under EP requires the MTP draft MoE to be built REPLICATED "
                    "(not EP-sharded); its propose issues a data-dependent number of MoE collectives "
                    "that an idle replica cannot match. Serve MTP without --enable-ep, or use a build "
                    "that replicates the MTP experts. (EAGLE3/DFlash/TiDAR work under EP.)"
                )
            dr = engine.dummy_req
            if self.cca_slots is not None and dr.uid not in self.cca_slots._slot_of:
                self.cca_slots._ensure_slots([dr])
            if self.gdn_slots is not None and dr.uid not in self.gdn_slots._slot_of:
                self.gdn_slots._ensure_slots([dr])
            proposer = self._proposer
            # num_draft==0 short-circuits _tidar_block_predict, so guard k>=1. For TiDAR clamp to the
            # block size; other proposers verify K+1 = num_draft+1 query rows, so k = num_draft.
            k = max(1, engine.spec_config.num_draft)
            if proposer is not None and hasattr(proposer, "block_size"):
                k = max(1, min(engine.spec_config.num_draft, proposer.block_size))
            mask_id = int(getattr(proposer, "mask_token_id", 0))
            for _ in range(n_ep):
                # Captured @ ep_cap_bs when set (matches the busy replica's block_predict+verify graph),
                # else eager self-agreed-N (the prior behavior). block_predict reads self._ep_common_bs.
                self._tidar_block_predict([dr], k, mask_id, skip_alloc=True)
        finally:
            self._ep_common_bs = None

    def _ep_spec_common_verify_bs(self, max_decode: int) -> "int | None":
        """The captured verify bs every replica pads to for a CAPTURED EP spec step, or None to stay
        EAGER (self-agreed-N). Deterministic per config + agreed max_decode, so every replica returns the
        SAME value (identical branch on all replicas — a divergence would wedge the MoE collective).

        Capturable ONLY when every EP-collective-issuing forward of the step is the SAME captured K+1
        verify graph — i.e. the two-forward path (busy: block_predict + verify; idle: n_ep× block_predict,
        all qlen=num_draft+1). Fused TiDAR (distinct fused_qlen graph) and the DDTree paths (own
        tree_qlen graph) mix graph shapes between busy and idle, so they keep the eager self-agreement;
        MTP (n_ep<0) can't match a fixed count. None also when verify graphs are off (--graph 0) or
        max_decode exceeds the largest captured verify bs (fall back to eager at N=max_decode)."""
        if (getattr(self, "_tidar_fused", False) or getattr(self, "_tidar_ddtree", False)
                or getattr(self, "_dflash_ddtree", False)):
            return None
        if self._spec_num_ep_forwards() < 0:  # MTP under EP — handled (raises) in the idle branch
            return None
        vbs = self.engine.graph_runner.verify_bs_list
        if not vbs or max_decode <= 0:
            return None
        bigger = [b for b in vbs if b >= max_decode]
        return min(bigger) if bigger else None

    def _spec_prefill_seeded(self, batch: Batch) -> None:
        """Prefill forward that ALSO captures the per-token target hidden over the prompt and seeds the
        proposer's persistent draft KV from it (the prompt-prefill draft-KV seed lever). Mirrors the
        normal prefill (`_forward` + `_process_last_data`) but runs the one forward with
        `return_hidden=True`, so the bonus token and the prompt hidden come from a SINGLE pass. Only
        reqs whose WHOLE prompt is in this forward (cached_len==0, not chunked) are seeded — a chunked
        or prefix-cache-hit prompt's earlier hidden isn't available here, so it falls back to the cold
        cache (still lossless, just no early-token lift)."""
        forward_input = self._prepare_batch(batch)
        # Plan the seed-eligible reqs + their hidden-row slices BEFORE the forward advances cached_len
        # (forward_batch calls complete_one). Prefill rows follow padded_reqs order, which equals
        # batch.reqs for a prefill (no CUDA-graph padding — see GraphRunner.pad_batch).
        real = {id(r) for r in batch.reqs}
        plan: List[Tuple[Req, int, int]] = []  # (req, hidden-row offset, prompt_len)
        offset = 0
        for req in batch.padded_reqs:
            ext = req.extend_len
            if (not isinstance(req, ChunkedReq)) and req.cached_len == 0 and id(req) in real:
                plan.append((req, offset, ext))
            offset += ext

        fi_batch, sample_args, input_mapping, output_mapping = forward_input
        fi_batch.input_ids = self.token_pool[input_mapping]
        out, last_hidden, aux_hidden = self.engine.forward_batch(
            fi_batch, sample_args, return_hidden=True
        )
        self.token_pool[output_mapping] = out.next_tokens_gpu
        self.decode_manager.filter_reqs(fi_batch.reqs)

        # Seed each eligible req's draft KV over its prompt slice, and carry the first-step seed hidden
        # (the prompt's LAST position produced the bonus token, so its hidden seeds the first propose —
        # exactly the row the decode-time carry would have used). Cloned off the big verify tensors so
        # they can be released. last_hidden/aux_hidden are still the prompt's hidden here (input_ids
        # has not yet had the bonus appended — that happens in _process_last_data below).
        for req, off, plen in plan:
            lh = last_hidden[off : off + plen] if last_hidden is not None else None
            ax = aux_hidden[:, off : off + plen] if aux_hidden is not None else None
            self._proposer.seed_prefill(req, lh, ax)
            if self._spec_needs_last_hidden and lh is not None:
                self._spec_last_hidden[req.uid] = lh[plen - 1].clone()
            if self._spec_capture_layer_ids and ax is not None:
                # Full-context: seed the buffer with the WHOLE prompt aux [num_aux, plen, hidden] so the
                # first block's drafter attends over the entire prompt (matches z-lab's prefill prefix).
                # Legacy: just the last prompt position [num_aux, hidden].
                self._spec_aux_hidden[req.uid] = (
                    ax[:, :plen].clone() if self._dflash_fullctx else ax[:, plen - 1].clone()
                )

        self._process_last_data((forward_input, out))

    @torch.inference_mode()
    def _tidar_block_predict(
        self, reqs: List[Req], k: int, mask_id: int, skip_alloc: bool = False, topk: int = 0
    ) -> List[List[int]]:
        """TiDAR forward #1 (self-draft): one causal target forward over ``[confirmed | mask×k]`` per
        req → k draft tokens (argmax at the k mask positions). Called by TiDARProposer.propose (bound
        at construction) because it needs this scheduler's batch machinery (token pool, paged-KV
        allocate, CCA/attn metadata) which a proposer can't reach.

        STATE-NEUTRAL — the real verify forward (#2) re-runs with the accepted drafts:
          * CCA/GDN recurrent state is snapshot before the forward and restored after (the mask block's
            advance is thrown away);
          * the speculative mask-KV pages allocated here are FREED before returning (allocate_paged is
            NOT idempotent — the verify path re-allocates cleanly), so no per-step page leak.
        Routes through the captured K+1 verify graph when shape+bs fit (k == num_draft, graphs on,
        EP gate); falls back to an eager forward otherwise. Both are state-neutral. Returns k drafts."""
        device = self.device
        page_table = self.engine.page_table

        # --- stage [confirmed@c0 | mask×k]: extend each req to k+1 query tokens --------------------
        saved_lens = [(r.device_len, r.cached_len) for r in reqs]
        m_rows: List[int] = []
        m_cols: List[int] = []
        for req in reqs:
            c0 = req.cached_len
            req.device_len = c0 + k + 1
            for j in range(k):
                m_rows.append(req.table_idx)
                m_cols.append(c0 + 1 + j)
        self.token_pool[
            torch.tensor(m_rows, dtype=torch.int64, device=device),
            torch.tensor(m_cols, dtype=torch.int64, device=device),
        ] = torch.full((len(m_rows),), mask_id, dtype=self.token_pool.dtype, device=device)

        # --- build the block-predict batch --------------------------------------------------------
        batch = Batch(reqs=reqs, phase="decode")
        batch.spec_verify = True  # multi-token extend → same paged-extend causal path as verify
        # skip_alloc: an EP idle-replica dummy call (see _spec_ep_loop). The reqs are the shared
        # dummy_req whose page_table row already points at the reserved NULL page; allocating real
        # pages here would leave stale (later-freed) page ids in that row and corrupt a subsequent
        # dummy forward. Reuse the null page (like _ep_prepare_dummy_prefill) and free nothing. The
        # forward's output is discarded — only its per-layer MoE collectives matter.
        if not skip_alloc:
            self.cache_manager.allocate_paged(reqs)

        # CAPTURE the block-predict forward (the advice's dispatch-tier fix). block_predict stages
        # [confirmed | mask×k] = k+1 uniform query tokens/req — the SAME fixed shape as the K+1 verify
        # graph — so when k == num_draft and the batch fits a captured bs it replays that captured graph
        # instead of an eager per-layer wide forward. This is the second (and last) eager wide forward in
        # the two-forward TiDAR path: the verify half already captures; block_predict was the only launch-
        # bound step left. Mirror the real verify's gating EXACTLY (_spec_decode_step): same use_vgraph
        # predicate, same pad_verify vs prepare_metadata branch, same EP gate. forward_verify re-checks
        # can_use_verify_graph internally, so consistency is by construction (block_predict inherits the
        # shipped verify path's behavior in every EP mode). Never captures on the skip_alloc dummy path
        # (the idle-replica MoE-lockstep call is self-agreed-N eager). STATE-NEUTRAL is preserved: the
        # verify capturer writes recurrent state to per-layer SCRATCH (not the live slots) and this call
        # never installs; the snapshot/restore below still rolls back the live slots either way.
        from minisgl.distributed import is_ep_over_tp

        # ep_bs: set by _spec_ep_loop during the coordinated EP spec lockstep — every replica agrees one
        # captured verify bs and replays the IDENTICAL graph, so the fixed-N MoE all_gather matches and
        # capture is safe under DP+EP (mirrors the plain-decode EP capture, ep.py). None off the EP
        # lockstep. When set it (a) LIFTS the EP eager gate and (b) forces pad to exactly ep_bs; it also
        # lets the skip_alloc idle-replica dummy CAPTURE (it must, to match the busy replica's forwards —
        # off the lockstep the dummy stays eager, self-agreed-N).
        ep_bs = getattr(self, "_ep_common_bs", None)
        use_vgraph = (
            self.engine.graph_runner.can_use_verify_graph(batch)
            and (not self.engine.enable_ep or is_ep_over_tp() or ep_bs is not None)
            and (not skip_alloc or ep_bs is not None)
        )
        if use_vgraph:
            if not getattr(self, "_tidar_bp_capture_logged", False):
                self._tidar_bp_capture_logged = True
                logger.info_rank0(
                    f"spec-decode: TiDAR block_predict CAPTURED (K+1 verify graph, qlen={k + 1}"
                    f"{', EP-coordinated' if ep_bs is not None else ''})")
            self.engine.graph_runner.pad_verify(batch, ep_bs)
        else:
            batch.padded_reqs = reqs
        batch.positions = _make_positions(batch, device)
        input_mapping = _make_input_tuple(batch, device)
        batch.out_loc = page_table[input_mapping]
        if not use_vgraph:
            self.engine.attn_backend.prepare_metadata(batch)
        batch.input_ids = self.token_pool[input_mapping]

        # recurrent metadata; snapshot state to roll back (this forward is discarded, never installed).
        # Under the captured path the recurrent verify capturer expects capture_verify_state metadata
        # (it stashes per-token conv/ssm into scratch during replay); the accepted-prefix install the real
        # verify does afterward is SKIPPED here, and the snapshot/restore makes the step state-neutral.
        cca_snapshot = gdn_snapshot = None
        if self.cca_slots is not None:
            from minisgl.cca.metadata import build_cca_metadata

            cca_idx = self.cca_slots.state_indices(batch)
            batch.cca_metadata = build_cca_metadata(batch, cca_idx, device)
            if use_vgraph:
                batch.cca_metadata.capture_verify_state = True
                batch.cca_metadata.verify_max_qlen = k + 1
            cca_snapshot = self.engine.cca_state.snapshot(cca_idx)
        if self.gdn_slots is not None:
            from minisgl.gdn.metadata import build_gdn_metadata

            gdn_idx = self.gdn_slots.state_indices(batch)
            batch.gdn_metadata = build_gdn_metadata(batch, gdn_idx, device)
            if use_vgraph:
                batch.gdn_metadata.capture_verify_state = True
                batch.gdn_metadata.verify_max_qlen = k + 1
            gdn_snapshot = self.engine.gdn_state.snapshot(gdn_idx)

        # --- one forward; argmax the k mask positions per req -------------------------------------
        logits = self.engine.forward_verify(batch)  # [sum(k+1), vocab]
        preds = logits.argmax(dim=-1).to(torch.int32).cpu()
        # DDTree: also stash the per-position top-K MARGINALS (ids + log-probs) of the k mask rows per
        # req, keyed by id(req), for build_draft_tree. Rows off+1..off+k are the block's L=k positions.
        self._ddtree_topk: dict[int, tuple] = {}
        if topk > 0:
            lp = torch.log_softmax(logits.float(), dim=-1)
            tv, ti = lp.topk(topk, dim=-1)  # [sum(k+1), topk]
            tv = tv.cpu().tolist(); ti = ti.cpu().tolist()
        drafts: List[List[int]] = []
        off = 0
        for _req in reqs:
            block = preds[off + 1 : off + 1 + k].tolist()  # rows 1..k == the mask positions
            drafts.append([int(t) for t in block])
            if topk > 0:
                rows = range(off + 1, off + 1 + k)
                self._ddtree_topk[id(_req)] = ([ti[r] for r in rows], [tv[r] for r in rows])
            off += k + 1

        # --- roll back the speculative state (recurrent + KV pages + lens) ------------------------
        if cca_snapshot is not None:
            self.engine.cca_state.restore(cca_snapshot)
        if gdn_snapshot is not None:
            self.engine.gdn_state.restore(gdn_snapshot)
        ps = self.cache_manager.page_size
        free_chunks: List[torch.Tensor] = []
        for (dl, cl), req in zip(saved_lens, reqs):
            # skip_alloc allocated nothing (null-page dummy) → free nothing; just restore lengths.
            if not skip_alloc:
                free_start = div_ceil(cl, ps) * ps
                free_end = div_ceil(req.device_len, ps) * ps  # req.device_len still = c0+k+1 here
                if free_end > free_start:
                    free_chunks.append(page_table[req.table_idx, free_start:free_end])
            req.device_len, req.cached_len = dl, cl
        if free_chunks:
            self.cache_manager._free(torch.cat(free_chunks))
        return drafts

    @torch.inference_mode()
    def _ddtree_tree_verify(self, reqs, trees):
        """Forward 2 of DDTree: run ONE target forward over [bonus | tree nodes] per req with an
        ANCESTOR-ONLY custom_mask, then greedy-walk each tree. STATE-NEUTRAL/throwaway — the tree KV is
        scattered and (on CCA) its conv reads wrong neighbours, so this ONLY discovers the accepted path;
        recurrent state is snapshot+restored and the speculative KV pages freed (like block_predict).
        Returns {id(req): (accepted_tokens, next_bonus)}.

        CUDA-graph-capturable at a FIXED tree size: each req's real tree (n_nodes ≤ budget+1) is PADDED
        up to ``tree_qlen = budget+1`` query rows so every verify batch has the same shape. Pad rows carry
        a dummy token at position c0 with an all-allowed mask row (their logits are discarded — the walk
        reads only the real n rows), so the padded forward is numerically identical to the exact-size one.
        The ancestor mask is fed through the captured graph's static mask buffer (see
        GraphRunner.capture_ddtree_verify_graphs); state neutrality is preserved by the snapshot/restore
        below (eager, around the replay) — the tree-verify never installs recurrent state."""
        from minisgl.spec.ddtree import ddtree_walk
        device = self.device
        page_table = self.engine.page_table
        NEG_INF = float("-inf")
        # Fixed padded query length = budget+1 (matches the captured graph). Fall back to the max real
        # tree size when no budget is configured (DDTree disabled path — should not happen here).
        budget = getattr(self, "_ddtree_budget", 0)
        # SEGMENTED (brief #3): stage seg_layout["n_query"] rows/req (real nodes + inserted ancestor
        # conv-context rows) so the CCA conv reads each node's true ancestor window. Fixed-topology only
        # (n_query is constant) → capturable. Else the plain node-per-row layout (tree_qlen = budget+1).
        seg = getattr(self, "_ddtree_seg", False)
        fuse = getattr(self, "_ddtree_fuse", False)
        seg_lay = getattr(self, "_ddtree_seg_layout", None)
        if fuse:
            # FUSED (#2): base seg tree rows + rank-0 replica rows. base rows carry tree tokens; repctx
            # rows carry the copied rank-0 node's token; replica mask rows carry mask_id.
            lay = self._ddtree_fused_layout
            rep = lay["rep"]; lpos = lay["positions"]; node_rows = lay["node_rows"]
            extra = lay["extra"]; n_base = lay["n_base"]; mask_id = self._ddtree_mask_id
            tree_qlen = lay["n_query"]
        elif seg:
            rep = seg_lay["rep"]; seg_pos = seg_lay["positions"]; node_rows = seg_lay["node_rows"]
            tree_qlen = seg_lay["n_query"]
        else:
            tree_qlen = (budget + 1) if budget else max(trees[id(r)].n_nodes for r in reqs)
        saved_lens = [(r.device_len, r.cached_len) for r in reqs]
        store_rows, store_cols, tok_vals, pos_list = [], [], [], []
        per_req = []  # (req, c0, tree, n)
        for req in reqs:
            c0 = req.cached_len
            tree = trees[id(req)]
            n = tree.n_nodes
            assert n <= tree_qlen, f"tree n_nodes={n} exceeds tree_qlen={tree_qlen} (budget={budget})"
            for r in range(tree_qlen):
                store_rows.append(req.table_idx)
                store_cols.append(c0 + r)
                if fuse:
                    if r < n_base:                             # base seg tree row
                        tok_vals.append(tree.token[rep[r]]); pos_list.append(c0 + lpos[r])
                    else:                                      # replica / repctx row
                        e = extra[r - n_base]
                        tok_vals.append(tree.token[e[3]] if e[0] == "repctx" else mask_id)
                        pos_list.append(c0 + lpos[r])
                elif seg:
                    # row r represents node rep[r] (a real node OR an ancestor ctx COPY); token = that
                    # node's id, RoPE pos c0 + its depth (seg_pos is c0=0-relative). Fixed n_query rows.
                    m = rep[r]
                    tok_vals.append(tree.token[m])
                    pos_list.append(c0 + seg_pos[r])
                else:
                    # real nodes r<n: token/pos of node r. pad rows r>=n: dummy token 0 at pos c0 (root).
                    tok_vals.append(tree.token[r] if r < n else 0)
                    pos_list.append(c0 + (tree.depth[r] if r < n else 0))
            req.device_len = c0 + tree_qlen
            per_req.append((req, c0, tree, n))
        rows_t = torch.tensor(store_rows, dtype=torch.int64, device=device)
        cols_t = torch.tensor(store_cols, dtype=torch.int64, device=device)
        self.token_pool[rows_t, cols_t] = torch.tensor(tok_vals, dtype=self.token_pool.dtype, device=device)
        batch = Batch(reqs=reqs, phase="decode")
        batch.spec_verify = True
        batch.ddtree_verify = True  # routes forward_verify to the DDTree tree-verify graph when capturable
        self.cache_manager.allocate_paged(reqs)
        batch.padded_reqs = reqs
        batch.positions = torch.tensor(pos_list, dtype=torch.int32, device=device)
        batch.out_loc = page_table[rows_t, cols_t]
        self.engine.attn_backend.prepare_metadata(batch)
        batch.input_ids = self.token_pool[rows_t, cols_t]
        max_kv = int(batch.attn_metadata.max_seqlen_k)
        # zeros base (all-allowed): cols beyond a req's context [c0+tree_qlen, max_kv) are never read
        # (the kernel bounds by per-req cache_seqlen). Pad rows stay all-allowed; real rows get the
        # ancestor mask (own+ancestor tree cols allowed, every other tree-local col denied).
        custom_mask = torch.zeros((tree_qlen * len(reqs), max_kv),
                                  dtype=torch.float32, device=device)
        # STATIC template: the ancestor structure is fixed, so slice-place the BAKED [n,n] block instead
        # of the per-node O(n*depth) host rebuild (the tree is full — n == tree_qlen, no pad rows). DYNAMIC
        # heap: build the ancestor mask per step (topology varies).
        tmpl_block = getattr(self, "_ddtree_template_block", None)
        seg_block = getattr(self, "_ddtree_seg_block", None)  # [n_query, n_query] baked seg mask (c0=0)
        fused_block = getattr(self, "_ddtree_fused_block", None) if fuse else None
        off = 0
        for (req, c0, tree, n) in per_req:
            block = custom_mask[off:off + tree_qlen]
            if fused_block is not None:
                block[:tree_qlen, c0 : c0 + tree_qlen] = fused_block  # baked tree+replica mask
            elif seg_block is not None:
                block[:tree_qlen, c0 : c0 + tree_qlen] = seg_block  # baked seg ancestor+ctx mask
            elif tmpl_block is not None:
                block[:tree_qlen, c0 : c0 + tree_qlen] = tmpl_block  # baked deny/allow ancestor mask
            else:
                for j in range(n):  # real rows only; pad rows [n, tree_qlen) stay all-allowed
                    block[j, c0 : c0 + tree_qlen] = NEG_INF  # deny the whole tree-local block first
                    a = j
                    while a != -1:  # then re-allow own column + the ancestor chain to the root
                        block[j, c0 + a] = 0.0
                        a = tree.parent[a]
            off += tree_qlen
        batch.attn_metadata.custom_mask = custom_mask
        # recurrent metadata with per-token verify-state capture (capture_verify_state=True routes the
        # GDN/CCA layers through the bit-stable verify kernels that write only SCRATCH; the real slots are
        # additionally snapshot+restored below, so the tree-verify is throwaway/state-neutral). This same
        # verify path is what the captured graph threads through static scratch buffers.
        cca_snap = gdn_snap = None
        if self.cca_slots is not None:
            from minisgl.cca.metadata import build_cca_metadata
            cca_idx = self.cca_slots.state_indices(batch)
            batch.cca_metadata = build_cca_metadata(batch, cca_idx, device)
            batch.cca_metadata.capture_verify_state = True
            batch.cca_metadata.verify_max_qlen = tree_qlen
            cca_snap = self.engine.cca_state.snapshot(cca_idx)
        if self.gdn_slots is not None:
            from minisgl.gdn.metadata import build_gdn_metadata
            gdn_idx = self.gdn_slots.state_indices(batch)
            batch.gdn_metadata = build_gdn_metadata(batch, gdn_idx, device)
            batch.gdn_metadata.capture_verify_state = True
            batch.gdn_metadata.verify_max_qlen = tree_qlen
            gdn_snap = self.engine.gdn_state.snapshot(gdn_idx)
        logits = self.engine.forward_verify(batch)
        argmax = logits.argmax(dim=-1).to(torch.int32).cpu().tolist()
        out = {}
        off = 0
        r0 = getattr(self, "_ddtree_rank0", None)
        replica_rows = self._ddtree_fused_layout["replica_rows"] if fuse else None
        Kfuse = int(os.environ.get("MINISGL_DDTREE_TOPK") or "8")
        for (req, c0, tree, n) in per_req:
            if fuse or seg:
                # node j's logit is at packed row node_rows[j] (real nodes are interleaved with ctx rows).
                node_argmax = [argmax[off + node_rows[j]] for j in range(n)]
            else:
                node_argmax = argmax[off:off + n]  # node j at row j
            acc, nb = ddtree_walk(node_argmax, tree)
            out[id(req)] = (acc, nb)
            if fuse:
                # FUSED next-block (#2): the replicas were drafted off the rank-0 top path. If the walk
                # DESCENDED rank-0 (each accepted token is the rank-0 child) AND the bonus CONTINUES
                # rank-0, the committed sequence == rank0[1..k+1], so replica R_{k+1} drafted the correct
                # next block — reuse its top-K as the next tree's marginals (skip block_predict next step,
                # ON-PATH). Otherwise the replicas are stale -> block_predict re-drafts next step (off-path).
                k = len(acc)
                depth_cap = len(replica_rows) - 1
                # ON-PATH = the walk DESCENDED the rank-0 chain (each accepted token is the rank-0 child)
                # and a replica R_{k+1} exists. R_{k+1} was drafted conditioned on rank0[1..k+1]; the
                # committed sequence is rank0[1..k] + bonus, so the last conditioning token differs from
                # the bonus — the SAME one-token speculation the shipping fused-TiDAR path makes (the tree
                # is a heuristic; F3 commits losslessly regardless). NOT requiring bonus==rank0[k+1] (which
                # can't hold: the walk stops precisely because the bonus is not a tree child).
                on_path = (k + 1 <= depth_cap
                           and all(acc[i] == tree.token[r0[i + 1]] for i in range(k)))
                fst = getattr(self, "_ddtree_fuse_stat", None) or {"on": 0, "tot": 0}
                fst["tot"] += 1; fst["on"] += int(on_path)
                self._ddtree_fuse_stat = fst
                if fst["tot"] % 100 == 0:
                    logger.info_rank0(f"[ddtree-fuse] on-path (F1 skipped) {fst['on']}/{fst['tot']} = "
                                      f"{fst['on'] / fst['tot']:.2f}")
                if on_path:
                    rr = replica_rows[k + 1]                       # R_{k+1}: block_len next-block rows
                    rowlog = torch.log_softmax(logits[[off + x for x in rr]].float(), dim=-1)
                    tv, ti = rowlog.topk(Kfuse, dim=-1)
                    req._ddtree_fused_next = (ti.cpu().tolist(), tv.cpu().tolist())  # [L][K] ids, logp
                else:
                    req._ddtree_fused_next = None
            off += tree_qlen
        # rollback recurrent state + speculative KV + lengths (throwaway)
        if cca_snap is not None:
            self.engine.cca_state.restore(cca_snap)
        if gdn_snap is not None:
            self.engine.gdn_state.restore(gdn_snap)
        ps = self.cache_manager.page_size
        free_chunks = []
        for (dl, cl), req in zip(saved_lens, reqs):
            fs = div_ceil(cl, ps) * ps
            fe = div_ceil(req.device_len, ps) * ps
            if fe > fs:
                free_chunks.append(page_table[req.table_idx, fs:fe])
            req.device_len, req.cached_len = dl, cl
        if free_chunks:
            self.cache_manager._free(torch.cat(free_chunks))
        return out
        return out

    @torch.inference_mode()
    def _spec_decode_step_ddtree(self, reqs: List[Req], B: int, mask_id: int) -> None:
        """DDTree spec step for TiDAR (correctness-first v1; MINISGL_TIDAR_DDTREE=1). Three forwards:
          1. block_predict(topk=K) -> per-position top-K MARGINALS + the argmax chain (state-neutral).
          2. _ddtree_tree_verify: build a B-budget draft TREE (build_draft_tree) and discover the
             accepted path (longest prefix of the target's argmax chain the tree covers).
          3. commit: a LINEAR verify over the contiguous [bonus | accepted-path] (correct causal + conv/
             CCA state) via _spec_decode_step's proven machinery, injected with the tree path as the
             drafts — it accepts all and commits KV/state/bonus.
        The tree buys a longer accepted path than the argmax chain; the 2-forward segmented-tree variant
        (removing forward 3) is the follow-up. K=MINISGL_DDTREE_TOPK, budget=MINISGL_DDTREE_BUDGET."""
        K = int(os.environ.get("MINISGL_DDTREE_TOPK") or "8")
        budget = self._ddtree_budget  # fixed at init so the tree size matches the captured tree_qlen
        from minisgl.spec.ddtree import build_draft_tree, fill_static_template
        tmpl = getattr(self, "_ddtree_template", None)
        fuse = getattr(self, "_ddtree_fuse", False)
        # FUSED (#2): a req that carries valid on-path fused marginals from the PREVIOUS step's rank-0
        # replica skips block_predict (F1) — that draft was already computed by the prior fused verify.
        # Off-path (or first step / fidelity check) reqs still run F1. Never fuses under a fidelity A/B.
        check = os.environ.get("MINISGL_DDTREE_FUSE_CHECK") == "1"
        fused_reqs = ([r for r in reqs if getattr(r, "_ddtree_fused_next", None) is not None]
                      if fuse and not check else [])
        need_f1 = [r for r in reqs if r not in fused_reqs]
        if need_f1:
            self._tidar_block_predict(need_f1, B, mask_id, topk=K)  # forward 1 (marginals) for these
        topk_map = self._ddtree_topk
        # FIDELITY CHECK: on-path fused marginals must match a fresh block_predict at the committed
        # position. When set, F1 runs for ALL reqs and we compare the stored fused top-K to it.
        if check and fuse:
            fmatch = getattr(self, "_ddtree_fmatch", None) or {"hit": 0, "tot": 0}
            for req in reqs:
                fn = getattr(req, "_ddtree_fused_next", None)
                if fn is None:
                    continue
                bp_ids, _ = topk_map[id(req)]
                fmatch["tot"] += 1
                # top-1 agreement per next-block position (the draft the tree's rank-0 child would take)
                if all(fn[0][m][0] == bp_ids[m][0] for m in range(len(bp_ids))):
                    fmatch["hit"] += 1
            self._ddtree_fmatch = fmatch
            if fmatch["tot"] and fmatch["tot"] % 20 == 0:
                logger.info_rank0(f"[ddtree-fuse-check] on-path fused==block_predict top-1 "
                                  f"{fmatch['hit']}/{fmatch['tot']} = {fmatch['hit']/fmatch['tot']:.3f}")
        # forward 2: tree-verify -> accepted path (build tree from fused marginals if present, else F1)
        trees = {}
        for req in reqs:
            fn = getattr(req, "_ddtree_fused_next", None) if (fuse and not check) else None
            ids, logp = fn if fn is not None else topk_map[id(req)]
            root = int(req.input_ids[req.cached_len])
            trees[id(req)] = (fill_static_template(tmpl, ids, root) if tmpl is not None
                              else build_draft_tree(logp, ids, budget, root))
        accepted = self._ddtree_tree_verify(reqs, trees)
        # metrics: tree accept-len vs argmax chain accept-len (the argmax chain is verified in forward 3)
        st = getattr(self, "_ddtree_stat", None) or {"tree": 0.0, "n": 0}
        for r in reqs:
            st["tree"] += len(accepted[id(r)][0]); st["n"] += 1
        self._ddtree_stat = st
        if st["n"] % 100 == 0:
            logger.info_rank0(f"[ddtree] mean tree accept-len={st['tree']/st['n']:.2f} over {st['n']} reqs "
                              f"(B={budget} K={K} block={B})")
        # forward 3: commit the accepted path per req via the linear verify (drafts = tree path)
        for req in reqs:
            req._tidar_drafts = accepted[id(req)][0]  # the accepted tokens as the drafts to commit
        self._spec_decode_step(reqs, ddtree_drafts=True)

    @torch.inference_mode()
    def _spec_decode_step_dflash_ddtree(self, reqs: List[Req]) -> None:
        """DDTree spec step for the DFlash block-diffusion drafter (MINISGL_DFLASH_DDTREE=1). Reuses the
        drafter-agnostic tree machinery (_ddtree_tree_verify) — the only DFlash-specific piece is where
        the per-position top-K MARGINALS come from: DFlash's ONE denoising forward inside propose (vs
        TiDAR's block_predict). Three forwards, mirroring _spec_decode_step_ddtree:
          1. DFlash propose(topk=K) -> per-position top-K marginals (ids+log-probs) stashed on the
             proposer, keyed by id(req). No persistent draft KV (DFlash rebuilds from captured aux).
          2. _ddtree_tree_verify: build a B-budget draft TREE (build_draft_tree, root = the confirmed
             token at cached_len) and discover the accepted path via ONE ancestor-masked target forward.
          3. commit: the linear verify over [bonus | accepted-path] (_spec_decode_step ddtree_drafts) —
             correct causal context, re-captures the target aux for the next block, emits + rolls back.
        K=MINISGL_DDTREE_TOPK (default 8), budget=MINISGL_DDTREE_BUDGET (default 32)."""
        K = int(os.environ.get("MINISGL_DDTREE_TOPK") or "8")
        budget = self._ddtree_budget  # fixed at init so the tree size matches the captured tree_qlen
        from minisgl.spec.ddtree import build_draft_tree, fill_static_template
        tmpl = getattr(self, "_ddtree_template", None)

        spec = self.engine.spec_config
        assert spec is not None and self._proposer is not None
        # ctx carries the target aux hidden captured at the PREVIOUS verify (DFlash's cross-block context).
        ctx = ProposeContext(
            self.device,
            last_hidden={r.uid: self._spec_last_hidden[r.uid]
                         for r in reqs if r.uid in self._spec_last_hidden}
            if self._spec_needs_last_hidden else None,
            aux_hidden={r.uid: self._spec_aux_hidden[r.uid]
                        for r in reqs if r.uid in self._spec_aux_hidden}
            if self._spec_capture_layer_ids else None,
        )
        # forward 1: DFlash denoise -> per-position top-K marginals (ignore the argmax chain it returns).
        self._proposer.propose(reqs, spec.num_draft, ctx, topk=K)
        topk_map = getattr(self._proposer, "_ddtree_topk", {})
        # forward 2: build the tree per req (root = confirmed token @cached_len) + tree-verify walk.
        trees = {}
        for req in reqs:
            root = int(req.input_ids[req.cached_len])
            entry = topk_map.get(id(req))
            if entry is None:  # no aux yet / budget-0 req -> degrade to plain decode
                if tmpl is not None:
                    # SEG/STATIC stage the FULL template topology (rep/node_rows reference all n_nodes),
                    # so a root-only tree (n_nodes=1) would index out of range. Fill the template with a
                    # dummy (all-root) marginal -> full topology, tokens all = root so the walk accepts
                    # nothing (same plain-decode degradation as the root-only tree).
                    L = getattr(self, "_ddtree_block_len", spec.num_draft)
                    trees[id(req)] = fill_static_template(tmpl, [[root] * K for _ in range(L)], root)
                else:
                    trees[id(req)] = build_draft_tree([], [], 0, root)
            else:
                ids, logp = entry
                trees[id(req)] = (fill_static_template(tmpl, ids, root) if tmpl is not None
                                  else build_draft_tree(logp, ids, budget, root))
        accepted = self._ddtree_tree_verify(reqs, trees)
        # metric: mean tree accept-len (the DDTree win at block-16 vs the argmax chain).
        st = getattr(self, "_ddtree_stat", None) or {"tree": 0.0, "n": 0}
        for r in reqs:
            st["tree"] += len(accepted[id(r)][0]); st["n"] += 1
        self._ddtree_stat = st
        if st["n"] % 100 == 0:
            logger.info_rank0(f"[ddtree] mean tree accept-len={st['tree']/st['n']:.2f} over {st['n']} reqs "
                              f"(B={budget} K={K} block={self._proposer._block_size})")
        # forward 3: commit the accepted path per req via the linear verify (drafts = tree path).
        for req in reqs:
            req._tidar_drafts = accepted[id(req)][0]
        self._spec_decode_step(reqs, ddtree_drafts=True)

    @torch.inference_mode()
    def _spec_decode_step_tidar_fused(self, reqs: List[Req], B: int, mask_id: int) -> None:
        """TiDAR FUSED single-forward step (Phase C): ONE forward over [confirmed | S | R_0..R_{B-1}]
        per req that BOTH verifies S (the prev block's drafts) AND pre-drafts the next block from B
        replicas — vs the two-forward path's separate block_predict + verify. The forward-count halves
        → speedup ≈ avg_accept+1 (vs (avg+1)/2). Lossless: the S-verify rows have correct causal/conv
        context; only replica draft QUALITY is capped by the flat conv (a later segmented-conv refine).

        Uses the validated pieces: fused_paged_layout (positions + custom_mask), the paged kernel's
        mask_bias arg (causal=0), verify_greedy (β=1), and the B.0 CCA verify-state capture/install.
        RoPE positions (§7.6, non-contiguous) are separate from the contiguous KV storage cols."""
        from minisgl.spec.tidar_mask import fused_paged_layout

        device = self.device
        page_table = self.engine.page_table
        # PROBE (MINISGL_TIDAR_FUSED_NOREP=1): drop the replicas -> query only [confirmed | S] and get
        # next drafts from a separate block_predict (NOT fused-fast). Isolates whether the losslessness
        # drift comes from replica contamination (mask/conv/KV) or the confirmed/S + state-install path.
        norep = os.environ.get("MINISGL_TIDAR_FUSED_NOREP") == "1"
        # SEGMENTED conv (MINISGL_TIDAR_SEG=1): give each replica R_r its correct conv left-context via
        # inserted ctx tokens (masked from attention) -> recovers draft acceptance (flat conv ~4x lower).
        seg = os.environ.get("MINISGL_TIDAR_SEG") == "1" and not norep
        # DUMP (MINISGL_TIDAR_DUMP=1): systematic draft-vs-reference diagnostic (see _tidar_fused_dump).
        dump = os.environ.get("MINISGL_TIDAR_DUMP") == "1" and not norep
        # TIME (MINISGL_TIDAR_TIME=1): per-step cost breakdown (stage/mask-build | forward | commit) —
        # the Step-0.5 cost pivot needs to know where the ~287ms/step goes. Syncs → adds overhead, so
        # a dedicated flag. Timestamps helper below.
        timeit = os.environ.get("MINISGL_TIDAR_TIME") == "1"
        # MIX (MINISGL_TIDAR_MIX_BETA < 1.0): logit-mixing "Trust-Diffusion" verify (TiDAR paper
        # §4.4.3 / Zyphra ZAYA1-8B-Diffusion's 7.7x sampler). Verify each draft position against
        # argmax(beta*p_ar + (1-beta)*R_i[0]) instead of pure p_ar. beta=1.0 (default) == the current
        # lossless-vs-AR-greedy path (bit-identical); beta<1.0 leans on the diffusion self-draft ->
        # higher acceptance but NOT lossless vs base AR. norep has no replicas -> forced back to 1.0.
        mix_beta = float(os.environ.get("MINISGL_TIDAR_MIX_BETA", "1.0"))
        if norep:
            mix_beta = 1.0

        def _tstamp():
            if timeit:
                torch.cuda.synchronize(device)
            return time.perf_counter()

        # PROFILE (MINISGL_TIDAR_PROFILE=1): capture a window of fused steps with torch.profiler to get
        # the per-KERNEL GPU-time breakdown (confirm/size the custom-mask attention hotspot). Enters the
        # profiler at step PROF_START, captures PROF_N steps, then dumps a key_averages table (sorted by
        # self CUDA time) + a chrome trace for TraceLens. rocprof is dead on gfx1201, so this is the tool.
        if os.environ.get("MINISGL_TIDAR_PROFILE") == "1":
            ps = self._prof_step = getattr(self, "_prof_step", 0) + 1
            PROF_START, PROF_N = 40, 8
            if ps == PROF_START:
                self._prof = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA])
                self._prof.__enter__()
            elif ps == PROF_START + PROF_N:
                self._prof.__exit__(None, None, None)
                logger.info_rank0("[tidar-prof] per-kernel GPU time over %d fused steps:" % PROF_N)
                logger.info_rank0("\n" + self._prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=25))
                try:
                    self._prof.export_chrome_trace("/engine/tools/fused_prof.pt.trace.json")
                    logger.info_rank0("[tidar-prof] trace -> /engine/tools/fused_prof.pt.trace.json")
                except Exception as e:  # noqa: BLE001
                    logger.info_rank0(f"[tidar-prof] trace export failed: {e}")

        t0 = _tstamp()
        tp = (int(self.engine.cca_state.conv_states.shape[-1])
              if (seg and self.engine.cca_state is not None) else 2)
        if seg:
            from minisgl.spec.tidar_mask import fused_paged_layout_segmented

        # --- bootstrap: first step has no carried drafts -> one block_predict per fresh req --------
        fresh = [r for r in reqs if getattr(r, "_tidar_drafts", None) is None]
        if fresh:
            boot = self._tidar_block_predict(fresh, B, mask_id)
            for r, d in zip(fresh, boot):
                r._tidar_drafts = d

        # DUMP fidelity gate: block_predict on the CURRENT (pre-step) committed == the IDEAL R_0 (both
        # condition on [committed|confirmed] and predict positions c0+1..c0+B). Captured here, before
        # staging, via the state-neutral two-forward call; compared to the fused R_0 in _tidar_fused_dump.
        # R_0 == bp0 => the paged fused forward faithfully reproduces block_predict (no port bug, the low
        # acceptance is the structural bonus-conditioning cap); a mismatch localizes a fused fidelity bug.
        bp0_map = {}
        if dump and getattr(self, "_tidar_dump_count", 0) < 8:
            for r in reqs:
                bp0_map[id(r)] = self._tidar_block_predict([r], B, mask_id)[0]

        # --- stage [confirmed | S=drafts | mask×B²] per req; build fused positions + custom_mask ---
        store_rows: List[int] = []
        store_cols: List[int] = []
        tok_vals: List[int] = []
        pos_list: List[int] = []
        per_req = []  # (req, c0, n_query, drafts, layout_or_None)
        for req in reqs:
            c0 = req.cached_len
            drafts = list(req._tidar_drafts)
            layout = None
            if seg:
                layout = fused_paged_layout_segmented(c0, B, tp, device=device)
                fpos = layout["positions"]; n_query = layout["n_query"]
                toks = []
                for (kind, _r, loc, abs_pos) in layout["rows"]:
                    if kind == "confirmed":
                        toks.append(int(req.input_ids[c0]))
                    elif kind == "S":
                        toks.append(drafts[loc])
                    elif kind == "ctx":  # actual token @ abs_pos (committed/confirmed) or a draft
                        toks.append(int(req.input_ids[abs_pos]) if abs_pos <= c0
                                    else drafts[abs_pos - (c0 + 1)])
                    else:  # R
                        toks.append(mask_id)
            else:
                fpos, _mask, n_query_full, _ = fused_paged_layout(c0, B, device=device)
                if norep:
                    n_query = 1 + B                                   # [confirmed | S] only
                    toks = [int(req.input_ids[c0])] + drafts
                    fpos = fpos[:n_query]
                else:
                    n_query = n_query_full
                    toks = [int(req.input_ids[c0])] + drafts + [mask_id] * (B * B)  # confirmed | S | R*
            assert len(toks) == n_query, (len(toks), n_query)
            for i in range(n_query):
                store_rows.append(req.table_idx)
                store_cols.append(c0 + i)
                tok_vals.append(toks[i])
            pos_list += fpos
            req.device_len = c0 + n_query  # extend_len = n_query -> cache_seqlens = context_len
            per_req.append((req, c0, n_query, drafts, layout))
        rows_t = torch.tensor(store_rows, dtype=torch.int64, device=device)
        cols_t = torch.tensor(store_cols, dtype=torch.int64, device=device)
        self.token_pool[rows_t, cols_t] = torch.tensor(tok_vals, dtype=self.token_pool.dtype, device=device)

        # --- build the fused batch (eager; graphs off for CCA serve) -------------------------------
        batch = Batch(reqs=reqs, phase="decode")
        batch.spec_verify = True
        # v2 S4: flag the FUSED custom-mask verify so forward_verify can route it through its captured
        # graph (can_use_fused_verify additionally checks every req stages fused_qlen tokens, so the
        # NOREP probe (1+B) and partial-K steps auto-fall-back to eager). No-op when graphs are off.
        batch.fused_verify = not norep
        self.cache_manager.allocate_paged(reqs)
        batch.padded_reqs = reqs
        batch.positions = torch.tensor(pos_list, dtype=torch.int32, device=device)  # §7.6 RoPE positions
        batch.out_loc = page_table[rows_t, cols_t]                                    # contiguous KV cols
        self.engine.attn_backend.prepare_metadata(batch)
        batch.input_ids = self.token_pool[rows_t, cols_t]
        # assemble the packed custom mask [total_q, max_kv]; each req's [n_query, context_len] block
        max_kv = int(batch.attn_metadata.max_seqlen_k)
        total_q = len(store_rows)
        custom_mask = torch.zeros(total_q, max_kv, dtype=torch.float32, device=device)
        off = 0
        for (req, c0, n_query, _drafts, layout) in per_req:
            if layout is not None:
                mask_blk = layout["mask"]                                   # segmented [n_query, c0+n_query]
            else:
                _, mask_blk, _, _ = fused_paged_layout(c0, B, device=device)  # [nq_full, c0+nq_full]
            custom_mask[off:off + n_query, : c0 + n_query] = mask_blk[:n_query, : c0 + n_query]
            off += n_query
        batch.attn_metadata.custom_mask = custom_mask

        # CCA verify-state capture (B.0) so we can install the accepted-prefix recurrent state.
        cca_state_indices = None
        if self.cca_slots is not None:
            from minisgl.cca.metadata import build_cca_metadata

            cca_state_indices = self.cca_slots.state_indices(batch)
            batch.cca_metadata = build_cca_metadata(batch, cca_state_indices, device)
            batch.cca_metadata.capture_verify_state = True
            batch.cca_metadata.verify_max_qlen = max(nq for (_, _, nq, _, _) in per_req)

        t_stage = _tstamp()
        logits = self.engine.forward_verify(batch)  # [total_q, vocab]
        t_fwd = _tstamp()
        vocab = logits.shape[-1]

        # --- verify S rows + select replica R_k; commit + carry ------------------------------------
        dump_cap: List[dict] = []
        bykacc: List[tuple] = []   # STEP-0a: (prev_k, n_drafts, n_accepted) to bucket accept by prior k
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        c_rows: List[int] = []
        c_cols: List[int] = []
        c_vals: List[int] = []
        free_chunks: List[torch.Tensor] = []
        install_batch_idx: List[int] = []
        install_t_index: List[int] = []
        ps = self.cache_manager.page_size
        n_proposed = n_accepted = n_emitted = 0
        off = 0
        for i, (req, c0, n_query, drafts, layout) in enumerate(per_req):
            lg = logits[off:off + n_query]
            off += n_query
            p_ar = lg[0:B + 1]                       # confirmed row + S rows -> predict positions c0..c0+B
            if mix_beta < 1.0:
                # Diffusion prediction of draft position i (conditioned on S[:i]) = R_i[0], the first
                # token of replica R_i — same conditioning AND predicted position as p_ar[i] (§4.4.3,
                # the "E->E''" case). Mix only the B draft rows; the bonus row (B) stays pure-AR.
                if layout is not None:                                                  # segmented
                    diff0 = lg[[layout["replica_rows"][r][1][0] for r in range(B)]]     # [B,V] = R_r[0]
                else:                                                                   # flat
                    diff0 = lg[B + 1:].reshape(B, B, vocab)[:, 0, :]                     # [B,V] = R_r[0]
                p_ar = p_ar.clone()  # lg is a view into logits — don't write the forward output in place
                p_ar[:B] = mix_beta * p_ar[:B] + (1.0 - mix_beta) * diff0
            target = p_ar.argmax(dim=-1).to(torch.int32).cpu().tolist()
            result = verify_greedy(drafts, target)  # emitted = drafts[:k] + bonus; num_accepted = k
            k = result.num_accepted
            n_proposed += len(drafts); n_accepted += k
            # STEP-0a de-risk: attribute THIS step's acceptance to the prior step's k (which selected the
            # replica that produced these carried drafts). If accept-after-k>=1 >> accept-after-k=0, the
            # fused cap is the k=0 bonus-conditioning effect and dissolves as diff_acc rises (see
            # docs/TRAINING_PLAN_ACCEPTANCE.md §"Step 0"). None on the very first (bootstrap) block.
            prev_k = getattr(req, "_tidar_drafts_from_k", None)
            if prev_k is not None and not norep:
                bykacc.append((prev_k, len(drafts), k))
            # next block starts AFTER the bonus (committed += drafts[:k]+bonus). R_r[0] sits at
            # position c0+1+r, so the replica drafting the post-bonus block is R_{k+1}, not R_k.
            r_sel = max(0, min(B - 1, k + 1))
            if norep:
                # probe: next drafts from a fresh block_predict (extra fwd) — isolates the verify path
                next_drafts = None  # filled after the commit (needs updated cached_len); see below
            elif layout is not None:                    # segmented: R_{r_sel} lives at these packed rows
                rrows = layout["replica_rows"][r_sel][1]
                next_drafts = [int(x) for x in lg[rrows].argmax(dim=-1).cpu().tolist()]
            else:                                       # flat: replicas are the contiguous tail rows
                rep_rows = lg[B + 1:].reshape(B, B, vocab)  # [replica r, token m, V]
                next_drafts = [int(x) for x in rep_rows[r_sel].argmax(dim=-1).cpu().tolist()]

            if dump and getattr(self, "_tidar_dump_count", 0) < 8:
                # capture ALL replicas' argmax [B][B] (not just the selected one) so the dump can tell
                # a selection/shift bug (some R_r matches gt) from a conditioning/fidelity bug (none do).
                if layout is not None:
                    reps_all = [[int(x) for x in lg[rr].argmax(dim=-1).cpu().tolist()]
                                for (_r, rr) in layout["replica_rows"]]
                else:
                    reps_all = lg[B + 1:].reshape(B, B, vocab).argmax(dim=-1).cpu().tolist()
                dump_cap.append(dict(req=req, c0=c0, drafts=list(drafts), target=list(target),
                                     k=k, r_sel=r_sel, reps=reps_all, next_drafts=list(next_drafts),
                                     bp0=bp0_map.get(id(req))))

            # emitted tokens, truncate at EOS
            keep: List[int] = []
            eos = False
            for tok in result.emitted:
                keep.append(tok)
                if (not req.sampling_params.ignore_eos) and tok in self.eos_token_ids:
                    eos = True
                    break
            n_emitted += len(keep)
            # confirmed@c0 already in pool + its KV computed this step; drafts[:k] already staged at
            # c0+1..c0+k (their KV computed this step); write only the bonus (emitted[-1]) at c0+len(keep).
            bonus_col = c0 + len(keep)
            c_rows.append(req.table_idx); c_cols.append(bonus_col); c_vals.append(keep[-1])
            req.append_host(torch.tensor(keep, dtype=req.input_ids.dtype))
            req.cached_len = c0 + len(keep)   # KV valid through c0+len(keep)-1 (confirmed + drafts[:k])
            req.device_len = req.cached_len + 1
            req._tidar_drafts = next_drafts
            req._tidar_drafts_from_k = k  # STEP-0a: k that selected R_{r_sel} producing next_drafts
            finished = eos or (not req.can_decode)
            # CCA state after the last KV-committed query token = query row len(keep)-1 (confirmed=row0,
            # drafts[:k] = rows 1..k; len(keep)=k+1 -> last committed row = k).
            if cca_state_indices is not None and not finished:
                install_batch_idx.append(i)
                install_t_index.append(len(keep) - 1)
            if keep:
                fr = ("stop" if eos else "length") if finished else None
                reply.append(DetokenizeMsg(uid=req.uid, next_token=keep[0], finished=finished,
                                           extra_tokens=keep[1:], finish_reason=fr))
            # free the speculative query KV cols beyond the committed prefix (rejected drafts + replicas)
            free_start = div_ceil(req.cached_len, ps) * ps
            free_end = div_ceil(c0 + n_query, ps) * ps
            if free_end > free_start:
                free_chunks.append(page_table[req.table_idx, free_start:free_end])
            if finished:
                new_finished_reqs.add(req)

        if c_vals:
            self.token_pool[
                torch.tensor(c_rows, dtype=torch.int64, device=device),
                torch.tensor(c_cols, dtype=torch.int64, device=device),
            ] = torch.tensor(c_vals, dtype=self.token_pool.dtype, device=device)
        if free_chunks:
            self.cache_manager._free(torch.cat(free_chunks))
        if cca_state_indices is not None and install_batch_idx:
            md = batch.cca_metadata
            sel = torch.tensor(install_batch_idx, dtype=torch.long, device=device)
            slots = cca_state_indices.to(torch.long)[sel]
            t_index = torch.tensor(install_t_index, dtype=torch.long, device=device)
            self.engine.cca_state.install_verify_state(md.conv_scratch, md.prev_scratch, slots, t_index)

        if dump and dump_cap:
            self._tidar_fused_dump(dump_cap, B, mask_id, new_finished_reqs)

        if norep:
            # probe: next drafts from a fresh block_predict on the committed prefix (snapshot/restores
            # the just-installed CCA state internally). Correctness-only — this makes the step 2-forward.
            still = [req for (req, _, _, _, _) in per_req if req not in new_finished_reqs]
            if still:
                boot = self._tidar_block_predict(still, B, mask_id)
                for r, d in zip(still, boot):
                    r._tidar_drafts = d

        for req in new_finished_reqs:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
        self.finished_reqs = new_finished_reqs
        self.send_result(reply)

        t_end = _tstamp()

        # MINISGL_SPEC_DEBUG=1: fused acceptance stats (the two-forward path logs [spec] separately).
        # accept_rate = accepted drafts / proposed (B/req/step); emitted/step includes the bonus token.
        if os.environ.get("MINISGL_SPEC_DEBUG") == "1":
            st = getattr(self, "_fused_stats", None)
            if st is None:
                st = self._fused_stats = {"prop": 0, "acc": 0, "emit": 0, "steps": 0, "byk": {},
                                          "t_stage": 0.0, "t_fwd": 0.0, "t_commit": 0.0}
            st["prop"] += n_proposed; st["acc"] += n_accepted; st["emit"] += n_emitted; st["steps"] += 1
            for pk, nd, ka in bykacc:  # STEP-0a: bucket accept by the prior step's k
                b = st["byk"].setdefault(pk, [0, 0])
                b[0] += nd; b[1] += ka
            if timeit:  # STEP-0.5 cost breakdown (only meaningful with the syncs enabled)
                st["t_stage"] += t_stage - t0; st["t_fwd"] += t_fwd - t_stage
                st["t_commit"] += t_end - t_fwd
            if st["steps"] % 50 == 0:
                ar = st["acc"] / max(1, st["prop"])
                # accept-after-prior-k: if k>=1 buckets >> k=0 bucket, the fused cap is the k=0
                # bonus-conditioning effect → dissolves as diff_acc rises (training gate 0a).
                byk = "  ".join(f"prevk={pk}:{b[1]}/{b[0]}={b[1]/max(1,b[0]):.2f}"
                                for pk, b in sorted(st["byk"].items()))
                logger.info_rank0(
                    f"[spec-fused] step={st['steps']} accept_rate={ar:.2f} "
                    f"draft_accepted={st['acc']}/{st['prop']} emitted/step={st['emit']/st['steps']:.2f}")
                logger.info_rank0(f"[spec-fused-0a] accept-by-prior-k:  {byk}")
                if timeit:
                    n = st["steps"]
                    logger.info_rank0(
                        f"[spec-fused-time] ms/step  stage={1e3*st['t_stage']/n:.1f}  "
                        f"forward={1e3*st['t_fwd']/n:.1f}  commit={1e3*st['t_commit']/n:.1f}  "
                        f"total={1e3*(st['t_stage']+st['t_fwd']+st['t_commit'])/n:.1f}")

    def _tidar_fused_dump(self, cap: List[dict], B: int, mask_id: int, finished: Set[Req]) -> None:
        """DIAGNOSTIC (MINISGL_TIDAR_DUMP=1): per-step draft-vs-reference dump for the fused replica
        mechanism — the systematic gate the reference `single_forward_ours.py --check-drafts` runs, to
        replace the plateaued blind one-at-a-time fixes. For a few steps it compares, POSITION BY
        POSITION, every replica R_0..R_{B-1} and the SELECTED next_drafts (R_{r_sel}) against:
          * ``gt`` = a fresh `_tidar_block_predict` on the just-committed prefix (the TWO-FORWARD path
            that measures 0.20 accept) — the ground-truth next block we WANT next_drafts to equal; and
          * the verify targets / bonus.
        It prints, for each R_r, a positionwise match string vs ``gt`` (direct) and vs ``gt`` shifted
        one position (drop R_r[0]). That separates the hypotheses:
          * SELECTION bug  — some R_r == gt but r != r_sel  -> fix r_sel.
          * SHIFT bug      — gt == R_r[1:] (the 'shift' column matches) -> drop the bonus-position token.
          * CONDITIONING   — no R_r matches gt at any shift, yet R_k[0] == bonus -> the replica for a
            rejection boundary conditions on the rejected draft[k], not the bonus (structural).
          * FIDELITY bug   — R_k[0] != bonus -> the fused forward itself diverges from block_predict
            (mask/positions/conv in the paged port), not an algorithm choice.
        `_tidar_block_predict` is state-neutral (snapshot/restore + page-free), so calling it here does
        not perturb the served stream; skip finished reqs (can't re-forward)."""
        def match(a: List[int], b: List[int]) -> str:
            return "".join("." if x == y else "X" for x, y in zip(a, b))
        for e in cap:
            req = e["req"]
            if req in finished:
                continue
            self._tidar_dump_count = getattr(self, "_tidar_dump_count", 0) + 1
            k, r_sel, target = e["k"], e["r_sel"], e["target"]
            drafts, reps, nd = e["drafts"], e["reps"], e["next_drafts"]
            bonus = target[k]                                  # committed token after k accepts
            rk = min(k, B - 1)                                 # replica conditioned on the accepted drafts
            bp0 = e.get("bp0")                                 # block_predict(pre-step committed) == ideal R_0
            gt = self._tidar_block_predict([req], B, mask_id)[0]   # two-forward next block (conditions on bonus)
            logger.info_rank0(f"[tidar-dump] #{self._tidar_dump_count} c0={e['c0']} k={k}/{B} r_sel={r_sel}")
            logger.info_rank0(f"  S drafts   = {drafts}")
            logger.info_rank0(f"  p_ar targ  = {target}  bonus=target[{k}]={bonus}")
            if bp0 is not None:
                logger.info_rank0(
                    f"  FIDELITY: R_0={reps[0]}  bp0(2fwd)={bp0}  R_0==bp0[{match(reps[0], bp0)}]"
                    f"  (mismatch => fused forward != block_predict = a PORT bug)")
            logger.info_rank0(f"  gt(2-fwd)  = {gt}   <- want next_drafts == this")
            for r in range(B):
                direct = match(reps[r], gt)
                shift = match(reps[r][1:], gt[: B - 1])
                tag = " <== r_sel" if r == r_sel else (" (R_k)" if r == rk else "")
                logger.info_rank0(f"  R_{r} = {reps[r]}  vs-gt[{direct}] vs-gt-shift1[{shift}]{tag}")
            logger.info_rank0(
                f"  next_drafts= {nd}  vs-gt[{match(nd, gt)}]   "
                f"FIDELITY R_k[0]={reps[rk][0]} bonus={bonus} match={reps[rk][0] == bonus}")

    def _req_spec_ok(self, req: Req) -> bool:
        """Whether a req may run through the spec step. Greedy reqs always can (lossless greedy verify).
        A non-greedy req can when sampled spec is enabled — unconstrained via verify_sampled, constrained
        via _verify_sampled_constrained (grammar-masked rejection). See docs/SAMPLED_SPEC_VERIFY.md.
        Thinking (reasoning-gate) requests DO speculate — the gate is enforced on the verify logits
        (EOS-suppression + budget force-</think>) in _spec_decode_step, see _apply_think_gate_spec."""
        sp = req.sampling_params
        return sp.is_greedy or self._spec_sampled

    def _bcast_drafts_tp(self, reqs: List[Req], drafts: List[List[int]]) -> List[List[int]]:
        """TP>1 lockstep: force every TP rank to verify rank0's drafts.

        The eager spec-verify's vocab-parallel lm-head all_gather (embedding.py `forward`/
        `logits_all_rows`) is sized by the number of scored rows = ``sum(len(draft)+1)``. The TP ranks
        run the SAME reqs, but each proposes drafts INDEPENDENTLY — a nondeterministic draft/verify
        kernel (atomics → an argmax near-tie flip) can make rank0 and rank1 disagree on a draft, so
        their verify batches get different row counts and the all_gather faults with an illegal
        address (the DFlash + EP-over-TP crash; same class as the structured-spec TP=2 divergence).
        Broadcasting rank0's drafts makes every rank build the byte-identical verify batch → identical
        collectives forever. LOSSLESS: ``verify_greedy`` corrects any draft, so the committed tokens
        are the target's greedy tokens regardless of which rank's drafts were verified — only the
        (already-nondeterministic) acceptance rate can move, never the output."""
        if self._tp_size <= 1:
            return drafts
        g = self.tp_cpu_group
        n = len(reqs)
        lens = torch.tensor(
            [len(d) for d in drafts] if self._tp_is_primary else [0] * n, dtype=torch.int64
        )
        g.broadcast(lens, root=0).wait()
        lens_l = [int(x) for x in lens.tolist()]
        total = sum(lens_l)
        if total == 0:
            return [[] for _ in range(n)]
        flat = (
            torch.tensor([t for d in drafts for t in d], dtype=torch.int64)
            if self._tp_is_primary else torch.zeros(total, dtype=torch.int64)
        )
        g.broadcast(flat, root=0).wait()
        out: List[List[int]] = []
        off = 0
        for L in lens_l:
            out.append([int(x) for x in flat[off:off + L]])
            off += L
        return out

    def _bcast_accept_tp(
        self, reqs: List[Req], results: List[Tuple[int, List[int], bool]]
    ) -> List[Tuple[int, List[int], bool]]:
        """TP>1 lockstep: force every rank to COMMIT rank0's accept outcome.

        The verify forward's per-position target logits are NOT bit-identical across TP ranks — the
        vocab-parallel lm-head all_gather and the MoE reductions accumulate with atomics, so `p` (and
        thus an argmax near-tie or a sampled rejection draw at min(1, p/q)) can flip on rank1 vs rank0.
        The accept path used to rely on "p is identical post-all_gather" and skip an outcome broadcast;
        that assumption is false, so the ranks could commit DIFFERENT tokens for a req -> it reaches
        EOS/length at different steps -> the ranks' decode req SETS drift -> `spec_ok` (the spec-vs-plain
        branch in `_spec_loop`) evaluates differently per rank -> one rank enters the gloo draft
        broadcast while the other enters a plain-decode all_gather -> the collectives desync and NCCL
        times out (60s) -> SIGABRT. Broadcasting rank0's (num_accepted, keep, eos) makes every rank
        commit the identical tokens, so reqs finish in lockstep and the branch never splits.

        LOSSLESS: only rank0 streams replies to the frontend (see run path), and no collective depends
        on the per-rank grammar matcher (its bitmask is applied host-side), so a non-primary rank's
        matcher drifting to rank0's tokens is invisible — its local verify decision is simply overwritten
        by rank0's. The committed sequence is always rank0's grammar-valid, greedy-correct output."""
        if self._tp_size <= 1:
            return results
        g = self.tp_cpu_group
        n = len(reqs)
        # header = [num_accepted*n | eos(0/1)*n | keep_len*n]; then the flat keep tokens.
        if self._tp_is_primary:
            na = [int(r[0]) for r in results]
            eos = [1 if r[2] else 0 for r in results]
            klen = [len(r[1]) for r in results]
            flat_keep = [int(t) for r in results for t in r[1]]
        else:
            na = [0] * n; eos = [0] * n; klen = [0] * n; flat_keep = []
        header = torch.tensor(na + eos + klen, dtype=torch.int64)
        g.broadcast(header, root=0).wait()
        h = [int(x) for x in header.tolist()]
        na, eos, klen = h[:n], h[n:2 * n], h[2 * n:3 * n]
        total = sum(klen)
        flat = (
            torch.tensor(flat_keep, dtype=torch.int64)
            if self._tp_is_primary else torch.zeros(total, dtype=torch.int64)
        )
        if total > 0:
            g.broadcast(flat, root=0).wait()
        fl = [int(x) for x in flat.tolist()]
        out: List[Tuple[int, List[int], bool]] = []
        off = 0
        for i in range(n):
            L = klen[i]
            out.append((na[i], fl[off:off + L], bool(eos[i])))
            off += L
        return out

    def _spec_decode_step(self, reqs: List[Req], ddtree_drafts: bool = False) -> None:
        spec = self.engine.spec_config
        assert spec is not None and self._proposer is not None
        device = self.device
        page_table = self.engine.page_table
        # MINISGL_SPEC_TIMING=1: per-phase wall-clock (propose / verify-forward / accept) to attribute
        # the synchronous spec step. Adds cuda.synchronize barriers, so DIAGNOSTIC-only.
        _timing = os.environ.get("MINISGL_SPEC_TIMING") == "1"
        if _timing:
            import time as _time
            torch.cuda.synchronize(device); _t0 = _time.perf_counter()

        # --- 1. propose drafts (proposer-specific: n-gram lookup / MTP head / draft model). The
        # proposer clamps per-req to the remaining budget; an empty list ⇒ plain decode for that req.
        # Draft-head proposers read the target hidden states captured at the PREVIOUS verify (keyed
        # by uid); n-gram declared neither, so these dicts are empty and ProposeContext is bare.
        capture = self._spec_needs_last_hidden or bool(self._spec_capture_layer_ids)
        ctx = ProposeContext(
            device,
            last_hidden={r.uid: self._spec_last_hidden[r.uid]
                         for r in reqs if r.uid in self._spec_last_hidden}
            if self._spec_needs_last_hidden else None,
            aux_hidden={r.uid: self._spec_aux_hidden[r.uid]
                        for r in reqs if r.uid in self._spec_aux_hidden}
            if self._spec_capture_layer_ids else None,
        )
        # DDTree forward 3: commit a pre-discovered accepted path (set on req._tidar_drafts by
        # _spec_decode_step_ddtree) instead of proposing — the linear verify accepts all + commits.
        if ddtree_drafts:
            drafts = [list(getattr(r, "_tidar_drafts", []) or []) for r in reqs]
        else:
            drafts = self._proposer.propose(reqs, spec.num_draft, ctx)
        # TP>1 lockstep: every rank verifies rank0's drafts so the eager lm-head all_gather sees an
        # identical row count on all ranks (else the verify batch desyncs → illegal-address fault).
        drafts = self._bcast_drafts_tp(reqs, drafts)

        # Partial-K → uniform padding for the verify GRAPH. `can_use_verify_graph` needs every req to
        # have exactly num_draft drafts (uniform qlen); a partial-K step (a req clamped near max_tokens
        # / a cold first block) otherwise falls to the EAGER verify — slower, and (under TP) the eager
        # lm-head all_gather is the desync surface the broadcast above guards. Padding each req's drafts
        # up to num_draft (filler 0 at the tail) makes the step uniform so it hits the captured GDN/CCA/
        # MLA verify graph instead. LOSSLESS: `drafts` (REAL) still drives accept — the target is sliced
        # to the real length per req, so the padded tail rows are verified-then-freed (their KV is
        # released by the normal rollback), and the bonus at the real length is causally correct (its
        # query position's input is the last REAL draft; the filler only ever feeds strictly-later,
        # discarded positions). Only when it actually helps: graphs captured, padded bs fits, real spec
        # work exists, and the step isn't already uniform. Skipped for the on-device accept path (it keys
        # accept on q_lens; padding would need a separate real-length arg) and for ddtree.
        staged_drafts = drafts
        pad_active = False
        if not ddtree_drafts:
            vbs = self.engine.graph_runner.verify_bs_list
            lens = [len(d) for d in drafts]
            if (vbs and len(reqs) <= vbs[-1] and any(L >= 1 for L in lens)
                    and not all(L == spec.num_draft for L in lens)):
                staged_drafts = [d + [0] * (spec.num_draft - len(d)) for d in drafts]
                pad_active = True
        if _timing:
            torch.cuda.synchronize(device); _t1 = _time.perf_counter()

        # --- 2. stage: extend each req to K_i+1 query tokens; write drafts into the token pool --
        # Confirmed token sits at position c0 (= cached_len); drafts go at c0+1 .. c0+K_i.
        d_rows: List[int] = []
        d_cols: List[int] = []
        d_vals: List[int] = []
        for req, d in zip(reqs, staged_drafts):  # staged (padded) drives the forward layout/qlen
            c0 = req.cached_len
            req.device_len = c0 + len(d) + 1  # extend_len = staged K_i+1 (uniform when pad_active)
            for j, tok in enumerate(d):
                d_rows.append(req.table_idx)
                d_cols.append(c0 + 1 + j)
                d_vals.append(tok)
        if d_vals:
            self.token_pool[
                torch.tensor(d_rows, dtype=torch.int64, device=device),
                torch.tensor(d_cols, dtype=torch.int64, device=device),
            ] = torch.tensor(d_vals, dtype=self.token_pool.dtype, device=device)

        # --- 3. build the verify batch (phase='decode' -> full per-position logits + extend) ---
        batch = Batch(reqs=reqs, phase="decode")
        batch.spec_verify = True  # multi-token; GDN/lm-head treat it like a prefill (see core.py)
        self.cache_manager.allocate_paged(reqs)
        # MLA verify CUDA graph: when every req drafted exactly num_draft (uniform K+1 query tokens)
        # and the batch fits a captured size, PAD with dummy verify reqs FIRST so positions/out_loc
        # cover the padded rows (dummy rows point at the dummy page — never stale real KV), then replay
        # the graph. Otherwise (partial-K step / graphs off) stay eager with dynamic metadata.
        # Under EP the verify MUST run eager: the in-graph MoE all_gather uses a FIXED N (the captured
        # bs), but an idle replica coordinates via the eager self-agreement path (moe.py case 3). Graph
        # (fixed N) vs eager (self-agreed N) would mismatch shapes across replicas and wedge the
        # collective. Eager verify -> every replica hits the self-coordinating path -> N agrees. The
        # _spec_ep_loop drives the idle replica's matching dummy forwards.
        # EP verify stays EAGER under DP+EP (the in-graph MoE all_gather pins a fixed N vs an idle
        # replica's self-agreed N) — UNLESS the EP spec lockstep coordinated a common verify bs
        # (_ep_common_bs): then every replica pads to that same captured bs and replays the IDENTICAL
        # graph, so the fixed-N all_gather matches (mirrors the plain-decode EP capture, ep.py). EP-OVER-TP
        # has no idle replica — the TP ranks run the SAME padded bs in lockstep, so capture is always safe.
        from minisgl.distributed import is_ep_over_tp
        ep_bs = getattr(self, "_ep_common_bs", None)
        use_vgraph = self.engine.graph_runner.can_use_verify_graph(batch) and (
            not self.engine.enable_ep or is_ep_over_tp() or ep_bs is not None
        )
        if use_vgraph:
            self.engine.graph_runner.pad_verify(batch, ep_bs)
        else:
            batch.padded_reqs = reqs
        batch.positions = _make_positions(batch, device)
        input_mapping = _make_input_tuple(batch, device)
        batch.out_loc = page_table[input_mapping]
        if not use_vgraph:
            self.engine.attn_backend.prepare_metadata(batch)
        batch.input_ids = self.token_pool[input_mapping]

        # GDN-hybrid: build the per-batch recurrent metadata (varlen, like a prefill — see
        # build_gdn_metadata + the spec_verify dispatch). Instead of snapshotting + re-advancing, the
        # verify forward CAPTURES the conv+ssm state AFTER each of the K+1 tokens into per-layer
        # scratch (gdn_prefill_verify / causal_conv1d_fwd_verify). After acceptance the scheduler
        # installs the state after the accepted prefix (index = committed-1) directly into the slot —
        # no 2x re-advance, and bit-exact vs 1-token decode. See SPEC_DECODE.md.
        gdn_state_indices = None
        if self.gdn_slots is not None:
            from minisgl.gdn.metadata import build_gdn_metadata

            gdn_state_indices = self.gdn_slots.state_indices(batch)
            batch.gdn_metadata = build_gdn_metadata(batch, gdn_state_indices, device)
            batch.gdn_metadata.capture_verify_state = True
            batch.gdn_metadata.verify_max_qlen = max(len(d) + 1 for d in staged_drafts)

        # CCA-hybrid (ZAYA): same lossless-verify pattern as GDN. Build the varlen (prefill-style)
        # CCA metadata (spec_verify already routes there) with capture ON: the CCA layer stashes the
        # per-token conv window + prev_hs (reconstructed in torch, no kernel — see capture_cca_verify_state)
        # and the scheduler installs the accepted-prefix state below. Without this the verify forward
        # advances CCA state to the last (rejected) draft with no rollback → not lossless.
        cca_state_indices = None
        if self.cca_slots is not None:
            from minisgl.cca.metadata import build_cca_metadata

            cca_state_indices = self.cca_slots.state_indices(batch)
            batch.cca_metadata = build_cca_metadata(batch, cca_state_indices, device)
            batch.cca_metadata.capture_verify_state = True
            batch.cca_metadata.verify_max_qlen = max(len(d) + 1 for d in staged_drafts)

        # --- 4. verify forward -> per-position argmax (greedy == sampling here) ----------------
        # Draft-head proposers also need the target's hidden states at the verified positions; the
        # engine returns them from the SAME forward (no extra pass). last_hidden [T, hidden], aux
        # [num_capture_layers, T, hidden] or None; T = sum(K_i+1). Stays on-device until we slice the
        # per-uid seed rows after acceptance (then drop the full tensors).
        last_hidden = aux_hidden = None
        if _timing:
            _t_stage = _time.perf_counter()  # CPU-side staging (steps 2-3) done; forward next
        if capture:
            logits, last_hidden, aux_hidden = self.engine.forward_verify(batch, return_hidden=True)
        else:
            logits = self.engine.forward_verify(batch)

        # Reasoning gate on the spec path: mask the verify logits IN PLACE for any req still inside <think>
        # (EOS-suppress under budget / force-</think> over budget) BEFORE the argmax/accept below, so a
        # thinking model speculates correctly instead of stopping mid-reasoning or truncating. The gate is
        # advanced from the committed rank0 tokens in pass 2. any_gated → those reqs take the host accept.
        any_gated = self._gate_mask_spec_logits(reqs, staged_drafts, logits)

        # Greedy acceptance + EOS truncation: either the on-device vectorized chain (one batched sync
        # of small per-req results — the concurrency lever) or the legacy per-position argmax .cpu() +
        # per-req Python verify_greedy/keep loop. The on-device path is byte-lossless (validated in
        # tools/validate_ondevice_accept.py + validate_eos_trunc.py) and only taken for an all-greedy,
        # unconstrained, non-ddtree batch — constrained reqs need the host matcher, so a batch with any
        # constrained req (or the FORCE_N0 / ddtree diagnostics) falls back to the per-req host path.
        force_n0 = os.environ.get("MINISGL_SPEC_FORCE_N0") == "1"
        any_constrained = any(r.sampling_params.is_constrained for r in reqs)
        # Sampled (rejection-sampling) verify engages for the non-greedy reqs in the batch. Per-step
        # generator seeded identically on every TP rank (same _spec_step) so the accept draws + residual/
        # bonus samples match across ranks (drafts are broadcast, p is identical post-all_gather) without
        # an outcome broadcast. Greedy reqs in the same batch still take verify_greedy below.
        any_sampled = self._spec_sampled and any(
            not r.sampling_params.is_greedy for r in reqs
        )
        gen: "torch.Generator | None" = None
        if any_sampled:
            gen = torch.Generator(device=device)
            gen.manual_seed(self._spec_seed_base + self._spec_step)
        self._spec_step += 1
        # The on-device EOS-truncate primitive compares against a SINGLE id; a multi-EOS model
        # (eos_token_ids is a set from generation_config) must use the host path to honor every stop
        # token. len<=1 => the on-device path is exact.
        use_ondevice = (
            self._spec_ondevice and not any_constrained and not ddtree_drafts and not force_n0
            and not pad_active  # padded layout: accept keys on real per-req len, not the staged q_lens
            and not any_sampled  # sampled reqs take the host rejection path (verify_sampled)
            and not any_gated    # reasoning-gate reqs need the host accept (</think> truncation + count)
            and len(self.eos_token_ids) <= 1
        )
        preds = None
        od_accepts: List[int] = []
        od_keeps: List[List[int]] = []
        od_eos: List[bool] = []
        if use_ondevice:
            # argmax stays on-device; the accept/EOS chain reads it and syncs only the small results.
            target_argmax = logits.argmax(dim=-1).to(torch.int32)  # [sum(K_i+1)], on GPU
            q_lens_t = torch.tensor(
                [len(d) + 1 for d in drafts], dtype=torch.int32, device=device
            )
            drafts_flat = [t for d in drafts for t in d]
            drafts_t = torch.tensor(drafts_flat, dtype=torch.int32, device=device)
            acc = accept_greedy_ondevice(target_argmax, drafts_t, q_lens_t, device)
            # use_ondevice already required len(eos_token_ids) <= 1, so there is exactly one stop id
            # (or none). Pull it; -1 (no real token id is negative) reproduces "never truncate".
            eos_id = next(iter(self.eos_token_ids), -1)
            ignore_mask = torch.tensor(
                [r.sampling_params.ignore_eos for r in reqs], dtype=torch.bool, device=device
            )
            trunc = truncate_at_eos_ondevice(
                acc.committed_flat, acc.committed_offsets, acc.committed_lens,
                eos_id, ignore_mask, device,
            )
            # The one batched sync: small [num_reqs] / [sum kept] host copies (vs the old per-position
            # preds + per-req Python). Replaces verify_greedy + the EOS keep-loop for every req.
            od_accepts = acc.num_accepted.cpu().tolist()
            kept_lens_host = trunc.kept_lens.cpu().tolist()
            kept_offsets_host = trunc.kept_offsets.cpu().tolist()
            kept_flat_host = trunc.kept_flat.cpu().tolist()
            od_eos = [bool(x) for x in trunc.kept_finished_eos.cpu().tolist()]
            od_keeps = [
                kept_flat_host[o : o + L] for o, L in zip(kept_offsets_host, kept_lens_host)
            ]
        else:
            preds = logits.argmax(dim=-1).to(torch.int32).cpu()  # [sum(K_i+1)]; this syncs
        if _timing:
            _t2 = _time.perf_counter()  # forward already synced by the .cpu() above

        # --- 5. accept + commit + rollback per req --------------------------------------------
        offset = 0
        total_emitted = 0
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        c_rows: List[int] = []
        c_cols: List[int] = []
        c_vals: List[int] = []
        free_chunks: List[torch.Tensor] = []
        accepted_counts: List[int] = []  # drafts accepted per req (drives proposer draft-state rollback)
        # GDN per-token-state install bookkeeping. For each still-running seq, record its BATCH index
        # (to gather the captured scratch) and the per-token index to install (= committed-1, the
        # state AFTER the last committed token). A finished seq frees its slot, so its state is moot.
        gdn_install_batch_idx: List[int] = []
        gdn_install_t_index: List[int] = []
        # Fresh per-uid target hidden seeds for the NEXT step's propose (draft-head proposers only).
        new_last_hidden: dict[int, torch.Tensor] = {}
        new_aux_hidden: dict[int, torch.Tensor] = {}
        # PASS 1: compute the accept OUTCOME (num_accepted, keep, eos) per req. This is the ONLY step
        # whose result can differ across TP ranks (the verify logits are not bit-identical), so it is
        # computed first for ALL reqs, then made rank0-authoritative below — before any commit.
        accept_results: List[Tuple[int, List[int], bool]] = []
        for i, (req, d, sd) in enumerate(zip(reqs, drafts, staged_drafts)):
            q_len = len(d) + 1            # REAL rows: drives accept (target slice + verify_greedy)
            staged_q_len = len(sd) + 1    # rows the forward actually laid out (== q_len unless padded)
            block_start = offset  # this req's first query row in the verify output ([sum staged q_len])
            offset += staged_q_len
            gate_tid = (
                self._grammar_think_gate.get(req.uid)
                if self._grammar_think_gate_enabled else None
            )
            if gate_tid is not None:
                # Reasoning phase: the verify logits for this req were already EOS-masked (under budget) or
                # </think>-forced (over budget) by _gate_mask_spec_logits. Accept UNCONSTRAINED (the schema
                # is gated off during reasoning, mirroring the plain path's all-ones bitmask) and TRUNCATE
                # at </think> so the answer regenerates next step, schema-constrained. The gate itself is
                # counted/opened from the committed rank0 tokens in pass 2.
                sp = req.sampling_params
                if gen is not None and not sp.is_greedy:
                    lblock = logits[block_start : block_start + q_len]
                    result = verify_sampled(
                        d, probs_from_logits(lblock, sp.temperature, sp.top_k, sp.top_p), gen
                    )
                else:
                    result = verify_greedy(d, preds[block_start : block_start + q_len].tolist())
                num_accepted_i = result.num_accepted
                keep = list(result.emitted)
                if gate_tid in keep:                 # </think> committed → reasoning ends here
                    j = keep.index(gate_tid)
                    keep = keep[: j + 1]
                    num_accepted_i = min(num_accepted_i, j)
                eos = False                          # EOS was masked out during the reasoning phase
                accept_results.append((num_accepted_i, list(keep), eos))
                continue
            if use_ondevice:
                # On-device chain already produced num_accepted + the EOS-truncated keep list for this
                # req (batch is unconstrained by the use_ondevice gate, so no matcher path). Byte-
                # identical to the host verify_greedy + keep-loop below.
                num_accepted_i = od_accepts[i]
                keep = list(od_keeps[i])
                eos = od_eos[i]
            elif gen is not None and not req.sampling_params.is_greedy:
                # Sampled rejection verify: accept draft_j w.p. min(1, p_j(draft_j)/q) (q=onehot(draft)),
                # residual/bonus sampled from p. Distributionally lossless.
                sp = req.sampling_params
                lblock = logits[block_start : block_start + q_len]
                matcher = (
                    self._grammar_matchers.get(req.uid) if sp.is_constrained else None
                )
                if matcher is not None:
                    # Constrained + sampled: grammar-masked rejection (masked p per position + matcher
                    # advance). A grammar-violating draft has masked p=0 -> always rejected.
                    result = self._verify_sampled_constrained(matcher, d, lblock, sp, gen, req.uid)
                else:
                    pblock = probs_from_logits(lblock, sp.temperature, sp.top_k, sp.top_p)
                    result = verify_sampled(d, pblock, gen)
            else:
                matcher = (
                    self._grammar_matchers.get(req.uid)
                    if req.sampling_params.is_constrained
                    else None
                )
                if matcher is not None:
                    # Structured output + spec: grammar-mask the verify argmax per position and advance
                    # the matcher through the accepted chain (lossless; see _verify_greedy_constrained).
                    # Needs the raw logit rows on host, not the precomputed unmasked argmax.
                    block = logits[block_start : block_start + q_len].float().cpu()
                    result = self._verify_greedy_constrained(
                        matcher, d, block, req.sampling_params.ignore_eos, req.uid
                    )
                else:
                    target = preds[block_start : block_start + q_len].tolist()
                    result = verify_greedy(d, target)
                if force_n0:
                    # Diagnostic: stage+verify drafts but accept none (emit only the bonus). Should be
                    # byte-identical to plain decode through the multi-query kernel — isolates whether
                    # the bug is in the verify forward vs. the accept/commit path.
                    result = result._replace(emitted=result.emitted[:1], num_accepted=0)

            if not use_ondevice:
                # Shared unpack for the sampled AND greedy/constrained branches (both produce `result`);
                # the use_ondevice branch already set num_accepted_i/keep/eos directly. This was the
                # site of the sampled+DDTree UnboundLocalError — the sampled branch never unpacked.
                num_accepted_i = result.num_accepted
                # Decide which emitted tokens to keep, truncating at EOS.
                keep = []
                eos = False
                for tok in result.emitted:
                    keep.append(tok)
                    if (not req.sampling_params.ignore_eos) and tok in self.eos_token_ids:
                        eos = True
                        break
            accept_results.append((num_accepted_i, list(keep), eos))

        # TP LOCKSTEP: commit rank0's accept outcome on EVERY rank so the ranks never drift on committed
        # tokens -> req finishes -> the spec-vs-plain branch in _spec_loop -> the collective sequence.
        # (No-op at TP=1.) See _bcast_accept_tp for why the per-rank verify can otherwise disagree.
        accept_results = self._bcast_accept_tp(reqs, accept_results)

        # PASS 2: commit + rollback per req using the rank0-authoritative outcome. Everything here is a
        # deterministic function of (num_accepted, keep, eos) + the already-synced drafts/reqs, so all
        # ranks perform identical KV/state mutations and finish the same reqs on the same step.
        offset = 0
        for i, (req, d, sd) in enumerate(zip(reqs, drafts, staged_drafts)):
            q_len = len(d) + 1
            staged_q_len = len(sd) + 1
            block_start = offset
            offset += staged_q_len
            num_accepted_i, keep, eos = accept_results[i]
            accepted_counts.append(num_accepted_i)
            # Reasoning gate advance (spec path): drive the budget count + </think> open-detect from the
            # COMMITTED rank0 tokens so every TP rank moves the gate identically (mirrors the plain path,
            # _process_last_data). keep was truncated at </think> in pass 1, so it is either all reasoning
            # (count all) or ends at </think> (count the prefix, then open the gate → schema/answer next step).
            if self._grammar_think_gate_enabled and req.uid in self._grammar_think_gate:
                _gtid = self._grammar_think_gate[req.uid]
                for _tok in keep:
                    if _tok == _gtid:
                        self._clear_think_gate(req.uid)
                        break
                    self._grammar_think_count[req.uid] = (
                        self._grammar_think_count.get(req.uid, 0) + 1
                    )
            c0 = req.cached_len

            # On-policy Draft-OPD capture (guarded; DFlash linear block only). One opdbuf record per
            # verify step: seed = target aux at the last committed position (what the CCA drafter's fc
            # seed conditions on — spec/dflash.py:454-461 slices aux[:, -1] then fuse_aux), anchor =
            # the confirmed token (input_ids[cached_len]), plus the drafter's block, its acceptance,
            # the target's correction, and the target top-K logits (soft-KL teacher). This matches the
            # train_drafter.py seed-fold recipe 1:1. See dflash-drafter/docs/CAPTURE_CONTRACT.md.
            if (self._opd_dir is not None and self._spec_capture_layer_ids and not use_ondevice
                    and req.sampling_params.is_greedy
                    and self._grammar_matchers.get(req.uid) is None):
                ax = self._spec_aux_hidden.get(req.uid)
                if ax is not None and 0 <= c0 < req.input_ids.shape[0]:
                    Kd = len(d)
                    seed = (ax[:, -1] if ax.dim() == 3 else ax).reshape(1, -1).half().cpu()
                    # Recompute the masked-argmax target for the reject-correct label (pass 1 no longer
                    # keeps `target` in scope). Guarded to not-use_ondevice above, so `preds` is set.
                    target = preds[block_start : block_start + q_len].tolist()
                    rc = int(target[num_accepted_i]) if num_accepted_i < Kd else -1
                    rec = {
                        "seed_in": seed,                                              # [1, n_aux*H]
                        "bonus": torch.tensor([int(req.input_ids[c0])], dtype=torch.long),
                        "draft_tokens": torch.tensor([list(d)], dtype=torch.long),    # [1, Kd]
                        "num_accepted": torch.tensor([int(num_accepted_i)], dtype=torch.long),
                        "reject_correct": torch.tensor([rc], dtype=torch.long),
                    }
                    if Kd > 0:
                        tk = min(16, logits.shape[-1])
                        lb = logits[block_start:block_start + Kd].float()             # [Kd, V]
                        idx = lb.topk(tk, dim=-1).indices
                        rec["target_topk_ids"] = idx.to(torch.long).cpu().unsqueeze(0)  # [1, Kd, tk]
                        rec["target_topk_logprob"] = (
                            torch.log_softmax(lb, -1).gather(-1, idx).cpu().unsqueeze(0))
                    self._opd_buf.append(rec)
                    if len(self._opd_buf) >= 512:
                        self._flush_opd()

            old_device_len = c0 + staged_q_len  # forward extended device_len by the STAGED qlen;
            #                                     free the padded tail pages too (rollback below)

            if os.environ.get("MINISGL_SPEC_DEBUG") in ("2", "3"):
                logger.info_rank0(
                    f"[spec-dbg] uid={req.uid} c0={c0} dev={req.device_len} "
                    f"conf={int(req.input_ids[c0])} k={len(d)} n={num_accepted_i} "
                    f"emit={keep}"
                )

            # Commit kept tokens to the host sequence + GPU token pool (positions c0+1 .. c0+len).
            for j, tok in enumerate(keep):
                c_rows.append(req.table_idx)
                c_cols.append(c0 + 1 + j)
                c_vals.append(tok)
            req.append_host(torch.tensor(keep, dtype=req.input_ids.dtype))
            req.cached_len = c0 + len(keep)  # KV valid through cached_len-1
            req.device_len = req.cached_len + 1
            total_emitted += len(keep)
            finished = eos or (not req.can_decode)
            # GDN: a still-running req's recurrent state must end after the accepted prefix. The verify
            # captured the state after each token; install scratch index committed-1 (state after the
            # last committed token). committed == len(keep) >= 1 (always >= the 1 bonus token) for a
            # non-EOS-truncated, still-running seq. Finished reqs free their slot, so state is moot.
            if (gdn_state_indices is not None or cca_state_indices is not None) and not finished:
                gdn_install_batch_idx.append(i)
                gdn_install_t_index.append(len(keep) - 1)

            # Draft-head seed: the target hidden at the row that PRODUCED the last committed token
            # (block_start + len(keep)-1 — same index the GDN install uses). The draft for the next
            # step is seeded from the accepted token's hidden state. Cloned off the big verify tensor
            # so the per-step [T, hidden] / [L, T, hidden] outputs can be released. Finished reqs are
            # gone next step, so we skip them.
            if capture and not finished and keep:
                row = block_start + len(keep) - 1
                if last_hidden is not None:
                    new_last_hidden[req.uid] = last_hidden[row].clone()
                if aux_hidden is not None:
                    if self._dflash_fullctx:
                        # Append THIS step's accepted positions' aux [num_aux, len(keep), hidden] to the
                        # running buffer. We only ever append ACCEPTED positions, so the buffer length
                        # stays == cached_len (committed) — no rollback needed on rejection.
                        new_slice = aux_hidden[:, block_start : block_start + len(keep)].clone()
                        prev = self._spec_aux_hidden.get(req.uid)
                        buf = (
                            torch.cat([prev, new_slice], dim=1)
                            if (prev is not None and prev.dim() == 3)
                            else new_slice
                        )
                        w = self._dflash_ctx_window
                        if w and buf.shape[1] > w:
                            buf = buf[:, -w:].contiguous()
                        new_aux_hidden[req.uid] = buf
                    else:
                        new_aux_hidden[req.uid] = aux_hidden[:, row].clone()

            # One message carries all of this step's committed tokens (the detokenizer keys
            # streaming state by uid and assumes one message per uid per batch).
            if keep:
                fr = ("stop" if eos else "length") if finished else None
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=keep[0],
                        finished=finished,
                        extra_tokens=keep[1:],
                        finish_reason=fr,
                    )
                )

            # Rollback: free WHOLE pages allocated for this step that fall entirely beyond the kept
            # run (page-size-aware: page_size 1 for MHA frees per token; 16 for MLA frees per page).
            # A partial page straddling cached_len stays (it holds valid KV); its rejected-draft tail
            # is overwritten as the seq grows back. The bonus token's slot is recomputed next step.
            ps = self.cache_manager.page_size
            free_start = div_ceil(req.cached_len, ps) * ps
            free_end = div_ceil(old_device_len, ps) * ps
            if free_end > free_start:
                # contiguous page-aligned token range; cache_manager._free strides by page_size.
                free_chunks.append(page_table[req.table_idx, free_start:free_end])

            if finished:
                new_finished_reqs.add(req)

        if c_vals:
            self.token_pool[
                torch.tensor(c_rows, dtype=torch.int64, device=device),
                torch.tensor(c_cols, dtype=torch.int64, device=device),
            ] = torch.tensor(c_vals, dtype=self.token_pool.dtype, device=device)
        if free_chunks:
            self.cache_manager._free(torch.cat(free_chunks))

        # GDN: install the captured accepted-prefix recurrent state into each still-running seq's slot.
        # The verify forward (gdn_prefill_verify) already computed the EXACT per-token conv+ssm state,
        # so this is a pure gather — no re-advance, no second forward. Bit-exact vs 1-token decode.
        if gdn_state_indices is not None and gdn_install_batch_idx:
            md = batch.gdn_metadata
            sel = torch.tensor(gdn_install_batch_idx, dtype=torch.long, device=device)
            slots = gdn_state_indices.to(torch.long)[sel]
            t_index = torch.tensor(gdn_install_t_index, dtype=torch.long, device=device)
            self.engine.gdn_state.install_verify_state(
                md.conv_scratch, md.ssm_scratch, slots, t_index
            )

        # CCA: install the captured accepted-prefix conv window + prev_hs into each still-running seq's
        # slot (same install_batch_idx/t_index bookkeeping — a model is GDN XOR CCA, never both).
        if cca_state_indices is not None and gdn_install_batch_idx:
            md = batch.cca_metadata
            sel = torch.tensor(gdn_install_batch_idx, dtype=torch.long, device=device)
            slots = cca_state_indices.to(torch.long)[sel]
            t_index = torch.tensor(gdn_install_t_index, dtype=torch.long, device=device)
            self.engine.cca_state.install_verify_state(
                md.conv_scratch, md.prev_scratch, slots, t_index
            )

        # Carry the fresh target hidden seeds to the next propose (replaces the consumed step's seeds
        # for these uids; finished uids drop out because they aren't in the fresh dict). No-op for
        # n-gram (capture False ⇒ both dicts stay empty).
        if capture:
            for uid in [r.uid for r in reqs]:
                self._spec_last_hidden.pop(uid, None)
                self._spec_aux_hidden.pop(uid, None)
            self._spec_last_hidden.update(new_last_hidden)
            self._spec_aux_hidden.update(new_aux_hidden)

        # Roll back any draft-owned state (draft KV / recurrent) to the accepted prefix. No-op for
        # n-gram; MTP/DFlash/EAGLE truncate their draft KV. (GDN backbone-state rollback is handled
        # separately in the verify forward path, not here — it is the target's state, not the draft's.)
        self._proposer.on_accept(reqs, accepted_counts)

        # Mean accepted-draft length (diagnostic, same 100-req cadence as the [ddtree] log). Gated to
        # the plain propose→verify path (ddtree_drafts=True is DDTree's commit forward — those drafts
        # are the already-discovered tree path, counted by the [ddtree] metric instead, so skip here).
        if not ddtree_drafts:
            ast = getattr(self, "_spec_accept_stat", None) or {"acc": 0.0, "n": 0}
            for na in accepted_counts:
                ast["acc"] += na; ast["n"] += 1
            self._spec_accept_stat = ast
            if ast["n"] % 100 == 0:
                logger.info_rank0(
                    f"[spec] mean accept-len={ast['acc']/ast['n']:.2f} over {ast['n']} reqs")
        # Metrics: one verify step for the batch; per-req draft/accepted/emitted totals (see
        # server/metrics.py -> minisgl_spec_*). Cheap int adds off the per-token path.
        if self._metrics_enabled:
            self._m_spec_steps += 1
            self._m_spec_draft_tokens += sum(len(d) for d in drafts)
            self._m_spec_accepted_tokens += sum(accepted_counts)
            self._m_spec_emitted_tokens += total_emitted

        for req in new_finished_reqs:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
        self.finished_reqs = new_finished_reqs
        self.send_result(reply)
        if _timing:
            torch.cuda.synchronize(device); _t3 = _time.perf_counter()
            ph = getattr(self, "_spec_phase", None)
            if ph is None:
                ph = self._spec_phase = {
                    "propose": 0.0, "stage": 0.0, "forward": 0.0, "accept": 0.0, "n": 0
                }
            ph["propose"] += _t1 - _t0
            ph["stage"] += _t_stage - _t1  # CPU-side verify-batch staging (steps 2-3)
            ph["forward"] += _t2 - _t_stage  # verify forward + preds.cpu() sync
            ph["accept"] += _t3 - _t2
            ph["n"] += 1
            if ph["n"] % 50 == 0:
                n = ph["n"]
                logger.info_rank0(
                    f"[spec-timing] step={n} propose={ph['propose']/n*1e3:.1f}ms "
                    f"stage={ph['stage']/n*1e3:.1f}ms forward={ph['forward']/n*1e3:.1f}ms "
                    f"accept={ph['accept']/n*1e3:.1f}ms "
                    f"total={(ph['propose']+ph['stage']+ph['forward']+ph['accept'])/n*1e3:.1f}ms"
                )
        self._spec_debug(reqs, drafts, total_emitted)

    def _spec_debug(self, reqs: List[Req], drafts: List[List[int]], emitted: int) -> None:
        # MINISGL_SPEC_DEBUG=1: accumulate acceptance stats and log the running mean every 50
        # steps on the primary rank. Inert otherwise (one dict lookup). proposed = sum K_i;
        # accepted drafts = emitted - num_reqs (each req emits 1 bonus + its accepted drafts).
        import os

        if os.environ.get("MINISGL_SPEC_DEBUG") != "1":
            return
        st = getattr(self, "_spec_stats", None)
        if st is None:
            st = self._spec_stats = {"steps": 0, "proposed": 0, "accepted": 0, "emitted": 0, "reqs": 0}
        st["steps"] += 1
        st["proposed"] += sum(len(d) for d in drafts)
        st["accepted"] += emitted - len(reqs)  # accepted drafts (excludes the per-req bonus)
        st["emitted"] += emitted
        st["reqs"] += len(reqs)
        if st["steps"] % 50 == 0:
            acc_rate = st["accepted"] / max(1, st["proposed"])
            toks_per_step = st["emitted"] / max(1, st["steps"])
            vinfo = getattr(self.engine.graph_runner, "_verify", None)
            vreplays = vinfo.get("replays", 0) if vinfo else 0
            logger.info_rank0(
                f"[spec] step={st['steps']} accept_rate={acc_rate:.2f} "
                f"draft_accepted={st['accepted']}/{st['proposed']} "
                f"emitted/step={toks_per_step:.2f} (reqs/step={st['reqs']/st['steps']:.1f}) "
                f"verify_graph_replays={vreplays}"
            )


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    reqs = batch.padded_reqs
    # DECODE fast path: every req extends by exactly one token, so positions == [cached_len per req].
    # One build over num_reqs elements — no per-req torch op (the hot path, run every decode step).
    if all(req.extend_len == 1 for req in reqs):
        indices_host = torch.tensor(
            [req.cached_len for req in reqs], dtype=torch.int32, pin_memory=True
        )
        return indices_host.to(device, non_blocking=True)
    # PREFILL / varlen: keep the per-req C-speed arange into the pinned buffer. A per-TOKEN Python
    # comprehension would be O(sum(extend_len)) Python — a real regression for long prompts (60k ctx).
    needed_size = sum(req.extend_len for req in reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len, req.device_len, dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # Vectorized: repeat each req.table_idx extend_len times via repeat_interleave — C-speed for both
    # decode (extend_len==1) and varlen prefill, no per-req .fill_() and no per-token Python list.
    reqs = batch.padded_reqs
    idx = torch.tensor([req.table_idx for req in reqs], dtype=torch.int64, pin_memory=True)
    lens = torch.tensor([req.extend_len for req in reqs], dtype=torch.int64)
    mapping_host = idx.repeat_interleave(lens)  # length == sum(extend_len) == len(batch.positions)
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
