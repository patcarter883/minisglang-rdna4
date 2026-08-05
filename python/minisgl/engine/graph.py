from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.layers.tp_overlap import inline_collectives, tp_overlap_chunks
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


@dataclass
class VerifyCaptureBuffer:
    """Static I/O buffers for a captured spec-decode VERIFY forward. Sized for the MAX token count
    bs*(K+1); a given replay uses the leading `bs*qlen` rows. `last_hidden`/`aux_hidden` are allocated
    only when the proposer needs target hidden states (MTP/EAGLE3); n-gram MLA spec captures logits
    only."""

    qlen: int
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    last_hidden: torch.Tensor | None
    aux_hidden: torch.Tensor | None

    @classmethod
    def init(cls, max_bs, qlen, vocab, hidden, num_aux, dtype, device) -> "VerifyCaptureBuffer":
        T = max_bs * qlen
        return cls(
            qlen=qlen,
            input_ids=torch.zeros(T, dtype=torch.int32, device=device),
            out_loc=torch.zeros(T, dtype=torch.int32, device=device),
            positions=torch.zeros(T, dtype=torch.int32, device=device),
            logits=torch.empty(T, vocab, dtype=torch.float32, device=device),
            last_hidden=(torch.empty(T, hidden, dtype=dtype, device=device)
                         if hidden is not None else None),
            aux_hidden=(torch.empty(num_aux, T, hidden, dtype=dtype, device=device)
                        if num_aux else None),
        )

    def view(self, qlen: int) -> "VerifyCaptureBuffer":
        """A narrower-width VIEW sharing the same allocations (adaptive verify width, spec/width.py).

        Every buffer here is FLAT over tokens (`T = padded_bs * qlen` leading rows), so a width whose
        qlen is smaller than the one this buffer was allocated at simply uses fewer leading rows —
        there is no per-width stride to get wrong and no reason to allocate a second set. The logits
        buffer alone is `max_bs*qlen*vocab*4` bytes (51 MB at bs=8, qlen=16, vocab=100352), so sharing
        it is what keeps a 3-width ladder affordable."""
        assert qlen <= self.qlen, (qlen, self.qlen)
        return VerifyCaptureBuffer(
            qlen=qlen,
            input_ids=self.input_ids, out_loc=self.out_loc, positions=self.positions,
            logits=self.logits, last_hidden=self.last_hidden, aux_hidden=self.aux_hidden,
        )

    def total(self, batch: Batch) -> int:
        return batch.padded_size * self.qlen

    def set_batch(self, batch: Batch) -> None:
        s = slice(self.total(batch))
        batch.input_ids = self.input_ids[s]
        batch.out_loc = self.out_loc[s]
        batch.positions = self.positions[s]

    def copy_from(self, batch: Batch) -> None:
        # batch holds padded_size*(K+1) tokens (scheduler built them over padded_reqs = real + dummy,
        # so the dummy tail carries valid dummy-page out_loc — never stale real KV slots). Copy ALL.
        s = slice(self.total(batch))
        self.input_ids[s] = batch.input_ids
        self.out_loc[s] = batch.out_loc
        self.positions[s] = batch.positions


@dataclass
class CanvasCaptureBuffer:
    """Static I/O for a captured BLOCK-DIFFUSION canvas step (DiffusionGemma).

    Flat over tokens, `T = bs * canvas_len` leading rows, exactly like VerifyCaptureBuffer — the
    canvas is a fixed-width multi-query batch and nothing about its buffers is per-request.

    Two fields have no analogue in any other capture family:

      * `self_cond` — the PREVIOUS denoising step's soft embedding, an INPUT that changes every
        step. The eager path passes `None` on the first step of a block and lets the model skip the
        self-conditioning MLP; a graph cannot branch, so this buffer is ZEROED for that case
        instead. That is exact, not an approximation: RMSNorm(0)=0, tanh-gelu(0)*0=0 and down_proj
        carries no bias, so the block contributes exactly zero and `x + 0.0` is bit-identical to
        skipping it (see DiffusionGemmaSelfConditioning.forward).
      * `hidden` — the OUTPUT is the backbone's final-norm hidden state, NOT logits. The LM head
        stays eager; see DiffusionGemmaForBlockDiffusion.forward_canvas_hidden for why (a captured
        head would pin ~1 GiB of vocab-sized transients per captured batch size).
    """

    qlen: int
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    self_cond: torch.Tensor
    hidden: torch.Tensor

    @classmethod
    def init(cls, max_bs: int, qlen: int, hidden_size: int, dtype, device) -> "CanvasCaptureBuffer":
        T = max_bs * qlen
        return cls(
            qlen=qlen,
            input_ids=torch.zeros(T, dtype=torch.int32, device=device),
            out_loc=torch.zeros(T, dtype=torch.int32, device=device),
            positions=torch.zeros(T, dtype=torch.int32, device=device),
            self_cond=torch.zeros(T, hidden_size, dtype=dtype, device=device),
            hidden=torch.empty(T, hidden_size, dtype=dtype, device=device),
        )

    def total(self, batch: Batch) -> int:
        return batch.padded_size * self.qlen

    def set_batch(self, batch: Batch) -> None:
        s = slice(self.total(batch))
        batch.input_ids = self.input_ids[s]
        batch.out_loc = self.out_loc[s]
        batch.positions = self.positions[s]

    def copy_from(self, batch: Batch, canvas_ids: torch.Tensor, self_cond: torch.Tensor) -> None:
        s = slice(self.total(batch))
        self.input_ids[s] = canvas_ids
        self.out_loc[s] = batch.out_loc
        self.positions[s] = batch.positions
        self.self_cond[s] = self_cond


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
    max_running_req: int | None = None,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        # Co-derive graph coverage from the admission cap so NO admissible decode batch size runs
        # fully eager. can_use_cuda_graph() gates on batch.size <= max_graph_bs; with the old hard
        # default (160 on non-H200) but max_running_req defaulting to 256, decode batches of 161..256
        # concurrent reqs silently fell back to a fully-eager forward — the worst case for launch
        # overhead. Cover max_running_req exactly instead: the graph-memory reservation in
        # Engine._graph_capture_bytes reproduces this same bs_list, so the KV pool is sized around the
        # larger capture (no capture-time OOM), and graph coverage scales with the concurrency the
        # operator actually configured. Fall back to the old free-memory heuristic only for direct
        # callers that don't supply max_running_req.
        if max_running_req is not None:
            cuda_graph_max_bs = max_running_req
        elif free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    # Decode graph bs-buckets: FINER granularity in the low range (small concurrent-decode batches),
    # coarser above. can_use_cuda_graph() replays the smallest bucket >= batch.size, so with the old
    # grid ([1,2,4] + range(8, max, 8)) a bs=9 decode replayed the bs=16 graph and pushed 7 dummy rows
    # through attention/GEMM/logits. The low buckets [1,2,4,8,12,16,24,32] cut worst-case low-bs padding
    # (bs 9-12 now pad to 12, not 16); the step widens to 16 past bs=32 because proportional padding is
    # small on big batches AND it keeps the graph COUNT bounded (each bucket costs capture time + a slice
    # of graph-pool memory). For max=256 this is ~22 graphs vs the old 35 — FEWER, despite the finer low
    # range. Engine._graph_capture_bytes reproduces THIS same list (calls this function), so the KV-pool
    # reservation stays matched to what is actually captured. Filter+append(max) also fixes a latent
    # over-reach in the old list (it returned [1,2,4] verbatim for max<4, capturing a bs above the cap).
    buckets = [1, 2, 4, 8, 12, 16, 24, 32]
    buckets += list(range(48, cuda_graph_max_bs + 1, 16))
    buckets.append(cuda_graph_max_bs)  # always cover the operator-configured max exactly
    return sorted({b for b in buckets if 1 <= b <= cuda_graph_max_bs})


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        gdn_state: object | None = None,
        cca_state: object | None = None,
        max_running_req: int | None = None,
        cam: object | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
            max_running_req=max_running_req,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        # GDN-hybrid models thread per-seq recurrent-state slots through static buffers for capture.
        self.gdn_capture = None
        if gdn_state is not None and self.max_graph_bs > 0:
            from minisgl.gdn.graph_capture import GDNGraphCapture

            self.gdn_capture = GDNGraphCapture(device, self.max_graph_bs)
        # CCA-hybrid (ZAYA) models thread their conv/prev_hs state slots the same way GDN does.
        self.cca_capture = None
        if cca_state is not None and self.max_graph_bs > 0:
            from minisgl.cca.graph_capture import CCAGraphCapture

            self.cca_capture = CCAGraphCapture(device, self.max_graph_bs)
        # CAM editable-memory decode tap: thread a static per-row bank buffer through the graph so the
        # L24 tap runs inside the captured decode (Phase 2). None when CAM is off or graphs are disabled.
        self.cam_capture = None
        if (cam is not None and getattr(cam, "enabled", False) and self.max_graph_bs > 0
                and not getattr(cam, "pointer_only", False)):
            # pointer-only CAM has no residual tap in the captured graph — delivery is a post-sample token
            # override (_process_last_data), so there is nothing to thread through the decode graph.
            from minisgl.cam.graph_capture import CAMGraphCapture

            inner = getattr(model, "model", model)
            self.cam_capture = CAMGraphCapture(cam, inner, device, self.max_graph_bs)
        # v2: stashed for the spec-VERIFY capturer (built in capture_verify_graphs, needs num_draft).
        self._cca_state = cca_state
        self.cca_verify = None
        # GDN-hybrid spec-VERIFY capturer (built in capture_verify_graphs, needs num_draft). Mirrors
        # the CCA verify capturer: per-GDN-layer conv/ssm static scratch threaded through the graph.
        self._gdn_state = gdn_state
        self.gdn_verify = None
        # v2 S4: the FUSED-verify capturer + its CCA-state static buffers (built in
        # capture_fused_verify_graphs, needs fused_qlen — a distinct Q from the K+1 verify above).
        self.cca_fused_verify = None
        self._fused_verify = None
        # DDTree draft-TREE verify capturer (built in capture_ddtree_verify_graphs, needs tree_qlen =
        # budget+1). State-NEUTRAL: the scheduler snapshots+restores the real recurrent slots around the
        # replay and NEVER installs, so the per-layer verify scratch is throwaway (pointer-stability only).
        self.gdn_ddtree_verify = None
        self.cca_ddtree_verify = None
        self._ddtree_verify = None
        # Spec-decode verify graphs are captured LATER (capture_verify_graphs), after the scheduler
        # builds the proposer + programs the target's aux-capture layers — None until then.
        self._verify = None
        # BLOCK-DIFFUSION canvas graphs (capture_canvas_graphs) — None for every model that is not a
        # block-diffusion checkpoint, which is what can_use_canvas_graph keys on.
        self._canvas = None
        self._verify_max_seq_len = max_seq_len
        self._verify_vocab = vocab_size
        import os as _os
        self._timing = ({"n": 0, "copy": 0.0, "prep": 0.0, "replay": 0.0}
                        if _os.environ.get("MINISGL_GRAPH_TIMING") == "1" else None)
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            if self.gdn_capture is not None:
                self.gdn_capture.prepare_for_capture(batch)
            if self.cca_capture is not None:
                self.cca_capture.prepare_for_capture(batch)
            if self.cam_capture is not None:
                self.cam_capture.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # inference_mode around BOTH the warmup and captured forwards: capture never needs
            # autograd, and with grad active the models' in-place-on-view ops (e.g. q_norm/k_norm
            # forward_inplace on a qkv-split view) trip the autograd view-guard, breaking capture.
            # Matches the serve forward path (Scheduler is @torch.inference_mode()); the offline LLM
            # path constructs the GraphRunner outside that decorator, so make it explicit here.
            with get_global_ctx().forward_batch(batch), torch.inference_mode():
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        if self.cam_capture is not None:
            self.cam_capture.after_capture()  # revert Python hook to eager single-bank path

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        # Env-gated per-step host-cost timing (MINISGL_GRAPH_TIMING=1). Diagnostics only: splits the
        # host wall of a captured decode step into copy_from (I/O staging), prepare_for_replay (attn +
        # recurrent metadata rebuild — the eager work NOT in the graph), and the g.replay() launch.
        # Prints running averages every 100 replays on rank0. Inert (no branches taken) when unset.
        if self._timing is not None:
            import time as _t
            t0 = _t.perf_counter()
            self.buffer.copy_from(batch)
            t1 = _t.perf_counter()
            g = self.graph_map[batch.padded_size]
            self.attn_backend.prepare_for_replay(batch)
            if self.gdn_capture is not None:
                self.gdn_capture.prepare_for_replay(batch)
            if self.cca_capture is not None:
                self.cca_capture.prepare_for_replay(batch)
            if self.cam_capture is not None:
                self.cam_capture.prepare_for_replay(batch)
            t2 = _t.perf_counter()
            g.replay()
            t3 = _t.perf_counter()
            tm = self._timing
            tm["n"] += 1
            tm["copy"] += t1 - t0
            tm["prep"] += t2 - t1
            tm["replay"] += t3 - t2
            if tm["n"] % 100 == 0:
                n = tm["n"]
                logger.info_rank0(
                    f"[graph-timing] n={n} host/step: copy_from={tm['copy']/n*1e3:.3f}ms "
                    f"prepare_for_replay={tm['prep']/n*1e3:.3f}ms replay_launch={tm['replay']/n*1e3:.3f}ms"
                )
            return self.buffer.logits[: batch.size]
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        if self.gdn_capture is not None:
            self.gdn_capture.prepare_for_replay(batch)
        if self.cca_capture is not None:
            self.cca_capture.prepare_for_replay(batch)
        if self.cam_capture is not None:
            self.cam_capture.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # ---- spec-decode VERIFY graph capture (MLA) ---------------------------------------------
    def capture_verify_graphs(
        self,
        model: BaseLLMModel,
        num_draft: int,
        bs_list: List[int],
        needs_hidden: bool,
        num_aux: int,
        hidden_size: int,
        dtype: torch.dtype,
        widths: "List[int] | None" = None,
    ) -> None:
        """Capture one verify graph per (width, bs). Called by the scheduler AFTER the proposer is
        built and aux-capture layers are programmed, so the captured forward stashes the aux/hidden the
        draft head consumes. Each graph runs model.forward over bs*(width+1) staged tokens; the
        MLA backend reads precomputed static verify indices (see MLABackend.init_verify_capture).

        `widths` is the ADAPTIVE VERIFY WIDTH ladder (spec/width.py `verify_width_ladder`): the small
        set of draft counts the scheduler is allowed to pick from per step. Defaults to `[num_draft]`,
        i.e. the single fixed width this used to capture. All widths share ONE VerifyCaptureBuffer
        (see `VerifyCaptureBuffer.view`) and ONE graph pool, so the incremental cost of a width is the
        graph-pool slice for its (width, bs) graphs, not another set of vocab-sized buffers."""
        if not bs_list or not hasattr(self.attn_backend, "init_verify_capture"):
            return logger.info_rank0("spec-verify CUDA graph: unsupported backend / disabled")
        widths = sorted({int(w) for w in (widths or [num_draft])})
        assert widths and widths[-1] <= num_draft, (widths, num_draft)
        if len(widths) > 1 and not hasattr(self.attn_backend, "set_verify_width"):
            # A backend that supports verify capture but cannot repoint its per-width statics would
            # replay a narrow graph against the WIDEST width's cu_seqlens/kbound/seq_idx — silently
            # wrong output, not an error. Capture the single widest instead. (hip + mla, the only two
            # backends with init_verify_capture, both implement it.)
            logger.warning_rank0(
                f"{type(self.attn_backend).__name__} has no set_verify_width; capturing ONE verify "
                f"width ({widths[-1]}) instead of {widths} — adaptive verify width is OFF."
            )
            widths = widths[-1:]
        qlen = widths[-1] + 1  # widest; the shared buffers/scratch are sized here
        dev = self.device
        max_bs = max(bs_list)
        self.attn_backend.init_verify_capture(self._verify_max_seq_len, bs_list, widths)
        # v2 S3: CCA-hybrid recurrent state through static verify buffers (per-CCA-layer conv/prev
        # scratch that the captured verify forward writes in place; see CCAVerifyGraphCapture).
        self.cca_verify = None
        if self._cca_state is not None:
            from minisgl.cca.graph_capture import CCAVerifyGraphCapture

            cs = self._cca_state
            self.cca_verify = CCAVerifyGraphCapture(
                dev, max_bs, widths,
                cca_layer_ids=range(cs.num_cca_layers),
                conv_dim=cs.conv_states.shape[2], conv_width=cs.conv_states.shape[3],
                hidden=cs.prev_hs.shape[2],
            )
        # GDN-hybrid recurrent state through static verify buffers (per-GDN-layer conv/ssm scratch the
        # captured verify forward writes in place; see GDNVerifyGraphCapture). conv_state shape is
        # (L, slots, conv_dim, conv_kernel-1); ssm_state (L, slots, num_v_heads, head_v_dim, head_k_dim).
        self.gdn_verify = None
        if self._gdn_state is not None:
            from minisgl.gdn.graph_capture import GDNVerifyGraphCapture

            gs = self._gdn_state
            cshape = gs.conv_state.shape
            sshape = gs.ssm_state.shape
            self.gdn_verify = GDNVerifyGraphCapture(
                dev, max_bs, widths,
                gdn_layer_ids=range(gs.num_gdn_layers),
                conv_dim=cshape[2], conv_width=cshape[3],
                num_v_heads=sshape[2], head_v_dim=sshape[3], head_k_dim=sshape[4],
                ssm_dtype=gs.ssm_dtype,
            )
        vbuf = VerifyCaptureBuffer.init(
            max_bs, qlen, self._verify_vocab,
            hidden_size if needs_hidden else None,
            num_aux, dtype, dev,
        )
        torch.cuda.synchronize(dev)
        free0 = get_free_memory(dev)
        logger.info_rank0(
            f"Capturing spec-verify CUDA graphs (widths={widths} -> qlens={[w + 1 for w in widths]}, "
            f"hidden={needs_hidden}, aux={num_aux}) sizes={sorted(bs_list)}; free {mem_GB(free0)}"
        )
        # DESCENDING width, then descending bs: the first (widest, biggest) capture allocates the
        # shared graph pool at its high-water mark and every narrower graph reuses that memory instead
        # of extending the pool. Capturing narrow-first would grow the pool once per width.
        by_qlen: Dict[int, dict] = {}
        pool = None
        todo = [(w, bs) for w in sorted(widths, reverse=True)
                for bs in sorted(bs_list, reverse=True)]
        for w, bs in tqdm(todo, desc="Capturing verify graphs", unit="graph",
                          disable=not get_tp_info().is_primary()):
            ql = w + 1
            ent = by_qlen.setdefault(ql, {"buf": vbuf.view(ql), "graphs": {}})
            graph = torch.cuda.CUDAGraph()
            # a dedicated dummy req with extend_len = ql (cached_len 0, device_len ql).
            vdummy = Req(
                input_ids=torch.zeros(ql, dtype=torch.int32, device="cpu"),
                table_idx=self.dummy_req.table_idx, cached_len=0, output_len=1, uid=-1,
                sampling_params=None, cache_handle=None,  # type: ignore
            )
            batch = Batch(reqs=[vdummy] * bs, phase="decode")
            batch.spec_verify = True
            batch.padded_reqs = batch.reqs
            self._set_verify_width(ql)
            self.attn_backend.prepare_verify_for_capture(batch)
            if self.cca_verify is not None:
                self.cca_verify.prepare_verify_for_capture(batch)
            if self.gdn_verify is not None:
                self.gdn_verify.prepare_verify_for_capture(batch)
            wbuf: VerifyCaptureBuffer = ent["buf"]
            wbuf.set_batch(batch)
            T = wbuf.total(batch)
            with get_global_ctx().forward_batch(batch), torch.inference_mode():
                self._run_verify_into(model, wbuf, T, needs_hidden)  # warmup
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self._run_verify_into(model, wbuf, T, needs_hidden)
            if pool is None:
                pool = graph.pool()
            ent["graphs"][bs] = graph
        self._verify = {"buf": vbuf, "widths": by_qlen, "qlens": sorted(by_qlen),
                        "qlen": qlen, "graphs": by_qlen[qlen]["graphs"],
                        "bs_list": sorted(bs_list), "needs_hidden": needs_hidden, "num_aux": num_aux}
        logger.info_rank0(f"spec-verify graphs captured; free {mem_GB(get_free_memory(dev))}")

    def _set_verify_width(self, qlen: int) -> None:
        """Repoint every per-width static (attention verify metadata, GDN/CCA recurrent scratch) at the
        buffers captured for `qlen` query rows per sequence. Must run BEFORE the prepare_* calls, which
        read those pointers to build the batch metadata."""
        setter = getattr(self.attn_backend, "set_verify_width", None)
        if setter is not None:
            setter(qlen)
        if self.cca_verify is not None:
            self.cca_verify.set_width(qlen)
        if self.gdn_verify is not None:
            self.gdn_verify.set_width(qlen)

    @staticmethod
    def _run_verify_into(model, vbuf: VerifyCaptureBuffer, T: int, needs_hidden: bool) -> None:
        if needs_hidden:
            logits, last_hidden, aux = model.forward(return_hidden=True)
            vbuf.logits[:T] = logits
            vbuf.last_hidden[:T] = last_hidden
            if aux is not None:
                vbuf.aux_hidden[:, :T] = aux
        else:
            vbuf.logits[:T] = model.forward()

    @property
    def verify_bs_list(self) -> "list[int]":
        """Captured spec-verify batch sizes (empty if verify graphs weren't captured). The scheduler
        reads this to decide whether padding a partial-K step up to uniform num_draft is worthwhile
        (only when the padded bs fits a captured size → the step hits the verify graph)."""
        return self._verify["bs_list"] if self._verify is not None else []

    @property
    def verify_widths(self) -> "list[int]":
        """Captured spec-verify WIDTHS (drafts/seq), ascending; empty if verify graphs weren't
        captured. The scheduler's adaptive-width controller may only ever choose from this set —
        anything else falls off the graph (see can_use_verify_graph)."""
        return [q - 1 for q in self._verify["qlens"]] if self._verify is not None else []

    def can_use_verify_graph(self, batch: Batch) -> bool:
        # capturable iff: graphs exist, every req in the step has the SAME extend_len, that extend_len
        # is one of the CAPTURED qlens (adaptive verify width — the scheduler picks the width and pads
        # to it), and the req count fits a captured bs. A ragged step (near max_tokens / first cold
        # step) falls back to eager. spec_verify is set by the scheduler.
        if self._verify is None or not getattr(batch, "spec_verify", False):
            return False
        if batch.size > self._verify["bs_list"][-1] or batch.size == 0:
            return False
        ql = batch.reqs[0].extend_len
        if ql not in self._verify["widths"]:
            return False
        return all(r.extend_len == ql for r in batch.reqs)

    def pad_verify(self, batch: Batch, bs: "int | None" = None) -> None:
        # bs=None: pad to the smallest captured verify size >= batch.size (the normal DP-local choice).
        # bs=<int>: pad to EXACTLY that captured size — used by the EP spec lockstep so every replica
        # replays the IDENTICAL verify graph (fixed-N MoE all_gather matches; see _spec_ep_loop). The
        # caller guarantees bs is a captured verify size and bs >= batch.size.
        if bs is None:
            bs = next(b for b in self._verify["bs_list"] if b >= batch.size)
        assert bs >= batch.size and bs in self._verify["bs_list"], (bs, batch.size)
        batch.padded_reqs = batch.reqs + [self._verify_dummy(batch)] * (bs - batch.size)

    def _verify_dummy(self, batch: Batch) -> Req:
        # Dummy rows must carry the SAME extend_len as the real ones (the step's chosen width), or the
        # padded batch would stage a different token count than the captured graph expects.
        ql = batch.reqs[0].extend_len if batch.reqs else self._verify["qlen"]
        return Req(
            input_ids=torch.zeros(ql, dtype=torch.int32, device="cpu"),
            table_idx=self.dummy_req.table_idx, cached_len=0, output_len=1, uid=-1,
            sampling_params=None, cache_handle=None,  # type: ignore
        )

    def replay_verify(self, batch: Batch, return_hidden: bool):
        """Replay the captured verify graph for `batch`. The scheduler has already PADDED the batch
        (pad_verify) and computed batch.input_ids / positions / out_loc over padded_reqs (eager);
        copy them into the static buffers, refresh the MLA verify metadata, replay, and slice the
        real-token outputs (the dummy-padded tail rows are discarded)."""
        v = self._verify
        v["replays"] = v.get("replays", 0) + 1
        # Adaptive verify width: the step's width is carried by the reqs themselves (can_use_verify_graph
        # already checked it is uniform AND captured), so pick that width's buffer view + graph and
        # repoint the per-width statics before the prepare_* calls read them.
        ql = batch.reqs[0].extend_len
        w = v["widths"][ql]
        vbuf: VerifyCaptureBuffer = w["buf"]
        vbuf.copy_from(batch)
        self._set_verify_width(ql)
        self.attn_backend.prepare_verify_for_replay(batch)
        if self.cca_verify is not None:
            self.cca_verify.prepare_verify_for_replay(batch)
        if self.gdn_verify is not None:
            self.gdn_verify.prepare_verify_for_replay(batch)
        w["graphs"][batch.padded_size].replay()
        n = batch.size * ql
        logits = vbuf.logits[:n]
        if not return_hidden:
            return logits
        last_hidden = vbuf.last_hidden[:n] if vbuf.last_hidden is not None else None
        aux = vbuf.aux_hidden[:, :n] if vbuf.aux_hidden is not None else None
        return logits, last_hidden, aux

    # ---- FUSED spec-verify graph capture (v2 S4: CCA custom-mask single-forward) -----------------
    def capture_fused_verify_graphs(
        self,
        model: BaseLLMModel,
        fused_qlen: int,
        bs_list: List[int],
    ) -> None:
        """Capture one FUSED-verify graph per bs in `bs_list`. The fused-TiDAR step stages `fused_qlen`
        query tokens/seq (`1+B+B²` flat / `1+B+B·(tp+B)` seg) and runs the paged-extend kernel with a
        dense custom_mask (causal=0). Distinct capture shape from the K+1 verify (`capture_verify_graphs`):
        qlen=fused_qlen, a static max-width mask buffer (HIPAttnBackend.init_fused_verify_capture), and
        a CCA-verify capturer at Q=fused_qlen. Logits-only (TiDAR self-draft reads logits, not hidden).
        Called by the scheduler after the TiDAR proposer is built (it knows B → fused_qlen)."""
        if not bs_list or not hasattr(self.attn_backend, "init_fused_verify_capture"):
            return logger.info_rank0("fused-verify CUDA graph: unsupported backend / disabled")
        dev = self.device
        max_bs = max(bs_list)
        self.attn_backend.init_fused_verify_capture(self._verify_max_seq_len, bs_list, fused_qlen)
        # CCA-hybrid recurrent state through static verify buffers at Q=fused_qlen (parameterised on
        # num_draft → Q = num_draft+1, so pass fused_qlen-1). Same in-place conv/prev scratch trick.
        self.cca_fused_verify = None
        if self._cca_state is not None:
            from minisgl.cca.graph_capture import CCAVerifyGraphCapture

            cs = self._cca_state
            self.cca_fused_verify = CCAVerifyGraphCapture(
                dev, max_bs, fused_qlen - 1,
                cca_layer_ids=range(cs.num_cca_layers),
                conv_dim=cs.conv_states.shape[2], conv_width=cs.conv_states.shape[3],
                hidden=cs.prev_hs.shape[2],
            )
        vbuf = VerifyCaptureBuffer.init(
            max_bs, fused_qlen, self._verify_vocab, None, 0, torch.float32, dev
        )
        graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        # dummy req with extend_len = fused_qlen (cached_len 0, device_len fused_qlen).
        fdummy = Req(
            input_ids=torch.zeros(fused_qlen, dtype=torch.int32, device="cpu"),
            table_idx=self.dummy_req.table_idx, cached_len=0, output_len=1, uid=-1,
            sampling_params=None, cache_handle=None,  # type: ignore
        )
        torch.cuda.synchronize(dev)
        free0 = get_free_memory(dev)
        logger.info_rank0(
            f"Capturing FUSED-verify CUDA graphs (qlen={fused_qlen}) sizes={sorted(bs_list)}; "
            f"free {mem_GB(free0)}"
        )
        pool = None
        for bs in tqdm(sorted(bs_list, reverse=True), desc="Capturing fused-verify graphs",
                       unit="batch", disable=not get_tp_info().is_primary()):
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[fdummy] * bs, phase="decode")
            batch.spec_verify = True
            batch.fused_verify = True
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_fused_verify_for_capture(batch)
            if self.cca_fused_verify is not None:
                self.cca_fused_verify.prepare_verify_for_capture(batch)
            vbuf.set_batch(batch)
            T = vbuf.total(batch)
            with get_global_ctx().forward_batch(batch), torch.inference_mode():
                self._run_verify_into(model, vbuf, T, False)  # warmup
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self._run_verify_into(model, vbuf, T, False)
            if pool is None:
                pool = graph.pool()
            graph_map[bs] = graph
        self._fused_verify = {"buf": vbuf, "graphs": graph_map, "qlen": fused_qlen,
                              "bs_list": sorted(bs_list)}
        logger.info_rank0(f"fused-verify graphs captured; free {mem_GB(get_free_memory(dev))}")

    def can_use_fused_verify(self, batch: Batch) -> bool:
        # capturable iff: fused graphs exist, the batch is a fused-verify step, every req stages exactly
        # fused_qlen query tokens (uniform), and the req count EXACTLY matches a captured bs. Unlike the
        # K+1 verify, the fused scheduler builds input_ids/positions/out_loc/custom_mask over `reqs`
        # (not padded_reqs), so a padded batch would under-fill the static buffers → require an exact bs
        # match and fall back to eager otherwise (still lossless). NOREP (qlen=1+B), partial/finished
        # steps, and bootstrap block_predict never set fused_verify / carry fused_qlen, so they fall back.
        if self._fused_verify is None or not getattr(batch, "fused_verify", False):
            return False
        ql = self._fused_verify["qlen"]
        if batch.size not in self._fused_verify["graphs"]:
            return False
        return all(r.extend_len == ql for r in batch.reqs)

    def replay_fused_verify(self, batch: Batch) -> torch.Tensor:
        """Replay the captured FUSED-verify graph. The scheduler has built input_ids/positions/out_loc
        + the dense custom_mask + cca_metadata (capture_verify_state=True) eagerly over `batch.reqs`;
        copy them into the static buffers, refresh the attn (page_table/cache_seqlens/mask) + CCA-state
        replay metadata, replay, and slice the real-token logits. The mask-build and the post-forward
        `install_verify_state` stay eager (the scheduler reads the static conv/prev scratch via
        `batch.cca_metadata`, which prepare_verify_for_replay repoints below)."""
        v = self._fused_verify
        v["replays"] = v.get("replays", 0) + 1
        if v["replays"] == 1:
            logger.info_rank0(f"fused-verify GRAPH REPLAY engaged (qlen={v['qlen']}, bs={batch.size})")
        vbuf: VerifyCaptureBuffer = v["buf"]
        vbuf.copy_from(batch)
        # attn: page_table/cache_seqlens + copy the dense mask into the static buffer (reads batch's
        # scheduler-built custom_mask, then swaps batch.attn_metadata for the static one).
        self.attn_backend.prepare_fused_verify_for_replay(batch)
        if self.cca_fused_verify is not None:
            self.cca_fused_verify.prepare_verify_for_replay(batch)
        v["graphs"][batch.padded_size].replay()
        n = batch.size * v["qlen"]
        return vbuf.logits[:n]

    # ---- DDTREE draft-TREE spec-verify graph capture (ancestor-mask single-forward, GDN/CCA) -------
    def capture_ddtree_verify_graphs(
        self,
        model: BaseLLMModel,
        tree_qlen: int,
        bs_list: List[int],
        max_ctx: int,
    ) -> None:
        """Capture one DDTree tree-verify graph per bs in `bs_list`. The DDTree step stages a draft TREE
        PADDED to a fixed `tree_qlen = budget+1` query tokens/seq and runs the paged-extend kernel with a
        dense ancestor `custom_mask` (causal=0). Distinct capture shape from the K+1 linear verify:
        qlen=tree_qlen, a static (capped-width) mask buffer (init_ddtree_verify_capture), and a recurrent
        verify capturer at Q=tree_qlen. STATE-NEUTRAL: the recurrent slots are snapshot/restored by the
        scheduler around replay and never installed, so the per-layer scratch is throwaway. Logits-only
        (DDTree reads the per-node argmax to walk the tree). Called by the scheduler after the proposer is
        built (it knows budget → tree_qlen)."""
        if not bs_list or not hasattr(self.attn_backend, "init_ddtree_verify_capture"):
            return logger.info_rank0("ddtree-verify CUDA graph: unsupported backend / disabled")
        # SWA-HYBRID (Laguna): NOT WIRED, and it must decline rather than crash. The static DDTree
        # metadata (`_ddtree_verify_metadata_static`) populates no swa_* fields — unlike the K+1
        # verify capture, which allocates a per-qlen ring block table + out_loc — so the first
        # sliding layer of the capture WARMUP trips `_swa_forward`'s "SWA metadata missing" assert
        # and takes the boot down. That made DDTree unbootable on the ONLY model whose DFlash
        # drafter has a capturable propose, i.e. DDTree could not be exercised at all. Declining
        # here leaves the tree-verify EAGER (the scheduler's own `prepare_metadata` DOES build the
        # SWA fields, so the eager path is complete and lossless — DDTree's verify is a plain
        # ancestor-masked forward), which is slower but runnable and testable.
        if getattr(self.attn_backend, "swa_kv", None) is not None and \
                getattr(self.attn_backend, "swa_window", 0) > 0:
            return logger.warning_rank0(
                "ddtree-verify CUDA graph: SKIPPED on an SWA-hybrid model — the DDTree static "
                "metadata has no sliding-window ring fields (attention/hip.py "
                "_ddtree_verify_metadata_static vs init_verify_capture). The tree-verify runs EAGER."
            )
        dev = self.device
        max_bs = max(bs_list)
        self.attn_backend.init_ddtree_verify_capture(
            self._verify_max_seq_len, bs_list, tree_qlen, max_ctx
        )
        # GDN-hybrid recurrent state through static verify buffers at Q=tree_qlen (param on num_draft →
        # Q=num_draft+1, so pass tree_qlen-1). Same in-place conv/ssm scratch trick as the K+1 verify.
        self.gdn_ddtree_verify = None
        if self._gdn_state is not None:
            from minisgl.gdn.graph_capture import GDNVerifyGraphCapture

            gs = self._gdn_state
            cshape = gs.conv_state.shape
            sshape = gs.ssm_state.shape
            self.gdn_ddtree_verify = GDNVerifyGraphCapture(
                dev, max_bs, tree_qlen - 1,
                gdn_layer_ids=range(gs.num_gdn_layers),
                conv_dim=cshape[2], conv_width=cshape[3],
                num_v_heads=sshape[2], head_v_dim=sshape[3], head_k_dim=sshape[4],
                ssm_dtype=gs.ssm_dtype,
            )
        # CCA-hybrid (ZAYA) analog — TiDAR-DDTree over a CCA backbone. Same throwaway scratch.
        self.cca_ddtree_verify = None
        if self._cca_state is not None:
            from minisgl.cca.graph_capture import CCAVerifyGraphCapture

            cs = self._cca_state
            self.cca_ddtree_verify = CCAVerifyGraphCapture(
                dev, max_bs, tree_qlen - 1,
                cca_layer_ids=range(cs.num_cca_layers),
                conv_dim=cs.conv_states.shape[2], conv_width=cs.conv_states.shape[3],
                hidden=cs.prev_hs.shape[2],
            )
        vbuf = VerifyCaptureBuffer.init(
            max_bs, tree_qlen, self._verify_vocab, None, 0, torch.float32, dev
        )
        graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        ddummy = Req(
            input_ids=torch.zeros(tree_qlen, dtype=torch.int32, device="cpu"),
            table_idx=self.dummy_req.table_idx, cached_len=0, output_len=1, uid=-1,
            sampling_params=None, cache_handle=None,  # type: ignore
        )
        torch.cuda.synchronize(dev)
        free0 = get_free_memory(dev)
        logger.info_rank0(
            f"Capturing DDTREE-verify CUDA graphs (qlen={tree_qlen}, "
            f"mask_ctx={self.attn_backend._dcap_max_kv}) sizes={sorted(bs_list)}; free {mem_GB(free0)}"
        )
        pool = None
        for bs in tqdm(sorted(bs_list, reverse=True), desc="Capturing ddtree-verify graphs",
                       unit="batch", disable=not get_tp_info().is_primary()):
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[ddummy] * bs, phase="decode")
            batch.spec_verify = True
            batch.ddtree_verify = True
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_ddtree_verify_for_capture(batch)
            if self.gdn_ddtree_verify is not None:
                self.gdn_ddtree_verify.prepare_verify_for_capture(batch)
            if self.cca_ddtree_verify is not None:
                self.cca_ddtree_verify.prepare_verify_for_capture(batch)
            vbuf.set_batch(batch)
            T = vbuf.total(batch)
            with get_global_ctx().forward_batch(batch), torch.inference_mode():
                self._run_verify_into(model, vbuf, T, False)  # warmup
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self._run_verify_into(model, vbuf, T, False)
            if pool is None:
                pool = graph.pool()
            graph_map[bs] = graph
        self._ddtree_verify = {"buf": vbuf, "graphs": graph_map, "qlen": tree_qlen,
                               "bs_list": sorted(bs_list)}
        logger.info_rank0(f"ddtree-verify graphs captured; free {mem_GB(get_free_memory(dev))}")

    def can_use_ddtree_verify(self, batch: Batch) -> bool:
        # capturable iff: ddtree graphs exist, the batch is a ddtree-verify step, every req stages exactly
        # tree_qlen query tokens (padded), the req count EXACTLY matches a captured bs, AND every req's
        # context fits the capped static mask width. Beyond the cap (or a non-exact bs / partial step) it
        # falls back to eager — still lossless. Mirrors can_use_fused_verify's exact-bs contract (the
        # scheduler builds input_ids/positions/mask over `reqs`, not padded_reqs).
        if self._ddtree_verify is None or not getattr(batch, "ddtree_verify", False):
            return False
        ql = self._ddtree_verify["qlen"]
        if batch.size not in self._ddtree_verify["graphs"]:
            return False
        max_kv = self.attn_backend._dcap_max_kv
        return all(r.extend_len == ql and r.device_len <= max_kv for r in batch.reqs)

    def replay_ddtree_verify(self, batch: Batch) -> torch.Tensor:
        """Replay the captured DDTree tree-verify graph. The scheduler built input_ids/positions/out_loc
        + the dense ancestor custom_mask + recurrent metadata (capture_verify_state=True) eagerly over
        `batch.reqs`; copy them into the static buffers, refresh attn (page_table/cache_seqlens/mask) +
        recurrent-state replay metadata, replay, and slice the real-token logits. State-neutral: the
        scheduler snapshots the real slots before and restores after (no install)."""
        v = self._ddtree_verify
        v["replays"] = v.get("replays", 0) + 1
        if v["replays"] == 1:
            logger.info_rank0(f"ddtree-verify GRAPH REPLAY engaged (qlen={v['qlen']}, bs={batch.size})")
        vbuf: VerifyCaptureBuffer = v["buf"]
        vbuf.copy_from(batch)
        self.attn_backend.prepare_ddtree_verify_for_replay(batch)
        if self.gdn_ddtree_verify is not None:
            self.gdn_ddtree_verify.prepare_verify_for_replay(batch)
        if self.cca_ddtree_verify is not None:
            self.cca_ddtree_verify.prepare_verify_for_replay(batch)
        v["graphs"][batch.padded_size].replay()
        n = batch.size * v["qlen"]
        return vbuf.logits[:n]

    # ---- BLOCK-DIFFUSION CANVAS graph capture (DiffusionGemma denoising step) ---------------------
    def capture_canvas_graphs(
        self,
        model: BaseLLMModel,
        canvas_len: int,
        bs_list: List[int],
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        """Capture one canvas-step graph per bs in `bs_list`.

        WHY A 256-TOKEN STEP IS WORTH CAPTURING, AND WHAT IT IS WORTH. Every other capture family here
        exists because the forward is TINY (one token per sequence) and therefore host-launch-bound. A
        canvas step is 256 tokens wide, which looks like a prefill — but it issues ~5k dispatches and
        a block runs the IDENTICAL forward k times (k = 12-19 measured) with nothing varying but the
        buffer contents. That is the textbook capture case.

        IT IS NOT A SPEEDUP HERE, and saying so in place is the point of this paragraph. Measured on
        the served checkpoint (bs=1, TP=2, marginal per-step, same harness on both legs): capture
        takes `fwd_issue` — the host launch loop — from 30.8 ms to 0.8 ms, a 97% reduction, and moves
        the STEP by -0.3% (181.6 -> 181.1 ms). Those 30.8 ms were entirely overlapped with GPU work;
        an independent rocprofv3 pass measures gfx activity at 100% median and found that removing
        38% of all dispatches bought 3.2%. There is no launch gap to reclaim. The step is
        GEMM-efficiency bound inside the backbone (127.7 ms of a 186 ms step).

        So this is carried for correctness — it is proven bit-identical, and eager-only is not "done"
        in this repo — and because its 30 ms is currently hidden behind the backbone: the same graph
        removes the same 30 ms from whatever the residual becomes, so its share grows as the rest of
        the stack lands. Do not quote it as a tok/s win.

        EXACT bs MATCH, no padding — deliberately, and unlike the decode/K+1-verify families. Padding
        a bs=3 canvas step up to a captured bs=4 would push a whole extra 256-token canvas through 30
        layers of attention and MoE, which is ~33% more compute to avoid ~1 ms of launch overhead.
        Anything not captured falls back to the eager forward, which is lossless (see
        `can_use_canvas_graph`). That is the same contract `can_use_fused_verify` uses, for the same
        reason.

        Logits-free: the captured region ends at the backbone's final norm and the LM head runs
        eagerly (`forward_canvas_hidden` / `canvas_logits`). See CanvasCaptureBuffer."""
        if not bs_list or not hasattr(self.attn_backend, "init_canvas_capture"):
            return logger.info_rank0("canvas CUDA graph: unsupported backend / disabled")
        dev = self.device
        max_bs = max(bs_list)
        self.attn_backend.init_canvas_capture(self._verify_max_seq_len, bs_list, canvas_len)
        cbuf = CanvasCaptureBuffer.init(max_bs, canvas_len, hidden_size, dtype, dev)
        # A canvas dummy owns canvas_len query tokens over its own (dummy-page) slots, cached_len 0 —
        # the same construction the verify capture uses, at the canvas width.
        cdummy = Req(
            input_ids=torch.zeros(canvas_len, dtype=torch.int32, device="cpu"),
            table_idx=self.dummy_req.table_idx, cached_len=0, output_len=1, uid=-1,
            sampling_params=None, cache_handle=None,  # type: ignore
        )
        torch.cuda.synchronize(dev)
        free0 = get_free_memory(dev)
        logger.info_rank0(
            f"Capturing CANVAS CUDA graphs (canvas={canvas_len}) sizes={sorted(bs_list)}; "
            f"free {mem_GB(free0)}"
        )
        graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        pool = None
        for bs in tqdm(sorted(bs_list, reverse=True), desc="Capturing canvas graphs",
                       unit="batch", disable=not get_tp_info().is_primary()):
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[cdummy] * bs, phase="decode")
            batch.canvas = True  # -> causal=0 on both geometries + the [window | canvas] ring rows
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_canvas_for_capture(batch)
            cbuf.set_batch(batch)
            T = cbuf.total(batch)
            with get_global_ctx().forward_batch(batch), torch.inference_mode():
                cbuf.hidden[:T] = model.forward_canvas_hidden(
                    cbuf.input_ids[:T], cbuf.self_cond[:T]
                )  # warmup
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    cbuf.hidden[:T] = model.forward_canvas_hidden(
                        cbuf.input_ids[:T], cbuf.self_cond[:T]
                    )
            if pool is None:
                pool = graph.pool()
            graph_map[bs] = graph
        self._canvas = {"buf": cbuf, "graphs": graph_map, "qlen": canvas_len,
                        "bs_list": sorted(bs_list), "model": model, "replays": 0,
                        # bs -> how many eager-vs-replay comparisons that size has had (cap 2 each).
                        "checked": {}}
        logger.info_rank0(f"canvas graphs captured; free {mem_GB(get_free_memory(dev))}")

    def can_use_canvas_graph(self, batch: Batch) -> bool:
        # capturable iff: canvas graphs exist, the batch IS a canvas step, the req count EXACTLY
        # matches a captured bs (no padding — see capture_canvas_graphs), and every req stages exactly
        # canvas_len query tokens. A block whose extent has been truncated, or a bs above the captured
        # set, falls back to the eager forward and is still lossless.
        if self._canvas is None or not getattr(batch, "canvas", False):
            return False
        if batch.size not in self._canvas["graphs"]:
            return False
        return all(r.extend_len == self._canvas["qlen"] for r in batch.reqs)

    @staticmethod
    def _canvas_metadata_diff(a, b, bs: int) -> None:
        """Log every field on which the SCHEDULER-built canvas metadata and the STATIC (graph)
        canvas metadata disagree. Host-side, no GPU work beyond the compares themselves; runs only
        on the two gated check replays per batch size."""
        if a is None or b is None:
            return logger.info_rank0("[canvas-graph] metadata diff: one side is None")
        names = [n for n in dir(a) if not n.startswith("_") and not callable(getattr(a, n, None))]
        for n in sorted(names):
            x, y = getattr(a, n, None), getattr(b, n, None)
            if x is None and y is None:
                continue
            if (x is None) != (y is None):
                logger.info_rank0(f"[canvas-graph] METADATA DIFF {n}: sched={x!r:.60} static={y!r:.60}")
                continue
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                # Compare the LIVE extent only: the static buffers are padded to their captured
                # width and the scheduler's are cut to the batch, so a shape difference alone is not
                # a divergence — a difference inside the read extent is.
                nmin = min(x.shape[0], y.shape[0])
                same_tail = (x.shape[1:] == y.shape[1:])
                if same_tail and torch.equal(x[:nmin].cpu(), y[:nmin].cpu()):
                    if x.shape != y.shape:
                        logger.info_rank0(
                            f"[canvas-graph] metadata {n}: values agree over [:{nmin}], shapes "
                            f"differ sched={tuple(x.shape)} static={tuple(y.shape)}")
                    continue
                if not same_tail:
                    w = min(x.shape[-1], y.shape[-1]) if x.dim() > 1 and y.dim() > 1 else 0
                    if w and torch.equal(x[:nmin, :w].cpu(), y[:nmin, :w].cpu()):
                        logger.info_rank0(
                            f"[canvas-graph] metadata {n}: values agree over [:{nmin},:{w}], row "
                            f"width differs sched={tuple(x.shape)} static={tuple(y.shape)}")
                        continue
                logger.info_rank0(
                    f"[canvas-graph] METADATA DIFF {n}: sched{tuple(x.shape)}={x.flatten()[:12].tolist()} "
                    f"static{tuple(y.shape)}={y.flatten()[:12].tolist()}")
            elif x != y:
                logger.info_rank0(f"[canvas-graph] METADATA DIFF {n}: sched={x!r} static={y!r}")

    def replay_canvas(
        self, batch: Batch, canvas_ids: torch.Tensor, self_conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Replay the captured canvas step and return the backbone hidden state `[bs*L, hidden]`.

        The first two replays AT EACH CAPTURED BATCH SIZE also run the eager forward and compare,
        because "the graph engaged" and "the graph computes the right thing" are different claims and
        only the second one matters. A canvas graph that addresses the wrong ring slots, or that lost
        `bidirectional`, produces fluent text and a plausible tok/s — there is nothing downstream that
        would notice.

        PER BATCH SIZE, not per serve, and that distinction is the whole point: every captured bs has
        its OWN cu_seqlens_q, its own `[bs, W+256]` ring block table and its own `bs*256` store-slot
        vector, so a bug in the per-bs row build is invisible to a check that only ever ran at bs=1.
        The first measurement of this did exactly that — it verified bs=1 four times and bs=2/3/4 not
        at all, because a concurrent serve's first two canvas steps happen while the other requests
        are still prefilling. Two replays each rather than one because the first step of a block
        carries a ZERO self-conditioning signal and would not exercise that input.

        The eager forward re-stores the same K/V into the same slots from the same inputs, so running
        it first is idempotent; it costs three extra forwards per captured size, twice.

        THE REFERENCE MUST BE IN THE GRAPH'S OWN COLLECTIVE REGIME, and getting that wrong is what
        made this gate report `max|delta|=1.575e+01` with nothing to attribute it to. `tp_overlap` is
        capture-TRANSPARENT: under capture every all_reduce is inline and `Gemma4DecoderLayer.forward`
        takes its no-row-split branch, while the eager path puts collectives on a side stream and (at
        the default `MINISGL_TP_AR_CHUNKS=2`) splits the FFN into two 128-row chunks. So a bare
        "graph vs eager" varies TWO things at once — the graph mechanism AND the program — and cannot
        say which one it caught. The reference below is therefore taken inside
        `inline_collectives()`, which pins the eager forward to the same program the capture recorded.
        The regime difference is not swept under the rug: it is measured separately, on the same
        inputs, and reported as its own number.

        AND IT RAISES. This check printed `*** NOT bit-identical ***` and served on, which is how a
        real divergence rode in the doc's §D6.1 "bit-identical" claim for a full measurement pass. A
        gate whose failure mode is a log line is not a gate."""
        v = self._canvas
        v["replays"] += 1
        checked = v["checked"]
        bs = batch.size
        check = checked.get(bs, 0) < 2
        if check:
            checked[bs] = checked.get(bs, 0) + 1
        cbuf: CanvasCaptureBuffer = v["buf"]
        ref = ref_again = ref_serve = None
        if check:
            # Eager references FIRST, off the scheduler-built (non-static) metadata, before
            # prepare_canvas_for_replay swaps in the static one.
            self.attn_backend.prepare_metadata(batch)
            model = v["model"]
            with get_global_ctx().forward_batch(batch), torch.inference_mode():
                with inline_collectives():
                    # THE reference: same inputs, same program as the captured region.
                    ref = model.forward_canvas_hidden(canvas_ids, self_conditioning).clone()
                    # Determinism control. If THIS is non-zero the eager forward is not a fixed
                    # function of its inputs (an atomic reduction, a race), and no graph-vs-eager
                    # number below means anything until that is fixed.
                    ref_again = model.forward_canvas_hidden(canvas_ids, self_conditioning).clone()
                # The eager FALLBACK path exactly as an uncaptured bs would run it: side-stream
                # collectives + the row split. Reported, not gated — see below.
                ref_serve = model.forward_canvas_hidden(canvas_ids, self_conditioning).clone()
        md_sched = getattr(batch, "attn_metadata", None) if check else None
        cbuf.copy_from(batch, canvas_ids, self_conditioning)
        self.attn_backend.prepare_canvas_for_replay(batch)
        ref_static = None
        if check:
            # FREE, host-side: the two metadata objects field by field. `prepare_metadata` and
            # `prepare_canvas_for_replay` are two INDEPENDENT implementations of the same canvas
            # geometry (`_build_swa_canvas_metadata` vs `_fill_swa_multiquery_static`), and a graph
            # can only be as right as the second one. Any line printed here is a divergence between
            # what the scheduler believes and what the graph reads.
            self._canvas_metadata_diff(md_sched, getattr(batch, "attn_metadata", None), bs)
            # THE LOCALISING PROBE. Same eager forward, same inputs, same collective regime — but
            # under the STATIC metadata `prepare_canvas_for_replay` just installed, which is what the
            # graph reads. Two deterministic forwards differing only in metadata:
            #   ref_static != ref  -> prepare_canvas_for_replay builds a DIFFERENT geometry than the
            #                         scheduler does, and the graph is faithfully replaying it.
            #   ref_static == ref  -> the metadata agrees and the divergence is in the captured
            #                         region itself (a pointer that moved, a pool alias).
            # It runs on the STATIC buffers the graph will read, so it must come AFTER copy_from and
            # BEFORE the replay; it re-stores the same K/V to the same slots, so it is idempotent.
            with get_global_ctx().forward_batch(batch), torch.inference_mode():
                with inline_collectives():
                    ref_static = v["model"].forward_canvas_hidden(
                        cbuf.input_ids[: batch.size * v["qlen"]],
                        cbuf.self_cond[: batch.size * v["qlen"]],
                    ).clone()
        v["graphs"][batch.size].replay()
        T = batch.size * v["qlen"]
        out = cbuf.hidden[:T]
        if check:
            first = out.clone()
            v["graphs"][batch.size].replay()  # replay determinism control
            second = cbuf.hidden[:T].clone()

            def _d(a, b):
                return (a.float() - b.float()).abs().max().item()

            d_graph = _d(first, ref)              # THE claim
            d_eager_self = _d(ref, ref_again)     # eager determinism
            d_graph_self = _d(first, second)      # replay determinism
            d_regime = _d(ref_serve, ref)         # cost of the eager path's overlap + row split
            d_meta = _d(ref_static, ref)          # STATIC metadata vs scheduler metadata, both eager
            d_vs_static = _d(first, ref_static)   # does the graph reproduce eager-under-ITS metadata
            scale = ref.float().abs().max().item()
            logger.info_rank0(
                f"[canvas-graph] REPLAY #{v['replays']} engaged, bs={bs} check {checked[bs]}/2 "
                f"(qlen={v['qlen']}, T={T}, ref max|x|={scale:.4g} over {tuple(out.shape)}): "
                f"graph vs eager[matched regime] max|delta|={d_graph:.3e} "
                + ("BIT-IDENTICAL" if d_graph == 0.0 else "*** NOT bit-identical ***")
                + f" | eager self-consistency={d_eager_self:.3e}"
                f" | replay self-consistency={d_graph_self:.3e}"
                f" | eager-fallback regime (side-stream AR + {tp_overlap_chunks()} row chunks) "
                f"vs matched={d_regime:.3e}"
                f" | LOCALISE: eager[static replay metadata] vs eager[scheduler metadata]"
                f"={d_meta:.3e}; graph vs eager[static replay metadata]={d_vs_static:.3e}"
            )
            if d_graph != 0.0 or d_graph_self != 0.0:
                raise RuntimeError(
                    "[canvas-graph] captured canvas step is NOT bit-identical to the eager forward "
                    f"it was captured from (bs={bs}, T={T}): graph vs matched-regime eager "
                    f"max|delta|={d_graph:.3e}, replay-vs-replay={d_graph_self:.3e}, "
                    f"eager-vs-eager={d_eager_self:.3e}, ref max|x|={scale:.4g}. "
                    "A replay that does not reproduce its own capture addresses different memory or "
                    "runs a different kernel than was recorded; every number taken on this graph is "
                    "void. Read GraphRunner.replay_canvas before relaxing this."
                )
        return out

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        if self._verify is not None:
            del self._verify
        if self._fused_verify is not None:
            del self._fused_verify
        if self._ddtree_verify is not None:
            del self._ddtree_verify
        if self._canvas is not None:
            del self._canvas
        gc.collect()
