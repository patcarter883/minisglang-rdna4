from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
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


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


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
        cam: object | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
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
        if cam is not None and getattr(cam, "enabled", False) and self.max_graph_bs > 0:
            from minisgl.cam.graph_capture import CAMGraphCapture

            inner = getattr(model, "model", model)
            self.cam_capture = CAMGraphCapture(cam, inner, device, self.max_graph_bs)
        # v2: stashed for the spec-VERIFY capturer (built in capture_verify_graphs, needs num_draft).
        self._cca_state = cca_state
        self.cca_verify = None
        # v2 S4: the FUSED-verify capturer + its CCA-state static buffers (built in
        # capture_fused_verify_graphs, needs fused_qlen — a distinct Q from the K+1 verify above).
        self.cca_fused_verify = None
        self._fused_verify = None
        # Spec-decode verify graphs are captured LATER (capture_verify_graphs), after the scheduler
        # builds the proposer + programs the target's aux-capture layers — None until then.
        self._verify = None
        self._verify_max_seq_len = max_seq_len
        self._verify_vocab = vocab_size
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
    ) -> None:
        """Capture one verify graph per bs in `bs_list`. Called by the scheduler AFTER the proposer is
        built and aux-capture layers are programmed, so the captured forward stashes the aux/hidden the
        draft head consumes. Each graph runs model.forward over bs*(num_draft+1) staged tokens; the
        MLA backend reads precomputed static verify indices (see MLABackend.init_verify_capture)."""
        if not bs_list or not hasattr(self.attn_backend, "init_verify_capture"):
            return logger.info_rank0("spec-verify CUDA graph: unsupported backend / disabled")
        qlen = num_draft + 1
        dev = self.device
        max_bs = max(bs_list)
        self.attn_backend.init_verify_capture(self._verify_max_seq_len, bs_list, num_draft)
        # v2 S3: CCA-hybrid recurrent state through static verify buffers (per-CCA-layer conv/prev
        # scratch that the captured verify forward writes in place; see CCAVerifyGraphCapture).
        self.cca_verify = None
        if self._cca_state is not None:
            from minisgl.cca.graph_capture import CCAVerifyGraphCapture

            cs = self._cca_state
            self.cca_verify = CCAVerifyGraphCapture(
                dev, max_bs, num_draft,
                cca_layer_ids=range(cs.num_cca_layers),
                conv_dim=cs.conv_states.shape[2], conv_width=cs.conv_states.shape[3],
                hidden=cs.prev_hs.shape[2],
            )
        vbuf = VerifyCaptureBuffer.init(
            max_bs, qlen, self._verify_vocab,
            hidden_size if needs_hidden else None,
            num_aux, dtype, dev,
        )
        graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        # a dedicated dummy req with extend_len = K+1 (cached_len 0, device_len K+1).
        vdummy = Req(
            input_ids=torch.zeros(qlen, dtype=torch.int32, device="cpu"),
            table_idx=self.dummy_req.table_idx, cached_len=0, output_len=1, uid=-1,
            sampling_params=None, cache_handle=None,  # type: ignore
        )
        torch.cuda.synchronize(dev)
        free0 = get_free_memory(dev)
        logger.info_rank0(
            f"Capturing spec-verify CUDA graphs (qlen={qlen}, hidden={needs_hidden}, aux={num_aux}) "
            f"sizes={sorted(bs_list)}; free {mem_GB(free0)}"
        )
        pool = None
        for bs in tqdm(sorted(bs_list, reverse=True), desc="Capturing verify graphs",
                       unit="batch", disable=not get_tp_info().is_primary()):
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[vdummy] * bs, phase="decode")
            batch.spec_verify = True
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_verify_for_capture(batch)
            if self.cca_verify is not None:
                self.cca_verify.prepare_verify_for_capture(batch)
            vbuf.set_batch(batch)
            T = vbuf.total(batch)
            with get_global_ctx().forward_batch(batch), torch.inference_mode():
                self._run_verify_into(model, vbuf, T, needs_hidden)  # warmup
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self._run_verify_into(model, vbuf, T, needs_hidden)
            if pool is None:
                pool = graph.pool()
            graph_map[bs] = graph
        self._verify = {"buf": vbuf, "graphs": graph_map, "qlen": qlen,
                        "bs_list": sorted(bs_list), "needs_hidden": needs_hidden, "num_aux": num_aux}
        logger.info_rank0(f"spec-verify graphs captured; free {mem_GB(get_free_memory(dev))}")

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

    def can_use_verify_graph(self, batch: Batch) -> bool:
        # capturable iff: graphs exist, every req has exactly num_draft drafts (uniform qlen), and the
        # req count fits a captured bs. Partial-K steps (near max_tokens / first cold step) fall back
        # to eager. spec_verify is set by the scheduler.
        if self._verify is None or not getattr(batch, "spec_verify", False):
            return False
        ql = self._verify["qlen"]
        if batch.size > self._verify["bs_list"][-1]:
            return False
        return all(r.extend_len == ql for r in batch.reqs)

    def pad_verify(self, batch: Batch) -> None:
        bs = next(b for b in self._verify["bs_list"] if b >= batch.size)
        batch.padded_reqs = batch.reqs + [self._verify_dummy(batch)] * (bs - batch.size)

    def _verify_dummy(self, batch: Batch) -> Req:
        ql = self._verify["qlen"]
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
        vbuf: VerifyCaptureBuffer = v["buf"]
        vbuf.copy_from(batch)
        self.attn_backend.prepare_verify_for_replay(batch)
        if self.cca_verify is not None:
            self.cca_verify.prepare_verify_for_replay(batch)
        v["graphs"][batch.padded_size].replay()
        n = batch.size * v["qlen"]
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

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map
        if self._verify is not None:
            del self._verify
        if self._fused_verify is not None:
            del self._fused_verify
        gc.collect()
