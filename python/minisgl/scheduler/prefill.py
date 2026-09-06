from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import align_down, init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager
    # HARD granularity a NON-FINAL prefill chunk's end must land on. 1 for every model that has no
    # such constraint; `indexer_compress_ratio` (4) when qwen4_exp's QSA selection is live.
    #
    # WHY IT EXISTS, and what it is NOT. `QSAPlan.build` raises on a chunk whose `cached_len` is not
    # a multiple of r, because the compressed index key of the group straddling that boundary would
    # have to be averaged from members this forward does not hold. That refusal is what capped the
    # shipped arm at CONC=1 (docs/measurements/QSA_INDEXER.md §4c), where it was attributed to PREFIX
    # REUSE — which is wrong, and worth stating plainly because the wrong fix (rounding the radix
    # match down) is a no-op here: qwen4_exp forces the NAIVE prefix cache (engine/config.py
    # `resolve_prefix_cache`: the PLE recurrent state is not covered by the snapshot store), so
    # `handle.cached_len` is ALWAYS 0. And a radix match cannot produce the failure anyway: every
    # radix `cached_len` is `align_down(..., page_size)` and QSA already requires
    # `page_size % r == 0`.
    #
    # The real source is chunk PACKING. `token_budget` is per STEP, not per request, and a request's
    # FINAL chunk is `remain_len` — an arbitrary number — so the leftover budget handed to the next
    # request in the same batch is arbitrary too, and becomes that request's first chunk. The two
    # recorded reproductions are exactly this arithmetic, not a prefix hit:
    #     16382 = 15*1024 + 1022  ->  leftover 2  ->  the second prompt's chunk is 2, so its NEXT
    #                                 chunk starts at cached_len=2   (48L TP=2, r2b.log)
    #      4087 =  3*1024 + 1015  ->  leftover 9  ->  cached_len=9   (4L TP=1,  r3.log)
    # Rounding the chunk END down to a multiple of r closes it at the source and costs at most r-1
    # tokens of a step's budget.
    chunk_gran: int = 1

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> "Req | None":
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        # Recurrent radix (GDN/CCA): keep every prefill SEGMENT boundary page-aligned so the linear-
        # attention recurrent-state slot can be snapshotted at that exact boundary (the losslessness
        # precondition — a snapshot at an unaligned length attached to the align_down radix node would
        # double-count the sub-page tail). Round this segment down to a page multiple, deferring the
        # <page_size remainder to the next (final) chunk. Guarded so it only fires when there IS an
        # aligned body AND tokens remain after it; the sub-page tail itself is never split again. Inert
        # for dense/MLA caches (is_recurrent_radix False) — those keep the historical single-forward.
        if self.cache_manager.is_recurrent_radix:
            aligned_end = align_down(cached_len + chunk_size, self.cache_manager.page_size)
            if cached_len < aligned_end < pending_req.input_len:
                chunk_size = aligned_end - cached_len
        # QSA group alignment (see `chunk_gran`). Applies to a NON-FINAL chunk only: the final chunk
        # ends at the prompt's own length, which nothing requires to be a multiple of anything (the
        # decode path sources its group members from the per-request raw-key RING, not from the
        # chunk). Returning None when no aligned chunk fits is deliberate and is the half the
        # existing recurrent-radix alignment above is missing: leaving a sub-granularity chunk in
        # place is precisely what produced `cached_len=2`.
        if self.chunk_gran > 1 and cached_len + chunk_size < pending_req.input_len:
            aligned_end = align_down(cached_len + chunk_size, self.chunk_gran)
            if aligned_end <= cached_len:
                return None
            chunk_size = aligned_end - cached_len
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            req = self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )
            if req is None:
                # No legally-aligned chunk fits this step's remaining budget (chunk_gran). Give the
                # resources back — `_try_allocate_one` already locked the handle and took a table
                # slot — and let the request be admitted on the next step with a full budget.
                self.cache_manager.unlock(cache_handle)
                self.table_manager.free(table_idx)
                return None
            return req

        return None


@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)
    # Prefix-cache accounting: hit_tokens = prefix tokens reused from the radix cache,
    # prompt_tokens = total prompt tokens seen. Counted ONCE per request, at first
    # admission (not per chunk), so hit_ratio = hit/prompt is a true prefix reuse rate.
    prefix_hit_tokens: int = 0
    prefix_prompt_tokens: int = 0
    # See PrefillAdder.chunk_gran. Set once by the scheduler from the model config.
    chunk_gran: int = 1

    def add_one_req(self, req: UserMsg) -> None:
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
            chunk_gran=self.chunk_gran,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            # "new" == not resuming an in-flight chunk: count prefix reuse exactly once.
            is_new_req = pending_req.chunked_req is None
            if req := adder.try_add_one(pending_req):
                if is_new_req:
                    self.prefix_hit_tokens += req.cached_len
                    self.prefix_prompt_tokens += pending_req.input_len
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
