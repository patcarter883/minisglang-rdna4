from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional

from .base import Proposer, ProposeContext

if TYPE_CHECKING:
    from minisgl.core import Req


__all__ = ["TiDARProposer"]


class TiDARProposer(Proposer):
    """TiDAR block-diffusion draft proposer — SELF-DRAFT on the target (no separate checkpoint).

    Unlike DFlash/EAGLE (which run a separate draft model in ``propose``), TiDAR drafts a whole block
    with ONE forward of the TARGET over ``[confirmed | mask×B]`` (block_predict): a causal target
    forward whose B mask positions argmax to B draft tokens. That forward needs the scheduler's batch
    machinery (token pool, paged-KV allocate, CCA metadata, attn metadata), and it must be
    STATE-NEUTRAL — the real verify forward re-runs with the accepted drafts. So the block_predict
    body lives in the scheduler (``Scheduler._tidar_block_predict``) where that machinery is; this
    proposer holds the config (block size B, mask id) and forwards ``propose`` to the bound callback.

    The proposer owns NO persistent state (the target's own KV is the context), so ``on_accept`` /
    ``free`` are no-ops and verification stays the existing linear ``verify_greedy`` (β=1 TiDAR
    acceptance). CCA verify-state losslessness is handled by the scheduler's capture+install (B.0).

    Config is read from ``tidar_config.json`` (mask_token_id, block_size) in the draft-model path if
    given, else the served model dir. Env knobs (GPU iteration): MINISGL_TIDAR_BLOCK caps B;
    MINISGL_TIDAR_MASK_ID overrides the mask token id.
    """

    needs_last_hidden = False
    capture_layer_ids: Optional[List[int]] = None

    def __init__(self, engine, num_draft: int, draft_model_path: Optional[str] = None) -> None:
        self._engine = engine
        self._num_draft = num_draft
        # tidar_config.json lives with the served model (self-draft) unless a path is given.
        cfg_dir = draft_model_path or getattr(engine, "model_path", None)
        cfg = self._load_tidar_config(cfg_dir)
        self.block_size = int(os.environ.get("MINISGL_TIDAR_BLOCK") or cfg.get("block_size") or 4)
        assert self.block_size >= 2, f"TiDAR block_size must be >= 2, got {self.block_size}"
        mask_id = os.environ.get("MINISGL_TIDAR_MASK_ID")
        self.mask_token_id = int(mask_id) if mask_id else int(cfg["mask_token_id"])
        # Bound by the scheduler right after construction (owns the batch machinery for block_predict).
        self._block_predict: Optional[Callable[[List["Req"], int, int], List[List[int]]]] = None
        self._dbg = os.environ.get("MINISGL_SPEC_DEBUG") in ("2", "3")

    @staticmethod
    def _load_tidar_config(cfg_dir: Optional[str]) -> dict:
        if cfg_dir:
            p = Path(cfg_dir) / "tidar_config.json"
            if p.exists():
                return json.loads(p.read_text())
        # Fall back to env-only (mask id + block via MINISGL_TIDAR_*); require the mask id then.
        env_mask = os.environ.get("MINISGL_TIDAR_MASK_ID")
        if env_mask:
            return {"mask_token_id": int(env_mask), "block_size": int(os.environ.get("MINISGL_TIDAR_BLOCK") or 4)}
        raise FileNotFoundError(
            f"TiDAR: no tidar_config.json in {cfg_dir!r} and MINISGL_TIDAR_MASK_ID unset — "
            "point --spec-draft-model-path at the served TiDAR model dir or set MINISGL_TIDAR_MASK_ID."
        )

    def bind_block_predict(
        self, fn: Callable[[List["Req"], int, int], List[List[int]]]
    ) -> None:
        """Scheduler injects its block_predict (self-draft target forward) after construction."""
        self._block_predict = fn

    def propose(self, reqs: List["Req"], num_draft: int, ctx: ProposeContext) -> List[List[int]]:
        assert self._block_predict is not None, (
            "TiDARProposer.block_predict not bound — the scheduler must call bind_block_predict()"
        )
        # Draft a block of K = min(num_draft, B) tokens via one target forward over [confirmed|mask×K].
        k = min(num_draft, self.block_size)
        if k <= 0:
            return [[] for _ in reqs]
        drafts = self._block_predict(reqs, k, self.mask_token_id)
        if self._dbg:
            for req, d in zip(reqs, drafts):
                print(f"[tidar-dbg] uid={req.uid} c0={req.cached_len} K={k} draft={d}", flush=True)
        return drafts

    def on_accept(self, reqs: List["Req"], num_accepted: List[int]) -> None:
        # Self-draft on the target: no draft-owned state. CCA/GDN backbone-state rollback is the
        # scheduler's verify capture+install, not the proposer's.
        return

    def free(self, uid: int) -> None:
        return
