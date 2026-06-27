from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import torch
    from minisgl.core import Req

    from .config import SpecConfig

__all__ = ["Proposer", "ProposeContext"]


class ProposeContext:
    """Per-step inputs a proposer may consume. n-gram needs none of it; draft-head proposers
    (MTP / EAGLE3 / DFlash) read the target's hidden states captured during the PREVIOUS verify.

    Fields are populated by the engine only when some proposer declares it needs them
    (`needs_last_hidden` / `capture_layer_ids`), so a pure-n-gram serve pays nothing.
    """

    __slots__ = ("last_hidden", "aux_hidden", "device")

    def __init__(
        self,
        device: "torch.device",
        last_hidden: "Optional[dict[int, torch.Tensor]]" = None,
        aux_hidden: "Optional[dict[int, torch.Tensor]]" = None,
    ) -> None:
        self.device = device
        # uid -> the target's final hidden state [hidden] at that req's last confirmed token
        # (post-final-norm, pre-lm_head). Consumed by MTP/EAGLE to seed the draft.
        self.last_hidden = last_hidden or {}
        # uid -> stacked aux hidden states [num_capture_layers, hidden] for DFlash/EAGLE3.
        self.aux_hidden = aux_hidden or {}


class Proposer(ABC):
    """Produces up to ``num_draft`` draft tokens per request for the spec-decode verify step.

    The verify / accept / commit / KV-rollback machinery in the scheduler is proposer-agnostic;
    only this `propose` (and the optional `on_accept` rollback of draft-owned state) varies. The
    four families and what they need from the engine:

      | proposer | model            | needs_last_hidden | capture_layer_ids | owns                |
      |----------|------------------|-------------------|-------------------|---------------------|
      | ngram    | none             | no                | None              | nothing             |
      | MTP      | target's head    | yes               | None              | head + 1-layer KV   |
      | DFlash   | separate ckpt    | (via aux)         | [N target layers] | trunk + draft KV    |
      | EAGLE3   | separate ckpt    | (via aux)         | [3 target layers] | trunk + draft KV    |

    See SPEC_DECODE.md for the per-family forward and the target-exposure contract.
    """

    # The engine captures the target's final hidden state during verify iff any proposer sets this.
    needs_last_hidden: bool = False
    # Target decoder-layer ids whose hidden states must be captured (DFlash/EAGLE3); None = no aux.
    capture_layer_ids: Optional[List[int]] = None

    @abstractmethod
    def propose(self, reqs: List["Req"], num_draft: int, ctx: ProposeContext) -> List[List[int]]:
        """Return up to ``num_draft`` draft token ids per req (empty list ⇒ plain decode step)."""

    def on_accept(self, reqs: List["Req"], num_accepted: List[int]) -> None:
        """Roll back any draft-owned state (draft KV / recurrent state) to the accepted prefix.
        Default no-op: n-gram owns no state. MTP/DFlash/EAGLE override to truncate their draft KV."""

    def free(self, uid: int) -> None:
        """Release any per-request draft-owned state (draft KV) for a finished/aborted request.
        Default no-op (n-gram); MTP/DFlash/EAGLE override to drop their persistent per-uid cache."""


def make_proposer(spec_config: "SpecConfig", engine=None) -> Proposer:
    """Construct the proposer for a spec config. `engine` is passed to model-based proposers
    (MTP/DFlash/EAGLE) for weight + hidden-state access; n-gram ignores it."""
    import os

    from .proposer import NgramProposer, _CaptureProbeProposer

    if spec_config.algorithm == "ngram":
        # Diagnostic seam (MINISGL_SPEC_CAPTURE_PROBE=<comma-separated layer ids>): an n-gram
        # proposer that also exercises the target hidden-state capture path and asserts shapes.
        # Inert unless the env is set — production n-gram serve is the plain NgramProposer below.
        probe = os.environ.get("MINISGL_SPEC_CAPTURE_PROBE")
        if probe:
            ids = [int(x) for x in probe.split(",") if x.strip() != ""]
            return _CaptureProbeProposer(
                num_draft=spec_config.num_draft,
                ngram_max=spec_config.ngram_max,
                ngram_min=spec_config.ngram_min,
                capture_layer_ids=ids,
            )
        return NgramProposer(
            num_draft=spec_config.num_draft,
            ngram_max=spec_config.ngram_max,
            ngram_min=spec_config.ngram_min,
        )
    if spec_config.algorithm == "mtp":
        from .mtp import MTPProposer

        if engine is None:
            raise ValueError("the MTP proposer needs the engine (model + device); none was passed")
        return MTPProposer(engine=engine, num_draft=spec_config.num_draft)
    raise ValueError(f"no proposer for spec algorithm {spec_config.algorithm!r}")
