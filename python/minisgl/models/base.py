from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    """Base for every *ForCausalLM. ``forward()`` returns lm_head logits for the active batch.

    Draft-head speculative decoding (MTP / EAGLE3 / DFlash) needs the *target's* hidden states from
    the verify forward, so the model exposes two optional capture seams. Both are OFF by default —
    a pure n-gram serve (or normal decode) pays nothing, because ``forward()`` keeps returning just
    logits and the inner ``model.forward`` skips the aux bookkeeping unless capture is programmed.

      * ``set_capture_layers(ids)`` — program a set of decoder-layer ids whose output hidden state
        (the residual stream AFTER that layer, what feeds the next layer) is stashed during the next
        forward. ``ids=None`` (or ``[]``) disables aux capture. Wired at engine/proposer init from
        the proposer's ``capture_layer_ids``.
      * ``forward(return_hidden=True)`` — return ``(logits, last_hidden, aux_hidden)`` instead of
        just ``logits``:
          - ``last_hidden`` : [num_tokens, hidden] post-final-norm, pre-lm_head (seeds MTP/EAGLE).
          - ``aux_hidden``  : [num_capture_layers, num_tokens, hidden] stacked in the order of the
            ``set_capture_layers`` id list, or ``None`` if no capture layers are programmed.
    """

    # decoder-layer ids whose output hidden is captured during forward (None/empty = no aux capture).
    _capture_layer_ids: Optional[List[int]] = None

    def set_capture_layers(self, ids: Optional[List[int]]) -> None:
        """Program which decoder layers stash their output hidden during the next forward.

        The concrete model propagates the set to its inner ``*Model`` (which owns the layer loop).
        Idempotent / cheap; pass ``None`` or ``[]`` to disable. Default impl raises so a model that
        is asked for aux capture but hasn't implemented it fails loudly rather than silently."""
        ids = list(ids) if ids else None
        self._capture_layer_ids = ids
        inner = getattr(self, "model", None)
        if inner is not None and hasattr(inner, "set_capture_layers"):
            inner.set_capture_layers(ids)
        elif ids:
            raise NotImplementedError(
                f"{type(self).__name__} does not support aux-hidden capture (set_capture_layers)"
            )

    @abstractmethod
    def forward(
        self, return_hidden: bool = False
    ) -> Union[
        "torch.Tensor",
        "Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]",
    ]:
        """Return lm_head logits; or ``(logits, last_hidden, aux_hidden)`` if ``return_hidden``."""
        ...
