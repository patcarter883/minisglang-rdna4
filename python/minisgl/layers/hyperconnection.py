"""Hyper-connections — the WIDE (`hc_count * hidden_size`) residual stream of Qwen3.8-Flash-Next.

WHAT A HYPER-CONNECTION IS
--------------------------
An ordinary transformer carries one `hidden_size` residual and wraps each block in
`x = x + block(norm(x))`. This architecture carries `hc_count` (= 4) parallel residual branches
concatenated into a single `hc_count * hidden_size` (= 10240) vector, and replaces BOTH halves of
that expression:

  * `mix`     — read the 4 branches down to the ONE `hidden_size` view the wrapped block consumes,
                weighting each branch channel by a data-dependent gate produced by a low-rank
                (rank `hc_lowrank` = 320) bottleneck over the whole wide vector.
  * `combine` — write the block's `hidden_size` output back into ALL 4 branches, each branch scaled
                by its own scalar gate.

Consequences that are easy to miss and fail silently:

  * There is **no `input_layernorm` and no `post_attention_layernorm`** in this checkpoint. The
    hyper-connection's own `hc_norm` is the pre-block norm — it is applied inside `mix`, and the
    same normed vector is what the `combine` gate reads. Adding a per-layer norm would be an
    unfillable key at load; omitting `hc_norm` would run un-normalized blocks.
  * There is **no final `model.norm`** either. The top-level `hyper_connection_mixer` is this same
    block built with `use_combine=False`; its `mix` is what folds the 10240-wide stream down to the
    2560 the `lm_head` consumes. It therefore ships 3 tensors, not 4.
  * `combine` GATES ON THE NORMED stream but ADDS TO THE UNNORMED one. Both are returned by `mix`
    as the residual pair precisely so the caller cannot accidentally re-derive the wrong one.

THE MATH (ported verbatim, not re-derived)
------------------------------------------
Reference: `sglang/srt/layers/hyperconnection.py::GatedResidual`, the nested `_mix_compute` /
`_combine_compute` closures plus `GroupedGemmaRMSNorm`, as wired by
`sglang/srt/models/qwen4_exp.py` (which builds every one of these with `hc_per_branch_norm=True`,
i.e. `hc_norm` is grouped over `hidden_size` and spans the full wide vector). With
`W_down [lowrank, hc*H]`, `W_up [hc*H, lowrank]`, `W_inject [hc, hc*H]`, `x` the wide stream:

    mix:
        n   = hc_norm(x)                                       # (T, hc*H), grouped (1+w) RMSNorm
        t   = silu(n @ W_down^T / hc)                          # (T, lowrank)
        g   = sigmoid(t @ W_up^T)                              # (T, hc*H)
        out = (g.unflatten(-1, (hc, H)) * n.unflatten(-1, (hc, H))).mean(dim=-2)   # (T, H)
        return out, (x, n)

    combine(y, (x, n)):
        b   = 2 * sigmoid(n @ W_inject^T / hc)                 # (T, hc)
        return (x.unflatten(-1, (hc, H)) + y.unsqueeze(-2) * b.unsqueeze(-1)).flatten(-2)

Every scalar there is load-bearing and silent if dropped: the `/ hc` appears TWICE (before the silu
and before the inject sigmoid) and is NOT the same as folding it into the weights; the reduction is
a `.mean` over branches, not a `.sum` (a factor of `hc` on the block input); the inject gate has a
leading `2 *`, so its neutral value is 1.0 and not 0.5. `tests/qwen4exp_hc_parity_test.py` pins all
of it against the reference source itself.

TENSOR PARALLELISM
------------------
NEVER SHARDED, and not by omission. The checkpoint's `quantization_config.ignore` list contains
`*hyper_connection*`, so these four tensors are bf16 on every rank; they total ~13 MB per block,
the rank-320 bottleneck has no clean split, and `mix` is a reduction over the branch axis that a
column split would have to all-reduce anyway. The only collective in a decoder layer stays the
wrapped block's own row-parallel all-reduce on its `hidden_size`-wide output, which happens BEFORE
`combine` — so every rank combines into an identical wide stream and the streams cannot drift.

GRAPH CAPTURE
-------------
Shape-static: two GEMMs, elementwise ops, and one reduction. No host sync, no data-dependent
control flow (the empty-batch guard is a Python shape test, constant for a given capture).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BaseOP
from .linear import LinearReplicated
from .norm import GroupedRMSNorm

# (unnormed wide stream, normed wide stream) — what `mix` hands to the matching `combine`.
HCResidual = Tuple[torch.Tensor, torch.Tensor]


class HyperConnection(BaseOP):
    """One hyper-connection block.

    Attribute names ARE the checkpoint's leaf names (the loader walks attributes), exactly:

        hc_norm.weight                  [hc*H]            grouped (1+w) RMSNorm, group = H
        input_mix_weight_down.weight    [lowrank, hc*H]
        input_mix_weight_up.weight      [hc*H, lowrank]
        block_inject_weight.weight      [hc, hc*H]        ONLY when `use_combine`

    `use_combine=False` builds the top-level `hyper_connection_mixer`, which owns 3 tensors and is
    used for its `mix` alone; calling `combine` on it raises rather than inventing a gate.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        hc_count: int,
        hc_lowrank: int,
        eps: float,
        use_combine: bool,
    ) -> None:
        if not hc_count or not hc_lowrank:
            raise ValueError(
                f"HyperConnection needs a non-zero hc_count/hc_lowrank, got "
                f"{hc_count!r}/{hc_lowrank!r} — a config that does not carry them is not a "
                f"hyper-connection architecture"
            )
        wide = hc_count * hidden_size
        self.hc_norm = GroupedRMSNorm(wide, group_size=hidden_size, eps=eps)
        self.input_mix_weight_down = LinearReplicated(wide, hc_lowrank, has_bias=False)
        self.input_mix_weight_up = LinearReplicated(hc_lowrank, wide, has_bias=False)
        if use_combine:
            self.block_inject_weight = LinearReplicated(wide, hc_count, has_bias=False)
        self._hc = hc_count
        self._hs = hidden_size
        self._wide = wide
        self._use_combine = use_combine

    # -- geometry ----------------------------------------------------------

    @property
    def hc_count(self) -> int:
        return self._hc

    @property
    def hidden_size(self) -> int:
        return self._hs

    @property
    def wide_size(self) -> int:
        """Width of the residual stream this block reads and writes (`hc_count * hidden_size`)."""
        return self._wide

    # -- compute -----------------------------------------------------------

    def mix(self, hyper_input: torch.Tensor) -> Tuple[torch.Tensor, HCResidual]:
        """Wide stream (..., hc*H) -> the (..., H) view the wrapped block consumes.

        Returns `(mixed, (hyper_input, normed))`. The second element is handed straight back to
        `combine`; it carries BOTH the unnormed stream (what the block output is added to) and the
        normed one (what the inject gate reads), because deriving either from the other at the
        combine site is exactly the mistake this pair exists to prevent."""
        if hyper_input.shape[-1] != self._wide:
            raise ValueError(
                f"hyper-connection mix expects a {self._wide}-wide residual "
                f"(hc_count {self._hc} x hidden {self._hs}), got {hyper_input.shape[-1]}"
            )
        normed = self.hc_norm.forward(hyper_input)
        if hyper_input.shape[0] == 0:
            # An idle rank still has to return the right shapes so the collectives downstream line
            # up; the GEMMs below are undefined on a 0-row batch under some backends.
            return hyper_input.new_empty((*hyper_input.shape[:-1], self._hs)), (
                hyper_input,
                normed,
            )
        # `/ self._hc` BEFORE the silu — it is a scale on the pre-activation, so it cannot be folded
        # into `input_mix_weight_down` without also changing where silu saturates.
        t = F.silu(self.input_mix_weight_down.forward(normed) / self._hc)
        gate = torch.sigmoid(self.input_mix_weight_up.forward(t))
        mixed = (
            gate.unflatten(-1, (self._hc, self._hs))
            * normed.unflatten(-1, (self._hc, self._hs))
        ).mean(dim=-2)
        return mixed, (hyper_input, normed)

    def combine(self, block_output: torch.Tensor, residuals: HCResidual) -> torch.Tensor:
        """Inject the wrapped block's (..., H) output back into all `hc_count` branches.

        `residuals` is the pair `mix` returned. Note the asymmetry, transcribed from the reference:
        the per-branch gate is computed from the NORMED stream, the addition lands on the UNNORMED
        one."""
        if not self._use_combine:
            raise RuntimeError(
                "this HyperConnection was built with use_combine=False (the top-level "
                "hyper_connection_mixer), so it ships no block_inject_weight and cannot combine"
            )
        hyper_input, normed = residuals
        if hyper_input.shape[-1] != self._wide:
            raise ValueError(
                f"hyper-connection combine expects a {self._wide}-wide residual, got "
                f"{hyper_input.shape[-1]}"
            )
        if block_output.shape[-1] != self._hs:
            raise ValueError(
                f"hyper-connection combine expects a {self._hs}-wide block output, got "
                f"{block_output.shape[-1]}"
            )
        if block_output.shape[0] == 0:
            return hyper_input
        # Leading 2x: the gate is neutral at 1.0, not 0.5. Dropping it halves every block's
        # contribution to the stream for all 48 layers — coherent-looking, uniformly wrong text.
        inject = 2.0 * torch.sigmoid(
            self.block_inject_weight.forward(normed) / self._hc
        )
        branches = hyper_input.unflatten(-1, (self._hc, self._hs))
        return (branches + block_output.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)

    def forward(self, *args, **kwargs):  # pragma: no cover - use mix()/combine()
        raise NotImplementedError(
            "a HyperConnection is not a single function: use .mix() before the wrapped block and "
            ".combine() after it"
        )


__all__ = ["HyperConnection", "HCResidual"]
