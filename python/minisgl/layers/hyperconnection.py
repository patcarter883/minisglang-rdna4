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

LAUNCH COUNT — WHY THIS FILE IS SHAPED THE WAY IT IS
----------------------------------------------------
MEASURED (`docs/measurements/HC_FUSION_2026-09-05/`, 40-layer TP=2 boot, graphs captured and
replayed): the 97 hyper-connection blocks issue **15.93 kernel launches per block = 1,545 per decode
step**, and cost **7.53 ms/step of GPU time even CAPTURED** — of which only 2.56 ms is bandwidth
(the two 6.55 MB low-rank weights). The other ~5 ms is ~1,000 kernels each doing a few microseconds
of work on a 20 KB tensor. Capture removes the HOST launch, not the node.

So the transcription below is deliberately written to issue FEWER, larger kernels while computing
the identical function. Four changes, each of them EXACT (no rounding moves in fp32) and each pinned
by `tests/qwen4exp_hc_parity_test.py`:

  1. `(1 + hc_norm.weight)` is a constant — memoized in `GroupedRMSNorm`, not re-added per call.
  2. The two `/ hc` scalings are folded into the weights ONCE, in `post_load`. Division by `hc` is a
     lossless exponent shift when `hc` is a power of two (it is 4 here), and scaling a matmul's
     weight is bit-identical to scaling its output because fp rounding commutes with an exact
     power-of-two scale. When `hc` is NOT a power of two the fold is refused and the runtime divide
     stays — correctness first, speed second.
  3. `block_inject_weight` [hc, wide] is CONCATENATED onto `input_mix_weight_down` [lowrank, wide]
     in `post_load` (the two `.weight` attributes become disjoint views of ONE buffer, so this costs
     no memory) and both are read by a single GEMV over the SAME `normed` input. The [4, 10240]
     inject GEMV was the single most expensive kernel in the block — 4 output rows means 4
     workgroups reducing 10240 elements each, ~12 us of pure latency for 80 KB of weights.
     It therefore moves from `combine` into `mix`, which is why `HCResidual` carries three elements.
  4. `combine`'s `2 *`, gate broadcast-multiply and residual add collapse into one `torch.addcmul`.

`post_load` is the hook for 2 and 3 and it is NOT optional-but-nice: without it the layer still
computes the right answer (the divides and the second GEMV are simply still there), so a path that
forgets it is SLOW, never WRONG. That direction is chosen on purpose.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from . import _tail_hip
from .base import BaseOP
from .linear import LinearReplicated
from .norm import GroupedRMSNorm

# What `mix` hands to the matching `combine`:
#   [0] the UNNORMED wide stream — what the block output is added to
#   [1] the NORMED wide stream — what the inject gate is computed from
#   [2] the inject gate's PRE-SIGMOID logits [.., hc], or None for a `use_combine=False` mixer
# [2] exists because the inject GEMV reads [1] and nothing else, so it can be issued in `mix`
# alongside the low-rank down-projection that reads the same vector — one GEMV instead of two. It is
# the logits and not the gate so that `combine` still owns the sigmoid (and the `2 *`, now folded
# into an `addcmul`), which keeps the gate's arithmetic in one place.
HCResidual = Tuple[torch.Tensor, torch.Tensor, "torch.Tensor | None"]


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
        # --- filled by post_load(); see the LAUNCH COUNT section of the module docstring ----------
        # `_fused_w` OWNS the storage that `input_mix_weight_down.weight` and (when combining)
        # `block_inject_weight.weight` become views of. Underscore-prefixed, so it is invisible to
        # `state_dict` / `load_state_dict` / `post_load`'s recursion exactly like `_hc`.
        self._fused_w: torch.Tensor | None = None
        # A `LinearReplicated` over the packed buffer, so the packed GEMV goes through the SAME
        # `LinearMethod` (kernel choice, decode-GEMV threshold, engage ledger entry) as the unpacked
        # call it replaces — not a second matmul path that could diverge from it.
        self._fused_lin: LinearReplicated | None = None
        # True once `/ hc` has been folded into `_fused_w`. False keeps the runtime divides.
        self._scale_folded = False
        self._prepared = False

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

    # -- weight preparation ------------------------------------------------

    def post_load(self) -> None:
        """Fold `/ hc` into the two weights that consume `normed`, and pack them into ONE buffer.

        BOTH are exact and both are pure launch-count work; neither changes the function computed.

        THE FOLD. `mix` scales the down-projection's OUTPUT by `1/hc` and `combine` scales the
        inject projection's output by `1/hc`. When `hc` is a power of two, `1/hc` is exact, and
        floating-point rounding commutes with an exact power-of-two scale — so scaling every weight
        instead of every output is BIT-identical at any dtype (bf16 included: the exponent moves,
        the significand does not), for the cost of zero kernels per step instead of two. `hc` is 4
        in this architecture; a non-power-of-two `hc` refuses the fold and keeps the divides, because
        "close enough" here would be a silent numeric change across all 48 layers.

        THE PACK. `input_mix_weight_down` [lowrank, wide] and `block_inject_weight` [hc, wide] read
        the SAME `normed` vector, so they are one GEMV with `lowrank + hc` output rows. The packed
        buffer becomes the storage and the two checkpoint-named `.weight` attributes become disjoint
        CONTIGUOUS VIEWS of it — same shapes, same dtype, same total bytes, so `state_dict()` and
        every byte-accounting consumer see exactly what they saw before. The originals are dropped.

        Idempotent by `_prepared`: a second call is a no-op rather than a second `/ hc` (which would
        be silent — a uniformly 4x-small pre-activation across every layer). `BaseOP.post_load`'s
        `_post_load_done` already prevents the chunked loader from double-finalizing, and this flag
        makes the guarantee local. A load that happens AFTER `post_load` is the one hazard this
        cannot see, and it is also the one thing the loader never does (it loads, then finalizes)."""
        super().post_load()
        if self._prepared:
            return
        self._prepared = True

        w_down = self.input_mix_weight_down.weight
        rows = [w_down]
        if self._use_combine:
            rows.append(self.block_inject_weight.weight)
        n_down = w_down.shape[0]

        fused = torch.empty(
            (sum(r.shape[0] for r in rows), self._wide),
            dtype=w_down.dtype,
            device=w_down.device,
        )
        off = 0
        for r in rows:
            fused[off : off + r.shape[0]].copy_(r)
            off += r.shape[0]

        # power of two -> `/ hc` is an exponent shift, i.e. exact, i.e. foldable
        if self._hc > 0 and (self._hc & (self._hc - 1)) == 0:
            fused.div_(self._hc)
            self._scale_folded = True

        self._fused_w = fused
        self.input_mix_weight_down.weight = fused[:n_down]
        if self._use_combine:
            self.block_inject_weight.weight = fused[n_down:]
            self._fused_lin = LinearReplicated(self._wide, fused.shape[0], has_bias=False)
            self._fused_lin.weight = fused

    def _fused_ok(self, x: torch.Tensor) -> bool:
        """May this call read both projections out of ONE GEMV?

        TWO conditions, and the second one is a numerics contract, not a nicety.

        1. The packed buffer is still the storage behind both checkpoint-named weights. Fail-CLOSED,
           and cheap (two `data_ptr()` reads, and under graph capture this runs once at capture time
           rather than per step). If anything rebound or moved either weight after `post_load` — a
           re-load, a residency move — the packed GEMV would silently read a stale copy, so the block
           falls back to the two separate GEMVs, which read whatever the weights are NOW and are
           therefore right either way (the `/ hc` fold travels WITH the values, so the fallback must
           not re-divide, and it does not).

        2. The kernel that will run it is ROW-INVARIANT — output row i does not depend on how many
           other rows are in the matrix. `minv_linear`'s HIP kernels are (each output row is an
           independent fp32 dot in a fixed 16-wide K order; that is the same property its
           M-invariance rests on, on the other axis) — but its **rocBLAS / `F.linear` fallback is
           NOT**: BLAS picks its blocking from N, so a [324, 10240] call and a [4, 10240] call
           accumulate the same row differently. MEASURED: packing under `F.linear` moves the inject
           logits by 9.54e-07 at fp32, which is ~3 ulp on `combine`'s output and fails
           `test_combine_matches_reference`. So the pack is asked EXACTLY where `minv_linear` says
           it will use its own kernel, and skipped where it will not — the fp32/CPU parity oracle
           lands in the second case, which is why it stays bit-exact."""
        fw = self._fused_w
        if fw is None:
            return False
        from .minv import minv_supported

        if not minv_supported(x, fw):
            return False
        if self.input_mix_weight_down.weight.data_ptr() != fw.data_ptr():
            return False
        if self._use_combine:
            n_down = self.input_mix_weight_down.weight.shape[0]
            if self.block_inject_weight.weight.data_ptr() != fw[n_down:].data_ptr():
                return False
        return True

    # -- compute -----------------------------------------------------------

    def mix(self, hyper_input: torch.Tensor) -> Tuple[torch.Tensor, HCResidual]:
        """Wide stream (..., hc*H) -> the (..., H) view the wrapped block consumes.

        Returns `(mixed, (hyper_input, normed, inject_logits))`. The triple is handed straight back
        to `combine`; it carries the unnormed stream (what the block output is added to), the normed
        one (what the inject gate reads), and the inject gate's PRE-SIGMOID logits, because deriving
        any of them from the other at the combine site is exactly the mistake this exists to
        prevent. `inject_logits` is produced HERE, by the same GEMV as the low-rank down-projection,
        because both read `normed` and nothing else — see the module docstring."""
        if hyper_input.shape[-1] != self._wide:
            raise ValueError(
                f"hyper-connection mix expects a {self._wide}-wide residual "
                f"(hc_count {self._hc} x hidden {self._hs}), got {hyper_input.shape[-1]}"
            )
        normed = self.hc_norm.forward(hyper_input)
        if hyper_input.shape[0] == 0:
            # An idle rank still has to return the right shapes so the collectives downstream line
            # up; the GEMMs below are undefined on a 0-row batch under some backends.
            empty_inj = (
                hyper_input.new_empty((*hyper_input.shape[:-1], self._hc))
                if self._use_combine
                else None
            )
            return hyper_input.new_empty((*hyper_input.shape[:-1], self._hs)), (
                hyper_input,
                normed,
                empty_inj,
            )

        # ONE GEMV over `normed` for both consumers of it (see post_load): rows [0, lowrank) are the
        # low-rank bottleneck, rows [lowrank, lowrank+hc) are the inject gate's logits.
        inject_logits = None
        if self._use_combine and self._fused_ok(normed):
            n_down = self.input_mix_weight_down.weight.shape[0]
            z = self._fused_lin.forward(normed)
            down, inject_logits = z[..., :n_down], z[..., n_down:]
        else:
            down = self.input_mix_weight_down.forward(normed)

        # The `/ self._hc` is a scale on the PRE-ACTIVATION (it changes where silu saturates), so it
        # cannot be dropped — only moved. `post_load` moved it into the weight, exactly, when `hc`
        # is a power of two; otherwise it is still right here.
        if not self._scale_folded:
            down = down / self._hc
        t = F.silu(down)
        gate_logits = self.input_mix_weight_up.forward(t)
        # ONE launch for sigmoid + per-branch multiply + mean, instead of three on a 20 KB tensor.
        # The kernel takes the PRE-sigmoid logits and rounds at the reference's rounding points, so
        # this is bit-exact with the fallback below, not merely close (tail_kernels.hip).
        if hasattr(_tail_hip, "hc_mix_epilogue") and _tail_hip.active(gate_logits, None):
            mixed = _tail_hip.hc_mix_epilogue(
                gate_logits.contiguous(), normed.contiguous(), self._hc
            )
            return mixed, (hyper_input, normed, inject_logits)
        gate = torch.sigmoid(gate_logits)
        mixed = (
            gate.unflatten(-1, (self._hc, self._hs))
            * normed.unflatten(-1, (self._hc, self._hs))
        ).mean(dim=-2)
        return mixed, (hyper_input, normed, inject_logits)

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
        hyper_input, normed, inject_logits = residuals
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
        if inject_logits is None:
            # `mix` did not have a packed buffer to read the gate out of (no `post_load`, or the
            # weights moved under it), so the gate's own GEMV runs here, as it always did.
            inject_logits = self.block_inject_weight.forward(normed)
            if not self._scale_folded:
                inject_logits = inject_logits / self._hc
        # ONE launch for sigmoid + the gated broadcast-add. The kernel takes the PRE-sigmoid
        # logits, keeps the leading 2x as an exact power-of-two scale on a product that is rounded
        # to the tensor dtype FIRST, and compiles with fp contract off — so it reproduces the
        # `torch.add(..., alpha=2.0)` chain below bit for bit rather than contracting to an FMA,
        # which is the ~1 ulp drift this method already refuses addcmul over.
        if hasattr(_tail_hip, "hc_combine") and _tail_hip.active(hyper_input, None):
            return _tail_hip.hc_combine(
                hyper_input.contiguous(), block_output.contiguous(),
                inject_logits.contiguous(), self._hc,
            )
        gate = torch.sigmoid(inject_logits)
        branches = hyper_input.unflatten(-1, (self._hc, self._hs))
        # Leading 2x: the gate is neutral at 1.0, not 0.5. Dropping it halves every block's
        # contribution to the stream for all 48 layers — coherent-looking, uniformly wrong text.
        # It rides on `add`'s `alpha` instead of being its own `2 * sigmoid(...)` launch. That is
        # BIT-IDENTICAL and not merely close: `2 *` is an exact power-of-two scale, so
        # `x + 2*(y*g)` and `x + y*(2*g)` round to the same number, verified at fp32 AND bf16 in
        # `test_combine_matches_reference`.
        #
        # NOT `torch.addcmul(..., value=2.0)`, which would also collapse the multiply and would be
        # ONE launch instead of two: it contracts to an FMA, so it skips the rounding of the product
        # and disagrees with the reference by ~1 ulp (measured 4.77e-07 at fp32). More accurate,
        # still different, and this file's contract is bit-exactness with the reference.
        return torch.add(
            branches, block_output.unsqueeze(-2) * gate.unsqueeze(-1), alpha=2.0
        ).flatten(-2)

    def forward(self, *args, **kwargs):  # pragma: no cover - use mix()/combine()
        raise NotImplementedError(
            "a HyperConnection is not a single function: use .mix() before the wrapped block and "
            ".combine() after it"
        )


__all__ = ["HyperConnection", "HCResidual"]
