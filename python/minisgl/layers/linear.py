from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.quant.method import UnquantizedLinearMethod
from minisgl.utils import div_even
from minisgl.weights.granule import ExpertContainer

from .base import BaseOP

if TYPE_CHECKING:
    from minisgl.quant.method import LinearMethod


class _LinearTPImpl(ExpertContainer, BaseOP):
    """Real implementation of a linear layer with tensor parallelism.

    Weight layout + the matmul are delegated to a LinearMethod (default unquantized
    `F.linear`); a W4A8 method swaps in quantized buffers + the WMMA kernel without
    changing the sharding/collective logic here."""

    # ── granule descriptor ──────────────────────────────────────────────────────────────────────
    # DENSE IS THE SAME MECHANISM, and — load-bearing — the same CLASS: `ExpertContainer` with the
    # granule axis declared as "the whole container is ONE granule". Not a parallel pair of
    # look-alike methods on this class, because a look-alike drifts: it silently lacked
    # `expert_slice` and `offload_refusal`, so the one surface a residency consumer would write
    # against worked for a 512-expert MoE stack and raised `AttributeError` on a bf16 QKV
    # projection. Sharing the mixin is "MoE and dense land together" discharged in code.
    #
    # The quant methods build the same repack-and-delete shape the MoE containers do
    # (`_w_packed_op` / `_scales_op` / `_zeros_op` / `_global_op`, checkpoint names deleted), so
    # the same underscore-inclusive walk finds them and the same fail-closed rule applies — which
    # is what makes dense offload a merge gate rather than a follow-on. And the BIAS is in the
    # granule: a moved weight with a left-behind bias is the same silent-corruption family as a
    # left-behind scale.
    _granule_dense = True

    # The dense compressed-tensors path makes the same undetectable-by-walk decode decision the MoE
    # one does (`W4A8LinearMethod.process_weights_after_load` -> `ct_packed_sign_convention` ->
    # `layer._ct_sign`), so it declares the same policy path. There is no w13/w2 pair to cross-check
    # here, but it still lands in `fingerprint()`, which is what two TP ranks compare: a rank that
    # resolved the nibble histogram the other way decodes every int4 weight of this linear off by 8
    # quanta, and nothing else in the descriptor would move.
    _granule_policy = ("_ct_sign.uint4b8",)

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        quant_method: "LinearMethod | None" = None,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self._method = quant_method or UnquantizedLinearMethod()
        self._method.create_weights(self, local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        x_fp8: torch.Tensor | None = None,
        act_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`x_fp8`/`act_scales`: the PRODUCER-quantized activation pair from the RMSNorm that
        produced `x` (`RMSNorm.forward_quant` / `RMSNormFused.forward_quant`). A method that cannot
        use it MUST ignore it rather than fail — `supports_producer_actquant` is the declaration, and
        a caller that has the pair should pass it unconditionally rather than gate on the checkpoint.

        THE PAIR IS BOUND TO `x`, ROW FOR ROW. Slicing `x` without slicing the pair scales every
        token by another token's amax — wrong numbers, right shapes, no error — so any row split
        above this must carry both (see `tp_overlap.rowchunked_ar_span`'s `row_aligned`)."""
        if x_fp8 is None or not getattr(self._method, "supports_producer_actquant", False):
            return self._method.apply(self, x, self.bias)
        return self._method.apply(self, x, self.bias, x_fp8=x_fp8, act_scales=act_scales)

    def forward_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU for a MERGED gate_up projection: this linear outputs [.., 2*inter] = [gate | up];
        return silu(gate) * up = [.., inter]. Prefers the quant method's FUSED gemm+silu kernel
        (one launch, no [.., 2*inter] HBM round-trip) at decode; falls back to the BIT-IDENTICAL
        silu_and_mul(self.forward(x)) for prefill / unquantized / unsupported shapes / biased layers."""
        fn = getattr(self._method, "apply_swiglu", None)
        if fn is not None and self.bias is None:
            out = fn(self, x)
            if out is not None:
                return out
        from .activation import silu_and_mul

        return silu_and_mul(self.forward(x))

    def post_load(self) -> None:
        proc = getattr(self._method, "process_weights_after_load", None)
        if proc is not None:
            proc(self)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix and computes the full output (no collective).

    May carry a quant_method: a replicated AWQ/W4A8 linear is used where a row-parallel split would
    violate a kernel tiling constraint (e.g. the GLM shared-expert down_proj, whose K=1536 stays a
    multiple of 512 only un-sharded; the W4A8 dense kernel needs K % 512 == 0). Its full output is
    added to the already-all-reduced routed output, so there is no double-count and no extra reduce.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        quant_method: "LinearMethod | None" = None,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
            quant_method=quant_method,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        quant_method: "LinearMethod | None" = None,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias, quant_method)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        quant_method: "LinearMethod | None" = None,
        has_v: bool = True,
    ):
        """One column-parallel GEMM for [q | k | v], or [q | k] with `has_v=False`. Per rank: local q
        heads, then local kv heads (replicated when they do not divide TP), so the loader stacks
        each rank's q/k/v shards."""
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        n_kv_parts = 2 if has_v else 1
        full_isize = hidden_size
        full_osize = (num_qo_heads + n_kv_parts * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + n_kv_parts * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias, quant_method)


class LinearOProj(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        quant_method: "LinearMethod | None" = None,
    ):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias, quant_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._method.apply(self, x, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        quant_method: "LinearMethod | None" = None,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            input_size, output_size, local_input_size, local_output_size, has_bias, quant_method
        )

    def forward(self, x: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        # reduce=False returns the per-rank PARTIAL (pre-all-reduce) so a caller can sum several
        # row-parallel partials and all_reduce ONCE (fuse a MoE block's shared+routed reduce).
        # sum_r(a_r + b_r) == sum_r a_r + sum_r b_r, so the fused reduce is exact.
        y = self._method.apply(self, x, self.bias)
        if self._tp_size > 1 and reduce:
            y = self._comm.all_reduce(y)
        return y
