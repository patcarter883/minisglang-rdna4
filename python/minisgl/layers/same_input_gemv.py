"""Several linears over the SAME input, run as ONE GEMV/GEMM — unquantised or quantised.

Built in post_load. Unquantised members: weights are concatenated into one buffer and each member's
`weight` becomes a row view of it (state dict, sharding and bytes unchanged); decode rows (M <= 16)
run one dense_bf16_gemv and get column views back, larger M runs the members as before.
Quantised members: see SameInputQuantLinear. Sets that do not qualify (mixed formats, biases,
differing K) stay separate and are logged once.
"""
from __future__ import annotations

from typing import Callable, List, Sequence

import torch

from minisgl.utils import init_logger

_logger = init_logger(__name__)

# The decode GEMV's M ceiling (layers/minv.py _DECODE_GEMV_MAXM / gdn _GDN_PROJ_GEMV_MAXM).
_MAXM = 16

_gemv_fn = None
_gemv_probed = False


def _gemv():
    global _gemv_fn, _gemv_probed
    if not _gemv_probed:
        _gemv_probed = True
        try:
            from fp8_wmma import dense_bf16_gemv

            _gemv_fn = dense_bf16_gemv
        except Exception:
            _gemv_fn = None
    return _gemv_fn


def _unquantised_weight(member) -> torch.Tensor | None:
    """The member's dense 16-bit weight if it is an unquantised, bias-free linear, else None."""
    from minisgl.quant.method import UnquantizedLinearMethod

    if isinstance(member, torch.nn.Linear):                       # gdn._GemvLinear and friends
        if member.bias is not None:
            return None
        return member.weight
    method = getattr(member, "_method", None)
    if not isinstance(method, UnquantizedLinearMethod) or getattr(member, "bias", None) is not None:
        return None
    return getattr(member, "weight", None)


class SameInputGemv:
    """One decode GEMV over the members' concatenated weights. Build with `SameInputGemv.build`."""

    def __init__(self, name: str, members: Sequence, merged: torch.Tensor, sizes: List[int]):
        self.name = name
        self.members = list(members)
        self.merged = merged
        self.sizes = sizes

    @classmethod
    def build(cls, name: str, members: Sequence) -> "SameInputGemv | None":
        """Fuse `members` (all reading the same input) or return None, leaving them untouched."""
        ws = [_unquantised_weight(m) for m in members]
        why = None
        if any(w is None for w in ws):
            why = "a member is quantised or biased"
        elif len({(w.dtype, w.device, w.shape[1]) for w in ws}) != 1:
            why = "members differ in dtype / device / K"
        elif ws[0].dtype not in (torch.bfloat16, torch.float16) or not ws[0].is_cuda:
            why = f"weights are {ws[0].dtype} on {ws[0].device}"
        elif ws[0].shape[1] % 8:
            why = f"K={ws[0].shape[1]} is not a multiple of 8"
        elif _gemv() is None:
            why = "fp8_wmma.dense_bf16_gemv is unavailable"
        if why is not None:
            _logger.info_rank0(f"[same-input-gemv] {name}: kept separate ({why})")
            return None
        merged = torch.cat([w.detach() for w in ws], dim=0).contiguous()
        sizes = [w.shape[0] for w in ws]
        off = 0
        for m, n in zip(members, sizes):
            view = merged[off:off + n]
            if isinstance(m, torch.nn.Linear):
                m.weight.data = view          # keep the Parameter object (and its name); drop the old storage
            else:
                m.weight = view
            off += n
        return cls(name, members, merged, sizes)

    def forward(self, x: torch.Tensor, separate: Callable[[torch.Tensor], List[torch.Tensor]], **kw
                ) -> List[torch.Tensor]:
        """The members' outputs, in order. Decode rows: one GEMV, split into column views (row
        stride = the merged width — consumers take views of a split projection already). Otherwise
        `separate(x)`, i.e. exactly the unfused path."""
        if (x.dim() == 2 and x.shape[0] <= _MAXM and x.dtype == self.merged.dtype):
            from minisgl._hip_engage import engaged

            engaged(f"fp8_wmma.dense_bf16_gemv[same_input:{self.name}]")
            return list(_gemv()(x.contiguous(), self.merged).split(self.sizes, dim=-1))
        return separate(x)


# ---- QUANTISED members: one method.apply over op-layout tensors stacked on the OUTPUT dim ----------
# Every dense quantised method converts its checkpoint tensors to an op layout in
# process_weights_after_load, and those layouts share one rule for where N lives:
#   _w_packed_op (N, K/8) int4 codes, _w_op (N, K) fp8 bytes, _global_op (N,) f32 NVFP4 global,
#   _scales_op (N,) per-channel (fp8 W8A8)  ->  N is dim 0;
#   _scales_op (K/g, N) and _zeros_op (K/g, N/8) GROUP-MAJOR (N contiguous for coalescing) -> dim 1.
# So an output-dim stack of the members is exactly the op layout of the stacked weight.
_STACK_DIM = {"_w_packed_op": 0, "_w_op": 0, "_global_op": 0, "_zeros_op": 1}
_OP_ATTRS = ("_w_packed_op", "_w_op", "_scales_op", "_zeros_op", "_global_op")


def _stack_dim(attr: str, t: torch.Tensor) -> int:
    if attr == "_scales_op":
        return 0 if t.dim() == 1 else 1
    return _STACK_DIM[attr]


class _MergedLayer:
    """Attribute holder shaped like a post-load _LinearTPImpl, for `method.apply(layer, x, ...)`."""


class SameInputQuantLinear:
    """Quantised members run as ONE method.apply over their stacked op-layout tensors, at every M.

    A group-major scale plane stacks on dim 1, so a member's slice is not contiguous: the members'
    op tensors are freed and the merged layer serves prefill too. A member used directly afterwards
    raises (missing op tensor) rather than running a stale copy.
    """

    def __init__(self, name: str, method, merged: _MergedLayer, sizes: List[int]):
        self.name, self.method, self.merged, self.sizes = name, method, merged, sizes

    @classmethod
    def build(cls, name: str, members: Sequence) -> "SameInputQuantLinear | str":
        """The fused op, or the reason it cannot be built (members are then left untouched)."""
        methods = [getattr(m, "_method", None) for m in members]
        if any(mt is None for mt in methods):
            return "a member has no quant method"
        if len({type(mt) for mt in methods}) != 1:
            return f"members use different methods {sorted({type(mt).__name__ for mt in methods})}"
        q = [getattr(mt, "quant", None) for mt in methods]
        if any(x is None for x in q) or len({(x.group_size, x.bits) for x in q}) != 1:
            return "members differ in group size / bits"
        if any(getattr(m, "bias", None) is not None for m in members):
            return "a member has a bias"
        present = [[a for a in _OP_ATTRS if getattr(m, a, None) is not None] for m in members]
        if not present[0] or any(p != present[0] for p in present):
            return f"op-layout tensors differ or are missing ({present})"
        if len({bool(getattr(m, "_w4a16", False)) for m in members}) != 1:
            return "members disagree on the W4A16 arm"
        merged = _MergedLayer()
        sizes = []
        for m in members:
            w = getattr(m, "_w_packed_op", None)
            if w is None:
                w = m._w_op
            sizes.append(w.shape[0])
        for a in present[0]:
            ts = [getattr(m, a) for m in members]
            d = _stack_dim(a, ts[0])
            if len({(t.dtype, t.dim()) for t in ts}) != 1 or len({t.shape[1 - d] for t in ts if t.dim() == 2}) > 1:
                return f"{a}: members' layouts are not stackable"
            setattr(merged, a, torch.cat(ts, dim=d).contiguous())
        if getattr(members[0], "_w4a16", False):
            merged._w4a16 = True
            merged._n_out = sum(sizes)
        for m in members:
            for a in present[0]:
                delattr(m, a)
            m._fused_into = name
        return cls(name, methods[0], merged, sizes)

    def forward(self, x: torch.Tensor, separate=None, **kw) -> List[torch.Tensor]:
        if kw.get("x_fp8") is not None and getattr(self.method, "supports_producer_actquant", False):
            y = self.method.apply(self.merged, x, None, **kw)
        else:
            y = self.method.apply(self.merged, x, None)
        return list(y.split(self.sizes, dim=-1))


def fuse_same_input(name: str, members: Sequence):
    """ONE entry point for every call site: the bf16 fusion when the members are unquantised, the
    op-layout fusion when they share a quantised method, else None (members untouched, reason
    logged once). Call from post_load, after the members' own post_load."""
    if all(_unquantised_weight(m) is not None for m in members):
        return SameInputGemv.build(name, members)
    q = SameInputQuantLinear.build(name, members)
    if isinstance(q, str):
        _logger.info_rank0(f"[same-input-gemv] {name}: kept separate ({q})")
        return None
    _logger.info_rank0(f"[same-input-gemv] {name}: {len(members)} quantised members -> one {type(q.method).__name__} call")
    return q
