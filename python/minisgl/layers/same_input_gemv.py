"""Several unquantised linears over the SAME input, run at decode as ONE GEMV.

WHY. On this box a decode GEMV launch costs a fixed ~5 us plus ~3.5 us of dead time between
graph-replayed kernels, so k small GEMVs over one input row cost far more than one GEMV k times as
wide — Gemma-4's q/k/v measured 23.1 us as three launches vs 14.2 us merged (rdna4 b934f811).

HOW, without touching a loader. The members keep their own parameters, names and TP layout; after
load their weights are concatenated ONCE into one (sum N_i, K) buffer and each member's `weight`
is re-pointed at its row slice of it. So the state dict, the checkpoint keys, the sharding and the
weight bytes are all unchanged, and every other path that reads `member.weight` (prefill, CAM's
differentiable F.linear, a debug dump) sees the same tensor it always did. Decode rows (M <= 16)
then run one `dense_bf16_gemv` over the buffer and hand back column views; anything else runs the
members exactly as before.

WHAT QUALIFIES, and why each condition. All members must be UNQUANTISED 16-bit dense weights with no
bias, the same K, dtype and device: the fused call is the same `dense_bf16_gemv` each member's own
decode path already ends in (UnquantizedLinearMethod -> minv_linear -> dense_bf16_gemv, and
gdn._GemvLinear -> dense_bf16_gemv), so the fusion changes the launch count and nothing else. A
quantised member, a mixed set (e.g. a checkpoint that quantises in_proj_qkvz but leaves in_proj_ba
bf16), or a member with a bias keeps the separate calls — `build` returns None and says why once.

NUMERICS. Each output column is a per-(row, col) fp32 dot in a fixed K order (M-invariant), but the
tiling table keys its split-K choice on the GEMV's N, so the merged call may associate a column's K
sum differently from the member's own call — last-ulp, and identical at every M.
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

    def forward(self, x: torch.Tensor, separate: Callable[[torch.Tensor], List[torch.Tensor]]
                ) -> List[torch.Tensor]:
        """The members' outputs, in order. Decode rows: one GEMV, split into column views (row
        stride = the merged width — consumers take views of a split projection already). Otherwise
        `separate(x)`, i.e. exactly the unfused path."""
        if (x.dim() == 2 and x.shape[0] <= _MAXM and x.dtype == self.merged.dtype):
            from minisgl._hip_engage import engaged

            engaged(f"fp8_wmma.dense_bf16_gemv[same_input:{self.name}]")
            return list(_gemv()(x.contiguous(), self.merged).split(self.sizes, dim=-1))
        return separate(x)
