"""THE drafter linear — one core, shared by every self-contained draft trunk.

Before this module there were TWO copies of the same class: `models/dflash.py::_PlainLinear` (used by
DFlash and, by import, the CCA drafter) and `models/glm_eagle3.py::_PlainLinear`. They started
identical and diverged, which is exactly the debt KERNEL_CORE_POLICY.md forbids for kernels and for
the same reason: every improvement made to one silently stranded the other. Concretely, the EAGLE3
copy missed BOTH improvements the DFlash copy received —

  * `device="meta"` construction. The drafter is built under `with torch.device(cuda)`, so a plain
    `torch.empty(out, in)` allocates an fp32 scaffold ON THE CARD that the loader is about to
    replace wholesale. At the ~1 GB drafters this class was written for that was merely wasteful; at
    Muse-Glimmer's 2.556B it is 10.2 GiB and the drafter OOMs during construction, before a single
    weight is read.
  * weight-only quantization (nvfp4 / fp8 / int8). EAGLE3's drafter simply cannot be quantized
    today, purely because of the copy.

So: ONE class, with weight format as a LOAD POLICY (never a subclass), and TP sharding as a second,
orthogonal policy.

WHY SHARDING. The drafter was replicated on every rank on the argument that it is "tiny" and that
replication keeps each rank's argmax identical with no collective. Both halves have expired:

  * It is not tiny. Muse-Glimmer's drafter is 2.556B — 2.48 GiB at fp8 and ~5.5 GiB at bf16, ON EACH
    16 GB card. Measured: the fp8 and bf16 arms cannot size a KV pool at all, so the drafter's
    precision is capped at 4-bit by the replication, not by anything about the model.
  * "No collective" was already untrue. The drafter borrows the target's TP-sharded `lm_head` via
    `logits_all_rows`, so propose already pays a collective every step.

A row-parallel all_reduce yields bit-identical outputs on every rank, so drafts stay in sync with no
extra broadcast — the sync property that motivated replication is preserved, not traded away.

SAFETY: sharding is opt-in per tensor AND self-disabling. At tp_size == 1, or when a dimension does
not divide the TP size, the tensor silently falls back to replicated — so TP=1 and awkward geometries
are byte-identical to the pre-change behaviour rather than a crash or a silent wrong answer.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

_QUANT_OPS: dict[str, object] = {}


def _quant_op(name: str):
    """Lazily resolve a `fp8_wmma` op by name, caching the miss as well as the hit.

    Same shape as `layers/embedding.py`'s `dense_bf16_gemv` resolver: one import attempt per op, so a
    build without the kernel package degrades to the dequant path instead of failing a boot.
    """
    if name not in _QUANT_OPS:
        try:
            _QUANT_OPS[name] = getattr(__import__("fp8_wmma", fromlist=[name]), name)
        except Exception:
            _QUANT_OPS[name] = None
    return _QUANT_OPS[name]


# (decode GEMV, prefill GEMM) per stored weight format. Both arms of a pair take the SAME weight
# layout — (N,K) bytes plus (N,) f32 per output channel — so one prepared weight serves every M, and
# the only thing M selects is which kernel reads it.
#   fp8  : FORMAT_MATRIX.md G1 (GEMV) + G2 (GEMM)
#   int8 : FORMAT_MATRIX.md G9, both halves
_QUANT_ARMS = {
    torch.float8_e4m3fn: ("dense_w8a16_gemv", "dense_w8a16_gemm"),
    torch.int8: ("dense_int8a16_gemv", "dense_int8a16_gemm"),
}

from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.layers.base import BaseOP
from minisgl.utils import init_logger

logger = init_logger(__name__)

SHARD_NONE = "none"
SHARD_COL = "col"   # split the OUTPUT rows; no collective (outputs are disjoint slices)
SHARD_ROW = "row"   # split the INPUT columns; partial sums -> all_reduce


class DraftLinear(BaseOP):
    """An [out, in] weight, no bias, with two orthogonal policies:

      * FORMAT (`load` / `load_quant`): bf16 | fp8_e4m3 | int8 | nvfp4-e2m1, per-tensor.
      * SHARDING (`shard`): none | col | row, resolved against the live TP size at construction.

    The checkpoint always arrives FULL; slicing to the local shard happens at load, before quant, so
    a quantized shard is quantized from its own rows (per-output-channel scales stay correct).
    """

    def __init__(self, in_features: int, out_features: int, shard: str = SHARD_NONE) -> None:
        tp = get_tp_info()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.full_in = in_features
        self.full_out = out_features

        # Resolve the shard policy NOW, against the real TP size and real divisibility. A tensor that
        # cannot be split evenly stays replicated rather than crashing the boot or, worse, silently
        # splitting unevenly.
        if self.tp_size <= 1 or shard == SHARD_NONE:
            shard = SHARD_NONE
        elif shard == SHARD_COL and out_features % self.tp_size != 0:
            logger.warning_rank0(
                f"DraftLinear: out_features={out_features} not divisible by tp={self.tp_size} — "
                "keeping this tensor REPLICATED")
            shard = SHARD_NONE
        elif shard == SHARD_ROW and in_features % self.tp_size != 0:
            logger.warning_rank0(
                f"DraftLinear: in_features={in_features} not divisible by tp={self.tp_size} — "
                "keeping this tensor REPLICATED")
            shard = SHARD_NONE
        self.shard = shard

        self.local_out = out_features // self.tp_size if shard == SHARD_COL else out_features
        self.local_in = in_features // self.tp_size if shard == SHARD_ROW else in_features

        # META, not the ambient device — see the module docstring. Keeps .shape/.dtype for the
        # loader's assertions and costs nothing.
        self.weight = torch.empty(self.local_out, self.local_in, device="meta")
        self._wq = None   # [out, in] fp8_e4m3 / int8
        self._ws = None   # [out, 1] per-output-channel scale
        self._w4 = None   # [out, in/8] int32-packed E2M1      (nvfp4)
        self._w4s = None  # [in/16, out] fp16 group scale, GROUP-MAJOR (nvfp4)
        self._comm = DistributedCommunicator() if shard == SHARD_ROW else None

    @property
    def full_shape(self) -> tuple:
        """Shape of the FULL checkpoint tensor this linear consumes. Loaders must assert against
        this, not against `weight.shape`, which is the local shard once sharding is on."""
        return (self.full_out, self.full_in)

    # ---- loading -------------------------------------------------------------------------------
    def _slice(self, w: torch.Tensor) -> torch.Tensor:
        """Take this rank's slice of a FULL [out, in] checkpoint tensor."""
        if self.shard == SHARD_COL:
            n = self.full_out // self.tp_size
            return w[self.tp_rank * n : (self.tp_rank + 1) * n].contiguous()
        if self.shard == SHARD_ROW:
            n = self.full_in // self.tp_size
            return w[:, self.tp_rank * n : (self.tp_rank + 1) * n].contiguous()
        return w

    def load(self, w: torch.Tensor, device) -> None:
        w = self._slice(w)
        assert tuple(w.shape) == (self.local_out, self.local_in), (
            f"DraftLinear shard mismatch: got {tuple(w.shape)}, want "
            f"{(self.local_out, self.local_in)} (shard={self.shard}, tp={self.tp_size})")
        self.weight = w.to(device)

    def load_quant(self, w: torch.Tensor, mode: str, compute_dtype, device) -> None:
        """RTN weight-only quant of a FULL [out, in] checkpoint tensor. Pass it on CPU so the fp32
        transient stays in host RAM and only the packed shard reaches the GPU.
        mode: 'fp8' | 'int8' | 'nvfp4'."""
        w = self._slice(w)
        if mode == "nvfp4":
            # 4-bit E2M1 + fp16 per-16-element group scale, riding the SAME shared W4A8 core the
            # target's NVFP4 linears use — a weight format is a load POLICY on that core, never a new
            # kernel (KERNEL_CORE_POLICY.md). Unlike fp8/int8 below it is not merely smaller: the
            # e2m1 kernel decodes to fp8 e4m3 in-register at the WMMA, so nothing is materialized.
            from minisgl.quant.nvfp4 import quantize_nvfp4_rtn

            packed, scales = quantize_nvfp4_rtn(w)
            self._w4 = packed.to(device)
            # GROUP-MAJOR (K//16, N): the op indexes scales as [g*N + n], so N must be contiguous for
            # the read to coalesce. Transposing here rather than in the encoder keeps the encoder in
            # checkpoint layout, which its round-trip test and the golden decoder expect.
            self._w4s = scales.transpose(0, 1).contiguous().to(device)
            self.weight = None
            return
        wf = w.float()
        amax = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)  # [out,1]
        if mode == "fp8":
            fmax = 448.0  # E4M3 max
            s = amax / fmax
            wq = (wf / s).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
        else:  # int8
            s = amax / 127.0
            wq = (wf / s).round().clamp(-127, 127).to(torch.int8)
        self._wq = wq.contiguous().to(device)
        self._ws = s.to(compute_dtype).to(device)
        # (N,) f32 per OUTPUT CHANNEL — the layout `fp8_wmma.dense_w8a16_gemv` wants. Cached at load
        # because it is a constant of the loaded weight; recomputing a squeeze+cast per forward on
        # the decode path is exactly the kind of per-step allocation this class exists to avoid.
        # (N,) f32 per OUTPUT CHANNEL — the layout both dense GEMVs want. fp8 and int8 share it:
        # the two ops differ only in the byte-decode policy inside one shared loader.
        self._ws_f32 = s.squeeze(-1).float().contiguous().to(device)
        self.weight = None

    def _w8a16_or_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """The fp8/int8 weight arm. Streams the e4m3 weight through a kernel when one applies.

        WHY THIS EXISTS. `F.linear(x, self._wq.to(x.dtype) * self._ws)` materialises a full
        [out, in] dequantised temporary on EVERY forward — ~9 B/elem against bf16's 2 — which is the
        mechanism behind `tools/serve.sh:324` "fp8 on this drafter is WORSE, 27.0 tok/s". The kernel
        that removes it (`fp8_wmma.dense_w8a16_gemv`, e4m3 weight x UNQUANTIZED bf16/fp16 act) has
        existed since 2026-09-07 and was recorded as closing gap G1 — but NOTHING EVER CALLED IT.
        `rdna4-hip-kernels/FORMAT_MATRIX.md` listed G1 as closed on the strength of the kernel alone.

        NOT BIT-IDENTICAL to the dequant path, and the op's own docstring says so: the activation is
        untouched (still bf16/fp16, never quantised per token), but the scale is applied once in fp32
        AFTER the K-sum instead of being folded into a bf16-ROUNDED weight before the GEMM. One
        rounding step is removed — plausibly more accurate — and the accumulation order differs. On a
        DRAFTER that matters: its whole job is agreeing with the target, so acceptance has to be
        measured, not assumed. `tests/draft_linear_w8a16_parity_test.py` pins the numeric distance.

        int8 takes the same treatment via `dense_int8a16_gemv` / `dense_int8a16_gemm` (gap G9) — the
        identical core with one byte-decode policy swapped, which is why it is a branch here and not
        a second code path. M > 16 goes to the GEMM arm rather than falling back: a drafter prefills
        too, and leaving prefill on the dequant path would have kept the [out, in] temporary on
        exactly the shapes where it costs most. Only K % 16 (a kernel shape limit) and a missing
        kernel package still fall back.
        """
        wq, ws = self._wq, self._ws
        arms = _QUANT_ARMS.get(wq.dtype)
        if arms is not None and x.is_cuda and self._ws_f32 is not None:
            shp = x.shape
            x2 = x.reshape(-1, shp[-1])
            if (x2.shape[-1] % 16) == 0:
                fn = _quant_op(arms[0] if x2.shape[0] <= 16 else arms[1])
                if fn is not None:
                    # e4m3 has no uint8 storage of its own in torch; the ops take the byte view.
                    wb = wq.view(torch.uint8) if wq.dtype is torch.float8_e4m3fn else wq
                    out = fn(x2.contiguous(), wb, self._ws_f32)
                    return out.reshape(*shp[:-1], out.shape[-1])
        return F.linear(x, wq.to(x.dtype) * ws)  # dequant [out,in] * [out,1]

    # ---- forward -------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._w4 is not None:
            from minisgl.quant import kernels

            # The kernel is 2-D and wants contiguous rows; callers pass both [T, H] (denoise) and
            # [N, Q, H] (the captured batched twin), so flatten and restore. Being explicit about
            # contiguity rather than trusting the caller is deliberate — the NoPE bug in
            # AttentionLayer was exactly a contiguity invariant a neighbouring op had been providing.
            shp = x.shape
            x2 = x.reshape(-1, shp[-1]).contiguous()
            out = kernels.w4a8_linear(x2, self._w4, self._w4s, None, 16, weight_is_e2m1=True)
            y = out.reshape(*shp[:-1], out.shape[-1])
        elif self._wq is not None:
            y = self._w8a16_or_dequant(x)
        else:
            y = F.linear(x, self.weight)
        if self._comm is not None:
            # Row-parallel: every rank holds a partial sum over its slice of the contraction dim.
            # The all_reduce is what makes the result bit-identical across ranks, which is what keeps
            # the drafters' argmax in sync without a separate broadcast.
            y = self._comm.all_reduce(y)
        return y


__all__ = ["DraftLinear", "SHARD_NONE", "SHARD_COL", "SHARD_ROW"]
