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
        self.weight = None

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
            y = F.linear(x, self._wq.to(x.dtype) * self._ws)  # dequant [out,in] * [out,1]
        else:
            y = F.linear(x, self.weight)
        if self._comm is not None:
            # Row-parallel: every rank holds a partial sum over its slice of the contraction dim.
            # The all_reduce is what makes the result bit-identical across ranks, which is what keeps
            # the drafters' argmax in sync without a separate broadcast.
            y = self._comm.all_reduce(y)
        return y


__all__ = ["DraftLinear", "SHARD_NONE", "SHARD_COL", "SHARD_ROW"]
