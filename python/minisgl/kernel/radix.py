from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def fast_compare_key(x: "torch.Tensor", y: "torch.Tensor") -> int:
    # Length of the common leading prefix of two 1-D int CPU tensors.
    # Torch port of the former tvm-ffi std::mismatch kernel (kernel/csrc/src/radix.cpp).
    n = min(x.shape[0], y.shape[0])
    if n == 0:
        return 0
    diff = (x[:n] != y[:n]).nonzero()
    return int(diff[0].item()) if diff.numel() > 0 else n
