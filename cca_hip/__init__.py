# CCA fused HIP custom op package.
"""cca_hip — native fused HIP kernels for the ZAYA1 Compressed Convolutional Attention (CCA) path
on gfx1201 (RDNA4). Vendored from vllm-gfx1201/zaya/cca_hip (single source of truth for the kernel).

AOT-compiled once (no Triton), it registers torch.ops.zaya_cca.{conv_state_decode, cca_decode_qk,
cca_prefill_qk}. Importing this loads the .so + the fake/meta impls (via .cca_op)."""
from .cca_op import (  # noqa: F401
    cca_decode_qk,
    cca_prefill_qk,
    conv_state_decode,
)
