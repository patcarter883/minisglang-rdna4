"""QSA — query-sparse attention for Qwen4-Exp, native HIP, Triton-free.

    config.py    the five checkpoint numbers, parsed once, plus the page-alignment requirement
    cache.py     the r-slot-per-request raw-key ring + the compressed-key cache (DSV4 addressing)
    ops.py       the `qsa_index` HIP selection kernels (score / top-k / expand), torch behind a flag
    runtime.py   the batch-derived plan and the four-stage per-layer selection driver

The ATTENTION half is deliberately not here: a sparse attention is not a new kernel, it is the
existing paged attention core visiting a different set of rows. `attention/hip.py::forward_sparse`
calls `attn_decode.flash_decode_paged` with the SELECTED physical slots as its block table at
page_size 1 — same kernel, same accumulation order, same code. See that method's docstring.
"""

from .cache import QSAIndexCache
from .config import QSAProfile, parse_qsa_profile
from .runtime import QSAPlan, QSARuntime, QSASelection

__all__ = [
    "QSAIndexCache",
    "QSAPlan",
    "QSAProfile",
    "QSARuntime",
    "QSASelection",
    "parse_qsa_profile",
]
