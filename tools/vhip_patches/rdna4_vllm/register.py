# SPDX-License-Identifier: Apache-2.0
"""Entry point for the `rdna4_vllm` plugin (vllm.general_plugins).

Runs from vllm/v1/worker/worker_base.py:247 — before the model is built and before Sampler is
constructed, which is what makes the PluggableLayer/CustomOp swaps take effect.

Each sub-wiring is independently switchable so a regression can be bisected without unloading the
whole plugin, but NONE of them fall back to Triton at runtime: a disabled switch means vLLM's stock
path, an ENABLED switch means our kernel or a loud error.
"""
from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_REGISTERED = False


def _on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) == "1"


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    from vllm.platforms import current_platform

    try:
        gfx = current_platform.get_device_capability()
    except Exception:
        gfx = None

    enabled = []

    if _on("VLLM_RDNA4_LMHEAD"):
        from rdna4_vllm import lmhead  # noqa: F401  (registers via decorator on import)

        enabled.append("lmhead(dense_bf16_gemv/dense_gemm)")

    # print(), not logger: vLLM configures handlers for the "vllm.*" logger hierarchy only, so an
    # init_logger("rdna4_vllm.register") record propagates to a handler-less root and is DROPPED.
    # Every other plugin in this image (gdn_hip/tail_hip/w4a8) prints for the same reason.
    if enabled:
        print(f"[rdna4_vllm] wired: {', '.join(enabled)}", flush=True)
    else:
        print("[rdna4_vllm] loaded but every sub-wiring is DISABLED", flush=True)
