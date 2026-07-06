"""One-time engagement log for the custom HIP kernels.

Verifies that each custom HIP kernel actually FIRES on the serve path — and, critically, keeps
firing under cudagraph capture (a graph-unsafe kernel would fail capture or silently fall back to a
non-HIP reference). Each kernel calls ``engaged(name)`` at its entry; the first call per name emits
``[hip-engage] <name>`` to the rank-0 log. Under cudagraph the entry runs during CAPTURE, so a
missing line after capture = that kernel did not run (fell back / gated off). Silence with
``MINISGL_HIP_ENGAGE_LOG=0``.
"""
from __future__ import annotations

import os

from minisgl.utils import init_logger

_logger = init_logger("hip-engage")
_seen: set[str] = set()
_ON = os.environ.get("MINISGL_HIP_ENGAGE_LOG", "1") != "0"


def engaged(name: str) -> None:
    if _ON and name not in _seen:
        _seen.add(name)
        _logger.info_rank0(f"[hip-engage] {name}")
