"""CAM (editable memory) serving path for minisgl.

Folds the memory-organ CAM "product-key store + trained tap at layer 24 + per-token gate router"
editing mechanism into the minisgl serve engine (Qwen3.5-4B). The public surface is `CAMMemory`
(see `memory.py`), constructed once at engine start from a checkpoint directory produced by the
memory-organ export (WS-B). The Qwen3.5 model's decoder-layer loop stages a per-request bank and
applies the tap after `tap_layer` — a byte-exact no-op when nothing is staged.

CPU-importable: importing this package pulls in only torch; it does not require a checkpoint or a GPU.
"""
from .memory import CAMMemory
from .runtime import CAMRuntime, get_cam_runtime

__all__ = ["CAMMemory", "CAMRuntime", "get_cam_runtime"]
