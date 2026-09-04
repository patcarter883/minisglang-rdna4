"""Qwen4-Exp PLE (per-layer n-gram embedding) block — bring-up tranche 1b.

Four pieces, deliberately separate because they fail in different places:

  `hashing.py`  recent token ids -> per-head hashes. Pure numpy, torch-free, VERIFIED against
                `transformers/models/qwen4_exp/modeling_qwen4_exp.py` and against the shipping
                checkpoint's own `layer_multipliers` / `ngram_heads_vocab_sizes` tensors.
  `state.py`    per-sequence recurrent state: the 9-wide dilated-conv window (device) and the
                2-token n-gram history (host), both indexed by the GDN slot id.
  `source.py`   the NVMe row gather + the single H2D into a static staging buffer. The table
                itself lives in `weights/row_table.py` and is NOT re-implemented here.
  `runtime.py`  what the scheduler stages per batch and what the layer reads (`Context.ple`).

The device arithmetic — projections, grouped norms, gate, dilated depthwise short conv — is in
`models/qwen4exp.py::Qwen4ExpPLE`, next to the parameters it uses.

`hashing.py` stays importable WITHOUT torch — deliberately. It is pure numpy, it is the piece whose
correctness has to be checkable anywhere (including a host where torch will not load), and pulling
the torch-backed submodules in at package import would take that away. Hence the lazy `__getattr__`
below: `from minisgl.ple.hashing import ...` and `from minisgl.ple import Qwen4ExpNGramHasher` both
work with no torch, while `from minisgl.ple import PLERuntime` imports it on demand.
"""
from .hashing import (
    DEFAULT_NGRAM_SEED,
    Qwen4ExpNGramHasher,
    build_head_vocab_sizes,
    build_layer_multipliers,
    shift_right_ignore_eos,
)

_LAZY = {
    "PLEBatch": ".runtime",
    "PLERuntime": ".runtime",
    "PLE_TABLE_GLOB": ".runtime",
    "build_ple_runtime": ".runtime",
    "ple_device_bytes": ".runtime",
    "ENV_PLE_FILES": ".source",
    "ENV_PLE_META_FILES": ".source",
    "PLEEmbeddingSource": ".source",
    "PLEStateCache": ".state",
    "PLEGraphCapture": ".graph_capture",
}


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(mod, __name__), name)

__all__ = [
    "DEFAULT_NGRAM_SEED",
    "ENV_PLE_FILES",
    "ENV_PLE_META_FILES",
    "PLEBatch",
    "PLEEmbeddingSource",
    "PLEGraphCapture",
    "PLERuntime",
    "PLEStateCache",
    "PLE_TABLE_GLOB",
    "Qwen4ExpNGramHasher",
    "build_head_vocab_sizes",
    "build_layer_multipliers",
    "build_ple_runtime",
    "ple_device_bytes",
    "shift_right_ignore_eos",
]
