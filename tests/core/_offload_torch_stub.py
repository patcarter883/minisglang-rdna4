"""Make the weight-offload PLANNING tests runnable on a machine with no working torch.

NOT a conftest, and not a mock of anything under test. Imported explicitly by the two planning test
modules, before they import `minisgl.weights`, so its blast radius is those two files.

WHY IT EXISTS. The planning half of weight offload is pure integer arithmetic -- byte models,
capacity inequalities, the f-sweep -- and it decides whether a serve boots at all. It must be
testable without a GPU, and on this box also without torch: the host's torch fails to load
(`libmpi_cxx.so.40`) and the engine runs in a container. Refusing to test the capacity arithmetic
until someone starts a container is how a capacity bug ships.

WHAT IT DOES. `minisgl.weights.placement` reaches torch only through `granule.py`'s module-scope
`import torch` and `minisgl.kvcache.host_arena`. Nothing in the placement or capacity path CALLS
torch. So when the real torch cannot be imported we install a module that satisfies those imports
and nothing more -- `torch.dtype`, `torch.Tensor`, `torch.device` and the scalar dtypes, all
referenced but not exercised at import time.

WHEN IT DOES NOTHING. If the real torch imports, the stub is never installed and the same tests run
against the real chain. So a green run in the container is a real run, and this file is a
portability shim rather than a fixture. `TORCH_IS_REAL` says which happened; any test needing real
tensor behaviour must branch on it (or live in a file that does not import this).

CAVEAT. `sys.modules` is process-global, so on a torch-broken host the stub is visible to every
other test collected in the same session. That is harmless in both realities -- on such a host those
tests could not have run anyway, and on a working host the stub is never installed -- but it is why
this is opt-in per file rather than a `conftest.py`.
"""

from __future__ import annotations

import sys
import types


def _install() -> bool:
    """Return True if a stub was installed (i.e. the real torch is unavailable)."""
    try:
        import torch  # noqa: F401

        return False
    except Exception:
        pass

    stub = types.ModuleType("torch")
    # `importlib.import_module` insists on a spec; a hand-built ModuleType has none.
    stub.__spec__ = types.SimpleNamespace(name="torch", loader=None, origin="offload-test-stub")

    class _DType:
        __slots__ = ("name", "itemsize")

        def __init__(self, name: str, itemsize: int) -> None:
            self.name = name
            self.itemsize = itemsize

        def __repr__(self) -> str:  # pragma: no cover - diagnostics only
            return f"torch.{self.name}"

    class _Tensor:  # the isinstance target in the walkers; never instantiated here
        pass

    class _Device:
        def __init__(self, *a, **k) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    stub.dtype = _DType
    stub.Tensor = _Tensor
    stub.device = _Device
    for name, size in (
        ("float16", 2), ("bfloat16", 2), ("float32", 4), ("float64", 8), ("int32", 4),
        ("int64", 8), ("int16", 2), ("int8", 1), ("uint8", 1), ("bool", 1),
        ("float8_e4m3fn", 1),
    ):
        setattr(stub, name, _DType(name, size))
    sys.modules["torch"] = stub
    return True


TORCH_IS_REAL = not _install()
