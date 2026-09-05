"""`read()` the checkpoint instead of faulting it in through `mmap`.

--------------------------------------------------------------------------------------------------
WHY THIS EXISTS — a measured 4x tax that is invisible in every profile bucket

`safetensors.safe_open(..., framework="pt")` mmaps the shard and hands out ZERO-COPY tensors. That
makes `get_tensor` look free (the 48-layer TP=2 boot charges it 7.25 s for 77.84 GB, an apparent
10.0 GiB/s) and moves the real cost — the page faults — into whoever first READS the bytes, which
here is `_shard_qwen4_exp` (`ckpt.shard`, 44.1 s) and `sharded.to(device)` (`ckpt.h2d`, 116.7 s).
Those two buckets therefore read as "the shard is slow" and "the H2D is slow" when what is actually
happening is 66.2 GiB of file I/O at ZFS's mmap rate.

MEASURED ON THIS BOX, one cold 337.7 MiB shard per leg (`docs/measurements/BOOT_TIMELINE_2026-09-06/
zfs_readpath.txt`):

    read() 4 MiB buffered      2209.4 MiB/s     1,272 ARC demand hits
    O_DIRECT preadv 4 MiB      2978.8 MiB/s         5 ARC demand hits
    mmap, 16 MiB slices         550.9 MiB/s    91,855 ARC demand hits   <-- what ships
    mmap, 4 K byte-walk         596.9 MiB/s    86,654 ARC demand hits

86,400 pages in 337.7 MiB and ~91,855 ARC lookups: ZFS's mmap path takes **one ARC lookup per 4 KiB
page**, and it is that per-page walk — not the drive, and not the ARC miss rate, which is ~0 — that
costs the 4x. The NVMe does 4.9 GB/s and `read()` reaches 2.2 GB/s of it; mmap reaches 0.55 GB/s.

Each rank streams 196 shards / 66.2 GiB, so the tax is ~90 s PER RANK, both ranks concurrently,
against a 618 s boot. It is pure overhead: the same bytes, through a different syscall.

--------------------------------------------------------------------------------------------------
WHY A READER AND NOT A PREFETCH

The obvious cheap fix — read the file sequentially first so the mmap faults hit a warm cache — does
NOT work here, and the ARC counters above are why: the slow mmap leg took 91,855 ARC **hits** and
only 128 misses. It was already warm. The cost is the per-page fault/lookup itself, so the only fix
is to stop faulting, i.e. to stop mmapping.

--------------------------------------------------------------------------------------------------
WHY IT IS BIT-IDENTICAL, AND HOW THAT IS CHECKED RATHER THAN ASSERTED

This reader must be indistinguishable from `safe_open` in THREE ways, because the weight-offload
arena is layout-sensitive: a different tensor ORDER changes the carve order, which changes
`carve_digest()` and every region address, which on a quantized checkpoint means dequantizing one
expert against another's scale — plausible text, no crash (see `PinnedWeightArena.allocate`).

 1. the same bytes for every tensor,
 2. the same dtype and shape,
 3. **the same `keys()` ORDER.** `safe_open.keys()` returns the names SORTED, while the shard's own
    JSON header is not in sorted order for the expert files (checked: `layer-00000-experts-*` is
    unsorted, `model-bf16-*` happens to be sorted). Reproducing the header order would therefore
    have silently reordered 200k carves. `keys()` below sorts, and `tools/offload/ckpt_read_parity.py`
    asserts the list equality against the real `safe_open` rather than trusting this paragraph.

`tools/offload/ckpt_read_parity.py` compares EVERY tensor of a whole shard, as raw uint8, against
`safe_open` — that is the merge gate, and it runs on CPU with no GPU lease.

Unknown dtypes RAISE. A silent fallback to mmap would make the fast path opt-out-by-accident and
hide exactly the kind of drift this file's own docstring warns about.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

# The safetensors dtype names this checkpoint family actually uses, plus the rest of the spec's
# fixed-width set. Anything outside it raises: see the module docstring.
_DTYPES: Dict[str, str] = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
    "I16": "int16",
    "U16": "uint16",
    "F16": "float16",
    "BF16": "bfloat16",
    "I32": "int32",
    "U32": "uint32",
    "F32": "float32",
    "I64": "int64",
    "U64": "uint64",
    "F64": "float64",
}

# 16 MiB. Large enough that the syscall count is irrelevant (21 calls for a 337.7 MiB shard) and
# small enough that the copy stays in L3-friendly steps. The measured 2209 MiB/s leg used 4 MiB;
# nothing above ~1 MiB changed the rate.
READ_BLOCK = 16 << 20


class CheckpointReadError(RuntimeError):
    """The shard is not a safetensors file this reader can serve verbatim."""


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        attr = _DTYPES[name]
    except KeyError:
        raise CheckpointReadError(
            f"safetensors dtype {name!r} has no mapping in ckpt_read._DTYPES. Refusing rather than "
            f"falling back to mmap: a silent fallback would make the read() path opt-out-by-accident "
            f"and the 4x ZFS mmap tax would come back invisibly. Add the mapping and re-run "
            f"tools/offload/ckpt_read_parity.py."
        ) from None
    dt = getattr(torch, attr, None)
    if dt is None:
        raise CheckpointReadError(
            f"this torch build has no torch.{attr} for safetensors dtype {name!r}"
        )
    return dt


def read_file_bytes(path: str) -> bytearray:
    """The whole shard, via `read()`.

    A fresh `bytearray` per file on purpose. CPython services an allocation this large with its own
    `mmap`, so freeing it returns the pages to the OS immediately and there is no arena growth
    across 196 shards — and a REUSED buffer would be a correctness bug, because `get_tensor` hands
    out zero-copy views and `fold_buf` can legitimately hold one across the next `get_tensor`.

    `buffering=0` so Python's io layer does not add a second copy on top of the kernel's.
    """
    size = os.path.getsize(path)
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    with open(path, "rb", buffering=0) as fh:
        while got < size:
            n = fh.readinto(view[got:got + min(READ_BLOCK, size - got)])
            if not n:
                raise CheckpointReadError(
                    f"{path}: read() returned EOF after {got} of {size} bytes. A short checkpoint "
                    f"read would load a TRUNCATED tensor with no other symptom"
                )
            got += n
    view.release()
    return buf


class ReadSafeOpen:
    """A `safetensors.safe_open` work-alike whose bytes came from `read()`, not `mmap`.

    Implements exactly the surface `weight.py` uses — `keys()`, `get_tensor()`, and the context
    manager — and nothing else, so an unsupported call fails with `AttributeError` at the call site
    instead of quietly diverging from `safe_open`'s semantics somewhere subtler.
    """

    __slots__ = ("path", "_buf", "_mv", "_index", "_data0", "_keys", "metadata")

    def __init__(self, path: str, framework: str = "pt", device: str = "cpu") -> None:
        if framework != "pt" or device != "cpu":
            raise CheckpointReadError(
                f"ReadSafeOpen serves framework='pt', device='cpu' only (got {framework!r}, "
                f"{device!r}); the device= form of safe_open loads straight to VRAM and this "
                f"reader would silently change WHERE the tensor lands"
            )
        self.path = path
        self._buf = read_file_bytes(path)
        self._mv = memoryview(self._buf)
        if len(self._buf) < 8:
            raise CheckpointReadError(f"{path}: {len(self._buf)} bytes is not a safetensors file")
        hdr_len = struct.unpack_from("<Q", self._buf, 0)[0]
        if 8 + hdr_len > len(self._buf):
            raise CheckpointReadError(
                f"{path}: header claims {hdr_len} bytes but the file is {len(self._buf)}"
            )
        header = json.loads(bytes(self._mv[8:8 + hdr_len]))
        self.metadata: Optional[Dict[str, Any]] = header.get("__metadata__")
        self._index: Dict[str, Dict[str, Any]] = {
            k: v for k, v in header.items() if k != "__metadata__"
        }
        self._data0 = 8 + hdr_len
        # SORTED, matching safe_open.keys(). See the module docstring: the shard headers are not in
        # sorted order and the carve order is layout-critical.
        self._keys: List[str] = sorted(self._index)

    # -- safe_open surface ----------------------------------------------------

    def keys(self) -> List[str]:
        return list(self._keys)

    def get_tensor(self, name: str) -> Any:
        import torch

        try:
            spec = self._index[name]
        except KeyError:
            raise KeyError(f"{name!r} not in {self.path}") from None
        start, end = spec["data_offsets"]
        shape: Tuple[int, ...] = tuple(spec["shape"])
        dt = _torch_dtype(spec["dtype"])
        a, b = self._data0 + int(start), self._data0 + int(end)
        if b > len(self._buf) or a > b:
            raise CheckpointReadError(
                f"{self.path}: {name} data_offsets [{start}, {end}) fall outside the {len(self._buf)}"
                f" byte shard"
            )
        if b == a:
            # `torch.frombuffer` refuses a zero-length buffer. An empty tensor still has to carry the
            # right dtype/shape or `load_state_dict` reports a shape mismatch instead of the real
            # problem.
            return torch.empty(shape, dtype=dt)
        return torch.frombuffer(self._mv[a:b], dtype=dt).reshape(shape)

    def __enter__(self) -> "ReadSafeOpen":
        return self

    def __exit__(self, *exc: Any) -> bool:
        # Deliberately NOT freeing the buffer: `get_tensor` returns zero-copy views and the caller
        # may legitimately still hold one (`fold_buf` pairs a `weight_scale` with its
        # `weight_scale_2`). Python frees the bytearray when the last view dies, which is the same
        # lifetime rule mmap gave.
        return False

    def __len__(self) -> int:
        return len(self._keys)


def safe_open(path: str, framework: str = "pt", device: str = "cpu") -> ReadSafeOpen:
    """Drop-in for `safetensors.safe_open` on the read path. Same name on purpose."""
    return ReadSafeOpen(path, framework=framework, device=device)
