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

# O_DIRECT alignment. 4096 covers this box's NVMe logical block size and ZFS's requirement; the
# buffer comes from an anonymous `mmap`, which is page-aligned by construction (a `bytearray` is
# NOT: CPython's allocator gives no alignment guarantee, and an unaligned O_DIRECT buffer fails
# with EINVAL, which would look like "this filesystem does not support O_DIRECT").
DIO_ALIGN = 4096

# Whole-file O_DIRECT is used up to this size; above it the reader preads ONE TENSOR AT A TIME.
# 1.5 GiB covers all 192 `layer-*-experts-*.safetensors` shards (337.7 MiB each, 64.8 GiB of the
# checkpoint's 72.6 GiB) with one buffer per file, and keeps the 3.4-10.0 GiB `model-bf16-*` body
# shards off a whole-file allocation — a 10 GiB transient buffer would itself be the memory event
# this whole change exists to avoid. Body tensors average ~12 MiB (821 keys in a 10 GiB shard), so
# per-tensor O_DIRECT is large-block there anyway.
WHOLE_FILE_CAP = 3 << 29


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


def _align_down(x: int, a: int = DIO_ALIGN) -> int:
    return x - (x % a)


def _align_up(x: int, a: int = DIO_ALIGN) -> int:
    return -(-x // a) * a


def _open_direct(path: str) -> Tuple[int, bool]:
    """`(fd, direct)`. Falls back to a buffered fd when O_DIRECT is refused.

    The fallback is NOT silent-by-design the way a fallback to mmap would be: buffered `read()` is
    still 4x mmap on this box (2209 vs 551 MiB/s) and still writes into a buffer this reader owns.
    What it loses is only the page-cache/ARC bypass. `ReadSafeOpen.direct` records which one ran so
    a boot report can say so.
    """
    flags = os.O_RDONLY
    odirect = getattr(os, "O_DIRECT", 0)
    if odirect:
        try:
            return os.open(path, flags | odirect), True
        except OSError:
            pass
    return os.open(path, flags), False


def _pread_into(fd: int, mv: memoryview, want: int, file_off: int, path: str) -> None:
    """Fill `mv[:want]` from `file_off`, in READ_BLOCK steps. `mv` must be page-aligned for O_DIRECT.

    O_DIRECT requires the OFFSET, the LENGTH and the BUFFER address to be block-aligned, so both the
    offset and the per-call length are aligned here and the tail of the last call is allowed to run
    past EOF: the kernel then returns a short count, which is not an error and is why the loop
    tolerates `got + n > want`. A count of ZERO before `want` is a truncated shard and raises —
    a short checkpoint read loads a truncated tensor with no other symptom.
    """
    got = 0
    while got < want:
        n = os.preadv(fd, [mv[got:got + min(READ_BLOCK, len(mv) - got)]], file_off + got)
        got_next = got + n
        # A short read that is NOT at EOF would leave `got` unaligned, and the NEXT O_DIRECT call
        # would then fail EINVAL on the buffer address — which reads as "this mount does not support
        # O_DIRECT" rather than as what it is. Refuse loudly instead. (The final call is allowed to
        # come up short: it is the one whose block-aligned length runs past EOF.)
        if got_next < want and n % DIO_ALIGN:
            raise CheckpointReadError(
                f"{path}: short read of {n} B (not a multiple of {DIO_ALIGN}) at offset "
                f"{file_off + got}, {got_next} of {want} B done. Continuing would misalign every "
                f"subsequent O_DIRECT read of this shard"
            )
        if n == 0:
            raise CheckpointReadError(
                f"{path}: read returned EOF after {got} of {want} bytes at offset {file_off}. "
                f"A short checkpoint read would load a TRUNCATED tensor with no other symptom"
            )
        got += n


def _aligned_buffer(nbytes: int):
    """A page-aligned, page-granular anonymous buffer, freed to the OS the moment it is dropped.

    `mmap.mmap(-1, n)` rather than `bytearray(n)`: O_DIRECT needs the ADDRESS aligned, and CPython
    gives no alignment guarantee for a bytearray. A fresh buffer per file/tensor on purpose — a
    REUSED one would be a correctness bug, because `get_tensor` hands out zero-copy views and
    `fold_buf` can legitimately hold one across the next `get_tensor`.
    """
    import mmap as _mmap

    return _mmap.mmap(-1, _align_up(max(nbytes, 1)))


def read_file_bytes(path: str):
    """The whole shard, via O_DIRECT `preadv` into one page-aligned anonymous buffer.

    Returns the buffer (an anonymous `mmap`); its first `os.path.getsize(path)` bytes are the file.

    O_DIRECT, not buffered `read()`, and neither is `mmap` — measured on this box's ZFS pool, one
    cold 337.7 MiB shard per leg (`docs/measurements/BOOT_TIMELINE_2026-09-06/zfs_readpath.txt`):

        O_DIRECT preadv 4 MiB      2978.8 MiB/s         5 ARC demand hits
        read() 4 MiB buffered      2209.4 MiB/s     1,272 ARC demand hits
        mmap, 16 MiB slices         550.9 MiB/s    91,855 ARC demand hits   <-- what shipped

    The RATE is only half the reason. The other half is the ARC/page-cache column, and on a
    48-layer TP=2 boot it is the half that dominates: by the time Stage B starts streaming, pinning
    the two 24.12 GiB arenas has driven the box's page cache from 23 GiB to 0.7 GiB and the ZFS ARC
    from 15.8 GiB to 2.9 GiB, with ~21 GiB pushed into zram. Every mmap fault after that point must
    ALLOCATE a page-cache page, which means reclaiming one, which means a zram compression — so the
    checkpoint read is paying box reclaim per 4 KiB. O_DIRECT reads into a buffer that already
    exists and caches nothing, so it takes no page-cache page and grows no ARC.
    """
    return _read_file(path)[0]


def _read_file(path: str):
    """`(buffer, direct)` — `read_file_bytes` plus whether O_DIRECT was granted."""
    size = os.path.getsize(path)
    buf = _aligned_buffer(size)
    mv = memoryview(buf)
    fd, direct = _open_direct(path)
    try:
        _pread_into(fd, mv, size, 0, path)
    finally:
        os.close(fd)
        mv.release()
    return buf, direct


class ReadSafeOpen:
    """A `safetensors.safe_open` work-alike whose bytes came from O_DIRECT `preadv`, not `mmap`.

    Implements exactly the surface `weight.py` uses — `keys()`, `get_tensor()`, and the context
    manager — and nothing else, so an unsupported call fails with `AttributeError` at the call site
    instead of quietly diverging from `safe_open`'s semantics somewhere subtler.

    TWO MODES, one rule: never hold more than one shard's worth of transient buffer.
      * `size <= WHOLE_FILE_CAP` (every expert shard): one aligned buffer, tensors are zero-copy
        views of it, exactly as mmap gave.
      * larger (the `model-bf16-*` body shards, up to 10 GiB): the header only, then ONE ALIGNED
        PREAD PER TENSOR into that tensor's own buffer. Same bytes, same order; the difference is
        that peak transient memory is one tensor (~12 MiB average here) instead of 10 GiB.
    """

    __slots__ = (
        "path", "_buf", "_mv", "_index", "_data0", "_keys", "metadata",
        "_size", "_fd", "_direct", "_whole",
    )

    def __init__(self, path: str, framework: str = "pt", device: str = "cpu") -> None:
        if framework != "pt" or device != "cpu":
            raise CheckpointReadError(
                f"ReadSafeOpen serves framework='pt', device='cpu' only (got {framework!r}, "
                f"{device!r}); the device= form of safe_open loads straight to VRAM and this "
                f"reader would silently change WHERE the tensor lands"
            )
        self.path = path
        self._size = os.path.getsize(path)
        if self._size < 8:
            raise CheckpointReadError(f"{path}: {self._size} bytes is not a safetensors file")
        self._whole = self._size <= WHOLE_FILE_CAP
        self._fd = -1
        self._direct = False
        if self._whole:
            self._buf, self._direct = _read_file(path)
            self._mv = memoryview(self._buf)
            head = self._mv
        else:
            # Header first, then keep the fd for per-tensor preads. The header read is aligned up to
            # a block like any other O_DIRECT read; 1 MiB comfortably covers the largest header in
            # this checkpoint family (821 keys ~= 190 KiB) and is re-read exactly if it does not.
            self._fd, self._direct = _open_direct(path)
            probe = min(_align_up(1 << 20), _align_up(self._size))
            hb = _aligned_buffer(probe)
            hmv = memoryview(hb)
            _pread_into(self._fd, hmv, min(probe, self._size), 0, path)
            need = 8 + struct.unpack_from("<Q", hmv, 0)[0]
            if need > probe:
                hmv.release()
                hb = _aligned_buffer(need)
                hmv = memoryview(hb)
                _pread_into(self._fd, hmv, min(_align_up(need), self._size), 0, path)
            self._buf = hb
            self._mv = hmv
            head = hmv
        hdr_len = struct.unpack_from("<Q", head, 0)[0]
        if 8 + hdr_len > self._size:
            raise CheckpointReadError(
                f"{path}: header claims {hdr_len} bytes but the file is {self._size}"
            )
        header = json.loads(bytes(head[8:8 + hdr_len]))
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

    @property
    def direct(self) -> bool:
        """Whether O_DIRECT was actually granted (whole-file mode reports its own open)."""
        return self._direct

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
        if b > self._size or a > b:
            raise CheckpointReadError(
                f"{self.path}: {name} data_offsets [{start}, {end}) fall outside the {self._size}"
                f" byte shard"
            )
        if b == a:
            # `torch.frombuffer` refuses a zero-length buffer. An empty tensor still has to carry the
            # right dtype/shape or `load_state_dict` reports a shape mismatch instead of the real
            # problem.
            return torch.empty(shape, dtype=dt)
        if self._whole:
            return torch.frombuffer(self._mv[a:b], dtype=dt).reshape(shape)
        # Per-tensor O_DIRECT read-around: the tensor's byte range rarely starts on a block, so read
        # from the block BELOW it and index into the result. The buffer is kept alive by the
        # memoryview `torch.frombuffer` holds, exactly as the whole-file mode's is.
        lo = _align_down(a)
        want = b - lo
        buf = _aligned_buffer(want)
        mv = memoryview(buf)
        _pread_into(self._fd, mv, want, lo, self.path)
        return torch.frombuffer(mv[a - lo:b - lo], dtype=dt).reshape(shape)

    def __enter__(self) -> "ReadSafeOpen":
        return self

    def __exit__(self, *exc: Any) -> bool:
        # The fd is closed here (nothing outlives it: in per-tensor mode each tensor owns its own
        # buffer already). The BUFFERS are deliberately NOT freed: `get_tensor` returns zero-copy
        # views and the caller may legitimately still hold one (`fold_buf` pairs a `weight_scale`
        # with its `weight_scale_2`). Python frees each buffer when the last view dies, which is the
        # same lifetime rule mmap gave.
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        return False

    def __len__(self) -> int:
        return len(self._keys)


def safe_open(path: str, framework: str = "pt", device: str = "cpu") -> ReadSafeOpen:
    """Drop-in for `safetensors.safe_open` on the read path. Same name on purpose."""
    return ReadSafeOpen(path, framework=framework, device=device)
