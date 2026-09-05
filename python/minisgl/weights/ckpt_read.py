"""O_DIRECT reader for safetensors shards, into ONE REUSED page-aligned buffer.

--------------------------------------------------------------------------------------------------
VERDICT, 2026-09-06, 48-layer TP=2, sequential, quiet box, both legs at floor=4 GiB, two separate
worktrees so neither leg could read the other's source
(docs/measurements/BOOT_TIMELINE_2026-09-06/r3/):

    boot                574.2 s  ->  254.6 s      (2.26x)
    stage_b             376.6 s  ->  111.1 s
    ckpt.h2d            194.0 s  ->   51.6 s
    stageb.post_load    106.1 s  ->    7.0 s      <- NOT a read; the box had been stopping it
    graph_capture        84.1 s  ->   33.6 s      <- NOT a read either
    swap-out         11.68 M pages -> 3.05 M   |  major faults 61.2 M -> 19.6 M

Bit-identical: digest a5e49a2e2b307ff3, 156 forecast / 156 actual regions, arena_pinned_bytes
25,895,632,896 x2, torch_fallbacks 0, seam_pointer_checked, 1236 keys, and the same greedy token ids
from both prompts on both ranks.

--------------------------------------------------------------------------------------------------
HISTORY, IN THE ORDER IT WAS MEASURED — this module shipped WRONG once, and the way it was wrong is
the whole design of what is here now.

2026-09-06 round 2 wired an O_DIRECT reader that allocated a FRESH whole-file buffer per shard. The
48-layer TP=2 A/B (docs/measurements/BOOT_TIMELINE_2026-09-06/r2/):

    boot           520.5 s  ->  1808.3 s     (3.5x WORSE)
    ckpt.h2d       174.7 s  ->  1382.1 s
    read total     ~297 s   ->    59.2 s     (5x BETTER -- 64.8 GiB at 1660 MiB/s)
    swap-out       6.18 M pages -> 44.58 M   |  major faults 46.9 M -> 132.8 M

The read got five times faster and the boot got three and a half times slower, because of WHERE the
bytes landed. `safe_open`'s pages are FILE-BACKED, so reclaiming one is free -- the kernel drops a
clean page. A fresh anonymous buffer per shard streams 64.8 GiB of ANONYMOUS memory through a box
whose two 24.12 GiB pinned arenas have already taken half of RAM, and reclaiming anonymous memory
here means COMPRESSING IT INTO ZRAM. The tensors handed to `.to(device)` live in those buffers,
which is why `ckpt.h2d` absorbed the entire regression.

That commit's own postmortem named the fix: ONE REUSED, pre-faulted buffer, so the anon footprint is
337 MiB for the whole boot instead of 64.8 GiB of churn. It did not land it because reuse needs a
guarantee that no `get_tensor` view outlives its shard, and that guarantee was not free.

THIS MODULE NOW HOLDS BOTH HALVES. `_SharedReadBuffer` is the single buffer; the guarantee is
MECHANICAL rather than argued -- see `_SharedReadBuffer.acquire`, which refuses to reuse a buffer
that still has a tensor view exported and raises instead of handing out another shard's bytes.

--------------------------------------------------------------------------------------------------
WHY THE READ PATH MATTERS AT ALL — measured on THIS pool, one cold 337.7 MiB expert shard per leg,
CPU only, quiet box (2026-09-06, four distinct cold shards so no leg warms another):

    reader                        rate          dCached      dARC
    mmap, 4 KiB walk          627.0 MiB/s     +0.33 GiB   +0.33 GiB   <-- what shipped
    read() into reused buf   4947.8 MiB/s     +0.00 GiB   +0.31 GiB
    O_DIRECT into reused buf 4967.0 MiB/s     +0.00 GiB   +0.00 GiB   <-- what this module does

The rate column is the smaller half. mmap costs TWICE the bytes in reclaimable footprint — a
page-cache page AND an ARC buffer per 4 KiB — while O_DIRECT into a buffer that already exists costs
NOTHING, and once the two 24.12 GiB pinned arenas have taken half of RAM, every one of those
allocations must reclaim, which here means a zram compression. That is why the boot's two BIGGEST
wins (`stageb.post_load`, `graph_capture`) are phases that never touch the checkpoint at all.

`posix_fadvise(POSIX_FADV_DONTNEED)` was tried first as the smaller, safer change — bound mmap's
footprint in place and touch nothing else. It returns 0 on this ZFS 2.4.3 mount and frees nothing,
so that option does not exist here. `tools/offload/zfs_readpath_probe.py` re-measures all of it.

--------------------------------------------------------------------------------------------------
WHAT WAS TRUE ABOUT THE ORIGINAL DIAGNOSIS (kept, because the rate numbers are still right)

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

# The O_DIRECT reader serves shards up to this size; `safe_open()` sends anything larger to
# safetensors' mmap, and its docstring carries the measurement that settled the boundary.
# 1.5 GiB covers all 192 `layer-*-experts-*.safetensors` shards (337.7 MiB each, 64.8 GiB of the
# checkpoint's 72.6 GiB) with one buffer per file, and keeps the 3.4-10.0 GiB `model-bf16-*` body
# shards off a whole-file allocation — a 10 GiB transient buffer would itself be the memory event
# this whole change exists to avoid.
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


class _SharedReadBuffer:
    """THE buffer. One page-aligned anonymous mapping, allocated once and refilled per shard.

    A fresh buffer per shard is what made round 2's O_DIRECT reader lose 3.5x (see the module
    docstring): 64.8 GiB of anonymous churn on a box that answers anonymous reclaim with zram
    compression. Reused, the checkpoint stream's anonymous footprint is ONE shard — 337.7 MiB for
    the whole boot — and it is faulted in exactly once.

    REUSE IS A CORRECTNESS HAZARD AND IS GATED MECHANICALLY. `get_tensor` hands out zero-copy views
    of this buffer; refilling it while one is alive would not crash, it would silently dequantize
    one expert against another expert's scale. So `acquire()` proves, before every refill, that the
    previous shard's views are all gone — by asking CPython to `resize()` the mapping, which raises
    `BufferError` while ANY buffer export (a `memoryview` slice, a `torch.frombuffer` storage) is
    outstanding. Same-size `resize` is an `mremap` no-op on Linux, so the check costs nothing and
    cannot be satisfied by inspection or by comment. A caller that retains a view across shards gets
    a loud `CheckpointReadError` naming the contract, not wrong weights.
    """

    __slots__ = ("_buf", "_cap", "gen")

    def __init__(self) -> None:
        self._buf: Any = None
        self._cap = 0
        # Bumped on every refill. `ReadSafeOpen` records the value it was read at and refuses to
        # serve a tensor once the buffer has moved on -- the second half of the guard, for the case
        # where the caller kept the HANDLE rather than a view.
        self.gen = 0

    def acquire(self, nbytes: int, path: str):
        """The buffer, sized >= `nbytes`, with no live view of the PREVIOUS shard in it."""
        import mmap as _mmap

        want = _align_up(max(nbytes, 1))
        if self._buf is not None:
            try:
                # Same size on purpose: this is an export CHECK, not a resize. It is also what
                # makes growth safe, below, since a grow is the same check plus a real mremap.
                self._buf.resize(self._cap)
            except BufferError:
                raise CheckpointReadError(
                    f"{path}: the shared checkpoint read buffer still has a tensor view exported "
                    f"from the PREVIOUS shard. Refilling it would silently replace that tensor's "
                    f"bytes with this shard's. Every `get_tensor` result must be consumed (or "
                    f"`.clone()`d) before the next shard is opened — see the reuse contract in "
                    f"`_SharedReadBuffer` and the `fold_buf` clone in `_load_qwen4_exp_weight`."
                ) from None
            if want > self._cap:
                self._buf.close()
                self._buf = None
        if self._buf is None:
            self._buf = _mmap.mmap(-1, want)
            self._cap = want
            # PRE-FAULT ONCE. O_DIRECT's `get_user_pages` faults its destination INSIDE the read
            # syscall; on an untouched mapping that turns the first read of every shard into a page
            # allocation under whatever memory pressure the box happens to be in — which is the
            # shape of the failure that made per-tensor O_DIRECT collapse to 53 MiB/s. One pass of
            # zeros here pays it once for the whole boot, in READ_BLOCK steps so the temporary is
            # 16 MiB rather than a second copy of the whole buffer.
            zeros = b"\0" * min(READ_BLOCK, want)
            done = 0
            while done < want:
                done += self._buf.write(zeros[: want - done])
            self._buf.seek(0)
        self.gen += 1
        return self._buf


_SHARED = _SharedReadBuffer()


def read_file_bytes(path: str):
    """The whole shard in the shared buffer, via O_DIRECT `preadv`.

    Returns `(buffer, generation)`; the buffer's first `os.path.getsize(path)` bytes are the file
    and are valid only until the next `read_file_bytes`/`ReadSafeOpen` call. See
    `_SharedReadBuffer`.
    """
    buf, _direct, gen = _read_file(path)
    return buf, gen


def _read_file(path: str):
    """`(buffer, direct, generation)` — the shared buffer refilled from `path`."""
    size = os.path.getsize(path)
    buf = _SHARED.acquire(size, path)
    mv = memoryview(buf)
    fd, direct = _open_direct(path)
    try:
        _pread_into(fd, mv, size, 0, path)
    finally:
        os.close(fd)
        mv.release()
    return buf, direct, _SHARED.gen


class ReadSafeOpen:
    """A `safetensors.safe_open` work-alike whose bytes came from O_DIRECT `preadv`, not `mmap`.

    Implements exactly the surface `weight.py` uses — `keys()`, `get_tensor()`, and the context
    manager — and nothing else, so an unsupported call fails with `AttributeError` at the call site
    instead of quietly diverging from `safe_open`'s semantics somewhere subtler.

    ONE BUFFER, ALL SHARDS. The whole file is read into the process-wide `_SharedReadBuffer` and
    tensors are zero-copy views of it, exactly as mmap gave -- but the buffer is REFILLED by the
    next shard, so those views are valid only until then. `_SharedReadBuffer.acquire` enforces that
    mechanically (it refuses to refill while a view is exported) and `get_tensor` refuses to serve a
    handle whose shard has already been overwritten, so a stale handle raises instead of returning
    another shard's bytes under this shard's names.

    `safe_open()` below, not this class, decides WHICH files come here: a 10 GiB body shard would
    make the shared buffer 10 GiB forever, which is the memory event this reader exists to avoid, so
    `ReadSafeOpen` REFUSES an oversized file rather than growing a second mode for it.
    """

    __slots__ = ("path", "_buf", "_mv", "_index", "_data0", "_keys", "metadata",
                 "_size", "_direct", "_gen")

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
        if self._size > WHOLE_FILE_CAP:
            raise CheckpointReadError(
                f"{path} is {self._size / (1 << 30):.2f} GiB, over the {WHOLE_FILE_CAP / (1 << 30):.2f}"
                f" GiB whole-file cap. Route it with ckpt_read.safe_open(), which sends oversized"
                f" shards to safetensors' mmap on purpose -- see that function for the measurement."
            )
        self._buf, self._direct, self._gen = _read_file(path)
        self._mv = memoryview(self._buf)
        head = self._mv
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

        # THE STALE-HANDLE FENCE. The buffer under `self._mv` belongs to whichever shard was read
        # LAST. Serving from a handle whose shard has been overwritten would return this shard's
        # NAME with another shard's BYTES — an expert dequantized against a different expert's
        # scale, which is plausible text and no crash. Refuse instead.
        if self._gen != _SHARED.gen:
            raise CheckpointReadError(
                f"{self.path}: get_tensor({name!r}) on a stale handle — the shared read buffer has "
                f"been refilled {_SHARED.gen - self._gen} time(s) since this shard was read. Open "
                f"one shard at a time, or route this call through safetensors' mmap."
            )
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
        return torch.frombuffer(self._mv[a:b], dtype=dt).reshape(shape)

    def __enter__(self) -> "ReadSafeOpen":
        return self

    def __exit__(self, *exc: Any) -> bool:
        # Drop THIS handle's own export of the shared buffer. It has to go, or the next shard's
        # `acquire()` would see an outstanding export that belongs to nobody and refuse to reuse.
        # Slices already handed to `torch.frombuffer` hold their own independent exports and are
        # NOT released here — that is exactly the condition `acquire()` is there to catch.
        # The read fd is already closed; `_read_file` closes it.
        self._mv.release()
        return False

    def __len__(self) -> int:
        return len(self._keys)


def safe_open(path: str, framework: str = "pt", device: str = "cpu") -> Any:
    """Drop-in for `safetensors.safe_open`, and the POLICY that decides which reader serves a shard.

    O_DIRECT for shards up to `WHOLE_FILE_CAP`; `safetensors.safe_open`'s mmap above it. That split
    is measured, not assumed, and it was measured the expensive way — by shipping the other answer
    and watching it lose.

    The first cut of this reader served large shards with ONE ALIGNED PREAD PER TENSOR, so that a
    10 GiB body shard never needed a 10 GiB buffer. It is byte-identical (the parity gate passes on
    it) and on an IDLE box it is only mildly slower than mmap — 601 vs 715 MiB/s on
    `model-bf16-00011.safetensors`, 0.84x. Inside the boot it collapses. Measured in the aborted
    2026-09-06 after-leg, at the point in Stage B where the two 24.12 GiB arenas are already pinned:
    the body chunk was still running after 13 MINUTES (it costs 15.6 s on the mmap path), reading
    9.6 MiB per `preadv` at **53 MiB/s**, both ranks pinned at 100 % of a core in userspace with the
    NVMe idle. The reason is that every one of those preads allocated a FRESH anonymous buffer, so
    O_DIRECT's `get_user_pages` had to fault ~12 MiB of new anon memory INSIDE the read syscall, on
    a box whose free memory the arena had just consumed. Per-tensor O_DIRECT converts a read into a
    page-allocation storm exactly when page allocation is the scarce thing.

    Whole-file O_DIRECT does not have that shape: one 337.7 MiB buffer amortises over 1536 tensors,
    and it measures 2.0-2.6x mmap on this pool with 5 ARC demand hits instead of 91,855. So the cap
    is what keeps the fast path on the 192 expert shards — 64.8 GiB of the checkpoint's 72.6 GiB —
    and leaves the 4 `model-bf16-*` body shards on exactly the reader they had before. A file over
    the cap is not a fallback-by-accident: `ReadSafeOpen` raises on one, and only this function is
    allowed to route around it.

    THE CAP IS ALSO A MEMORY BOUND, not just a routing rule: `_SharedReadBuffer` grows to the
    largest shard it is ever handed and never shrinks, so routing one 10 GiB body shard here would
    cost 10 GiB of resident anonymous memory for the rest of the process — per rank — which is the
    exact quantity this reader exists to keep at zero. On this checkpoint the buffer settles at
    337.7 MiB.
    """
    if os.path.getsize(path) <= WHOLE_FILE_CAP:
        return ReadSafeOpen(path, framework=framework, device=device)
    import safetensors

    return safetensors.safe_open(path, framework=framework, device=device)
