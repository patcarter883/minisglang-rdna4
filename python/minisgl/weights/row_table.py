"""NVMe-resident row-gather tables: enormous embedding tables that are LOOKED UP, never multiplied.

## Why this is a different problem from weight offload

The rest of `minisgl/weights/` moves *expert weights* — a GEMM reads them at full bandwidth every
step, so they must be pinned host memory the GPU can stream from. A PLE / n-gram table is the
opposite shape of problem:

  * it is a pure **gather**. A hash of the recent tokens picks a handful of rows and those rows are
    added to the hidden state. No matmul, so no bandwidth wall;
  * the rows are tiny and few relative to the table. Qwen3.8-Flash-Next's n-gram embedding is
    **320,001,536 rows x 160 B = 51.2 GB**, and one token reads **16 rows = 2,560 B**, once per
    forward pass. At 36 tok/s that is ~90 KB/s of random reads — orders of magnitude below what any
    NVMe sustains.

So the right home for such a table is **the page cache over an mmap'd file**: no pinned memory, no
VRAM, nothing competing with the expert arena, and the kernel keeps the hot rows resident for free.

`RadixArk/Qwen3.8-Flash-Next-NVFP4` is built for exactly this — the table is isolated into
`model-plefp8-00000..00009.safetensors` (51.20 GB) with the other 84.00 GB of the model in separate
files, so these ten files can be mmap'd and *nothing else in them* is paid for. (A checkpoint that
interleaves the n-gram shards with MoE weights, as `tcclaviger/...-MXFP4-FP8` does, defeats that.)

**Not** what vLLM upstream does: `VLLM_PLE_CPU_OFFLOAD=1` holds the whole table in host RAM (>=51 GB)
and is CUDA-only. On a 96 GB box that collides head-on with the pinned expert arena.

## What this deliberately does not need

No `hipMemUnmap`/`hipMemMap`, no fixed-VA repointing, no residency management. That matters: dynamic
residency is **dead on this box** (P6 — a remap at a live VA silently serves the stale physical page
and returns `hipSuccess`). A gather from an mmap into a staging buffer never touches that machinery.

The module is model-agnostic: it knows *files, byte offsets, a row stride, a row count, a codec*.
The model layer supplies row indices; `NgramHeads` below is the only Qwen-shaped piece and is just
arithmetic over two tensors the checkpoint ships.
"""
from __future__ import annotations

import json
import mmap
import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# fp8 e4m3 (OCP "fn": no infinities, NaN == 0x7F/0xFF)
# ---------------------------------------------------------------------------


def _build_e4m3_lut() -> np.ndarray:
    """256-entry uint8 -> float32 decode table.

    Built arithmetically rather than by reinterpreting bits, so the module stays torch-free and
    importable on a host where torch will not load. `test_row_table.py` checks it against
    `torch.float8_e4m3fn` element-for-element.
    """
    b = np.arange(256, dtype=np.uint32)
    sign = np.where((b >> 7) & 1, -1.0, 1.0).astype(np.float64)
    exp = ((b >> 3) & 0xF).astype(np.int32)
    mant = (b & 0x7).astype(np.float64)
    # exp == 0 is subnormal: value = mant * 2^-9 (i.e. 2^(1-bias) * mant/8, bias 7)
    sub = mant * (2.0**-9)
    nrm = (1.0 + mant / 8.0) * np.power(2.0, exp.astype(np.float64) - 7.0)
    out = sign * np.where(exp == 0, sub, nrm)
    out[(exp == 0xF) & (mant == 0x7)] = np.nan  # the only NaN encodings: 0x7F / 0xFF
    return out.astype(np.float32)


_E4M3_LUT = _build_e4m3_lut()


def dequant_f8_e4m3(raw: np.ndarray, scale: float) -> np.ndarray:
    """(n, row_bytes) uint8 -> (n, row_bytes) float32, one byte per value."""
    return _E4M3_LUT[raw] * np.float32(scale)


#: Width of every safetensors dtype this module may have to size — including the small I64 metadata
#: tensors, which are NOT valid row-table element types.
DTYPE_BYTES: Dict[str, int] = {
    "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
    "F64": 8, "I64": 8, "U64": 8,
}

#: The subset a row table may be built from — a dtype is only safe here if one value is one
#: independently addressable unit, so a row can be sliced without decoding its neighbours.
CODECS: Dict[str, int] = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4}


# ---------------------------------------------------------------------------
# safetensors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorLoc:
    path: str
    offset: int  # absolute byte offset of element 0 in `path`
    shape: List[int]
    dtype: str

    @property
    def nbytes(self) -> int:
        try:
            n = DTYPE_BYTES[self.dtype]
        except KeyError:
            raise ValueError(f"unknown safetensors dtype {self.dtype!r} for {self.path}") from None
        for d in self.shape:
            n *= d
        return n


def read_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header dict, absolute offset of the data section)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
    return hdr, 8 + hlen


def index_safetensors(paths: Sequence[str]) -> Dict[str, TensorLoc]:
    """name -> TensorLoc across several shard files. Reads headers only, never weights."""
    out: Dict[str, TensorLoc] = {}
    for p in paths:
        hdr, data_start = read_safetensors_header(p)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            beg, _end = meta["data_offsets"]
            out[name] = TensorLoc(
                path=p, offset=data_start + beg, shape=list(meta["shape"]), dtype=meta["dtype"]
            )
    return out


def read_tensor(loc: TensorLoc) -> np.ndarray:
    """Materialise a SMALL tensor (scales, offsets, vocab sizes). Never use on the table itself."""
    np_dtype = {"I64": "<i8", "I32": "<i4", "F32": "<f4", "F16": "<f2", "BF16": "<u2"}.get(loc.dtype)
    if np_dtype is None:
        raise ValueError(f"read_tensor: unsupported dtype {loc.dtype} for {loc.path}")
    with open(loc.path, "rb") as f:
        f.seek(loc.offset)
        buf = f.read(loc.nbytes)
    arr = np.frombuffer(buf, dtype=np_dtype)
    if loc.dtype == "BF16":  # widen to f32: bf16 is the high half of an f32
        w = np.zeros((arr.size, 2), dtype=np.uint16)
        w[:, 1] = arr
        arr = w.reshape(-1).view("<f4")
    return arr.reshape(loc.shape)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


class ShardedRowTable:
    """One logical [total_rows, row_elems] table spread over many equal-height shard tensors.

    The checkpoint splits the n-gram embedding into 128 tensors of [2,500,012, 160] because a single
    51 GB tensor is unwieldy, not because the shards mean anything — row `r` is shard `r // H`, local
    row `r % H`. Shards may share a file; each distinct file is mmap'd once.

    Holds no pinned and no device memory. `MAP_NORESERVE` keeps a 51 GB mapping off the commit
    charge; `MADV_RANDOM` stops the kernel doing 128 KB readahead for every 160-byte row, which would
    turn ~90 KB/s of real demand into ~75 MB/s of wasted I/O.
    """

    def __init__(self, shards: Sequence[TensorLoc], *, scale: float = 1.0,
                 advise_random: bool = True, shard_ids: Sequence[int] | None = None,
                 workers: int = 0, auto_prefetch: bool = True,
                 small_gather: str = "mmap") -> None:
        if not shards:
            raise ValueError("no shards")
        row_elems = shards[0].shape[1]
        height = shards[0].shape[0]
        for i, s in enumerate(shards):
            if s.shape[1] != row_elems:
                raise ValueError(f"shard {i}: row length {s.shape[1]} != {row_elems}")
            if s.shape[0] != height:
                # Uniform height is what makes row -> shard a division instead of a search.
                raise ValueError(
                    f"shard {i}: height {s.shape[0]} != {height}. Non-uniform shard heights need a "
                    f"cumulative-offset lookup; this class assumes uniform."
                )
            if s.dtype != shards[0].dtype:
                raise ValueError(f"shard {i}: dtype {s.dtype} != {shards[0].dtype}")

        self.shards = list(shards)
        self.dtype = shards[0].dtype
        self.row_elems = int(row_elems)
        self.rows_per_shard = int(height)
        self.row_bytes = self.row_elems * CODECS[self.dtype]
        self.scale = float(scale)
        self.workers = int(workers)
        #: Threaded gathers from this many rows up. WAS 256, on the reading that "the thread pool
        #: costs more than the faults it overlaps" for a 16-row decode step (609 us prefetch+mmap vs
        #: 802 us threaded). That was true of the pool as it was THEN BUILT: `_gather_raw_threaded`
        #: created and joined a fresh ThreadPoolExecutor on EVERY call, so the setup it was being
        #: charged for was an artifact, not a property of the mechanism. `_pool` now keeps one.
        #: RE-MEASURED 2026-09-18 on the real 51.2 GB table, arms INTERLEAVED with fresh random ids
        #: per arm per iteration (running each arm's block back-to-back warms the ARC monotonically
        #: and hands the win to whoever went last -- that error made one probe report serial pread
        #: 8x faster than the same probe had minutes earlier). Medians, us:
        #:
        #:   rows   mmap+prefetch   serial pread   threaded      dCached over the block
        #:     16       529             74            323        mmap accumulates, both preads ~0
        #:     64      2305            245            648
        #:    256      7136           5150           2029        mmap +55.6 MiB, pread +0.1 MiB
        #:
        #: and on a COLD table (first touch, nothing resident) the ordering differs: mmap 1135,
        #: threaded 1115, serial pread 2231. Serial pread's median is the best of the three when the
        #: rows are ARC-resident and the WORST when they are not -- its p90 reaches 23 ms at 256 rows
        #: -- so it is not the default: with an 8 GiB ARC over a 49 GB table most n-gram rows miss.
        #: Threaded is >= mmap in BOTH regimes and buffers nothing in the page cache, so the
        #: threshold drops to the decode step's own width. Below 16 is unmeasured.
        self.threaded_min_rows = 16
        #: Mechanism for a SUB-THRESHOLD gather: "mmap" (fancy-index the mapped view, with
        #: MADV_WILLNEED when auto_prefetch is on) or "pread" (serial os.pread, no page-cache copy
        #: -- see `_gather_raw_pread`). Both return identical bytes; they differ in latency and in
        #: how much of the box they leave buffered.
        self.small_gather = str(small_gather)
        if self.small_gather not in ("mmap", "pread"):
            raise ValueError(f"small_gather must be 'mmap' or 'pread', got {small_gather!r}")
        #: Issue MADV_WILLNEED before a non-threaded gather. Advisory, so it can never return wrong
        #: data; on an already-resident range it is a cheap no-op syscall with no I/O.
        self.auto_prefetch = bool(auto_prefetch)

        # Global row ids are defined by the shard's INDEX IN THE CHECKPOINT, not its position in
        # this list. The checkpoint assigns shards to files in string order — `model-plefp8-00000`
        # holds shard_0, shard_1, shard_10..shard_19, NOT shard_0..shard_12 — so a partially loaded
        # table (validation, or a lazily fetched subset) has a sparse, non-monotonic id set. Mapping
        # slot->id explicitly is what keeps `gather` addressing the same rows a complete table would.
        ids = list(range(len(shards))) if shard_ids is None else [int(i) for i in shard_ids]
        if len(ids) != len(shards):
            raise ValueError(f"shard_ids has {len(ids)} entries for {len(shards)} shards")
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate shard ids: {sorted(ids)}")
        self.shard_ids = ids
        self.n_shards_total = max(ids) + 1
        self.n_rows = self.rows_per_shard * self.n_shards_total
        self.complete = sorted(ids) == list(range(self.n_shards_total))
        # shard id -> slot in self._views, or -1 when that shard was not supplied.
        self._slot_of_id = np.full(self.n_shards_total, -1, dtype=np.int64)
        for slot, sid in enumerate(ids):
            self._slot_of_id[sid] = slot

        #: Persistent gather pool (see `_pool`); created on first threaded gather.
        self._exec = None
        self._exec_workers = 0
        self._maps: Dict[str, mmap.mmap] = {}
        self._fds: Dict[str, int] = {}
        self._views: List[np.ndarray] = []
        page = mmap.ALLOCATIONGRANULARITY
        for s in shards:
            if s.path not in self._maps:
                fd = os.open(s.path, os.O_RDONLY)
                size = os.fstat(fd).st_size
                flags = mmap.MAP_SHARED | getattr(mmap, "MAP_NORESERVE", 0)
                mm = mmap.mmap(fd, size, flags=flags, prot=mmap.PROT_READ)
                if advise_random and hasattr(mm, "madvise"):
                    mm.madvise(mmap.MADV_RANDOM)
                self._fds[s.path] = fd
                self._maps[s.path] = mm
            mm = self._maps[s.path]
            need = s.offset + self.rows_per_shard * self.row_bytes
            if need > len(mm):
                raise ValueError(
                    f"{os.path.basename(s.path)} is too short for its own header: shard needs bytes "
                    f"up to {need:,} but the file is {len(mm):,} ({need - len(mm):,} missing). "
                    f"The usual cause is a partial or in-progress download — the safetensors header "
                    f"is written first, so an incomplete file still advertises every tensor."
                )
            view = np.frombuffer(mm, dtype=np.uint8, count=self.rows_per_shard * self.row_bytes,
                                 offset=s.offset)
            self._views.append(view.reshape(self.rows_per_shard, self.row_bytes))
        assert page  # silence linters; kept for the documented granularity note

    # -- gather ------------------------------------------------------------

    def _locate(self, ids: np.ndarray):
        """(n,) global row ids -> (slots, absolute file offsets). Raises if a shard is absent."""
        shard_of = ids // self.rows_per_shard
        local = ids - shard_of * self.rows_per_shard
        slots = self._slot_of_id[shard_of]
        if (slots < 0).any():
            missing = sorted(set(shard_of[slots < 0].tolist()))
            raise KeyError(
                f"rows fall in shard(s) {missing[:8]} which were not loaded "
                f"({len(self.shards)} of {self.n_shards_total} shards present). "
                f"Pass the full `model-plefp8-*.safetensors` set for a complete table."
            )
        base = np.array([s.offset for s in self.shards], dtype=np.int64)[slots]
        return slots, base + local * self.row_bytes

    def prefetch(self, row_ids) -> None:
        """Ask the kernel to start reading these rows, without waiting.

        `MADV_WILLNEED` is asynchronous, so issuing it for a whole batch lets the device queue many
        reads at once; a subsequent `gather` then finds the pages resident instead of taking one
        serialised major fault per row. This is the cheap half of fixing the latency bound — no
        threads, one syscall per row, and it is advisory so it can never return wrong data.

        Offsets are sorted first: the kernel merges adjacent requests, and the NVMe queue is happier
        with ascending LBAs than with the random order the hash produces.
        """
        ids = np.asarray(row_ids, dtype=np.int64)
        if ids.size == 0:
            return
        if ids.min() < 0 or ids.max() >= self.n_rows:
            raise IndexError(f"row id outside [0, {self.n_rows})")
        slots, offs = self._locate(ids)
        page = mmap.PAGESIZE
        for slot in np.unique(slots):
            sel = slots == slot
            mm = self._maps[self.shards[int(slot)].path]
            o = np.sort(offs[sel])
            starts = (o // page) * page
            ends = ((o + self.row_bytes + page - 1) // page) * page
            # Coalesce overlapping/adjacent page ranges so hot regions cost one call, not many.
            keep = np.empty(starts.size, dtype=bool)
            keep[0] = True
            np.greater(starts[1:], ends[:-1], out=keep[1:])
            grp = np.cumsum(keep) - 1
            g_start = starts[keep]
            g_end = np.maximum.reduceat(ends, np.flatnonzero(keep))
            for a, b in zip(g_start.tolist(), g_end.tolist()):
                try:
                    mm.madvise(mmap.MADV_WILLNEED, a, b - a)
                except (OSError, ValueError):
                    return  # advisory only — never let a hint failure break a gather
            del grp

    def _gather_raw_threaded(self, ids: np.ndarray) -> np.ndarray:
        """Fetch rows with `os.pread` from a thread pool.

        `os.pread` releases the GIL, so N threads give N genuinely concurrent NVMe requests — which
        is the whole game here, because a single-threaded gather is bound by ~15 us of page-fault
        latency per row and reaches only ~10 MB/s regardless of what the drive can do. This is what
        the DGX recipe's `WORKERS=32` is buying.
        """
        slots, offs = self._locate(ids)
        paths = [self.shards[int(s)].path for s in slots]
        out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
        rb = self.row_bytes
        n_workers = min(self.workers, max(1, ids.size))
        bounds = np.linspace(0, ids.size, n_workers + 1).astype(np.int64)

        def work(lo: int, hi: int) -> None:
            for i in range(lo, hi):
                fd = self._fds[paths[i]]
                out[i] = np.frombuffer(os.pread(fd, rb, int(offs[i])), dtype=np.uint8)

        ex = self._pool(n_workers)
        list(ex.map(lambda b: work(*b), list(zip(bounds[:-1], bounds[1:]))))
        return out

    def _pool(self, n_workers: int):
        """One PERSISTENT pool, not a fresh one per gather.

        A `with ThreadPoolExecutor(...)` per call creates and joins `n_workers` OS threads on every
        gather. At a 16-row decode step that setup is most of the call, which is what made the
        threaded arm look like the wrong mechanism for small requests and pinned
        `threaded_min_rows` at 256. Sized to the largest worker count asked for so far and reused;
        `ex.map` over a fixed set of workers needs no per-call ownership.
        """
        from concurrent.futures import ThreadPoolExecutor

        if self._exec is None or self._exec_workers < n_workers:
            if self._exec is not None:
                self._exec.shutdown(wait=True)
            self._exec = ThreadPoolExecutor(max_workers=n_workers,
                                            thread_name_prefix="rowtable")
            self._exec_workers = n_workers
        return self._exec

    def _gather_raw_pread(self, ids: np.ndarray) -> np.ndarray:
        """Fetch rows with SERIAL `os.pread` — no thread pool, no mmap, no page-cache copy.

        THE GAP THIS FILLS. The table below picked mmap+prefetch for a decode-sized gather because
        the only pread arm measured was the THREADED one, where pool setup dominates at 16 rows
        (802 us vs 609 us). Serial pread was never measured: it has neither the pool cost nor mmap's
        page-cache copy.

        WHY THE FOOTPRINT DIFFERS, measured for the checkpoint reader on this same ZFS pool
        (`ckpt_read.py`, one cold 337.7 MiB shard):

            mmap, 4 KiB walk           627 MiB/s   dCached +0.33 GiB   dARC +0.33 GiB
            read() into reused buf    4948 MiB/s   dCached +0.00 GiB   dARC +0.31 GiB
            O_DIRECT into reused buf  4967 MiB/s   dCached +0.00 GiB   dARC +0.00 GiB

        OpenZFS intercepts read()/pread() at the VFS layer and serves from the ARC, so a pread costs
        ONE cache copy; an mmap'd page costs a page-cache page AND its ARC buffer, i.e. the bytes are
        buffered twice. mmap also pays roughly one ARC lookup per 4 KiB page (~91,855 lookups per
        337.7 MiB, against 1,272 for read()), which is the tax that actually dominates a scattered
        row gather. O_DIRECT would drop the ARC copy too, but its 4096-byte alignment requirement is
        a poor fit for small scattered n-gram rows, and it buys only 19 MiB/s over read() here.

        Offsets are NOT sorted: with one row per call there is nothing to merge, and sorting would
        cost a permutation to undo afterwards.
        """
        slots, offs = self._locate(ids)
        out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
        rb = self.row_bytes
        fds = [self._fds[self.shards[int(s)].path] for s in slots]
        for i in range(ids.size):
            buf = os.pread(fds[i], rb, int(offs[i]))
            if len(buf) != rb:
                # pread is permitted a short read; refuse rather than hand back a row that is part
                # stale buffer. A truncated PLE row is silently wrong output, not a crash.
                raise OSError(
                    f"short pread on {os.path.basename(self.shards[int(slots[i])].path)}: got "
                    f"{len(buf)} of {rb} bytes at offset {int(offs[i])}"
                )
            out[i] = np.frombuffer(buf, dtype=np.uint8)
        return out

    def gather_raw(self, row_ids) -> np.ndarray:
        """(n,) global row ids -> (n, row_bytes) uint8. Faults in only the pages actually touched."""
        ids = np.asarray(row_ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        if ids.min() < 0 or ids.max() >= self.n_rows:
            raise IndexError(
                f"row id outside [0, {self.n_rows}): min={ids.min()} max={ids.max()}"
            )
        # Two mechanisms, and which one wins INVERTS with request size (measured on the full 51.2 GB
        # table, cold cache, idle box):
        #
        #   rows        baseline    prefetch    workers=16/32
        #   16 (decode)  2320 us      609 us       802 us     <- prefetch wins; pool setup dominates
        #   8192         906 ms       177 ms        66 ms     <- threads win, 13.8x vs 5.1x
        #   65536       6199 ms      1326 ms       497 ms     <- threads win, 12.5x vs 4.7x
        #
        # So pick per call rather than committing to one. Threads give real I/O concurrency because
        # os.pread drops the GIL; MADV_WILLNEED gives async readahead with no threads at all, which
        # is what a 16-row decode step actually wants.
        if self.workers > 1 and ids.size >= self.threaded_min_rows:
            return self._gather_raw_threaded(ids)
        if self.small_gather == "pread":
            return self._gather_raw_pread(ids)
        if self.auto_prefetch:
            self.prefetch(ids)
        shard_of = ids // self.rows_per_shard
        local = ids - shard_of * self.rows_per_shard
        slots = self._slot_of_id[shard_of]
        out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
        # Group by shard so each mmap view is fancy-indexed once, not once per row.
        for slot in np.unique(slots):
            sel = slots == slot
            out[sel] = self._views[int(slot)][local[sel]]
        return out

    def gather(self, row_ids) -> np.ndarray:
        """(n,) global row ids -> (n, row_elems) float32, dequantised and scaled."""
        raw = self.gather_raw(row_ids)
        if self.dtype == "F8_E4M3":
            return dequant_f8_e4m3(raw, self.scale)
        if self.dtype == "BF16":
            w = np.zeros((raw.shape[0], self.row_elems, 2), dtype=np.uint8)
            w[:, :, 1] = raw.reshape(raw.shape[0], self.row_elems, 2)[:, :, 1]
            return w.reshape(raw.shape[0], -1).view("<f4") * np.float32(self.scale)
        if self.dtype in ("F16", "F32"):
            npd = "<f2" if self.dtype == "F16" else "<f4"
            return raw.view(npd).astype(np.float32) * np.float32(self.scale)
        raise ValueError(f"no decoder for {self.dtype}")

    def gather_into(self, row_ids, out: np.ndarray) -> np.ndarray:
        """Dequantise into a caller-owned buffer — e.g. a pinned staging region carved from the
        weight arena — so the hot path does one H2D copy and allocates nothing per token."""
        vals = self.gather(row_ids)
        if out.shape != vals.shape:
            raise ValueError(f"out shape {out.shape} != gathered {vals.shape}")
        out[...] = vals
        return out

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._exec is not None:
            self._exec.shutdown(wait=True)
            self._exec, self._exec_workers = None, 0
        self._views = []
        for mm in self._maps.values():
            mm.close()
        for fd in self._fds.values():
            os.close(fd)
        self._maps, self._fds = {}, {}

    def __enter__(self) -> "ShardedRowTable":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        part = "" if self.complete else f" PARTIAL {len(self.shards)}/{self.n_shards_total}"
        return (
            f"ShardedRowTable({len(self.shards)} shards over {len(self._maps)} files{part}, "
            f"{self.n_rows:,} rows x {self.row_bytes} B = "
            f"{self.n_rows * self.row_bytes / 1e9:.2f} GB, {self.dtype}, scale={self.scale:g})"
        )


# ---------------------------------------------------------------------------
# n-gram head addressing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NgramHeads:
    """The checkpoint's multi-head hash layout.

    16 heads, each owning a contiguous band of the table. Vocab sizes are primes just above
    20,000,000 (20000003, 20000023, ...) — the usual trick so a modulo spreads evenly — and
    `offsets` is exactly their running prefix sum. Their sum is 90 rows short of the table height,
    so the last rows are padding and are never addressable. That is a property worth asserting
    rather than assuming: an off-by-one in this arithmetic silently reads a *different head's*
    embeddings, which degrades quality without ever erroring.
    """

    offsets: np.ndarray  # (n_heads,) int64
    vocab_sizes: np.ndarray  # (n_heads,) int64

    @property
    def n_heads(self) -> int:
        return int(self.offsets.size)

    def validate(self, n_rows: int) -> None:
        if self.offsets.shape != self.vocab_sizes.shape:
            raise ValueError("offsets and vocab_sizes differ in length")
        expected = np.concatenate([[0], np.cumsum(self.vocab_sizes)[:-1]])
        if not np.array_equal(self.offsets, expected):
            raise ValueError("offsets are not the prefix sum of vocab_sizes")
        total = int(self.offsets[-1] + self.vocab_sizes[-1])
        if total > n_rows:
            raise ValueError(f"heads address {total} rows but the table has {n_rows}")

    def row_ids(self, hashes: np.ndarray) -> np.ndarray:
        """(..., n_heads) arbitrary integer hashes -> (..., n_heads) global row ids.

        Each head maps into its own band, so a hash collision can never cross heads.
        """
        h = np.asarray(hashes, dtype=np.int64)
        if h.shape[-1] != self.n_heads:
            raise ValueError(f"expected last dim {self.n_heads}, got {h.shape[-1]}")
        return self.offsets + np.mod(h, self.vocab_sizes)


# ---------------------------------------------------------------------------
# Qwen4-Exp convenience loader
# ---------------------------------------------------------------------------

PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding"


def open_qwen4exp_ngram_table(
    ple_files: Sequence[str],
    meta_files: Sequence[str] = (),
    *,
    layer_prefix: str = PLE_PREFIX,
    scale_override: float | None = None,
    workers: int = 0,
    auto_prefetch: bool = True,
    small_gather: str = "mmap",
):
    """Open the n-gram table from a Qwen4-Exp checkpoint's `model-plefp8-*.safetensors` set.

    `meta_files` may name additional shards holding `ngram_heads_offsets` / `ngram_heads_vocab_sizes`
    (they live in a bf16 shard, not the plefp8 set). Returns `(table, heads)`; `heads` is None if the
    metadata was not supplied.
    """
    idx = index_safetensors(list(ple_files) + list(meta_files))

    # Enumerate every shard PRESENT rather than probing 0,1,2,... and stopping at the first gap:
    # the checkpoint distributes shards to files in STRING order, so `model-plefp8-00000` carries
    # shard_0, shard_1, shard_10..shard_19. A sequential probe finds two shards and silently builds
    # a table with the wrong height.
    pat = re.compile(
        rf"^{re.escape(layer_prefix)}\.ngram_embedding\.shard_(\d+)\.weight$"
    )
    found = sorted(
        ((int(m.group(1)), name) for name in idx if (m := pat.match(name))),
    )
    if not found:
        raise KeyError(
            f"no '{layer_prefix}.ngram_embedding.shard_*.weight' tensors in the given files. "
            f"Note only ONE layer carries the PLE block (ple_layer_ids is 1-BASED, so [2] means "
            f"tensors named layers.1.ple.*)."
        )
    shard_ids = [i for i, _ in found]

    scale = 1.0
    sk = f"{layer_prefix}.ngram_embedding.weight_scale"
    if sk in idx:
        scale = float(np.asarray(read_tensor(idx[sk])).reshape(-1)[0])
    elif scale_override is None:
        # The scale lives in the LAST plefp8 file, so a caller validating against a single shard
        # file will not have it. Silently using 1.0 would scale every embedding ~5000x and still
        # "work", so say so.
        raise KeyError(
            f"'{sk}' not found in the given files (it lives in the last model-plefp8-* shard). "
            f"Pass scale_override=... to proceed with a partial file set — defaulting to 1.0 would "
            f"silently mis-scale every row by ~1/{1/0.0002:.0f}."
        )
    if scale_override is not None:
        scale = float(scale_override)

    table = ShardedRowTable(
        [idx[n] for _, n in found], scale=scale, shard_ids=shard_ids, workers=workers,
        auto_prefetch=auto_prefetch, small_gather=small_gather
    )

    heads = None
    ok, vk = f"{layer_prefix}.ngram_heads_offsets", f"{layer_prefix}.ngram_heads_vocab_sizes"
    if ok in idx and vk in idx:
        heads = NgramHeads(
            offsets=read_tensor(idx[ok]).astype(np.int64),
            vocab_sizes=read_tensor(idx[vk]).astype(np.int64),
        )
        heads.validate(table.n_rows)
    return table, heads
