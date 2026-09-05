#!/usr/bin/env python3
"""Which way of getting a cold safetensors shard off ZFS is fastest? Measure, do not assume.

`ckpt_read_parity.py` proved the read() reader byte-identical but reported it 0.86x — SLOWER —
which contradicts `zfs_readpath.txt`'s 2209 vs 551 MiB/s. Both cannot be right, and the parity
harness is the suspect: it reads each shard TWICE (the mmap leg warms ARC for the read() leg), and
it charges both legs a full uint8 `.sum()` over 14 GiB that dilutes any I/O difference.

So: EVERY LEG GETS ITS OWN COLD SHARDS, drawn from disjoint slices of the 196 so no leg can warm
another, and each leg does the identical downstream work (build every tensor, touch every byte).
The only thing that varies is HOW the bytes arrive.

  A  safe_open (mmap, faulted lazily by the touch)          <-- what ships
  B  read() into an anonymous mmap buffer                    <-- ReadSafeOpen
  C  safe_open + MADV_WILLNEED on the mapping
  D  safe_open + MAP_POPULATE-equivalent (MADV_POPULATE_READ)

C and D are in here because they would be three-line fixes that keep safetensors' own mmap: if the
ZFS tax is per-fault rather than per-page-lookup, a bulk prefault removes it without a reader at
all. If they lose, the reader is justified; if they win, the reader is dead code and should not
merge. `zfs_readpath.txt` already predicts they LOSE (its slow mmap leg took 91,855 ARC *hits* and
128 misses — it was warm and still ran at 551 MiB/s), but a prediction is not a measurement.

CPU only. No GPU, no lease.
"""
import ctypes
import ctypes.util
import mmap
import os
import sys
import time

sys.path.insert(0, "/engine/python")

import safetensors  # noqa: E402
import torch  # noqa: E402

from minisgl.weights.ckpt_read import ReadSafeOpen, read_file_bytes  # noqa: E402

MODEL = os.environ.get("RATE_MODEL", "/model")
PER_LEG = int(os.environ.get("RATE_PER_LEG", "6"))
MIB = 1 << 20

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
MADV_WILLNEED = 3
MADV_POPULATE_READ = 22  # Linux 5.14+


def touch(f, keys) -> int:
    """Build every tensor and read every byte — the work the boot actually does downstream."""
    n = 0
    for k in keys:
        t = f.get_tensor(k)
        v = t.contiguous().reshape(-1)
        if v.numel():
            v.view(torch.uint8).sum(dtype=torch.int64)
            n += v.numel() * v.element_size()
    return n


def leg_mmap(path, advise=None):
    f = safetensors.safe_open(path, framework="pt", device="cpu")
    if advise is not None:
        # Advise the WHOLE file through our own mapping of the same pages: safetensors does not
        # expose its mapping, and the page cache is shared, so advising this mapping prefaults the
        # cache safetensors' mapping will then hit.
        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            rc = _libc.madvise(ctypes.c_void_p(addr), len(mm), advise)
            if rc != 0:
                print(f"    (madvise {advise} -> errno {ctypes.get_errno()})", flush=True)
            n = touch(f, list(f.keys()))
            del f
            mm.close()
            return n
    return touch(f, list(f.keys()))


def leg_read(path):
    f = ReadSafeOpen(path)
    return touch(f, f.keys())


LEGS = [
    ("A_mmap_lazy          ", lambda p: leg_mmap(p)),
    ("B_read_into_anon     ", leg_read),
    ("C_mmap_WILLNEED      ", lambda p: leg_mmap(p, MADV_WILLNEED)),
    ("D_mmap_POPULATE_READ ", lambda p: leg_mmap(p, MADV_POPULATE_READ)),
]

# Expert shards only, and a DISJOINT slice per leg so no leg warms another. They are uniform
# (337.7 MiB, 1536 keys, identical dtype mix), which is what makes the legs comparable at all.
shards = sorted(f for f in os.listdir(MODEL)
                if f.startswith("layer-") and f.endswith(".safetensors"))
need = PER_LEG * len(LEGS)
if len(shards) < need:
    sys.exit(f"need {need} expert shards, found {len(shards)}")
# Spread across the file list rather than taking a contiguous run, so no leg is systematically
# favoured by whatever the last boot happened to leave in ARC.
picks = {name: shards[i::len(LEGS)][:PER_LEG] for i, (name, _) in enumerate(LEGS)}

print(f"model={MODEL}  {len(shards)} expert shards, {PER_LEG} cold shards per leg", flush=True)
print(f"raw read() of one untouched shard, for reference:", flush=True)
_ref = shards[-1]
_t = time.perf_counter()
_b = read_file_bytes(os.path.join(MODEL, _ref))
_dt = time.perf_counter() - _t
print(f"  {_ref}  {len(_b) / MIB:.1f} MiB in {_dt:.3f} s = {len(_b) / MIB / _dt:.1f} MiB/s "
      f"(no tensors built, no touch)", flush=True)
del _b

results = []
for name, fn in LEGS:
    tot = 0
    t0 = time.perf_counter()
    for s in picks[name]:
        tot += fn(os.path.join(MODEL, s))
    dt = time.perf_counter() - t0
    results.append((name, tot, dt))
    print(f"  {name} {tot / MIB:9.1f} MiB in {dt:7.3f} s = {tot / MIB / dt:8.1f} MiB/s", flush=True)

base = results[0][1] / results[0][2]
print("\n=== vs A (what ships) ===")
for name, tot, dt in results:
    r = tot / MIB / dt
    print(f"  {name} {r:8.1f} MiB/s   {r / base:5.2f}x")
