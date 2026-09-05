"""Which read path grows the box's RECLAIMABLE FOOTPRINT? CPU-only, no lease, no torch.

Every earlier round priced the checkpoint read as a RATE and drew the wrong conclusion from it:
a rate table says O_DIRECT wins, and round 2 shipped O_DIRECT and lost 3.5x. What decides a 48-layer
TP=2 boot is the OTHER column — how much reclaimable memory the read leaves behind on a box whose
two 24.12 GiB pinned arenas have already taken half of RAM. So this probe reports both, and each leg
gets a DISTINCT cold shard so no leg warms another.

Measured 2026-09-06 on `zpcachyoshome/home` (ZFS 2.4.3, 16 GiB ARC cap, 45.9 GiB zram swap):

    mmap, 4 KiB walk          614.9 MiB/s   dCached +0.33 GiB  dARC +0.33 GiB   <- what shipped
    read() into reused buf   1653.8 MiB/s   dCached +0.00 GiB  dARC +0.31 GiB
    O_DIRECT into reused buf 1829.1 MiB/s   dCached +0.00 GiB  dARC +0.00 GiB   <- what ships now

mmap costs TWICE the bytes — a page-cache page AND an ARC buffer per 4 KiB — at a third of the rate.
The `posix_fadvise(POSIX_FADV_DONTNEED)` legs are here because bounding mmap's footprint IN PLACE
would have been the smaller, safer change: it returns 0 on this mount and frees nothing.

The first leg walks one byte per 16 MiB block on purpose. It is a control, not a reader: it shows
that a walk which does not touch every page also does not pay for every page, so only the 4 KiB leg
is the apples-to-apples mmap number.

    python tools/offload/zfs_readpath_probe.py <four distinct cold shards>
"""
import ctypes, mmap, os, sys, time

def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, _, r = line.partition(":")
        d[k] = int(r.split()[0]) * 1024
    return d

def arc():
    for line in open("/proc/spl/kstat/zfs/arcstats"):
        if line.startswith("size "):
            return int(line.split()[-1])
    return 0

def box():
    m = meminfo()
    return m["Cached"], arc(), m["MemAvailable"]

def show(tag, b0, b1, nbytes, dt):
    print(f"{tag:28s} {nbytes/2**20/dt:8.1f} MiB/s   dCached {(b1[0]-b0[0])/2**30:+7.2f} GiB   "
          f"dARC {(b1[1]-b0[1])/2**30:+7.2f} GiB   dAvail {(b1[2]-b0[2])/2**30:+7.2f} GiB")

files = sys.argv[1:]
BLK = 16 << 20

# ---- leg 1: mmap byte-walk (what safetensors does) ----
f = files[0]
n = os.path.getsize(f)
b0 = box(); t = time.perf_counter()
fd = os.open(f, os.O_RDONLY)
mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
s = 0
mv = memoryview(mm)
for off in range(0, n, BLK):
    s += mv[off:off+BLK][0]
dt = time.perf_counter() - t
b1 = box(); show("mmap walk", b0, b1, n, dt)
# fadvise DONTNEED test on the SAME fd/inode while still mapped, then after munmap
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_long, ctypes.c_long, ctypes.c_int]
POSIX_FADV_DONTNEED = 4
rc = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
b2 = box()
print(f"  fadvise(DONTNEED) while mapped rc={rc}  dCached {(b2[0]-b1[0])/2**30:+7.2f} GiB  dARC {(b2[1]-b1[1])/2**30:+7.2f} GiB")
del mv; mm.close()
b3 = box()
print(f"  after munmap                      dCached {(b3[0]-b2[0])/2**30:+7.2f} GiB  dARC {(b3[1]-b2[1])/2**30:+7.2f} GiB")
rc = libc.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)
b4 = box()
print(f"  fadvise(DONTNEED) after munmap rc={rc}  dCached {(b4[0]-b3[0])/2**30:+7.2f} GiB  dARC {(b4[1]-b3[1])/2**30:+7.2f} GiB")
os.close(fd)

# ---- leg 2: buffered read() into ONE reused pre-faulted buffer ----
f = files[1]
n = os.path.getsize(f)
buf = mmap.mmap(-1, BLK)          # anon, page aligned
buf.write(b"\0" * BLK); buf.seek(0)   # pre-fault
mvb = memoryview(buf)
b0 = box(); t = time.perf_counter()
fd = os.open(f, os.O_RDONLY)
got = 0
while got < n:
    k = os.preadv(fd, [mvb], got)
    if not k:
        break
    got += k
dt = time.perf_counter() - t
os.close(fd)
b1 = box(); show("read() reused buf", b0, b1, got, dt)

# ---- leg 3: O_DIRECT into ONE reused pre-faulted buffer ----
f = files[2]
n = os.path.getsize(f)
b0 = box(); t = time.perf_counter()
try:
    fd = os.open(f, os.O_RDONLY | os.O_DIRECT)
except OSError as e:
    print("O_DIRECT open failed:", e); fd = None
if fd is not None:
    got = 0
    while got < n:
        k = os.preadv(fd, [mvb], got)
        if not k:
            break
        got += k
    dt = time.perf_counter() - t
    os.close(fd)
    b1 = box(); show("O_DIRECT reused buf", b0, b1, got, dt)

# ---- leg 4: mmap again on a 4th cold shard, for a same-session rate reference ----
f = files[3]
n = os.path.getsize(f)
b0 = box(); t = time.perf_counter()
fd = os.open(f, os.O_RDONLY)
mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
mv = memoryview(mm); s = 0
for off in range(0, n, 4096):
    s += mv[off]
dt = time.perf_counter() - t
b1 = box(); show("mmap 4K walk", b0, b1, n, dt)
del mv; mm.close(); os.close(fd)
print("box now: Cached %.2f GiB  ARC %.2f GiB  Avail %.2f GiB" % tuple(x/2**30 for x in box()))
