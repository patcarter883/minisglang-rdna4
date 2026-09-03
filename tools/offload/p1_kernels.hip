// P1 probe kernels: device-side reads from host-located pages behind a device VA.
//
// Companion to tools/offload/p1_host_read_bw.py.  Everything here is a thin
// `extern "C"` host wrapper plus three read kernels; the VMM plumbing
// (hipMemAddressReserve / hipMemCreate(location=Host) / hipMemMap /
// hipMemSetAccess) lives in the Python driver so the probe can report exactly
// which HIP call failed.
//
// Build (the driver does this automatically):
//   /opt/rocm/bin/hipcc -O3 -std=c++17 --offload-arch=gfx1201 -fPIC -shared \
//       -o tools/offload/_build/p1_kernels.so tools/offload/p1_kernels.hip
//
// Contract for every read kernel:
//   * the buffer has been filled so that the u32 at dword index i equals i
//     (p1_fill_range(base, 0, bytes, 0)),
//   * the kernel touches a PREFIX of the buffer, every covered byte exactly
//     once, and atomically accumulates the u32 sum of what it read,
//   * so the driver can check sum == M*(M-1)/2 mod 2^32 with M = cover/4.
//     A wrong checksum means the read did not actually land on the pages we
//     think it did -- the probe then refuses to report a bandwidth number.

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

typedef unsigned int u32;
typedef unsigned long long u64;

// Our own error codes are negative so they can never collide with hipError_t.
#define P1_EBADARG   (-101)
#define P1_EDIVIS    (-102)
#define P1_ENOMEM    (-103)
#define P1_EPATTERN  (-104)
#define P1_ETOOBIG   (-105)

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) return (int)_e; } while (0)

// Drop any error left in HIP's thread-local last-error slot by an EARLIER,
// unrelated call, so the hipGetLastError() after a launch reports only that
// launch.
//
// This is load-bearing, not hygiene.  hipGetLastError() returns AND clears a
// sticky per-thread slot, and the driver side of this probe deliberately makes
// calls that are EXPECTED to fail: it probes hipMemCreate with location types
// (HostNuma, HostNumaCurrent) that gfx1201 rejects with hipErrorInvalidValue,
// precisely in order to discover which ones work.  Measured on this box: after
// one such refused hipMemCreate, the very next p1_fill_range on a freshly
// hipMalloc'd, completely unrelated device buffer returned rc=1 "invalid
// argument", and the call after that returned 0 -- a phantom failure inherited
// from a probe of a different API, which aborted the whole run.  Clearing
// before the launch also stops the converse bug: a real launch failure being
// silently consumed by an unrelated earlier hipGetLastError() and never
// reported.
#define P1_CLEAR_STICKY_ERROR() do { (void)hipGetLastError(); } while (0)

// ---------------------------------------------------------------- utilities

__device__ __forceinline__ u32 sum4(uint4 v) { return v.x + v.y + v.z + v.w; }

__device__ __forceinline__ void wave_atomic_add(u32 acc, u32 *gsum) {
    for (int o = warpSize / 2; o > 0; o >>= 1) acc += __shfl_down(acc, o, warpSize);
    if ((threadIdx.x % warpSize) == 0) atomicAdd(gsum, acc);
}

// ---------------------------------------------------------------- fill

// dword at global index (off_u32 + j) := base_val + j   for j in [0, n_u32)
__global__ void k_fill_range(u32 *p, size_t off_u32, size_t n_u32, u32 base_val) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    for (; i < n_u32; i += st) p[off_u32 + i] = base_val + (u32)i;
}

// ---------------------------------------------------------------- pattern 0: linear stream

__global__ void k_linear(const uint4 *__restrict__ p, size_t n4, u32 *gsum) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    u32 acc = 0;
    for (; i < n4; i += st) acc += sum4(p[i]);
    wave_atomic_add(acc, gsum);
}

// ---------------------------------------------------------------- pattern 1: tiled / strided
//
// Models the grouped-GEMM's walk over an expert row: contiguous `seg` bursts,
// but the traversal order is transposed so consecutive workgroups are
// `stride` bytes apart -- the K-walk over an N-major packed weight
// (w_base = w_rep + e*(N/16)*ktiles*32).  Every byte of every covered row is
// read exactly once, so the checksum still applies.

__global__ void k_strided(const uint4 *__restrict__ base, u32 *gsum,
                          size_t rows, size_t row_u4, size_t stride_u4, size_t seg_u4) {
    const size_t nstr = row_u4 / stride_u4;      // strides per row
    const size_t nsub = stride_u4 / seg_u4;      // segments per stride
    const size_t segs_per_row = nstr * nsub;
    const size_t total = rows * segs_per_row;

    const int lanes = (int)seg_u4;               // threads cooperating on one segment
    const int groups = blockDim.x / lanes;       // segments in flight per workgroup
    const int g = threadIdx.x / lanes;
    const int l = threadIdx.x % lanes;

    u32 acc = 0;
    for (size_t s = (size_t)blockIdx.x * groups + g; s < total;
         s += (size_t)gridDim.x * groups) {
        const size_t r = s / segs_per_row;
        const size_t k = s % segs_per_row;
        const size_t o = k / nstr;               // which segment within a stride (column)
        const size_t j = k % nstr;               // which stride (row)  -> transposed walk
        acc += sum4(base[r * row_u4 + j * stride_u4 + o * seg_u4 + l]);
    }
    wave_atomic_add(acc, gsum);
}

// ---------------------------------------------------------------- pattern 2: random 128 B
//
// nlines must be a power of two; multiplying by an odd constant mod 2^k is a
// bijection, so every line is still visited exactly once (prefix cover).

__global__ void k_random(const uint4 *__restrict__ base, u32 *gsum,
                         size_t nlines, size_t line_u4) {
    const int lanes = (int)line_u4;
    const int groups = blockDim.x / lanes;
    const int g = threadIdx.x / lanes;
    const int l = threadIdx.x % lanes;
    const size_t mask = nlines - 1;

    u32 acc = 0;
    for (size_t s = (size_t)blockIdx.x * groups + g; s < nlines;
         s += (size_t)gridDim.x * groups) {
        const size_t p = (s * 0x9E3779B1ull) & mask;
        acc += sum4(base[p * line_u4 + l]);
    }
    wave_atomic_add(acc, gsum);
}

// ---------------------------------------------------------------- line checksums (coherence arm)

__global__ void k_line_sums(const u32 *__restrict__ p, size_t off_u32,
                            size_t nlines, u32 line_u32, u32 *out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    for (; i < nlines; i += st) {
        u32 a = 0;
        const u32 *q = p + off_u32 + i * line_u32;
        for (u32 j = 0; j < line_u32; ++j) a += q[j];
        out[i] = a;
    }
}

// ---------------------------------------------------------------- host side

static u32 *g_sum[16] = {nullptr};

static int ensure_sum(u32 **out) {
    int dev = 0;
    CK(hipGetDevice(&dev));
    if (dev < 0 || dev >= 16) return P1_EBADARG;
    if (g_sum[dev] == nullptr) {
        void *p = nullptr;
        CK(hipMalloc(&p, sizeof(u32)));
        g_sum[dev] = (u32 *)p;
    }
    *out = g_sum[dev];
    return 0;
}

extern "C" {

const char *p1_err(int e) {
    switch (e) {
        case P1_EBADARG:  return "P1: bad argument";
        case P1_EDIVIS:   return "P1: divisibility precondition violated";
        case P1_ENOMEM:   return "P1: allocation failed";
        case P1_EPATTERN: return "P1: unknown pattern id";
        case P1_ETOOBIG:  return "P1: request too large";
        default: break;
    }
    if (e < 0) return "P1: unknown internal error";
    return hipGetErrorString((hipError_t)e);
}

int p1_runtime_version(int *v) { CK(hipRuntimeGetVersion(v)); return 0; }
int p1_driver_version(int *v)  { CK(hipDriverGetVersion(v));  return 0; }
int p1_device_count(int *n)    { CK(hipGetDeviceCount(n));    return 0; }
int p1_set_device(int d)       { CK(hipSetDevice(d));         return 0; }
int p1_sync(void)              { CK(hipDeviceSynchronize());  return 0; }

int p1_mem_info(size_t *freeb, size_t *totalb) {
    CK(hipMemGetInfo(freeb, totalb));
    return 0;
}

int p1_device_info(int dev, char *name, int nlen, char *arch, int alen,
                   char *pci, int plen, int *mp_count, int *warp,
                   int *clock_khz, unsigned long long *total_mem) {
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, dev));
    if (name && nlen > 0) { snprintf(name, nlen, "%s", p.name); }
    if (arch && alen > 0) { snprintf(arch, alen, "%s", p.gcnArchName); }
    if (pci && plen > 0)  { CK(hipDeviceGetPCIBusId(pci, plen, dev)); }
    if (mp_count)  *mp_count = p.multiProcessorCount;
    if (warp)      *warp = p.warpSize;
    if (clock_khz) *clock_khz = p.clockRate;
    if (total_mem) *total_mem = (unsigned long long)p.totalGlobalMem;
    return 0;
}

int p1_malloc_device(void **p, size_t bytes) { CK(hipMalloc(p, bytes)); return 0; }
int p1_free_device(void *p)                  { CK(hipFree(p));          return 0; }
int p1_host_malloc(void **p, size_t bytes, unsigned int flags) {
    CK(hipHostMalloc(p, bytes, flags));
    return 0;
}
int p1_host_free(void *p) { CK(hipHostFree(p)); return 0; }
int p1_host_dev_ptr(void **dp, void *hp) {
    CK(hipHostGetDevicePointer(dp, hp, 0));
    return 0;
}

// dword at global index (off_bytes/4 + j) := base_val + j
int p1_fill_range(void *p, size_t off_bytes, size_t len_bytes, unsigned int base_val) {
    if ((off_bytes & 3u) || (len_bytes & 3u)) return P1_EDIVIS;
    const size_t n = len_bytes / 4;
    if (n == 0) return P1_EBADARG;
    const int threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    P1_CLEAR_STICKY_ERROR();
    hipLaunchKernelGGL(k_fill_range, dim3((unsigned)blocks), dim3(threads), 0, 0,
                       (u32 *)p, off_bytes / 4, n, base_val);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    return 0;
}

// pattern: 0 = linear, 1 = strided/tiled, 2 = random 128 B
// Returns the covered byte count (out_cover) and the event-timed kernel ms.
int p1_run(int pattern, const void *base, size_t bytes, int blocks, int threads,
           size_t row_bytes, size_t seg_bytes, size_t stride_bytes,
           unsigned int *out_sum, size_t *out_cover, float *out_ms) {
    if (!base || bytes < 16 || blocks < 1 || threads < 1) return P1_EBADARG;
    if (threads % 32 != 0 || threads > 1024) return P1_EBADARG;

    // ---- validate and precompute BEFORE any resource is created, so an
    // argument error can never leak an event or leave a half-timed run.
    size_t cover = 0, n4 = 0, rows = 0, seg_u4 = 0, nlines = 0, line_u4 = 0;
    if (pattern == 0) {
        n4 = bytes / 16;
        cover = n4 * 16;
    } else if (pattern == 1) {
        if (seg_bytes == 0 || stride_bytes == 0 || row_bytes == 0) return P1_EBADARG;
        if (seg_bytes % 16 || stride_bytes % seg_bytes || row_bytes % stride_bytes)
            return P1_EDIVIS;
        seg_u4 = seg_bytes / 16;
        if (seg_u4 == 0 || (size_t)threads % seg_u4) return P1_EDIVIS;
        rows = bytes / row_bytes;
        if (rows == 0) return P1_EBADARG;
        cover = rows * row_bytes;
    } else if (pattern == 2) {
        if (seg_bytes == 0 || seg_bytes % 16) return P1_EDIVIS;
        line_u4 = seg_bytes / 16;
        if ((size_t)threads % line_u4) return P1_EDIVIS;
        nlines = bytes / seg_bytes;
        if (nlines == 0) return P1_EBADARG;
        size_t n2 = 1;
        while ((n2 << 1) <= nlines) n2 <<= 1;       // largest power of two <= nlines
        nlines = n2;
        cover = nlines * seg_bytes;
    } else {
        return P1_EPATTERN;
    }

    u32 *gs = nullptr;
    int rc = ensure_sum(&gs);
    if (rc) return rc;
    CK(hipMemset(gs, 0, sizeof(u32)));
    CK(hipDeviceSynchronize());

    // Cleared OUTSIDE the event window: hipGetLastError is a host-side TLS read,
    // and doing it between the two event records would add host latency to a
    // measurement whose whole purpose is a bandwidth number.
    P1_CLEAR_STICKY_ERROR();

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0, 0));

    if (pattern == 0) {
        hipLaunchKernelGGL(k_linear, dim3((unsigned)blocks), dim3(threads), 0, 0,
                           (const uint4 *)base, n4, gs);
    } else if (pattern == 1) {
        hipLaunchKernelGGL(k_strided, dim3((unsigned)blocks), dim3(threads), 0, 0,
                           (const uint4 *)base, gs, rows, row_bytes / 16,
                           stride_bytes / 16, seg_u4);
    } else {
        hipLaunchKernelGGL(k_random, dim3((unsigned)blocks), dim3(threads), 0, 0,
                           (const uint4 *)base, gs, nlines, line_u4);
    }

    CK(hipGetLastError());
    CK(hipEventRecord(e1, 0));
    CK(hipEventSynchronize(e1));
    float ms = 0.f;
    CK(hipEventElapsedTime(&ms, e0, e1));
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));

    u32 s = 0;
    CK(hipMemcpy(&s, gs, sizeof(u32), hipMemcpyDeviceToHost));
    if (out_sum) *out_sum = s;
    if (out_cover) *out_cover = cover;
    if (out_ms) *out_ms = ms;
    return 0;
}

// kind: 1 = H2D, 2 = D2H, 3 = D2D, 4 = Default
int p1_memcpy_bw(void *dst, const void *src, size_t bytes, int kind, int reps,
                 float *out_ms /* array[reps] */) {
    if (!dst || !src || bytes == 0 || reps < 1) return P1_EBADARG;
    hipMemcpyKind k;
    switch (kind) {
        case 1: k = hipMemcpyHostToDevice; break;
        case 2: k = hipMemcpyDeviceToHost; break;
        case 3: k = hipMemcpyDeviceToDevice; break;
        case 4: k = hipMemcpyDefault; break;
        default: return P1_EBADARG;
    }
    hipStream_t s;
    CK(hipStreamCreate(&s));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    for (int i = 0; i < reps; ++i) {
        CK(hipEventRecord(e0, s));
        CK(hipMemcpyAsync(dst, src, bytes, k, s));
        CK(hipEventRecord(e1, s));
        CK(hipEventSynchronize(e1));
        CK(hipEventElapsedTime(&out_ms[i], e0, e1));
    }
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipStreamDestroy(s));
    return 0;
}

// Per-line u32 checksums over [off, off+nlines*line_bytes) copied back to host.
int p1_line_checksums(const void *p, size_t off_bytes, size_t nlines,
                      unsigned int line_bytes, unsigned int *out_host) {
    if (!p || !out_host || nlines == 0) return P1_EBADARG;
    if ((off_bytes & 3u) || (line_bytes & 3u)) return P1_EDIVIS;
    if (nlines > (1u << 22)) return P1_ETOOBIG;
    void *d = nullptr;
    CK(hipMalloc(&d, nlines * sizeof(u32)));
    const int threads = 256;
    size_t blocks = (nlines + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    P1_CLEAR_STICKY_ERROR();
    hipLaunchKernelGGL(k_line_sums, dim3((unsigned)blocks), dim3(threads), 0, 0,
                       (const u32 *)p, off_bytes / 4, nlines, line_bytes / 4, (u32 *)d);
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) { (void)hipFree(d); return (int)le; }
    hipError_t ce = hipMemcpy(out_host, d, nlines * sizeof(u32), hipMemcpyDeviceToHost);
    (void)hipFree(d);
    CK(ce);
    return 0;
}

// Direct placement query.  A QUERY CAN LIE (this whole probe exists because
// hipMemCreate(location=Host) can return hipSuccess for device memory), so the
// driver cross-checks this against the VRAM/MemAvailable deltas and the
// measured bandwidth.  Never trusted on its own.
// mem_type follows hipMemoryType: 0=Host 1=Device 2=Array 3=Unified 4=Managed.
int p1_pointer_attrs(const void *p, int *mem_type, int *device, void **dev_ptr,
                     void **host_ptr, int *is_managed, unsigned int *flags) {
    hipPointerAttribute_t a;
    memset(&a, 0, sizeof(a));
    hipError_t e = hipPointerGetAttributes(&a, p);
    if (e != hipSuccess) return (int)e;
    if (mem_type)   *mem_type = (int)a.type;
    if (device)     *device = a.device;
    if (dev_ptr)    *dev_ptr = a.devicePointer;
    if (host_ptr)   *host_ptr = a.hostPointer;
    if (is_managed) *is_managed = a.isManaged;
    if (flags)      *flags = a.allocationFlags;
    return 0;
}

// CPU-side ordering primitives for the coherence arm.
void p1_cpu_fence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

void p1_cpu_flush(void *p, size_t bytes) {
    // clflush the range, then fence.  Harmless on already-uncached memory.
    char *q = (char *)p;
    for (size_t i = 0; i < bytes; i += 64) __builtin_ia32_clflush(q + i);
    __builtin_ia32_sfence();
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

}  // extern "C"
