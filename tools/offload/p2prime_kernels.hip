// P2prime probe kernels: mixed-media grouped MoE decode-GEMV over an EXPLICIT TWO-STACK
// weight layout (device hipMalloc stack + pinned hipHostMalloc(Mapped) host stack).
//
// Companion to tools/offload/p2prime_two_stack_moe.py.
//
// WHY THIS FILE EXISTS
//   Phase 0 (P1/P2/P3) established that `hipMemCreate(location.type = hipMemLocationTypeHost)`
//   silently returns DEVICE VRAM on this box, so the plan's single-VA mixed-media expert stack is
//   NOT CONSTRUCTIBLE.  The only working host mechanism is `hipHostMalloc(...Mapped)` +
//   `hipHostGetDevicePointer`, which yields a SEPARATE virtual address that cannot be interleaved
//   with a device allocation.  Therefore a mixed-media MoE layer must be built as TWO STACKS with a
//   per-expert pointer table selecting which stack each routed expert reads from.  That is exactly
//   what this file constructs.
//
// WHY THERE IS NO FORKED KERNEL HERE
//   KERNEL_CORE_POLICY.md: a new weight/placement scheme is a *loader policy* on the existing
//   shared core, never a new kernel.  `gemv_decode::gemv_decode_impl` already funnels every
//   per-expert base-pointer computation through exactly three WLoad hooks:
//
//       wq_e = WLoad::wq_expert(w_data,   e, N, K);
//       ws_e = WLoad::ws_expert(w_scales, e, N, num_groups);
//       wz_e = WLoad::wz_expert(w_zeros,  e, N, num_groups);
//
//   so a two-stack layout is a 3-method policy, `TwoStackInt4A16GemvLoader`, that inherits
//   everything else (decode32/dot32/accum/accum_bylane/wscale_epi/uses_act_scale_v) from the
//   SHIPPED `gemv_decode::Int4A16GemvLoader<Elem>` and only replaces `w + e*stride` with a lookup
//   in a per-expert `__constant__` pointer table.  The measured inner loop, the register
//   allocation, the tiling and the numerics are therefore the shipped ones BY CONSTRUCTION -- the
//   probe cannot accidentally measure a probe-shaped kernel.
//
//   The cost of the indirection is one scalar (block-uniform) `__constant__` load per block, and
//   the driver measures it: `p2p_*` entry points take a `use_table` flag that selects the STOCK
//   loader instead, giving an A/B against the unmodified shipped path on identical data.  Per the
//   repo's A/B rule the baseline is the OLD CODE, not an emulation of it.
//
// BUILD (the Python driver does this automatically):
//   /opt/rocm/bin/hipcc -O3 -std=c++17 --offload-arch=gfx1201 -fPIC -shared \
//       -I <rdna4-hip-kernels>/fp8_wmma/fp8_wmma_rocm \
//       -o tools/offload/_build/p2prime_kernels.so tools/offload/p2prime_kernels.hip
//
// DTYPE: activations/outputs are fp16 (`__half`) only.  The shared core is templated on the
// activation element and the bf16 instantiation differs by one template argument, but every extra
// dtype doubles the instantiation count (and the compile time) of a probe whose answer -- the SHAPE
// of per-layer time vs miss count -- is set by weight traffic, not by the activation type.  The
// driver rejects any other `--act-dtype` explicitly rather than silently substituting one.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "gemv_decode.h"   // the SHIPPED decode-GEMV core + Int4A16GemvLoader (rdna4-hip-kernels)

typedef unsigned int u32;
typedef unsigned long long u64;

// Our own error codes are negative so they can never collide with hipError_t.
#define P2P_EBADARG   (-101)
#define P2P_EDIVIS    (-102)
#define P2P_ETOOBIG   (-103)
#define P2P_EUNSUP    (-104)
#define P2P_ETABLE    (-105)

#define CK(x) do { hipError_t _e = (x); if (_e != hipSuccess) return (int)_e; } while (0)

// Drop any error left in HIP's thread-local sticky slot by an EARLIER, unrelated call, so the
// hipGetLastError() after a launch reports only that launch.  P1 hit a phantom rc=1 inherited from
// a deliberately-refused hipMemCreate probe; the same hazard applies here because the driver
// probes allocation limits.
#define P2P_CLEAR_STICKY() do { (void)hipGetLastError(); } while (0)

// ===========================================================================
// The per-expert two-stack pointer table.
// ===========================================================================
//
// One entry per expert, holding that expert's granule base in WHICHEVER stack currently owns it.
// A granule is one expert's slice of EVERY tensor in the container (packed weights, scales and --
// when present -- zero-points); the three tables are always written together by the driver, because
// dequantizing one expert's weights against another's scale is a silent wrong-numbers bug with no
// crash.
//
// `__constant__` (not a kernel argument) because the shipped core's signature is fixed and must not
// be forked to carry a probe argument.  The index is block-uniform (`e = expert_ids[block_idx]`),
// so this codegens to a scalar load issued once per block and amortized over the entire K sweep.
#define P2P_MAX_E 1024
#define P2P_E_MASK (P2P_MAX_E - 1)

__constant__ const int*    c_wq_tab[P2P_MAX_E];
__constant__ const __half* c_ws_tab[P2P_MAX_E];
__constant__ const int*    c_wz_tab[P2P_MAX_E];

// The policy.  Everything except the three base-pointer hooks is inherited from the shipped loader.
// The `& P2P_E_MASK` is a hardware-cheap bound: a corrupt expert id can then only read a valid
// table slot instead of walking off the end of constant memory (an unrecoverable GPU page fault
// that would abort the process and lose the run's artifact).
template <typename Elem>
struct TwoStackInt4A16GemvLoader : public gemv_decode::Int4A16GemvLoader<Elem> {
    using Base    = gemv_decode::Int4A16GemvLoader<Elem>;
    using WT      = typename Base::WT;         // int  (8 int4 nibbles per int32)
    using WScaleT = typename Base::WScaleT;    // __half, per (channel, group)

    __device__ static __forceinline__ const WT* wq_expert(const WT*, long e, int, int) {
        return c_wq_tab[e & P2P_E_MASK];
    }
    __device__ static __forceinline__ const WScaleT* ws_expert(const WScaleT*, long e, int, int) {
        return c_ws_tab[e & P2P_E_MASK];
    }
    __device__ static __forceinline__ const int* wz_expert(const int* z, long e, int, int) {
        return (z != nullptr) ? c_wz_tab[e & P2P_E_MASK] : nullptr;
    }
};

using StockLoader = gemv_decode::Int4A16GemvLoader<__half>;
using TableLoader = TwoStackInt4A16GemvLoader<__half>;
using AT = __half;

// ===========================================================================
// Support kernels: fill / checksum-read / cache thrash
// ===========================================================================

// dword at global index (off_u32 + j) := base_val + j   (P1's contract, verbatim)
__global__ void k_fill_seq(u32 *p, size_t off_u32, size_t n_u32, u32 base_val) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    for (; i < n_u32; i += st) p[off_u32 + i] = base_val + (u32)i;
}

// Deterministic PRNG fill (splitmix32 on the global dword index).  Used for the PACKED int4
// weights and zero-points, where a sequential ramp would make every expert's nibbles correlated and
// could let a systematic cancellation hide a wrong-expert read behind a plausible number.  The
// stream is a pure function of (seed, index), so the DEVICE and HOST stacks can be given the same
// content (mirror, for the timing arm) or different content (for the provenance arm) by seed alone,
// with no host-side buffer and no PCIe upload of 1.2 GiB.
__device__ __forceinline__ u32 splitmix32(u32 x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    return x ^ (x >> 16);
}

__global__ void k_fill_hash(u32 *p, size_t off_u32, size_t n_u32, u32 seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    for (; i < n_u32; i += st) p[off_u32 + i] = splitmix32(seed ^ (u32)(i * 2654435761u));
}

// fp16 uniform in [lo, hi).  Scales and activations must be BOUNDED and finite: a hash fill
// reinterpreted as fp16 produces inf/NaN, and a NaN anywhere makes every downstream bit-comparison
// trivially "equal" (NaN != NaN fails, so the comparison reports a mismatch) or trivially useless.
__global__ void k_fill_half_uniform(__half *p, size_t n, u32 seed, float lo, float hi) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    const float span = hi - lo;
    for (; i < n; i += st) {
        const u32 h = splitmix32(seed ^ (u32)(i * 2246822519u));
        p[i] = __float2half(lo + span * ((float)(h >> 8) * (1.0f / 16777216.0f)));
    }
}

// fp32 uniform in [lo, hi) -- topk_weights for the scatter arm.
__global__ void k_fill_float_uniform(float *p, size_t n, u32 seed, float lo, float hi) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    const float span = hi - lo;
    for (; i < n; i += st) {
        const u32 h = splitmix32(seed ^ (u32)(i * 3266489917u));
        p[i] = lo + span * ((float)(h >> 8) * (1.0f / 16777216.0f));
    }
}

__device__ __forceinline__ u32 sum4(uint4 v) { return v.x + v.y + v.z + v.w; }

__device__ __forceinline__ void wave_atomic_add(u32 acc, u32 *gsum) {
    for (int o = warpSize / 2; o > 0; o >>= 1) acc += __shfl_down(acc, o, warpSize);
    if ((threadIdx.x % warpSize) == 0) atomicAdd(gsum, acc);
}

// Linear stream read with an exact prefix checksum.  A wrong checksum means the read did not land
// on the pages we think it did, and the driver then refuses to report a bandwidth number -- this is
// the arm that decides "is the host stack actually on the far side of PCIe".
__global__ void k_linear_read(const uint4 *__restrict__ p, size_t n4, u32 *gsum) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    u32 acc = 0;
    for (; i < n4; i += st) acc += sum4(p[i]);
    wave_atomic_add(acc, gsum);
}

// Evict the L2 / MALL between timed reps.  WITHOUT THIS THE PROBE MEASURES NOTHING USEFUL: ten
// experts of gemm1 weights is ~15 MiB, which fits this card's 64 MB Infinity Cache outright, so a
// second rep on the same route would read host-resident weights out of on-die cache and report a
// host miss as free -- a fabricated "cliff at zero cost".  The driver ALSO redraws the routed
// expert set every rep, so this is belt and braces.
__global__ void k_thrash(const uint4 *__restrict__ p, size_t n4, u32 *gsum) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t st = (size_t)gridDim.x * blockDim.x;
    u32 acc = 0;
    for (; i < n4; i += st) acc += sum4(p[i]);
    wave_atomic_add(acc, gsum);
}

static u32 *g_sum[16] = {nullptr};

static int ensure_sum(u32 **out) {
    int dev = 0;
    CK(hipGetDevice(&dev));
    if (dev < 0 || dev >= 16) return P2P_EBADARG;
    if (g_sum[dev] == nullptr) {
        void *p = nullptr;
        CK(hipMalloc(&p, sizeof(u32)));
        g_sum[dev] = (u32 *)p;
    }
    *out = g_sum[dev];
    return 0;
}

// ===========================================================================
// Launch dispatch
// ===========================================================================
//
// The template parameter ladders below mirror the SHIPPED launchers
// (`launch_mmq_regdirect_w4a16_moe_gemv_silu_gfx1201` and `run_w4a16_moe_gemv`) so the probe
// launches the same instantiation production would for the same shape.  MMAX is restricted to
// {8, 16}: MMAX comes from `min(block_m, T)` and this probe is a DECODE probe (T <= 16), so the 32
// and 64 ladders are unreachable and instantiating them would only cost compile time.  The driver
// refuses T > 16 rather than silently clamping.

struct GemmArgs {
    const unsigned char* x;
    const int*    w;          // stack base (ignored when use_table, but still passed through)
    const __half* s;
    const int*    z;          // may be nullptr
    const int*    sti;
    const int*    eid;
    const int*    ntp;
    AT*           out;        // non-scatter output (P, N) or (P, inter) for SILU
    float*        out_scatter;
    const float*  topk_w;
    int T, N, K, group_size, top_k, block_m, num_valid, out_top_k;
    bool e2m1;
};

#define P2P_LAUNCH(LOADER, SILU, SCATTER, NW, MMAX, COLS, BYLANE, GRID, STREAM)              \
    gemv_decode::gemv_decode_core<AT, LOADER, /*GATHER=*/true, SILU, SCATTER,                \
                                  NW, MMAX, COLS, BYLANE>                                    \
        <<<(GRID), dim3((NW) * 32), 0, (STREAM)>>>(                                          \
            a.x, /*act_scales=*/nullptr, a.w, a.s, a.z, a.sti, a.eid, a.ntp, a.out,          \
            a.out_scatter, a.topk_w, a.T, a.N, a.K, a.group_size, a.top_k, a.block_m,        \
            a.num_valid, a.out_top_k, a.e2m1)

// ---- gemm1 + fused SiLU (SILU=true, SCATTER=false, COLS=1), both tilings ----
template <class L>
static int launch_gemm1(const GemmArgs& a, int nw, bool bylane, dim3 grid, int mmax,
                        hipStream_t stream) {
#define G1(NW, MM, BL) P2P_LAUNCH(L, true, false, NW, MM, 1, BL, grid, stream)
#define G1_MM(NW, BL) do { if (mmax == 8) G1(NW, 8, BL); else G1(NW, 16, BL); } while (0)
#define G1_NW(BL) do {                                                                       \
        if (nw == 4) G1_MM(4, BL);                                                           \
        else if (nw == 16) G1_MM(16, BL);                                                    \
        else if (nw == 32) G1_MM(32, BL);                                                    \
        else G1_MM(8, BL); } while (0)
    if (mmax != 8 && mmax != 16) return P2P_EUNSUP;
    if (nw != 4 && nw != 8 && nw != 16 && nw != 32) return P2P_EUNSUP;
    P2P_CLEAR_STICKY();
    if (bylane) G1_NW(true); else G1_NW(false);
    CK(hipGetLastError());
    return 0;
#undef G1_NW
#undef G1_MM
#undef G1
}

// ---- gemm2 (down-proj), K-on-lanes only, scatter optional ----
template <class L>
static int launch_gemm2(const GemmArgs& a, int nw, int cols, bool scatter, dim3 grid, int mmax,
                        hipStream_t stream) {
#define G2(NW, MM, CO, SC) P2P_LAUNCH(L, false, SC, NW, MM, CO, false, grid, stream)
#define G2_MM(NW, CO, SC) do { if (mmax == 8) G2(NW, 8, CO, SC); else G2(NW, 16, CO, SC); } while (0)
#define G2_CO(NW, SC) do {                                                                   \
        if (cols == 8) G2_MM(NW, 8, SC);                                                     \
        else if (cols == 4) G2_MM(NW, 4, SC);                                                \
        else if (cols == 2) G2_MM(NW, 2, SC);                                                \
        else G2_MM(NW, 1, SC); } while (0)
#define G2_NW(SC) do {                                                                       \
        if (nw == 4) G2_CO(4, SC);                                                           \
        else if (nw == 16) G2_CO(16, SC);                                                    \
        else if (nw == 32) G2_CO(32, SC);                                                    \
        else G2_CO(8, SC); } while (0)
    if (mmax != 8 && mmax != 16) return P2P_EUNSUP;
    if (nw != 4 && nw != 8 && nw != 16 && nw != 32) return P2P_EUNSUP;
    if (cols != 1 && cols != 2 && cols != 4 && cols != 8) return P2P_EUNSUP;
    P2P_CLEAR_STICKY();
    if (scatter) G2_NW(true); else G2_NW(false);
    CK(hipGetLastError());
    return 0;
#undef G2_NW
#undef G2_CO
#undef G2_MM
#undef G2
}

// ===========================================================================
// extern "C" surface
// ===========================================================================

extern "C" {

const char *p2p_err(int e) {
    switch (e) {
        case P2P_EBADARG: return "P2prime: bad argument";
        case P2P_EDIVIS:  return "P2prime: divisibility precondition violated";
        case P2P_ETOOBIG: return "P2prime: request too large";
        case P2P_EUNSUP:  return "P2prime: unsupported tiling (no instantiation for this shape)";
        case P2P_ETABLE:  return "P2prime: expert table too large (> P2P_MAX_E)";
        default: break;
    }
    if (e < 0) return "P2prime: unknown internal error";
    return hipGetErrorString((hipError_t)e);
}

int p2p_max_experts(void) { return P2P_MAX_E; }

int p2p_runtime_version(int *v) { CK(hipRuntimeGetVersion(v)); return 0; }
int p2p_driver_version(int *v)  { CK(hipDriverGetVersion(v));  return 0; }
int p2p_device_count(int *n)    { CK(hipGetDeviceCount(n));    return 0; }
int p2p_set_device(int d)       { CK(hipSetDevice(d));         return 0; }
int p2p_sync(void)              { CK(hipDeviceSynchronize());  return 0; }

int p2p_mem_info(size_t *freeb, size_t *totalb) { CK(hipMemGetInfo(freeb, totalb)); return 0; }

int p2p_device_info(int dev, char *name, int nlen, char *arch, int alen, char *pci, int plen,
                    int *mp_count, int *warp, int *clock_khz, unsigned long long *total_mem) {
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, dev));
    if (name && nlen > 0) snprintf(name, nlen, "%s", p.name);
    if (arch && alen > 0) snprintf(arch, alen, "%s", p.gcnArchName);
    if (pci && plen > 0)  CK(hipDeviceGetPCIBusId(pci, plen, dev));
    if (mp_count)  *mp_count = p.multiProcessorCount;
    if (warp)      *warp = p.warpSize;
    if (clock_khz) *clock_khz = p.clockRate;
    if (total_mem) *total_mem = (unsigned long long)p.totalGlobalMem;
    return 0;
}

int p2p_malloc_device(void **p, size_t bytes) { CK(hipMalloc(p, bytes)); return 0; }
int p2p_free_device(void *p)                  { CK(hipFree(p));          return 0; }
int p2p_host_malloc(void **p, size_t bytes, unsigned int flags) {
    CK(hipHostMalloc(p, bytes, flags));
    return 0;
}
int p2p_host_free(void *p) { CK(hipHostFree(p)); return 0; }
int p2p_host_dev_ptr(void **dp, void *hp) { CK(hipHostGetDevicePointer(dp, hp, 0)); return 0; }

// kind: 1 = H2D, 2 = D2H, 3 = D2D, 4 = Default
int p2p_memcpy(void *dst, const void *src, size_t n, int kind) {
    hipMemcpyKind k;
    switch (kind) {
        case 1: k = hipMemcpyHostToDevice; break;
        case 2: k = hipMemcpyDeviceToHost; break;
        case 3: k = hipMemcpyDeviceToDevice; break;
        default: k = hipMemcpyDefault; break;
    }
    CK(hipMemcpy(dst, src, n, k));
    return 0;
}

int p2p_memset(void *p, int v, size_t n) { CK(hipMemset(p, v, n)); CK(hipDeviceSynchronize()); return 0; }

int p2p_memcpy_bw(void *dst, const void *src, size_t bytes, int kind, int reps, float *out_ms) {
    if (!dst || !src || bytes == 0 || reps < 1) return P2P_EBADARG;
    hipMemcpyKind k;
    switch (kind) {
        case 1: k = hipMemcpyHostToDevice; break;
        case 2: k = hipMemcpyDeviceToHost; break;
        case 3: k = hipMemcpyDeviceToDevice; break;
        default: k = hipMemcpyDefault; break;
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

int p2p_fill_seq(void *p, size_t off_bytes, size_t len_bytes, unsigned int base_val) {
    if ((off_bytes & 3u) || (len_bytes & 3u)) return P2P_EDIVIS;
    const size_t n = len_bytes / 4;
    if (n == 0) return P2P_EBADARG;
    const int threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    P2P_CLEAR_STICKY();
    hipLaunchKernelGGL(k_fill_seq, dim3((unsigned)blocks), dim3(threads), 0, 0,
                       (u32 *)p, off_bytes / 4, n, base_val);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    return 0;
}

#define P2P_FILL_LAUNCH(KERN, N, ...) do {                                                   \
        const int _thr = 256;                                                                \
        size_t _blk = ((N) + _thr - 1) / _thr;                                               \
        if (_blk > 65535) _blk = 65535;                                                      \
        if (_blk == 0) return P2P_EBADARG;                                                   \
        P2P_CLEAR_STICKY();                                                                  \
        hipLaunchKernelGGL(KERN, dim3((unsigned)_blk), dim3(_thr), 0, 0, __VA_ARGS__);       \
        CK(hipGetLastError());                                                               \
        CK(hipDeviceSynchronize());                                                          \
    } while (0)

int p2p_fill_hash(void *p, size_t off_bytes, size_t len_bytes, unsigned int seed) {
    if ((off_bytes & 3u) || (len_bytes & 3u)) return P2P_EDIVIS;
    const size_t n = len_bytes / 4;
    P2P_FILL_LAUNCH(k_fill_hash, n, (u32 *)p, off_bytes / 4, n, seed);
    return 0;
}

int p2p_fill_half_uniform(void *p, size_t n, unsigned int seed, float lo, float hi) {
    if (!p || n == 0 || !(hi > lo)) return P2P_EBADARG;
    P2P_FILL_LAUNCH(k_fill_half_uniform, n, (__half *)p, n, seed, lo, hi);
    return 0;
}

int p2p_fill_float_uniform(void *p, size_t n, unsigned int seed, float lo, float hi) {
    if (!p || n == 0 || !(hi > lo)) return P2P_EBADARG;
    P2P_FILL_LAUNCH(k_fill_float_uniform, n, (float *)p, n, seed, lo, hi);
    return 0;
}

// Linear kernel read with prefix checksum.  `out_sum` must equal M*(M-1)/2 mod 2^32 with
// M = out_cover/4 when the region was filled by p2p_fill_seq(base_val = 0).
int p2p_read_bw(const void *base, size_t bytes, int blocks, int threads,
                unsigned int *out_sum, size_t *out_cover, float *out_ms) {
    if (!base || bytes < 16 || blocks < 1 || threads < 1) return P2P_EBADARG;
    if (threads % 32 != 0 || threads > 1024) return P2P_EBADARG;
    const size_t n4 = bytes / 16;
    if (n4 == 0) return P2P_EBADARG;
    u32 *gs = nullptr;
    int rc = ensure_sum(&gs);
    if (rc) return rc;
    CK(hipMemset(gs, 0, sizeof(u32)));
    CK(hipDeviceSynchronize());
    P2P_CLEAR_STICKY();
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0, 0));
    hipLaunchKernelGGL(k_linear_read, dim3((unsigned)blocks), dim3(threads), 0, 0,
                       (const uint4 *)base, n4, gs);
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
    if (out_cover) *out_cover = n4 * 16;
    if (out_ms) *out_ms = ms;
    return 0;
}

int p2p_thrash(const void *scratch, size_t bytes) {
    if (!scratch || bytes < 16) return P2P_EBADARG;
    u32 *gs = nullptr;
    int rc = ensure_sum(&gs);
    if (rc) return rc;
    P2P_CLEAR_STICKY();
    hipLaunchKernelGGL(k_thrash, dim3(4096), dim3(256), 0, 0,
                       (const uint4 *)scratch, bytes / 16, gs);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    return 0;
}

// Upload the per-expert two-stack pointer tables.  All three MUST be written together: scales and
// zero-points travel WITH the weights or an expert is dequantized against another expert's scale
// (plausible numbers, no crash).  `wz` may be NULL when the checkpoint has no zero-points.
int p2p_set_tables(const void **wq, const void **ws, const void **wz, int n) {
    if (!wq || !ws || n <= 0) return P2P_EBADARG;
    if (n > P2P_MAX_E) return P2P_ETABLE;
    CK(hipMemcpyToSymbol(HIP_SYMBOL(c_wq_tab), wq, (size_t)n * sizeof(void *)));
    CK(hipMemcpyToSymbol(HIP_SYMBOL(c_ws_tab), ws, (size_t)n * sizeof(void *)));
    if (wz) CK(hipMemcpyToSymbol(HIP_SYMBOL(c_wz_tab), wz, (size_t)n * sizeof(void *)));
    CK(hipDeviceSynchronize());
    return 0;
}

// Read one table slot back through a kernel-visible path so the driver can prove the upload landed
// (a hipMemcpyToSymbol that returns hipSuccess is not proof -- that is the exact class of lie that
// invalidated the first P2).
__global__ void k_read_tab(int idx, unsigned long long *out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = (unsigned long long)(uintptr_t)c_wq_tab[idx & P2P_E_MASK];
        out[1] = (unsigned long long)(uintptr_t)c_ws_tab[idx & P2P_E_MASK];
        out[2] = (unsigned long long)(uintptr_t)c_wz_tab[idx & P2P_E_MASK];
    }
}

int p2p_get_table_entry(int idx, unsigned long long *out3) {
    if (!out3 || idx < 0) return P2P_EBADARG;
    void *d = nullptr;
    CK(hipMalloc(&d, 3 * sizeof(unsigned long long)));
    P2P_CLEAR_STICKY();
    hipLaunchKernelGGL(k_read_tab, dim3(1), dim3(64), 0, 0, idx, (unsigned long long *)d);
    hipError_t le = hipGetLastError();
    if (le != hipSuccess) { (void)hipFree(d); return (int)le; }
    hipError_t ce = hipMemcpy(out3, d, 3 * sizeof(unsigned long long), hipMemcpyDeviceToHost);
    (void)hipFree(d);
    CK(ce);
    return 0;
}

// ---- the two timed entry points -------------------------------------------
//
// Both time `reps` INDEPENDENT launches, one hipEvent pair each, and return the per-rep ms array.
// The driver redraws the route and rewrites the table BETWEEN calls, so a "rep" here is a
// back-to-back burst on one fixed route; the cold, route-rotated sample is built by the driver
// calling this with reps=1 many times.  Both are recorded because they answer different questions
// (cold = the serving regime, hot = the cache-resident bound).

// gemm1 + fused SiLU: x (T, K) fp16, w13 (E, 2*inter, K/8) int32, scales (E, K/group, 2*inter) fp16,
// out (P, inter) fp16.  N == 2*inter.
int p2p_gemm1_silu(int use_table,
                   const void *x, const void *w, const void *s, const void *z,
                   const int *sti, const int *eid, const int *ntp, void *out,
                   int T, int N, int K, int group_size, int top_k, int block_m,
                   int num_valid, int e2m1,
                   int nw, int cols_unused, int bylane, int grid_x, int grid_y,
                   int reps, float *out_ms) {
    (void)cols_unused;
    if (!x || !sti || !eid || !ntp || !out || reps < 1 || !out_ms) return P2P_EBADARG;
    // `w`/`s` are STOCK-loader bases only.  In table mode TwoStackInt4A16GemvLoader::wq_expert /
    // ws_expert DISCARD them and return c_wq_tab[e] / c_ws_tab[e], so the driver passes NULL on
    // purpose (launch_raw's docstring: a null base makes an ignored table fault loudly instead of
    // quietly reading a valid stack and reporting a device-speed "host" number).  Demanding
    // s != NULL unconditionally rejected that deliberate, documented call -- w was already
    // exempted here and s was simply missed, so the table path could never launch.
    if (!use_table && (!w || !s)) return P2P_EBADARG;
    if (N % 2) return P2P_EDIVIS;
    if (K % 32 || group_size % 32 || block_m % 8 || block_m > 64) return P2P_EDIVIS;
    const int mreal_cap = (block_m < T) ? block_m : T;
    if (mreal_cap > 16) return P2P_EUNSUP;          // decode probe: MMAX ladder is {8, 16} only
    const int mmax = (mreal_cap <= 8) ? 8 : 16;

    GemmArgs a{};
    a.x = (const unsigned char *)x;
    a.w = (const int *)w;
    a.s = (const __half *)s;
    a.z = (const int *)z;
    a.sti = sti; a.eid = eid; a.ntp = ntp;
    a.out = (AT *)out;
    a.out_scatter = nullptr; a.topk_w = nullptr;
    a.T = T; a.N = N; a.K = K; a.group_size = group_size; a.top_k = top_k;
    a.block_m = block_m; a.num_valid = num_valid; a.out_top_k = top_k; a.e2m1 = (e2m1 != 0);

    const dim3 grid((unsigned)grid_x, (unsigned)grid_y);
    hipStream_t stream = 0;
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    for (int i = 0; i < reps; ++i) {
        CK(hipEventRecord(e0, stream));
        int rc = use_table ? launch_gemm1<TableLoader>(a, nw, bylane != 0, grid, mmax, stream)
                           : launch_gemm1<StockLoader>(a, nw, bylane != 0, grid, mmax, stream);
        if (rc) { (void)hipEventDestroy(e0); (void)hipEventDestroy(e1); return rc; }
        CK(hipEventRecord(e1, stream));
        CK(hipEventSynchronize(e1));
        CK(hipEventElapsedTime(&out_ms[i], e0, e1));
    }
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipDeviceSynchronize());
    return 0;
}

// gemm2 (down-proj): x (P, inter) fp16, w2 (E, hidden, inter/8) int32,
// scales (E, inter/group, hidden) fp16.  scatter=0 -> out (P, N) fp16 (BIT-COMPARABLE, used by the
// correctness arm); scatter=1 -> atomicAdd into out_scatter (T, N) fp32 (production decode shape,
// NOT bit-reproducible because the atomic reduction order varies -- timing arm only).
int p2p_gemm2(int use_table, int scatter,
              const void *x, const void *w, const void *s, const void *z,
              const int *sti, const int *eid, const int *ntp,
              void *out, float *out_scatter, const float *topk_w,
              int T, int N, int K, int group_size, int top_k, int block_m,
              int num_valid, int e2m1,
              int nw, int cols, int grid_x, int grid_y,
              int reps, float *out_ms) {
    if (!x || !sti || !eid || !ntp || reps < 1 || !out_ms) return P2P_EBADARG;
    if (!use_table && (!w || !s)) return P2P_EBADARG;   // stock-loader bases; see p2p_gemm1_silu
    if (scatter ? (out_scatter == nullptr || topk_w == nullptr) : (out == nullptr))
        return P2P_EBADARG;
    if (K % 32 || group_size % 32 || block_m % 8 || block_m > 64) return P2P_EDIVIS;
    const int mreal_cap = (block_m < T) ? block_m : T;
    if (mreal_cap > 16) return P2P_EUNSUP;
    const int mmax = (mreal_cap <= 8) ? 8 : 16;

    GemmArgs a{};
    a.x = (const unsigned char *)x;
    a.w = (const int *)w;
    a.s = (const __half *)s;
    a.z = (const int *)z;
    a.sti = sti; a.eid = eid; a.ntp = ntp;
    a.out = (AT *)out;
    a.out_scatter = out_scatter; a.topk_w = topk_w;
    a.T = T; a.N = N; a.K = K; a.group_size = group_size; a.top_k = top_k;
    a.block_m = block_m; a.num_valid = num_valid; a.out_top_k = top_k; a.e2m1 = (e2m1 != 0);

    const dim3 grid((unsigned)grid_x, (unsigned)grid_y);
    hipStream_t stream = 0;
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    for (int i = 0; i < reps; ++i) {
        if (scatter) CK(hipMemsetAsync(out_scatter, 0, (size_t)T * N * sizeof(float), stream));
        CK(hipEventRecord(e0, stream));
        int rc = use_table
            ? launch_gemm2<TableLoader>(a, nw, cols, scatter != 0, grid, mmax, stream)
            : launch_gemm2<StockLoader>(a, nw, cols, scatter != 0, grid, mmax, stream);
        if (rc) { (void)hipEventDestroy(e0); (void)hipEventDestroy(e1); return rc; }
        CK(hipEventRecord(e1, stream));
        CK(hipEventSynchronize(e1));
        CK(hipEventElapsedTime(&out_ms[i], e0, e1));
    }
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipDeviceSynchronize());
    return 0;
}

// Host-side tiling selection, taken from the SHIPPED table so the probe launches the instantiation
// production would.  Uses the production env prefix on purpose: if the served config overrides the
// tiling, the probe must inherit that override, and the driver records the env it saw.
int p2p_select_tiling(int K, int mreal_cap, int group_size, int *bylane, int *nwarps, int *cols) {
    if (!bylane || !nwarps || !cols) return P2P_EBADARG;
    const gemv_decode::GemvTiling t = gemv_decode::select_gemv_tiling(
        K, mreal_cap, group_size, /*k_per_lane_slot=*/32, /*bylane_available=*/true,
        "VLLM_W4A16_MOE_GEMV_");
    *bylane = t.bylane ? 1 : 0;
    *nwarps = t.nwarps;
    *cols   = t.cols;
    return 0;
}

// Direct placement query.  A QUERY CAN LIE -- this whole probe family exists because
// hipMemCreate(location=Host) returns hipSuccess for device memory -- so the driver cross-checks
// this against the VRAM / MemAvailable deltas AND the achieved bandwidth.  Never trusted alone.
int p2p_pointer_attrs(const void *p, int *mem_type, int *device, void **dev_ptr, void **host_ptr,
                      int *is_managed, unsigned int *flags) {
    hipPointerAttribute_t attr;
    memset(&attr, 0, sizeof(attr));
    hipError_t e = hipPointerGetAttributes(&attr, p);
    if (e != hipSuccess) return (int)e;
    if (mem_type)   *mem_type = (int)attr.type;
    if (device)     *device = attr.device;
    if (dev_ptr)    *dev_ptr = attr.devicePointer;
    if (host_ptr)   *host_ptr = attr.hostPointer;
    if (is_managed) *is_managed = attr.isManaged;
    if (flags)      *flags = attr.allocationFlags;
    return 0;
}

}  // extern "C"
