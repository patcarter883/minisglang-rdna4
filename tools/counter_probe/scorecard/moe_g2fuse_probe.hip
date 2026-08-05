// moe_g2fuse_probe.hip — torch-free standalone launcher for the PRODUCTION fused MoE decode
// gemm2 kernel `moe_gemm2_gather_reduce_core`, so it can be profiled with rocprofv3 hardware
// counters.
//
// WHY IT EXISTS. The kernel lives in
//   /home/pat/code/rdna4-hip-kernels-cscore/fp8_wmma/fp8_wmma_rocm/moe_kernel.hip
// (git worktree @ 2ade7c1), which `#include <ATen/ATen.h>`s — so it CANNOT be included from a
// torch-free harness, and a .so built against one image does not load in another. The two device
// bodies this probe needs (`moe_gemm2_gather_reduce_core` and its `moe_build_inverse_map_kernel`
// prelude) depend on NOTHING but gemv_decode.h, so they are COPIED HERE VERBATIM, byte for byte,
// inside the same `namespace w4a8_fp8_moe`. DO NOT "clean up", reformat or restructure them: any
// edit invalidates the measurement, because the point of this harness is that the instructions
// profiled here are the instructions the serve runs.
//
// WHAT IS TRANSCRIBED vs INFERRED (read this before trusting a number):
//   TRANSCRIBED VERBATIM from launch_mmq_fp8_moe_gemm2_gather_reduce_gfx1201 (moe_kernel.hip:1527):
//     - constexpr int NW = 8; cols = 4 (env VLLM_W4A8_MOE_G2FUSE_COLS, validated to {1,2,4,8});
//     - per_block = NW*cols; grid = dim3(ceil(N/per_block), M); block = dim3(NW*32);
//     - the BYLANE gate: select_gemv_tiling(K, /*mreal_cap=*/1, group_size, /*k_per_lane_slot=*/32,
//       /*bylane_available=*/true, "VLLM_W4A8_MOE_G2FUSE_", bylane_min_group_v<Int4Fp8GemvLoader>)
//       .bylane && K <= 320  ->  core<OT, Int4Fp8GemvLoader, NW, 1, /*BYLANE=*/true> with
//       bgrid = dim3(ceil(N/(NW*32)), M) == dim3(ceil(N/256), M), same block;
//     - the inverse-map prelude grid: dim3((P+255)/256) x dim3(256).
//     select_gemv_tiling itself is CALLED, not re-implemented — it is a host-inline in gemv_decode.h
//     and is torch-free, so the probe gets the production table (and its env overrides) exactly.
//   INFERRED / SUBSTITUTED (unavoidable in a torch-free probe — all of it is INPUT, not tiling):
//     - the act-quant prelude (moe_compute_act_fp8_kernel) is NOT run; x_fp8/act_scales are filled
//       directly with plausible e4m3 bytes / 1.0f. It is a separate kernel dispatch in production
//       and is not the kernel under test; skipping it changes no launch parameter of the fused
//       kernel, only the VALUES in x_fp8 (which do not affect counters — shapes and access patterns
//       do). If you ever profile with --pmc and see the act-quant kernel in a serve trace, it is
//       that separate dispatch, not this one.
//     - the routing metadata is SYNTHETIC round-robin (slot i -> expert i % E), not a real router
//       histogram. This fixes the expert-block layout and therefore the weight-row footprint the
//       kernel streams; a real skewed histogram would reuse expert weights across slots and could
//       look BETTER in L2. Round-robin is the pessimal/most-uniform case and is stated here so a
//       reader knows the number is not a serve replay.
//     - topk_weights are 1/top_k, act_scales 1.0f, weights a repeating byte pattern.
//     - OT (the intermediate rounding dtype) defaults to __hip_bfloat16, the served dtype;
//       G2FUSE_OT=fp16 selects __half. In production this is dispatched off x.scalar_type().
//
// NOTE on the output buffer: the FUSED kernel does NOT atomicAdd into `out` — both the BYLANE and
// the K-on-lanes path do a plain `out[t*N+nc] = red;` store (the atomic scatter belongs to the OLD
// unfused gemm2 epilogue this kernel replaces). `out` is zeroed anyway: it is cheap, it makes the
// tail columns (nc >= N is skipped, so grid-tail elements are never written) deterministic, and it
// keeps the probe honest if the kernel's epilogue ever changes back.
//
// BUILD (CPU only, no GPU needed):
//   docker run --rm -v /home/pat/code/minisgl-rdna4-cscore/tools/counter_probe:/probe:ro \
//     -v /home/pat/code/rdna4-hip-kernels-cscore:/kern:ro \
//     --entrypoint bash rocm/dev-ubuntu-24.04:7.14.0-full -lc \
//     'hipcc -O3 --offload-arch=gfx1201 -I/kern/fp8_wmma/fp8_wmma_rocm -o /tmp/g2 \
//        /probe/scorecard/moe_g2fuse_probe.hip'
//
// RUN:  ./g2 [M E top_k K N group_size block_m [iters]]
//       defaults:  1 32 8 512 2048 128 16      (iters defaults to 3)
//       K=512 -> K-on-lanes (COLS=4);  K=256 (the SERVED gemm2 K) -> BYLANE.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gemv_decode.h"

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) {                      \
    std::fprintf(stderr, "[g2fuse] HIP error %s at %s:%d\n",                                \
                 hipGetErrorString(_e), __FILE__, __LINE__); std::exit(1); } } while (0)

static void fatal(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void fatal(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::fprintf(stderr, "[g2fuse] FATAL: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::exit(1);
}

// ============================================================================
// ==== BEGIN VERBATIM COPY from moe_kernel.hip @ e82e2eb — DO NOT EDIT ====
// Re-extracted mechanically from the live source (not re-typed) when the top_k split on
// grid.z landed, so the probe measures the kernel the serve runs, split included.
// Both bodies below are byte-for-byte the production sources (moe_kernel.hip lines ~899-908 and
// ~973-1066), kept inside their original `namespace w4a8_fp8_moe`. Only the surrounding torch
// launchers were dropped. Any change here silently invalidates every counter this probe collects.
// ============================================================================
namespace w4a8_fp8_moe {

// `arrival` is the fused gemm2's k-slice arrival counter (see the SPLIT note on
// moe_gemm2_gather_reduce_core). It is zeroed HERE, inside a kernel that already launches, rather
// than by its own at::zeros — a per-MoE-layer fill dispatch is exactly the launch-count cost this
// file spends paragraphs removing elsewhere, and the counter must be re-zeroed on every cudagraph
// REPLAY, which a captured prelude kernel gives for free. nullptr/0 when the slice count is 1.
static __global__ void moe_build_inverse_map_kernel(
    const int* __restrict__ sorted_token_ids,       // (P,)
    const int* __restrict__ num_tokens_post_padded,  // (1,)
    int*       __restrict__ meta,                    // [0,M_tk) slot_to_row, pre-filled -1;
                                                     // [M_tk, M_tk+n_arrival) k-slice counters
    int P, int M_tk, int n_arrival = 0) {
    // ONE restrict pointer for both regions ON PURPOSE. They live in the same allocation (the
    // counters ride in slot_to_row's tail so a split costs no extra dispatch), so handing the
    // compiler two `__restrict__` pointers into it would be an aliasing LIE: UB, and the kind that
    // pays out as dropped stores under a vectoriser rather than as a fault. Derived from one
    // pointer, the question does not arise.
    // (Honest provenance: this was written while chasing a max|d| 1.3e-1 that appeared only at
    // M >= 16. That turned out to be the PARITY HARNESS, not this — it compared legs run in
    // separate PROCESSES, and moe_align's row assignment within an expert block is not stable
    // across processes, so the two legs were not reading the same buf2 rows. The aliasing removal
    // is kept because it is correct, NOT because it was measured to fix anything.)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_arrival;
         i += gridDim.x * blockDim.x)
        meta[M_tk + i] = 0;
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= P || r >= num_tokens_post_padded[0]) return;
    const int slot = sorted_token_ids[r];
    if (slot < M_tk) meta[slot] = r;                 // valid (token,slot) -> its row
}

template <typename OT, class WLoad, int NWARPS, int COLS, bool BYLANE = false>
static __global__ void __launch_bounds__(NWARPS * 32) moe_gemm2_gather_reduce_core(
    const unsigned char*           __restrict__ x_fp8,        // (P, K) act-quant fp8, row = padded row
    const float*                   __restrict__ act_scales,   // (P,) per-padded-row act scale
    const typename WLoad::WT*      __restrict__ w_data,       // (E, N, K/8) int32 packed
    const typename WLoad::WScaleT* __restrict__ w_scales,     // (E, N, K/group) fp16
    const int*                     __restrict__ w_zeros,      // (E, N/8, K/group) int32 or nullptr
    const int*                     __restrict__ expert_ids,   // (P/block_m,) int32
    int*                           __restrict__ slot_to_row,  // (M*top_k [+ M*gridDim.x],) int32
    const float*                   __restrict__ topk_weights, // (M*top_k,) fp32
    float*                         __restrict__ out,          // (M, N) fp32
    int M, int N, int K, int group_size, int top_k, int block_m, bool weight_is_e2m1,
    float*                         __restrict__ partial = nullptr,  // (top_k, M, N) fp32; k_slices>1
    int k_slices = 1) {
    // The k-slice arrival counters are the TAIL of slot_to_row, derived from the same pointer —
    // never a second __restrict__ into the same allocation (that would be UB). See the prelude.
    int* arrival = slot_to_row + M * top_k;

    const int t       = blockIdx.y;          // token
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;

    const int num_groups = K / group_size;
    const int col_base   = (blockIdx.x * NWARPS + warp_id) * COLS;
    const int sbase      = t * top_k;
    // k_slices is grid-uniform, so every branch on it is uniform across the block — no divergence,
    // and the __syncthreads() in the fixup is reached by all threads or by none.
    const int kz    = (k_slices > 1) ? (int)blockIdx.z : 0;
    const long pbase = (long)t * N;                  // partial[k][t][*] = partial + k*M*N + t*N

    __shared__ int s_last;

    if constexpr (BYLANE) {
        // One column per LANE (COLS unused): each lane sweeps ALL of K in registers, so there is no
        // cross-lane reduce and no minimum K. Same epilogue order as the K-on-lanes path — the
        // ascending-k fp32 fold of tw * (float)cvt_out<OT>(acc*asc*wsc) — so only the dot's internal
        // summation order differs.
        const int nc = (blockIdx.x * NWARPS + warp_id) * 32 + lane;
        float red1 = 0.0f;
        for (int k = kz; k < top_k; k += k_slices) {
            const int r = slot_to_row[sbase + k];
            if (r < 0) {                                      // missing slot -> contributes 0
                if (k_slices > 1 && nc < N)
                    __hip_atomic_store(&partial[(long)k * M * N + pbase + nc], 0.0f,
                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                continue;
            }
            const int e = expert_ids[r / block_m];
            const typename WLoad::WT*      wq_e = WLoad::wq_expert(w_data,   e, N, K);
            const typename WLoad::WScaleT* ws_e = WLoad::ws_expert(w_scales, e, N, num_groups);
            const int*                     wz_e = WLoad::wz_expert(w_zeros,  e, N, num_groups);
            const unsigned char* x_r = x_fp8 + (long)r * K;
            const float          asc = act_scales[r];
            const float          tw  = topk_weights[sbase + k];
            if (nc < N) {
                float acc[1] = {0.0f};
                WLoad::template accum_bylane<1, /*GATHER=*/false>(
                    acc, 1, x_r, /*s_src=*/nullptr, wq_e, ws_e, wz_e, nc, K, group_size,
                    weight_is_e2m1);
                const float wsc = WLoad::wscale_epi(ws_e, nc);
                const OT rounded = gemv_decode::cvt_out<OT>(acc[0] * asc * wsc);
                const float term = tw * (float)rounded;
                if (k_slices > 1)
                    __hip_atomic_store(&partial[(long)k * M * N + pbase + nc], term,
                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                else red1 += term;
            }
        }
        if (k_slices == 1) {
            if (nc < N) out[pbase + nc] = red1;
            return;
        }
        // Last slice to arrive folds the token's top_k terms in ASCENDING k — the identical left
        // fold the unsplit path does in `red1`, hence max|Δ|=0.
        __threadfence();                 // RELEASE: this block's partial writes, before it arrives
        __syncthreads();
        if (threadIdx.x == 0)
            s_last = (__hip_atomic_fetch_add(&arrival[t * (int)gridDim.x + (int)blockIdx.x], 1,
                                             __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT)
                      == k_slices - 1);
        __syncthreads();
        if (!s_last) return;
        __threadfence();                 // ACQUIRE: see the OTHER slices' partials, not a stale L1
        if (nc < N) {
            float red = 0.0f;
            for (int k = 0; k < top_k; ++k)
                red += __hip_atomic_load(&partial[(long)k * M * N + pbase + nc],
                                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            out[pbase + nc] = red;
        }
        return;
    }

    // per-column fp32 reduce accumulator (only lane 0 holds the real running sum).
    float red[COLS];
    #pragma unroll
    for (int c = 0; c < COLS; ++c) red[c] = 0.0f;

    // OUTER over the token's top_k experts (ascending k == gather_reduce fold order).
    for (int k = kz; k < top_k; k += k_slices) {
        const int r = slot_to_row[sbase + k];
        if (r < 0) {                                          // missing slot -> contributes 0
            if (k_slices > 1 && lane == 0) {
                #pragma unroll
                for (int c = 0; c < COLS; ++c) {
                    const int nc = col_base + c;
                    if (nc < N)
                        __hip_atomic_store(&partial[(long)k * M * N + pbase + nc], 0.0f,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                }
            }
            continue;
        }
        const int e = expert_ids[r / block_m];
        const typename WLoad::WT*      wq_e = WLoad::wq_expert(w_data,   e, N, K);
        const typename WLoad::WScaleT* ws_e = WLoad::ws_expert(w_scales, e, N, num_groups);
        const int*                     wz_e = WLoad::wz_expert(w_zeros,  e, N, num_groups);
        // pre-sorted buf2: padded row r reads its OWN activation row (identity gather,
        // like the engine's gemm2 with top_k=1 -> s_src = arange(P)[r]/1 == r).
        const unsigned char* x_r = x_fp8 + (long)r * K;
        const float          asc = act_scales[r];
        const float          tw  = topk_weights[sbase + k];
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const int nc = col_base + c;
            float acc[1] = {0.0f};
            if (nc < N)
                WLoad::template accum<1, /*GATHER=*/false>(
                    acc, 1, x_r, /*s_src=*/nullptr, wq_e, ws_e, wz_e, nc, K, group_size, lane, weight_is_e2m1);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc[0] += __shfl_xor(acc[0], off);
            if (lane == 0 && nc < N) {
                const float wsc = WLoad::wscale_epi(ws_e, nc);
                const OT rounded = gemv_decode::cvt_out<OT>(acc[0] * asc * wsc);
                const float term = tw * (float)rounded;
                if (k_slices > 1)
                    __hip_atomic_store(&partial[(long)k * M * N + pbase + nc], term,
                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                else red[c] += term;
            }
        }
    }
    if (k_slices == 1) {
        if (lane == 0) {
            #pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int nc = col_base + c;
                if (nc < N) out[pbase + nc] = red[c];
            }
        }
        return;
    }
    __threadfence();                     // RELEASE (see the BYLANE twin above)
    __syncthreads();
    if (threadIdx.x == 0)
        s_last = (__hip_atomic_fetch_add(&arrival[t * (int)gridDim.x + (int)blockIdx.x], 1,
                                         __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT)
                  == k_slices - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();                     // ACQUIRE
    if (lane == 0) {
        #pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const int nc = col_base + c;
            if (nc < N) {
                float acc2 = 0.0f;
                for (int k = 0; k < top_k; ++k)
                    acc2 += __hip_atomic_load(&partial[(long)k * M * N + pbase + nc],
                                              __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                out[pbase + nc] = acc2;
            }
        }
    }
}

}  // namespace w4a8_fp8_moe

// ==== END VERBATIM COPY ====

// ============================================================================
// Harness
// ============================================================================
using gemv_decode::Int4Fp8GemvLoader;

// The template's per-call accumulator cap. The fused core declares `float acc[1]` and passes
// M_real=1 to WLoad::accum/accum_bylane, i.e. its MMAX-like compile-time bound is 1 and the real
// rows it accumulates per call is also 1 (identity gather: padded row r reads its OWN activation).
// See check_caps() — this is asserted at runtime rather than assumed, because the last time a
// compile-time array bound was smaller than the real rows per block, the launch took a Memory
// Fault and cost a GPU window.
static constexpr int ACC_MMAX        = 1;
static constexpr int ROWS_PER_ACCUM  = 1;

template <class T>
static T* dalloc_bytes(size_t n, int fill) {
    T* p = nullptr;
    HIP_CHECK(hipMalloc(&p, n * sizeof(T)));
    HIP_CHECK(hipMemset(p, fill, n * sizeof(T)));
    return p;
}

template <class T>
static T* dupload(const std::vector<T>& h) {
    T* p = nullptr;
    HIP_CHECK(hipMalloc(&p, h.size() * sizeof(T)));
    HIP_CHECK(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

// ---- Pre-launch cap / shape guard. Every check here is a launch-time INVARIANT of the copied
// kernel body or of the loader it calls; violating one is an out-of-bounds access, not a wrong
// answer. Print and exit(1) — never launch and take a Memory Fault.
static void check_caps(int M, int E, int top_k, int K, int N, int group_size, int block_m,
                       int P, int nblk, bool bylane, int cols) {
    if (ACC_MMAX < ROWS_PER_ACCUM)
        fatal("acc[%d] compile-time cap < rows accumulated per call (%d). The kernel would write "
              "past its accumulator. Refusing to launch.", ACC_MMAX, ROWS_PER_ACCUM);
    if (M <= 0 || E <= 0 || top_k <= 0 || K <= 0 || N <= 0 || group_size <= 0 || block_m <= 0)
        fatal("all of M E top_k K N group_size block_m must be > 0 "
              "(got %d %d %d %d %d %d %d).", M, E, top_k, K, N, group_size, block_m);
    // WLoad::accum / accum_bylane read a v4i_t (4 x int32 = 16 B) at &wq[base] with base a multiple
    // of 4, over ppr = K/PACK_FACTOR words. ppr % 4 != 0 => the LAST chunk reads past the row end
    // AND the 16-B load is misaligned.
    if (K % 8 != 0)
        fatal("K=%d is not a multiple of PACK_FACTOR=8: the int4 row is K/8 int32 words.", K);
    const int ppr = K / 8;
    if (ppr % 4 != 0)
        fatal("K=%d -> ppr=K/8=%d is not a multiple of 4. The loader sweeps 4 int32 (32 nibbles) "
              "per step and would read past the weight row. Use K %% 32 == 0.", K, ppr);
    if (K % group_size != 0)
        fatal("K=%d is not a multiple of group_size=%d (num_groups = K/group_size).", K, group_size);
    const int num_groups = K / group_size;
    if (group_size < gemv_decode::bylane_min_group_v<Int4Fp8GemvLoader>)
        fatal("group_size=%d < Int4Fp8GemvLoader::bylane_min_group_v=%d.",
              group_size, gemv_decode::bylane_min_group_v<Int4Fp8GemvLoader>);
    // group_size == 16 makes each 32-k chunk straddle TWO scale groups; the loader then reads
    // ws[g+1], so num_groups must be even and >= 2.
    if (group_size < 32 && num_groups < 2)
        fatal("group_size=%d (<32) folds ws[g] and ws[g+1] per chunk, but num_groups=%d.",
              group_size, num_groups);
    if (P % block_m != 0)
        fatal("P=%d is not a multiple of block_m=%d — the routing metadata is malformed.",
              P, block_m);
    if (nblk != P / block_m)
        fatal("expert_ids has %d entries but P/block_m = %d.", nblk, P / block_m);
    // expert_ids[r/block_m] with r < P must stay inside the (P/block_m,) array, and each entry must
    // name a real expert (wq_expert multiplies it by N*K/8).
    if (nblk <= 0) fatal("nblk=%d: no expert blocks were built.", nblk);
    if (bylane && cols != 1)
        fatal("BYLANE must be instantiated with COLS=1 (COLS multiplies the resident weight-row "
              "footprint and is unused on that path); got COLS=%d.", cols);
    if (!bylane && cols != 1 && cols != 2 && cols != 4 && cols != 8)
        fatal("COLS=%d is not one of {1,2,4,8}.", cols);
    (void)E; (void)top_k; (void)N;
}

// ---- One launch of the (prelude + fused) pair, at the production tiling. Templated on OT and the
// COLS/BYLANE the production launcher selected.
struct Args {
    const unsigned char* x_fp8;
    const float*  act_scales;
    const int*    w_data;
    const __half* w_scales;
    const int*    w_zeros;        // nullptr — the served W4A8 checkpoints are symmetric
    const int*    expert_ids;
    int*          slot_to_row;    // (M*top_k [+ M*gridDim.x]) — the arrival counters are its tail
    const float*  topk_weights;
    float*        out;
    int M, N, K, group_size, top_k, block_m;
    bool weight_is_e2m1;
    float*        partial;        // (top_k, M, N) fp32, nullptr when k_slices == 1
    int           k_slices;
};

template <typename OT, int COLS, bool BYLANE>
static void launch_fused(const Args& a, dim3 grid, dim3 block) {
    w4a8_fp8_moe::moe_gemm2_gather_reduce_core<OT, Int4Fp8GemvLoader, 8, COLS, BYLANE>
        <<<grid, block>>>(a.x_fp8, a.act_scales, a.w_data, a.w_scales, a.w_zeros, a.expert_ids,
                          a.slot_to_row, a.topk_weights, a.out, a.M, a.N, a.K, a.group_size,
                          a.top_k, a.block_m, a.weight_is_e2m1, a.partial, a.k_slices);
}

template <typename OT>
static void launch_dispatch(const Args& a, dim3 grid, dim3 block, bool bylane, int cols) {
    if (bylane) { launch_fused<OT, 1, true>(a, grid, block); return; }
    if (cols == 8)      launch_fused<OT, 8, false>(a, grid, block);
    else if (cols == 4) launch_fused<OT, 4, false>(a, grid, block);
    else if (cols == 2) launch_fused<OT, 2, false>(a, grid, block);
    else                launch_fused<OT, 1, false>(a, grid, block);
}

int main(int argc, char** argv) {
    // argv: M E top_k K N group_size block_m [iters]
    const int M          = (argc > 1) ? std::atoi(argv[1]) : 1;
    const int E          = (argc > 2) ? std::atoi(argv[2]) : 32;
    const int top_k      = (argc > 3) ? std::atoi(argv[3]) : 8;
    const int K          = (argc > 4) ? std::atoi(argv[4]) : 512;
    const int N          = (argc > 5) ? std::atoi(argv[5]) : 2048;
    const int group_size = (argc > 6) ? std::atoi(argv[6]) : 128;
    const int block_m    = (argc > 7) ? std::atoi(argv[7]) : 16;
    const int iters      = (argc > 8) ? std::atoi(argv[8]) : 3;   // keep SMALL: --pmc serializes

    const bool use_fp16 = [] {
        const char* e = std::getenv("G2FUSE_OT");
        return e != nullptr && std::string(e) == "fp16";
    }();

    // ---- device props. multiProcessorCount on gfx1201 reports WGPs (32 on the 9070 XT / 9070),
    // NOT CUs. Never assert 64 here: gemv_decode::warn_if_under_occupied compares grid blocks to
    // this same number, so a "64-CU" assumption would mis-read every occupancy verdict.
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    std::printf("[g2fuse] device %d: %s (%s)\n", dev, prop.name, prop.gcnArchName);
    std::printf("[g2fuse]   multi_processor_count = %d  (gfx1201 reports WGPs, not CUs -- 32 on the "
                "9070 XT; do NOT assert 64)\n", prop.multiProcessorCount);
    std::printf("[g2fuse]   warpSize=%d  maxThreadsPerBlock=%d  sharedMemPerBlock=%zu  "
                "regsPerBlock=%d  clockRate=%d kHz  totalGlobalMem=%.2f GiB\n",
                prop.warpSize, prop.maxThreadsPerBlock, (size_t)prop.sharedMemPerBlock,
                prop.regsPerBlock, prop.clockRate, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // ---- routing metadata. CONTRACT (moe_align_block_size): sorted_token_ids holds FLAT topk
    // indices 0..numel-1 (numel = M*top_k) grouped into per-expert CONTIGUOUS runs, each run padded
    // up to a multiple of block_m with the SENTINEL value `numel`; expert_ids[b] names the expert
    // owning block b; num_tokens_post_padded[0] = P = sorted_token_ids.size(). Slots are assigned
    // round-robin (slot i -> expert i % E) -- see the INFERRED note at the top of this file.
    const int numel = M * top_k;
    std::vector<std::vector<int>> per_expert(E);
    for (int i = 0; i < numel; ++i) per_expert[i % E].push_back(i);

    std::vector<int> h_sti;     // (P,)
    std::vector<int> h_eid;     // (P/block_m,)
    for (int e = 0; e < E; ++e) {
        const int cnt = (int)per_expert[e].size();
        if (cnt == 0) continue;                                   // experts with no tokens emit no block
        const int padded = ((cnt + block_m - 1) / block_m) * block_m;
        for (int j = 0; j < padded; ++j)
            h_sti.push_back(j < cnt ? per_expert[e][j] : numel);   // SENTINEL == numel
        for (int b = 0; b < padded / block_m; ++b) h_eid.push_back(e);
    }
    const int P    = (int)h_sti.size();
    const int nblk = (int)h_eid.size();
    const std::vector<int> h_ntp{P};

    // ---- production tiling, transcribed from launch_mmq_fp8_moe_gemm2_gather_reduce_gfx1201.
    constexpr int NW = 8;
    int cols = 4;
    if (const char* e = std::getenv("VLLM_W4A8_MOE_G2FUSE_COLS")) cols = std::atoi(e);
    if (cols != 1 && cols != 2 && cols != 4 && cols != 8) cols = 4;
    const int per_block = NW * cols;
    dim3 grid((N + per_block - 1) / per_block, M);
    dim3 block(NW * 32);

    const gemv_decode::GemvTiling tl = gemv_decode::select_gemv_tiling(
        K, /*mreal_cap=*/1, group_size, /*k_per_lane_slot=*/32, /*bylane_available=*/true,
        "VLLM_W4A8_MOE_G2FUSE_",
        gemv_decode::bylane_min_group_v<gemv_decode::Int4Fp8GemvLoader>);
    const bool bylane = (tl.bylane && K <= 320);   // MEASURED crossover ~300, NOT the table's K<=1024
    int eff_cols = cols;
    if (bylane) {
        const int per_block_bl = NW * 32;          // one column per lane
        grid = dim3((N + per_block_bl - 1) / per_block_bl, M);
        eff_cols = 1;                              // BYLANE is instantiated with COLS=1
        gemv_decode::warn_if_under_occupied("moe_gemm2_gather_reduce[bylane]", grid, NW * 32);
    }

    std::printf("[g2fuse] shape: M=%d E=%d top_k=%d K=%d N=%d group_size=%d block_m=%d "
                "-> numel=%d P=%d nblk=%d  (OT=%s)\n",
                M, E, top_k, K, N, group_size, block_m, numel, P, nblk,
                use_fp16 ? "__half" : "__hip_bfloat16");
    std::printf("[g2fuse] BRANCH = %s  |  moe_gemm2_gather_reduce_core<%s, Int4Fp8GemvLoader, "
                "NWARPS=%d, COLS=%d, BYLANE=%s>\n",
                bylane ? "BYLANE (N-on-lanes)" : "K-on-lanes (COLS)",
                use_fp16 ? "__half" : "__hip_bfloat16", NW, eff_cols, bylane ? "true" : "false");
    std::printf("[g2fuse]   tiling table said bylane=%d nwarps=%d cols=%d; gate is "
                "(tl.bylane && K<=320) -> %d\n", (int)tl.bylane, tl.nwarps, tl.cols, (int)bylane);
    std::printf("[g2fuse]   grid=(%u,%u,%u)  block=(%u,%u,%u)  = %ld blocks x %u threads\n",
                grid.x, grid.y, grid.z, block.x, block.y, block.z,
                (long)grid.x * grid.y * grid.z, block.x);

    check_caps(M, E, top_k, K, N, group_size, block_m, P, nblk, bylane, eff_cols);

    // ---- buffers. Sizes are the production ones; VALUES are synthetic (see header note).
    auto* d_x_fp8 = dalloc_bytes<unsigned char>((size_t)P * K, 0x38);         // e4m3 ~1.0
    auto* d_wdata = dalloc_bytes<int>((size_t)E * N * (K / 8), 0x11);          // int4 nibbles = 1
    auto* d_wscal = dalloc_bytes<__half>((size_t)E * N * (K / group_size), 0x3c);  // ~1.06 in fp16
    const int* d_wzero = nullptr;                                             // symmetric int4

    std::vector<float> h_asc((size_t)P, 1.0f);
    std::vector<float> h_tw((size_t)numel, 1.0f / (float)top_k);
    std::vector<int>   h_str((size_t)numel, -1);                              // slot_to_row pre-filled -1

    auto* d_asc  = dupload(h_asc);
    auto* d_tw   = dupload(h_tw);
    auto* d_str  = dupload(h_str);   // resized below when the k-slice split is on
    auto* d_sti  = dupload(h_sti);
    auto* d_eid  = dupload(h_eid);
    auto* d_ntp  = dupload(h_ntp);

    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, (size_t)M * N * sizeof(float)));
    HIP_CHECK(hipMemset(d_out, 0, (size_t)M * N * sizeof(float)));

    // ---- (2) inverse map: slot -> padded row. The fused kernel takes slot_to_row, NOT
    // sorted_token_ids. Same launch config as the production prelude.
    // ---- k-slice split on grid.z (G2FUSE_SPLIT_K; 1 = the pre-split grid, for the "before" row).
    // The probe takes the count EXPLICITLY rather than re-deriving it, because a counter run wants
    // to name the configuration it is measuring, not discover it.
    int k_slices = 1;
    if (const char* e = std::getenv("G2FUSE_SPLIT_K")) k_slices = std::atoi(e);
    if (k_slices < 1) k_slices = 1;
    while (k_slices > 1 && top_k % k_slices != 0) k_slices--;   // slices must divide top_k
    const int n_arrival = (k_slices > 1) ? (int)grid.x * M : 0;

    // slot_to_row carries the arrival counters in its TAIL (production layout: one allocation, one
    // pointer, zeroed by the prelude that already launches).
    if (n_arrival > 0) {
        HIP_CHECK(hipFree(d_str));
        std::vector<int> h_str2(h_str);
        h_str2.resize((size_t)numel + n_arrival, -1);
        d_str = dupload(h_str2);
    }
    float* d_partial = nullptr;
    if (k_slices > 1) {
        HIP_CHECK(hipMalloc(&d_partial, (size_t)top_k * M * N * sizeof(float)));
        HIP_CHECK(hipMemset(d_partial, 0, (size_t)top_k * M * N * sizeof(float)));
    }
    grid.z = k_slices;

    w4a8_fp8_moe::moe_build_inverse_map_kernel<<<dim3((P + 255) / 256), dim3(256)>>>(
        d_sti, d_ntp, d_str, P, numel, n_arrival);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());

    // Post-hoc validation of the map: every slot must have found a real padded row in [0,P).
    // A -1 here would mean the routing metadata is malformed (the kernel tolerates it by skipping,
    // which would silently under-count the work being profiled).
    {
        std::vector<int> back((size_t)numel, -2);
        HIP_CHECK(hipMemcpy(back.data(), d_str, (size_t)numel * sizeof(int), hipMemcpyDeviceToHost));
        int missing = 0, oob = 0;
        for (int i = 0; i < numel; ++i) {
            if (back[i] < 0) ++missing;
            else if (back[i] >= P) ++oob;
        }
        if (oob) fatal("slot_to_row has %d entries >= P=%d — x_fp8/act_scales would be read out of "
                       "bounds. Refusing to launch.", oob, P);
        if (missing)
            std::printf("[g2fuse] WARNING: %d/%d slots have no padded row (contribute 0). The "
                        "profiled work is smaller than the shape implies.\n", missing, numel);
        else
            std::printf("[g2fuse] slot_to_row: %d/%d slots mapped, all in [0,%d).\n",
                        numel, numel, P);
    }

    std::printf("[g2fuse] SPLIT: k_slices=%d -> grid=(%u,%u,%u) = %u blocks, %u waves\n",
                k_slices, grid.x, grid.y, grid.z, grid.x * grid.y * grid.z,
                grid.x * grid.y * grid.z * NW);
    Args a{d_x_fp8, d_asc, d_wdata, d_wscal, d_wzero, d_eid, d_str, d_tw, d_out,
           M, N, K, group_size, top_k, block_m, /*weight_is_e2m1=*/false, d_partial, k_slices};

    // ---- WARM-UP outside the counted region, so lazy code-object load / first-touch page-in is
    // not attributed to the measured dispatches.
    if (use_fp16) launch_dispatch<__half>(a, grid, block, bylane, eff_cols);
    else          launch_dispatch<__hip_bfloat16>(a, grid, block, bylane, eff_cols);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    std::printf("[g2fuse] warm-up OK; running %d counted iteration(s)\n", iters);

    // ---- counted region.
    for (int it = 0; it < iters; ++it) {
        if (use_fp16) launch_dispatch<__half>(a, grid, block, bylane, eff_cols);
        else          launch_dispatch<__hip_bfloat16>(a, grid, block, bylane, eff_cols);
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());

    // Cheap non-degeneracy check: with these synthetic inputs every output column is the same
    // nonzero value, so an all-zero out means the kernel wrote nothing.
    {
        std::vector<float> o((size_t)std::min(N, 8), 0.0f);
        HIP_CHECK(hipMemcpy(o.data(), d_out, o.size() * sizeof(float), hipMemcpyDeviceToHost));
        std::printf("[g2fuse] out[0,0..%zu] =", o.size() - 1);
        for (float v : o) std::printf(" %g", v);
        std::printf("\n");
    }

    std::printf("[g2fuse] DONE (%d iters, branch=%s)\n", iters, bylane ? "bylane" : "k-on-lanes");

    HIP_CHECK(hipFree(d_x_fp8)); HIP_CHECK(hipFree(d_wdata)); HIP_CHECK(hipFree(d_wscal));
    HIP_CHECK(hipFree(d_asc));   HIP_CHECK(hipFree(d_tw));    HIP_CHECK(hipFree(d_str));
    HIP_CHECK(hipFree(d_sti));   HIP_CHECK(hipFree(d_eid));   HIP_CHECK(hipFree(d_ntp));
    HIP_CHECK(hipFree(d_out));
    if (d_partial) HIP_CHECK(hipFree(d_partial));
    return 0;
}
