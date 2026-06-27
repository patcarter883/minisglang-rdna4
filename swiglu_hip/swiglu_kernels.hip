// Native HIP fused bf16/fp16 SwiGLU MLP for gfx1201 (RDNA4) — Task B (#18).
//
// Target: the dense always-on shared expert (qwen3_5_moe / qwen2_moe `*MoeSharedExpert`) at
// decode (M<=2). It is two unquantized `F.linear` GEMVs + silu_and_mul. At M=1 with a small
// intermediate (inter=512, per-card 256) hipBLAS tiles for large M and launches too few blocks,
// reaching only ~14% of the 644 GB/s weight-read BW (measured, tools/swiglu_bench.py). This kernel
// is output-parallel (one block per output channel) so every weight row is read exactly once across
// many concurrent blocks -> approaches peak BW at M=1.
//
//   out = down( silu(x @ Wg^T) * (x @ Wu^T) )
//   x  : [M, K]          (K = hidden)               bf16/fp16
//   Wgu: [2*inter, K]    (F.linear layout [out,in]; rows [0,inter)=gate, [inter,2*inter)=up)
//   Wd : [K, inter]      (F.linear layout [out,in])
//   out: [M, K]          bf16/fp16
//
// fp32 accumulation throughout (matches F.linear + the fp32-internal silu_and_mul reference).
// STATUS: compiles-pending / numerics-pending GPU. Validate with swiglu_hip_parity.py.

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

using bf16 = __hip_bfloat16;
constexpr int WARP = 32;
constexpr int WARPS_PER_BLOCK = 8;             // 256 threads
constexpr int THREADS = WARP * WARPS_PER_BLOCK;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}

// Stage 1: h[m,c] = silu(dot(Wg[c], x[m])) * dot(Wu[c], x[m]).  One WARP per output channel c
// (warp-shuffle reduce, no shared mem / no block sync). grid.x = ceil(inter/WARPS_PER_BLOCK),
// grid.y = M. Each warp reads gate row c + up row c once -> all of Wgu read exactly once.
template <typename scalar_t>
__global__ __launch_bounds__(THREADS) void gate_up_silu_kernel(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ wgu, scalar_t* __restrict__ h,
    int M, int K, int inter) {
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);  // output channel
    if (c >= inter) return;
    const int m = blockIdx.y;
    const scalar_t* xr = x + (long)m * K;
    const scalar_t* wg = wgu + (long)c * K;               // gate row c
    const scalar_t* wu = wgu + (long)(inter + c) * K;     // up   row c
    float ag = 0.0f, au = 0.0f;
    for (int k = lane; k < K; k += WARP) {
        const float xv = (float)xr[k];
        ag += xv * (float)wg[k];
        au += xv * (float)wu[k];
    }
    ag = warp_reduce_sum(ag);
    au = warp_reduce_sum(au);
    if (lane == 0) {
        const float s = ag / (1.0f + __expf(-ag));        // silu(gate)
        h[(long)m * inter + c] = (scalar_t)(s * au);
    }
}

// Stage 2: out[m,o] = dot(Wd[o], h[m]).  One WARP per output channel o.
template <typename scalar_t>
__global__ __launch_bounds__(THREADS) void down_kernel(
    const scalar_t* __restrict__ h, const scalar_t* __restrict__ wd, scalar_t* __restrict__ out,
    int M, int hidden, int inter) {
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);  // output channel
    if (o >= hidden) return;
    const int m = blockIdx.y;
    const scalar_t* hr = h + (long)m * inter;
    const scalar_t* wo = wd + (long)o * inter;
    float acc = 0.0f;
    for (int k = lane; k < inter; k += WARP) acc += (float)hr[k] * (float)wo[k];
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[(long)m * hidden + o] = (scalar_t)acc;
}

template <typename scalar_t>
void run(const at::Tensor& x, const at::Tensor& wgu, const at::Tensor& wd, at::Tensor& h,
         at::Tensor& out, int M, int K, int inter) {
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 g1((inter + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, M);
    gate_up_silu_kernel<scalar_t><<<g1, THREADS, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        reinterpret_cast<const scalar_t*>(wgu.data_ptr()),
        reinterpret_cast<scalar_t*>(h.data_ptr()), M, K, inter);
    dim3 g2((K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, M);  // hidden == K
    down_kernel<scalar_t><<<g2, THREADS, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(h.data_ptr()),
        reinterpret_cast<const scalar_t*>(wd.data_ptr()),
        reinterpret_cast<scalar_t*>(out.data_ptr()), M, K, inter);
}

}  // namespace

// ---------------------------------------------------------------------------- launcher
void launch_fused_swiglu(const at::Tensor& x, const at::Tensor& wgu, const at::Tensor& wd,
                         at::Tensor& h, at::Tensor& out) {
    const int K = x.size(-1);
    const int M = x.numel() / K;
    const int inter = wd.size(-1);
    if (x.scalar_type() == at::kBFloat16) {
        run<bf16>(x, wgu, wd, h, out, M, K, inter);
    } else {
        run<__half>(x, wgu, wd, h, out, M, K, inter);
    }
}
