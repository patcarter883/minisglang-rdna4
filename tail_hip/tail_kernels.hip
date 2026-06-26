// Native HIP "tail" elementwise kernels for gfx1201 — RMSNorm, fused residual-add RMSNorm,
// gated SiLU-mul, and NeoX partial RoPE. Triton-free (these are torch elementwise in minisgl /
// csrc-HIP in vLLM today — NOT Triton — so this is about completing the native-HIP set and cutting
// per-op launch/alloc overhead on the launch-sensitive decode path, not removing Triton).
//
// Conventions matched to minisgl-rdna4 (the parity oracles):
//   RMSNorm  : fp32-internal, in-dtype out; plus_one => gain = (weight + 1) (Qwen3.5/3.6/Gemma).
//   SiLU-mul : gated over [..., 2D] -> silu(x[:D]) * x[D:], fp32-internal.
//   RoPE     : NeoX rotate-half, partial rotary_dim<=head_size; cos_sin_cache row =
//              cat(cos[rd/2], sin[rd/2]); dims [rd:head_size] pass through.
//
// STATUS: compiles-pending / numerics-pending GPU. Validate with tail_hip_parity.py.

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

using bf16 = __hip_bfloat16;
constexpr int NORM_THREADS = 256;   // 8 warps

// Block sum-reduce (blockDim multiple of 32, <=1024). Returns total to ALL threads.
__device__ __forceinline__ float block_reduce_sum(float v, float* shared /*[32]*/) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    const int nwarps = blockDim.x >> 5;
    v = (threadIdx.x < nwarps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

// out[row] = rmsnorm(x[row]) * gain;  gain = plus_one ? (w+1) : w
__global__ __launch_bounds__(NORM_THREADS) void rms_norm_kernel(
    const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out,
    int N, int D, float eps, int plus_one) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const bf16* xr = x + (long)row * D;
    bf16* outr = out + (long)row * D;
    __shared__ float red[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) { float v = (float)xr[i]; ss += v * v; }
    const float inv = rsqrtf(block_reduce_sum(ss, red) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float g = (float)w[i] + (plus_one ? 1.0f : 0.0f);
        outr[i] = (bf16)((float)xr[i] * inv * g);
    }
}

// fused residual-add RMSNorm: residual <- x + residual (in place); out <- rmsnorm(residual) * gain.
__global__ __launch_bounds__(NORM_THREADS) void rms_norm_add_kernel(
    const bf16* __restrict__ x, bf16* __restrict__ residual, const bf16* __restrict__ w,
    bf16* __restrict__ out, int N, int D, float eps, int plus_one) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const bf16* xr = x + (long)row * D;
    bf16* rr = residual + (long)row * D;
    bf16* outr = out + (long)row * D;
    __shared__ float red[32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        bf16 rb = (bf16)((float)xr[i] + (float)rr[i]);   // new residual, rounded to bf16 (the
        rr[i] = rb;                                       // residual stream IS bf16 — match minisgl
        float rf = (float)rb;                             // RMSNormFused: variance on the bf16 sum
        ss += rf * rf;
    }
    const float inv = rsqrtf(block_reduce_sum(ss, red) / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float g = (float)w[i] + (plus_one ? 1.0f : 0.0f);
        outr[i] = (bf16)((float)rr[i] * inv * g);
    }
}

// out[n,c] = silu(x[n,c]) * x[n, D+c],  x:[N,2D] out:[N,D]
__global__ void silu_mul_kernel(const bf16* __restrict__ x, bf16* __restrict__ out, long N, int D) {
    const long total = N * (long)D;
    for (long o = (long)blockIdx.x * blockDim.x + threadIdx.x; o < total;
         o += (long)gridDim.x * blockDim.x) {
        const long row = o / D;
        const int col = o % D;
        const float a = (float)x[row * 2 * D + col];
        const float b = (float)x[row * 2 * D + D + col];
        out[o] = (bf16)((a / (1.0f + __expf(-a))) * b);
    }
}

// NeoX partial RoPE on x:[N, H*hs]. cache:[max_pos, rd] = cat(cos[rd/2], sin[rd/2]).
__global__ void rope_kernel(const bf16* __restrict__ x, const int* __restrict__ pos,
                            const float* __restrict__ cache, bf16* __restrict__ out,
                            long N, int H, int hs, int rd) {
    const int d = rd >> 1;
    const long total = N * (long)H * hs;
    for (long o = (long)blockIdx.x * blockDim.x + threadIdx.x; o < total;
         o += (long)gridDim.x * blockDim.x) {
        const int j = o % hs;
        if (j >= rd) { out[o] = x[o]; continue; }        // pass-through tail dims
        const long tmp = o / hs;
        const int n = (int)(tmp / H);
        const int p = pos[n];
        const int half = (j < d) ? j : (j - d);
        const float cs = cache[(long)p * rd + half];
        const float sn = cache[(long)p * rd + d + half];
        const float xj = (float)x[o];
        if (j < d) out[o] = (bf16)(xj * cs - (float)x[o + d] * sn);
        else       out[o] = (bf16)(xj * cs + (float)x[o - d] * sn);
    }
}

}  // namespace

// ---------------------------------------------------------------------------- launchers
void launch_rms_norm(const at::Tensor& x, const at::Tensor& w, at::Tensor& out, double eps,
                     int64_t plus_one) {
    const int D = x.size(-1);
    const int N = x.numel() / D;
    auto stream = at::cuda::getCurrentCUDAStream();
    rms_norm_kernel<<<N, NORM_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16*>(x.data_ptr()), reinterpret_cast<const bf16*>(w.data_ptr()),
        reinterpret_cast<bf16*>(out.data_ptr()), N, D, (float)eps, (int)plus_one);
}

void launch_rms_norm_add(const at::Tensor& x, at::Tensor& residual, const at::Tensor& w,
                         at::Tensor& out, double eps, int64_t plus_one) {
    const int D = x.size(-1);
    const int N = x.numel() / D;
    auto stream = at::cuda::getCurrentCUDAStream();
    rms_norm_add_kernel<<<N, NORM_THREADS, 0, stream>>>(
        reinterpret_cast<const bf16*>(x.data_ptr()), reinterpret_cast<bf16*>(residual.data_ptr()),
        reinterpret_cast<const bf16*>(w.data_ptr()), reinterpret_cast<bf16*>(out.data_ptr()),
        N, D, (float)eps, (int)plus_one);
}

void launch_silu_mul(const at::Tensor& x, at::Tensor& out) {
    const int D = out.size(-1);
    const long N = out.numel() / D;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)std::min<long>(65535, (N * D + 255) / 256);
    silu_mul_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const bf16*>(x.data_ptr()), reinterpret_cast<bf16*>(out.data_ptr()), N, D);
}

void launch_rope(const at::Tensor& x, const at::Tensor& pos, const at::Tensor& cache,
                 at::Tensor& out, int64_t head_size, int64_t rotary_dim) {
    const int hs = (int)head_size;
    const int H = x.size(-1) / hs;
    const long N = x.numel() / (long)x.size(-1);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int blocks = (int)std::min<long>(65535, (N * H * hs + 255) / 256);
    rope_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const bf16*>(x.data_ptr()), pos.data_ptr<int>(),
        cache.data_ptr<float>(), reinterpret_cast<bf16*>(out.data_ptr()), N, H, hs, (int)rotary_dim);
}
