// Native RXF ("Rotated eXtra Fast") W4A8 HIP kernels for gfx1201 (RDNA4), Triton-free.
//
// Migrated from the Triton RXF kernels in vllm-gfx1201/paroquant_rotation/rxf_kernels.py.
// RXF is W4(NL-codebook)-A8(int8) with a fixed block-diagonal Hadamard (FWHT) rotation
// applied to the activation at runtime and to the weights offline, so the rotation cancels
// in the dot product ( (H x) . (H w)^T == x . w^T ) while spreading activation outliers
// across each size-S group to tighten the 4-bit weight scale.
//
// THREE ops, all integer-exact on the weight side (no fp8 round-trip):
//   rotate_quant_int8 : [M,K] act  -> (int8 [M,K] rotated+quantized, fp32 [M] per-token scale)
//   linear            : int8 q . NL-coded int4 weight  -> bf16 [M,N]   (rocwmma int8 WMMA / GEMV)
//   moe_gemm          : grouped per-expert int8 GEMM (scalar v0; WMMA is a perf follow-up)
//
// WHY int8 WMMA is viable on RDNA4 (the migration's pivotal fact): gfx12 retains the RDNA3
// integer WMMA — __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12, wrapped by rocwmma's
// int8_t fragments (rocwmma/internal/wmma_impl.hpp:981). There is NO K=32 int8 variant (only
// iu4 has one), but RXF's group size 32 == exactly two 16-wide int8 WMMA K-steps, so the
// per-group fp16 weight scale lands cleanly on a 2-kstep boundary.
//
// Numerics matched bit-for-bit to the Triton reference (rxf_hip_parity.py is the oracle):
//   FWHT  : in-place decimation butterfly, partner at distance h for h in {1,2,...,S/2};
//           normalize by 0.1767766953 for S==32 (== 1/sqrt(32)) else 1/sqrt(S).
//   quant : per-token symmetric int8; scale = absmax/127; q = round(x*127/absmax) in [-127,127].
//   NL    : weight_packed uint8 [N,K/2], low nibble = even channel, high = odd; index -> nl[idx]
//           int8 codebook; per-group fp16 weight_scale [N,K/32] multiplies the int32 partial;
//           per-token act scale multiplies the fp32 accumulator once at the end.

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

using bf16 = __hip_bfloat16;

// ----------------------------------------------------------------------------------------
// 1. rotate_quant_int8 : block-diagonal FWHT-S over each size-S group of a row, then a
//    per-token symmetric int8 quant. One block per row; each thread owns whole S-groups
//    (S floats in registers -> no intra-group sync), rotated values staged in dynamic smem
//    for the row absmax reduction + quantize passes.
// ----------------------------------------------------------------------------------------

__device__ __forceinline__ float block_reduce_max(float v, float* shared /*[32]*/) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
    if (lane == 0) shared[wid] = v;
    __syncthreads();
    const int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
        if (lane == 0) shared[0] = v;
    }
    __syncthreads();
    return shared[0];
}

template <typename scalar_t, int SPAN>
__global__ void rotate_quant_kernel(
    const scalar_t* __restrict__ x,   // [M, K]
    int8_t* __restrict__ q,           // [M, K]
    float* __restrict__ scale,        // [M]
    int M, int K, float norm) {
    extern __shared__ float srow[];   // [K] rotated row, fp32
    __shared__ float red[32];
    const int row = blockIdx.x;
    if (row >= M) return;
    const scalar_t* xr = x + (long)row * K;
    const int nG = K / SPAN;

    // Phase 1: rotate each S-group in registers, write back * norm.
    for (int g = threadIdx.x; g < nG; g += blockDim.x) {
        float buf[SPAN];
        const int base = g * SPAN;
        #pragma unroll
        for (int i = 0; i < SPAN; ++i) buf[i] = (float)xr[base + i];
        #pragma unroll
        for (int h = 1; h < SPAN; h <<= 1) {
            #pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                if ((i & h) == 0) {
                    const float a = buf[i], b = buf[i | h];
                    buf[i] = a + b;
                    buf[i | h] = a - b;
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < SPAN; ++i) srow[base + i] = buf[i] * norm;
    }
    __syncthreads();

    // Phase 2: per-token absmax -> scale.
    float local = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) local = fmaxf(local, fabsf(srow[i]));
    const float absmax = fmaxf(block_reduce_max(local, red), 1e-12f);
    const float inv = 127.0f / absmax;
    if (threadIdx.x == 0) scale[row] = absmax / 127.0f;

    // Phase 3: symmetric int8 quant (round half away from zero, clamp [-127,127]).
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = roundf(srow[i] * inv);
        v = fminf(fmaxf(v, -127.0f), 127.0f);
        q[(long)row * K + i] = (int8_t)v;
    }
}

// ----------------------------------------------------------------------------------------
// 2. linear : int8 q [M,K] . NL-coded int4 weight [N,K] -> bf16 [M,N].
//    WMMA path (M>2): 16x16 output tile per warp; per group of 32 (= 2 int8 K-steps) the
//    int32 partial is scaled by the group's fp16 weight scale into a fp32 smem accumulator
//    (store-to-smem epilogue, attn_hip-style -> no WMMA fragment-layout assumption escapes).
//    GEMV path (M<=2): one thread per output, scalar int8 dot.
// ----------------------------------------------------------------------------------------

constexpr int WD = 16;          // int8 WMMA is 16x16x16 on gfx12
constexpr int GROUP = 32;       // RXF group == 2 WMMA K-steps
using FragAq  = rocwmma::fragment<rocwmma::matrix_a, WD, WD, WD, int8_t, rocwmma::row_major>;
using FragWk  = rocwmma::fragment<rocwmma::matrix_b, WD, WD, WD, int8_t, rocwmma::col_major>;
using FragI32 = rocwmma::fragment<rocwmma::accumulator, WD, WD, WD, int32_t>;

__global__ __launch_bounds__(32) void linear_wmma_kernel(
    const int8_t* __restrict__ q,        // [M, K]
    const float* __restrict__ a_scale,   // [M]
    const uint8_t* __restrict__ wpk,     // [N, K/2]
    const __half* __restrict__ wsc,      // [N, K/32]
    const int8_t* __restrict__ nl,       // [16]
    const float* __restrict__ bias,      // [N] or nullptr
    bf16* __restrict__ out,              // [M, N]
    int M, int N, int K) {
    const int n0 = blockIdx.x * WD;
    const int m0 = blockIdx.y * WD;
    const int tid = threadIdx.x;         // single warp, 32 lanes

    __shared__ int8_t sA[WD][GROUP];
    __shared__ int8_t sW[WD][GROUP];
    __shared__ int32_t sPart[WD][WD];
    __shared__ float sAcc[WD][WD];
    __shared__ int8_t sNL[16];

    if (tid < 16) sNL[tid] = nl[tid];
    for (int i = tid; i < WD * WD; i += 32) sAcc[i / WD][i % WD] = 0.0f;
    __syncthreads();

    const int nGroups = K / GROUP;
    const int kpr = K / 2;               // packed bytes per weight row
    const int gpr = K / GROUP;           // scale groups per weight row

    for (int g = 0; g < nGroups; ++g) {
        const int kb = g * GROUP;
        // load activation [16][32] and unpacked weight codes [16][32] for this group
        for (int idx = tid; idx < WD * GROUP; idx += 32) {
            const int r = idx / GROUP, kk = idx % GROUP;
            const int m = m0 + r, n = n0 + r;
            sA[r][kk] = (m < M) ? q[(long)m * K + kb + kk] : (int8_t)0;
            int8_t code = 0;
            if (n < N) {
                const uint8_t byte = wpk[(long)n * kpr + ((kb + kk) >> 1)];
                const int nib = (kk & 1) ? (byte >> 4) : (byte & 0x0F);
                code = sNL[nib];
            }
            sW[r][kk] = code;
        }
        __syncthreads();

        FragI32 acc;
        rocwmma::fill_fragment(acc, 0);
        #pragma unroll
        for (int ks = 0; ks < GROUP / WD; ++ks) {
            FragAq af;
            FragWk wf;
            rocwmma::load_matrix_sync(af, &sA[0][ks * WD], GROUP);
            rocwmma::load_matrix_sync(wf, &sW[0][ks * WD], GROUP);
            rocwmma::mma_sync(acc, af, wf, acc);
        }
        rocwmma::store_matrix_sync(&sPart[0][0], acc, WD, rocwmma::mem_row_major);
        __syncthreads();

        // scale this group's int32 partial by the per-(n,group) fp16 weight scale.
        for (int idx = tid; idx < WD * WD; idx += 32) {
            const int m = idx / WD, n = idx % WD;
            const int nn = n0 + n;
            if (nn < N) {
                const float ws = __half2float(wsc[(long)nn * gpr + g]);
                sAcc[m][n] += (float)sPart[m][n] * ws;
            }
        }
        __syncthreads();
    }

    // epilogue: per-token act scale, optional bias, write bf16.
    for (int idx = tid; idx < WD * WD; idx += 32) {
        const int m = idx / WD, n = idx % WD;
        const int mm = m0 + m, nn = n0 + n;
        if (mm < M && nn < N) {
            float v = sAcc[m][n] * a_scale[mm];
            if (bias) v += bias[nn];
            out[(long)mm * N + nn] = (bf16)v;
        }
    }
}

__global__ void linear_gemv_kernel(
    const int8_t* __restrict__ q,        // [M, K]
    const float* __restrict__ a_scale,   // [M]
    const uint8_t* __restrict__ wpk,     // [N, K/2]
    const __half* __restrict__ wsc,      // [N, K/32]
    const int8_t* __restrict__ nl,       // [16]
    const float* __restrict__ bias,      // [N] or nullptr
    bf16* __restrict__ out,              // [M, N]
    int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;
    __shared__ int8_t sNL[16];
    if (threadIdx.x < 16) sNL[threadIdx.x] = nl[threadIdx.x];
    __syncthreads();

    const uint8_t* wn = wpk + (long)n * (K / 2);
    const __half* wsn = wsc + (long)n * (K / GROUP);
    const int8_t* qm = q + (long)m * K;
    float acc = 0.0f;
    for (int g = 0; g < K / GROUP; ++g) {
        const int kb = g * GROUP;
        int part = 0;
        #pragma unroll
        for (int kk = 0; kk < GROUP; ++kk) {
            const uint8_t byte = wn[(kb + kk) >> 1];
            const int nib = (kk & 1) ? (byte >> 4) : (byte & 0x0F);
            part += (int)qm[kb + kk] * (int)sNL[nib];
        }
        acc += (float)part * __half2float(wsn[g]);
    }
    acc *= a_scale[m];
    if (bias) acc += bias[n];
    out[(long)m * N + n] = (bf16)acc;
}

// ----------------------------------------------------------------------------------------
// 3. moe_gemm (scalar v0) : grouped per-expert int8 GEMM. Mirrors the proven fp8 MoE v0
//    dispatch (sorted_token_ids / expert_ids / num_tokens_post_padded) but with int8 dot +
//    NL codebook + per-token act scale. WMMA grouping is a perf follow-up.
// ----------------------------------------------------------------------------------------

__global__ void moe_gemm_kernel(
    const int8_t* __restrict__ q,        // [T, K] (gemm1) or [P, K] (gemm2, ident gather)
    const float* __restrict__ a_scale,   // [T] or [P]
    const uint8_t* __restrict__ wpk,     // [E, N, K/2]
    const __half* __restrict__ wsc,      // [E, N, K/32]
    const int8_t* __restrict__ nl,       // [16]
    const int* __restrict__ sorted_ids,  // [P]
    const int* __restrict__ expert_ids,  // [P/block_m]
    const int* __restrict__ ntp,         // [1] num_tokens_post_padded
    bf16* __restrict__ out,              // [P, N]
    int N, int K, int top_k, int block_m, int num_valid_tokens) {
    const int row_pad = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    if (row_pad >= ntp[0]) return;
    const int offs = sorted_ids[row_pad];
    if (offs >= num_valid_tokens) return;          // padding slot
    const int src = offs / top_k;                   // activation row
    const int e = expert_ids[row_pad / block_m];

    const long ppr = K / 2;
    const long gpr = K / GROUP;
    const uint8_t* wn = wpk + ((long)e * N + col) * ppr;
    const __half* wsn = wsc + ((long)e * N + col) * gpr;
    const int8_t* qsrc = q + (long)src * K;

    float acc = 0.0f;
    for (int g = 0; g < K / GROUP; ++g) {
        const int kb = g * GROUP;
        int part = 0;
        #pragma unroll
        for (int kk = 0; kk < GROUP; ++kk) {
            const uint8_t byte = wn[(kb + kk) >> 1];
            const int nib = (kk & 1) ? (byte >> 4) : (byte & 0x0F);
            part += (int)qsrc[kb + kk] * (int)nl[nib];
        }
        acc += (float)part * __half2float(wsn[g]);
    }
    out[(long)row_pad * N + col] = (bf16)(acc * a_scale[src]);
}

}  // namespace

// ============================== launchers ==============================

void launch_rotate_quant(const at::Tensor& x, at::Tensor& q, at::Tensor& scale, int64_t span) {
    const int K = x.size(-1);
    const int M = x.numel() / K;
    TORCH_CHECK(span == 32, "rxf_hip: only Hadamard span 32 is supported (this migration)");
    TORCH_CHECK(K % span == 0, "rxf_hip: K must be a multiple of span");
    const float norm = 0.1767766953f;  // == 1/sqrt(32), matched to the Triton reference literal
    const int threads = 256;
    const size_t shmem = (size_t)K * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rxf_rotate_quant", [&] {
        rotate_quant_kernel<scalar_t, 32><<<M, threads, shmem, stream>>>(
            x.data_ptr<scalar_t>(), q.data_ptr<int8_t>(), scale.data_ptr<float>(), M, K, norm);
    });
}

void launch_linear(const at::Tensor& q, const at::Tensor& a_scale, const at::Tensor& wpk,
                   const at::Tensor& wsc, const at::Tensor& nl, const at::Tensor& bias,
                   at::Tensor& out, bool has_bias) {
    const int M = q.size(0), K = q.size(1), N = wpk.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;
    if (M <= 2) {
        const int TPB = 256;
        dim3 grid((N + TPB - 1) / TPB, M);
        linear_gemv_kernel<<<grid, TPB, 0, stream>>>(
            q.data_ptr<int8_t>(), a_scale.data_ptr<float>(), wpk.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(wsc.data_ptr()), nl.data_ptr<int8_t>(),
            bias_ptr, reinterpret_cast<bf16*>(out.data_ptr()), M, N, K);
    } else {
        dim3 grid((N + WD - 1) / WD, (M + WD - 1) / WD);
        linear_wmma_kernel<<<grid, 32, 0, stream>>>(
            q.data_ptr<int8_t>(), a_scale.data_ptr<float>(), wpk.data_ptr<uint8_t>(),
            reinterpret_cast<const __half*>(wsc.data_ptr()), nl.data_ptr<int8_t>(),
            bias_ptr, reinterpret_cast<bf16*>(out.data_ptr()), M, N, K);
    }
}

void launch_moe_gemm(const at::Tensor& q, const at::Tensor& a_scale, const at::Tensor& wpk,
                     const at::Tensor& wsc, const at::Tensor& nl, const at::Tensor& sorted_ids,
                     const at::Tensor& expert_ids, const at::Tensor& ntp, at::Tensor& out,
                     int64_t top_k, int64_t block_m, int64_t num_valid_tokens) {
    const int P = sorted_ids.size(0);
    const int N = wpk.size(1), K = wpk.size(2) * 2;
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(64, 4);
    dim3 grid((N + block.x - 1) / block.x, (P + block.y - 1) / block.y);
    moe_gemm_kernel<<<grid, block, 0, stream>>>(
        q.data_ptr<int8_t>(), a_scale.data_ptr<float>(), wpk.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(wsc.data_ptr()), nl.data_ptr<int8_t>(),
        sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(), ntp.data_ptr<int>(),
        reinterpret_cast<bf16*>(out.data_ptr()), N, K, (int)top_k, (int)block_m,
        (int)num_valid_tokens);
}
