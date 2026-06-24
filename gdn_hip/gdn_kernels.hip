// Native HIP kernels for the Qwen3-Next / Qwen3.5 Gated Delta Net (GDN) linear-attention path,
// for AMD RDNA4 (gfx1201). AOT-compiled ONCE (no per-shape Triton JIT/autotune) and exposed as
// framework-agnostic torch.ops.gdn_hip.* — consumable from minisgl AND vLLM (both route GDN through
// the same flash-linear-attention recurrence). Correctness-first v1: faithful to the fla reference
// (fused_recurrent_gated_delta_rule), fp32 throughout; the chunked throughput optimization is later.
//
// Math (per sequence, per value-head; state S is [V x K], V=head_v_dim, K=head_k_dim), per token:
//   q,k l2-normalized over K; q *= scale (=1/sqrt(K))
//   g    = -exp(A_log) * softplus(a + dt_bias)      (decay)         beta = sigmoid(b)   (write gate)
//   S   *= exp(g)                                   (gated decay)
//   v   -= S @ k                                    (delta correction)
//   v   *= beta
//   S   += outer(v, k)                              (rank-1 write)
//   o    = S @ q                                    (readout)
// Lifted verbatim from fused_recurrent_gated_delta_rule_packed_decode_kernel / _fwd_kernel.
//
// Layout: one thread block per (sequence, value-head); blockDim.x == V; thread r owns state row
// S[r, 0..K-1] in registers (K<=GDN_MAX_K) across the whole token loop. q/k for the current token
// live in LDS (shared across the V rows). No cross-row dependency -> no block reduction in the step.

#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef GDN_MAX_K
#define GDN_MAX_K 128  // head_k_dim (Qwen3.5/3.6 GDN = 128); state row held in registers
#endif

__device__ __forceinline__ float gdn_softplus(float x) {
  // matches the Triton ref: threshold 20 to avoid overflow in log1p(exp(x))
  return x <= 20.0f ? log1pf(__expf(x)) : x;
}

// Cooperative L2 norm of a length-K vector held in LDS `vec` (blockDim.x threads, V>=?).
// Returns 1/sqrt(sum(vec^2)+eps). Result broadcast via LDS scalar slot.
__device__ __forceinline__ float gdn_inv_l2(const float* __restrict__ vec, int K,
                                            float* __restrict__ red, int tid, int nthreads) {
  float partial = 0.0f;
  for (int j = tid; j < K; j += nthreads) partial += vec[j] * vec[j];
  red[tid] = partial;
  __syncthreads();
  // simple tree reduction over nthreads (power-of-two not required; serial tail by thread 0)
  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] + 1e-6f);
  __syncthreads();
  return inv;
}

// One recurrence step for thread r's state row s_row[0..K-1] (registers). q/k post-l2norm in LDS.
__device__ __forceinline__ float gdn_step(float* s_row, const float* q_s, const float* k_s,
                                          float v_r, float g, float beta, int K) {
  const float eg = __expf(g);
  // S *= exp(g);  delta = v - S@k
  float t = 0.0f;
#pragma unroll 4
  for (int j = 0; j < K; ++j) { s_row[j] *= eg; t += s_row[j] * k_s[j]; }
  float vr = (v_r - t) * beta;
  // S += outer(v,k);  o = S@q
  float o = 0.0f;
#pragma unroll 4
  for (int j = 0; j < K; ++j) { s_row[j] += vr * k_s[j]; o += s_row[j] * q_s[j]; }
  return o;
}

// ---------------------------------------------------------------------------------------------
// DECODE: one token per sequence. Computes g/beta inline from a,b,A_log,dt_bias. Updates state
// in place at slot = state_indices[seq]. Skips (writes zero) when slot <= 0 (NULL block).
//   q,k:        [B, H,  K]   v: [B, HV, V]   a,b: [B, HV]   A_log,dt_bias: [HV]
//   ssm_state:  [num_slots, HV, V, K]   state_indices: [B]   out: [B, HV, V]
//   GQA: head h = hv / (HV / H).
// grid = (B * HV), block = V threads.
__global__ void gdn_decode_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    float* __restrict__ ssm_state, const long* __restrict__ state_indices, float* __restrict__ out,
    int B, int H, int HV, int K, int V, float scale, int use_l2norm) {
  const int blk = blockIdx.x, r = threadIdx.x;
  const int bi = blk / HV, hv = blk % HV, hq = hv / (HV / H);

  extern __shared__ float smem[];
  float* q_s = smem;            // [K]
  float* k_s = smem + K;        // [K]
  float* red = smem + 2 * K;    // [blockDim.x] reduction scratch

  // load q,k for (bi,hq) into LDS
  for (int j = r; j < K; j += blockDim.x) {
    q_s[j] = q[(bi * H + hq) * K + j];
    k_s[j] = k[(bi * H + hq) * K + j];
  }
  __syncthreads();
  if (use_l2norm) {
    float invq = gdn_inv_l2(q_s, K, red, r, blockDim.x);
    float invk = gdn_inv_l2(k_s, K, red, r, blockDim.x);
    for (int j = r; j < K; j += blockDim.x) { q_s[j] *= invq; k_s[j] *= invk; }
    __syncthreads();
  }
  for (int j = r; j < K; j += blockDim.x) q_s[j] *= scale;
  __syncthreads();

  const long slot = state_indices[bi];
  if (r < V) {
    float* o_p = out + (bi * HV + hv) * V + r;
    if (slot <= 0) { *o_p = 0.0f; return; }
    const float x = a[bi * HV + hv] + dt_bias[hv];
    const float g = -__expf(A_log[hv]) * gdn_softplus(x);
    const float beta = 1.0f / (1.0f + __expf(-b[bi * HV + hv]));
    float* s_row = ssm_state + ((slot * HV + hv) * V + r) * K;
    float s_reg[GDN_MAX_K];
#pragma unroll 4
    for (int j = 0; j < K; ++j) s_reg[j] = s_row[j];
    float o = gdn_step(s_reg, q_s, k_s, v[(bi * HV + hv) * V + r], g, beta, K);
#pragma unroll 4
    for (int j = 0; j < K; ++j) s_row[j] = s_reg[j];
    *o_p = o;
  }
}

// ---------------------------------------------------------------------------------------------
// PREFILL: varlen, one block per (sequence, value-head); loops over the sequence's tokens carrying
// the state row in registers, writes o per token, persists the final state to slot. g,beta are
// computed inline per token from a,b (per-token) and A_log,dt_bias (per-head).
//   q,k:  [T, H, K]   v: [T, HV, V]   a,b: [T, HV]   (T = total tokens across the batch)
//   cu_seqlens: [N+1]   state_indices: [N]   has_initial_state: [N] (uint8)
//   ssm_state: [num_slots, HV, V, K]   out: [T, HV, V]
// grid = (N * HV), block = V threads.
__global__ void gdn_prefill_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    const int* __restrict__ cu_seqlens, const long* __restrict__ state_indices,
    const unsigned char* __restrict__ has_initial_state,
    float* __restrict__ ssm_state, float* __restrict__ out,
    int N, int H, int HV, int K, int V, float scale, int use_l2norm) {
  const int blk = blockIdx.x, r = threadIdx.x;
  const int n = blk / HV, hv = blk % HV, hq = hv / (HV / H);
  const int bos = cu_seqlens[n], eos = cu_seqlens[n + 1];
  const long slot = state_indices[n];
  if (slot <= 0 || bos >= eos) return;

  extern __shared__ float smem[];
  float* q_s = smem;             // [K]
  float* k_s = smem + K;         // [K]
  float* red = smem + 2 * K;     // [blockDim.x]

  // state row in registers; init from slot if has_initial_state, else zero
  float s_reg[GDN_MAX_K];
  if (r < V) {
    float* s_row = ssm_state + ((slot * HV + hv) * V + r) * K;
    const bool use_init = has_initial_state[n] != 0;
#pragma unroll 4
    for (int j = 0; j < K; ++j) s_reg[j] = use_init ? s_row[j] : 0.0f;
  }

  const float Alog = A_log[hv], dtb = dt_bias[hv];
  for (int t = bos; t < eos; ++t) {
    for (int j = r; j < K; j += blockDim.x) {
      q_s[j] = q[(t * H + hq) * K + j];
      k_s[j] = k[(t * H + hq) * K + j];
    }
    __syncthreads();
    if (use_l2norm) {
      float invq = gdn_inv_l2(q_s, K, red, r, blockDim.x);
      float invk = gdn_inv_l2(k_s, K, red, r, blockDim.x);
      for (int j = r; j < K; j += blockDim.x) { q_s[j] *= invq; k_s[j] *= invk; }
      __syncthreads();
    }
    for (int j = r; j < K; j += blockDim.x) q_s[j] *= scale;
    __syncthreads();

    if (r < V) {
      const float x = a[t * HV + hv] + dtb;
      const float g = -__expf(Alog) * gdn_softplus(x);
      const float beta = 1.0f / (1.0f + __expf(-b[t * HV + hv]));
      float o = gdn_step(s_reg, q_s, k_s, v[(t * HV + hv) * V + r], g, beta, K);
      out[(t * HV + hv) * V + r] = o;
    }
    __syncthreads();  // q_s/k_s reused next token
  }
  if (r < V) {
    float* s_row = ssm_state + ((slot * HV + hv) * V + r) * K;
#pragma unroll 4
    for (int j = 0; j < K; ++j) s_row[j] = s_reg[j];
  }
}

// ---------------------------------------------------------------------------------------------
// Depthwise causal conv1d, DECODE update (one token/seq). conv_state holds the last (W-1) inputs
// per channel; we append the new token, dot with the kernel, optional SiLU, and roll the state.
//   x:       [B, C]   weight: [C, W]   bias: [C] or null   conv_state: [num_slots, C, W-1]
//   state_indices: [B]   out: [B, C]   (silu applied iff activation!=0)
// grid = (B), block = min(C, 1024); each thread strides over channels.
__global__ void causal_conv1d_update_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    float* __restrict__ conv_state, const long* __restrict__ state_indices, float* __restrict__ out,
    int B, int C, int W, int activation) {
  const int bi = blockIdx.x;
  const long slot = state_indices[bi];
  const int Wm1 = W - 1;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float acc = bias ? bias[c] : 0.0f;
    const float* wc = weight + c * W;
    float* st = (slot > 0) ? conv_state + (slot * C + c) * Wm1 : nullptr;
    // window = [state(W-1) , x_new]; conv = sum_{i<W} window[i]*w[i]
    float xnew = x[bi * C + c];
    if (st) {
      for (int i = 0; i < Wm1; ++i) acc += st[i] * wc[i];
    }
    acc += xnew * wc[Wm1];
    // roll state left, append new token at tail
    if (st) {
      for (int i = 0; i < Wm1 - 1; ++i) st[i] = st[i + 1];
      st[Wm1 - 1] = xnew;
    }
    if (activation) acc = acc / (1.0f + __expf(-acc));  // SiLU
    out[bi * C + c] = acc;
  }
}

// Depthwise causal conv1d, PREFILL (varlen). For each sequence, run the causal conv over its tokens
// using the optional initial conv_state as left padding, write outputs, and store the trailing
// (W-1) inputs back to the slot for the subsequent decode.
//   x:    [T, C]  (token-major)   weight: [C, W]   bias: [C] or null
//   cu_seqlens: [N+1]   state_indices: [N]   has_initial_state: [N]   conv_state: [num_slots,C,W-1]
//   out:  [T, C]
// grid = (N, ceil(C/CT)); block = CT threads over channels. Simple per-channel sliding window.
__global__ void causal_conv1d_fwd_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias,
    const int* __restrict__ cu_seqlens, const long* __restrict__ state_indices,
    const unsigned char* __restrict__ has_initial_state, float* __restrict__ conv_state,
    float* __restrict__ out, int N, int C, int W, int activation) {
  const int n = blockIdx.x;
  const int bos = cu_seqlens[n], eos = cu_seqlens[n + 1];
  const long slot = state_indices[n];
  const int Wm1 = W - 1;
  const bool use_init = has_initial_state && has_initial_state[n] != 0;
  for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < C; c += gridDim.y * blockDim.x) {
    const float* wc = weight + c * W;
    const float bc = bias ? bias[c] : 0.0f;
    float* st = (slot > 0) ? conv_state + (slot * C + c) * Wm1 : nullptr;
    // local history ring of last W-1 inputs, seeded from conv_state (or zero)
    float hist[16];  // W-1 <= 15 (GDN W=4); generous bound
    for (int i = 0; i < Wm1; ++i) hist[i] = (use_init && st) ? st[i] : 0.0f;
    for (int t = bos; t < eos; ++t) {
      float xnew = x[t * C + c];
      float acc = bc;
      for (int i = 0; i < Wm1; ++i) acc += hist[i] * wc[i];
      acc += xnew * wc[Wm1];
      for (int i = 0; i < Wm1 - 1; ++i) hist[i] = hist[i + 1];
      hist[Wm1 - 1] = xnew;
      if (activation) acc = acc / (1.0f + __expf(-acc));
      out[t * C + c] = acc;
    }
    if (st) for (int i = 0; i < Wm1; ++i) st[i] = hist[i];  // persist trailing window
  }
}

// ---------------------------------------------------------------------------------------------
// RMSNormGated: y = (x / rms(x)) * weight * silu(z), normalized over the last dim D (norm-before-gate
// = the Qwen3.5 GDN convention). x,z,out: [M, D]; weight: [D]. grid=(M), block=min(D,1024).
__global__ void rmsnorm_gated_kernel(
    const float* __restrict__ x, const float* __restrict__ z, const float* __restrict__ weight,
    float* __restrict__ out, int M, int D, float eps) {
  const int m = blockIdx.x;
  extern __shared__ float red[];
  float partial = 0.0f;
  for (int j = threadIdx.x; j < D; j += blockDim.x) { float xv = x[m * D + j]; partial += xv * xv; }
  red[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / D + eps);
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    const float zv = z[m * D + j];
    const float gate = zv / (1.0f + __expf(-zv));  // SiLU(z)
    out[m * D + j] = x[m * D + j] * inv * weight[j] * gate;
  }
}

// ============================== launchers (called from bindings.cpp) ==========================
static inline float* FP(const at::Tensor& t) { return t.data_ptr<float>(); }
static inline const float* CFP(const at::Tensor& t) { return t.data_ptr<float>(); }

void launch_gdn_decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                       const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                       const at::Tensor& dt_bias, at::Tensor& ssm_state,
                       const at::Tensor& state_indices, at::Tensor& out,
                       double scale, int64_t use_l2norm) {
  const int B = q.size(0), H = q.size(1), K = q.size(2);
  const int HV = v.size(1), V = v.size(2);
  dim3 grid(B * HV), block(V);
  size_t shmem = (2 * K + V) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(gdn_decode_kernel, grid, block, shmem, stream,
      CFP(q), CFP(k), CFP(v), CFP(a), CFP(b), CFP(A_log), CFP(dt_bias),
      FP(ssm_state), state_indices.data_ptr<long>(), FP(out),
      B, H, HV, K, V, (float)scale, (int)use_l2norm);
}

void launch_gdn_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                        const at::Tensor& a, const at::Tensor& b, const at::Tensor& A_log,
                        const at::Tensor& dt_bias, const at::Tensor& cu_seqlens,
                        const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                        at::Tensor& ssm_state, at::Tensor& out, double scale, int64_t use_l2norm) {
  const int H = q.size(1), K = q.size(2);
  const int HV = v.size(1), V = v.size(2);
  const int N = state_indices.size(0);
  dim3 grid(N * HV), block(V);
  size_t shmem = (2 * K + V) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(gdn_prefill_kernel, grid, block, shmem, stream,
      CFP(q), CFP(k), CFP(v), CFP(a), CFP(b), CFP(A_log), CFP(dt_bias),
      cu_seqlens.data_ptr<int>(), state_indices.data_ptr<long>(),
      has_initial_state.data_ptr<unsigned char>(), FP(ssm_state), FP(out),
      N, H, HV, K, V, (float)scale, (int)use_l2norm);
}

void launch_causal_conv1d_update(const at::Tensor& x, const at::Tensor& weight,
                                 const c10::optional<at::Tensor>& bias, at::Tensor& conv_state,
                                 const at::Tensor& state_indices, at::Tensor& out,
                                 int64_t activation) {
  const int B = x.size(0), C = x.size(1), W = weight.size(1);
  dim3 grid(B), block(C < 1024 ? C : 1024);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(causal_conv1d_update_kernel, grid, block, 0, stream,
      CFP(x), CFP(weight), bias.has_value() ? CFP(*bias) : nullptr,
      FP(conv_state), state_indices.data_ptr<long>(), FP(out), B, C, W, (int)activation);
}

void launch_causal_conv1d_fwd(const at::Tensor& x, const at::Tensor& weight,
                              const c10::optional<at::Tensor>& bias, const at::Tensor& cu_seqlens,
                              const at::Tensor& state_indices, const at::Tensor& has_initial_state,
                              at::Tensor& conv_state, at::Tensor& out, int64_t activation) {
  const int C = x.size(1), W = weight.size(1);
  const int N = state_indices.size(0);
  const int CT = C < 256 ? C : 256;
  dim3 grid(N, (C + CT - 1) / CT), block(CT);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(causal_conv1d_fwd_kernel, grid, block, 0, stream,
      CFP(x), CFP(weight), bias.has_value() ? CFP(*bias) : nullptr,
      cu_seqlens.data_ptr<int>(), state_indices.data_ptr<long>(),
      has_initial_state.data_ptr<unsigned char>(), FP(conv_state), FP(out), N, C, W, (int)activation);
}

void launch_rmsnorm_gated(const at::Tensor& x, const at::Tensor& z, const at::Tensor& weight,
                          at::Tensor& out, double eps) {
  const int M = x.size(0), D = x.size(1);
  dim3 grid(M), block(D < 1024 ? D : 1024);
  size_t shmem = block.x * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  hipLaunchKernelGGL(rmsnorm_gated_kernel, grid, block, shmem, stream,
      CFP(x), CFP(z), CFP(weight), FP(out), M, D, (float)eps);
}
