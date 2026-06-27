// Native HIP MLA (Multi-head Latent Attention) DECODE for gfx1201 (RDNA4) — Triton-free.
//
// For DeepSeek-style MLA models (GLM-4.7-Flash: Glm4MoeLite, kv_lora_rank=512, qk_rope_head_dim=64,
// qk_nope=192, v_head_dim=256). MLA compresses the KV cache to a per-token LATENT (no per-head K/V):
// the cache stores [c_KV (kv_lora_rank=512)  ‖  k_rope (qk_rope_head_dim=64)] = 576 dims, SHARED
// across all query heads. None of our GQA/MHA kernels apply, hence a dedicated MLA kernel.
//
// ABSORBED decode form (the efficient one vLLM/SGLang use): W_UK is absorbed into the q up-proj and
// W_UV into the output proj OUTSIDE this kernel, so the kernel receives an absorbed per-head query
//   q = [ q_nope_absorbed (kv_lora_rank=512)  ‖  q_rope (qk_rope_head_dim=64) ]   (LATENT+ROPE=576)
// and attends over the shared latent cache:
//   score[j] = (q · cache[j]) * scale         // dot over all 576 dims (nope-latent + rope)
//   attn     = softmax_j(score)               // causal
//   out      = Σ_j attn[j] * c_KV[j]          // V = the FIRST `LATENT` dims of the cache (512)
// Output is the per-head LATENT-dim context; the W_UV up-projection to v_head_dim happens outside.
//
// Structurally this is a flash-DECODE (M=1, pure FMA + warp-shuffle online softmax, like attn_decode)
// with an ASYMMETRIC qk_dim=576 / v_dim=512 and num_kv_heads=1 (shared latent). No WMMA.
//
// SCOPE (v0, correctness-first): bf16 latent cache, paged (block_table + kv_block_stride), causal,
// kv_lora_rank=512 + qk_rope_head_dim=64. NOT YET: PREFILL (the materialized/absorbed prefill form),
// fp8 latent cache, the W_UK/W_UV absorption (done in the model layer). Validate with mla_hip_parity.py.

#undef __HIP_NO_HALF_CONVERSIONS__
#undef __HIP_NO_HALF_OPERATORS__
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

constexpr int WARP = 32;
constexpr int NWARPS = 8;
constexpr int THREADS = WARP * NWARPS;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}

// OCP e4m3 (torch float8_e4m3fn) -> f32 via the gfx12 hardware convert (exact), same intrinsic the
// attn_decode fp8 path and w4a8_fp8_wmma kernel use. One byte per latent-cache element.
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    return __builtin_amdgcn_cvt_f32_fp8(static_cast<int>(b), 0);
}

// One CTA per (head, seq). The latent cache is shared across heads. LATENT = kv_lora_rank,
// ROPE = qk_rope_head_dim; QK = LATENT+ROPE (score dim), V = LATENT (output dim).
template <int LATENT, int ROPE>
__global__ __launch_bounds__(THREADS) void mla_decode_kernel(
    const __hip_bfloat16* __restrict__ Q,            // [B, num_heads, LATENT+ROPE]  (absorbed)
    const __hip_bfloat16* __restrict__ latent_cache, // [num_blocks, block_size, LATENT+ROPE]
    const int* __restrict__ block_table,             // [B, max_blocks]
    const int* __restrict__ context_lens,            // [B]
    __hip_bfloat16* __restrict__ O,                  // [B, num_heads, LATENT]
    const unsigned int num_heads,
    const unsigned int block_size,
    const unsigned int max_blocks,
    const long kv_block_stride,
    const float scale,
    const unsigned int sliding_window) {
    constexpr int QK = LATENT + ROPE;
    constexpr int QK_PL = QK / WARP;      // q/k elements per lane (576/32 = 18)
    constexpr int V_PL = LATENT / WARP;   // output elements per lane (512/32 = 16) — the first V_PL

    const unsigned int head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (head >= num_heads) return;
    const unsigned int ctx_len = (unsigned int)context_lens[b];
    if (ctx_len == 0) return;

    const unsigned int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int* bt = block_table + (size_t)b * max_blocks;

    // this lane's query elements (d = lane, lane+32, ...): all QK dims (nope-latent + rope)
    const __hip_bfloat16* q_ptr = Q + (size_t)(b * num_heads + head) * QK;
    float q_reg[QK_PL];
    #pragma unroll
    for (int i = 0; i < QK_PL; i++) q_reg[i] = (float)q_ptr[i * WARP + lane];

    float m = -1e30f, l = 0.0f;
    float o_reg[V_PL];                    // output accumulates only the LATENT (V) dims
    #pragma unroll
    for (int i = 0; i < V_PL; i++) o_reg[i] = 0.0f;

    for (unsigned int j = warp_id; j < ctx_len; j += NWARPS) {
        if (sliding_window > 0 && (ctx_len - 1 - j) >= sliding_window) continue;
        const size_t base = (size_t)(unsigned int)bt[j / block_size] * kv_block_stride
                            + (size_t)(j % block_size) * QK;
        const __hip_bfloat16* kptr = latent_cache + base;
        // score = q · cache[j]  over all QK dims
        float kv[QK_PL];
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK_PL; i++) { kv[i] = (float)kptr[i * WARP + lane]; partial += q_reg[i] * kv[i]; }
        float score = warp_reduce_sum(partial) * scale;

        const float m_new = fmaxf(m, score);
        const float correct = __expf(m - m_new);
        const float p = __expf(score - m_new);
        // V = the first LATENT dims (c_KV) — reuse the kv[] we already loaded for the score
        #pragma unroll
        for (int i = 0; i < V_PL; i++) o_reg[i] = o_reg[i] * correct + p * kv[i];
        l = l * correct + p;
        m = m_new;
    }

    // cross-warp combine
    __shared__ float s_m[NWARPS];
    __shared__ float s_l[NWARPS];
    __shared__ float s_o[NWARPS][LATENT];
    if (lane == 0) { s_m[warp_id] = m; s_l[warp_id] = l; }
    #pragma unroll
    for (int i = 0; i < V_PL; i++) s_o[warp_id][i * WARP + lane] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float m_g = -1e30f;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) m_g = fmaxf(m_g, s_m[w]);
        float l_g = 0.0f, w_corr[NWARPS];
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) { w_corr[w] = __expf(s_m[w] - m_g); l_g += s_l[w] * w_corr[w]; }
        const float inv_l = (l_g > 0.0f) ? 1.0f / l_g : 0.0f;
        __hip_bfloat16* o_ptr = O + (size_t)(b * num_heads + head) * LATENT;
        #pragma unroll
        for (int i = 0; i < V_PL; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS; w++) acc += s_o[w][i * WARP + lane] * w_corr[w];
            o_ptr[i * WARP + lane] = (__hip_bfloat16)(acc * inv_l);
        }
    }
}

// fp8 LATENT-CACHE variant — the latent cache is OCP e4m3 (torch float8_e4m3fn), one byte/element,
// same paged layout. Per-tensor descale (matches the minisgl FP8_PER_TENSOR convention used by
// flash_decode_paged_fp8): k_descale folds into the pre-softmax SCORE, v_descale folds linearly into
// the OUTPUT. The latent cache is a SINGLE tensor [c_KV(LATENT) ‖ k_rope(ROPE)] with one storage
// scale, so the caller passes k_descale == v_descale == cache_descale (the two-arg form keeps the
// general per-tensor signature). q stays bf16. Dequant via the gfx12 hardware e4m3 convert.
template <int LATENT, int ROPE>
__global__ __launch_bounds__(THREADS) void mla_decode_fp8_kernel(
    const __hip_bfloat16* __restrict__ Q,            // [B, num_heads, LATENT+ROPE]  (absorbed, bf16)
    const unsigned char* __restrict__ latent_cache,  // [num_blocks, block_size, LATENT+ROPE]  e4m3
    const int* __restrict__ block_table,             // [B, max_blocks]
    const int* __restrict__ context_lens,            // [B]
    __hip_bfloat16* __restrict__ O,                  // [B, num_heads, LATENT]
    const unsigned int num_heads,
    const unsigned int block_size,
    const unsigned int max_blocks,
    const long kv_block_stride,
    const float scale,
    const float k_descale,
    const float v_descale,
    const unsigned int sliding_window) {
    constexpr int QK = LATENT + ROPE;
    constexpr int QK_PL = QK / WARP;      // q/k elements per lane (576/32 = 18)
    constexpr int V_PL = LATENT / WARP;   // output elements per lane (512/32 = 16) — the first V_PL

    const unsigned int head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (head >= num_heads) return;
    const unsigned int ctx_len = (unsigned int)context_lens[b];
    if (ctx_len == 0) return;

    const unsigned int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int* bt = block_table + (size_t)b * max_blocks;
    const float score_scale = scale * k_descale;   // k descale folds into the pre-softmax score

    const __hip_bfloat16* q_ptr = Q + (size_t)(b * num_heads + head) * QK;
    float q_reg[QK_PL];
    #pragma unroll
    for (int i = 0; i < QK_PL; i++) q_reg[i] = (float)q_ptr[i * WARP + lane];

    float m = -1e30f, l = 0.0f;
    float o_reg[V_PL];
    #pragma unroll
    for (int i = 0; i < V_PL; i++) o_reg[i] = 0.0f;

    for (unsigned int j = warp_id; j < ctx_len; j += NWARPS) {
        if (sliding_window > 0 && (ctx_len - 1 - j) >= sliding_window) continue;
        const size_t base = (size_t)(unsigned int)bt[j / block_size] * kv_block_stride
                            + (size_t)(j % block_size) * QK;
        const unsigned char* kptr = latent_cache + base;
        float kv[QK_PL];
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK_PL; i++) { kv[i] = e4m3_to_f32(kptr[i * WARP + lane]); partial += q_reg[i] * kv[i]; }
        float score = warp_reduce_sum(partial) * score_scale;

        const float m_new = fmaxf(m, score);
        const float correct = __expf(m - m_new);
        const float p = __expf(score - m_new);
        #pragma unroll
        for (int i = 0; i < V_PL; i++) o_reg[i] = o_reg[i] * correct + p * kv[i];
        l = l * correct + p;
        m = m_new;
    }

    __shared__ float s_m[NWARPS];
    __shared__ float s_l[NWARPS];
    __shared__ float s_o[NWARPS][LATENT];
    if (lane == 0) { s_m[warp_id] = m; s_l[warp_id] = l; }
    #pragma unroll
    for (int i = 0; i < V_PL; i++) s_o[warp_id][i * WARP + lane] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float m_g = -1e30f;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) m_g = fmaxf(m_g, s_m[w]);
        float l_g = 0.0f, w_corr[NWARPS];
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) { w_corr[w] = __expf(s_m[w] - m_g); l_g += s_l[w] * w_corr[w]; }
        // v descale folds linearly into the output; combine with 1/l.
        const float out_scale = (l_g > 0.0f) ? (v_descale / l_g) : 0.0f;
        __hip_bfloat16* o_ptr = O + (size_t)(b * num_heads + head) * LATENT;
        #pragma unroll
        for (int i = 0; i < V_PL; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS; w++) acc += s_o[w][i * WARP + lane] * w_corr[w];
            o_ptr[i * WARP + lane] = (__hip_bfloat16)(acc * out_scale);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// MLA multi-query VERIFY (speculative decoding). Same absorbed flash-decode math as mla_decode,
// but q packs several query tokens per sequence (confirmed + K drafts), each attending the SHARED
// paged latent under a per-query causal bound. One CTA per (head, query-row); the query row's
// causal context length is q_kbound[r] = cached_len + qi + 1 (qi = within-seq query index), so
// query qi attends latent positions [0, cached_len+qi] — itself + prefix + earlier drafts. This
// attends the paged latent DIRECTLY (no prefix re-materialization), preserving the spec speedup.
template <int LATENT, int ROPE>
__global__ __launch_bounds__(THREADS) void mla_verify_kernel(
    const __hip_bfloat16* __restrict__ Q,            // [total_q, num_heads, LATENT+ROPE] (absorbed)
    const __hip_bfloat16* __restrict__ latent_cache, // [num_blocks, block_size, LATENT+ROPE]
    const int* __restrict__ block_table,             // [B, max_blocks]
    const int* __restrict__ q_seq_idx,               // [total_q] -> seq (block_table row)
    const int* __restrict__ q_kbound,                // [total_q] -> causal context length
    __hip_bfloat16* __restrict__ O,                  // [total_q, num_heads, LATENT]
    const unsigned int num_heads,
    const unsigned int block_size,
    const unsigned int max_blocks,
    const long kv_block_stride,
    const float scale,
    const unsigned int sliding_window) {
    constexpr int QK = LATENT + ROPE;
    constexpr int QK_PL = QK / WARP;
    constexpr int V_PL = LATENT / WARP;

    const unsigned int head = blockIdx.x;
    const unsigned int r = blockIdx.y;               // query row (packed token index)
    if (head >= num_heads) return;
    const unsigned int ctx_len = (unsigned int)q_kbound[r];
    if (ctx_len == 0) return;
    const unsigned int b = (unsigned int)q_seq_idx[r];

    const unsigned int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int* bt = block_table + (size_t)b * max_blocks;

    const __hip_bfloat16* q_ptr = Q + (size_t)(r * num_heads + head) * QK;
    float q_reg[QK_PL];
    #pragma unroll
    for (int i = 0; i < QK_PL; i++) q_reg[i] = (float)q_ptr[i * WARP + lane];

    float m = -1e30f, l = 0.0f;
    float o_reg[V_PL];
    #pragma unroll
    for (int i = 0; i < V_PL; i++) o_reg[i] = 0.0f;

    for (unsigned int j = warp_id; j < ctx_len; j += NWARPS) {
        if (sliding_window > 0 && (ctx_len - 1 - j) >= sliding_window) continue;
        const size_t base = (size_t)(unsigned int)bt[j / block_size] * kv_block_stride
                            + (size_t)(j % block_size) * QK;
        const __hip_bfloat16* kptr = latent_cache + base;
        float kv[QK_PL];
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK_PL; i++) { kv[i] = (float)kptr[i * WARP + lane]; partial += q_reg[i] * kv[i]; }
        float score = warp_reduce_sum(partial) * scale;

        const float m_new = fmaxf(m, score);
        const float correct = __expf(m - m_new);
        const float p = __expf(score - m_new);
        #pragma unroll
        for (int i = 0; i < V_PL; i++) o_reg[i] = o_reg[i] * correct + p * kv[i];
        l = l * correct + p;
        m = m_new;
    }

    __shared__ float s_m[NWARPS];
    __shared__ float s_l[NWARPS];
    __shared__ float s_o[NWARPS][LATENT];
    if (lane == 0) { s_m[warp_id] = m; s_l[warp_id] = l; }
    #pragma unroll
    for (int i = 0; i < V_PL; i++) s_o[warp_id][i * WARP + lane] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float m_g = -1e30f;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) m_g = fmaxf(m_g, s_m[w]);
        float l_g = 0.0f, w_corr[NWARPS];
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) { w_corr[w] = __expf(s_m[w] - m_g); l_g += s_l[w] * w_corr[w]; }
        const float inv_l = (l_g > 0.0f) ? 1.0f / l_g : 0.0f;
        __hip_bfloat16* o_ptr = O + (size_t)(r * num_heads + head) * LATENT;
        #pragma unroll
        for (int i = 0; i < V_PL; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS; w++) acc += s_o[w][i * WARP + lane] * w_corr[w];
            o_ptr[i * WARP + lane] = (__hip_bfloat16)(acc * inv_l);
        }
    }
}

// fp8 latent-cache verify variant (e4m3 cache; q bf16). k_descale folds into the score, v_descale
// into the output — identical descale convention to mla_decode_fp8.
template <int LATENT, int ROPE>
__global__ __launch_bounds__(THREADS) void mla_verify_fp8_kernel(
    const __hip_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ latent_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ q_seq_idx,
    const int* __restrict__ q_kbound,
    __hip_bfloat16* __restrict__ O,
    const unsigned int num_heads,
    const unsigned int block_size,
    const unsigned int max_blocks,
    const long kv_block_stride,
    const float scale,
    const float k_descale,
    const float v_descale,
    const unsigned int sliding_window) {
    constexpr int QK = LATENT + ROPE;
    constexpr int QK_PL = QK / WARP;
    constexpr int V_PL = LATENT / WARP;

    const unsigned int head = blockIdx.x;
    const unsigned int r = blockIdx.y;
    if (head >= num_heads) return;
    const unsigned int ctx_len = (unsigned int)q_kbound[r];
    if (ctx_len == 0) return;
    const unsigned int b = (unsigned int)q_seq_idx[r];

    const unsigned int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
    const int* bt = block_table + (size_t)b * max_blocks;
    const float score_scale = scale * k_descale;

    const __hip_bfloat16* q_ptr = Q + (size_t)(r * num_heads + head) * QK;
    float q_reg[QK_PL];
    #pragma unroll
    for (int i = 0; i < QK_PL; i++) q_reg[i] = (float)q_ptr[i * WARP + lane];

    float m = -1e30f, l = 0.0f;
    float o_reg[V_PL];
    #pragma unroll
    for (int i = 0; i < V_PL; i++) o_reg[i] = 0.0f;

    for (unsigned int j = warp_id; j < ctx_len; j += NWARPS) {
        if (sliding_window > 0 && (ctx_len - 1 - j) >= sliding_window) continue;
        const size_t base = (size_t)(unsigned int)bt[j / block_size] * kv_block_stride
                            + (size_t)(j % block_size) * QK;
        const unsigned char* kptr = latent_cache + base;
        float kv[QK_PL];
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < QK_PL; i++) { kv[i] = e4m3_to_f32(kptr[i * WARP + lane]); partial += q_reg[i] * kv[i]; }
        float score = warp_reduce_sum(partial) * score_scale;

        const float m_new = fmaxf(m, score);
        const float correct = __expf(m - m_new);
        const float p = __expf(score - m_new);
        #pragma unroll
        for (int i = 0; i < V_PL; i++) o_reg[i] = o_reg[i] * correct + p * kv[i];
        l = l * correct + p;
        m = m_new;
    }

    __shared__ float s_m[NWARPS];
    __shared__ float s_l[NWARPS];
    __shared__ float s_o[NWARPS][LATENT];
    if (lane == 0) { s_m[warp_id] = m; s_l[warp_id] = l; }
    #pragma unroll
    for (int i = 0; i < V_PL; i++) s_o[warp_id][i * WARP + lane] = o_reg[i];
    __syncthreads();

    if (warp_id == 0) {
        float m_g = -1e30f;
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) m_g = fmaxf(m_g, s_m[w]);
        float l_g = 0.0f, w_corr[NWARPS];
        #pragma unroll
        for (int w = 0; w < NWARPS; w++) { w_corr[w] = __expf(s_m[w] - m_g); l_g += s_l[w] * w_corr[w]; }
        const float out_scale = (l_g > 0.0f) ? (v_descale / l_g) : 0.0f;
        __hip_bfloat16* o_ptr = O + (size_t)(r * num_heads + head) * LATENT;
        #pragma unroll
        for (int i = 0; i < V_PL; i++) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NWARPS; w++) acc += s_o[w][i * WARP + lane] * w_corr[w];
            o_ptr[i * WARP + lane] = (__hip_bfloat16)(acc * out_scale);
        }
    }
}

}  // namespace

// q:[B,num_heads,LATENT+ROPE]  latent_cache:[num_blocks,block_size,LATENT+ROPE]
// block_table:[B,max_blocks]  context_lens:[B]  ->  out:[B,num_heads,LATENT]
void launch_mla_decode(const at::Tensor& q, const at::Tensor& latent_cache,
                       const at::Tensor& block_table, const at::Tensor& context_lens, at::Tensor& out,
                       double scale, int64_t sliding_window, int64_t kv_block_stride) {
    const unsigned int B = q.size(0);
    const unsigned int num_heads = q.size(1);
    const unsigned int qk = q.size(2);                 // LATENT + ROPE
    const unsigned int block_size = latent_cache.size(1);
    const unsigned int max_blocks = block_table.size(1);
    const unsigned int latent = out.size(2);
    const unsigned int rope = qk - latent;
    const float sc = (float)scale;
    const long kvbs = (kv_block_stride > 0) ? (long)kv_block_stride : (long)block_size * qk;

    dim3 grid(num_heads, B, 1);
    dim3 block(THREADS, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* cp = reinterpret_cast<const __hip_bfloat16*>(latent_cache.data_ptr());
    const int* btp = block_table.data_ptr<int>();
    const int* clp = context_lens.data_ptr<int>();
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCH(L, R)                                                                          \
    mla_decode_kernel<L, R><<<grid, block, 0, stream>>>(                                      \
        qp, cp, btp, clp, op, num_heads, block_size, max_blocks, kvbs, sc,                    \
        (unsigned)sliding_window)

    // GLM-4.7-Flash / DeepSeek: kv_lora_rank=512, qk_rope_head_dim=64.
    if (latent == 512 && rope == 64) { LAUNCH(512, 64); }
    else {
        TORCH_CHECK(false, "mla_hip: unsupported (kv_lora_rank=", latent, ", qk_rope=", rope,
                    ") — add a template instantiation (LATENT%32==0, ROPE%32==0)");
    }
#undef LAUNCH
}

// fp8 latent-cache variant. latent_cache is OCP e4m3 (uint8/float8_e4m3fn); k_descale folds into the
// score, v_descale into the output (per-tensor, == cache_descale for a single-scale latent cache).
// q:[B,num_heads,LATENT+ROPE] bf16  latent_cache:[num_blocks,block_size,LATENT+ROPE] e4m3  -> [B,num_heads,LATENT]
void launch_mla_decode_fp8(const at::Tensor& q, const at::Tensor& latent_cache,
                           const at::Tensor& block_table, const at::Tensor& context_lens,
                           at::Tensor& out, double scale, double k_descale, double v_descale,
                           int64_t sliding_window, int64_t kv_block_stride) {
    const unsigned int B = q.size(0);
    const unsigned int num_heads = q.size(1);
    const unsigned int qk = q.size(2);                 // LATENT + ROPE
    const unsigned int block_size = latent_cache.size(1);
    const unsigned int max_blocks = block_table.size(1);
    const unsigned int latent = out.size(2);
    const unsigned int rope = qk - latent;
    const float sc = (float)scale;
    const long kvbs = (kv_block_stride > 0) ? (long)kv_block_stride : (long)block_size * qk;

    dim3 grid(num_heads, B, 1);
    dim3 block(THREADS, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* cp = reinterpret_cast<const unsigned char*>(latent_cache.data_ptr());
    const int* btp = block_table.data_ptr<int>();
    const int* clp = context_lens.data_ptr<int>();
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCH_FP8(L, R)                                                                      \
    mla_decode_fp8_kernel<L, R><<<grid, block, 0, stream>>>(                                  \
        qp, cp, btp, clp, op, num_heads, block_size, max_blocks, kvbs, sc,                    \
        (float)k_descale, (float)v_descale, (unsigned)sliding_window)

    if (latent == 512 && rope == 64) { LAUNCH_FP8(512, 64); }
    else {
        TORCH_CHECK(false, "mla_hip fp8: unsupported (kv_lora_rank=", latent, ", qk_rope=", rope,
                    ") — add a template instantiation (LATENT%32==0, ROPE%32==0)");
    }
#undef LAUNCH_FP8
}

// MLA multi-query VERIFY. q:[total_q,num_heads,LATENT+ROPE]  latent_cache:[blocks,block_size,QK]
// block_table:[B,max_blocks]  q_seq_idx:[total_q]  q_kbound:[total_q]  ->  out:[total_q,num_heads,LATENT]
void launch_mla_verify(const at::Tensor& q, const at::Tensor& latent_cache,
                       const at::Tensor& block_table, const at::Tensor& q_seq_idx,
                       const at::Tensor& q_kbound, at::Tensor& out, double scale,
                       int64_t sliding_window, int64_t kv_block_stride) {
    const unsigned int total_q = q.size(0);
    const unsigned int num_heads = q.size(1);
    const unsigned int qk = q.size(2);
    const unsigned int block_size = latent_cache.size(1);
    const unsigned int max_blocks = block_table.size(1);
    const unsigned int latent = out.size(2);
    const unsigned int rope = qk - latent;
    const float sc = (float)scale;
    const long kvbs = (kv_block_stride > 0) ? (long)kv_block_stride : (long)block_size * qk;

    dim3 grid(num_heads, total_q, 1);
    dim3 block(THREADS, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* cp = reinterpret_cast<const __hip_bfloat16*>(latent_cache.data_ptr());
    const int* btp = block_table.data_ptr<int>();
    const int* sip = q_seq_idx.data_ptr<int>();
    const int* kbp = q_kbound.data_ptr<int>();
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCH_V(L, R)                                                                        \
    mla_verify_kernel<L, R><<<grid, block, 0, stream>>>(                                      \
        qp, cp, btp, sip, kbp, op, num_heads, block_size, max_blocks, kvbs, sc,               \
        (unsigned)sliding_window)
    if (latent == 512 && rope == 64) { LAUNCH_V(512, 64); }
    else { TORCH_CHECK(false, "mla_hip verify: unsupported (kv_lora_rank=", latent, ", qk_rope=", rope, ")"); }
#undef LAUNCH_V
}

void launch_mla_verify_fp8(const at::Tensor& q, const at::Tensor& latent_cache,
                           const at::Tensor& block_table, const at::Tensor& q_seq_idx,
                           const at::Tensor& q_kbound, at::Tensor& out, double scale,
                           double k_descale, double v_descale, int64_t sliding_window,
                           int64_t kv_block_stride) {
    const unsigned int total_q = q.size(0);
    const unsigned int num_heads = q.size(1);
    const unsigned int qk = q.size(2);
    const unsigned int block_size = latent_cache.size(1);
    const unsigned int max_blocks = block_table.size(1);
    const unsigned int latent = out.size(2);
    const unsigned int rope = qk - latent;
    const float sc = (float)scale;
    const long kvbs = (kv_block_stride > 0) ? (long)kv_block_stride : (long)block_size * qk;

    dim3 grid(num_heads, total_q, 1);
    dim3 block(THREADS, 1, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    const auto* qp = reinterpret_cast<const __hip_bfloat16*>(q.data_ptr());
    const auto* cp = reinterpret_cast<const unsigned char*>(latent_cache.data_ptr());
    const int* btp = block_table.data_ptr<int>();
    const int* sip = q_seq_idx.data_ptr<int>();
    const int* kbp = q_kbound.data_ptr<int>();
    auto* op = reinterpret_cast<__hip_bfloat16*>(out.data_ptr());

#define LAUNCH_VF(L, R)                                                                       \
    mla_verify_fp8_kernel<L, R><<<grid, block, 0, stream>>>(                                  \
        qp, cp, btp, sip, kbp, op, num_heads, block_size, max_blocks, kvbs, sc,               \
        (float)k_descale, (float)v_descale, (unsigned)sliding_window)
    if (latent == 512 && rope == 64) { LAUNCH_VF(512, 64); }
    else { TORCH_CHECK(false, "mla_hip verify fp8: unsupported (kv_lora_rank=", latent, ", qk_rope=", rope, ")"); }
#undef LAUNCH_VF
}
