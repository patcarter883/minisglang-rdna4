// The ONE CPU MoE compute core.  Parameterised on a WLoad policy (wload.hpp); nothing in this
// file knows what a scale byte means.
//
// Shape: decode-time GEMV, y[N] = W[N,K] . x[K], W held 4-bit E2M1 + a per-group scale.
// The workload is DDR-bandwidth-bound (measured 55 GB/s ceiling on this box vs a 374 GB/s
// cache-resident compute ceiling), so the core keeps ACTIVATIONS IN FP32 -- there is no reason
// to pay the accuracy of an int8/q8_0 activation quantization to buy compute we do not need.
//
// Per 32 weights (16 bytes of codes):
//   1 load, 3 int ops to split nibbles, 2 vpmovzxbd, 2 vpermps through the 16-float E2M1 table,
//   1 policy scale vector, 2 mul, 2 fmadd.  No horizontal reduction until the row ends.
//
// The even/odd nibble split makes lane i of the "even" vector weight k = k0 + 2i, so the
// activation is pre-deinterleaved ONCE per matrix into xe[j] = x[2j], xo[j] = x[2j+1].
#pragma once

#include "wload.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

// Tiles of E2M1 codes to prefetch ahead in the VNNI core; 0 = rely on the HW streamer.
#ifndef CPU_MOE_VNNI_PF
#define CPU_MOE_VNNI_PF 0
#endif

// x -> (xe, xo) with xe[j] = x[2j], xo[j] = x[2j+1].  K/2 floats each.
static inline void deinterleave(const float* __restrict x, int K, float* __restrict xe,
                                float* __restrict xo) {
    for (int j = 0; j < K / 2; ++j) {
        xe[j] = x[2 * j];
        xo[j] = x[2 * j + 1];
    }
}

// ---------------------------------------------------------------------------------------------
// THE CORE.  rows [r0, r1) of one E2M1 matrix.  ACCUMULATE ? out += s*dot : out = dot -- the two
// things a MoE layer needs from a GEMV.
//
// Rows are processed RB at a time so the activation loads amortize: the x vector is shared by
// every row, so an RB-row block issues 2 zmm x-loads per 32 weights instead of 2*RB, and gives
// 2*RB independent accumulator chains to hide the 4-cycle FMA latency.
template <class WL, bool ACCUMULATE, int RB>
static inline void gemv_block(const uint8_t* __restrict codes,
                              const typename WL::scale_t* __restrict scales,
                              const typename WL::ctx_t& ctx, const float* __restrict xe,
                              const float* __restrict xo, int K, int r, float* __restrict out,
                              float acc_scale) {
    const __m512 lut = e2m1_lut();
    const __m128i nib = _mm_set1_epi8(0x0F);
    const size_t cstride = (size_t)K >> 1;
    const size_t sstride = (size_t)K / WL::GROUP;

    const uint8_t* __restrict cp[RB];
    const typename WL::scale_t* __restrict sp[RB];
    __m512 ae[RB], ao[RB];
    for (int j = 0; j < RB; ++j) {
        cp[j] = codes + (size_t)(r + j) * cstride;
        sp[j] = scales + (size_t)(r + j) * sstride;
        ae[j] = _mm512_setzero_ps();
        ao[j] = _mm512_setzero_ps();
    }

    for (int k0 = 0; k0 < K; k0 += 32) {
        const __m512 xv_e = _mm512_loadu_ps(xe + (k0 >> 1));
        const __m512 xv_o = _mm512_loadu_ps(xo + (k0 >> 1));
        for (int j = 0; j < RB; ++j) {
            const __m128i b = _mm_loadu_si128((const __m128i*)(cp[j] + (k0 >> 1)));
            const __m512i lo = _mm512_cvtepu8_epi32(_mm_and_si128(b, nib));
            const __m512i hi = _mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(b, 4), nib));
            const __m512 sv = WL::blk_scale(sp[j], k0, ctx);
            ae[j] = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_permutexvar_ps(lo, lut), sv), xv_e, ae[j]);
            ao[j] = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_permutexvar_ps(hi, lut), sv), xv_o, ao[j]);
        }
    }
    for (int j = 0; j < RB; ++j) {
        const float dot = _mm512_reduce_add_ps(_mm512_add_ps(ae[j], ao[j]));
        if (ACCUMULATE) out[r + j] += acc_scale * dot;
        else out[r + j] = dot;
    }
}

template <class WL, bool ACCUMULATE>
static inline void gemv_e2m1(const uint8_t* __restrict codes,
                             const typename WL::scale_t* __restrict scales,
                             const typename WL::ctx_t& ctx, const float* __restrict xe,
                             const float* __restrict xo, int N, int K, int r0, int r1,
                             float* __restrict out, float acc_scale) {
    (void)N;
    // Row-block width. Measured on this box (znver4, gcc 16.2): RB=1 wins; see RESULTS.
#ifndef CPU_MOE_ROW_BLOCK
#define CPU_MOE_ROW_BLOCK 1
#endif
    constexpr int RB = CPU_MOE_ROW_BLOCK;
    int r = r0;
    for (; r + RB <= r1; r += RB)
        gemv_block<WL, ACCUMULATE, RB>(codes, scales, ctx, xe, xo, K, r, out, acc_scale);
    for (; r < r1; ++r)
        gemv_block<WL, ACCUMULATE, 1>(codes, scales, ctx, xe, xo, K, r, out, acc_scale);
}

// =============================================================================================
// THE INT8-ACTIVATION (VNNI) CORE.
// =============================================================================================
// Same contract as gemv_e2m1 above -- rows [.,.) of one E2M1 matrix, ACCUMULATE ? out += s*dot
// : out = dot -- and the same WLoad policy split: the ONLY policy-dependent line in the loop is
// WL::tile_scale().  Two policies (D: e4m3 0.5625 B/w, E: fp16 0.625 B/w) share this body.
//
// It is a SECOND core, not a fourth WLoad on the first one, and that is deliberate rather than
// a copy-paste fork.  KERNEL_CORE_POLICY permits exactly one exemption -- "a genuinely
// different algorithm or tiling" -- and this is it: VPDPBUSD reduces 4 k values into each int32
// lane, so the lane index must be a ROW, and the resident bytes must be tiled 16 rows x 16 k.
// The fp32 core puts k in the lanes and reads row-major.  You cannot express one as a policy on
// the other; every line of the inner loop differs (vpshufb vs vpermps, u8xs8 vs fp32 FMA, no
// horizontal reduction vs one per row, broadcast activation vs vector activation).
// What IS shared and must stay shared: the E2M1 codebook (kE2M1I8 is checked against kE2M1 by
// e2m1_tables_agree()), the WLoad scale semantics, ExpertSlab, silu_mul, the row-range
// partition contract, and the accumulate/store output convention.
//
// Per tile (256 weights, 128 B of codes):
//   2 loads, 2 shifts + 4 ands, 4 vpshufb, 4 vpdpbusd, 3 vpaddd, 1 vpsubd, 1 cvtdq2ps,
//   1 policy scale vector, 1 mul, 1 fmadd.  VPDPBUSD retires 64 weights per instruction against
//   VFMADD's 16, and there is no int->float conversion in the weight path at all.

// A quantized activation vector: int8 codes, one fp32 scale per group of 16, and the group sum
// that removes VPDPBUSD's +16 unsigned bias.  Groups match the WEIGHT groups, so the two scales
// multiply into a single per-(row,group) constant.
struct QAct {
    const int8_t* q;      // K
    const float* sc;      // K/16
    const int32_t* sum;   // K/16   (sum of q over the group)
};

// fp32 -> int8, symmetric, per group of 16.  Round-to-nearest-even via cvtps_epi32.
// amax/127 (not /128) so the endpoints are representable; an all-zero group gets scale 0.
static inline void quantize_act_g16_range(const float* __restrict x, int g0, int g1,
                                          int8_t* __restrict q, float* __restrict sc,
                                          int32_t* __restrict sum) {
    for (int g = g0; g < g1; ++g) {
        const __m512 v = _mm512_loadu_ps(x + g * 16);
        const float amax = _mm512_reduce_max_ps(_mm512_abs_ps(v));
        const float s = amax / 127.0f;
        const float inv = (amax > 0.0f) ? 127.0f / amax : 0.0f;
        const __m512i qi = _mm512_cvtps_epi32(_mm512_mul_ps(v, _mm512_set1_ps(inv)));
        _mm_storeu_si128((__m128i*)(q + g * 16), _mm512_cvtsepi32_epi8(qi));
        sc[g] = s;
        sum[g] = _mm512_reduce_add_epi32(qi);
    }
}

static inline void quantize_act_g16(const float* __restrict x, int K, int8_t* __restrict q,
                                    float* __restrict sc, int32_t* __restrict sum) {
    quantize_act_g16_range(x, 0, K / 16, q, sc, sum);
}

// ---------------------------------------------------------------------------------------------
// Scalar twin of gemv_e2m1_vnni, in double.  Not a performance path -- it exists so the tile
// layout, the +16 bias correction and the two scale foldings are checked independently of the
// intrinsics, and so a policy can be added without trusting the SIMD to validate itself.
template <class WL>
static inline void gemv_e2m1_vnni_ref(const uint8_t* codes, const typename WL::scale_t* scales,
                                      const typename WL::ctx_t& ctx, const QAct& x, int N, int K,
                                      double* out) {
    const int NG = K / WL::GROUP;
    for (int rb = 0; rb < N / VNNI_RB; ++rb) {
        for (int r = 0; r < VNNI_RB; ++r) {
            double s = 0;
            for (int g = 0; g < NG; ++g) {
                const uint8_t* B = codes + ((size_t)rb * NG + g) * VNNI_TILE_W;
                double gs = 0;
                for (int j = 0; j < 16; ++j) {
                    int b, code;
                    if (j < 4)        { b = r * 4 + j;             code = B[b] & 15; }
                    else if (j < 8)   { b = r * 4 + (j - 4);       code = B[b] >> 4; }
                    else if (j < 12)  { b = 64 + r * 4 + (j - 8);  code = B[b] & 15; }
                    else              { b = 64 + r * 4 + (j - 12); code = B[b] >> 4; }
                    gs += 0.5 * (double)kE2M1I8[code] * (double)x.q[g * 16 + j];
                }
                s += gs * (double)WL::scale_ref(scales + ((size_t)rb * NG + g) * VNNI_RB, r, ctx) *
                     (double)WL::post_scale(ctx) * (double)x.sc[g];
            }
            out[rb * VNNI_RB + r] = s;
        }
    }
}

// rows [rb0*16, rb1*16) -- row BLOCKS, because the tile is the unit of the layout.
template <class WL, bool ACCUMULATE>
static inline void gemv_e2m1_vnni(const uint8_t* __restrict codes,
                                  const typename WL::scale_t* __restrict scales,
                                  const typename WL::ctx_t& ctx, const QAct& x, int N, int K,
                                  int rb0, int rb1, float* __restrict out, float acc_scale) {
    (void)N;
    const __m512i lut = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*)kE2M1U8));
    const __m512i mlo = _mm512_set1_epi8(0x0F);
    const int NG = K / WL::GROUP;
    const __m512 as = _mm512_set1_ps(acc_scale);
    // per-MATRIX scalar (the NVFP4 per-tensor global) folded once, not per tile
    const float post = WL::post_scale(ctx);

    for (int rb = rb0; rb < rb1; ++rb) {
        __m512 yacc = _mm512_setzero_ps();
        const uint8_t* __restrict wp = codes + (size_t)rb * NG * VNNI_TILE_W;
        const typename WL::scale_t* __restrict sp = scales + (size_t)rb * NG * VNNI_RB;
        for (int g = 0; g < NG; ++g, wp += VNNI_TILE_W, sp += VNNI_RB) {
#if CPU_MOE_VNNI_PF
            _mm_prefetch((const char*)(wp + CPU_MOE_VNNI_PF * VNNI_TILE_W), _MM_HINT_T0);
#endif
            const __m512i z0 = _mm512_loadu_si512((const void*)wp);
            const __m512i z1 = _mm512_loadu_si512((const void*)(wp + 64));
            const __m512i w0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(z0, mlo));
            const __m512i w1 =
                _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(z0, 4), mlo));
            const __m512i w2 = _mm512_shuffle_epi8(lut, _mm512_and_si512(z1, mlo));
            const __m512i w3 =
                _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(z1, 4), mlo));
            const int32_t* xq32 = (const int32_t*)(x.q + (size_t)g * VNNI_GS);
            // Four INDEPENDENT accumulators, not one chain: VPDPBUSD is ~4-cycle latency on
            // znver4 and a 4-long chain per tile would serialise the loop.
            const __m512i z = _mm512_setzero_si512();
            const __m512i a0 = _mm512_dpbusd_epi32(z, w0, _mm512_set1_epi32(xq32[0]));
            const __m512i a1 = _mm512_dpbusd_epi32(z, w1, _mm512_set1_epi32(xq32[1]));
            const __m512i a2 = _mm512_dpbusd_epi32(z, w2, _mm512_set1_epi32(xq32[2]));
            const __m512i a3 = _mm512_dpbusd_epi32(z, w3, _mm512_set1_epi32(xq32[3]));
            __m512i acc = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
            // remove the +16 unsigned bias: sum((w+16)*xq) - 16*sum(xq)
            acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(16 * x.sum[g]));
            // ONE scalar folds the E2M1 x2, the activation group scale and the per-tensor
            // global; only the 16 per-row weight scales stay a vector.
            const __m512 sc = _mm512_mul_ps(WL::tile_scale(sp, ctx),
                                            _mm512_set1_ps(0.5f * x.sc[g] * post));
            yacc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), sc, yacc);
        }
        float* o = out + (size_t)rb * VNNI_RB;
        if (ACCUMULATE)
            _mm512_storeu_ps(o, _mm512_fmadd_ps(yacc, as, _mm512_loadu_ps(o)));
        else
            _mm512_storeu_ps(o, yacc);
    }
}

// ---------------------------------------------------------------------------------------------
// One expert's resident slab.  Codes and scales for a projection are adjacent, so each of the
// three projections is one ~900 KB sequential run -- the slab size the DDR gather measurement
// found fastest (54-55 GB/s at 256 KB-1 MB, falling to 51 at 3 MB and 41.6 at 16 KB).
template <class WL>
struct ExpertSlab {
    const uint8_t* gate_c;
    const typename WL::scale_t* gate_s;
    const uint8_t* up_c;
    const typename WL::scale_t* up_s;
    const uint8_t* down_c;
    const typename WL::scale_t* down_s;
    typename WL::ctx_t gate_ctx, up_ctx, down_ctx;
};

static inline float silu_mul(float g, float u) { return g / (1.0f + expf(-g)) * u; }
