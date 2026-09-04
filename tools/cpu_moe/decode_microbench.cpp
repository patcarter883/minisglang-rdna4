// Why is the fp32-activation core only ~8.8 GB/s/core when the VNNI ceiling kernel hits 57?
// Isolate the WEIGHT-DECODE sequence (L1-resident, no DDR, no scale-LUT chain) and time the
// candidate instruction mixes head to head.
//
//   g++ -O3 -march=znver4 -std=c++17 -o decode_microbench decode_microbench.cpp
#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <vector>

static double now() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

alignas(64) static const float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                            -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
// E2M1 -> fp16 is a pure HIGH-BYTE table: every fp16 low byte is 0x00.
//   0 .5 1 1.5 2 3 4 6  ->  0x0000 3800 3C00 3E00 4000 4200 4400 4600
alignas(16) static const int8_t kE2M1_F16HI[16] = {
    0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46,
    (int8_t)0x80, (int8_t)0xB8, (int8_t)0xBC, (int8_t)0xBE,
    (int8_t)0xC0, (int8_t)0xC2, (int8_t)0xC4, (int8_t)0xC6};
// E2M1 -> exact int8 (2x magnitude), the VNNI representation: {0,1,2,3,4,6,8,12}
alignas(16) static const int8_t kE2M1_I8[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                                0, -1, -2, -3, -4, -6, -8, -12};

static const size_t NB = 1 << 16;  // 64 KB of codes = 128K weights, L1/L2 resident
static const int REP = 4000;

// FOUR independent accumulator chains per variant.  With one chain every variant measures
// vfmadd LATENCY (4 cycles) instead of issue throughput, which made the first run of this
// microbench report the no-decode roofline as SLOWER than the real kernel.
#define TIME(name, WPI, BODY)                                                        \
    {                                                                                \
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();                   \
        __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();                   \
        __m512i i0 = _mm512_setzero_si512(), i1 = _mm512_setzero_si512();            \
        __m512i i2 = _mm512_setzero_si512(), i3 = _mm512_setzero_si512();            \
        const double t0 = now();                                                     \
        for (int r = 0; r < REP; ++r)                                                \
            for (size_t i = 0; i + 64 <= NB; i += 32) {                              \
                const uint8_t* cp = codes.data() + i;                                \
                BODY                                                                 \
            }                                                                        \
        const double dt = now() - t0;                                                \
        const double w = (double)REP * (NB / 32) * (WPI);                            \
        printf("  %-34s %7.2f Gweight/s  %6.2f GB/s  %5.2f w/cycle@5.05GHz\n", name, \
               w / dt / 1e9, w * 0.625 / dt / 1e9, w / dt / 5.05e9);                 \
        __m512 acc = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));    \
        __m512i iacc = _mm512_add_epi32(_mm512_add_epi32(i0, i1),                    \
                                        _mm512_add_epi32(i2, i3));                   \
        if (_mm512_reduce_add_ps(acc) + (float)_mm512_reduce_add_epi32(iacc) ==      \
            12345.678f)                                                              \
            puts("");                                                                \
    }

int main() {
    std::vector<uint8_t> codes(NB);
    for (size_t i = 0; i < NB; ++i) codes[i] = (uint8_t)(i * 131 + 7);
    std::vector<float> x(1 << 16, 1.0f);
    const __m512 lut = _mm512_load_ps(kE2M1);
    const __m512i lut_f16 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*)kE2M1_F16HI));
    const __m512i lut_i8 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*)kE2M1_I8));
    const __m128i m0F128 = _mm_set1_epi8(0x0F);
    const __m512i m0F512 = _mm512_set1_epi8(0x0F);
    const __m512i zero = _mm512_setzero_si512();
    const __m512 sv = _mm512_set1_ps(0.01f);
    printf("decode-only throughput, L1-resident codes, single thread\n");

    // (A) what the core does today: vpmovzxbd + vpermps, 32 weights per 16 code bytes
    TIME("A cvtepu8_epi32 + vpermps", 32, {
        const __m128i b = _mm_loadu_si128((const __m128i*)cp);
        const __m512i lo = _mm512_cvtepu8_epi32(_mm_and_si128(b, m0F128));
        const __m512i hi = _mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(b, 4), m0F128));
        s0 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_permutexvar_ps(lo, lut), sv),
                             _mm512_loadu_ps(x.data() + (i & 0x3FFF)), s0);
        s1 = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_permutexvar_ps(hi, lut), sv),
                             _mm512_loadu_ps(x.data() + (i & 0x3FFF) + 16), s1);
    })

    // (B) vpshufb -> fp16 high bytes -> vcvtph2ps, 64 weights per 32 code bytes
    TIME("B vpshufb->fp16->vcvtph2ps", 64, {
        const __m256i b = _mm256_loadu_si256((const __m256i*)cp);
        const __m512i bb = _mm512_castsi256_si512(b);
        const __m512i lo = _mm512_and_si512(bb, m0F512);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(bb, 4), m0F512);
        const __m512i he = _mm512_shuffle_epi8(lut_f16, lo);
        const __m512i ho = _mm512_shuffle_epi8(lut_f16, hi);
        const __m512i e0 = _mm512_unpacklo_epi8(zero, he);
        const __m512i e1 = _mm512_unpackhi_epi8(zero, he);
        const __m512i o0 = _mm512_unpacklo_epi8(zero, ho);
        const __m512i o1 = _mm512_unpackhi_epi8(zero, ho);
        const __m512 f0 = _mm512_cvtph_ps(_mm512_castsi512_si256(e0));
        const __m512 f1 = _mm512_cvtph_ps(_mm512_castsi512_si256(e1));
        const __m512 f2 = _mm512_cvtph_ps(_mm512_castsi512_si256(o0));
        const __m512 f3 = _mm512_cvtph_ps(_mm512_castsi512_si256(o1));
        const __m512 xv = _mm512_loadu_ps(x.data() + (i & 0x3FFF));
        s0 = _mm512_fmadd_ps(_mm512_mul_ps(f0, sv), xv, s0);
        s1 = _mm512_fmadd_ps(_mm512_mul_ps(f1, sv), xv, s1);
        s2 = _mm512_fmadd_ps(_mm512_mul_ps(f2, sv), xv, s2);
        s3 = _mm512_fmadd_ps(_mm512_mul_ps(f3, sv), xv, s3);
    })

    // (C) VNNI: vpshufb -> exact int8, vpdpbusd, NO int->float per weight.
    //     128 weights per 64 code bytes, 2 shuffles + 2 dot products.
    TIME("C vpshufb->int8->vpdpbusd", 128, {
        const __m512i b = _mm512_loadu_si512((const __m512i*)cp);
        const __m512i lo = _mm512_shuffle_epi8(lut_i8, _mm512_and_si512(b, m0F512));
        const __m512i hi = _mm512_shuffle_epi8(lut_i8, _mm512_and_si512(_mm512_srli_epi16(b, 4), m0F512));
        i0 = _mm512_dpbusd_epi32(i0, _mm512_set1_epi8(3), lo);
        i1 = _mm512_dpbusd_epi32(i1, _mm512_set1_epi8(3), hi);
    })

    // (D) A, but with the scale multiply removed (is the extra vmulps material?)
    TIME("D  A without the scale multiply", 32, {
        const __m128i b = _mm_loadu_si128((const __m128i*)cp);
        const __m512i lo = _mm512_cvtepu8_epi32(_mm_and_si128(b, m0F128));
        const __m512i hi = _mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(b, 4), m0F128));
        s0 = _mm512_fmadd_ps(_mm512_permutexvar_ps(lo, lut),
                             _mm512_loadu_ps(x.data() + (i & 0x3FFF)), s0);
        s1 = _mm512_fmadd_ps(_mm512_permutexvar_ps(hi, lut),
                             _mm512_loadu_ps(x.data() + (i & 0x3FFF) + 16), s1);
    })

    // (E) pure FMA roofline: no decode at all
    TIME("E fp32 fmadd roofline (no decode)", 32, {
        (void)cp;
        s0 = _mm512_fmadd_ps(sv, _mm512_loadu_ps(x.data() + (i & 0x3FFF)), s0);
        s1 = _mm512_fmadd_ps(sv, _mm512_loadu_ps(x.data() + (i & 0x3FFF) + 16), s1);
    })
    return 0;
}
