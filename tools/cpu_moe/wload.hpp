// WLoad policies for the CPU MoE core.
//
// KERNEL CORE POLICY (rdna4-hip-kernels/KERNEL_CORE_POLICY.md), applied on the host side:
// a new weight format is a LOADER POLICY on the one shared core, never a forked kernel.
// Everything below supplies exactly two things:
//
//   scale_t          the resident per-group scale element type
//   GROUP            how many weights share one scale
//   blk_scale()      -> the 16-lane float scale vector for one 32-weight block of the core
//
// The E2M1 nibble unpack, the permute-LUT dequant, the FMA and the accumulator all live ONCE,
// in moe_core.hpp.  Adding int4/W8A16/a different scale encoding means adding a struct here.
#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cmath>

// OCP E2M1 codebook, indexed by the raw 4-bit code (bit3 = sign).  Must match
// python/minisgl/quant/mxfp4.py::FP4_E2M1_LUT and the HIP kernel's e2m1_to_e4m3 LUT.
alignas(64) static const float kE2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static inline __m512 e2m1_lut() { return _mm512_load_ps(kE2M1); }

// --------------------------------------------------------------------------------------------
// Policy A: NVFP4 exactly as the checkpoint stores it.
//   codes  uint8 (N, K/2)   E2M1, low nibble = lower K
//   scale  e4m3  (N, K/16)
//   plus a per-TENSOR f32 global (`weight_scale_2` in this checkpoint), a MULTIPLIER.
//
// The per-tensor global is folded into a 256-entry float table (one table per matrix), which is
// the fp32 analogue of quant/nvfp4.py::fold_nvfp4_scale -- exact, and it costs ZERO resident
// bytes, so the CPU side holds 2.7648 MB/expert against the GPU path's 3.072 MB (fp16 scales).
// NOTE the sign of the fold: quant/nvfp4.py DIVIDES by `weight_global_scale` (llm-compressor's
// 448*6/amax); this checkpoint's `weight_scale_2` is its RECIPROCAL, so we MULTIPLY.  See
// make_fixture.py for the evidence that pins it.
struct WLoadNvfp4E4m3 {
    using scale_t = uint8_t;
    static constexpr int GROUP = 16;
    static constexpr bool TILED = false;
    static constexpr const char* NAME = "nvfp4_e4m3_g16";
    struct ctx_t { const float* lut256; };  // e4m3(byte) / weight_global_scale

    static inline __m512 blk_scale(const scale_t* srow, int k0, const ctx_t& c) {
        const int g = k0 >> 4;  // 32 weights = 2 groups
        const float s0 = c.lut256[srow[g]];
        const float s1 = c.lut256[srow[g + 1]];
        // even lanes hold k0+0,2,..,30 -> lanes 0..7 are in group g, lanes 8..15 in group g+1.
        // The odd-lane block (k0+1,3,..,31) partitions identically, so ONE vector serves both.
        return _mm512_insertf32x8(_mm512_set1_ps(s0), _mm256_set1_ps(s1), 1);
    }
    // Scalar twin of blk_scale, for references/tests.  Same policy, no SIMD.
    static inline float scale_ref(const scale_t* srow, int k, const ctx_t& c) {
        return c.lut256[srow[k / GROUP]];
    }
};

// --------------------------------------------------------------------------------------------
// Policy B: NVFP4 already folded to the fp16 per-group scale the HIP W4A8 path holds in VRAM.
// Byte-identical to the GPU-resident layout, so a hybrid (some layers streamed, some computed
// here) can share ONE host table instead of keeping two.
struct WLoadNvfp4Fp16 {
    using scale_t = uint16_t;
    static constexpr int GROUP = 16;
    static constexpr bool TILED = false;
    static constexpr const char* NAME = "nvfp4_fp16_g16";
    struct ctx_t { const float* lut256; };  // unused

    static inline __m512 blk_scale(const scale_t* srow, int k0, const ctx_t&) {
        const int g = k0 >> 4;
        const __m128i h = _mm_cvtsi32_si128((int)srow[g] | ((int)srow[g + 1] << 16));
        const __m128 f = _mm_cvtph_ps(h);  // [s0, s1, ., .]
        return _mm512_insertf32x8(_mm512_set1_ps(_mm_cvtss_f32(f)),
                                  _mm256_set1_ps(_mm_cvtss_f32(_mm_movehdup_ps(f))), 1);
    }
    static inline float scale_ref(const scale_t* srow, int k, const ctx_t&) {
        return _cvtsh_ss(srow[k / GROUP]);
    }
};

// --------------------------------------------------------------------------------------------
// Policy C: OCP MXFP4 -- identical E2M1 codes, an E8M0 power-of-two scale, group 32.
// Present to prove the core really is format-parameterised (different GROUP, different scale
// decode, zero lines of the core touched), and because the survey's transcode-to-MXFP4 option
// would land here.
struct WLoadMxfp4E8m0 {
    using scale_t = uint8_t;
    static constexpr int GROUP = 32;
    static constexpr bool TILED = false;
    static constexpr const char* NAME = "mxfp4_e8m0_g32";
    struct ctx_t { const float* lut256; };  // 2^(b-127)

    static inline __m512 blk_scale(const scale_t* srow, int k0, const ctx_t& c) {
        return _mm512_set1_ps(c.lut256[srow[k0 >> 5]]);
    }
    static inline float scale_ref(const scale_t* srow, int k, const ctx_t& c) {
        return c.lut256[srow[k / GROUP]];
    }
};

// =============================================================================================
// VNNI (int8-activation) POLICIES
// =============================================================================================
// Same three things as above -- scale_t, GROUP, a scale accessor -- for the int8-activation
// core in moe_core.hpp.  They differ from policies A/B ONLY in the resident TILING, which is
// forced by the arithmetic and is the one exemption KERNEL_CORE_POLICY names ("a genuinely
// different algorithm or tiling"):
//
//   fp32 core : lane = a k index, one row at a time, activation loaded as a vector.
//   VNNI core : lane = a ROW, activation BROADCAST as an int32 of 4 int8s, because VPDPBUSD
//               reduces 4 k per int32 lane.  Putting rows in lanes is what makes the group
//               scale a plain 16-lane vector multiply and removes every horizontal reduction.
//
// Tile (16 rows x 16 k) = 128 B of E2M1 codes + 16 rows' worth of scale.  Nibble placement,
// which the repacker in cpu_moe_layer.cpp and the scalar reference must both honour:
//     j in [0,4)   -> byte  r*4 + (j)      low nibble
//     j in [4,8)   -> byte  r*4 + (j-4)    high nibble
//     j in [8,12)  -> byte  64 + r*4 + (j-8)   low nibble
//     j in [12,16) -> byte  64 + r*4 + (j-12)  high nibble
// so byte i of the 64-B half decodes to row i/4, k offset i%4 -- i.e. int32 lane j == row j.
static inline void build_e4m3_lut(float* out, float global_mul);  // defined below

static constexpr int VNNI_RB = 16;       // rows per tile
static constexpr int VNNI_GS = 16;       // k per tile == the NVFP4 group
static constexpr int VNNI_TILE_W = 128;  // bytes of codes per tile

// E2M1 as an EXACT int8: the codebook magnitudes {0,.5,1,1.5,2,3,4,6} doubled are the integers
// {0,1,2,3,4,6,8,12}, so 4-bit float -> int8 is lossless and the implicit /2 is folded into the
// group scale.  +16 makes it unsigned for VPDPBUSD's (u8 x s8) form; the bias is removed with a
// per-group  16*sum(xq[g])  correction that every row in the tile shares.
alignas(16) static const int8_t kE2M1I8[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
alignas(16) static const uint8_t kE2M1U8[16] = {16, 17, 18, 19, 20, 22, 24, 28,
                                                16, 15, 14, 13, 12, 10,  8,  4};

// The int8 codebook is a SECOND encoding of kE2M1, so it is checked against it, not trusted.
static inline bool e2m1_tables_agree() {
    for (int i = 0; i < 16; ++i) {
        if ((float)kE2M1I8[i] != kE2M1[i] * 2.0f) return false;
        if ((int)kE2M1U8[i] != kE2M1I8[i] + 16) return false;
    }
    return true;
}

// 16 e4m3 bytes -> 16 floats, by bit surgery rather than a 256-entry gather (vgatherdps on
// znver4 is ~20 cycles, which at one gather per 256 weights would dominate the loop).
//
// SPECIALISED to POSITIVE NORMAL e4m3, i.e. bytes 0x08..0x7E.  For those,
//     value = (1 + m/8) * 2^(e-7)  ->  fp32 bits = ((e<<3)|m) << 20  +  120<<23
// and since (e<<3)|m IS the byte once the sign bit is known zero, the whole decode is
//     cvtepu8 -> shift -> add.  Three ops per 16 scales.
//
// That is legal because it was MEASURED, not assumed: over all 629,145,600 weight_scale bytes
// in layers 0-3 of this checkpoint (tools/cpu_moe scale-byte census) the exponent field spans
// 8..15 and the byte range is 64..126 -- zero subnormals (e==0), zero NaN slots (0x7F/0xFF),
// zero negatives, zero zeros.  A block scale is an amax divided by a positive constant, so
// none of those CAN occur in a well-formed NVFP4 tensor.
//
// The general case still exists, in build_e4m3_lut() below, and is what WLoadVnniE4m3's
// scale_ref() (and therefore the scalar reference and --mode selftest) uses.  The two are
// interchangeable only on the range above, so e4m3_scales_normpos() re-checks every byte at
// LOAD time and the loader dies loudly if a checkpoint ever violates it.  Do not delete that
// guard to make a new checkpoint load.
//
// Cost of getting this wrong the general way: the fully-branched version (19 ops, subnormal
// blend + NaN mask) made policy D 29.7% SLOWER than the fp16-scale policy E at 1 thread, where
// the loop is ALU-bound rather than DDR-bound -- it inverted the layout decision.
static inline __m512 e4m3x16_normpos_to_ps(const uint8_t* p) {
    const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)p));
    return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_slli_epi32(v, 20),
                                                _mm512_set1_epi32(120 << 23)));
}

// The precondition above, checked on real bytes at load time.  Returns the offending byte.
static inline bool e4m3_scales_normpos(const uint8_t* p, size_t n, int* bad) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] < 0x08 || p[i] > 0x7E) { *bad = p[i]; return false; }
    return true;
}

// --------------------------------------------------------------------------------------------
// Policy D: NVFP4 in the checkpoint's OWN bytes (e4m3 group scale + per-tensor multiplier),
// VNNI-tiled.  0.5625 B/weight = 2,764,800 B/expert -- the same 10%-smaller, ~1800x more
// accurate resident layout policy A proved, so int8 activations cost NO extra DDR traffic.
struct WLoadVnniE4m3 {
    using scale_t = uint8_t;
    static constexpr int GROUP = 16;
    static constexpr bool TILED = true;
    static constexpr const char* NAME = "vnni_nvfp4_e4m3_g16";

    // THE SECOND SCALE LEVEL HAS TWO SHAPES AND THEY ARE NOT INTERCHANGEABLE.
    //   gmul  a per-MATRIX multiplier -- what a RAW checkpoint leaf carries (`weight_scale_2` is
    //         one f32 per tensor), and what cpu_moe_layer.cpp's bench reads straight off disk.
    //   gvec  a per-OUTPUT-CHANNEL f32 vector of length N, in ROW order -- what the ENGINE holds
    //         (`_GroupedNvFp4Experts._global_op`, (E, N) f32). It is a vector rather than a scalar
    //         because the loader's gate|up merge and per-expert stack combine differently-scaled
    //         matrices: after the merge the global is constant on each contiguous output-channel
    //         RANGE, which an N-vector expresses and a scalar cannot (quant/nvfp4.py, "WHY THE
    //         GLOBAL IS A PER-OUTPUT-CHANNEL VECTOR").
    // gvec WINS when set; gmul is then unused. Serving the engine's tensors through the scalar
    // form would apply one channel's global to all N, so the two are kept distinct rather than
    // collapsed into "the global".
    struct ctx_t {
        float gmul = 1.0f;
        const float* gvec = nullptr;  // N floats, row-major; nullptr -> use gmul
    };

    // the 16 RAW row-scales of one tile.  The second level is NOT applied here: as a scalar it
    // folds into the per-group constant (post_scale), and as a vector it is constant along k so it
    // folds ONCE PER ROW BLOCK at the end of the row (post_vec) -- one zmm multiply per ~K/16
    // tiles either way, never one per tile.
    static inline __m512 tile_scale(const scale_t* sp, const ctx_t&) {
        return e4m3x16_normpos_to_ps(sp);
    }
    // Applied per row BLOCK by the core, and only for policies that ask for it.
    static constexpr bool POST_PER_ROW = true;
    static inline float post_scale(const ctx_t& c) { return c.gvec ? 1.0f : c.gmul; }
    static inline __m512 post_vec(const ctx_t& c, int rb) {
        return c.gvec ? _mm512_loadu_ps(c.gvec + (size_t)rb * VNNI_RB) : _mm512_set1_ps(1.0f);
    }
    // Scalar twin of post_scale x post_vec[lane], for the reference.
    static inline float post_ref(const ctx_t& c, int rb, int r) {
        return c.gvec ? c.gvec[(size_t)rb * VNNI_RB + r] : c.gmul;
    }
    // Scalar twin: the GENERAL e4m3 decode, not the specialised one, so the reference does not
    // inherit the fast path's precondition.
    static inline float scale_ref(const scale_t* sp, int r, const ctx_t&) {
        float t[256];
        build_e4m3_lut(t, 1.0f);
        return t[sp[r]];
    }
};

// --------------------------------------------------------------------------------------------
// Policy E: the same weights with the global pre-folded into an fp16 group scale -- the layout
// the GPU W4A8 path holds, and the one the ceiling probe measured.  0.625 B/weight, i.e. +11%
// DDR traffic per token.  Present so the scale-width cost can be A/B'd against policy D on
// identical bytes rather than argued about.
struct WLoadVnniFp16 {
    using scale_t = uint16_t;
    static constexpr int GROUP = 16;
    static constexpr bool TILED = true;
    static constexpr const char* NAME = "vnni_nvfp4_fp16_g16";
    struct ctx_t { float gmul; };  // 1.0f; already folded at repack time

    static inline __m512 tile_scale(const scale_t* sp, const ctx_t&) {
        return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)sp));
    }
    // No second level AT ALL: it is inside the fp16 scale. There is deliberately no `gvec` field
    // here, so a caller that has an engine-shaped per-channel global cannot hand it to this policy
    // and have it silently ignored -- the code does not compile instead.
    static constexpr bool POST_PER_ROW = false;
    static inline float post_scale(const ctx_t&) { return 1.0f; }  // folded in at repack time
    static inline float post_ref(const ctx_t&, int, int) { return 1.0f; }
    static inline float scale_ref(const scale_t* sp, int r, const ctx_t&) {
        return _cvtsh_ss(sp[r]);
    }
};

// e4m3 (float8_e4m3fn: bias 7, no inf, 0x7F/0xFF = NaN) -> float, 256-entry table.
static inline void build_e4m3_lut(float* out, float global_mul) {
    for (int b = 0; b < 256; ++b) {
        const float s = (b & 0x80) ? -1.0f : 1.0f;
        const int e = (b >> 3) & 0xF;
        const int m = b & 0x7;
        float v;
        if (e == 0)
            v = (float)m / 8.0f * 0.015625f;  // 2^-6
        else if (e == 15 && m == 7)
            v = 0.0f;  // NaN slot; never present in a real weight_scale
        else
            v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
        out[b] = s * v * global_mul;
    }
}
