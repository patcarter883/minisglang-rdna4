// The CPU MoE expert tier as a LOADABLE LIBRARY — the `.so` `weights/cpu_worker.NativeBackend`
// declares an ABI for and refuses to guess at.
//
// WHY THIS FILE EXISTS SEPARATELY FROM cpu_moe_layer.cpp
//   That file is a BENCH: it mmaps checkpoint shards through a `plan.txt`, holds shapes as
//   compile-time constants, and exits from `main`. The serve needs the same core driven from
//   tensors the ENGINE already owns, at runtime shapes, with no files. Both include the one
//   `moe_core.hpp` / `wload.hpp` pair, so there is exactly one kernel core (KERNEL_CORE_POLICY).
//
// TWO POLICIES, CHOSEN AT `cpu_moe_open2` BY WHAT THE ENGINE ACTUALLY HOLDS
//   `sizing._CPU_WLOAD_BY_SCHEME` maps NVFP4 -> `vnni_nvfp4_e4m3_g16`, the checkpoint's own e4m3
//   group scale (0.5625 B/weight): 10% smaller and ~1800x more accurate than the fp16 fold. Until
//   2026-09-05 that named bytes the engine no longer had — `_GroupedNvFp4Experts.post_load` folded
//   the e4m3 block scale and the global into ONE fp16 per-group scale and deleted the originals —
//   so this library was hardcoded to policy E (`vnni_nvfp4_fp16_g16`, 0.625 B/weight). The NVFP4
//   containers now keep BOTH levels (`_scales_op` e4m3 + `_global_op` (E,N) f32), so policy D is
//   reachable and is what an NVFP4 serve uses. Policy E stays compiled in and callable: it is the
//   A/B comparand these files' measurements were written against, and it is still what a
//   fold-at-the-leaf container would present.
//
//   THE POLICY IS NOT INFERRED FROM THE POINTERS. It is passed to `cpu_moe_open2` and then the
//   run entry point REFUSES the mismatched call shape in both directions (globals with policy E,
//   no globals with policy D). Guessing off a pointer being null would turn "the loader forgot the
//   global" into a silent ~4.8e3x error per weight instead of an error code.
//
//   THE SECOND SCALE LEVEL IS A PER-OUTPUT-CHANNEL VECTOR, NOT A SCALAR. `_global_op` is (E, N)
//   f32 because the loader's gate|up merge and expert stack combine differently-scaled matrices
//   (quant/nvfp4.py). `WLoadVnniE4m3::ctx_t` therefore takes `gvec`, and the core applies it as one
//   zmm multiply per 16-row block — exact, because the global does not depend on k.
//
// THREE ABI DEVIATIONS FROM `NativeBackend`'s DOCSTRING, all forced by the engine's data layout:
//   1. There is no single `const void* table`. The engine holds w13 (E, 2I, H) and w2 (E, H, I) as
//      two stacked tensors per layer, gate and up FUSED over rows. Passing base pointers plus
//      the per-expert strides derived from (hidden, inter) describes that exactly; flattening it
//      into the bench's slab would mean a third full copy of 27 GiB.
//   2. `lut256` is gone. It is policy A/B state (the e4m3 -> float table with the global folded in);
//      the VNNI policies decode e4m3 by bit surgery (`e4m3x16_normpos_to_ps`) and carry the global
//      separately, so there is no table to pass.
//   3. The globals are two MORE base pointers, per layer, for the same reason as (1).
//   `out` is still WRITTEN, never accumulated, for the reason the docstring gives.
//
// THE PACK ENTRY POINT IS PART OF THE ABI, NOT A HELPER
//   The VNNI core reads a 16x16 TILED layout (`moe_core.hpp`, `VNNI_TILE_W`); the engine's tensors
//   are row-major. The permutation is byte-count-preserving, so it is done IN PLACE at boot
//   (through a per-matrix scratch buffer) and costs zero resident bytes. Doing it in Python would
//   be a nibble shuffle over ~10^10 weights.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <ctime>
#include <vector>

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include "moe_core.hpp"

namespace {

// ---------------------------------------------------------------------------------------------
// PACK: engine layout -> VNNI tiles. Pure permutation; `dst` and `src` are the same byte count.
//
// Codes. `_w_op` is (E, N, K/8) int32 with code j at bits [4j, 4j+3] (quant/mxfp4.py::
// pack_codes_to_int32). On a little-endian machine byte b of that word therefore holds codes 2b
// (low nibble) and 2b+1 (high) — BYTE-IDENTICAL to the (N, K/2) uint8 `weight_packed` the bench's
// `tile_codes` consumes. So the int32-ness is a viewing convention and nothing here has to undo it.
void tile_codes_rt(uint8_t* __restrict dst, const uint8_t* __restrict src, int N, int K) {
    const int NG = K / VNNI_GS, NRB = N / VNNI_RB;
    const size_t rstride = (size_t)K / 2;
    for (int rb = 0; rb < NRB; ++rb) {
        for (int g = 0; g < NG; ++g) {
            uint8_t* B = dst + ((size_t)rb * NG + g) * VNNI_TILE_W;
            memset(B, 0, VNNI_TILE_W);
            for (int r = 0; r < VNNI_RB; ++r) {
                const uint8_t* srow = src + (size_t)(rb * VNNI_RB + r) * rstride;
                for (int j = 0; j < 16; ++j) {
                    const int k = g * 16 + j;
                    const int code = (k & 1) ? (srow[k >> 1] >> 4) : (srow[k >> 1] & 0xF);
                    int b, hi;
                    if (j < 4)      { b = r * 4 + j;              hi = 0; }
                    else if (j < 8) { b = r * 4 + (j - 4);        hi = 1; }
                    else if (j < 12){ b = 64 + r * 4 + (j - 8);   hi = 0; }
                    else            { b = 64 + r * 4 + (j - 12);  hi = 1; }
                    B[b] |= (uint8_t)(code << (hi ? 4 : 0));
                }
            }
        }
    }
}

// Scales. `_scales_op` is (E, K/16, N) GROUP-MAJOR — post_load transposes for the HIP op's
// coalesced `[g*N + n]` read (layers/moe.py). The VNNI core wants tile-major
// `[(rb*NG + g)*16 + r]`. Both index the same element, so this is a gather, not a conversion —
// which is why ONE template serves both scale widths: the permutation is identical and only
// sizeof(T) differs. A second copy of this loop for uint8 is how the two layouts would drift.
//
// ROW ORDER IS PRESERVED (tile rb holds source rows rb*16 .. rb*16+15, in order), and that is what
// lets the per-output-channel global stay an UNPERMUTED (N,) vector indexed `gvec[rb*16 + r]`.
template <class T>
void tile_scales_gm(T* __restrict dst, const T* __restrict src, int N, int K) {
    const int NG = K / VNNI_GS, NRB = N / VNNI_RB;
    for (int rb = 0; rb < NRB; ++rb)
        for (int g = 0; g < NG; ++g)
            for (int r = 0; r < VNNI_RB; ++r)
                dst[((size_t)rb * NG + g) * VNNI_RB + r] =
                    src[(size_t)g * N + (rb * VNNI_RB + r)];  // group-major source
}

// ---------------------------------------------------------------------------------------------
// The runtime-shaped Runner. Same phase structure and same barrier as the bench's TILED path; the
// only differences are that every dimension is a member instead of a constant, and that w13 is one
// FUSED (2I, H) matrix so phase A is a single row-block split over 2I rather than two over I.
struct Barrier {
    std::atomic<int> count{0};
    std::atomic<int> sense{0};
    int nthreads = 1;
    void wait(int& local_sense) {
        local_sense ^= 1;
        if (count.fetch_add(1, std::memory_order_acq_rel) == nthreads - 1) {
            count.store(0, std::memory_order_relaxed);
            sense.store(local_sense, std::memory_order_release);
        } else {
            while (sense.load(std::memory_order_acquire) != local_sense) _mm_pause();
        }
    }
};

// The WLoad policy, as an ABI-stable integer. The NAMES are what `weights/sizing.py`'s
// `_CPU_WLOAD_BY_SCHEME` and `cpu_native.py` key on; `cpu_moe_policies()` publishes them so a
// Python caller can ASSERT that this build supports what the checkpoint needs instead of
// discovering it by producing wrong numbers.
enum : int { POL_FP16 = 0, POL_E4M3 = 1, POL_N = 2 };
static const char* const kPolicyNames[POL_N] = {WLoadVnniFp16::NAME, WLoadVnniE4m3::NAME};

// The ctx the core wants, built from the per-expert global slice. `if constexpr` rather than a
// common ctx shape: `WLoadVnniFp16::ctx_t` deliberately has NO `gvec` field, so handing a
// per-channel global to the fp16 policy fails to compile rather than being ignored.
template <class WL>
static inline typename WL::ctx_t wctx_for(const float* gvec) {
    typename WL::ctx_t c{};
    c.gmul = 1.0f;
    if constexpr (WL::POST_PER_ROW) c.gvec = gvec;
    return c;
}

struct Ctx {
    int policy = POL_FP16;
    int hidden = 0, inter = 0, topk_max = 0, T = 1;
    // per-expert strides, in ELEMENTS of the respective type. The SCALE stride is the same NUMBER
    // for both policies (one scale per 16 weights); only the element WIDTH differs, and that is
    // carried by the pointer type inside run_slice_t.
    size_t w13_c_stride = 0, w13_s_stride = 0, w2_c_stride = 0, w2_s_stride = 0;
    // per-expert stride of the (E, N) f32 per-output-channel global: N, i.e. 2*inter and hidden.
    size_t w13_g_stride = 0, w2_g_stride = 0;

    // --- the job, rewritten per token ---
    const uint8_t* w13c = nullptr;
    const void* w13s = nullptr;      // scale_t depends on the policy; typed in run_slice_t
    const float* w13g = nullptr;     // (E, 2I) f32, or null under POL_FP16
    const uint8_t* w2c = nullptr;
    const void* w2s = nullptr;
    const float* w2g = nullptr;      // (E, H) f32, or null under POL_FP16
    const int32_t* sel = nullptr;
    const float* rw = nullptr;
    int topk = 0;
    QAct xq{};
    std::vector<int8_t> xq_q;
    std::vector<float> xq_sc;
    std::vector<int32_t> xq_sum;
    std::vector<float> gu;    // topk NOT needed: 2I scratch, reused per expert
    std::vector<float> h;     // topk * I
    std::vector<int8_t> hq;   // topk * I
    std::vector<float> hsc;   // topk * I/16
    std::vector<int32_t> hsum;
    std::vector<float> y;     // H

    Barrier bar;
    std::vector<std::thread> th;
    std::atomic<int> gen{0}, quit{0}, done{0};
    int main_sense = 0;
    std::atomic<long long> calls{0};   // THE COUNTER (the engaged() ledger is a set; this is not)
    std::atomic<long long> tokens{0};
    std::vector<int> cpus;

    static inline void split(int n, int t, int T, int* r0, int* r1) {
        const int q = n / T, rem = n % T;
        *r0 = t * q + (t < rem ? t : rem);
        *r1 = *r0 + q + (t < rem ? 1 : 0);
    }

    // ONE body, instantiated per policy. The policy-dependent parts are exactly two: the scale
    // element type, and whether a per-output-channel global rides along. Everything about the
    // phase structure, the barriers and the row-range split is shared, which is the whole point of
    // WLoad being a policy rather than a second kernel.
    template <class WL>
    void run_slice_t(int t, int& ls) {
        using S = typename WL::scale_t;
        const S* __restrict w13s_ = (const S*)w13s;
        const S* __restrict w2s_ = (const S*)w2s;
        const int I = inter, H = hidden, N13 = 2 * I, rbI = I / VNNI_RB;
        // ---- phase A: the FUSED w13 -> gate|up, then SiLU.
        //
        // The split is over the GATE half's row blocks, and each thread then takes the MATCHING
        // row blocks of the up half (`+ rbI`). That is what keeps the whole phase barrier-free:
        // thread t produces gu[a0*16 .. a1*16) and gu[I + a0*16 .. I + a1*16), which is exactly
        // the pair its own SiLU rows consume. Splitting the fused 2I range instead would let a
        // thread own gate rows whose up rows belong to another thread, costing two barriers per
        // expert (22 per token at top-10) in a loop whose whole budget is ~0.5 ms.
        //
        // Legal because the gate/up boundary is at row I and I is a multiple of VNNI_RB, so no
        // tile ever straddles it.
        int a0, a1;
        split(rbI, t, T, &a0, &a1);
        for (int i = 0; i < topk; ++i) {
            const int e = sel[i];
            const uint8_t* wc = w13c + (size_t)e * w13_c_stride;
            const S* ws = w13s_ + (size_t)e * w13_s_stride;
            // The global is indexed by the FUSED matrix's output channel, so ONE (2I,) slice
            // covers both the gate half and the up half — the same reason `torch.cat(dim=0)`
            // carries it through the merge with no special case.
            const auto wctx = wctx_for<WL>(w13g ? w13g + (size_t)e * w13_g_stride : nullptr);
            gemv_e2m1_vnni<WL, false>(wc, ws, wctx, xq, N13, H, a0, a1, gu.data(), 0.0f);
            gemv_e2m1_vnni<WL, false>(wc, ws, wctx, xq, N13, H, a0 + rbI, a1 + rbI, gu.data(),
                                      0.0f);
            float* hh = h.data() + (size_t)i * I;
            for (int r = a0 * VNNI_RB; r < a1 * VNNI_RB; ++r) hh[r] = silu_mul(gu[r], gu[I + r]);
        }
        bar.wait(ls);
        // ---- re-quantize h over the flat (expert, group) list
        int q0, q1;
        split(topk * (I / 16), t, T, &q0, &q1);
        quantize_act_g16_range(h.data(), q0, q1, hq.data(), hsc.data(), hsum.data());
        bar.wait(ls);
        // ---- phase B: down, partitioned by HIDDEN row block so each thread owns disjoint y
        int b0, b1;
        split(H / VNNI_RB, t, T, &b0, &b1);
        for (int r = b0 * VNNI_RB; r < b1 * VNNI_RB; ++r) y[r] = 0.0f;
        for (int i = 0; i < topk; ++i) {
            const int e = sel[i];
            const QAct hqa{hq.data() + (size_t)i * I, hsc.data() + (size_t)i * (I / 16),
                           hsum.data() + (size_t)i * (I / 16)};
            const auto wctx = wctx_for<WL>(w2g ? w2g + (size_t)e * w2_g_stride : nullptr);
            gemv_e2m1_vnni<WL, true>(w2c + (size_t)e * w2_c_stride,
                                     w2s_ + (size_t)e * w2_s_stride, wctx, hqa, H, I, b0, b1,
                                     y.data(), rw[i]);
        }
        bar.wait(ls);
    }

    // The ONE dispatch, on a value fixed at open() — not on a pointer being null, and not per
    // tile. Both arms are fully inlined instantiations of the body above.
    void run_slice(int t, int& ls) {
        if (policy == POL_E4M3) run_slice_t<WLoadVnniE4m3>(t, ls);
        else run_slice_t<WLoadVnniFp16>(t, ls);
    }

    void pin(int t) {
        if ((int)cpus.size() <= t) return;
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(cpus[t], &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }

    void worker(int t) {
        pin(t);
        int mygen = 0, ls = 0;
        for (;;) {
            // SPIN -> YIELD -> SLEEP, not a pure spin.
            //
            // The bench spins because it runs layers back to back and owns the box. A serve does
            // not: these threads exist for the whole process life, and a pure `_mm_pause` loop
            // would hold 100% of `threads-1` physical cores through the ~10-minute checkpoint
            // load and through every idle moment between requests — competing with the very
            // engine this tier is supposed to leave core 0 to, and reproducing §1.5's cliff (a
            // descheduled spinner costs a whole timeslice) instead of avoiding it.
            //
            // The three tiers match the three timescales: the ~us gaps BETWEEN LAYERS inside a
            // decode step (spin — a condvar round trip would be most of the budget), the ~ms gaps
            // between steps (yield), and the unbounded idle of a loaded-but-unqueried serve
            // (sleep). The sleep is 200 us, so the worst case it adds to a layer is 200 us against
            // a ~500 us layer, and only on the FIRST layer after an idle gap.
            int spins = 0;
            while (gen.load(std::memory_order_acquire) == mygen) {
                if (quit.load(std::memory_order_acquire)) return;
                if (spins < 4000) { _mm_pause(); ++spins; }
                else if (spins < 4200) { std::this_thread::yield(); ++spins; }
                else {
                    struct timespec ts{0, 200000};
                    nanosleep(&ts, nullptr);
                }
            }
            mygen = gen.load(std::memory_order_acquire);
            if (quit.load(std::memory_order_acquire)) return;
            run_slice(t, ls);
            done.fetch_add(1, std::memory_order_release);
        }
    }

    void run_one() {
        done.store(0, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
        int ls = main_sense;
        run_slice(0, ls);
        main_sense = ls;
        while (done.load(std::memory_order_acquire) != T - 1) _mm_pause();
    }
};

}  // namespace

extern "C" {

// The policy names this BUILD supports, comma-separated. The Python side asserts membership
// before packing anything: an .so from before this change exports no such symbol and no
// `cpu_moe_pack_e4m3`, so an e4m3 checkpoint meets a refusal instead of the fp16 packer reading
// half a slab. A POSITIVE check (is this policy advertised?) rather than a negative one (is the
// dtype the one I know is broken?), so a future scale format is refused by default.
const char* cpu_moe_policies(void) { return "vnni_nvfp4_fp16_g16,vnni_nvfp4_e4m3_g16"; }

// Pack ONE matrix. `scratch` must be >= max(N*K/2, N*K/16*sizeof(scale_t)) bytes; the caller owns
// it so the pack can run over many matrices without churning allocations.
//
// BOTH ENTRY POINTS ARE THE SAME PERMUTATION. They are separate symbols, not one with a dtype
// argument, because the SYMBOL is the ABI: a Python caller that resolves `cpu_moe_pack_e4m3`
// cannot then be handed a build that does not have it.
int cpu_moe_pack_fp16(uint8_t* codes, uint16_t* scales, int N, int K, void* scratch) {
    if (N % VNNI_RB || K % VNNI_GS) return 1;
    const size_t cb = (size_t)N * K / 2, sn = (size_t)N * K / 16;
    memcpy(scratch, codes, cb);
    tile_codes_rt(codes, (const uint8_t*)scratch, N, K);
    memcpy(scratch, scales, sn * sizeof(uint16_t));
    tile_scales_gm<uint16_t>(scales, (const uint16_t*)scratch, N, K);
    return 0;
}

// Same, for the checkpoint's own 1-byte e4m3 block scale.
//
// It CHECKS THE PRECONDITION OF THE FAST DECODE on these exact bytes, and reports the offender.
// `e4m3x16_normpos_to_ps` is specialised to positive-normal e4m3 (0x08..0x7E) — three ops instead
// of nineteen, which is what makes the smaller layout also the faster one — and a byte outside
// that range decodes to a plausible wrong number, not to a fault. The census that established the
// range covered layers 0-3 of one checkpoint; this covers every byte actually served.
// Returns 0, or 1 for a shape refusal, or -(bad_byte)-1 (so, negative) for a normpos violation.
int cpu_moe_pack_e4m3(uint8_t* codes, uint8_t* scales, int N, int K, void* scratch) {
    if (N % VNNI_RB || K % VNNI_GS) return 1;
    const size_t cb = (size_t)N * K / 2, sn = (size_t)N * K / 16;
    int bad = 0;
    if (!e4m3_scales_normpos(scales, sn, &bad)) return -bad - 1;
    memcpy(scratch, codes, cb);
    tile_codes_rt(codes, (const uint8_t*)scratch, N, K);
    memcpy(scratch, scales, sn);
    tile_scales_gm<uint8_t>(scales, (const uint8_t*)scratch, N, K);
    return 0;
}

void* cpu_moe_open2(int hidden, int inter, int topk_max, int threads, const int* cpus, int ncpus,
                    int policy) {
    if (hidden % VNNI_RB || inter % VNNI_RB || (2 * inter) % VNNI_RB) return nullptr;
    if (hidden % VNNI_GS || inter % VNNI_GS) return nullptr;
    if (policy < 0 || policy >= POL_N) return nullptr;
    Ctx* c = new Ctx();
    c->policy = policy;
    c->w13_g_stride = (size_t)2 * inter;
    c->w2_g_stride = (size_t)hidden;
    c->hidden = hidden;
    c->inter = inter;
    c->topk_max = topk_max;
    c->T = threads < 1 ? 1 : threads;
    c->w13_c_stride = (size_t)2 * inter * hidden / 2;
    c->w13_s_stride = (size_t)2 * inter * hidden / 16;
    c->w2_c_stride = (size_t)hidden * inter / 2;
    c->w2_s_stride = (size_t)hidden * inter / 16;
    c->xq_q.assign(hidden, 0);
    c->xq_sc.assign(hidden / 16, 0.f);
    c->xq_sum.assign(hidden / 16, 0);
    c->gu.assign((size_t)2 * inter, 0.f);
    c->h.assign((size_t)topk_max * inter, 0.f);
    c->hq.assign((size_t)topk_max * inter, 0);
    c->hsc.assign((size_t)topk_max * inter / 16, 0.f);
    c->hsum.assign((size_t)topk_max * inter / 16, 0);
    c->y.assign(hidden, 0.f);
    c->xq = QAct{c->xq_q.data(), c->xq_sc.data(), c->xq_sum.data()};
    for (int i = 0; i < ncpus; ++i) c->cpus.push_back(cpus[i]);
    c->bar.nthreads = c->T;
    for (int t = 1; t < c->T; ++t) c->th.emplace_back([c, t] { c->worker(t); });
    return c;
}

// The original 6-argument spelling, pinned to policy E. Kept so an older caller keeps working.
void* cpu_moe_open(int hidden, int inter, int topk_max, int threads, const int* cpus, int ncpus) {
    return cpu_moe_open2(hidden, inter, topk_max, threads, cpus, ncpus, POL_FP16);
}

// What this handle was opened for. Lets a caller report the policy it is ACTUALLY serving
// through rather than the one it believes it asked for.
const char* cpu_moe_policy_name(void* h) {
    if (!h) return "";
    const int p = ((Ctx*)h)->policy;
    return (p >= 0 && p < POL_N) ? kPolicyNames[p] : "";
}

void cpu_moe_close(void* h) {
    if (!h) return;
    Ctx* c = (Ctx*)h;
    c->quit.store(1, std::memory_order_release);
    c->gen.fetch_add(1, std::memory_order_acq_rel);
    for (auto& t : c->th) if (t.joinable()) t.join();
    delete c;
}

// M tokens, one layer. `out` is (M, hidden) fp32, WRITTEN.
//
// The token loop is SERIAL and that is a real limit, stated rather than hidden: the core is a
// GEMV, so M tokens re-read the expert weights M times. It is correct at any M and only economic
// at M=1. A prefill chunk pays M x the decode cost.
int cpu_moe_run2(void* hh,
                 const uint8_t* w13c, const void* w13s, const float* w13g,
                 const uint8_t* w2c, const void* w2s, const float* w2g,
                 const float* x, const int32_t* ids, const float* rw,
                 int M, int topk, float* out) {
    if (!hh) return 1;
    Ctx* c = (Ctx*)hh;
    if (topk > c->topk_max || topk < 1) return 2;
    // THE CALL SHAPE MUST MATCH THE POLICY, IN BOTH DIRECTIONS, AND BOTH ARE SILENT IF ALLOWED:
    //   e4m3 without globals -> every weight short by its per-channel multiplier (~4.8e3x here),
    //   fp16 with globals    -> a global applied twice, once folded and once as a vector.
    // Neither faults; both produce finite, fluent, wrong output. So they are error codes.
    const bool need_g = (c->policy == POL_E4M3);
    if (need_g && (!w13g || !w2g)) return 3;
    if (!need_g && (w13g || w2g)) return 4;
    c->w13c = w13c; c->w13s = w13s; c->w13g = w13g;
    c->w2c = w2c; c->w2s = w2s; c->w2g = w2g;
    c->topk = topk;
    for (int m = 0; m < M; ++m) {
        const float* xm = x + (size_t)m * c->hidden;
        quantize_act_g16(xm, c->hidden, c->xq_q.data(), c->xq_sc.data(), c->xq_sum.data());
        c->sel = ids + (size_t)m * topk;
        c->rw = rw + (size_t)m * topk;
        if (c->T > 1) c->run_one();
        else { int ls = c->main_sense; c->run_slice(0, ls); c->main_sense = ls; }
        memcpy(out + (size_t)m * c->hidden, c->y.data(), (size_t)c->hidden * sizeof(float));
    }
    c->calls.fetch_add(1, std::memory_order_relaxed);
    c->tokens.fetch_add(M, std::memory_order_relaxed);
    return 0;
}

// The original spelling, for a policy-E handle. Passes no globals, which `cpu_moe_run2` then
// requires to be consistent with the handle's policy.
int cpu_moe_run(void* hh,
                const uint8_t* w13c, const uint16_t* w13s,
                const uint8_t* w2c, const uint16_t* w2s,
                const float* x, const int32_t* ids, const float* rw,
                int M, int topk, float* out) {
    return cpu_moe_run2(hh, w13c, (const void*)w13s, nullptr, w2c, (const void*)w2s, nullptr,
                        x, ids, rw, M, topk, out);
}

// Pin the SUBMITTING thread — the one that runs slice 0. It must be called ON that thread, which
// is `CpuMoEWorker`'s dispatcher, NOT the boot thread that called `cpu_moe_open`. Pinning inside
// `open()` would have pinned the engine's main Python thread to a CPU-tier core for the life of
// the process, which is the exact opposite of "leave core 0 to the engine".
void cpu_moe_pin_self(void* h) {
    if (!h) return;
    ((Ctx*)h)->pin(0);
}

// THE COUNTER. `engaged()` is a SET and saturates at one; these are monotone and per-call, so a
// tier that bound 21 layers and executed 3 of them is distinguishable from one that executed all 21.
long long cpu_moe_calls(void* h) { return h ? ((Ctx*)h)->calls.load() : -1; }
long long cpu_moe_tokens(void* h) { return h ? ((Ctx*)h)->tokens.load() : -1; }

}  // extern "C"
