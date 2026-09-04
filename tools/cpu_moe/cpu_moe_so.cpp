// The CPU MoE expert tier as a LOADABLE LIBRARY — the `.so` `weights/cpu_worker.NativeBackend`
// declares an ABI for and refuses to guess at.
//
// WHY THIS FILE EXISTS SEPARATELY FROM cpu_moe_layer.cpp
//   That file is a BENCH: it mmaps checkpoint shards through a `plan.txt`, holds shapes as
//   compile-time constants, and exits from `main`. The serve needs the same core driven from
//   tensors the ENGINE already owns, at runtime shapes, with no files. Both include the one
//   `moe_core.hpp` / `wload.hpp` pair, so there is exactly one kernel core (KERNEL_CORE_POLICY).
//
// THE POLICY IS `WLoadVnniFp16`, AND THAT IS A CHECKPOINT FACT, NOT A PREFERENCE
//   `sizing._CPU_WLOAD_BY_SCHEME` maps NVFP4 -> `vnni_nvfp4_e4m3_g16`, the checkpoint's own e4m3
//   group scale (0.5625 B/weight). That policy is 10% smaller and ~1800x more accurate — but it
//   reads bytes the ENGINE NO LONGER HAS. `_GroupedNvFp4Experts.post_load` folds the e4m3 block
//   scale and the per-tensor global into ONE fp16 per-group scale and then `del`s the checkpoint
//   copies (layers/moe.py:405-412). What is resident, and therefore what the CPU tier's baked
//   pageable copy holds, is exactly policy E: E2M1 codes + fp16 folded group scale, 0.625 B/weight.
//   Reaching the e4m3 layout needs a re-read of the raw checkpoint at bake time (a repacker), which
//   is what `resolve_weight_plan(cpu_repacked=)` gates and what nothing implements. So this library
//   serves policy E and the plan must claim `layout_fraction = 1.0`. The 10% is real and unclaimed.
//
// TWO ABI DEVIATIONS FROM `NativeBackend`'s DOCSTRING, both forced by the same fact — the engine
// does not hold experts as one contiguous per-expert slab:
//   1. There is no single `const void* table`. The engine holds w13 (E, 2I, H) and w2 (E, H, I) as
//      two stacked tensors per layer, gate and up FUSED over rows. Passing four base pointers plus
//      the per-expert strides derived from (hidden, inter) describes that exactly; flattening it
//      into the bench's slab would mean a third full copy of 27 GiB.
//   2. `lut256` is gone. It is policy A/B state (the e4m3 -> float table with the global folded in);
//      policy E's global is already inside the fp16 scale, so `WLoadVnniFp16::post_scale()` is
//      1.0f and there is no table to pass.
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

// Scales. `_scales_op` is (E, K/16, N) fp16 GROUP-MAJOR — post_load transposes for the HIP op's
// coalesced `[g*N + n]` read (layers/moe.py:410-411). The VNNI core wants tile-major
// `[(rb*NG + g)*16 + r]`. Both index the same element, so this is a gather, not a conversion.
void tile_scales_gm(uint16_t* __restrict dst, const uint16_t* __restrict src, int N, int K) {
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

using WL = WLoadVnniFp16;

struct Ctx {
    int hidden = 0, inter = 0, topk_max = 0, T = 1;
    // per-expert strides, in ELEMENTS of the respective type
    size_t w13_c_stride = 0, w13_s_stride = 0, w2_c_stride = 0, w2_s_stride = 0;

    // --- the job, rewritten per token ---
    const uint8_t* w13c = nullptr;
    const uint16_t* w13s = nullptr;
    const uint8_t* w2c = nullptr;
    const uint16_t* w2s = nullptr;
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
    WL::ctx_t wctx{1.0f};     // policy E: the global is already inside the fp16 scale

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

    void run_slice(int t, int& ls) {
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
            const uint16_t* ws = w13s + (size_t)e * w13_s_stride;
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
            gemv_e2m1_vnni<WL, true>(w2c + (size_t)e * w2_c_stride,
                                     w2s + (size_t)e * w2_s_stride, wctx, hqa, H, I, b0, b1,
                                     y.data(), rw[i]);
        }
        bar.wait(ls);
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

// Pack ONE matrix. `scratch` must be >= max(N*K/2, N*K/16*2) bytes; the caller owns it so the
// pack can run over many matrices without churning allocations.
int cpu_moe_pack_fp16(uint8_t* codes, uint16_t* scales, int N, int K, void* scratch) {
    if (N % VNNI_RB || K % VNNI_GS) return 1;
    const size_t cb = (size_t)N * K / 2, sn = (size_t)N * K / 16;
    memcpy(scratch, codes, cb);
    tile_codes_rt(codes, (const uint8_t*)scratch, N, K);
    memcpy(scratch, scales, sn * sizeof(uint16_t));
    tile_scales_gm(scales, (const uint16_t*)scratch, N, K);
    return 0;
}

void* cpu_moe_open(int hidden, int inter, int topk_max, int threads, const int* cpus, int ncpus) {
    if (hidden % VNNI_RB || inter % VNNI_RB || (2 * inter) % VNNI_RB) return nullptr;
    if (hidden % VNNI_GS || inter % VNNI_GS) return nullptr;
    Ctx* c = new Ctx();
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
int cpu_moe_run(void* hh,
                const uint8_t* w13c, const uint16_t* w13s,
                const uint8_t* w2c, const uint16_t* w2s,
                const float* x, const int32_t* ids, const float* rw,
                int M, int topk, float* out) {
    if (!hh) return 1;
    Ctx* c = (Ctx*)hh;
    if (topk > c->topk_max || topk < 1) return 2;
    c->w13c = w13c; c->w13s = w13s; c->w2c = w2c; c->w2s = w2s;
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
