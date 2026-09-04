// Standalone CPU MoE expert path for Qwen3.8-Flash-Next-NVFP4, one layer, decode (M=1).
//
//   build : g++ -O3 -march=znver4 -std=c++17 -o cpu_moe_layer cpu_moe_layer.cpp -lpthread
//   run   : ./cpu_moe_layer --plan <fixture>/plan.txt [--plan ...] --ref <fixture>/ref
//                           [--mode verify|bench|selftest] [--threads N] [--iters N]
//                           [--policy e4m3|fp16] [--json]
//
// Weights are gathered out of the real safetensors shards into a packed resident table, one
// 2.7648 MB slab per expert, laid out so each projection is one contiguous ~900 KB run.
#include "moe_core.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

static constexpr int HID = 2560;   // hidden_size
static constexpr int INT_ = 640;   // moe_intermediate_size
static constexpr int NGU = 640;    // gate/up rows
static constexpr int KGU = 2560;   // gate/up cols
static constexpr int NDN = 2560;   // down rows
static constexpr int KDN = 640;    // down cols

// ---- resident slab geometry, policy A (e4m3 scales, checkpoint-native) ----------------------
static constexpr size_t GU_C = (size_t)NGU * KGU / 2;   // 819200
static constexpr size_t GU_S8 = (size_t)NGU * KGU / 16; // 102400
static constexpr size_t DN_C = (size_t)NDN * KDN / 2;   // 819200
static constexpr size_t DN_S8 = (size_t)NDN * KDN / 16; // 102400
static constexpr size_t SLAB8 = 2 * (GU_C + GU_S8) + DN_C + DN_S8;    // 2'764'800
static constexpr size_t SLAB16 = 2 * (GU_C + 2 * GU_S8) + DN_C + 2 * DN_S8;  // 3'072'000

struct PlanRec {
    int shard;
    size_t w_off, w_len, s_off, s_len;
    int n, k;
    float global;
};

struct Plan {
    std::vector<std::string> shards;
    int n_experts = 0;
    std::vector<PlanRec> rec;  // 3 per expert: gate, up, down
};

static void die(const char* m) {
    fprintf(stderr, "FATAL: %s (%s)\n", m, strerror(errno));
    exit(1);
}

static Plan read_plan(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) die(path);
    Plan p;
    char line[4096];
    if (!fgets(line, sizeof line, f)) die("plan: empty");
    int ns = 0;
    if (sscanf(line, "SHARDS %d", &ns) != 1) die("plan: SHARDS");
    for (int i = 0; i < ns; ++i) {
        if (!fgets(line, sizeof line, f)) die("plan: shard path");
        line[strcspn(line, "\n")] = 0;
        p.shards.emplace_back(line);
    }
    if (!fgets(line, sizeof line, f)) die("plan: EXPERTS");
    if (sscanf(line, "EXPERTS %d", &p.n_experts) != 1) die("plan: EXPERTS");
    p.rec.resize((size_t)p.n_experts * 3);
    while (fgets(line, sizeof line, f)) {
        int e, pi, sh, n, k;
        size_t wo, wl, so, sl;
        double g;
        if (sscanf(line, "E %d %d %d %zu %zu %zu %zu %d %d %lf", &e, &pi, &sh, &wo, &wl, &so, &sl,
                   &n, &k, &g) != 10)
            continue;
        p.rec[(size_t)e * 3 + pi] = PlanRec{sh, wo, wl, so, sl, n, k, (float)g};
    }
    fclose(f);
    return p;
}

static const uint8_t* map_file(const std::string& path, size_t* len) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) die(path.c_str());
    struct stat st;
    if (fstat(fd, &st)) die("fstat");
    void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) die("mmap shard");
    close(fd);
    *len = st.st_size;
    return (const uint8_t*)m;
}

static void* big_alloc(size_t bytes) {
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) die("mmap table");
    madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

// -------------------------------------------------------------------------------------------
// The resident table.  One instantiation per WLoad policy; the only thing that differs is how
// the scale bytes are written at build time.
template <class WL>
struct Table {
    uint8_t* base = nullptr;
    size_t slab = 0;
    int n_experts = 0;
    std::vector<float> luts;  // 256 per (expert, proj)
    std::vector<ExpertSlab<WL>> ex;
    size_t off_gc, off_gs, off_uc, off_us, off_dc, off_ds;

    void layout() {
        const size_t SS = sizeof(typename WL::scale_t);
        off_gc = 0;
        off_gs = off_gc + GU_C;
        off_uc = off_gs + GU_S8 * SS;
        off_us = off_uc + GU_C;
        off_dc = off_us + GU_S8 * SS;
        off_ds = off_dc + DN_C;
        slab = off_ds + DN_S8 * SS;
    }
};

// ---- Emit: the LOADER half of a WLoad policy.  Turns the raw checkpoint bytes for ONE matrix
// into that policy's resident form.  Row-major policies memcpy the codes; VNNI policies repack
// them into 16x16 tiles.  Nothing else in the harness knows a layout exists.
static void tile_codes(uint8_t* __restrict dst, const uint8_t* __restrict src, int N, int K) {
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
                    if (j < 4)       { b = r * 4 + j;             hi = 0; }
                    else if (j < 8)  { b = r * 4 + (j - 4);       hi = 1; }
                    else if (j < 12) { b = 64 + r * 4 + (j - 8);  hi = 0; }
                    else             { b = 64 + r * 4 + (j - 12); hi = 1; }
                    B[b] |= (uint8_t)(code << (hi ? 4 : 0));
                }
            }
        }
    }
}

template <class WL>
struct Emit;

// Policy A: verbatim e4m3 bytes; the per-tensor global goes into the 256-entry LUT.
template <>
struct Emit<WLoadNvfp4E4m3> {
    static void go(uint8_t* dc, uint8_t* ds, const uint8_t* wsrc, const uint8_t* ssrc, int N,
                   int K, float gmul, float* lut, WLoadNvfp4E4m3::ctx_t& ctx) {
        memcpy(dc, wsrc, (size_t)N * K / 2);
        memcpy(ds, ssrc, (size_t)N * K / 16);
        build_e4m3_lut(lut, gmul);
        ctx.lut256 = lut;
    }
};
// Policy B: fold e4m3/global into an fp16 per-group scale -- byte-identical to what the HIP
// W4A8 path holds in VRAM (quant/nvfp4.py::fold_nvfp4_scale, at fp16).
template <>
struct Emit<WLoadNvfp4Fp16> {
    static void go(uint8_t* dc, uint8_t* ds, const uint8_t* wsrc, const uint8_t* ssrc, int N,
                   int K, float gmul, float* lut, WLoadNvfp4Fp16::ctx_t& ctx) {
        memcpy(dc, wsrc, (size_t)N * K / 2);
        float f[256];
        build_e4m3_lut(f, gmul);
        uint16_t* d = (uint16_t*)ds;
        const size_t n = (size_t)N * K / 16;
        for (size_t i = 0; i < n; ++i)
            d[i] = (uint16_t)_cvtss_sh(f[ssrc[i]], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        for (int i = 0; i < 256; ++i) lut[i] = 0.0f;
        ctx.lut256 = lut;
    }
};
// Policy D: the SAME checkpoint bytes, VNNI-tiled.  Scale bytes stay e4m3 (0.5625 B/weight) and
// the global stays a runtime multiplier, so DDR traffic is byte-for-byte identical to policy A.
template <>
struct Emit<WLoadVnniE4m3> {
    static void go(uint8_t* dc, uint8_t* ds, const uint8_t* wsrc, const uint8_t* ssrc, int N,
                   int K, float gmul, float*, WLoadVnniE4m3::ctx_t& ctx) {
        // The in-loop scale decode is specialised to positive-normal e4m3 (wload.hpp).  Prove
        // the precondition on THESE bytes rather than trusting the census that established it.
        int bad = 0;
        if (!e4m3_scales_normpos(ssrc, (size_t)N * K / 16, &bad)) {
            fprintf(stderr,
                    "FATAL: weight_scale byte 0x%02X is outside the positive-normal e4m3 range "
                    "0x08..0x7E that e4m3x16_normpos_to_ps assumes.\n"
                    "This checkpoint needs the general decode (build_e4m3_lut); do not remove "
                    "this check to make it load.\n",
                    bad);
            exit(1);
        }
        tile_codes(dc, wsrc, N, K);
        const int NG = K / VNNI_GS;
        for (int rb = 0; rb < N / VNNI_RB; ++rb)
            for (int g = 0; g < NG; ++g)
                for (int r = 0; r < VNNI_RB; ++r)
                    ds[((size_t)rb * NG + g) * VNNI_RB + r] =
                        ssrc[(size_t)(rb * VNNI_RB + r) * NG + g];
        ctx.gmul = gmul;
    }
};
// Policy E: VNNI-tiled with the global pre-folded into an fp16 group scale -- the 0.625 B/weight
// layout the ceiling probe measured.  Present only to price the scale width against D.
template <>
struct Emit<WLoadVnniFp16> {
    static void go(uint8_t* dc, uint8_t* ds, const uint8_t* wsrc, const uint8_t* ssrc, int N,
                   int K, float gmul, float*, WLoadVnniFp16::ctx_t& ctx) {
        tile_codes(dc, wsrc, N, K);
        float f[256];
        build_e4m3_lut(f, gmul);
        uint16_t* d = (uint16_t*)ds;
        const int NG = K / VNNI_GS;
        for (int rb = 0; rb < N / VNNI_RB; ++rb)
            for (int g = 0; g < NG; ++g)
                for (int r = 0; r < VNNI_RB; ++r)
                    d[((size_t)rb * NG + g) * VNNI_RB + r] = (uint16_t)_cvtss_sh(
                        f[ssrc[(size_t)(rb * VNNI_RB + r) * NG + g]],
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        ctx.gmul = 1.0f;
    }
};

template <class WL>
static Table<WL> build_table(const std::vector<Plan>& plans,
                             const std::vector<std::vector<const uint8_t*>>& maps, int cap) {
    Table<WL> t;
    t.layout();
    int total = 0;
    for (auto& p : plans) total += p.n_experts;
    if (cap > 0 && cap < total) total = cap;
    t.n_experts = total;
    t.base = (uint8_t*)big_alloc((size_t)total * t.slab);
    t.luts.assign((size_t)total * 3 * 256, 0.0f);
    t.ex.resize(total);

    // (plan, expert-within-plan) for each resident slot, so the fill can be handed to a pool.
    std::vector<std::pair<int, int>> src_of(total);
    {
        int e = 0;
        for (size_t pi = 0; pi < plans.size() && e < total; ++pi)
            for (int i = 0; i < plans[pi].n_experts && e < total; ++i, ++e)
                src_of[e] = {(int)pi, i};
    }

    // The VNNI repack is a scalar nibble shuffle over every weight in the table (~10^10 for a
    // 5.6 GiB residency).  Single-threaded that is minutes of setup before a millisecond
    // measurement, so the FILL is threaded.  This is load-time work and is not timed anywhere.
    const int BT = std::max(1, (int)std::thread::hardware_concurrency());
    std::atomic<int> next{0};
    auto fill = [&] {
        for (;;) {
            const int e = next.fetch_add(1, std::memory_order_relaxed);
            if (e >= total) return;
            const Plan& p = plans[src_of[e].first];
            const int i = src_of[e].second;
            uint8_t* s = t.base + (size_t)e * t.slab;
            const size_t co[3] = {t.off_gc, t.off_uc, t.off_dc};
            const size_t so[3] = {t.off_gs, t.off_us, t.off_ds};
            typename WL::ctx_t* cx[3] = {&t.ex[e].gate_ctx, &t.ex[e].up_ctx, &t.ex[e].down_ctx};
            for (int pr = 0; pr < 3; ++pr) {
                const PlanRec& r = p.rec[(size_t)i * 3 + pr];
                const uint8_t* src = maps[src_of[e].first][r.shard];
                // r.global is this checkpoint's weight_scale_2, a MULTIPLIER -- see wload.hpp
                // and make_fixture.py for the four pieces of evidence that pin the sign.
                Emit<WL>::go(s + co[pr], s + so[pr], src + r.w_off, src + r.s_off, r.n, r.k,
                             r.global, &t.luts[((size_t)e * 3 + pr) * 256], *cx[pr]);
            }
            using ST = typename WL::scale_t;
            t.ex[e].gate_c = s + t.off_gc;
            t.ex[e].gate_s = (const ST*)(s + t.off_gs);
            t.ex[e].up_c = s + t.off_uc;
            t.ex[e].up_s = (const ST*)(s + t.off_us);
            t.ex[e].down_c = s + t.off_dc;
            t.ex[e].down_s = (const ST*)(s + t.off_ds);
        }
    };
    std::vector<std::thread> bt;
    for (int i = 1; i < BT; ++i) bt.emplace_back(fill);
    fill();
    for (auto& x : bt) x.join();
    return t;
}

// -------------------------------------------------------------------------------------------
// Thread pool: persistent workers, sense-reversing spin barrier.
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

struct Job {
    const int* sel = nullptr;
    const float* rw = nullptr;
    int topk = 0;
    // fp32-activation core
    const float* xe = nullptr;
    const float* xo = nullptr;
    // int8-activation core: the layer input, quantized once per token
    QAct xq{};
    float* h = nullptr;   // topk * INT_   (SiLU(gate)*up, fp32, both cores)
    // int8-activation core: h re-quantized between the two phases.  SHARED, filled
    // cooperatively between barriers -- every thread's down-rows read every expert's h.
    int8_t* hq = nullptr;      // topk * INT_
    float* hsc = nullptr;      // topk * INT_/16
    int32_t* hsum = nullptr;   // topk * INT_/16
    float* y = nullptr;   // HID
};

template <class WL>
struct Runner {
    const Table<WL>* tab;
    Job job;
    Barrier bar;
    std::vector<std::thread> th;
    std::atomic<int> gen{0};
    std::atomic<int> quit{0};
    std::atomic<int> done{0};
    int T = 1;

    static void pin(int cpu) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(cpu, &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }

    void worker(int t) {
        pin(t);
        std::vector<float> he((size_t)32 * INT_ / 2), ho((size_t)32 * INT_ / 2);
        std::vector<float> gbuf(INT_), ubuf(INT_);
        int mygen = 0, ls = 0;
        for (;;) {
            while (gen.load(std::memory_order_acquire) == mygen) {
                if (quit.load(std::memory_order_acquire)) return;
                _mm_pause();
            }
            mygen = gen.load(std::memory_order_acquire);
            // stop() sets quit THEN bumps gen, so a worker woken by that bump must re-check
            // here.  Without it the worker ran one more slice against a dead Job and parked in
            // a barrier main had already left -- a hang, not a crash.
            if (quit.load(std::memory_order_acquire)) return;
            run_slice(t, he.data(), ho.data(), gbuf.data(), ubuf.data(), ls);
            done.fetch_add(1, std::memory_order_release);
        }
    }

    // rows [r0,r1) for thread t out of n
    static inline void split(int n, int t, int T, int* r0, int* r1) {
        const int q = n / T, rem = n % T;
        *r0 = t * q + (t < rem ? t : rem);
        *r1 = *r0 + q + (t < rem ? 1 : 0);
    }

    void run_slice(int t, float* he, float* ho, float* gbuf, float* ubuf, int& ls) {
        const Job& j = job;
        if constexpr (WL::TILED) {
            // ---- int8-activation path -----------------------------------------------------
            // Identical phase structure to the fp32 path; the row partition is in row BLOCKS
            // of VNNI_RB because the tile is the unit of the layout, and there is one extra
            // barrier because h must be re-quantized before the down projection.
            int a0, a1;
            split(INT_ / VNNI_RB, t, T, &a0, &a1);
            for (int i = 0; i < j.topk; ++i) {
                const ExpertSlab<WL>& s = tab->ex[j.sel[i]];
                gemv_e2m1_vnni<WL, false>(s.gate_c, s.gate_s, s.gate_ctx, j.xq, NGU, KGU, a0, a1,
                                          gbuf, 0.0f);
                gemv_e2m1_vnni<WL, false>(s.up_c, s.up_s, s.up_ctx, j.xq, NGU, KGU, a0, a1, ubuf,
                                          0.0f);
                float* hh = j.h + (size_t)i * INT_;
                for (int r = a0 * VNNI_RB; r < a1 * VNNI_RB; ++r) hh[r] = silu_mul(gbuf[r], ubuf[r]);
            }
            bar.wait(ls);
            // ---- re-quantize h, partitioned over the flat (expert, group) list -------------
            int q0, q1;
            split(j.topk * (INT_ / 16), t, T, &q0, &q1);
            quantize_act_g16_range(j.h, q0, q1, j.hq, j.hsc, j.hsum);
            bar.wait(ls);
            // ---- phase B: down, partitioned by HIDDEN row block, disjoint y ----------------
            int b0, b1;
            split(HID / VNNI_RB, t, T, &b0, &b1);
            for (int r = b0 * VNNI_RB; r < b1 * VNNI_RB; ++r) j.y[r] = 0.0f;
            for (int i = 0; i < j.topk; ++i) {
                const ExpertSlab<WL>& s = tab->ex[j.sel[i]];
                const QAct hq{j.hq + (size_t)i * INT_, j.hsc + (size_t)i * (INT_ / 16),
                              j.hsum + (size_t)i * (INT_ / 16)};
                gemv_e2m1_vnni<WL, true>(s.down_c, s.down_s, s.down_ctx, hq, NDN, KDN, b0, b1,
                                         j.y, j.rw[i]);
            }
            bar.wait(ls);
            return;
        } else {
        // ---- phase A: gate + up + SiLU, partitioned by intermediate row -------------------
        int a0, a1;
        split(INT_, t, T, &a0, &a1);
        for (int i = 0; i < j.topk; ++i) {
            const ExpertSlab<WL>& s = tab->ex[j.sel[i]];
            gemv_e2m1<WL, false>(s.gate_c, s.gate_s, s.gate_ctx, j.xe, j.xo, NGU, KGU, a0, a1,
                                 gbuf, 0.0f);
            gemv_e2m1<WL, false>(s.up_c, s.up_s, s.up_ctx, j.xe, j.xo, NGU, KGU, a0, a1, ubuf,
                                 0.0f);
            float* hh = j.h + (size_t)i * INT_;
            for (int r = a0; r < a1; ++r) hh[r] = silu_mul(gbuf[r], ubuf[r]);
        }
        bar.wait(ls);
        // ---- phase B: down, partitioned by HIDDEN row so each thread owns disjoint y -----
        for (int i = 0; i < j.topk; ++i) {
            const float* hh = j.h + (size_t)i * INT_;
            deinterleave(hh, KDN, he + (size_t)i * (KDN / 2), ho + (size_t)i * (KDN / 2));
        }
        int b0, b1;
        split(HID, t, T, &b0, &b1);
        for (int r = b0; r < b1; ++r) j.y[r] = 0.0f;
        for (int i = 0; i < j.topk; ++i) {
            const ExpertSlab<WL>& s = tab->ex[j.sel[i]];
            gemv_e2m1<WL, true>(s.down_c, s.down_s, s.down_ctx, he + (size_t)i * (KDN / 2),
                                ho + (size_t)i * (KDN / 2), NDN, KDN, b0, b1, j.y, j.rw[i]);
        }
        bar.wait(ls);
        }
    }

    void start(int nthreads) {
        T = nthreads;
        bar.nthreads = T;
        for (int t = 1; t < T; ++t) th.emplace_back([this, t] { worker(t); });
    }
    void run() {
        done.store(0, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
        static thread_local std::vector<float> he, ho, gb, ub;
        if (he.empty()) {
            he.resize((size_t)32 * INT_ / 2);
            ho.resize((size_t)32 * INT_ / 2);
            gb.resize(INT_);
            ub.resize(INT_);
        }
        int ls = main_sense;
        run_slice(0, he.data(), ho.data(), gb.data(), ub.data(), ls);
        main_sense = ls;
        while (done.load(std::memory_order_acquire) != T - 1) _mm_pause();
    }
    int main_sense = 0;
    void stop() {
        quit.store(1, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
        for (auto& x : th) x.join();
    }
};

// -------------------------------------------------------------------------------------------
static double now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static std::vector<uint8_t> slurp(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) die(p.c_str());
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) die("fread");
    fclose(f);
    return v;
}

// xorshift, so the expert draw is reproducible and costs nothing in the timed region
static inline uint64_t xs(uint64_t& s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

// -------------------------------------------------------------------------------------------
// selftest: prove the core is genuinely format-parameterised by running a THIRD policy
// (MXFP4, group 32, E8M0 power-of-two scale) against a scalar reference.  No checkpoint data
// exists in that format; this only asserts the core, not model quality.
template <class WL>
static double selftest_policy(const char* name) {
    const int N = 64, K = 512;
    std::vector<uint8_t> codes((size_t)N * K / 2);
    std::vector<typename WL::scale_t> scales((size_t)N * K / WL::GROUP);
    std::vector<float> x(K), xe(K / 2), xo(K / 2), out(N), ref(N);
    std::vector<float> lut(256);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (auto& c : codes) c = (uint8_t)(xs(s) & 0xFF);
    for (auto& v : scales) v = (typename WL::scale_t)(xs(s) % 200 + 40);
    for (int i = 0; i < K; ++i) x[i] = (float)((int)(xs(s) % 2001) - 1000) / 500.0f;
    if (WL::GROUP == 32)
        for (int i = 0; i < 256; ++i) lut[i] = ldexpf(1.0f, i - 127);
    else
        build_e4m3_lut(lut.data(), 1.0f);
    deinterleave(x.data(), K, xe.data(), xo.data());
    typename WL::ctx_t ctx{lut.data()};
    gemv_e2m1<WL, false>(codes.data(), scales.data(), ctx, xe.data(), xo.data(), N, K, 0, N,
                         out.data(), 0.0f);
    for (int n = 0; n < N; ++n) {
        double acc = 0;
        for (int k = 0; k < K; ++k) {
            const uint8_t byte = codes[(size_t)n * K / 2 + k / 2];
            const int code = (k & 1) ? (byte >> 4) : (byte & 0xF);
            const double sc =
                WL::scale_ref(&scales[(size_t)n * K / WL::GROUP], k, ctx);
            acc += (double)kE2M1[code] * sc * (double)x[k];
        }
        ref[n] = (float)acc;
    }
    double num = 0, den = 0;
    for (int n = 0; n < N; ++n) {
        num += (double)(out[n] - ref[n]) * (out[n] - ref[n]);
        den += (double)ref[n] * ref[n];
    }
    const double rel = sqrt(num / (den > 0 ? den : 1));
    printf("  selftest %-16s rel_rms=%.3e  %s\n", name, rel, rel < 1e-5 ? "PASS" : "FAIL");
    return rel;
}

// Same idea for the int8-activation core: run the intrinsics and the scalar-double twin on the
// SAME quantized activation, so this isolates the tile layout / bias correction / scale folding
// from the activation-quantization error.  That error is measured separately, on the real
// checkpoint, by --mode verify.
template <class WL>
static double selftest_vnni(const char* name) {
    const int N = 64, K = 512;
    const int NT = (N / VNNI_RB) * (K / VNNI_GS);
    std::vector<uint8_t> codes((size_t)NT * VNNI_TILE_W);
    std::vector<typename WL::scale_t> scales((size_t)NT * VNNI_RB);
    std::vector<float> x(K), out(N), xsc(K / 16);
    std::vector<int8_t> xq(K);
    std::vector<int32_t> xsum(K / 16);
    std::vector<double> ref(N);
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (auto& c : codes) c = (uint8_t)(xs(s) & 0xFF);
    // Scale bytes are drawn from the POSITIVE-NORMAL e4m3 range 0x08..0x7E, which is the range
    // the real checkpoint occupies (census: 629,145,600 bytes, all in 0x40..0x7E) and the one
    // e4m3x16_normpos_to_ps is specialised to.  Widening this without also widening the decoder
    // is what the load-time guard in Emit<WLoadVnniE4m3> exists to catch.
    for (auto& v : scales) v = (typename WL::scale_t)(xs(s) % 0x77 + 0x08);
    for (int i = 0; i < K; ++i) x[i] = (float)((int)(xs(s) % 2001) - 1000) / 500.0f;
    quantize_act_g16(x.data(), K, xq.data(), xsc.data(), xsum.data());
    typename WL::ctx_t ctx{};
    ctx.gmul = 1.0f;
    const QAct qa{xq.data(), xsc.data(), xsum.data()};
    gemv_e2m1_vnni<WL, false>(codes.data(), scales.data(), ctx, qa, N, K, 0, N / VNNI_RB,
                              out.data(), 0.0f);
    gemv_e2m1_vnni_ref<WL>(codes.data(), scales.data(), ctx, qa, N, K, ref.data());
    double num = 0, den = 0;
    for (int n = 0; n < N; ++n) {
        num += (out[n] - ref[n]) * (out[n] - ref[n]);
        den += ref[n] * ref[n];
    }
    const double rel = sqrt(num / (den > 0 ? den : 1));
    printf("  selftest %-20s rel_rms=%.3e  %s\n", name, rel, rel < 1e-6 ? "PASS" : "FAIL");
    return rel;
}

// -------------------------------------------------------------------------------------------
// One policy, end to end.  Everything below is policy-agnostic; the WLoad (and, for TILED
// policies, the activation dtype it implies) is the only thing that varies.
template <class WL>
static int run_policy(const std::vector<Plan>& plans,
                      const std::vector<std::vector<const uint8_t*>>& maps, int cap,
                      const std::string& mode, const std::string& refdir, int threads, int iters,
                      int warm) {
    const double t0 = now();
    Table<WL> tab = build_table<WL>(plans, maps, cap);
    const size_t bpe = tab.slab;
    const int NE = tab.n_experts;
    fprintf(stderr, "[table] policy=%s experts=%d slab=%zu total=%.2f GiB build=%.1fs\n",
            WL::NAME, NE, bpe, (double)NE * bpe / (1 << 30), now() - t0);

    // scratch shared by verify and bench
    const int TOPK_MAX = 32;
    std::vector<float> h((size_t)TOPK_MAX * INT_), y(HID);
    std::vector<int8_t> hq((size_t)TOPK_MAX * INT_);
    std::vector<float> hsc((size_t)TOPK_MAX * INT_ / 16);
    std::vector<int32_t> hsum((size_t)TOPK_MAX * INT_ / 16);
    std::vector<float> xe(KGU / 2), xo(KGU / 2), xsc(KGU / 16);
    std::vector<int8_t> xq(KGU);
    std::vector<int32_t> xsum(KGU / 16);

    auto make_job = [&](const int* sel, const float* rw, int topk, const float* x) {
        Job j;
        j.sel = sel; j.rw = rw; j.topk = topk;
        deinterleave(x, KGU, xe.data(), xo.data());
        j.xe = xe.data(); j.xo = xo.data();
        quantize_act_g16(x, KGU, xq.data(), xsc.data(), xsum.data());
        j.xq = QAct{xq.data(), xsc.data(), xsum.data()};
        j.h = h.data(); j.hq = hq.data(); j.hsc = hsc.data(); j.hsum = hsum.data();
        j.y = y.data();
        return j;
    };

    if (mode == "verify") {
        auto xb = slurp(refdir + "/x.bin");
        auto selb = slurp(refdir + "/sel.bin");
        auto rwb = slurp(refdir + "/rw.bin");
        auto yb = slurp(refdir + "/y_f64.bin");
        const float* x = (const float*)xb.data();
        const int* sel = (const int*)selb.data();
        const float* rw = (const float*)rwb.data();
        const double* yref = (const double*)yb.data();
        const int topk = (int)(selb.size() / 4);
        for (int T : {1, 2, 4, 8}) {
            std::fill(y.begin(), y.end(), 0.0f);
            Runner<WL> R;
            R.tab = &tab;
            R.job = make_job(sel, rw, topk, x);
            R.start(T);
            R.run();
            R.stop();
            double num = 0, den = 0, maxabs = 0;
            for (int i = 0; i < HID; ++i) {
                const double d = (double)y[i] - yref[i];
                num += d * d;
                den += yref[i] * yref[i];
                if (fabs(d) > maxabs) maxabs = fabs(d);
            }
            printf("verify T=%-2d policy=%-20s rel_rms=%.3e max_abs=%.3e ref_rms=%.4f\n", T,
                   WL::NAME, sqrt(num / den), maxabs, sqrt(den / HID));
        }
        return 0;
    }

    // ---- bench --------------------------------------------------------------------------
    const int topk = 10;
    std::vector<float> x(HID);
    uint64_t s = 0xDEADBEEF12345ull;
    for (int i = 0; i < HID; ++i) x[i] = (float)((int)(xs(s) % 2001) - 1000) / 2000.0f;
    std::vector<int> sel(topk);
    std::vector<float> rw(topk, 0.1f);

    printf("{\"policy\":\"%s\",\"experts\":%d,\"table_gib\":%.3f,\"bytes_per_expert\":%zu,"
           "\"topk\":%d,\"threads\":[",
           WL::NAME, NE, (double)NE * bpe / (1 << 30), bpe, topk);
    bool first = true;
    for (int T : {1, 2, 3, 4, 6, 8, 12, 16}) {
        if (T > threads) break;
        Runner<WL> R;
        R.tab = &tab;
        R.job = make_job(sel.data(), rw.data(), topk, x.data());
        R.start(T);
        for (int i = 0; i < warm; ++i) {
            for (int e = 0; e < topk; ++e) sel[e] = (int)(xs(s) % NE);
            R.run();
        }
        std::vector<double> ms;
        for (int i = 0; i < iters; ++i) {
            for (int e = 0; e < topk; ++e) sel[e] = (int)(xs(s) % NE);
            const double a = now();
            R.run();
            ms.push_back((now() - a) * 1e3);
        }
        R.stop();
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2], p5 = ms[ms.size() / 20];
        const double B = (double)topk * bpe;
        printf("%s{\"T\":%d,\"ms_med\":%.4f,\"ms_p5\":%.4f,\"gb_s_med\":%.2f,\"gb_s_p5\":%.2f,"
               "\"gb_s_per_thread_med\":%.2f,\"tok_s_37L\":%.2f}",
               first ? "" : ",", T, med, p5, B / (med * 1e-3) / 1e9, B / (p5 * 1e-3) / 1e9,
               B / (med * 1e-3) / 1e9 / T, 1000.0 / (med * 37.0));
        first = false;
        fflush(stdout);
    }
    printf("]}\n");
    return 0;
}

int main(int argc, char** argv) {
    std::vector<std::string> planpaths;
    std::string refdir, mode = "verify", policy = "e4m3";
    int threads = 8, iters = 200, cap = 0, warm = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&] { return std::string(argv[++i]); };
        if (a == "--plan") planpaths.push_back(nx());
        else if (a == "--ref") refdir = nx();
        else if (a == "--mode") mode = nx();
        else if (a == "--policy") policy = nx();
        else if (a == "--threads") threads = atoi(nx().c_str());
        else if (a == "--iters") iters = atoi(nx().c_str());
        else if (a == "--warm") warm = atoi(nx().c_str());
        else if (a == "--cap") cap = atoi(nx().c_str());
    }
    if (mode == "selftest") {
        double a = selftest_policy<WLoadNvfp4E4m3>("nvfp4_e4m3_g16");
        double b = selftest_policy<WLoadNvfp4Fp16>("nvfp4_fp16_g16");
        double c = selftest_policy<WLoadMxfp4E8m0>("mxfp4_e8m0_g32");
        // int8-activation core, against its own scalar-double twin on identical inputs
        const bool tab_ok = e2m1_tables_agree();
        printf("  e2m1 int8 codebook == kE2M1*2 : %s\n", tab_ok ? "PASS" : "FAIL");
        double d = selftest_vnni<WLoadVnniE4m3>("vnni_nvfp4_e4m3_g16");
        double e = selftest_vnni<WLoadVnniFp16>("vnni_nvfp4_fp16_g16");
        return (a < 1e-5 && b < 1e-5 && c < 1e-5 && d < 1e-6 && e < 1e-6 && tab_ok) ? 0 : 1;
    }
    if (planpaths.empty()) {
        fprintf(stderr, "need --plan\n");
        return 2;
    }

    std::vector<Plan> plans;
    std::vector<std::vector<const uint8_t*>> maps;
    for (auto& p : planpaths) {
        plans.push_back(read_plan(p.c_str()));
        std::vector<const uint8_t*> m;
        for (auto& s : plans.back().shards) {
            size_t l;
            m.push_back(map_file(s, &l));
        }
        maps.push_back(m);
    }

    if (policy == "e4m3")  return run_policy<WLoadNvfp4E4m3>(plans, maps, cap, mode, refdir, threads, iters, warm);
    if (policy == "fp16")  return run_policy<WLoadNvfp4Fp16>(plans, maps, cap, mode, refdir, threads, iters, warm);
    if (policy == "vnni")  return run_policy<WLoadVnniE4m3>(plans, maps, cap, mode, refdir, threads, iters, warm);
    if (policy == "vnni16") return run_policy<WLoadVnniFp16>(plans, maps, cap, mode, refdir, threads, iters, warm);
    fprintf(stderr, "unknown --policy %s (e4m3|fp16|vnni|vnni16)\n", policy.c_str());
    return 2;
}
