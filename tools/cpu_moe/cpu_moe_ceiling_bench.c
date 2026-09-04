// CPU-side MoE feasibility bench for Qwen3.8-Flash-Next-NVFP4 on Ryzen 7 7800X3D.
//   mode=gather   : random 3.072 MB expert-slab gather out of a multi-GB table (DDR read BW)
//   mode=stream   : sequential read (DDR peak reference)
//   mode=gemv     : AVX-512 VNNI E2M1(NVFP4) dequant+GEMV, reports GB/s of WEIGHTS consumed
//   mode=interf   : gather + a rate-limited "PCIe DMA emulator" reader, reports both
//   mode=verify   : numeric check of the GEMV kernel vs a scalar reference
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <math.h>
#include <sys/mman.h>
#include <immintrin.h>

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void pin(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

// cpu id for logical thread i: fill physical cores 0..7 first, then SMT siblings 8..15
static int cpu_for(int i) { return i; }

static void *alloc_huge(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(p, bytes, MADV_HUGEPAGE);
    // touch every 4K page to fault it in (and randomize content so XOR isn't trivial)
    uint64_t s = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < bytes; i += 4096) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *(volatile uint64_t *)((char *)p + i) = s;
    }
    return p;
}

static inline uint64_t xs(uint64_t *s) {
    uint64_t x = *s; x ^= x << 13; x ^= x >> 7; x ^= x << 17; *s = x; return x;
}

// ---------------------------------------------------------------- read kernels
static inline __m512i read_range(const uint8_t *p, size_t n, __m512i acc) {
    for (size_t i = 0; i < n; i += 256) {
        __m512i a = _mm512_load_si512((const void *)(p + i));
        __m512i b = _mm512_load_si512((const void *)(p + i + 64));
        __m512i c = _mm512_load_si512((const void *)(p + i + 128));
        __m512i d = _mm512_load_si512((const void *)(p + i + 192));
        acc = _mm512_xor_si512(acc, _mm512_xor_si512(_mm512_xor_si512(a, b),
                                                     _mm512_xor_si512(c, d)));
    }
    return acc;
}

// ---------------------------------------------------------------- shared state
static uint8_t *g_table;
static size_t   g_table_bytes;
static size_t   g_slab;         // bytes per expert slab
static size_t   g_nslab;
static int      g_nthreads;
static double   g_seconds;
static volatile int g_go = 0, g_stop = 0;
static double   g_dma_gbs = 0;  // target rate for the DMA emulator (0 = off / unlimited)

typedef struct { int id; double bytes; double secs; uint64_t sink; } res_t;

static void *th_gather(void *arg) {
    res_t *r = (res_t *)arg;
    pin(cpu_for(r->id));
    uint64_t s = 0x1234567 + 0x9e37u * (uint64_t)(r->id + 1);
    for (int i = 0; i < 64; i++) xs(&s);
    __m512i acc = _mm512_setzero_si512();
    while (!g_go) sched_yield();
    double t0 = now(); double bytes = 0;
    while (!g_stop) {
        for (int e = 0; e < 10; e++) {            // one layer's top-10 expert gather
            size_t idx = xs(&s) % g_nslab;
            acc = read_range(g_table + idx * g_slab, g_slab, acc);
            bytes += (double)g_slab;
        }
    }
    r->secs = now() - t0; r->bytes = bytes;
    r->sink = (uint64_t)_mm512_reduce_add_epi64(acc);
    return NULL;
}

static void *th_stream(void *arg) {
    res_t *r = (res_t *)arg;
    pin(cpu_for(r->id));
    size_t chunk = g_table_bytes / g_nthreads;
    chunk &= ~(size_t)255;
    const uint8_t *base = g_table + (size_t)r->id * chunk;
    __m512i acc = _mm512_setzero_si512();
    while (!g_go) sched_yield();
    double t0 = now(); double bytes = 0;
    while (!g_stop) { acc = read_range(base, chunk, acc); bytes += (double)chunk; }
    r->secs = now() - t0; r->bytes = bytes;
    r->sink = (uint64_t)_mm512_reduce_add_epi64(acc);
    return NULL;
}

// rate-limited sequential reader standing in for the GPU's PCIe DMA pull from host DDR
static res_t g_dma_res;
static int g_ndma = 1;
static void *th_dma(void *arg) {
    res_t *r = (res_t *)arg;
    pin(7 - r->id);  // dedicated physical cores, top-down; a real DMA engine costs no core at all
    size_t chunk = 1u << 20;
    __m512i acc = _mm512_setzero_si512();
    size_t off = 0;
    while (!g_go) sched_yield();
    double t0 = now(); double bytes = 0;
    double target = g_dma_gbs * 1e9 / (double)g_ndma;
    while (!g_stop) {
        if (off + chunk > g_table_bytes) off = 0;
        acc = read_range(g_table + off, chunk, acc);
        off += chunk * 7;  // stride so it is not the same lines as before
        if (off >= g_table_bytes) off %= g_table_bytes;
        off &= ~(size_t)255;
        bytes += (double)chunk;
        if (target > 0) {
            double want = bytes / target;
            double have = now() - t0;
            if (want > have) {
                struct timespec ts;
                double d = want - have;
                ts.tv_sec = (time_t)d; ts.tv_nsec = (long)((d - ts.tv_sec) * 1e9);
                nanosleep(&ts, NULL);
            }
        }
    }
    r->secs = now() - t0; r->bytes = bytes;
    r->sink = (uint64_t)_mm512_reduce_add_epi64(acc);
    return NULL;
}

// ============================================================ NVFP4 CPU GEMV ==
// Resident layout (we own it; identical footprint to the GPU-side layout: 0.625 B/weight)
//   For a matrix (N rows, K cols), row-blocks of 16, groups of 16 k:
//     tile(rb,g) = 128 B of packed E2M1 nibbles + 32 B of fp16 per-row group scales.
//   Nibble placement inside the 128 B so unpack is 2 loads + 4 shuffles:
//     B[b].lo   -> row (b/4), k = g*16 + 0 + (b%4)      b in [0,64)
//     B[b].hi   -> row (b/4), k = g*16 + 4 + (b%4)
//     B[64+b].lo-> row (b/4), k = g*16 + 8 + (b%4)
//     B[64+b].hi-> row (b/4), k = g*16 +12 + (b%4)
// Math: E2M1 magnitudes {0,.5,1,1.5,2,3,4,6} * 2 are integers {0,1,2,3,4,6,8,12}, so an
// E2M1 code maps EXACTLY to int8; +16 makes it unsigned for VPDPBUSD (u8 x s8), and the
// +16 bias is removed with a per-group  16*sum(xq[g])  correction shared by all 16 rows.
#define RB 16
#define GS 16
#define TILE_W 128
#define TILE_S 32

static const int8_t E2M1_I8[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

typedef struct {
    int N, K;
    uint8_t *w;    // (N/RB) * (K/GS) * TILE_W
    uint8_t *s;    // (N/RB) * (K/GS) * TILE_S  (fp16)
    size_t bytes;
} qmat_t;

static void qmat_alloc(qmat_t *m, int N, int K, void *arena, size_t *cursor) {
    m->N = N; m->K = K;
    size_t nt = (size_t)(N / RB) * (K / GS);
    m->w = (uint8_t *)arena + *cursor; *cursor += nt * TILE_W;
    m->s = (uint8_t *)arena + *cursor; *cursor += nt * TILE_S;
    m->bytes = nt * (TILE_W + TILE_S);
}

// fill with pseudo-random codes/scales
static void qmat_fill(qmat_t *m, uint64_t seed) {
    uint64_t s = seed | 1;
    size_t nt = (size_t)(m->N / RB) * (m->K / GS);
    for (size_t i = 0; i < nt * TILE_W; i++) m->w[i] = (uint8_t)(xs(&s) & 0xFF);
    _Float16 *sc = (_Float16 *)m->s;
    for (size_t i = 0; i < nt * (TILE_S / 2); i++)
        sc[i] = (_Float16)(0.002f + 0.02f * ((xs(&s) >> 40) & 1023) / 1023.0f);
}

static uint8_t E2M1_U8[16];
static void init_lut(void) { for (int i = 0; i < 16; i++) E2M1_U8[i] = (uint8_t)(E2M1_I8[i] + 16); }

static void gemv(const qmat_t *m, const int8_t *xq, const float *xsc, const int32_t *xsum,
                 float *y, int rb0, int rb1) {
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)E2M1_U8));
    const __m512i mlo = _mm512_set1_epi8(0x0F);
    const int NG = m->K / GS;
    for (int rb = rb0; rb < rb1; rb++) {
        __m512 yacc = _mm512_setzero_ps();
        const uint8_t *wp = m->w + (size_t)rb * NG * TILE_W;
        const uint8_t *sp = m->s + (size_t)rb * NG * TILE_S;
        for (int g = 0; g < NG; g++, wp += TILE_W, sp += TILE_S) {
            __m512i z0 = _mm512_loadu_si512((const void *)wp);
            __m512i z1 = _mm512_loadu_si512((const void *)(wp + 64));
            __m512i w0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(z0, mlo));
            __m512i w1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(z0, 4), mlo));
            __m512i w2 = _mm512_shuffle_epi8(lut, _mm512_and_si512(z1, mlo));
            __m512i w3 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(z1, 4), mlo));
            const int32_t *xq32 = (const int32_t *)(xq + (size_t)g * GS);
            __m512i acc = _mm512_setzero_si512();
            acc = _mm512_dpbusd_epi32(acc, w0, _mm512_set1_epi32(xq32[0]));
            acc = _mm512_dpbusd_epi32(acc, w1, _mm512_set1_epi32(xq32[1]));
            acc = _mm512_dpbusd_epi32(acc, w2, _mm512_set1_epi32(xq32[2]));
            acc = _mm512_dpbusd_epi32(acc, w3, _mm512_set1_epi32(xq32[3]));
            acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(16 * xsum[g]));
            __m512 f  = _mm512_cvtepi32_ps(acc);
            __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)sp));
            sc = _mm512_mul_ps(sc, _mm512_set1_ps(0.5f * xsc[g]));
            yacc = _mm512_fmadd_ps(f, sc, yacc);
        }
        _mm512_storeu_ps(y + rb * RB, yacc);
    }
}

// scalar reference on the same quantized inputs (integer-exact path, fp32 accumulate)
static void gemv_ref(const qmat_t *m, const int8_t *xq, const float *xsc, float *y) {
    const int NG = m->K / GS;
    for (int rb = 0; rb < m->N / RB; rb++) {
        for (int r = 0; r < RB; r++) {
            double s = 0;
            for (int g = 0; g < NG; g++) {
                const uint8_t *B = m->w + ((size_t)rb * NG + g) * TILE_W;
                const _Float16 *SC = (const _Float16 *)(m->s + ((size_t)rb * NG + g) * TILE_S);
                double gs = 0;
                for (int j = 0; j < 16; j++) {
                    int b, code;
                    if (j < 4)       { b = r * 4 + j;      code = B[b] & 15; }
                    else if (j < 8)  { b = r * 4 + (j - 4);  code = B[b] >> 4; }
                    else if (j < 12) { b = 64 + r * 4 + (j - 8); code = B[b] & 15; }
                    else             { b = 64 + r * 4 + (j - 12); code = B[b] >> 4; }
                    gs += 0.5 * (double)E2M1_I8[code] * (double)xq[g * 16 + j];
                }
                s += gs * (double)(float)SC[r] * (double)xsc[g];
            }
            y[rb * RB + r] = (float)s;
        }
    }
}

// ------------------------------------------------------- gemv driver (threaded)
static qmat_t *g_mats;      // g_nmat matrices forming the resident "expert table"
static int      g_nmat;
static int8_t  *g_xq;
static float   *g_xsc;
static int32_t *g_xsum;

static void *th_gemv(void *arg) {
    res_t *r = (res_t *)arg;
    pin(cpu_for(r->id));
    float *y = aligned_alloc(64, 4096 * sizeof(float));
    uint64_t s = 0xabcdef + 0x9e37u * (uint64_t)(r->id + 1);
    for (int i = 0; i < 32; i++) xs(&s);
    while (!g_go) sched_yield();
    double t0 = now(); double bytes = 0;
    while (!g_stop) {
        int mi = (int)(xs(&s) % (uint64_t)g_nmat);
        const qmat_t *m = &g_mats[mi];
        gemv(m, g_xq, g_xsc, g_xsum, y, 0, m->N / RB);
        bytes += (double)m->bytes;
    }
    r->secs = now() - t0; r->bytes = bytes; r->sink = (uint64_t)y[0];
    free(y);
    return NULL;
}

// ---------------------------------------------------------------------- main
static double run_threads(void *(*fn)(void *), int nthreads, double secs, double *out_gbs) {
    pthread_t th[32]; res_t res[32];
    memset(res, 0, sizeof(res));
    g_go = 0; g_stop = 0;
    for (int i = 0; i < nthreads; i++) { res[i].id = i; pthread_create(&th[i], NULL, fn, &res[i]); }
    struct timespec ts = {0, 50000000}; nanosleep(&ts, NULL);
    double t0 = now(); g_go = 1;
    while (now() - t0 < secs) { struct timespec q = {0, 5000000}; nanosleep(&q, NULL); }
    g_stop = 1;
    double wall = now() - t0, tot = 0; uint64_t sink = 0;
    for (int i = 0; i < nthreads; i++) { pthread_join(th[i], NULL); tot += res[i].bytes; sink ^= res[i].sink; }
    if (sink == 0xdeadbeefULL) fprintf(stderr, "sink\n");
    *out_gbs = tot / wall / 1e9;
    return wall;
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "gather";
    size_t table_mb = argc > 2 ? (size_t)atol(argv[2]) : 6144;
    g_nthreads = argc > 3 ? atoi(argv[3]) : 8;
    g_seconds  = argc > 4 ? atof(argv[4]) : 3.0;
    size_t slab_kb = argc > 5 ? (size_t)atol(argv[5]) : 3000;   // 3.072 MB expert
    g_dma_gbs = argc > 6 ? atof(argv[6]) : 0.0;
    g_ndma    = argc > 7 ? atoi(argv[7]) : 1;
    init_lut();

    g_slab = slab_kb * 1024;
    g_table_bytes = table_mb * 1024 * 1024;
    g_table_bytes -= g_table_bytes % g_slab;
    g_nslab = g_table_bytes / g_slab;

    if (!strcmp(mode, "verify")) {
        static uint8_t arena[64 << 20] __attribute__((aligned(64)));
        size_t cur = 0; qmat_t m; qmat_alloc(&m, 1280, 2560, arena, &cur);
        qmat_fill(&m, 42);
        int8_t *xq = aligned_alloc(64, 2560);
        float  *xsc = aligned_alloc(64, (2560 / GS) * sizeof(float));
        int32_t *xsum = aligned_alloc(64, (2560 / GS) * sizeof(int32_t));
        uint64_t s = 7;
        for (int i = 0; i < 2560; i++) xq[i] = (int8_t)((xs(&s) & 0xFF) - 128);
        for (int g = 0; g < 2560 / GS; g++) {
            xsc[g] = 0.01f + 0.001f * (g % 7);
            int32_t t = 0; for (int j = 0; j < GS; j++) t += xq[g * GS + j];
            xsum[g] = t;
        }
        float *y = aligned_alloc(64, 1280 * 4), *yr = aligned_alloc(64, 1280 * 4);
        gemv(&m, xq, xsc, xsum, y, 0, 1280 / RB);
        gemv_ref(&m, xq, xsc, yr);
        double maxrel = 0, maxabs = 0, ref_rms = 0;
        for (int i = 0; i < 1280; i++) {
            double d = fabs(y[i] - yr[i]);
            if (d > maxabs) maxabs = d;
            double rel = d / (fabs(yr[i]) + 1e-9);
            if (rel > maxrel) maxrel = rel;
            ref_rms += (double)yr[i] * yr[i];
        }
        ref_rms = sqrt(ref_rms / 1280);
        printf("{\"verify\":{\"max_abs\":%.6g,\"max_rel\":%.6g,\"ref_rms\":%.6g,\"y0\":%.6g,\"yr0\":%.6g}}\n",
               maxabs, maxrel, ref_rms, y[0], yr[0]);
        return 0;
    }

    if (!strcmp(mode, "gemv") || !strcmp(mode, "gemvsmall")) {
        // build an "expert table": each matrix = one expert's gate_up (1280x2560) so the
        // per-matrix footprint is 2.048 MB; a pair of them ~ one 3.072 MB expert.
        int N = 1280, K = 2560;
        size_t per = (size_t)(N / RB) * (K / GS) * (TILE_W + TILE_S);
        g_nmat = (int)(g_table_bytes / per);
        if (g_nmat < 1) g_nmat = 1;
        size_t need = (size_t)g_nmat * per;
        void *arena = alloc_huge(need + 4096);
        g_mats = calloc(g_nmat, sizeof(qmat_t));
        size_t cur = 0;
        for (int i = 0; i < g_nmat; i++) { qmat_alloc(&g_mats[i], N, K, arena, &cur); }
        // fill only the first few (content is irrelevant to timing; keeps setup fast)
        for (int i = 0; i < g_nmat && i < 8; i++) qmat_fill(&g_mats[i], 1000 + i);
        g_xq  = aligned_alloc(64, K);
        g_xsc = aligned_alloc(64, (K / GS) * sizeof(float));
        g_xsum = aligned_alloc(64, (K / GS) * sizeof(int32_t));
        uint64_t s = 5;
        for (int i = 0; i < K; i++) g_xq[i] = (int8_t)((xs(&s) & 0xFF) - 128);
        for (int g = 0; g < K / GS; g++) {
            g_xsc[g] = 0.01f; int32_t t = 0;
            for (int j = 0; j < GS; j++) t += g_xq[g * GS + j];
            g_xsum[g] = t;
        }
        double gbs; run_threads(th_gemv, g_nthreads, g_seconds, &gbs);
        printf("{\"mode\":\"%s\",\"threads\":%d,\"table_mb\":%zu,\"n_mat\":%d,\"weights_gb_s\":%.3f}\n",
               mode, g_nthreads, table_mb, g_nmat, gbs);
        return 0;
    }

    g_table = alloc_huge(g_table_bytes);

    if (!strcmp(mode, "stream")) {
        double gbs; run_threads(th_stream, g_nthreads, g_seconds, &gbs);
        printf("{\"mode\":\"stream\",\"threads\":%d,\"table_mb\":%zu,\"gb_s\":%.3f}\n",
               g_nthreads, table_mb, gbs);
        return 0;
    }
    if (!strcmp(mode, "gather")) {
        double gbs; run_threads(th_gather, g_nthreads, g_seconds, &gbs);
        printf("{\"mode\":\"gather\",\"threads\":%d,\"table_mb\":%zu,\"slab_kb\":%zu,\"gb_s\":%.3f}\n",
               g_nthreads, table_mb, slab_kb, gbs);
        return 0;
    }
    if (!strcmp(mode, "interf")) {
        pthread_t th[32], dth[8]; res_t res[32], dres[8];
        memset(res, 0, sizeof(res)); memset(dres, 0, sizeof(dres));
        g_go = 0; g_stop = 0;
        for (int i = 0; i < g_nthreads; i++) { res[i].id = i; pthread_create(&th[i], NULL, th_gather, &res[i]); }
        for (int i = 0; i < g_ndma; i++) { dres[i].id = i; pthread_create(&dth[i], NULL, th_dma, &dres[i]); }
        struct timespec ts = {0, 50000000}; nanosleep(&ts, NULL);
        double t0 = now(); g_go = 1;
        while (now() - t0 < g_seconds) { struct timespec q = {0, 5000000}; nanosleep(&q, NULL); }
        g_stop = 1;
        double wall = now() - t0, tot = 0, dtot = 0;
        for (int i = 0; i < g_nthreads; i++) { pthread_join(th[i], NULL); tot += res[i].bytes; }
        for (int i = 0; i < g_ndma; i++) { pthread_join(dth[i], NULL); dtot += dres[i].bytes; }
        printf("{\"mode\":\"interf\",\"moe_threads\":%d,\"dma_threads\":%d,\"dma_target_gb_s\":%.2f,"
               "\"moe_gb_s\":%.3f,\"dma_gb_s\":%.3f,\"total_gb_s\":%.3f}\n",
               g_nthreads, g_ndma, g_dma_gbs, tot / wall / 1e9, dtot / wall / 1e9,
               (tot + dtot) / wall / 1e9);
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
}
