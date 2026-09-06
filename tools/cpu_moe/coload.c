// Co-load generator: stands in for the engine while the MoE kernel is measured on other cores.
// Answers the additivity question directly instead of by argument -- "MoE on 2 cores leaves the
// engine alone" is a claim about interference, and interference has two channels (execution
// resources and DDR) that behave completely differently (RESULTS_PERCORE section 5).
//
//   ./coload <mode> <nthreads> <first_core> <seconds>
//     mode=compute : AVX-512 FMA chains, cache-resident.  Costs cores, ~no DDR.
//     mode=stream  : sequential DDR reads.  Costs DDR, ~no execution resources.
//     mode=idle    : nothing (control).
// Prints what it actually achieved so the co-load is a measured quantity, not an assumption.
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static volatile int g_stop = 0;
static int g_first, g_mode, g_n;
static uint8_t *g_buf;
static size_t g_bufsz;
static _Atomic long long g_bytes;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void *th(void *a) {
    const int id = (int)(long)a;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(g_first + id, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    if (g_mode == 0) {  // compute
        __m512 x[8];
        for (int i = 0; i < 8; i++) x[i] = _mm512_set1_ps(1.0f + i);
        const __m512 c = _mm512_set1_ps(1.0000001f);
        long long n = 0;
        while (!g_stop) {
            for (int r = 0; r < 1000; r++)
                for (int i = 0; i < 8; i++) x[i] = _mm512_fmadd_ps(x[i], c, c);
            n += 8000;
        }
        atomic_fetch_add(&g_bytes, n);
        float sink[16];
        _mm512_storeu_ps(sink, x[0]);
        if (sink[0] == 12345.678f) puts("");
    } else if (g_mode == 1) {  // stream
        const size_t chunk = g_bufsz / g_n;
        uint8_t *base = g_buf + (size_t)id * chunk;
        __m512i acc = _mm512_setzero_si512();
        long long n = 0;
        while (!g_stop) {
            for (size_t o = 0; o < chunk; o += 256) {
                acc = _mm512_add_epi64(acc, _mm512_stream_load_si512((void *)(base + o)));
                acc = _mm512_add_epi64(acc, _mm512_stream_load_si512((void *)(base + o + 64)));
                acc = _mm512_add_epi64(acc, _mm512_stream_load_si512((void *)(base + o + 128)));
                acc = _mm512_add_epi64(acc, _mm512_stream_load_si512((void *)(base + o + 192)));
            }
            n += (long long)chunk;
        }
        atomic_fetch_add(&g_bytes, n);
        if (_mm512_reduce_add_epi64(acc) == 1234567) puts("");
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: coload <compute|stream|idle> <nthreads> <first_core> <seconds>\n");
        return 2;
    }
    g_mode = !strcmp(argv[1], "compute") ? 0 : (!strcmp(argv[1], "stream") ? 1 : 2);
    g_n = atoi(argv[2]);
    g_first = atoi(argv[3]);
    const double secs = atof(argv[4]);
    if (g_mode == 2) {
        struct timespec ts = {(time_t)secs, 0};
        nanosleep(&ts, NULL);
        printf("{\"coload\":\"idle\"}\n");
        return 0;
    }
    if (g_mode == 1) {
        g_bufsz = 4ull << 30;
        g_buf = mmap(NULL, g_bufsz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (g_buf == MAP_FAILED) return 1;
        madvise(g_buf, g_bufsz, MADV_HUGEPAGE);
        memset(g_buf, 1, g_bufsz);
    }
    pthread_t t[64];
    const double t0 = now();
    for (int i = 0; i < g_n; i++) pthread_create(&t[i], NULL, th, (void *)(long)i);
    struct timespec ts;
    ts.tv_sec = (time_t)secs;
    ts.tv_nsec = (long)((secs - ts.tv_sec) * 1e9);
    nanosleep(&ts, NULL);
    g_stop = 1;
    for (int i = 0; i < g_n; i++) pthread_join(t[i], NULL);
    const double el = now() - t0;
    if (g_mode == 0)
        printf("{\"coload\":\"compute\",\"threads\":%d,\"cores\":\"%d-%d\",\"gflop_s\":%.1f}\n",
               g_n, g_first, g_first + g_n - 1,
               (double)atomic_load(&g_bytes) * 16 * 2 / el / 1e9);
    else
        printf("{\"coload\":\"stream\",\"threads\":%d,\"cores\":\"%d-%d\",\"gb_s\":%.2f}\n", g_n,
               g_first, g_first + g_n - 1, (double)atomic_load(&g_bytes) / el / 1e9);
    return 0;
}
