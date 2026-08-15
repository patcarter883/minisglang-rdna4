// p2p.hip — what does a cross-card transfer actually cost on this box? torch-free.
//
// This is the number that decides tensor-parallel vs pipeline-parallel here. TP=2 pays TWO
// all-reduces per transformer layer; PP=2 pays ONE activation handoff per micro-batch for the whole
// model. If the link is slow, that ratio (order 100x fewer bytes) is PP's entire case; if the link
// is fast, TP's per-layer sync is cheap and PP only buys pipeline bubbles.
//
// Reports: whether P2P is even possible between the two cards (on separate PCIe root ports it often
// is not, and the copy silently bounces through system RAM), the achieved bandwidth at sizes that
// match real activation tensors, and the LATENCY floor, which is what a bs=1 decode step actually
// pays per stage boundary.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

#define OK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
  printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); return 2;} } while(0)

int main() {
    int n = 0; OK(hipGetDeviceCount(&n));
    printf("devices visible: %d\n", n);
    if (n < 2) { printf("need 2 cards; lease with -n 2 and forward both\n"); return 1; }

    for (int i = 0; i < 2; ++i) {
        hipDeviceProp_t p; OK(hipGetDeviceProperties(&p, i));
        printf("  cuda:%d %s  pciBus=%02x:%02x.%x\n", i, p.name, p.pciBusID, p.pciDeviceID, 0);
    }
    int can01 = 0, can10 = 0;
    OK(hipDeviceCanAccessPeer(&can01, 0, 1));
    OK(hipDeviceCanAccessPeer(&can10, 1, 0));
    printf("\nP2P: 0->1 %s   1->0 %s\n", can01 ? "YES" : "NO", can10 ? "YES" : "NO");
    if (can01) { OK(hipSetDevice(0)); hipDeviceEnablePeerAccess(1, 0); }
    if (can10) { OK(hipSetDevice(1)); hipDeviceEnablePeerAccess(0, 0); }
    printf("(NO means the copy is staged through system RAM, at roughly half the bandwidth and\n"
           " with the host in the loop — which is what separate root ports usually give you.)\n");

    // Sizes chosen to bracket real tensors: a bs=1 decode hidden state, a 6-token MTP batch, a
    // 2048-token prefill chunk, and a bulk size for the asymptote.
    struct S { const char* what; size_t bytes; };
    const S sizes[] = {
        {"hidden bs=1   (1 x 2048 x bf16)",      1ull * 2048 * 2},
        {"hidden bs=6   (6 x 2048 x bf16)",      6ull * 2048 * 2},
        {"hidden bs=64  (64 x 2048 x bf16)",     64ull * 2048 * 2},
        {"prefill chunk (2048 x 2048 x bf16)",   2048ull * 2048 * 2},
        {"bulk 64 MB",                           64ull << 20},
    };

    printf("\n%-38s %12s %12s %12s\n", "transfer", "us", "GB/s", "us (D2H+H2D)");
    for (const S& s : sizes) {
        void *a = nullptr, *b = nullptr, *h = nullptr;
        OK(hipSetDevice(0)); OK(hipMalloc(&a, s.bytes));
        OK(hipSetDevice(1)); OK(hipMalloc(&b, s.bytes));
        OK(hipHostMalloc(&h, s.bytes));
        const int iters = s.bytes > (1u << 20) ? 50 : 500;

        OK(hipSetDevice(0));
        for (int i = 0; i < 10; ++i) OK(hipMemcpyPeer(b, 1, a, 0, s.bytes));
        OK(hipDeviceSynchronize());
        hipEvent_t e0, e1; OK(hipEventCreate(&e0)); OK(hipEventCreate(&e1));
        OK(hipEventRecord(e0));
        for (int i = 0; i < iters; ++i) OK(hipMemcpyPeer(b, 1, a, 0, s.bytes));
        OK(hipEventRecord(e1)); OK(hipEventSynchronize(e1));
        float ms = 0; OK(hipEventElapsedTime(&ms, e0, e1));
        const double us = ms * 1e3 / iters;

        // The explicit host round-trip, i.e. what you pay if P2P is unavailable and you stage it.
        OK(hipEventRecord(e0));
        for (int i = 0; i < iters; ++i) {
            OK(hipMemcpy(h, a, s.bytes, hipMemcpyDeviceToHost));
            OK(hipSetDevice(1)); OK(hipMemcpy(b, h, s.bytes, hipMemcpyHostToDevice));
            OK(hipSetDevice(0));
        }
        OK(hipEventRecord(e1)); OK(hipEventSynchronize(e1));
        float ms2 = 0; OK(hipEventElapsedTime(&ms2, e0, e1));

        printf("%-38s %12.1f %12.1f %12.1f\n", s.what, us,
               s.bytes / (us * 1e-6) / 1e9, ms2 * 1e3 / iters);
        OK(hipFree(a)); OK(hipSetDevice(1)); OK(hipFree(b)); OK(hipSetDevice(0)); OK(hipHostFree(h));
    }
    return 0;
}
