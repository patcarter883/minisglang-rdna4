// cache_gap_probe.hip — does CACHE STATE explain the isolated-vs-in-serve gap?
//
// THE ANOMALY THIS EXISTS TO EXPLAIN. The same decode GEMV kernels, at the same shapes, read
// 55-81% of the 706.6 GB/s roofline in an isolated benchmark and 13.0-69.5% inside the serve. Nobody
// has explained the difference. The leading hypothesis: in a microbench the weights stay resident in
// the 64 MB MALL across back-to-back reps, while in the serve the ~2,800 elementwise/norm dispatches
// between two GEMMs each do a full HBM round-trip of the activation and EVICT that working set — so
// every served GEMM pays cold and the microbench never does.
//
// This probe puts the SAME kernel through three cache states and reports the same statistic for
// each, so the hypothesis is testable rather than plausible:
//
//   hot    one weight buffer, back-to-back reps.        <- the condition "81% of roofline" was taken at
//   rot    R copies of the weights, cycled, sized to
//          exceed the MALL by BYTES.                    <- every launch reads cold, no interference
//   evict  hot weights, but a flush kernel streams
//          FLUSH_MB between reps.                       <- reproduces what the serve does BETWEEN GEMMs
//
// `evict` sweeps FLUSH_MB. That sweep is the actual measurement: it says how many bytes of
// interleaved traffic are needed to move the isolated number onto the in-serve number. Comparing
// that answer with the elementwise byte volume MEASURED from the serve trace (analyze_ksweep.py
// prints it per step) is what turns a plausible story into a confirmed or refuted one.
//
// TORCH-FREE ON PURPOSE. These kernels ship as torch extensions and a .so built against one image
// will not load in another, so a torch harness is pinned to the ROCm 7.2.1 serve image where `--pmc`
// hangs. Instantiating the same `gemv_decode_core<>` template body directly makes the probe
// compilable under ROCm 7.14, which is the only stack where gfx1201 counters work at all.
//
// FIDELITY: the tilings and grids are transcribed from the production launchers exactly as
// tools/counter_probe/scorecard/kernel_probes.hip does — an equalised tiling would be a
// fairer-looking and wronger experiment. Values are memset patterns; counters and timings do not
// depend on the numerics.
//
//   hipcc -O3 --offload-arch=gfx1201 -I<kernels>/fp8_wmma/fp8_wmma_rocm -o cgp cache_gap_probe.hip
//   ./cgp <loader> <M> <N> <K> <cond> <flush_mb> <reps>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "gemv_decode.h"

using OT = __hip_bfloat16;
using gemv_decode::gemv_decode_core;
using gemv_decode::Fp8DenseGemvLoader;
using gemv_decode::Int4Fp8GemvLoader;
using gemv_decode::Bf16GemvLoader;

#define HIP_CHECK(x) do { hipError_t _e=(x); if(_e!=hipSuccess){ \
  std::fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(_e),__LINE__); std::exit(1);} } while(0)

// The ONLY roofline denominator this repo uses: 706.6 GB/s at the 1380 MHz memory OC. Never 640,
// never 644, and never a previously ACHIEVED bench figure — quoting an achieved throughput as the
// ceiling makes every kernel look more finished than it is.
static const double ROOFLINE_GBS = 706.6;

template <class T> static T* dalloc(size_t n, int fill) {
  T* p=nullptr; HIP_CHECK(hipMalloc(&p, n*sizeof(T)));
  HIP_CHECK(hipMemset(p, fill, n*sizeof(T)));
  return p;
}

// A deliberate cache-flusher. It READS `bytes` and writes a reduced value, so the read traffic
// actually displaces the MALL rather than being optimised away. Grid-strided so the flush itself is
// bandwidth-bound and does not become a launch-latency measurement.
__global__ void flush_kernel(const float4* __restrict__ src, float* __restrict__ sink, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n4; i += stride) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // Never actually true — keeps the loads live without adding a store per element.
  if (acc == 1.2345e-30f) sink[0] = acc;
}

struct Timing { double mean_ns, min_ns, p90_ns; };

// OUTER repeats of an INNER rep loop. The inner loop amortises launch latency; the outer repeats
// give a distribution. The same statistic is reported for every condition — taking a min for `hot`
// and a mean for `evict` would manufacture the very gap under test.
template <class F>
static Timing timeit(F&& body, int reps, int outer) {
  hipEvent_t a, b; HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i) body();                       // warm: never attribute code-object load
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<double> per;
  for (int o = 0; o < outer; ++o) {
    HIP_CHECK(hipEventRecord(a));
    for (int r = 0; r < reps; ++r) body();
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    per.push_back((double)ms * 1e6 / reps);
  }
  HIP_CHECK(hipEventDestroy(a)); HIP_CHECK(hipEventDestroy(b));
  std::sort(per.begin(), per.end());
  double sum = 0; for (double v : per) sum += v;
  return { sum / per.size(), per.front(), per[(size_t)(0.9 * (per.size() - 1))] };
}

enum class Loader { BF16, FP8, INT4 };

int main(int argc, char** argv) {
  auto S=[&](int i,const char* d){ return std::string((argc>i)?argv[i]:d); };
  auto A=[&](int i,int d){ return (argc>i)?std::atoi(argv[i]):d; };

  std::string ldr = S(1,"bf16");
  int M = A(2,1), N = A(3,32768), K = A(4,2048);
  std::string cond = S(5,"hot");
  int flush_mb = A(6,0);
  int reps = A(7,50), outer = A(8,7), gs = A(9,128);
  int rot_mb = A(10,192);                    // 3x the 64 MB MALL

  Loader L = (ldr=="fp8") ? Loader::FP8 : (ldr=="int4") ? Loader::INT4 : Loader::BF16;

  // multi_processor_count reports WGPs, not CUs: the 64-CU 9070 XT answers 32 and the 56-CU 9070
  // answers 28. Printed so every row carries which card produced it; never assert 64.
  hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p, 0));
  std::fprintf(stderr,"[dev] %s gfx=%s WGP=%d clk=%dkHz\n",
               p.name, p.gcnArchName, p.multiProcessorCount, p.clockRate);

  // Weight bytes per launch, by loader. This is the byte model the GB/s is computed from; the
  // activation and output are negligible at M=1 but are included so the number is honest.
  // fp8 carries PER-COLUMN fp32 scales (N floats) and int4 carries PER-GROUP fp16 scales
  // (N*K/gs halves) — transcribed from run_dense in kernel_probes.hip, not assumed symmetric.
  size_t wbytes = (L==Loader::BF16) ? (size_t)N*K*2
                : (L==Loader::FP8)  ? (size_t)N*K*1 + (size_t)N*4
                                    : (size_t)N*K/2 + (size_t)N*(K/gs)*2;
  size_t abytes = (size_t)M*K*((L==Loader::BF16)?2:1) + (size_t)M*N*2;
  size_t bytes  = wbytes + abytes;

  // How many weight copies to rotate through so the working set exceeds the MALL by BYTES. A
  // rotation counted in COPIES rather than bytes is the classic way to think you rotated and not
  // have: at these shapes 2 copies of a 12 MB weight still sit entirely inside 64 MB.
  int rot = 1;
  if (cond=="rot") rot = std::max(2, (int)(((size_t)rot_mb<<20) / std::max<size_t>(wbytes,1)) + 1);

  // ---- allocate ---------------------------------------------------------------------------------
  auto* x  = dalloc<OT>((size_t)M*K, 0x3c);
  auto* xf = dalloc<unsigned char>((size_t)M*K, 0x38);
  auto* as = dalloc<float>(std::max(M,1), 0);
  auto* o  = dalloc<OT>((size_t)M*N, 0);

  std::vector<void*> W(rot), WS(rot);
  for (int i=0;i<rot;++i) {
    if (L==Loader::BF16)      W[i] = dalloc<OT>((size_t)N*K, 0x3c);
    else if (L==Loader::FP8) { W[i] = dalloc<unsigned char>((size_t)N*K, 0x38);
                               WS[i]= dalloc<float>((size_t)N, 0); }        // per-COLUMN fp32
    else                     { W[i] = dalloc<int>((size_t)N*K/8, 0x11);
                               WS[i]= dalloc<__half>((size_t)N*(K/gs), 0x3c); }
  }

  size_t fn4 = ((size_t)flush_mb<<20) / sizeof(float4);
  float4* fbuf = nullptr; float* fsink = nullptr;
  if (cond=="evict" && flush_mb>0) {
    HIP_CHECK(hipMalloc(&fbuf, fn4*sizeof(float4)));
    HIP_CHECK(hipMemset(fbuf, 0x11, fn4*sizeof(float4)));
    fsink = dalloc<float>(1,0);
  }

  // ---- the production tiling, transcribed ---------------------------------------------------------
  // bf16: from w8a8_dense::run_bf16_gemv. mreal_cap is PINNED at 16 and does NOT vary with M — BYLANE
  // and K-on-lanes reduce K in different orders, so the tiling must be a pure function of (N,K) or
  // the same row differs by a ULP between M=1 and M=8 and spec-verify losslessness breaks.
  const bool bylane = (K <= 1024) && (N >= 32*64);
  const int  nw_bf  = bylane ? 8 : (K <= 512 ? 8 : 16);
  const int  co_bf  = bylane ? 1 : (K <= 1024 ? 2 : 1);
  // fp8/int4 DENSE decode tilings, transcribed from run_dense (kernel_probes.hip) — the fp8 and int4
  // arms do NOT share a tiling, and using the MoE tiling here (nw = K<=1024 ? 8 : 32) would time a
  // kernel the dense path never launches.
  const int  nw_f8 = 8,  mm_f8 = 8, co_f8 = 1;      // grid = ceil(N / 8)
  const int  nw_i4 = 16, mm_i4 = 1, co_i4 = 2;      // grid = ceil(N / 32)

  int idx = 0;
  auto one = [&]() {
    void* w  = W[idx % rot];
    void* ws = WS.empty() ? nullptr : WS[idx % rot];
    ++idx;
    if (L==Loader::BF16) {
      const int per = bylane ? nw_bf*32*co_bf : nw_bf*co_bf;
      dim3 g((N + per - 1)/per), b(nw_bf*32);
      if (bylane)
        gemv_decode_core<OT, Bf16GemvLoader<OT>, false,false,false, 8,1,1, true>
          <<<g,b>>>((const unsigned char*)x, nullptr, (OT*)w, nullptr,nullptr,
                    nullptr,nullptr,nullptr, o, nullptr,nullptr, M,N,K, 0,0,0,0,0, false);
      else if (nw_bf==16 && co_bf==1)
        gemv_decode_core<OT, Bf16GemvLoader<OT>, false,false,false, 16,1,1, false>
          <<<g,b>>>((const unsigned char*)x, nullptr, (OT*)w, nullptr,nullptr,
                    nullptr,nullptr,nullptr, o, nullptr,nullptr, M,N,K, 0,0,0,0,0, false);
      else
        gemv_decode_core<OT, Bf16GemvLoader<OT>, false,false,false, 16,1,2, false>
          <<<g,b>>>((const unsigned char*)x, nullptr, (OT*)w, nullptr,nullptr,
                    nullptr,nullptr,nullptr, o, nullptr,nullptr, M,N,K, 0,0,0,0,0, false);
    } else if (L==Loader::FP8) {
      // gs is passed as 0 for fp8: its scales are per-COLUMN, not per-group.
      dim3 g((N + nw_f8*co_f8 - 1)/(nw_f8*co_f8)), b(nw_f8*32);
      gemv_decode_core<OT, Fp8DenseGemvLoader, false,false,false, nw_f8,mm_f8,co_f8>
        <<<g,b>>>(xf, as, (unsigned char*)w, (float*)ws, nullptr,
                  nullptr,nullptr,nullptr, o, nullptr,nullptr, M,N,K, 0,0,0,0,0, false);
    } else {
      dim3 g((N + nw_i4*co_i4 - 1)/(nw_i4*co_i4)), b(nw_i4*32);
      gemv_decode_core<OT, Int4Fp8GemvLoader, false,false,false, nw_i4,mm_i4,co_i4>
        <<<g,b>>>(xf, as, (int*)w, (__half*)ws, nullptr,
                  nullptr,nullptr,nullptr, o, nullptr,nullptr, M,N,K, gs,0,0,0,0, false);
    }
  };

  auto body = [&]() {
    one();
    if (cond=="evict" && flush_mb>0)
      flush_kernel<<<2048, 256>>>(fbuf, fsink, fn4);
  };

  Timing t = timeit(body, reps, outer);
  HIP_CHECK(hipGetLastError());

  // The flush launch is INSIDE the timed body, so subtract its own cost to report the GEMV's time.
  // Measured separately rather than modelled — the flush is bandwidth-bound and its cost is not a
  // constant across sizes.
  double flush_ns = 0;
  if (cond=="evict" && flush_mb>0) {
    Timing tf = timeit([&]{ flush_kernel<<<2048,256>>>(fbuf, fsink, fn4); }, reps, outer);
    flush_ns = tf.mean_ns;
  }
  double gemv_ns = t.mean_ns - flush_ns;
  double gbs = (double)bytes / gemv_ns;                  // B/ns == GB/s
  std::printf("CSV,%s,%d,%d,%d,%s,%d,%d,%.1f,%.1f,%.1f,%.1f,%zu,%.1f,%.1f\n",
              ldr.c_str(), M, N, K, cond.c_str(), flush_mb, rot,
              t.mean_ns, t.min_ns, flush_ns, gemv_ns, bytes, gbs, 100.0*gbs/ROOFLINE_GBS);
  std::fprintf(stderr,
      "[%s] %s M=%d N=%d K=%d flush=%dMB rot=%d  total %.0f ns  flush %.0f ns  "
      "GEMV %.0f ns  %.1f GB/s = %.1f%% of %.1f\n",
      cond.c_str(), ldr.c_str(), M,N,K, flush_mb, rot,
      t.mean_ns, flush_ns, gemv_ns, gbs, 100.0*gbs/ROOFLINE_GBS, ROOFLINE_GBS);
  return 0;
}
