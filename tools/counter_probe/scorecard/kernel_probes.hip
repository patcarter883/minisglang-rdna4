// kernel_probes.hip — torch-free launcher for the decode kernels whose REGIME CLAIMS are under test.
//
// Covers, in one binary (mode selected by argv[1]):
//   bf16 : Bf16GemvLoader dense decode GEMV      -> "dense GEMV is at its floor"
//   moe1 : MoE gemm1 (GATHER + SILU)             -> "MoE decode is reduction-floor bound"
//   moe2 : MoE gemm2 (GATHER, down-projection)   -> same, and the gemm2-slower-than-gemm1 datum
//   dense: fp8 + int4 dense decode GEMV          -> "decode GEMV is at 81% of HBM"
//
// TORCH-FREE ON PURPOSE. These kernels ship as torch extensions, but a .so built against one image
// will not load in another, so profiling them in a DIFFERENT ROCm than we serve on would mean
// rebuilding the whole package there. This instantiates the same `gemv_decode_core<>` template body
// the torch launchers instantiate, so the only thing that differs from production is who calls it.
//
// FIDELITY RULES (each one decides whether the numbers mean anything):
//   1. PRODUCTION TILINGS, NOT CONVENIENT ONES. Every launch below reproduces the (NWARPS, MMAX,
//      COLS, BYLANE) and grid that the real launcher computes for that shape and M -- transcribed
//      from w8a8_dense_kernel.hip run_bf16_gemv and moe_kernel.hip run_moe_gemm1_silu/run_moe_gemm.
//      Equalising tilings across kernels would be a fairer-LOOKING and wronger experiment.
//   2. REAL ROUTING BUFFERS. The MoE core's cost is dominated by what the routing metadata makes it
//      do (which rows are real, which are padding). Feeding it zeros or an identity map would
//      measure a kernel that never runs in production -- padding rows are compacted and SKIPPED, so
//      a probe with all-real rows would overstate the work by ~16x at decode.
//   3. VALUES ARE IRRELEVANT, SHAPES AND METADATA ARE NOT. Weights/activations are memset patterns;
//      counters do not depend on the numerics. Routing metadata IS constructed correctly.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gemv_decode.h"

using OT = __hip_bfloat16;
using gemv_decode::gemv_decode_core;
using gemv_decode::Fp8DenseGemvLoader;
using gemv_decode::Int4Fp8GemvLoader;
using gemv_decode::Bf16GemvLoader;

#define HIP_CHECK(x) do { hipError_t _e=(x); if(_e!=hipSuccess){ \
  std::fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(_e),__LINE__); std::exit(1);} } while(0)

template <class T> static T* dalloc(size_t n, int fill) {
  T* p=nullptr; HIP_CHECK(hipMalloc(&p, n*sizeof(T)));
  HIP_CHECK(hipMemset(p, fill, n*sizeof(T)));
  return p;
}
template <class T> static T* dupload(const std::vector<T>& v) {
  T* p=nullptr; HIP_CHECK(hipMalloc(&p, v.size()*sizeof(T)));
  HIP_CHECK(hipMemcpy(p, v.data(), v.size()*sizeof(T), hipMemcpyHostToDevice));
  return p;
}

// -------------------------------------------------------------------------------------------
// Routing metadata, built to the contract in moe/moe_rocm/moe_kernels.hip (moe_align_kernel).
//
//   sorted_ids[P]  flat topk indices 0..numel-1, grouped into per-expert CONTIGUOUS RUNS, each run
//                  padded up to a multiple of block_m with the SENTINEL `numel` -- an out-of-range
//                  index the GEMM skips via `offs < num_valid_tokens`. That skip is the whole
//                  reason decode padding is free, so the sentinel must be exact.
//   expert_ids[nblk]        the expert owning each block, ascending.
//   num_tokens_post_pad[1]  sum over experts of round_up(count[e], block_m).
// -------------------------------------------------------------------------------------------
struct Routing {
  std::vector<int> sorted_ids, expert_ids, ntp;
  int P = 0, nblk = 0, numel = 0;
};

static Routing build_routing(int M, int top_k, int E, int block_m) {
  Routing r;
  r.numel = M * top_k;
  // Round-robin assignment of the M*top_k slots to experts. Decode routing is not uniform in
  // reality, but the counter-relevant property is the BLOCK COUNT and the real-rows-per-block
  // ratio, and round-robin reproduces the decode regime exactly: at M=1/top_k=8 that is 8 experts
  // holding 1 real row each in a block_m=16 block, i.e. the 1/16 occupancy the claim is about.
  std::vector<std::vector<int>> per_expert(E);
  for (int i = 0; i < r.numel; ++i) per_expert[i % E].push_back(i);

  for (int e = 0; e < E; ++e) {
    int cnt = (int)per_expert[e].size();
    if (cnt == 0) continue;
    int padded = ((cnt + block_m - 1) / block_m) * block_m;
    for (int i = 0; i < cnt; ++i) r.sorted_ids.push_back(per_expert[e][i]);
    for (int i = cnt; i < padded; ++i) r.sorted_ids.push_back(r.numel);   // SENTINEL
    for (int b = 0; b < padded / block_m; ++b) r.expert_ids.push_back(e);
  }
  r.P = (int)r.sorted_ids.size();
  r.nblk = (int)r.expert_ids.size();
  r.ntp = {r.P};
  return r;
}

// Compile-time tiling tags. These live at file scope because a LOCAL struct may not have static
// data members; each one names one (NWARPS, MMAX, COLS, BYLANE) combination the production
// launchers can select, so the dispatch below is a transcription of their `if` ladders.
template <int NW_, int CO_, bool BL_> struct BfTile {
  static constexpr int nw = NW_; static constexpr int co = CO_; static constexpr bool bl = BL_;
};
template <int NW_, int MM_> struct Moe1Tile {
  static constexpr int nw = NW_; static constexpr int mm = MM_;
};
template <int NW_, int MM_, int CO_> struct Moe2Tile {
  static constexpr int nw = NW_; static constexpr int mm = MM_; static constexpr int co = CO_;
};

// ===========================================================================================
// bf16 dense decode GEMV — transcribed from w8a8_dense::run_bf16_gemv (w8a8_dense_kernel.hip).
//
//   tl      = select_gemv_tiling(K, mreal_cap=16, group_size=32, k_per_lane_slot=8, bylane=true)
//   bylane  = tl.bylane && N >= 32*64          <- the N-aware correction the table cannot make
//   MMAX    = (M<=1) ? 1 : (M<=8) ? 8 : 16
//   grid    = ceil(N / (BYLANE ? NW*32*COLS : NW*COLS)),  block = NW*32
//
// mreal_cap is PINNED at 16 and does NOT vary with M: BYLANE and K-on-lanes reduce K in different
// orders, so the tiling must be a pure function of (N,K) or the same row differs by a ULP between
// M=1 and M=8 and spec-verify / chunked-prefill / radix losslessness breaks.
// ===========================================================================================
static void run_bf16(int M, int N, int K) {
  auto* x  = dalloc<OT>((size_t)M*K, 0x3c);
  auto* w  = dalloc<OT>((size_t)N*K, 0x3c);
  auto* o  = dalloc<OT>((size_t)M*N, 0);

  const bool bylane_pref = (K <= 1024);
  const bool bylane = bylane_pref && (N >= 32*64);
  const int  nw   = bylane ? 8 : (K <= 512 ? 8 : 16);
  const int  cols = bylane ? 1 : (K <= 1024 ? 2 : 1);   // mreal_cap=16 => cols_def 2 for K<=1024, else 1

  auto go = [&](auto tag) {
    constexpr int NW   = decltype(tag)::nw;
    constexpr int CO   = decltype(tag)::co;
    constexpr bool BL  = decltype(tag)::bl;
    constexpr int MM   = 1;                              // M=1 decode
    const int per = BL ? NW*32*CO : NW*CO;
    dim3 g((N + per - 1)/per), b(NW*32);
    std::printf("[bf16] M=%d N=%d K=%d bylane=%d nw=%d cols=%d grid=%d blk=%d\n",
                M,N,K,(int)BL,NW,CO,g.x,b.x);
    gemv_decode_core<OT, Bf16GemvLoader<OT>, false,false,false, NW,MM,CO, BL>
      <<<g,b>>>(reinterpret_cast<const unsigned char*>(x), nullptr, w, nullptr, nullptr,
                nullptr,nullptr,nullptr, o, nullptr,nullptr, M,N,K, 0,0,0,0,0, false);
  };
  auto launch = [&]{
    if (bylane)                       go(BfTile<8,1,true>{});
    else if (nw==8  && cols==2)       go(BfTile<8,2,false>{});
    else if (nw==16 && cols==2)       go(BfTile<16,2,false>{});
    else                              go(BfTile<16,1,false>{});
  };
  launch(); HIP_CHECK(hipDeviceSynchronize());          // warm: don't attribute lazy code-object load
  for (int i=0;i<3;++i) launch();
  HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipGetLastError());
  (void)hipFree(x); (void)hipFree(w); (void)hipFree(o);
}

// ===========================================================================================
// MoE gemm1 (gate_up + SiLU) — transcribed from run_moe_gemm1_silu, Gemv branch.
//   nw = (K<=1024) ? 8 : 32 ; COLS = 1 ; BYLANE = never
//   mreal_cap = min(block_m, T) ; MMAX = <=8?8 : <=16?16 : <=32?32 : 64
//   grid = dim3(ceil(inter/nw), P/block_m) ; block = nw*32
// N is the FULL gate_up width 2*inter; the grid spans `inter` because SILU pairs column j with j+inter.
// ===========================================================================================
static void run_moe1(int M, int E, int top_k, int inter, int K, int gs, int block_m) {
  Routing r = build_routing(M, top_k, E, block_m);
  const int N = 2*inter;
  auto* xf = dalloc<unsigned char>((size_t)M*K, 0x38);
  auto* as = dalloc<float>(M, 0);
  auto* wq = dalloc<int>((size_t)E*N*(K/8), 0x11);
  auto* ws = dalloc<__half>((size_t)E*N*(K/gs), 0x3c);
  auto* o  = dalloc<OT>((size_t)r.P*inter, 0);
  int* sti = dupload(r.sorted_ids);
  int* eid = dupload(r.expert_ids);
  int* ntp = dupload(r.ntp);

  const int nw = (K <= 1024) ? 8 : 32;
  const int mreal_cap = block_m < M ? block_m : M;
  auto go = [&](auto tag) {
    constexpr int NW = decltype(tag)::nw;
    constexpr int MM = decltype(tag)::mm;
    // Same guard as gemm2: MMAX must cover the REAL rows compacted into __shared__ int[MMAX].
    // For gemm1 that is min(block_m, rows routed to one expert), bounded by block_m.
    int real_max = 0;
    { std::vector<int> cnt(r.nblk,0);
      for (int b2=0;b2<r.nblk;++b2) for (int q=0;q<block_m;++q) {
        int idx=b2*block_m+q; if (idx<(int)r.sorted_ids.size() && r.sorted_ids[idx]<r.numel) cnt[b2]++; }
      for (int b2=0;b2<r.nblk;++b2) real_max = cnt[b2]>real_max?cnt[b2]:real_max; }
    if (MM < real_max) {
      std::fprintf(stderr, "[moe1] FATAL MMAX=%d < real rows/block=%d — would fault\n", MM, real_max);
      std::exit(1);
    }
    dim3 g((inter + NW - 1)/NW, r.P/block_m), b(NW*32);
    std::printf("[moe1] M=%d E=%d tk=%d inter=%d K=%d bm=%d P=%d nblk=%d nw=%d MMAX=%d grid=(%d,%d) blk=%d\n",
                M,E,top_k,inter,K,block_m,r.P,r.nblk,NW,MM,g.x,g.y,b.x);
    gemv_decode_core<OT, Int4Fp8GemvLoader, true,true,false, NW,MM,1>
      <<<g,b>>>(xf, as, wq, ws, nullptr, sti, eid, ntp, o, nullptr,nullptr,
                M, N, K, gs, top_k, block_m, r.numel, 0, false);
  };
  auto launch = [&]{
    if (nw==8) { if (mreal_cap<=8) go(Moe1Tile<8,8>{});  else go(Moe1Tile<8,16>{}); }
    else       { if (mreal_cap<=8) go(Moe1Tile<32,8>{}); else go(Moe1Tile<32,16>{}); }
  };
  launch(); HIP_CHECK(hipDeviceSynchronize());
  for (int i=0;i<3;++i) launch();
  HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipGetLastError());
  (void)hipFree(xf);(void)hipFree(as);(void)hipFree(wq);(void)hipFree(ws);(void)hipFree(o);
  (void)hipFree(sti);(void)hipFree(eid);(void)hipFree(ntp);
}

// ===========================================================================================
// MoE gemm2 (down projection) — transcribed from run_moe_gemm, Gemv branch, SCATTER=false.
//   nw   = (K<=1024) ? 8 : 32
//   cols = (K<=1024) ? (mreal_cap<=8 ? 4 : <=16 ? 2 : 1) : 1
//   grid = dim3(ceil(N/(nw*cols)), P/block_m) ; block = nw*32
// The served decode call uses top_k=1 with sorted_token_ids = arange(P): gemm2 consumes the ALREADY
// SORTED rows produced by gemm1, so a "token" here IS a padded row and s_src = offs/1 = offs.
// ===========================================================================================
static void run_moe2(int M, int E, int top_k, int inter, int N, int gs, int block_m) {
  Routing r = build_routing(M, top_k, E, block_m);
  const int K = inter;
  const int T = r.P;
  std::vector<int> ar(r.P); for (int i=0;i<r.P;++i) ar[i]=i;   // top_k=1 => identity row map

  auto* xf = dalloc<unsigned char>((size_t)T*K, 0x38);
  auto* as = dalloc<float>(T, 0);
  auto* wq = dalloc<int>((size_t)E*N*(K/8), 0x11);
  auto* ws = dalloc<__half>((size_t)E*N*(K/gs), 0x3c);
  auto* o  = dalloc<OT>((size_t)r.P*N, 0);
  int* sti = dupload(ar);
  int* eid = dupload(r.expert_ids);
  int* ntp = dupload(r.ntp);

  const int nw = (K <= 1024) ? 8 : 32;
  // mreal_cap = min(block_m, T) and for gemm2 T IS P, not M: gemm2 consumes the already-sorted rows
  // gemm1 produced, so every one of the block_m rows in a block is REAL (top_k=1, sorted=arange(P),
  // so `offs < num_valid_tokens` never rejects). Deriving mreal_cap from M instead gave MMAX=8 while
  // 16 real rows were compacted into the `__shared__ int[MMAX]` row arrays -> out-of-bounds shared
  // write -> memory fault. MMAX must cover the REAL rows per block, which here is all block_m of them.
  const int T_cap = block_m < T ? block_m : T;
  const int mreal_cap = T_cap;
  const int cols = (K <= 1024) ? (mreal_cap<=8 ? 4 : (mreal_cap<=16 ? 2 : 1)) : 1;
  auto go = [&](auto tag) {
    constexpr int NW = decltype(tag)::nw;
    constexpr int MM = decltype(tag)::mm;
    constexpr int CO = decltype(tag)::co;
    // Fail LOUDLY rather than faulting. The core compacts up to `block_m` real rows into
    // __shared__ int[MMAX]; if MMAX < real-rows-per-block it writes out of bounds and the only
    // symptom is a Memory Fault Error naming the kernel, which reads like a kernel bug rather than
    // a harness misconfiguration. This is the check that would have saved a GPU window.
    if (MM < T_cap) {
      std::fprintf(stderr, "[moe2] FATAL MMAX=%d < real rows/block=%d — would fault\n", MM, T_cap);
      std::exit(1);
    }
    dim3 g((N + NW*CO - 1)/(NW*CO), r.P/block_m), b(NW*32);
    std::printf("[moe2] M=%d E=%d tk=%d N=%d K=%d bm=%d P=%d nblk=%d nw=%d MMAX=%d cols=%d grid=(%d,%d) blk=%d\n",
                M,E,top_k,N,K,block_m,r.P,r.nblk,NW,MM,CO,g.x,g.y,b.x);
    gemv_decode_core<OT, Int4Fp8GemvLoader, true,false,false, NW,MM,CO>
      <<<g,b>>>(xf, as, wq, ws, nullptr, sti, eid, ntp, o, nullptr,nullptr,
                T, N, K, gs, 1, block_m, T, 0, false);
  };
  auto launch = [&]{
    if (cols==4)      go(Moe2Tile<8,8,4>{});
    else if (cols==2) go(Moe2Tile<8,16,2>{});
    else              go(Moe2Tile<8,32,1>{});
  };
  launch(); HIP_CHECK(hipDeviceSynchronize());
  for (int i=0;i<3;++i) launch();
  HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipGetLastError());
  (void)hipFree(xf);(void)hipFree(as);(void)hipFree(wq);(void)hipFree(ws);(void)hipFree(o);
  (void)hipFree(sti);(void)hipFree(eid);(void)hipFree(ntp);
}

// ===========================================================================================
// dense fp8 + int4 decode GEMV, production tilings (same as gemv_counters.hip).
// `stream` forces the weight working set ABOVE the 64 MB Infinity Cache so the measurement is true
// HBM traffic; the default cache-resident shape is kept as the control, because conflating the two
// is exactly how an inflated "% of HBM" gets published.
// ===========================================================================================
static void run_dense(int M, int N, int K, int gs) {
  auto* xf = dalloc<unsigned char>((size_t)M*K, 0x38);
  auto* as = dalloc<float>(M, 0);
  auto* o  = dalloc<OT>((size_t)M*N, 0);
  auto* w8 = dalloc<unsigned char>((size_t)N*K, 0x38);
  auto* s8 = dalloc<float>(N, 0);
  auto* w4 = dalloc<int>((size_t)N*(K/8), 0x11);
  auto* s4 = dalloc<__half>((size_t)N*(K/gs), 0x3c);

  constexpr int F_NW=8, F_MMAX=8, F_COLS=1;
  dim3 fg((N + F_NW - 1)/F_NW), fb(F_NW*32);
  constexpr int I_NW=16, I_MMAX=1, I_COLS=2;
  dim3 ig((N + I_NW*I_COLS - 1)/(I_NW*I_COLS)), ib(I_NW*32);
  std::printf("[dense] M=%d N=%d K=%d gs=%d | fp8 grid=%d blk=%d | int4 grid=%d blk=%d\n",
              M,N,K,gs,fg.x,fb.x,ig.x,ib.x);

  auto f8 = [&]{ gemv_decode_core<OT, Fp8DenseGemvLoader, false,false,false, F_NW,F_MMAX,F_COLS>
      <<<fg,fb>>>(xf, as, w8, s8, nullptr, nullptr,nullptr,nullptr, o, nullptr,nullptr,
                  M,N,K, 0,0,0,0,0, false); };
  auto i4 = [&]{ gemv_decode_core<OT, Int4Fp8GemvLoader, false,false,false, I_NW,I_MMAX,I_COLS>
      <<<ig,ib>>>(xf, as, w4, s4, nullptr, nullptr,nullptr,nullptr, o, nullptr,nullptr,
                  M,N,K, gs,0,0,0,0, false); };
  f8(); i4(); HIP_CHECK(hipDeviceSynchronize());
  for (int i=0;i<3;++i) f8();
  for (int i=0;i<3;++i) i4();
  HIP_CHECK(hipDeviceSynchronize()); HIP_CHECK(hipGetLastError());
  (void)hipFree(xf);(void)hipFree(as);(void)hipFree(o);(void)hipFree(w8);(void)hipFree(s8);(void)hipFree(w4);(void)hipFree(s4);
}

int main(int argc, char** argv) {
  std::string mode = (argc>1) ? argv[1] : "dense";
  auto A=[&](int i,int d){ return (argc>i)?std::atoi(argv[i]):d; };

  // multi_processor_count reports WGPs, not CUs: the 64-CU 9070 XT answers 32 and the 56-CU 9070
  // answers 28. Printed so every result carries which card it came from; never assert 64.
  hipDeviceProp_t p; HIP_CHECK(hipGetDeviceProperties(&p, 0));
  std::printf("[dev] %s gfx=%s WGP(multi_processor_count)=%d clk=%dkHz\n",
              p.name, p.gcnArchName, p.multiProcessorCount, p.clockRate);

  if      (mode=="bf16")  run_bf16(A(2,1), A(3,2048), A(4,256));
  else if (mode=="moe1")  run_moe1(A(2,1), A(3,32), A(4,8), A(5,512), A(6,2048), A(7,128), A(8,16));
  else if (mode=="moe2")  run_moe2(A(2,1), A(3,32), A(4,8), A(5,512), A(6,2048), A(7,128), A(8,16));
  else if (mode=="dense") run_dense(A(2,1), A(3,4096), A(4,4096), A(5,128));
  else { std::fprintf(stderr,"unknown mode %s\n", mode.c_str()); return 2; }
  return 0;
}
