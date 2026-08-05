// gemv_counters.hip — measure the VALU/VMEM/LDS breakdown of the int4 vs fp8 DECODE GEMV.
//
// WHY: "int4 decode GEMV is VALU-bound" has been the standing regime call for this repo, and it is an
// INFERENCE — from an ISA count (4950 single-issue v_cvt_f32_ubyte in the int4 path, zero in fp8) plus
// a ledger of failed optimisation attempts. It was never measured, because rocprofv3 --pmc was believed
// to hang on gfx1201. It does not hang on ROCm 7.14 (see probe_counters.sh). This harness is what turns
// that inference into a measurement.
//
// TORCH-FREE ON PURPOSE. The kernels are torch extensions, but a .so built against one image will not
// load in another, so profiling them in a DIFFERENT ROCm than we serve on would mean rebuilding the
// whole package there. Instead this instantiates gemv_decode_core<> directly — the same template body
// the torch launcher instantiates — so the only thing that differs from production is who calls it.
//
// FIDELITY RULES (both were nearly got wrong; they decide whether the numbers mean anything):
//   1. SAME SHAPE, BOTH KERNELS. Counters are shape-sensitive. A comparison at two different shapes
//      measures the shapes, not the loaders.
//   2. PRODUCTION TILINGS, NOT CONVENIENT ONES. Each path gets the (NWARPS, MMAX, COLS) and grid its
//      real launcher computes at decode M=1 — fp8: nw=8/MMAX=8/COLS=1 (w8a8_dense_kernel.hip run_gemv);
//      int4: nw=16/MMAX=1/COLS=2 (w4a8_fp8_wmma_kernel.hip DEC_MMAX, VLLM_W4A8_DECODE_GEMV_NW def 16,
//      cols=2 at M<=4). Equalising the tilings would be a fairer-looking and WRONGER experiment: COLS=2
//      is the int4 path's ILP mitigation for exactly the issue-bound behaviour under test.
//
// BUILD+RUN: see probe_counters.sh / gemv_counters.sh (needs the kernels repo mounted for gemv_decode.h).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gemv_decode.h"

using OT = __hip_bfloat16;
using gemv_decode::gemv_decode_core;
using gemv_decode::Fp8DenseGemvLoader;
using gemv_decode::Int4Fp8GemvLoader;

#define HIP_CHECK(x) do { hipError_t _e=(x); if(_e!=hipSuccess){ \
  std::fprintf(stderr,"HIP %s @%d\n",hipGetErrorString(_e),__LINE__); std::exit(1);} } while(0)

template <class T> static T* dalloc(size_t n, int fill) {
  T* p=nullptr; HIP_CHECK(hipMalloc(&p, n*sizeof(T)));
  HIP_CHECK(hipMemset(p, fill, n*sizeof(T)));   // values are irrelevant to counters; shapes are not
  return p;
}

int main(int argc, char** argv) {
  // Decode shape. Default N=K=4096 is the shape the int4 COLS=2 tuning note quotes (44->53% HBM),
  // so a regime call made here is directly comparable to the number that motivated the claim.
  const int M     = (argc>1)?std::atoi(argv[1]):1;
  const int N     = (argc>2)?std::atoi(argv[2]):4096;
  const int K     = (argc>3)?std::atoi(argv[3]):4096;
  const int GS    = (argc>4)?std::atoi(argv[4]):128;
  const int iters = (argc>5)?std::atoi(argv[5]):3;   // keep SMALL: --pmc serializes dispatches

  auto* x_fp8 = dalloc<unsigned char>((size_t)M*K, 0x38);
  auto* ascal = dalloc<float>(M, 0);
  auto* out   = dalloc<OT>((size_t)M*N, 0);

  // fp8 dense weights: raw e4m3 [N,K] + per-output-channel float scale.
  auto* w_fp8 = dalloc<unsigned char>((size_t)N*K, 0x38);
  auto* s_fp8 = dalloc<float>(N, 0);

  // int4 W4A8 weights: PACK_FACTOR=8 nibbles per int32 [N,K/8] + per-group fp16 scale [N,K/GS].
  auto* w_i4  = dalloc<int>((size_t)N*(K/8), 0x11);
  auto* s_i4  = dalloc<__half>((size_t)N*(K/GS), 0x3c);

  // ---- fp8 dense decode GEMV: run_gemv() in w8a8_dense_kernel.hip ----
  constexpr int F_NW=8, F_MMAX=8, F_COLS=1;
  dim3 f_grid((N + F_NW - 1) / F_NW), f_block(F_NW*32);
  auto launch_fp8 = [&]{
    gemv_decode_core<OT, Fp8DenseGemvLoader, false,false,false, F_NW,F_MMAX,F_COLS>
      <<<f_grid,f_block>>>(x_fp8, ascal, w_fp8, s_fp8, nullptr, nullptr,nullptr,nullptr,
                           out, nullptr,nullptr, M,N,K, 0,0,0,0,0, false); };

  // ---- int4 W4A8 decode GEMV: DEC_MMAX/DEC_COLS in w4a8_fp8_wmma_kernel.hip (non-magic path) ----
  constexpr int I_NW=16, I_MMAX=1, I_COLS=2;
  dim3 i_grid((N + I_NW*I_COLS - 1) / (I_NW*I_COLS)), i_block(I_NW*32);
  auto launch_i4 = [&]{
    gemv_decode_core<OT, Int4Fp8GemvLoader, false,false,false, I_NW,I_MMAX,I_COLS>
      <<<i_grid,i_block>>>(x_fp8, ascal, w_i4, s_i4, nullptr, nullptr,nullptr,nullptr,
                           out, nullptr,nullptr, M,N,K, GS,0,0,0,0, false); };

  // Warm up BOTH before the counted region so lazy code-object load is not what gets attributed.
  launch_fp8(); launch_i4(); HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());

  for (int i=0;i<iters;++i) launch_fp8();
  for (int i=0;i<iters;++i) launch_i4();
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());

  std::printf("[gemv_counters] M=%d N=%d K=%d gs=%d iters=%d | fp8 grid=%d blk=%d | int4 grid=%d blk=%d\n",
              M,N,K,GS,iters, f_grid.x, f_block.x, i_grid.x, i_block.x);
  return 0;
}
