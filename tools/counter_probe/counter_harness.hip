// counter_harness.hip — the harness you profile for COUNTERS. No python. No torch. On purpose.
//
// WHY THIS EXISTS
// ---------------
// rocprofiler-sdk processes every code object as it loads, and that work is single-threaded. PyTorch's
// ROCm build ships enormous fat binaries carrying tens of thousands of kernel symbols, so a counter run
// over ANY torch process pays that cost before your first real dispatch. Measured on this box: a probe
// doing 10 trivial matmuls under `--pmc` had not finished after 15 minutes; the matmuls were
// microseconds of it. roctxProfilerPause cannot rescue this — gating stops RECORDING, but the
// interception machinery stays armed for every dispatch and every code-object load, so it shrinks your
// output files and post-processing while doing nothing about startup. Different bottleneck.
//
// This binary loads ONE small code object: your kernel. Profiling it is near-instant, which is the
// entire point — a counter profile you can run in seconds is a counter profile you will actually use
// while iterating.
//
// BUILD + RUN (never hand-roll docker run; the compose `run` service injects the lease devices):
//   RDNA4_HIP_KERNELS=<your worktree> MINISGL_CMD='bash /kernels/opt_loop/bin/build_counter_harness.sh' \
//     docker compose -f /home/pat/code/minisgl-rdna4/docker-compose.yml --profile run run --rm run
//   gpu-lease -n 2 -- env RDNA4_HIP_KERNELS=<worktree> \
//     MINISGL_CMD='rocprofv3 --pmc SQ_WAVES -f csv -d /tmp/c -- /tmp/counter_harness' \
//     docker compose -f ... --profile run run --rm run
//
// ADAPTING IT TO YOUR KERNEL
// --------------------------
// Replace the demo kernel + launch below with yours. Two rules that keep this honest:
//   1. SHAPES MUST BE THE CAPTURED ONES. Counters are shape-sensitive; a harness at a convenient shape
//      measures a kernel nobody runs. Take the shapes from a trace of the real serve (WHAT=hotspots),
//      then replay exactly those here.
//   2. VALUES DO NOT MATTER, SHAPES DO. Fill buffers with a fixed cheap pattern. Do NOT pull in a
//      random-number library or a data loader "for realism" — you are counting hardware events, and
//      every extra dependency is another code object the profiler must chew through.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(x)                                                                    \
  do {                                                                                  \
    hipError_t _e = (x);                                                                \
    if (_e != hipSuccess) {                                                             \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),             \
                   __FILE__, __LINE__);                                                  \
      std::exit(1);                                                                     \
    }                                                                                   \
  } while (0)

// ---------------------------------------------------------------------------------------------
// DEMO KERNEL — replace with the kernel under study.
// Deliberately trivial: this file's job is to prove the harness mechanism (one small code object ->
// fast profiling), not to be a benchmark. A real target #includes its package header instead.
// ---------------------------------------------------------------------------------------------
__global__ void saxpy(const float* __restrict__ x, float* __restrict__ y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char** argv) {
  // ITERS: how many times the kernel is launched. Keep it SMALL. Counter collection serializes
  // dispatches, and every extra launch is another serialized, instrumented dispatch — 3 is usually
  // enough to read a counter, and 1000 just multiplies the cost you came here to avoid.
  const int iters = (argc > 1) ? std::atoi(argv[1]) : 3;
  const int n = (argc > 2) ? std::atoi(argv[2]) : (1 << 20);

  float *dx = nullptr, *dy = nullptr;
  HIP_CHECK(hipMalloc(&dx, sizeof(float) * n));
  HIP_CHECK(hipMalloc(&dy, sizeof(float) * n));

  // Fixed host pattern -> no RNG dependency, no extra code objects. Values are irrelevant to counters.
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i & 1023) * 0.001f;
  HIP_CHECK(hipMemcpy(dx, h.data(), sizeof(float) * n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dy, h.data(), sizeof(float) * n, hipMemcpyHostToDevice));

  const int block = 256;
  const int grid = (n + block - 1) / block;

  // One untimed warmup so the first dispatch's lazy setup is not what you measure.
  hipLaunchKernelGGL(saxpy, dim3(grid), dim3(block), 0, 0, dx, dy, 2.0f, n);
  HIP_CHECK(hipDeviceSynchronize());

  for (int it = 0; it < iters; ++it) {
    hipLaunchKernelGGL(saxpy, dim3(grid), dim3(block), 0, 0, dx, dy, 2.0f, n);
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::printf("[counter_harness] %d launches, n=%d — done\n", iters, n);
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
  return 0;
}
