// Native HIP moe_align_block_size for gfx1201 (RDNA4) — Triton-free, no vLLM dependency.
//
// W4A8 grouped-MoE needs the routed tokens grouped per expert and each expert's run padded to a
// multiple of block_size, so the grouped GEMM can process whole [block_size]-row tiles per expert.
// This replaces the `from vllm... import moe_align_block_size` host op on the decode path.
//
// OUTPUT CONTRACT (bit-compatible with vLLM moe_align_block_size, pad_sorted_ids=True), the form the
// w4a8_fp8_wmma grouped GEMM consumes:
//   sorted_ids[P]      : flat topk indices (0..numel-1) grouped into per-expert contiguous runs, each
//                        run padded UP to a multiple of block_size with the SENTINEL = numel (an
//                        out-of-range token the GEMM skips). P = max_num_tokens_padded.
//   expert_ids[nblk]   : the expert index owning each block (ascending by expert; nblk=cdiv(P,bs)).
//   num_tokens_post_pad: Σ_e round_up(count[e], block_size).
// Token ORDER within an expert's run is arbitrary (the downstream gather/scatter-reduce sums per
// ORIGINAL token, so the final MoE output is invariant to intra-expert order) — we only guarantee a
// VALID alignment, validated for equivalence against the vLLM reference.
//
// Single workgroup, single launch: at decode (numel = M*top_k is tiny) the path is launch-bound, so
// one kernel beats a multi-launch count/scan/scatter. Shared cnt[E]+off[E] (E up to 4096 -> 32 KB).

#include <hip/hip_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace {

constexpr int THREADS = 256;

// One block. s[] holds cnt[E] (token count, then reused as the scatter running-counter) and off[E]
// (each expert's start offset in sorted_ids). numel small at decode; the serial scan over E and the
// block-fill over nblk run on thread 0 (both O(E + nblk), negligible).
__global__ void moe_align_kernel(const int* __restrict__ topk_ids, int* __restrict__ sorted_ids,
                                 int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_pad,
                                 const int numel, const int num_experts, const int block_size,
                                 const int P, const int num_blocks) {
    extern __shared__ int s[];
    int* cnt = s;                       // [num_experts]
    int* off = s + num_experts;         // [num_experts]
    const int tid = threadIdx.x, T = blockDim.x;

    // sentinel-fill all of sorted_ids (padding slots stay = numel)
    for (int i = tid; i < P; i += T) sorted_ids[i] = numel;
    for (int e = tid; e < num_experts; e += T) cnt[e] = 0;
    __syncthreads();

    // count tokens per expert
    for (int t = tid; t < numel; t += T) atomicAdd(&cnt[topk_ids[t]], 1);
    __syncthreads();

    // exclusive prefix offsets (in tokens) + per-block expert ids + total padded count
    if (tid == 0) {
        int tok = 0, blk = 0;
        for (int e = 0; e < num_experts; e++) {
            off[e] = tok;
            const int padded = ((cnt[e] + block_size - 1) / block_size) * block_size;
            const int nblk = padded / block_size;
            for (int b = 0; b < nblk; b++) expert_ids[blk + b] = e;
            blk += nblk;
            tok += padded;
        }
        *num_tokens_post_pad = tok;
        for (int b = blk; b < num_blocks; b++) expert_ids[b] = 0;  // unused tail (GEMM bounds by ntp)
    }
    __syncthreads();

    // reuse cnt as the per-expert running counter for the scatter
    for (int e = tid; e < num_experts; e += T) cnt[e] = 0;
    __syncthreads();

    // scatter each flat token into its expert's run (order within a run is arbitrary)
    for (int t = tid; t < numel; t += T) {
        const int e = topk_ids[t];
        const int r = atomicAdd(&cnt[e], 1);
        sorted_ids[off[e] + r] = t;
    }
}

}  // namespace

void launch_moe_align(const at::Tensor& topk_ids, int64_t num_experts, int64_t block_size,
                      at::Tensor& sorted_ids, at::Tensor& expert_ids, at::Tensor& ntp) {
    const int numel = (int)topk_ids.numel();
    const int P = (int)sorted_ids.size(0);
    const int num_blocks = (int)expert_ids.size(0);
    const size_t shmem = 2 * (size_t)num_experts * sizeof(int);
    auto stream = at::cuda::getCurrentCUDAStream();
    moe_align_kernel<<<1, THREADS, shmem, stream>>>(
        topk_ids.data_ptr<int>(), sorted_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
        ntp.data_ptr<int>(), numel, (int)num_experts, (int)block_size, P, num_blocks);
}
