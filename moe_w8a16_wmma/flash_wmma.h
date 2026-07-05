// flash_wmma.h — shared rocWMMA bf16 flash-attention spine for gfx1201 (RDNA4).
//
// The attention family (attn_hip, attn_prefill_paged, mla_prefill, ...) each re-implements the same
// two rocWMMA matmuls: QK^T scores and P@V accumulation, both stored to fp32 smem for the row-wise
// online softmax (the softmax/rescale math stays in each kernel — it is intentionally decoupled from
// the WMMA fragments). This header factors ONLY those two matmuls, as `__forceinline__` device
// functions that reproduce the inline rocWMMA call sequence verbatim, so the emitted WMMA is
// byte-identical to the hand-written spine (proven via A/B torch.equal). This is the rocWMMA-side
// analog of what w4a8_tile::WmmaFp8 + TileConfig did for the fp8 GEMM family.
//
// Convention: smem operands are addressed as flat pointers with an explicit leading dimension `ld`,
// i.e. element [r][c] lives at base + r*ld + c — matching `bf16_t smem[ROWS][ld]` indexing exactly.

#pragma once

#include <rocwmma/rocwmma.hpp>

namespace flash_wmma {

using bf16_t = rocwmma::bfloat16_t;

constexpr int WM = 16, WN = 16, WK = 16;

// QK^T operands: A = Q rows (matrix_a, row_major), B = K rows read as K^T (matrix_b, col_major).
using FragQ   = rocwmma::fragment<rocwmma::matrix_a, WM, WN, WK, bf16_t, rocwmma::row_major>;
using FragKt  = rocwmma::fragment<rocwmma::matrix_b, WM, WN, WK, bf16_t, rocwmma::col_major>;
// P@V operands: A = P (matrix_a, row_major), B = V rows (matrix_b, row_major).
using FragP   = rocwmma::fragment<rocwmma::matrix_a, WM, WN, WK, bf16_t, rocwmma::row_major>;
using FragV   = rocwmma::fragment<rocwmma::matrix_b, WM, WN, WK, bf16_t, rocwmma::row_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, WM, WN, WK, float>;

// QK^T scores for one query m-tile (owned by the calling warp):
//   for each n-tile: S[m,n] = sum_{ks<KSTEPS} Q[m,ks] . K[n,ks]   -> stored row-major to sS[m*16][n*16].
// sQ/sK are bf16 with leading dim ldQ/ldK; sS is fp32 with leading dim ldS. `m` is the m-tile index.
template <int N_TILES, int KSTEPS>
__device__ __forceinline__ void qk_scores(const bf16_t* sQ, int ldQ, const bf16_t* sK, int ldK,
                                          float* sS, int ldS, int m) {
    for (int nt = 0; nt < N_TILES; nt++) {
        FragAcc acc;
        rocwmma::fill_fragment(acc, 0.0f);
        for (int ks = 0; ks < KSTEPS; ks++) {
            FragQ qf;
            FragKt kf;
            rocwmma::load_matrix_sync(qf, sQ + (m * WM) * ldQ + ks * WK, ldQ);
            rocwmma::load_matrix_sync(kf, sK + (nt * WN) * ldK + ks * WK, ldK);
            rocwmma::mma_sync(acc, qf, kf, acc);
        }
        rocwmma::store_matrix_sync(sS + (m * WM) * ldS + nt * WN, acc, ldS, rocwmma::mem_row_major);
    }
}

// P@V accumulation for one query m-tile (owned by the calling warp):
//   for each d-tile: O[m,d] = sum_{nt<N_TILES} P[m,nt] . V[nt,d]  -> stored row-major to sPV[m*16][d*16].
// sP/sV are bf16 with leading dim ldP/ldV; sPV is fp32 with leading dim ldPV. `m` is the m-tile index.
template <int D_TILES, int N_TILES>
__device__ __forceinline__ void pv_accumulate(const bf16_t* sP, int ldP, const bf16_t* sV, int ldV,
                                              float* sPV, int ldPV, int m) {
    for (int dt = 0; dt < D_TILES; dt++) {
        FragAcc acc;
        rocwmma::fill_fragment(acc, 0.0f);
        for (int nt = 0; nt < N_TILES; nt++) {
            FragP pf;
            FragV vf;
            rocwmma::load_matrix_sync(pf, sP + (m * WM) * ldP + nt * WN, ldP);
            rocwmma::load_matrix_sync(vf, sV + (nt * WN) * ldV + dt * WN, ldV);
            rocwmma::mma_sync(acc, pf, vf, acc);
        }
        rocwmma::store_matrix_sync(sPV + (m * WM) * ldPV + dt * WN, acc, ldPV, rocwmma::mem_row_major);
    }
}

}  // namespace flash_wmma
