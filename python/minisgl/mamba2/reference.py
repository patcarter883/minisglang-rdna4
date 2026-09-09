"""Mamba-2 (SSD) recurrence — the REFERENCE implementation the HIP kernels are graded against.

Phase 1 of docs/NEMOTRON35_LIGHTNING_PLAN.md. Pure torch, CPU-runnable, no HIP, no engine. Three
implementations of the same recurrence, in increasing order of how much they resemble a kernel:

    mamba2_sequential   one token at a time, the definition. Obviously correct, uselessly slow.
    mamba2_chunked      the chunked "state-space duality" form the kernel will implement.
    mamba2_decode       a single decode step against a carried state.

The point of having all three is that the chunked form is where the bugs live. It is not a
transcription of the sequential form — it re-associates the recurrence into a
(diagonal block) + (state passing) decomposition so the inner work becomes two GEMMs, and every
term in that decomposition is an opportunity to drop a decay factor or an off-by-one in the
segment sum. Establishing `chunked == sequential` in float64 on the HOST, before any HIP is
written, is what makes a later kernel mismatch mean "the kernel is wrong" instead of
"one of these two things is wrong".

THE RECURRENCE (Nemotron-H / Mamba-2, scalar-per-head A):

    dt = clamp(softplus(dt_raw + dt_bias), t_min, t_max)      [L, H]
    A  = -exp(A_log)                                          [H]        one scalar per head
    S  <- exp(dt*A) * S  +  dt * (x .outer. B)                [H, P, N]  P=head_dim, N=state
    y  =  S @ C  +  D * x                                     [H, P]

B and C are shared across heads in GROUPS: `n_groups` of them, each serving H/n_groups heads
(64 heads / 8 groups = 8 heads per group here). That grouping is the reason B/C are indexed
`g = h // (H // n_groups)` everywhere below and never `h`.

NOT in scope here: the causal conv1d in front (`gdn_hip.causal_conv1d_*` already computes it — same
op, only `conv_dim` differs), the gated RMSNorm behind it (`gdn_hip.rmsnorm_gated`, identical), and
the in_proj/out_proj GEMMs. This file is only the part minisgl does not already have.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "discretize_dt",
    "mamba2_sequential",
    "mamba2_chunked",
    "mamba2_decode",
    "segment_sum",
]


def discretize_dt(
    dt_raw: torch.Tensor,
    dt_bias: torch.Tensor,
    time_step_min: float = 0.001,
    time_step_max: float = 0.1,
) -> torch.Tensor:
    """`clamp(softplus(dt + dt_bias), t_min, t_max)` — Nemotron-H ships t_min/t_max as
    `time_step_min`/`time_step_max` (0.001 / 0.1) rather than the `time_step_limit` tuple other
    Mamba-2 configs use, so the caller passes floats and this stays config-shape agnostic."""
    return torch.clamp(F.softplus(dt_raw + dt_bias), min=time_step_min, max=time_step_max)


def _group_expand(t: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[..., G, N] -> [..., H, N] by repeating each group across the H/G heads it serves.

    `repeat_interleave` on the GROUP axis, not `repeat`: head h belongs to group h // (H//G), so the
    groups must expand in blocks (g,g,g,g,...) and NOT cycle (g0,g1,...,g0,g1,...). Getting this
    backwards is silent — the shapes agree and the numbers are wrong."""
    g = t.shape[-2]
    assert num_heads % g == 0, f"num_heads {num_heads} not divisible by n_groups {g}"
    return t.repeat_interleave(num_heads // g, dim=-2)


def mamba2_sequential(
    x: torch.Tensor,          # [L, H, P]  post-conv, post-silu
    dt: torch.Tensor,         # [L, H]     ALREADY discretized
    A_log: torch.Tensor,      # [H]
    B: torch.Tensor,          # [L, G, N]
    C: torch.Tensor,          # [L, G, N]
    D: torch.Tensor,          # [H]
    initial_state: Optional[torch.Tensor] = None,   # [H, P, N]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The definition, one token at a time. Returns (y [L,H,P], final_state [H,P,N])."""
    L, H, P = x.shape
    N = B.shape[-1]
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    A = -torch.exp(A_log.to(dtype))                          # [H]
    Bx = _group_expand(B.to(dtype), H)                       # [L, H, N]
    Cx = _group_expand(C.to(dtype), H)                       # [L, H, N]
    xf, dtf, Df = x.to(dtype), dt.to(dtype), D.to(dtype)

    S = (torch.zeros(H, P, N, dtype=dtype, device=x.device)
         if initial_state is None else initial_state.to(dtype).clone())
    y = torch.empty(L, H, P, dtype=dtype, device=x.device)
    for t in range(L):
        decay = torch.exp(dtf[t] * A)                        # [H]
        # outer product per head: [H,P,1] * [H,1,N]
        upd = (dtf[t].unsqueeze(-1) * xf[t]).unsqueeze(-1) * Bx[t].unsqueeze(-2)
        S = decay.view(H, 1, 1) * S + upd
        y[t] = torch.einsum("hpn,hn->hp", S, Cx[t]) + Df.unsqueeze(-1) * xf[t]
    return y, S


def segment_sum(t: torch.Tensor) -> torch.Tensor:
    """Lower-triangular cumulative sums: out[..., i, j] = sum(t[..., j+1 : i+1]) for j <= i, else -inf.

    This is the log-domain decay between two positions in a chunk: `exp(out[i,j])` is the product of
    every decay factor applied to a contribution entering at j by the time it is read at i. The
    strictly-lower mask before the cumsum and the inclusive mask after it are two DIFFERENT masks
    and swapping them shifts every decay by one position."""
    T = t.shape[-1]
    # [..., T(i), T(j)] holding t[**i**], not t[j]: the cumsum below runs over i, so each column j
    # must accumulate the decays of the steps AFTER j. Broadcasting t along j (rather than along i)
    # is the whole trick, and getting it backwards yields t[j]*(i-j) — a plausible-looking matrix
    # that is not a segment sum.
    tt = t.unsqueeze(-1).expand(*t.shape, T)
    strict = torch.tril(torch.ones(T, T, dtype=torch.bool, device=t.device), diagonal=-1)
    tt = tt.masked_fill(~strict, 0)
    out = torch.cumsum(tt, dim=-2)
    incl = torch.tril(torch.ones(T, T, dtype=torch.bool, device=t.device), diagonal=0)
    return out.masked_fill(~incl, -torch.inf)


def mamba2_chunked(
    x: torch.Tensor,          # [L, H, P]
    dt: torch.Tensor,         # [L, H]  ALREADY discretized
    A_log: torch.Tensor,      # [H]
    B: torch.Tensor,          # [L, G, N]
    C: torch.Tensor,          # [L, G, N]
    D: torch.Tensor,          # [H]
    chunk_size: int = 128,
    initial_state: Optional[torch.Tensor] = None,   # [H, P, N]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The chunked SSD form — THE ONE THE KERNEL IMPLEMENTS. Returns (y [L,H,P], final_state).

    Per chunk the output is the sum of two terms:

      * DIAGONAL — contributions that both enter and are read inside this chunk. A masked
        (C·Bᵀ) attention-like matrix weighted by the intra-chunk decay, times X. Two GEMMs.
      * STATE-PASSING — the state carried in at the chunk boundary, decayed to each position and
        read by C. One GEMV per position, plus one GEMM to produce the state carried OUT.

    `L` need not be a multiple of `chunk_size`; the tail chunk is handled at its true length, which
    is the case a kernel gets wrong first because the padded and unpadded decay matrices differ."""
    L, H, P = x.shape
    N = B.shape[-1]
    dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    A = -torch.exp(A_log.to(dtype))
    Bx = _group_expand(B.to(dtype), H)                       # [L, H, N]
    Cx = _group_expand(C.to(dtype), H)                       # [L, H, N]
    xf, dtf, Df = x.to(dtype), dt.to(dtype), D.to(dtype)

    S = (torch.zeros(H, P, N, dtype=dtype, device=x.device)
         if initial_state is None else initial_state.to(dtype).clone())
    ys = []
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        T = end - start
        xc = xf[start:end]                                   # [T, H, P]
        bc = Bx[start:end]                                   # [T, H, N]
        cc = Cx[start:end]
        dc = dtf[start:end]                                  # [T, H]
        dA = dc * A                                          # [T, H]  log-decay per step

        # --- state-passing term: the incoming state decayed to each position, read by C ---
        cum = torch.cumsum(dA, dim=0)                        # [T, H]  log decay from chunk start
        y_state = torch.einsum("hpn,thn->thp", S, cc) * torch.exp(cum).unsqueeze(-1)

        # --- diagonal term: intra-chunk contributions under the causal decay mask ---
        Lmat = torch.exp(segment_sum(dA.transpose(0, 1)))    # [H, T(i), T(j)]
        CB = torch.einsum("thn,shn->hts", cc, bc)            # [H, T, T]
        M = CB * Lmat * dc.transpose(0, 1).unsqueeze(-2)     # weight by dt at the SOURCE position j
        y_diag = torch.einsum("hts,shp->thp", M, xc)

        ys.append(y_state + y_diag + Df.view(1, H, 1) * xc)

        # --- state carried out: decay the incoming state across the whole chunk, add this chunk's
        # contributions each decayed from its own position to the chunk end ---
        total = cum[-1]                                      # [H]
        w = torch.exp(total.unsqueeze(0) - cum) * dc         # [T, H]
        S = torch.exp(total).view(H, 1, 1) * S + torch.einsum("thp,thn,th->hpn", xc, bc, w)

    return torch.cat(ys, dim=0), S


def mamba2_decode(
    x: torch.Tensor,          # [H, P]  one token, post-conv, post-silu
    dt: torch.Tensor,         # [H]     ALREADY discretized
    A_log: torch.Tensor,      # [H]
    B: torch.Tensor,          # [G, N]
    C: torch.Tensor,          # [G, N]
    D: torch.Tensor,          # [H]
    state: torch.Tensor,      # [H, P, N]  UPDATED IN PLACE, as the kernel will
) -> torch.Tensor:
    """One decode step. Returns y [H, P]; `state` is advanced in place.

    In place because that is the kernel's contract and because a decode step that returns a new
    state tensor hides the thing spec decode has to be able to undo — see the replay/rollback
    requirement in the plan (N1)."""
    H, P = x.shape
    dtype = state.dtype
    A = -torch.exp(A_log.to(dtype))
    Bx = _group_expand(B.to(dtype).unsqueeze(0), H).squeeze(0)   # [H, N]
    Cx = _group_expand(C.to(dtype).unsqueeze(0), H).squeeze(0)
    xf, dtf = x.to(dtype), dt.to(dtype)

    decay = torch.exp(dtf * A)                                   # [H]
    upd = (dtf.unsqueeze(-1) * xf).unsqueeze(-1) * Bx.unsqueeze(-2)
    state.mul_(decay.view(H, 1, 1)).add_(upd)
    return torch.einsum("hpn,hn->hp", state, Cx) + D.to(dtype).unsqueeze(-1) * xf
