"""Autograd (backward) support for the native Triton-free GDN HIP ops.

The `torch.ops.gdn_hip.*` kernels are FORWARD-ONLY: they run the gated-delta-rule prefill, the
depthwise causal conv, and the gated RMSNorm as opaque HIP custom ops with no registered backward.
That is fine for serving (inference), but it means the GDN linear-attention path cannot be
backprop-ed through — which a downstream trainer (CAM/memory-organ trains a small adapter by
backprop THROUGH a frozen Qwen3.5-4B, whose 24 GDN layers currently fall back to fla-Triton kernels
that HANG on RDNA4 when the residual is perturbed) needs.

This module makes the three TRAINING-relevant ops differentiable, WITHOUT touching the default
inference forward path. The strategy is the gradient-checkpointing / recompute pattern:

    forward :  call the fast native op unchanged (no hang, this is also the serve path).
    backward:  RECOMPUTE a pure-torch, differentiable equivalent of the op's forward on the SAVED
               ORIGINAL inputs (with requires_grad_()), then let torch.autograd.grad() produce the
               input grads. Correctness of the gradient comes for free from torch autograd over a
               faithful reference; it is Triton-free (pure torch elementwise/matmul on ROCm) so it
               cannot hang.

The pure-torch references (`ref_gdn_prefill_core`, `ref_causal_conv1d_fwd`, `ref_rmsnorm_gated`)
implement EXACTLY the math the HIP kernels implement (lifted from gdn_kernels.hip):

  gdn_prefill (per seq, per value-head; state S is [V,K]):
      q,k l2-normed over K with rsqrt(sumsq + 1e-6); q *= scale (= 1/sqrt(K))
      g    = -exp(A_log) * softplus(a + dt_bias)        beta = sigmoid(b)
      S   *= exp(g);  v -= S@k;  v *= beta;  S += outer(v,k);  o = S@q
  causal_conv1d_fwd (depthwise causal, left zero-pad, per channel):
      acc_t = bias_c + sum_{i} window[i]*w[c,i];  out = SiLU(acc) if activation
  rmsnorm_gated (norm-before-gate, over last dim D):
      out = x * rsqrt(mean(x^2) + eps) * weight * SiLU(z)

Scope / limitations of v1 (the training-prefill case) — stated plainly:
  * SINGLE full-sequence prefill: cu_seqlens = [0, T], has_initial_state = False. Varlen batches
    and non-zero initial state are NOT supported in the backward (the recompute reference here
    ignores state_indices/has_initial_state and treats the whole [T,...] buffer as one sequence
    starting from zero state). Serving still uses the native forward, which handles varlen — only
    TRAINING is constrained to this case.
  * ssm_state / conv_state are in-place OUTPUT buffers with ZERO initial state; they are treated as
    non-differentiable. The differentiable wrapper allocates a FRESH state buffer internally so the
    caller's tensor is not mutated and autograd never sees an in-place mutation of a saved input.
  * The `recurrent` op (gdn_prefill) is EXACT, so its native-forward-vs-reference-backward pairing
    is exact (gradcheck-clean). The `wmma` op's forward differs from the reference by ~1e-3 (fp16
    matmul operands), so a backward recomputed from the reference is an APPROXIMATE gradient for the
    wmma forward — correct to the reference, ~1e-3 off the exact wmma jacobian. Prefer the recurrent
    op for training if exactness matters (GDN_HIP_WMMA_PREFILL=0).

Opt-in — the inference path is untouched:
  * Differentiable wrapper functions `gdn_prefill_train`, `gdn_prefill_wmma_train`,
    `causal_conv1d_fwd_train`, `rmsnorm_gated_train` — call these instead of the raw ops where you
    want gradients. They return the native forward output and carry a grad_fn.
  * `enable()` registers an autograd formula on the raw ops themselves (via
    torch.library.register_autograd) so an UNCHANGED call site
    (torch.ops.gdn_hip.gdn_prefill(...)) becomes differentiable process-wide. Off by default.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

# ----------------------------------------------------------------------------------------------
# Pure-torch DIFFERENTIABLE references (faithful to gdn_kernels.hip). These are the correctness
# anchor: their forward must match the op forward, and their gradient is what we hand back.
# ----------------------------------------------------------------------------------------------

_L2_EPS = 1e-6  # matches gdn_inv_l2 / the chunked kernels: inv = rsqrt(sumsq + 1e-6)


def _compute_dtype(*ts: torch.Tensor) -> torch.dtype:
    """The float dtype to compute the reference in. The HIP kernels up-cast every I/O element to
    fp32 in-register, so the recompute-backward runs in fp32 (matching the kernel). But a float64
    gradcheck feeds float64 inputs and needs float64 arithmetic to be self-consistent; forcing fp32
    there would floor the reference at ~1e-7 and fail the check. So: compute in the HIGHER of fp32
    and the input dtype — fp32 for bf16/fp16/fp32 activations (the real training path), float64 only
    when a gradcheck deliberately passes float64."""
    hi = torch.float32
    for t in ts:
        if t is not None and t.is_floating_point() and t.dtype == torch.float64:
            hi = torch.float64
    return hi


def _softplus(x: torch.Tensor) -> torch.Tensor:
    # matches gdn_softplus: threshold 20 to avoid overflow in log1p(exp(x))
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


def _l2norm_kernel(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize over the last dim with the KERNEL's convention: x * rsqrt(sum(x^2) + 1e-6).

    NB: this is NOT F.normalize (which is x / max(||x||, eps)); the HIP gdn_inv_l2 adds eps INSIDE
    the rsqrt, so we replicate that for a faithful gradient."""
    inv = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + _L2_EPS)
    return x * inv


# ----------------------------------------------------------------------------------------------
# The gated-delta recurrence, two equivalent scans. Both take the ALREADY-prepared per-token tensors
# (qh,kh l2-normed + q-scaled + GQA-expanded to [T,HV,K]; g,beta [T,HV]) and return core [T,HV,V].
# ----------------------------------------------------------------------------------------------

def _gdn_scan_recurrent(qh, kh, v, g, beta):
    """Exact per-token recurrence (the reference-of-record). O(T) sequential steps; O(T) state tensors
    retained by autograd -> memory-heavy in the recompute-backward (the OOM lever the chunk fixes)."""
    T, HV, K = qh.shape
    Vv = v.shape[2]
    S = torch.zeros(HV, Vv, K, dtype=v.dtype, device=v.device)
    outs = []
    for t in range(T):
        S = S * torch.exp(g[t]).view(HV, 1, 1)              # decay
        kt, qt = kh[t], qh[t]                               # [HV,K]
        Sk = torch.einsum("hvk,hk->hv", S, kt)             # [HV,V]
        vt = (v[t] - Sk) * beta[t].unsqueeze(-1)           # delta pseudo-value
        S = S + vt.unsqueeze(-1) * kt.unsqueeze(1)         # rank-1 update
        outs.append(torch.einsum("hvk,hk->hv", S, qt))     # o = S@q (post-update)
    return torch.stack(outs, dim=0)                         # [T,HV,V]


def _unit_lower_solve(M, RHS):
    """Solve (I + M) U = RHS for U, where M is strictly-lower-triangular [C,C,HV] and RHS is [C,HV,V],
    by forward substitution: U_i = RHS_i - sum_{j<i} M[i,j] U_j. BLAS-free (portable CPU+ROCm) and
    differentiable — the intra-chunk delta-rule coupling is C steps of cheap [HV,V] ops, not the
    per-token [HV,V,K] state carry the recurrence pays."""
    C = RHS.shape[0]
    U = []
    for i in range(C):
        u_i = RHS[i]
        if i > 0:
            u_i = u_i - torch.einsum("jh,jhv->hv", M[i, :i], torch.stack(U, dim=0))
        U.append(u_i)
    return torch.stack(U, dim=0)                            # [C,HV,V]


def _gdn_chunk(S_prev, q_c, k_c, v_c, g_c, beta_c):
    """One chunk of the gated-delta recurrence in closed form (the WY / forward-substitution
    representation). Returns (S_next, O_c). EXACT — algebraically identical to the per-token scan,
    just reassociated into matmuls + one unit-lower-triangular solve per head.

    Within-chunk cumulative gate G_i = sum_{l<=i} g_l (<=0). All decay factors used are exp(G_i - G_j)
    with i>=j, i.e. <=1 (numerically safe); the strictly-upper entries (i<j, would exp to >1) are
    clamp(max=0)-ed before exp and then masked out, so no inf * 0 -> nan."""
    C, HV, K = q_c.shape
    G = torch.cumsum(g_c, dim=0)                            # [C,HV]
    dG = (G.unsqueeze(1) - G.unsqueeze(0)).clamp(max=0.0)   # [C,C,HV] (i,j)=G_i-G_j, <=0 kept, >0->0
    decay = torch.exp(dG)                                   # [C,C,HV], entries in (0,1]
    eG = torch.exp(G)                                       # [C,HV] = b_i (state decay from chunk start)
    KK = torch.einsum("ihk,jhk->ijh", k_c, k_c)            # k_i . k_j
    QK = torch.einsum("ihk,jhk->ijh", q_c, k_c)            # q_i . k_j
    ar = torch.arange(C, device=q_c.device)
    strict = (ar.unsqueeze(1) > ar.unsqueeze(0)).unsqueeze(-1)   # [C,C,1] i>j
    lower = (ar.unsqueeze(1) >= ar.unsqueeze(0)).unsqueeze(-1)   # [C,C,1] i>=j
    # (I + M) U = RHS ; M[i,j] = beta_i * decay_ij * (k_i.k_j), strictly lower.
    M = (beta_c.unsqueeze(1) * decay * KK) * strict         # [C,C,HV]
    Spk = torch.einsum("hvk,ihk->ihv", S_prev, k_c)        # [C,HV,V] S_prev @ k_i
    RHS = beta_c.unsqueeze(-1) * (v_c - eG.unsqueeze(-1) * Spk)   # [C,HV,V]
    U = _unit_lower_solve(M, RHS)                          # (I+M) U = RHS, forward substitution
    # O_i = sum_{j<=i} decay_ij (q_i.k_j) u_j   +   b_i (S_prev @ q_i)
    P = ((decay * QK) * lower).permute(2, 0, 1)            # [HV,C,C]
    O = torch.einsum("hij,jhv->ihv", P, U) \
        + eG.unsqueeze(-1) * torch.einsum("hvk,ihk->ihv", S_prev, q_c)
    # S_next = b_{C-1} S_prev + sum_j exp(G_{C-1}-G_j) u_j k_j^T
    cfac = torch.exp(G[-1].unsqueeze(0) - G)               # [C,HV], <=1
    S_next = eG[-1].view(HV, 1, 1) * S_prev \
        + torch.einsum("jhv,jhk->hvk", cfac.unsqueeze(-1) * U, k_c)
    return S_next, O


def _gdn_scan_chunked(qh, kh, v, g, beta, chunk):
    """Chunked scan: closed-form intra-chunk (matmuls) + sequential inter-chunk state carry. Faster
    than the per-token loop (fewer, larger ops) and memory-bounded — each chunk is gradient-
    checkpointed, so the recompute-backward holds ~one chunk's activations, not the whole sequence."""
    T, HV, K = qh.shape
    Vv = v.shape[2]
    S = torch.zeros(HV, Vv, K, dtype=v.dtype, device=v.device)
    outs = []
    use_ckpt = torch.is_grad_enabled()
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        args = (S, qh[c0:c1], kh[c0:c1], v[c0:c1], g[c0:c1], beta[c0:c1])
        if use_ckpt:
            S, O_c = torch.utils.checkpoint.checkpoint(_gdn_chunk, *args, use_reentrant=False)
        else:
            S, O_c = _gdn_chunk(*args)
        outs.append(O_c)
    return torch.cat(outs, dim=0)                           # [T,HV,V]


def ref_gdn_prefill_core(
    q: torch.Tensor,      # [T, H, K]   (activation dtype; up-cast to fp32 here)
    k: torch.Tensor,      # [T, H, K]
    v: torch.Tensor,      # [T, HV, V]
    a: torch.Tensor,      # [T, HV]
    b: torch.Tensor,      # [T, HV]
    A_log: torch.Tensor,  # [HV]  (fp32)
    dt_bias: torch.Tensor,  # [HV] (fp32)
    scale: float,
    use_l2norm: bool = True,
) -> torch.Tensor:
    """Pure-torch differentiable equivalent of gdn_prefill / gdn_prefill_wmma for ONE full sequence
    starting from ZERO state (cu_seqlens=[0,T], has_initial_state=False). Returns core [T, HV, V].

    Faithful to the recurrent kernel (gdn_prefill_kernel + gdn_step): per value-head hv, GQA head
    hq = hv // (HV // H); q,k l2-normed (kernel eps), q scaled; g/beta from a,b,A_log,dt_bias; the
    rank-1 gated-delta recurrence carried in fp32. Vectorized over value-heads (the batch dim), with
    a python loop over the T recurrence steps (T is small in training-prefill; each step is a batched
    [HV,V,K] update). SSM-state Frobenius clamp (SSM_STATE_MAX_NORM=1000) is a no-op in-range and is
    intentionally omitted from the reference (it never engages on training-scale sequences)."""
    T, H, K = q.shape
    HV, Vv = v.shape[1], v.shape[2]
    f32 = _compute_dtype(q, k, v, a, b, A_log, dt_bias)
    q = q.to(f32); k = k.to(f32); v = v.to(f32); a = a.to(f32); b = b.to(f32)
    A_log = A_log.to(f32); dt_bias = dt_bias.to(f32)

    if use_l2norm:
        qn = _l2norm_kernel(q) * scale        # [T, H, K]
        kn = _l2norm_kernel(k)                # [T, H, K]
    else:
        qn = q * scale
        kn = k

    # expand GQA: value-head hv uses key/query head hv // (HV//H)
    rep = HV // H
    qh = qn.repeat_interleave(rep, dim=1)     # [T, HV, K]
    kh = kn.repeat_interleave(rep, dim=1)     # [T, HV, K]

    # per-token, per-value-head gate + decay
    g = -torch.exp(A_log) * _softplus(a + dt_bias)   # [T, HV]
    beta = torch.sigmoid(b)                           # [T, HV]

    # dispatch: chunked closed-form scan (fast + memory-bounded) for long sequences, exact per-token
    # recurrence otherwise. Both are algebraically identical (parity + gradcheck in
    # tools/gdn_backward_gradcheck.py). GDN_REF_CHUNK sets the chunk width (0 -> always recurrent).
    chunk = int(os.environ.get("GDN_REF_CHUNK", "64"))
    if chunk and T > chunk:
        return _gdn_scan_chunked(qh, kh, v, g, beta, chunk)
    return _gdn_scan_recurrent(qh, kh, v, g, beta)


def ref_causal_conv1d_fwd(
    x: torch.Tensor,          # [T, C]  token-major (activation dtype)
    weight: torch.Tensor,     # [C, W]  (fp32)
    bias: torch.Tensor | None,  # [C] or None
    activation: bool = True,
) -> torch.Tensor:
    """Pure-torch differentiable equivalent of causal_conv1d_fwd for ONE sequence, zero initial
    state. Depthwise causal conv (left zero-pad of W-1) + optional SiLU. Faithful to
    causal_conv1d_fwd_kernel: out_t,c = SiLU( bias_c + sum_{i<W} window[i]*w[c,i] ), where window is
    the trailing W inputs ending at t (zeros before the start)."""
    T, C = x.shape
    W = weight.shape[1]
    f32 = _compute_dtype(x, weight, bias)
    xf = x.to(f32)
    # depthwise causal conv via F.conv1d with groups=C. Input [1, C, T]; left-pad W-1.
    xin = xf.transpose(0, 1).unsqueeze(0)             # [1, C, T]
    xin = F.pad(xin, (W - 1, 0))                      # causal left pad
    wt = weight.to(f32).unsqueeze(1)                  # [C, 1, W]
    bt = bias.to(f32) if bias is not None else None
    acc = F.conv1d(xin, wt, bias=bt, groups=C)        # [1, C, T]
    acc = acc.squeeze(0).transpose(0, 1)              # [T, C]
    if activation:
        acc = F.silu(acc)
    return acc


def ref_rmsnorm_gated(
    x: torch.Tensor,        # [M, D] (activation dtype)
    z: torch.Tensor,        # [M, D]
    weight: torch.Tensor,   # [D]    (fp32)
    eps: float,
) -> torch.Tensor:
    """Pure-torch differentiable equivalent of rmsnorm_gated: norm-before-gate over the last dim.
    out = x * rsqrt(mean(x^2) + eps) * weight * SiLU(z). Faithful to rmsnorm_gated_kernel."""
    f32 = _compute_dtype(x, z, weight)
    xf = x.to(f32); zf = z.to(f32); wf = weight.to(f32)
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return xf * inv * wf * F.silu(zf)


# ----------------------------------------------------------------------------------------------
# autograd.Function wrappers: native forward, pure-torch recompute backward.
# ----------------------------------------------------------------------------------------------

def _make_cu_seqlens(T: int, device) -> torch.Tensor:
    return torch.tensor([0, T], dtype=torch.int32, device=device)


def _native_bwd_ok(name: str) -> bool:
    """True if the native HIP backward op `name` should be used: on by default, off via
    GDN_HIP_NATIVE_BWD=0, and auto-disabled if the op isn't registered (older .so) so the pure-torch
    recompute still works. Checked at backward time, when the extension is definitely loaded."""
    return os.environ.get("GDN_HIP_NATIVE_BWD", "1") != "0" and hasattr(torch.ops.gdn_hip, name)


class _GDNPrefillFn(torch.autograd.Function):
    """Differentiable gdn_prefill / gdn_prefill_wmma (recurrent or wmma native forward; reference
    recompute backward). Single-seq, zero-initial-state training case."""

    @staticmethod
    def forward(ctx, q, k, v, a, b, A_log, dt_bias, scale, use_l2norm, wmma):
        T = q.shape[0]
        HV, Vv, Kk = v.shape[1], v.shape[2], q.shape[2]
        dev = q.device
        # Fresh state buffer so the caller's tensors are never mutated; slot 1 valid, slot 0 NULL.
        ssm_state = torch.zeros(2, HV, Vv, Kk, dtype=torch.float32, device=dev)
        cu = _make_cu_seqlens(T, dev)
        state_idx = torch.tensor([1], dtype=torch.long, device=dev)
        has_init = torch.zeros(1, dtype=torch.uint8, device=dev)
        op = torch.ops.gdn_hip.gdn_prefill_wmma if wmma else torch.ops.gdn_hip.gdn_prefill
        out = op(q.contiguous(), k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous(),
                 A_log, dt_bias, cu, state_idx, has_init, ssm_state, float(scale), int(use_l2norm))
        ctx.save_for_backward(q, k, v, a, b, A_log, dt_bias)
        ctx.scale = float(scale)
        ctx.use_l2norm = bool(use_l2norm)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, a, b, A_log, dt_bias = ctx.saved_tensors
        with torch.enable_grad():
            qd = q.detach().to(torch.float32).requires_grad_(True)
            kd = k.detach().to(torch.float32).requires_grad_(True)
            vd = v.detach().to(torch.float32).requires_grad_(True)
            ad = a.detach().to(torch.float32).requires_grad_(True)
            bd = b.detach().to(torch.float32).requires_grad_(True)
            Ad = A_log.detach().to(torch.float32).requires_grad_(True)
            dd = dt_bias.detach().to(torch.float32).requires_grad_(True)
            ref = ref_gdn_prefill_core(qd, kd, vd, ad, bd, Ad, dd, ctx.scale, ctx.use_l2norm)
            grads = torch.autograd.grad(ref, (qd, kd, vd, ad, bd, Ad, dd),
                                        grad_out.to(torch.float32))
        gq, gk, gv, ga, gb, gA, gd = grads

        def cast(g, ref_t):
            return None if g is None else g.to(ref_t.dtype)
        # order matches forward's non-self args: q,k,v,a,b,A_log,dt_bias,scale,use_l2norm,wmma
        return (cast(gq, q), cast(gk, k), cast(gv, v), cast(ga, a), cast(gb, b),
                cast(gA, A_log), cast(gd, dt_bias), None, None, None)


class _CausalConv1dFwdFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, activation):
        T, C = x.shape
        dev = x.device
        conv_state = torch.zeros(2, C, weight.shape[1] - 1, dtype=torch.float32, device=dev)
        cu = _make_cu_seqlens(T, dev)
        state_idx = torch.tensor([1], dtype=torch.long, device=dev)
        has_init = torch.zeros(1, dtype=torch.uint8, device=dev)
        out = torch.ops.gdn_hip.causal_conv1d_fwd(
            x.contiguous(), weight, bias, cu, state_idx, has_init, conv_state, int(activation))
        ctx.save_for_backward(x, weight, bias if bias is not None else torch.empty(0))
        ctx.has_bias = bias is not None
        ctx.activation = bool(activation)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias_s = ctx.saved_tensors
        if _native_bwd_ok("causal_conv1d_bwd"):
            bias = bias_s if ctx.has_bias else None
            dx, dw, db = torch.ops.gdn_hip.causal_conv1d_bwd(
                grad_out.contiguous().to(x.dtype), x.contiguous(), weight, bias, int(ctx.activation))
            gb = db.to(bias_s.dtype) if ctx.has_bias else None
            return dx, dw.to(weight.dtype), gb, None
        with torch.enable_grad():
            xd = x.detach().to(torch.float32).requires_grad_(True)
            wd = weight.detach().to(torch.float32).requires_grad_(True)
            if ctx.has_bias:
                bd = bias_s.detach().to(torch.float32).requires_grad_(True)
                inputs = (xd, wd, bd)
                ref = ref_causal_conv1d_fwd(xd, wd, bd, ctx.activation)
            else:
                inputs = (xd, wd)
                ref = ref_causal_conv1d_fwd(xd, wd, None, ctx.activation)
            grads = torch.autograd.grad(ref, inputs, grad_out.to(torch.float32))
        gx = grads[0].to(x.dtype)
        gw = grads[1].to(weight.dtype)
        gb = grads[2].to(bias_s.dtype) if ctx.has_bias else None
        return gx, gw, gb, None


class _RMSNormGatedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, weight, eps):
        out = torch.ops.gdn_hip.rmsnorm_gated(x.contiguous(), z.contiguous(), weight, float(eps))
        ctx.save_for_backward(x, z, weight)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, z, weight = ctx.saved_tensors
        if _native_bwd_ok("rmsnorm_gated_bwd"):
            dx, dz, dw = torch.ops.gdn_hip.rmsnorm_gated_bwd(
                grad_out.contiguous().to(x.dtype), x.contiguous(), z.contiguous(), weight, ctx.eps)
            return dx, dz, dw.to(weight.dtype), None
        with torch.enable_grad():
            xd = x.detach().to(torch.float32).requires_grad_(True)
            zd = z.detach().to(torch.float32).requires_grad_(True)
            wd = weight.detach().to(torch.float32).requires_grad_(True)
            ref = ref_rmsnorm_gated(xd, zd, wd, ctx.eps)
            gx, gz, gw = torch.autograd.grad(ref, (xd, zd, wd), grad_out.to(torch.float32))
        return gx.to(x.dtype), gz.to(z.dtype), gw.to(weight.dtype), None


# ----------------------------------------------------------------------------------------------
# Public differentiable wrappers (opt-in; do NOT change the default inference forward path).
# ----------------------------------------------------------------------------------------------

def gdn_prefill_train(q, k, v, a, b, A_log, dt_bias, scale, use_l2norm=1):
    """Differentiable gdn_prefill (RECURRENT native forward, EXACT). Single-seq, zero initial state.
    Returns core [T, HV, V] with a grad_fn. Grads flow to q,k,v,a,b,A_log,dt_bias."""
    return _GDNPrefillFn.apply(q, k, v, a, b, A_log, dt_bias, scale, use_l2norm, False)


def gdn_prefill_wmma_train(q, k, v, a, b, A_log, dt_bias, scale, use_l2norm=1):
    """Differentiable gdn_prefill_wmma (WMMA native forward, ~1e-3 vs recurrent). The backward is
    recomputed from the EXACT reference, so the gradient is exact-to-the-reference and ~1e-3 off the
    true wmma jacobian. Use gdn_prefill_train for an exact forward+backward pairing."""
    return _GDNPrefillFn.apply(q, k, v, a, b, A_log, dt_bias, scale, use_l2norm, True)


def causal_conv1d_fwd_train(x, weight, bias, activation=1):
    """Differentiable causal_conv1d_fwd (depthwise causal conv + SiLU). Single-seq, zero initial
    state. Grads flow to x, weight, and bias (if given)."""
    return _CausalConv1dFwdFn.apply(x, weight, bias, activation)


def rmsnorm_gated_train(x, z, weight, eps):
    """Differentiable rmsnorm_gated (norm-before-gate + SiLU gate). Grads flow to x, z, weight."""
    return _RMSNormGatedFn.apply(x, z, weight, eps)


# ----------------------------------------------------------------------------------------------
# Process-wide opt-in: register an autograd formula on the RAW ops so unchanged call sites become
# differentiable. Off by default (serving never calls enable()).
# ----------------------------------------------------------------------------------------------

_ENABLED = False


def enable() -> None:
    """Register recompute-backward autograd formulas on the FUNCTIONAL gdn_hip ops so an UNCHANGED
    forward call participates in autograd via the pure-torch reference recompute. Idempotent; does
    NOT alter forward numerics (serve path is the native op either way).

    IMPORTANT — only functional ops can take a raw autograd formula. `gdn_prefill`,
    `gdn_prefill_wmma`, and `causal_conv1d_fwd` mutate their state argument in place
    (`Tensor(a!) ssm_state` / `Tensor(a!) conv_state`), and torch.library.register_autograd rejects
    a non-functional operator. For those the differentiable entry point is the `*_train`
    autograd.Function wrapper (single-seq, zero initial state), which the training layer path calls
    directly — NOT this raw-op registration. Only `rmsnorm_gated` (schema `-> Tensor`, no mutation)
    is registered here, so the layer's final norm-gate is differentiable for free. Registrations that
    fail on a non-functional op are logged and skipped, not fatal."""
    global _ENABLED
    if _ENABLED:
        return

    def _try_register(opname, backward, setup):
        try:
            torch.library.register_autograd(opname, backward, setup_context=setup)
            return True
        except RuntimeError as e:
            if "non-functional" in str(e):
                print(f"[gdn_hip.autograd] skip raw-op autograd on {opname} (in-place state mutation; "
                      f"use the *_train wrapper for the differentiable path)")
                return False
            raise

    def _prefill_setup(ctx, inputs, output):
        # inputs: (q,k,v,a,b,A_log,dt_bias,cu_seqlens,state_indices,has_initial_state,ssm_state,
        #          scale,use_l2norm)
        (q, k, v, a, b, A_log, dt_bias, cu, sidx, hinit, ssm, scale, use_l2) = inputs
        ctx.save_for_backward(q, k, v, a, b, A_log, dt_bias)
        ctx.scale = float(scale)
        ctx.use_l2norm = bool(use_l2)

    def _prefill_backward(ctx, grad_out):
        q, k, v, a, b, A_log, dt_bias = ctx.saved_tensors
        with torch.enable_grad():
            ins = [t.detach().to(torch.float32).requires_grad_(True)
                   for t in (q, k, v, a, b, A_log, dt_bias)]
            ref = ref_gdn_prefill_core(*ins, ctx.scale, ctx.use_l2norm)
            grads = torch.autograd.grad(ref, ins, grad_out.to(torch.float32))
        outs = [g.to(t.dtype) for g, t in zip(grads, (q, k, v, a, b, A_log, dt_bias))]
        # grads for (q,k,v,a,b,A_log,dt_bias, cu,state_indices,has_initial_state,ssm_state,
        #            scale,use_l2norm)
        return (*outs, None, None, None, None, None, None)

    _try_register("gdn_hip::gdn_prefill", _prefill_backward, _prefill_setup)
    _try_register("gdn_hip::gdn_prefill_wmma", _prefill_backward, _prefill_setup)

    def _conv_setup(ctx, inputs, output):
        # inputs: (x, weight, bias, cu_seqlens, state_indices, has_initial_state, conv_state,
        #          activation)
        (x, weight, bias, cu, sidx, hinit, conv_state, activation) = inputs
        ctx.save_for_backward(x, weight, bias if bias is not None else torch.empty(0))
        ctx.has_bias = bias is not None
        ctx.activation = bool(activation)

    def _conv_backward(ctx, grad_out):
        x, weight, bias_s = ctx.saved_tensors
        with torch.enable_grad():
            xd = x.detach().to(torch.float32).requires_grad_(True)
            wd = weight.detach().to(torch.float32).requires_grad_(True)
            if ctx.has_bias:
                bd = bias_s.detach().to(torch.float32).requires_grad_(True)
                ref = ref_causal_conv1d_fwd(xd, wd, bd, ctx.activation)
                gx, gw, gb = torch.autograd.grad(ref, (xd, wd, bd), grad_out.to(torch.float32))
                gb = gb.to(bias_s.dtype)
            else:
                ref = ref_causal_conv1d_fwd(xd, wd, None, ctx.activation)
                gx, gw = torch.autograd.grad(ref, (xd, wd), grad_out.to(torch.float32))
                gb = None
        # grads for (x, weight, bias, cu, state_indices, has_initial_state, conv_state, activation)
        return gx.to(x.dtype), gw.to(weight.dtype), gb, None, None, None, None, None

    _try_register("gdn_hip::causal_conv1d_fwd", _conv_backward, _conv_setup)

    def _norm_setup(ctx, inputs, output):
        (x, z, weight, eps) = inputs
        ctx.save_for_backward(x, z, weight)
        ctx.eps = float(eps)

    def _norm_backward(ctx, grad_out):
        x, z, weight = ctx.saved_tensors
        with torch.enable_grad():
            xd = x.detach().to(torch.float32).requires_grad_(True)
            zd = z.detach().to(torch.float32).requires_grad_(True)
            wd = weight.detach().to(torch.float32).requires_grad_(True)
            ref = ref_rmsnorm_gated(xd, zd, wd, ctx.eps)
            gx, gz, gw = torch.autograd.grad(ref, (xd, zd, wd), grad_out.to(torch.float32))
        # grads for (x, z, weight, eps)
        return gx.to(x.dtype), gz.to(z.dtype), gw.to(weight.dtype), None

    _try_register("gdn_hip::rmsnorm_gated", _norm_backward, _norm_setup)

    _ENABLED = True


__all__ = [
    "ref_gdn_prefill_core", "ref_causal_conv1d_fwd", "ref_rmsnorm_gated",
    "gdn_prefill_train", "gdn_prefill_wmma_train", "causal_conv1d_fwd_train", "rmsnorm_gated_train",
    "enable",
]
