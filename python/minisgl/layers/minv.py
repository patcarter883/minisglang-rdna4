"""M-invariant dense GEMM — the engine's standing rule for bf16/fp16 linears.

WHY THIS EXISTS
    torch's F.linear dispatches to rocBLAS, whose kernel / tile / split-K reduction ORDER is chosen by
    the problem shape (M,N,K). Float addition is non-associative, so linear(x[:m],W) is NOT bit-identical
    to linear(x[:M],W)[:m] when m != M: the same token's output changes by ~1 bf16 ULP depending on how
    many tokens share the batch. This silently corrupts every pathway where a token is computed at
    different M and expected to match:
      * prefix / radix caching  (cached continuation at M=new_len vs fresh at M=full_len)
      * chunked prefill         (a long prompt split by max_extend runs each chunk at a different M)
      * spec-decode VERIFY      (K+1 tokens/seq at variable M vs sequential decode at M=1)
    A coarse int8/fp8 activation downcast downstream amplifies the ULP seed into flipped greedy tokens
    (observed: ZAYA GSM8K 45->25 with CCA prefix caching on).

THE FIX
    Route bf16/fp16 dense linears through our OWN fixed-tile WMMA GEMM (the moe_bf16_wmma spine driven
    dense: single expert, identity routing). It does the FULL K-reduction per output tile in a fixed
    order with NO split-K, so the reduction order does not depend on M -> M-invariant BY CONSTRUCTION,
    and WE own the tile so it cannot drift under a rocBLAS/hipBLASLt upgrade. Verified bit-exact across
    M for the ZAYA projection shapes; parity vs F.linear is ~1 bf16 ULP (a different but fixed order).

SCOPE ("as close to M-invariant as realistically possible")
    Applied for eager bf16/fp16 GEMMs whose K is WMMA-friendly (IN % 16 == 0). Falls back to F.linear
    (with a one-time warning) for: fp32/other dtypes, IN not a multiple of 16, or under cudagraph
    capture (static shapes are already self-consistent, and the arange/route tensors would allocate
    mid-capture). Integer (int8) matmuls are already exact/M-invariant; quantized-expert and attention
    kernels are already fixed-tile HIP.
"""
from __future__ import annotations

import os

import torch

_BLOCK_M = int(os.environ.get("MINISGL_MINV_BLOCK_M", "64"))  # WMMA M-tile (mult of 16, <=128)
_BN = int(os.environ.get("MINISGL_MINV_BN", "64"))            # WMMA N-tile (divides most OUT dims)
# Large-M path: the deep-pipelined dense_gemm_pipe kernel runs at rocBLAS parity at large M (gate_up
# M=4096: ~113 vs rocBLAS ~114 TFLOPS, cca_q/down M=1024 BEAT it) after the RDNA4 LDS bank-conflict pad.
# It is BIT-IDENTICAL to the register-direct (rd) and LDS kernels (same 16-wide K-reduction order, no
# split-K), so switching rd<->pipe by M stays M-invariant. rd still wins the small-M decode hot path
# (register-direct, no LDS staging / __syncthreads overhead), so we only reach for pipe at M >= _PIPE_M.
_PIPE_M = int(os.environ.get("MINISGL_MINV_PIPE_M", "512"))   # M threshold to switch rd -> pipe
_PIPE_MI = int(os.environ.get("MINISGL_MINV_PIPE_MI", "2"))   # pipe register-block M-subtiles/warp
_PIPE_PBK = int(os.environ.get("MINISGL_MINV_PIPE_PBK", "64"))  # pipe K-chunk (needs IN % PBK == 0)
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        try:
            from minisgl.utils import init_logger

            init_logger(__name__).warning(msg)
        except Exception:
            pass


# Decode fast path through the shared bf16/fp16 GEMV — see the note in minv_linear.
#
# UNCONDITIONAL — no env gate. Both sides of the threshold are individually M-invariant (the GEMV
# per-(row,col) in a fixed K-order; dense_gemm with a fixed full-K reduction), so the only hazard is
# the CROSSING between them. MAXM=16 is the M ceiling the GEMV accepts, so ordinary decode and
# spec-decode VERIFY (M=K+1) land on the SAME kernel rather than opposite sides of the threshold —
# that is what keeps verify bit-matching sequential decode.
# Measured vs the old threshold of 2: +19.8% at conc=4, +7.9% at conc=8, neutral at bs=1
# (tools/_maxm_ab.sh).
_DECODE_GEMV_MAXM = 16
_decode_gemv_fn = None
_decode_gemv_w8a16_fn = None
_decode_gemv_probed = False


def _get_decode_gemv():
    """Lazily resolve fp8_wmma.dense_bf16_gemv (None if the kernel package is unavailable)."""
    _probe_decode_gemv()
    return _decode_gemv_fn


def _probe_decode_gemv():
    """Resolve BOTH decode-GEMV entry points once. The W8A16 op may be absent on an older kernel
    package, in which case the bf16 twin still resolves and the engine simply streams more bytes."""
    global _decode_gemv_fn, _decode_gemv_w8a16_fn, _decode_gemv_probed
    if _decode_gemv_probed:
        return
    _decode_gemv_probed = True
    try:
        from fp8_wmma import dense_bf16_gemv

        _decode_gemv_fn = dense_bf16_gemv
    except Exception:
        _decode_gemv_fn = None
    try:
        from fp8_wmma import dense_w8a16_gemv

        _decode_gemv_w8a16_fn = dense_w8a16_gemv
    except Exception:
        _decode_gemv_w8a16_fn = None


# =====================================================================================
# W8A16 decode GEMV — fp8 (e4m3) WEIGHTS, unquantized bf16/fp16 activations. THE DEFAULT.
#
# WHY. Profiled on the served Qwen3.6-35B-A3B TP=2 decode step, this GEMV is the single largest
# GPU consumer: 3.81 ms of a 10.99 ms TPOT (46% of all kernel time) across ~291 launches, and it
# decomposes exactly as `t = 3.93 us/dispatch + bytes / 648.1 GB/s`. The kernel BODY is already at
# 100% of the measured HBM read ceiling at every grid size — no tiling, spill, LDS or occupancy
# headroom exists (scratch 0, 13-27 VGPR, 16 B LDS, loads already b128), and three separate
# launch-count reductions (column-concat, KSPLIT, PREQUANT) each measured 0.0% e2e. The ONLY lever
# left is fewer BYTES, and every one of the 1.834 GB/step it streams was still unquantized bf16
# while the MoE experts running beside it were already W4A16.
#
# WHAT. One fp8 companion per weight tensor: raw e4m3 bytes + ONE fp32 scale per output channel,
# folded into the kernel epilogue. Halves the weight stream. Not bit-exact vs bf16 (that is stated,
# not hidden), but M-INVARIANCE — the property this module exists to defend — is preserved EXACTLY:
# each output is an independent per-(row, col) fp32 accumulation over K in a fixed lane-strided
# order, so row r at M=1 is bit-identical to row r at M=16, which is what keeps spec-verify,
# chunked prefill and radix caching lossless.
#
# NO FLAG. This is the default path; the bf16 twin remains as an in-code fallback for shapes the
# fp8 lane slot cannot take (K % 16 != 0), for weights on the KEEP-WIDE list below, and for the one
# context where the companion cannot be built (see below). The caller never has to know.
#
# SCOPE — NOT every dense weight. See the KEEP-WIDE registry immediately below: `lm_head` and the
# Qwen3.5/3.6 shared-expert `down_proj` are excluded under KERNEL_CORE_POLICY.md RULE 3 because the
# checkpoint ships them unquantized on purpose. MEASURED with those two excluded, Qwen3.6-35B-A3B
# TP=2 decode M=1: 89.50 -> 99.35 tok/s median (+11.0%), 4 interleaved reps, non-overlapping
# (cand_min 96.7 > base_max 90.5), against a base-vs-base control of 0.9967x.
#
# WHEN THE COMPANION IS BUILT. Lazily, on the first decode-shaped call for a given weight — which
# in a real serve is the EAGER warmup forward that GraphRunner runs immediately before each
# `torch.cuda.graph(...)` capture, at the same batch size. Allocating mid-capture is illegal, so if
# a weight somehow first arrives while capturing we fall back to bf16 for that call rather than
# fault. Building at warmup rather than at load time is deliberate: it leaves the KV-pool sizing
# (which happens earlier) byte-identical, so an A/B of this change is not confounded by a different
# pool. The cost is VRAM: +0.5 byte per QUANTIZED dense weight element on top of the retained bf16
# master. MEASURED on the 35B TP=2 config WITH the RULE-3 exclusions: +0.65 GiB/card (graph-capture
# free-memory delta 0.25 -> 0.90 GiB), taken from the post-KV slack. Keeping the LM head wide is
# most of the gap to the un-scoped ~0.92 GiB — its own companion alone is 254 MiB/card.
_W8A16_E4M3_MAX = 448.0
# Rows per quantisation chunk: the fp32 cast is materialised, so a whole-tensor cast spikes
# 4 bytes/elem of transient at warmup, right where the graph-capture reserve is tightest. 8192 rows
# caps it at ~67 MB at K=2048. With the LM head KEPT WIDE the largest companion left on the served
# model is the GDN in_proj_qkvz (6144 rows), so this chunks nothing today; it stays because it is
# what makes the builder safe at any N rather than at the sizes we happen to serve.
_W8A16_QUANT_ROWS = 8192
# key: (weight.data_ptr(), N, K) -> (e4m3 bytes uint8 (N,K), fp32 per-channel scale (N,))
_w8a16_companions: dict = {}

# ---------------------------------------------------------------------------------------------
# KEEP-WIDE registry — the tensors W8A16 must NEVER touch.
#
# WHY (KERNEL_CORE_POLICY.md, RULE 3: "a WLoad policy may EXIST anywhere; APPLYING it to an
# unquantized tensor is not yours to decide"). A W8A16 loader is a legitimate policy on the shared
# decode-GEMV core, but pointing it at a tensor the CHECKPOINT deliberately shipped wide is a model-
# quality decision, not a kernel decision — and a kernel-parity gate cannot clear it, because parity
# only says "the kernel computes what it claims", never "the model is still as good". The publisher
# skips exactly the tensors where quantization costs the most, so "it went faster" is not a licence.
#
# Marked here, per the policy's table:
#   * lm_head            — produces every token's logits; error lands directly on sampling.
#   * shared-expert down_proj (Qwen3.5/3.6 MoE) — small, always-on, and explicitly EXCLUDED from
#     the routed-expert quant by the checkpoint (Qwen3_5MoeSharedExpert builds it with
#     create_linear_method(None) while the routed experts carry int4/mxfp4).
#
# HOW. The policy lives ON THE TENSOR, not on the call site: a weight is marked once where it is
# owned (each module's post_load, after load_state_dict has installed the final tensor), and
# `w8a16_companion` returns None for it so the EXISTING in-code bf16 fallback carries it. There is
# no per-call-site branch, no module-name match and no shape heuristic — `K == 256` would silently
# catch an unrelated tensor the day a config changes. No env var either: if it merges, it is on.
#
# DO NOT "optimise this away". Deleting a mark here re-quantizes a tensor its publisher chose not
# to, and nothing in the parity suite will fail when you do.
_keep_wide: set = set()


def _weight_key(w: torch.Tensor):
    """Identity used by BOTH the keep-wide set and the companion cache: a weight is the storage it
    points at plus its 2-D shape. Same keying, so a mark can never miss the tensor it named."""
    return (w.data_ptr(), int(w.shape[0]), int(w.shape[1]))


def keep_wide(w: torch.Tensor) -> None:
    """Mark `w` as NEVER-QUANTIZE on the decode-GEMV path (see the note above). Call once from the
    owning module's `post_load`, i.e. after `load_state_dict` has installed the final tensor —
    marking the meta-device placeholder built in `__init__` would key on a stale pointer."""
    if w is None or w.dim() != 2:
        return
    _keep_wide.add(_weight_key(w))
    _warn_once(f"keep_wide_{w.shape[0]}x{w.shape[1]}",
               f"minv: KEEP-WIDE ({int(w.shape[0])}, {int(w.shape[1])}) {w.dtype} -- excluded from "
               f"the W8A16 decode GEMV (KERNEL_CORE_POLICY.md RULE 3: the checkpoint left this "
               f"tensor unquantized on purpose)")


def is_kept_wide(w: torch.Tensor) -> bool:
    """True iff `w` was marked never-quantize. Public so a probe/test can assert the exclusion took
    effect rather than inferring it from a kernel name in a log."""
    return w.dim() == 2 and _weight_key(w) in _keep_wide


def w8a16_companion(w: torch.Tensor):
    """The fp8 companion for weight `w`, building + caching it on first use.

    Returns None when the weight is KEPT WIDE (RULE 3 — the caller then runs the bf16 GEMV), or
    when the companion cannot be produced *right now* (mid-graph-capture, where a fresh allocation
    is illegal). Callers fall back to the bf16 GEMV in both cases."""
    if is_kept_wide(w):
        return None
    key = _weight_key(w)
    ent = _w8a16_companions.get(key)
    if ent is not None:
        return ent
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _warn_once("w8a16_capture",
                   "minv: W8A16 decode-GEMV companion missing at CUDA-graph capture time -> this "
                   "weight stays on the bf16 GEMV inside the graph (no fault, just more bytes). "
                   "Expected only if the pre-capture warmup forward did not touch this linear.")
        return None
    N, K = int(w.shape[0]), int(w.shape[1])
    q = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty((N,), dtype=torch.float32, device=w.device)
    worst = 0.0
    for r0 in range(0, N, _W8A16_QUANT_ROWS):
        r1 = min(r0 + _W8A16_QUANT_ROWS, N)
        blk = w[r0:r1].to(torch.float32)
        # Per-OUTPUT-CHANNEL (row) amax -> one fp32 scale per row, applied once in the epilogue.
        s = blk.abs().amax(dim=1).clamp_min_(1e-12) / _W8A16_E4M3_MAX
        scale[r0:r1] = s
        qb = blk.div(s.unsqueeze(1)).clamp_(-_W8A16_E4M3_MAX, _W8A16_E4M3_MAX).to(
            torch.float8_e4m3fn)
        q[r0:r1] = qb
        # Measure the representation error on the REAL weights, elementwise (no GEMM, no BLAS):
        # this is the whole numerics argument for W8A16 and it should be reported, not assumed.
        # Synthetic-weight parity on this part turned out to be unreliable at LM-head shapes, so
        # this is the number that actually characterises the served model.
        err = (qb.to(torch.float32).mul_(s.unsqueeze(1)) - blk).abs_().amax()
        worst = max(worst, float(err) / max(1e-30, float(blk.abs().amax())))
        del blk, qb
    _warn_once(f"w8a16_shape_{N}x{K}",
               f"minv W8A16: companion built for ({N}, {K}) -- max relative weight error "
               f"{worst:.4f} (per-output-channel e4m3), {N * K / 2**20:.0f} MiB added")
    ent = (q.view(torch.uint8), scale)
    _w8a16_companions[key] = ent
    return ent


def decode_gemv(x: torch.Tensor, w: torch.Tensor):
    """The small-M dense decode GEMV every call site shares. W8A16 (fp8 weight / native activation)
    by DEFAULT; the unquantized bf16 twin for shapes the fp8 lane slot cannot take AND for weights
    marked KEEP-WIDE (RULE 3 — see the registry above; `w8a16_companion` returns None for those, so
    the exclusion needs no branch here and no call site knows about it). Returns None if no decode
    GEMV applies at all, so the caller can keep its own fallback.

    Both paths run the SAME `gemv_decode_core` under different WLoad policies and are individually
    M-invariant, and the choice between them is a pure function of the weight shape — never of M —
    so it can never make row r depend on the batch size."""
    _probe_decode_gemv()
    if _decode_gemv_fn is None:
        return None
    xc = x.contiguous()
    if _decode_gemv_w8a16_fn is not None and w.shape[-1] % 16 == 0:
        ent = w8a16_companion(w)
        if ent is not None:
            return _decode_gemv_w8a16_fn(xc, ent[0], ent[1])
    if w.shape[-1] % 8 == 0:
        return _decode_gemv_fn(xc, w)
    return None


def minv_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """True iff `minv_linear` will run the M-invariant kernel (else it falls back to F.linear)."""
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    if weight.shape[-1] % 16 != 0:  # IN must be WMMA-friendly (full-K reduction in 16-wide steps)
        return False
    # Under CUDA-graph capture we STILL use dense_gemm (the whole point of removing rocBLAS): its only
    # allocations are F.pad + the output tensor, both via torch's caching allocator, which is capture-
    # safe (the graph memory pool). The old F.linear fallback here reintroduced rocBLAS/hipblaslt AND
    # faulted the spec-verify capture (hipblaslt's own workspace malloc is illegal mid-capture). Set
    # MINISGL_MINV_CAPTURE=0 to restore the fallback if a capture regression appears.
    if (os.environ.get("MINISGL_MINV_CAPTURE", "1") == "0"
            and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
        return False
    return True


def minv_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                *, block_m: int = _BLOCK_M, BN: int = _BN) -> torch.Tensor:
    """M-invariant drop-in for F.linear: C = x @ weight^T (+bias). Accepts N-D x (flattened to 2-D).
    Falls back to F.linear when the M-invariant kernel is unavailable for this dtype/shape/context."""
    import torch.nn.functional as F

    if not minv_supported(x, weight):
        if weight.dtype in (torch.bfloat16, torch.float16) and weight.shape[-1] % 16 != 0:
            _warn_once(f"K{weight.shape[-1]}",
                       f"minv_linear: IN={weight.shape[-1]} not a multiple of 16 -> F.linear fallback "
                       f"(this GEMM is NOT M-invariant; see layers/minv.py)")
        return F.linear(x, weight, bias)

    # DECODE fast path: the WMMA-tiled dense_gemm is built for large M and stalls at M=1 — it pads M
    # up to block_m (63/64 of the tile is padding) and the decode profile puts this chokepoint at
    # 2.58 ms/step (dense_gemm_rd, 160 calls) + 1.03 ms/step (the ragged dense_gemm on OUT=1
    # shared_expert_gate, 40 calls) = 3.6 ms of a 13.19 ms kernel step. The shared decode GEMV
    # (gemv_decode_core<Bf16GemvLoader>, packed v_dot2_f32_bf16) runs the same shapes at 1.5-3x —
    # it is the same trade the LM head already took (embedding.py) and the GDN projections took in
    # gdn/layer.py::_GemvLinear.
    #
    # M-invariant (row i is a per-(row,col) fp32 dot in a fixed K-order, bit-identical at any M —
    # measured max|d|=0.0 across M=1/2/8/16). It is NOT bit-identical to the dense_gemm family, so
    # the threshold is a crossing; MAXM=16 puts decode AND spec-verify on this side of it.
    if (x.dim() == 2 and x.shape[0] <= _DECODE_GEMV_MAXM and bias is None
            and x.dtype == weight.dtype):          # minv_supported() gated weight.dtype, not x's
        out = decode_gemv(x, weight)
        if out is not None:
            return out

    import dense_gemm as _dg

    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).contiguous()
    M, IN = x2.shape
    OUT = weight.shape[0]
    # All three dense_gemm kernels (rd / pipe / lds) run the FULL K-reduction per output tile in the
    # identical fixed 16-wide order with NO split-K, so every one is bit-identical per output row and
    # interchangeable -> the dispatch below is M-invariant regardless of which kernel a given M picks.
    #   * ragged OUT (OUT % BN != 0): the LDS kernel masks it (register-direct can't mask a ragged tile).
    #   * large M (>= _PIPE_M, full tiles, IN % PBK == 0): the deep-pipelined kernel at ~rocBLAS parity.
    #   * else (small-M decode hot path): register-direct (LDS-bypass), which wins there.
    # Full-tile kernels need block_m | M and BN | OUT; pad M up to the tile (padded rows are sliced off
    # and never touch the real rows).
    if OUT % BN != 0:
        out = _dg.dense_gemm(x2, weight, block_m, BN)  # LDS kernel: handles ragged OUT
    elif M >= _PIPE_M and IN % _PIPE_PBK == 0:
        pbm = 256 if M >= 1024 else 128                # deeper M -> more warps sharing the LDS B tile
        pbn = 128 if OUT % 128 == 0 else BN            # wider N-tile when OUT allows (fewer A reloads)
        Mp = ((M + pbm - 1) // pbm) * pbm
        xp = x2 if Mp == M else torch.nn.functional.pad(x2, (0, 0, 0, Mp - M))
        out = _dg.dense_gemm_pipe(xp, weight, pbm, pbn, _PIPE_MI, _PIPE_PBK)[:M]
    else:
        Mp = ((M + block_m - 1) // block_m) * block_m
        xp = x2 if Mp == M else torch.nn.functional.pad(x2, (0, 0, 0, Mp - M))
        out = _dg.dense_gemm_rd(xp, weight, block_m, BN)[:M]
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], OUT)
