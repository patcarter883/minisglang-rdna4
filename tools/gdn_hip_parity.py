"""Task 4-#19 — numeric parity for the native gdn_hip HIP kernels (GPU), across I/O dtypes.

Each gdn_hip op is checked against a pure-torch reference of the EXACT math it implements (the fla
gated-delta-rule recurrence + depthwise causal conv + gated RMSNorm). The kernels are now templated
on the ACTIVATION I/O dtype (fp32/fp16/bf16) and up-cast to fp32 in-register for the math; per-head
params (A_log/dt_bias/weight/bias) and the recurrent STATE (ssm/conv) stay fp32. So this harness runs
each check at all three dtypes: the activation inputs (q/k/v/a/b/x/z) are rounded to the test dtype
and the REFERENCE is fed the SAME rounded values, so what's measured is the kernel's own error (RELATIVE
to the output magnitude), NOT input rounding. A faithful kernel sits near the dtype's mantissa floor
(fp32~1e-3, fp16~1e-2, bf16~5e-2 rel); a real indexing/LDS/register bug or NaN blows past it.

Run inside the combined ROCm image UNDER a 1-card lease (executes HIP kernels):
    .../gpu-lease.sh -n 1 -- bash -c 'docker run ... python /engine/tools/gdn_hip_parity.py'
(The gdn_hip_C*.so must be built first: cd gdn_hip && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import gdn_hip  # loads the .so + registers torch.ops.gdn_hip.*

DEV = "cuda"
torch.manual_seed(0)

# Real GDN geometry (Qwen3.5/3.6): num_k_heads=16, num_v_heads=32, head_k_dim=head_v_dim=128.
H, HV, K, V = 16, 32, 128, 128
SCALE = K ** -0.5

# --- per-run dtype state (set by main()'s dtype loop) ----------------------------------------------
DT = torch.float32                     # current activation I/O dtype under test
# base RELATIVE threshold (max|Δ| / mean|ref|) per dtype = a few x the mantissa floor. Calibrated to
# the measured kernel-vs-fp32-reference error: fp32 non-wmma is ~1e-6 (exact); the floors rise with the
# dtype's mantissa (fp16 ~8e-3, bf16 ~6-8e-2 — the mean denominator inflates a single max-element's
# low-bit output rounding). The TIGHT fp32 tier + the isfinite guard are the real bug catchers (a logic
# bug shows at fp32 too, since one templated kernel serves all dtypes); the looser low-precision tiers
# confirm the bf16/fp16 path runs and stays within gross rounding.
_BASE_THR = {torch.float32: 4e-3, torch.float16: 1.5e-2, torch.bfloat16: 1.2e-1}


def to_dt(t: torch.Tensor) -> torch.Tensor:
    """Cast an activation tensor to the test dtype for the KERNEL call (exact when t is already
    DT-rounded in fp32 storage)."""
    return t.to(DT)


def rnd(*ts: torch.Tensor):
    """DT-round the activation inputs but keep them in fp32 storage, so the torch REFERENCE computes
    on exactly the values the kernel sees. (No-op at fp32.)"""
    if DT == torch.float32:
        return ts if len(ts) > 1 else ts[0]
    out = tuple(t.to(DT).float() for t in ts)
    return out if len(out) > 1 else out[0]


def _report(name: str, got: torch.Tensor, ref: torch.Tensor, tol_mult: float = 1.0) -> bool:
    d = (got.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().mean().item()
    rel = d / (scale + 1e-9)
    thr = _BASE_THR[DT] * tol_mult
    ok = rel <= thr and torch.isfinite(got).all().item()
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:42s} max|Δ|={d:.3e} rel={rel:.3e} (thr={thr:.1e})")
    return ok


def _softplus(x):
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


def ref_step(S, q, k, v, g, beta):
    """One recurrence step. S:[V,K], q,k:[K], v:[V] -> o:[V], updates S."""
    S = S * torch.exp(g)
    v = v - S @ k          # [V]
    v = v * beta
    S = S + torch.outer(v, k)
    o = S @ q              # [V]
    return o, S


def check_decode() -> bool:
    B = 4
    num_slots = 8
    q = torch.randn(B, H, K, device=DEV)
    k = torch.randn(B, H, K, device=DEV)
    v = torch.randn(B, HV, V, device=DEV)
    a = torch.randn(B, HV, device=DEV)
    b = torch.randn(B, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    state = torch.randn(num_slots, HV, V, K, device=DEV)
    # slots: mix of valid + one NULL (0) to exercise the skip
    idx = torch.tensor([1, 0, 3, 5], dtype=torch.long, device=DEV)
    q, k, v, a, b = rnd(q, k, v, a, b)  # reference sees the DT-rounded activation inputs

    ref_state = state.clone()
    ref_out = torch.zeros(B, HV, V, device=DEV)
    for bi in range(B):
        slot = int(idx[bi])
        if slot <= 0:
            continue
        for hv in range(HV):
            hq = hv // (HV // H)
            qn = F.normalize(q[bi, hq], dim=-1, eps=1e-6) * SCALE
            kn = F.normalize(k[bi, hq], dim=-1, eps=1e-6)
            g = -torch.exp(A_log[hv]) * _softplus(a[bi, hv] + dt_bias[hv])
            beta = torch.sigmoid(b[bi, hv])
            o, S = ref_step(ref_state[slot, hv], qn, kn, v[bi, hv].clone(), g, beta)
            ref_out[bi, hv] = o
            ref_state[slot, hv] = S

    got_state = state.clone()
    got_out = torch.ops.gdn_hip.gdn_decode(to_dt(q), to_dt(k), to_dt(v), to_dt(a), to_dt(b), A_log, dt_bias, got_state,
                                           idx, SCALE, 1)
    ok = _report("gdn_decode.out", got_out, ref_out)
    ok &= _report("gdn_decode.state", got_state[idx[idx > 0]], ref_state[idx[idx > 0]])
    return ok


def check_prefill() -> bool:
    lens = [5, 3]
    N = len(lens)
    T = sum(lens)
    num_slots = 6
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=torch.int32, device=DEV)
    q = torch.randn(T, H, K, device=DEV)
    k = torch.randn(T, H, K, device=DEV)
    v = torch.randn(T, HV, V, device=DEV)
    a = torch.randn(T, HV, device=DEV)
    b = torch.randn(T, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    state = torch.randn(num_slots, HV, V, K, device=DEV)
    idx = torch.tensor([1, 4], dtype=torch.long, device=DEV)
    has_init = torch.tensor([1, 0], dtype=torch.uint8, device=DEV)
    q, k, v, a, b = rnd(q, k, v, a, b)  # reference sees the DT-rounded activation inputs

    ref_state = state.clone()
    ref_out = torch.zeros(T, HV, V, device=DEV)
    for n in range(N):
        slot = int(idx[n])
        bos = int(cu[n])
        for hv in range(HV):
            hq = hv // (HV // H)
            S = ref_state[slot, hv].clone() if has_init[n] else torch.zeros(V, K, device=DEV)
            for t in range(bos, int(cu[n + 1])):
                qn = F.normalize(q[t, hq], dim=-1, eps=1e-6) * SCALE
                kn = F.normalize(k[t, hq], dim=-1, eps=1e-6)
                g = -torch.exp(A_log[hv]) * _softplus(a[t, hv] + dt_bias[hv])
                beta = torch.sigmoid(b[t, hv])
                o, S = ref_step(S, qn, kn, v[t, hv].clone(), g, beta)
                ref_out[t, hv] = o
            ref_state[slot, hv] = S

    got_state = state.clone()
    got_out = torch.ops.gdn_hip.gdn_prefill(to_dt(q), to_dt(k), to_dt(v), to_dt(a), to_dt(b), A_log, dt_bias, cu, idx,
                                            has_init, got_state, SCALE, 1)
    ok = _report("gdn_prefill.out", got_out, ref_out)
    ok &= _report("gdn_prefill.state", got_state[idx], ref_state[idx])
    return ok


def check_prefill_verify() -> bool:
    """gdn_prefill_verify: the spec-decode oracle. Same recurrence as gdn_prefill, but it MUST also
    snapshot the ssm state AFTER each token into scratch[t, n, hv]. Validate that scratch[t] equals
    the recurrent state after exactly t+1 tokens (a sequential decode-style scan), AND that the final
    out + final ssm_state match the plain gdn_prefill. This is THE bit that makes GDN spec bit-exact:
    the scheduler installs scratch[accepted_count-1] as the post-accept state — no re-advance."""
    lens = [6, 4]  # K+1 verify windows of different lengths
    N = len(lens)
    T = sum(lens)
    max_qlen = max(lens)
    num_slots = 6
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=torch.int32, device=DEV)
    q = torch.randn(T, H, K, device=DEV)
    k = torch.randn(T, H, K, device=DEV)
    v = torch.randn(T, HV, V, device=DEV)
    a = torch.randn(T, HV, device=DEV)
    b = torch.randn(T, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    state = torch.randn(num_slots, HV, V, K, device=DEV)
    idx = torch.tensor([1, 4], dtype=torch.long, device=DEV)
    has_init = torch.tensor([1, 0], dtype=torch.uint8, device=DEV)
    q, k, v, a, b = rnd(q, k, v, a, b)

    # Reference: per-token state via the same recurrent scan as gdn_prefill, capturing state[t].
    ref_state = state.clone()
    ref_out = torch.zeros(T, HV, V, device=DEV)
    ref_scr = torch.zeros(max_qlen, N, HV, V, K, device=DEV)
    for n in range(N):
        slot = int(idx[n])
        bos = int(cu[n])
        for hv in range(HV):
            hq = hv // (HV // H)
            S = ref_state[slot, hv].clone() if has_init[n] else torch.zeros(V, K, device=DEV)
            for t in range(bos, int(cu[n + 1])):
                qn = F.normalize(q[t, hq], dim=-1, eps=1e-6) * SCALE
                kn = F.normalize(k[t, hq], dim=-1, eps=1e-6)
                g = -torch.exp(A_log[hv]) * _softplus(a[t, hv] + dt_bias[hv])
                beta = torch.sigmoid(b[t, hv])
                o, S = ref_step(S, qn, kn, v[t, hv].clone(), g, beta)
                ref_out[t, hv] = o
                ref_scr[t - bos, n, hv] = S  # state AFTER token t
            ref_state[slot, hv] = S

    got_state = state.clone()
    got_out, got_scr = torch.ops.gdn_hip.gdn_prefill_verify(
        to_dt(q), to_dt(k), to_dt(v), to_dt(a), to_dt(b), A_log, dt_bias, cu, idx, has_init,
        got_state, max_qlen, SCALE, 1)
    ok = _report("gdn_prefill_verify.out", got_out, ref_out)
    ok &= _report("gdn_prefill_verify.final_state", got_state[idx], ref_state[idx])
    # per-token scratch: only the first lens[n] t-slots of each seq are written. Check each.
    for n in range(N):
        valid = got_scr[: lens[n], n]   # [lens[n], HV, V, K]
        ok &= _report(f"gdn_prefill_verify.scratch[seq{n}]", valid, ref_scr[: lens[n], n])
    # the LAST written scratch slot per seq must equal the final ssm_state (what plain prefill keeps)
    for n in range(N):
        ok &= _report(f"gdn_prefill_verify.scratch_last==final[seq{n}]",
                      got_scr[lens[n] - 1, n], got_state[int(idx[n])])
    return ok


def check_conv_fwd_verify() -> bool:
    """causal_conv1d_fwd_verify: per-token conv-state capture for spec rollback. scratch[t] = the
    trailing (W-1)-input window AFTER token t (what the next decode would convolve against). Validate
    against the same sliding-window reference as conv_fwd, capturing the window each step."""
    lens = [6, 4]
    N, C, W = 2, 256, 4
    T = sum(lens)
    max_qlen = max(lens)
    num_slots = 6
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=torch.int32, device=DEV)
    x = torch.randn(T, C, device=DEV)
    weight = torch.randn(C, W, device=DEV)
    bias = torch.randn(C, device=DEV)
    state = torch.randn(num_slots, C, W - 1, device=DEV)
    idx = torch.tensor([1, 4], dtype=torch.long, device=DEV)
    has_init = torch.tensor([1, 0], dtype=torch.uint8, device=DEV)
    x = rnd(x)

    ref_state = state.clone()
    ref_out = torch.zeros(T, C, device=DEV)
    ref_scr = torch.zeros(max_qlen, N, C, W - 1, device=DEV)
    for n in range(N):
        slot = int(idx[n])
        bos = int(cu[n])
        hist = ref_state[slot].clone() if has_init[n] else torch.zeros(C, W - 1, device=DEV)
        for t in range(bos, int(cu[n + 1])):
            win = torch.cat([hist, x[t].unsqueeze(-1)], dim=-1)  # [C, W]
            acc = (win * weight).sum(-1) + bias
            ref_out[t] = F.silu(acc)
            hist = win[:, 1:]
            ref_scr[t - bos, n] = hist  # trailing window AFTER token t
        ref_state[slot] = hist

    got_state = state.clone()
    got_out, got_scr = torch.ops.gdn_hip.causal_conv1d_fwd_verify(
        to_dt(x), weight, bias, cu, idx, has_init, got_state, max_qlen, 1)
    ok = _report("conv1d_fwd_verify.out", got_out, ref_out)
    ok &= _report("conv1d_fwd_verify.final_state", got_state[idx], ref_state[idx])
    for n in range(N):
        ok &= _report(f"conv1d_fwd_verify.scratch[seq{n}]", got_scr[: lens[n], n], ref_scr[: lens[n], n])
    for n in range(N):
        ok &= _report(f"conv1d_fwd_verify.scratch_last==final[seq{n}]",
                      got_scr[lens[n] - 1, n], got_state[int(idx[n])])
    return ok


def check_prefill_chunked() -> bool:
    """Chunked prefill vs the recurrent kernel (the validated oracle), on sequences spanning several
    GDN_CHUNK=32 chunks + a partial final chunk. Mild decay (A_log~-2) so gamma doesn't underflow —
    the regime where the chunked ratio formulation and the recurrent step form are comparable."""
    lens = [40, 70]  # 40 = 1.25 chunks; 70 = 2.19 chunks
    N, T = len(lens), sum(lens)
    num_slots = 6
    cu = torch.tensor([0, 40, 110], dtype=torch.int32, device=DEV)
    q = torch.randn(T, H, K, device=DEV)
    k = torch.randn(T, H, K, device=DEV)
    v = torch.randn(T, HV, V, device=DEV)
    a = torch.randn(T, HV, device=DEV)
    b = torch.randn(T, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV) * 0.5 - 2.0  # exp(A_log)~0.05-0.3 -> mild per-token decay
    dt_bias = torch.randn(HV, device=DEV)
    state = torch.randn(num_slots, HV, V, K, device=DEV)
    idx = torch.tensor([1, 4], dtype=torch.long, device=DEV)
    has_init = torch.tensor([1, 0], dtype=torch.uint8, device=DEV)
    q, k, v, a, b = rnd(q, k, v, a, b)  # both kernels see identical DT-rounded inputs

    st_ref = state.clone()
    out_ref = torch.ops.gdn_hip.gdn_prefill(to_dt(q), to_dt(k), to_dt(v), to_dt(a), to_dt(b), A_log, dt_bias, cu, idx,
                                            has_init, st_ref, SCALE, 1)
    st_ch = state.clone()
    out_ch = torch.ops.gdn_hip.gdn_prefill_chunked(to_dt(q), to_dt(k), to_dt(v), to_dt(a), to_dt(b), A_log, dt_bias, cu,
                                                   idx, has_init, st_ch, SCALE, 1)
    ok = _report("prefill_chunked.out (vs recurrent)", out_ch, out_ref, tol_mult=2.0)
    ok &= _report("prefill_chunked.state", st_ch[idx], st_ref[idx], tol_mult=2.0)
    return ok


def check_prefill_wmma() -> bool:
    """WMMA (matrix-core) chunked prefill. The RECURRENT kernel is the ground-truth oracle (per-token
    multiplicative exp(g) — robust to any decay). Two regimes, both on SHORT (<16-token) single
    partial chunks + multi-chunk + partial finals (the geometry mix that a naive kernel got wrong):

      (A) STRONG decay A_log~N(0,.5): cumulative gamma underflows ~1e-3 over a chunk. The naive
          k/gamma fp16 absorption went to NaN here; the stable log-space kernel must stay FINITE and
          match the recurrent oracle. NB: gdn_prefill_chunked (the SCALAR chunked op) ALSO NaNs here
          — it forms gam[j]/gam[i]=0/0 in fp32 — so it is NOT a valid oracle under strong decay; the
          WMMA kernel is strictly more robust. We therefore check (A) against recurrent ONLY.
      (B) MILD decay A_log~N(-2,.5): all three kernels are valid -> three-way agreement, incl. the
          scalar-chunked cross-check.
    fp16 matmul operands -> looser tol (8e-3)."""
    lens = [5, 11, 3, 8, 40, 70, 16, 33]  # short single sub-chunks + multi-chunk + partial finals
    N, T = len(lens), sum(lens)
    num_slots = 16
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=torch.int32, device=DEV)
    q = torch.randn(T, H, K, device=DEV)
    k = torch.randn(T, H, K, device=DEV)
    v = torch.randn(T, HV, V, device=DEV)
    a = torch.randn(T, HV, device=DEV)
    b = torch.randn(T, HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    state = torch.randn(num_slots, HV, V, K, device=DEV)
    idx = torch.tensor([1, 4, 6, 2, 9, 11, 13, 15], dtype=torch.long, device=DEV)
    has_init = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8, device=DEV)
    q, k, v, a, b = rnd(q, k, v, a, b)  # all kernels see identical DT-rounded inputs

    def _args(A_log):
        return (to_dt(q), to_dt(k), to_dt(v), to_dt(a), to_dt(b), A_log, dt_bias, cu, idx, has_init)

    # (A) strong decay -> recurrent oracle only (+ finiteness regression guard)
    A_strong = torch.randn(HV, device=DEV) * 0.5  # exp(A_log)~0.4-2.7
    st_rec, st_w = state.clone(), state.clone()
    out_rec = torch.ops.gdn_hip.gdn_prefill(*_args(A_strong), st_rec, SCALE, 1)
    out_w = torch.ops.gdn_hip.gdn_prefill_wmma(*_args(A_strong), st_w, SCALE, 1)
    fin = torch.isfinite(out_w).all().item() and torch.isfinite(st_w).all().item()
    if not fin:
        print("  [FAIL] prefill_wmma produced non-finite values (NaN/Inf) — decay overflow regression")
    ok = fin
    ok &= _report("prefill_wmma.out  [strong decay] (vs recurrent)", out_w, out_rec, tol_mult=8.0)
    ok &= _report("prefill_wmma.state[strong decay] (vs recurrent)", st_w[idx], st_rec[idx], tol_mult=8.0)

    # (B) mild decay -> three-way agreement (recurrent + scalar-chunked both valid)
    A_mild = torch.randn(HV, device=DEV) * 0.5 - 2.0  # exp(A_log)~0.05-0.3
    st_rec2, st_ch, st_w2 = state.clone(), state.clone(), state.clone()
    out_rec2 = torch.ops.gdn_hip.gdn_prefill(*_args(A_mild), st_rec2, SCALE, 1)
    out_ch = torch.ops.gdn_hip.gdn_prefill_chunked(*_args(A_mild), st_ch, SCALE, 1)
    out_w2 = torch.ops.gdn_hip.gdn_prefill_wmma(*_args(A_mild), st_w2, SCALE, 1)
    ok &= _report("prefill_wmma.out  [mild decay] (vs recurrent)", out_w2, out_rec2, tol_mult=8.0)
    ok &= _report("prefill_wmma.out  [mild decay] (vs scalar-chunked)", out_w2, out_ch, tol_mult=8.0)
    ok &= _report("prefill_wmma.state[mild decay] (vs scalar-chunked)", st_w2[idx], st_ch[idx], tol_mult=8.0)
    return ok


def check_conv_update() -> bool:
    B, C, W = 4, 256, 4
    num_slots = 6
    x = torch.randn(B, C, device=DEV)
    weight = torch.randn(C, W, device=DEV)
    bias = torch.randn(C, device=DEV)
    state = torch.randn(num_slots, C, W - 1, device=DEV)
    idx = torch.tensor([1, 0, 3, 5], dtype=torch.long, device=DEV)
    x = rnd(x)  # reference sees the DT-rounded input

    ref_state = state.clone()
    ref_out = torch.zeros(B, C, device=DEV)
    for bi in range(B):
        slot = int(idx[bi])
        win = torch.zeros(C, W, device=DEV)
        if slot > 0:
            win[:, : W - 1] = ref_state[slot]
        win[:, W - 1] = x[bi]
        acc = (win * weight).sum(-1) + bias
        ref_out[bi] = F.silu(acc)
        if slot > 0:
            ref_state[slot] = win[:, 1:]  # roll left, append new at tail

    got_state = state.clone()
    got_out = torch.ops.gdn_hip.causal_conv1d_update(to_dt(x), weight, bias, got_state, idx, 1)
    ok = _report("conv1d_update.out", got_out, ref_out)
    ok &= _report("conv1d_update.state", got_state[idx[idx > 0]], ref_state[idx[idx > 0]])
    return ok


def check_conv_fwd() -> bool:
    lens = [5, 3]
    N, T, C, W = 2, 8, 256, 4
    num_slots = 6
    cu = torch.tensor([0, 5, 8], dtype=torch.int32, device=DEV)
    x = torch.randn(T, C, device=DEV)
    weight = torch.randn(C, W, device=DEV)
    bias = torch.randn(C, device=DEV)
    state = torch.randn(num_slots, C, W - 1, device=DEV)
    idx = torch.tensor([1, 4], dtype=torch.long, device=DEV)
    has_init = torch.tensor([1, 0], dtype=torch.uint8, device=DEV)
    x = rnd(x)  # reference sees the DT-rounded input

    ref_state = state.clone()
    ref_out = torch.zeros(T, C, device=DEV)
    for n in range(N):
        slot = int(idx[n])
        hist = ref_state[slot].clone() if has_init[n] else torch.zeros(C, W - 1, device=DEV)
        for t in range(int(cu[n]), int(cu[n + 1])):
            win = torch.cat([hist, x[t].unsqueeze(-1)], dim=-1)  # [C, W]
            acc = (win * weight).sum(-1) + bias
            ref_out[t] = F.silu(acc)
            hist = win[:, 1:]
        ref_state[slot] = hist

    got_state = state.clone()
    got_out = torch.ops.gdn_hip.causal_conv1d_fwd(to_dt(x), weight, bias, cu, idx, has_init, got_state, 1)
    ok = _report("conv1d_fwd.out", got_out, ref_out)
    ok &= _report("conv1d_fwd.state", got_state[idx], ref_state[idx])
    return ok


def check_rmsnorm_gated() -> bool:
    M, D = 64, 128
    x = torch.randn(M, D, device=DEV)
    z = torch.randn(M, D, device=DEV)
    weight = torch.randn(D, device=DEV)
    eps = 1e-5
    x, z = rnd(x, z)  # reference sees the DT-rounded inputs
    inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    ref = x * inv * weight * F.silu(z)
    got = torch.ops.gdn_hip.rmsnorm_gated(to_dt(x), to_dt(z), weight, eps)
    return _report("rmsnorm_gated", got, ref)


def check_ssm_state_bf16(steps: int = 512) -> bool:
    """bf16 ssm_state cache (MINISGL_SSM_BF16): run the SAME decode stream through an fp32-state and a
    bf16-state buffer in lockstep and compare. Tests (1) the per-step cast load/store is correct and
    (2) long-context recurrent STABILITY — the gated decay (S*=exp(g), g<0) is contractive, so bf16
    state-storage error must stay BOUNDED over many steps (this is what the Triton path relied on),
    not blow up. Inputs are bf16 (the serve I/O dtype)."""
    torch.manual_seed(1234)  # deterministic (the metric is a vector L2/cos, not a noisy max-element)
    B, num_slots = 2, 4
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    idx = torch.tensor([1, 3], dtype=torch.long, device=DEV)
    st32 = torch.zeros(num_slots, HV, V, K, device=DEV, dtype=torch.float32)
    st16 = torch.zeros(num_slots, HV, V, K, device=DEV, dtype=torch.bfloat16)
    worst_l2, worst_cos = 0.0, 1.0
    o16 = None
    for _ in range(steps):
        q = torch.randn(B, H, K, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(B, H, K, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(B, HV, V, device=DEV, dtype=torch.bfloat16)
        a = torch.randn(B, HV, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(B, HV, device=DEV, dtype=torch.bfloat16)
        o32 = torch.ops.gdn_hip.gdn_decode(q, k, v, a, b, A_log, dt_bias, st32, idx, SCALE, 1)
        o16 = torch.ops.gdn_hip.gdn_decode(q, k, v, a, b, A_log, dt_bias, st16, idx, SCALE, 1)
        f32, f16 = o32.float().flatten(), o16.float().flatten()
        worst_l2 = max(worst_l2, ((f16 - f32).norm() / (f32.norm() + 1e-9)).item())
        worst_cos = min(worst_cos, F.cosine_similarity(f16, f32, dim=0).item())
    # bf16-state storage must leave the decode OUTPUT directionally intact over a long stream: relative
    # L2 small and cosine ~1 (the contractive gated decay keeps the bf16 rounding BOUNDED, not
    # accumulating). L2/cos are robust vector metrics (the per-element max/mean ratio is too noisy here).
    ok = (worst_l2 < 0.05) and (worst_cos > 0.998) and bool(torch.isfinite(o16).all().item())
    print(f"  [{'PASS' if ok else 'FAIL'}] ssm_state bf16 ({steps} decode steps) "
          f"worst_out_L2rel={worst_l2:.3e} (<0.05)  worst_out_cos={worst_cos:.5f} (>0.998)")
    return ok


def main() -> None:
    global DT
    assert torch.cuda.is_available(), "needs a GPU (run under a lease)"
    print(f"=== gdn_hip parity vs torch reference (device={torch.cuda.get_device_name()}) ===")
    checks = {
        "gdn_decode": check_decode,
        "gdn_prefill": check_prefill,
        "gdn_prefill_verify": check_prefill_verify,
        "causal_conv1d_fwd_verify": check_conv_fwd_verify,
        "gdn_prefill_chunked": check_prefill_chunked,
        "gdn_prefill_wmma": check_prefill_wmma,
        "causal_conv1d_update": check_conv_update,
        "causal_conv1d_fwd": check_conv_fwd,
        "rmsnorm_gated": check_rmsnorm_gated,
    }
    allok = True
    for dt, label in [(torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        DT = dt
        torch.manual_seed(0)  # identical random inputs across dtypes (comparability)
        print(f"\n--- I/O dtype: {label}  (base rel-thr={_BASE_THR[dt]:.1e}) ---")
        results = {n: fn() for n, fn in checks.items()}
        ok = all(results.values())
        allok &= ok
        print(f"  >>> {label}: {'ALL PASS' if ok else 'FAIL'}")
    print("\n--- bf16 ssm_state cache (recurrent stability) ---")
    allok &= check_ssm_state_bf16()
    print("\n" + "=" * 60)
    print("RESULT:", "ALL PASS — gdn_hip numerics faithful across fp32/fp16/bf16 (+ bf16 ssm_state)"
          if allok else "FAIL (see above)")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
