"""TWO-BUILD bit-exactness recorder for the surface commit 8acbc8e (e4m3 WLoad policy) rewrote.

WHY THIS FILE EXISTS. The e4m3 commit's own message says, verbatim: "NOT bit-exact-proven for the
MoE int4 gemv/tiled/fused-silu paths — that needs the repo's GPU max|delta|=0 A/B, which this
workflow could not run. Do not report those as bit-exact until it does." The in-tree recorders that
support the two-build `.pt` protocol (`parity_gemv.py`, `parity_w4a16_moe_gemv.py`) do NOT reach
that surface: they fix group_size to 32/128, and the served checkpoint is NVFP4 group 16; and they
only ever ask `kernel="gemv"`, so the register-tiled `wmma` arms — including
`moe_gemm1_silu_{alds,ashuffle}`, the hunk the commit itself names as highest-risk — are never
launched.

WHAT IS BEING COMPARED. Not e4m3-vs-fp16 (those are different functions of different inputs, and
the e4m3 policy does not exist on the baseline build at all). The question is the REGRESSION
question: the shipped fp16-group-scale path was refactored underneath the new policy — four accum
bodies collapsed onto one `consume_chunk()`, two hand-written gemm1_silu kernels moved onto the
shared `WLoad::stage_b`, a `wscale_epi` hook added to the dense SILU epilogue — so does the fp16
path still compute exactly what it computed before? Every case here is therefore recorded on BOTH
builds with fp16 scales, and the gate is max|delta| == 0.

THE FLOOR IS ESTABLISHED FIRST, per tensor, and this is not ceremony. `mmq_fp8_moe_gemm_scatter`
accumulates through a global atomicAdd whose reduction ORDER varies run to run; the engine's own
`quant/kernels.py` documents it as not bit-exact. Comparing two builds on such a tensor would
manufacture a failure out of a known property. So each build is dumped TWICE, in two separate
processes, and a cross-build delta is only evidence where the same-build delta is 0.

Usage (once per build):
    PARITY_OUT=/out/base_a python parity_e4m3_merge.py
    PARITY_OUT=/out/base_b python parity_e4m3_merge.py
then `compare_parity.py` pairwise.
"""
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import fp8_wmma as F  # noqa: E402  (resolved via PYTHONPATH to the build under test)

DEV = "cuda"
OUT = os.environ["PARITY_OUT"]
os.makedirs(OUT, exist_ok=True)

results: dict = {}
errors: dict = {}

# The SERVED geometry of Qwen3.8-Flash-Next at TP=2, which is the only reason these particular
# numbers appear: hidden 2560, moe_intermediate 640 -> 320 per rank, group_size 16 (NVFP4),
# top_k 10. E is cut to 16 so the probe fits beside a live serve's leftovers; E does not select a
# kernel, K/N/group/block_m do.
HID = 2560
INTER = 320
GROUP = 16
TOPK = 10
E = 16
BLOCK_M = 32


def build_routing(T, e_count, top_k, block_m, seed=0):
    """moe_align emulation — copied from local/bench_moe_flag_w4.py so this file does not depend on
    the package's cwd-relative `local.` import working inside two different trees."""
    g = torch.Generator().manual_seed(seed)
    M = T * top_k
    experts = torch.randint(0, e_count, (M,), generator=g)
    order = torch.argsort(experts, stable=True)
    exp_sorted = experts[order]
    ids_sorted = order.to(torch.int32)
    btok, bexp = [], []
    for e in range(e_count):
        rows = ids_sorted[exp_sorted == e]
        n = rows.numel()
        npad = ((n + block_m - 1) // block_m) * block_m
        run = torch.cat([rows, torch.full((npad - n,), M, dtype=torch.int32)])
        for b in range(npad // block_m):
            btok.append(run[b * block_m:(b + 1) * block_m])
            bexp.append(e)
    # THE PAD SENTINEL IS AN OUT-OF-RANGE ROW INDEX, and it is dereferenced.
    # `moe_align` pads each expert's run with `M` (= T*top_k), one past the last real row, and the
    # grouped DECODE arm (`mmq_fp8_moe_gemm1_silu(kernel="gemv")`) LOADS `x[sorted_token_ids[p]]`
    # before it masks. With `x` allocated at exactly M rows that read runs off the end of the
    # allocation: MEASURED here as `Memory access fault by GPU node-1 ... Page not present` on the
    # served NVFP4 shape (K=2560, 2*inter=640, group 16, block_m 32, top_k 10). The in-tree
    # recorders build routing the same way and never hit it only because their tensors are small
    # enough that the extra row lands in allocator slack.
    # Callers here therefore over-allocate `x` by one block_m. The valid rows are untouched
    # (torch.randn fills row-major, so rows 0..M-1 are identical either way) and the pad rows are
    # discarded by `rec_moe`.
    sti = torch.cat(btok).to(DEV)
    eid = torch.tensor(bexp, dtype=torch.int32, device=DEV)
    ntp = torch.tensor([sti.numel()], dtype=torch.int32, device=DEV)
    return sti, eid, ntp, M


_xg = torch.Generator(device=DEV)


def xrand(*shape, dt, seed):
    _xg.manual_seed(seed)
    return torch.randn(*shape, device=DEV, dtype=dt, generator=_xg) * 0.1


def rand_w4(e_count, N, K, group, seed):
    """Packed int4 weights + fp16 GROUP scales + int32 packed zeros, in the (E, K/group, ...)
    group-major layout the bindings assert."""
    ng = K // group
    g = torch.Generator(device=DEV).manual_seed(seed)
    wp = torch.randint(-(2**31), 2**31 - 1, (e_count, N, K // 8),
                       dtype=torch.int32, device=DEV, generator=g)
    sc = (torch.rand(e_count, N, ng, device=DEV, generator=g) * 0.02 + 0.004).half()
    wz = torch.randint(-(2**31), 2**31 - 1, (e_count, N // 8, ng),
                       dtype=torch.int32, device=DEV, generator=g)
    return wp, sc.transpose(-2, -1).contiguous(), wz.transpose(-2, -1).contiguous()


def rec(name, t):
    results[name] = t.detach().float().cpu().clone()


def rec_moe(name, out, sti, M):
    """Record a grouped-MoE (P, N) result SPLIT into its valid rows and its PAD rows.

    THIS SPLIT IS THE WHOLE GATE. `moe_align` pads each expert's run up to a multiple of block_m
    with the sentinel token id `M`, which is out of range by construction, and the kernels skip
    those slots — so the corresponding output rows are never written and still hold whatever
    `torch.empty` handed them. A recorder that dumps the raw (P, N) tensor is therefore comparing
    uninitialized memory, and it does not fail loudly: it reports the op as nondeterministic and
    every downstream verdict becomes "cannot discriminate".

    That is not hypothetical. The previous round's gate log (`logs/gate_a.log`) called 215 of its
    360 tensors self-nondeterministic, and the ops it DID find reproducible are dominated by
    `g2fuse` — whose output is (M, N) and has no pad rows at all. That gate could therefore not
    answer the question it was built for. Whether every remaining case is the same cause is not
    assumed here: the split makes it MEASURABLE, because a pad-row block that is noise on both
    builds while the valid block is bit-exact says so directly.

    So the valid rows are recorded under `name` (the gate), and the pad rows under `name#pad`
    (evidence, expected to be noise on both builds — recorded rather than dropped so the claim
    "the pad rows are the noise" is checkable from the dumps instead of asserted here).
    """
    valid = (sti < M)
    rec(name, out[valid])
    npad = int((~valid).sum().item())
    if npad:
        rec(f"{name}#pad", out[~valid])


ONLY = os.environ.get("PARITY_ONLY", "")     # substring filter, for bisecting a hard fault
SKIP = [s for s in os.environ.get("PARITY_SKIP", "").split(",") if s]


def case(name):
    """Run a recorded case, keeping a refusal as DATA. A binding that rejects a shape on one build
    and accepts it on the other is itself a finding, and a Python-level crash here would lose every
    case after it.

    A GPU MEMORY FAULT IS NOT CATCHABLE — it kills the process — so the name is printed BEFORE the
    launch and the queue is drained after it. Without both, an async fault is attributed to whatever
    happened to be running when the driver noticed, which is how a bad shape gets blamed on an
    innocent kernel."""
    def deco(fn):
        if ONLY and ONLY not in name:
            return fn
        if any(s and s in name for s in SKIP):
            print(f"  SKIP    {name} (PARITY_SKIP)", flush=True)
            errors[name] = "skipped by PARITY_SKIP"
            return fn
        print(f"  run     {name}", flush=True)
        try:
            fn()
            torch.cuda.synchronize()
        except BaseException as e:  # noqa: BLE001 - recorded, never swallowed
            errors[name] = f"{type(e).__name__}: {e}"
            print(f"  REFUSED {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return fn
    return deco


# =============================================================================================
# 1. DENSE decode GEMV + the SILU epilogue.  gemv_decode.h: the four accum bodies collapsed onto
#    consume_chunk(), and :1588's SILU epilogue gained a wscale_epi call it never had. If that hook
#    is not a no-op for the fp16 policy, THIS is where it shows.
# =============================================================================================
for dt in (torch.float16, torch.bfloat16):
    for group in (16, 32, 128):
        for e2 in (False, True):
            if group == 16 and not e2:
                continue  # group 16 is the NVFP4 tiling; AWQ zeros do not come at 16

            @case(f"dense_gemv/{dt}/g{group}/e2{e2}")
            def _(dt=dt, group=group, e2=e2):
                N, K = 512, HID
                wp, sc, wz = rand_w4(1, N, K, group, seed=100 + group + int(e2))
                for M in (1, 2, 4, 8):
                    x = xrand(M, K, dt=dt, seed=1000 + M + group)
                    o = F.mmq_fp8_gemm(x, wp[0], sc[0], kernel="decode_gemv",
                                       w_zeros=None if e2 else wz[0], weight_is_e2m1=e2)
                    rec(f"dense_gemv/{dt}/g{group}/e2{e2}/M{M}", o)

            @case(f"dense_gemv_silu/{dt}/g{group}/e2{e2}")
            def _(dt=dt, group=group, e2=e2):
                N, K = 2 * INTER, HID          # N = 2*inter, the gated layout the epilogue assumes
                wp, sc, wz = rand_w4(1, N, K, group, seed=200 + group + int(e2))
                for M in (1, 2, 4, 8):
                    x = xrand(M, K, dt=dt, seed=1500 + M + group)
                    o = F.mmq_fp8_gemm_silu(x, wp[0], sc[0],
                                            w_zeros=None if e2 else wz[0], weight_is_e2m1=e2)
                    rec(f"dense_gemv_silu/{dt}/g{group}/e2{e2}/M{M}", o)

# =============================================================================================
# 2. GROUPED (MoE) gemm1 + fused SILU, BOTH arms.  kernel="wmma" is the register-tiled
#    moe_gemm1_silu_{alds,ashuffle} pair — the hunk 8acbc8e names as highest-risk, because it
#    replaced a per-nibble decode loop with the v_perm E2M1 gather and byte-parallel unpack.
#    kernel="gemv" is the decode arm. T is swept across the tile/gemv crossover.
# =============================================================================================
for dt in (torch.float16, torch.bfloat16):
    for group in (16, 32, 128):
        for e2 in (False, True):
            if group == 16 and not e2:
                continue
            for kern in ("wmma", "gemv"):

                @case(f"moe_gemm1_silu/{dt}/g{group}/e2{e2}/{kern}")
                def _(dt=dt, group=group, e2=e2, kern=kern):
                    K, N = HID, 2 * INTER
                    wp, sc, wz = rand_w4(E, N, K, group, seed=300 + group + int(e2))
                    for T in (1, 2, 8, 64):
                        sti, eid, ntp, M = build_routing(T, E, TOPK, BLOCK_M, seed=T)
                        x = xrand(M + BLOCK_M, K, dt=dt, seed=2000 + T + group)
                        o = F.mmq_fp8_moe_gemm1_silu(
                            x, wp, sc, sti, eid, ntp, TOPK, BLOCK_M, kernel=kern,
                            w_zeros=None if e2 else wz, weight_is_e2m1=e2)
                        rec_moe(f"moe_gemm1_silu/{dt}/g{group}/e2{e2}/{kern}/T{T}", o, sti, M)

                @case(f"moe_gemm/{dt}/g{group}/e2{e2}/{kern}")
                def _(dt=dt, group=group, e2=e2, kern=kern):
                    K, N = INTER, HID          # the gemm2 (down-proj) shape
                    wp, sc, wz = rand_w4(E, N, K, group, seed=400 + group + int(e2))
                    for T in (1, 2, 8, 64):
                        sti, eid, ntp, M = build_routing(T, E, TOPK, BLOCK_M, seed=T)
                        x = xrand(M + BLOCK_M, K, dt=dt, seed=2500 + T + group)
                        o = F.mmq_fp8_moe_gemm(
                            x, wp, sc, sti, eid, ntp, TOPK, BLOCK_M, kernel=kern,
                            w_zeros=None if e2 else wz, weight_is_e2m1=e2)
                        rec_moe(f"moe_gemm/{dt}/g{group}/e2{e2}/{kern}/T{T}", o, sti, M)

# =============================================================================================
# 3. FUSED decode gemm2 + gather-reduce (gemv_decode.h's BYLANE core — four loop shapes over the
#    one consume_chunk this commit created).
# =============================================================================================
for dt in (torch.float16,):
    for group in (16, 32):
        for e2 in (False, True):
            if group == 16 and not e2:
                continue

            @case(f"moe_g2_gather_reduce/{dt}/g{group}/e2{e2}")
            def _(dt=dt, group=group, e2=e2):
                K, N = INTER, HID
                wp, sc, wz = rand_w4(E, N, K, group, seed=500 + group + int(e2))
                for T in (1, 2, 4):
                    sti, eid, ntp, M = build_routing(T, E, TOPK, BLOCK_M, seed=T)
                    P = int(sti.numel())
                    buf2 = xrand(P, K, dt=dt, seed=3000 + T + group)
                    # SEEDED. Unseeded `torch.rand` draws from the global RNG, which differs
                    # per PROCESS — so the two dumps of the SAME build got different routing
                    # weights and the op looked run-to-run nondeterministic. `topk_weights` is
                    # consumed by exactly the two ops that showed a floor, which is what gave
                    # it away. The floor must measure the KERNEL, not the harness's inputs.
                    _twg = torch.Generator(device=DEV).manual_seed(9000 + T)
                    # (T, top_k), NOT (T*top_k, top_k). gather_reduce REDUCES over the top_k axis,
                    # so its output has one row per TOKEN and `topk_weights.shape[0]` is what sizes
                    # it. Passing M = T*top_k made the op allocate T*top_k output rows of which only
                    # the first T were ever written — the other rows were uninitialised, and that,
                    # not the kernel, was the 0.03-0.5 "run-to-run floor" this op appeared to have.
                    tw = torch.rand(T, TOPK, device=DEV, dtype=torch.float32, generator=_twg)
                    o = F.mmq_fp8_moe_gemm2_gather_reduce(
                        buf2, wp, sc, sti, eid, ntp, tw, TOPK, BLOCK_M,
                        w_zeros=None if e2 else wz, weight_is_e2m1=e2)
                    rec(f"moe_g2_gather_reduce/{dt}/g{group}/e2{e2}/T{T}", o)

# =============================================================================================
# 4. TILED SCATTER (moe_gemm_tiled.h).  ATOMIC — recorded so the same-build floor can price it,
#    NOT because a cross-build delta here would be evidence on its own.
# =============================================================================================
for group in (16, 32):
    for e2 in (False, True):
        if group == 16 and not e2:
            continue
        for sk in (1, 2):

            @case(f"moe_scatter/g{group}/e2{e2}/sk{sk}")
            def _(group=group, e2=e2, sk=sk):
                K, N = INTER, HID
                wp, sc, wz = rand_w4(E, N, K, group, seed=600 + group + int(e2))
                for T in (2, 8):
                    sti, eid, ntp, M = build_routing(T, E, TOPK, BLOCK_M, seed=T)
                    P = int(sti.numel())
                    x = xrand(P, K, dt=torch.float16, seed=3500 + T + group)
                    # SEEDED. Unseeded `torch.rand` draws from the global RNG, which differs
                    # per PROCESS — so the two dumps of the SAME build got different routing
                    # weights and the op looked run-to-run nondeterministic. `topk_weights` is
                    # consumed by exactly the two ops that showed a floor, which is what gave
                    # it away. The floor must measure the KERNEL, not the harness's inputs.
                    _twg = torch.Generator(device=DEV).manual_seed(9000 + T)
                    tw = torch.rand(M, TOPK, device=DEV, dtype=torch.float32, generator=_twg)
                    out = torch.zeros(M, N, device=DEV, dtype=torch.float32)
                    F.mmq_fp8_moe_gemm_scatter(
                        x, wp, sc, sti, eid, ntp, tw, out, TOPK, BLOCK_M, kernel="wmma",
                        w_zeros=None if e2 else wz, weight_is_e2m1=e2, split_k=sk)
                    rec(f"moe_scatter/g{group}/e2{e2}/sk{sk}/T{T}", out)

# =============================================================================================
# 5. The FLAG (register-tiled flagship) kernels — moe_gemm_flag.h took 4 lines in this commit.
#    They gate on group_size in {32, 64, 128} and block_m == 128, so NVFP4's 16 never reaches them;
#    that is exactly why they are asked here at 32/128 rather than assumed unaffected.
# =============================================================================================
for dt in (torch.float16, torch.bfloat16):
    for group in (32, 128):
        for e2 in (False, True):

            @case(f"moe_flag/{dt}/g{group}/e2{e2}")
            def _(dt=dt, group=group, e2=e2):
                K, N = HID, 2 * INTER
                bm = 128
                wp, sc, wz = rand_w4(E, N, K, group, seed=700 + group + int(e2))
                wp2, sc2, wz2 = rand_w4(E, HID, INTER, group, seed=800 + group + int(e2))
                for T in (8, 64):
                    sti, eid, ntp, M = build_routing(T, E, TOPK, bm, seed=T)
                    x = xrand(M + bm, K, dt=dt, seed=4000 + T + group)
                    o1 = F.mmq_fp8_moe_gemm1_silu_flag(
                        x, wp, sc, sti, eid, ntp, TOPK, bm,
                        w_zeros=None if e2 else wz, weight_is_e2m1=e2)
                    rec_moe(f"moe_flag_silu/{dt}/g{group}/e2{e2}/T{T}", o1, sti, M)
                    x2 = xrand(M + bm, INTER, dt=dt, seed=4500 + T + group)
                    o2 = F.mmq_fp8_moe_gemm_flag(
                        x2, wp2, sc2, sti, eid, ntp, TOPK, bm,
                        w_zeros=None if e2 else wz2, weight_is_e2m1=e2)
                    rec_moe(f"moe_flag_gemm/{dt}/g{group}/e2{e2}/T{T}", o2, sti, M)


torch.save(results, os.path.join(OUT, "out.pt"))
with open(os.path.join(OUT, "errors.txt"), "w") as f:
    for k in sorted(errors):
        f.write(f"{k}\t{errors[k]}\n")
print(f"\nsaved {len(results)} tensors, {len(errors)} refused cases -> {OUT}", flush=True)
