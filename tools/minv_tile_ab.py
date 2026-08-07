"""End-to-end A/B of the minv tile selection, through the real `minv_linear`, against rocBLAS.

WHAT WENT WRONG IN THE PREVIOUS HARNESS (`minv_ab.py`) — two independent defects, both the same
root cause as the three already recorded in docs/CONTINUE_bf16_kernel_parity.md: the harness and
the thing it measures silently disagreeing.

  1. IT RAN HOT. It built a rotation of weight copies and then called `Ws[0]` every time. A hot B
     flatters `rd` specifically, because rd's entire cost model is re-reading B and a resident B
     makes the re-read free. Note the obvious fix is NOT enough: a counter inside the timed lambda
     still measures hot, because `torch.cuda.CUDAGraph` bakes the weight POINTER in at capture and
     every replay re-reads that one buffer. The rotation has to be unrolled INTO the graph —
     `reps` distinct captured calls over `ncopy` distinct buffers — which is what `gbench` below
     does, and what dense_gemm/local/sweep_policy.py always did.

  2. THE BASELINE LEG WAS NOT THE OLD CODE. It emulated "pinned" by setting the new
     `_BLOCK_M_OVERRIDE/_BN_OVERRIDE` to 64/64. But the OLD code never fed `_BLOCK_M` to the pipe
     arm at all — pipe ran its own staircase, `pbm = 256 if (M>=512 or (OUT>=65536 and M>=192))
     else (128 if M>=256 else 64)`, with `mi = _PIPE_MI if pbm>=256 else 1`. Forcing the override
     to 64 pins pipe at pbm=64/mi=1 everywhere, a config the old engine would never launch above
     M=256. That fabricates a slow baseline on exactly the pipe cells, and it is why `lm_head
     M=512` appeared to be a 3.02x -> 1.18x rout: the derived rule picks pbm=256 there, which is
     what the old code ALREADY did. The cell was new-vs-itself with a handicap.

     So this harness does not emulate. It loads the two real source files as two modules and calls
     both. `--old-src` is the pre-change minv.py (`git show <base>:python/minisgl/layers/minv.py`).

Both legs run the SAME kernel build (`dense_gemm` from /opt/kernels) — only the Python selection
differs — so this isolates the selection change from everything else.

PROVENANCE (the rule that makes a green A/B mean anything): the harness refuses to run if the two
sources hash the same, records the ACTUAL kernel dispatch each leg makes per cell by wrapping the
`dense_gemm` entry points, prints them, and reports how many cells actually differ. A cell where
old and new dispatch identically MUST come out at ~0%; if it does not, the measurement is noise
and the whole column should be distrusted.

Timed as CUDA-graph replay (device time; a per-call sync has a ~40 us wall floor on this box,
larger than most of these kernels).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import time

import torch
import torch.nn.functional as F

DEV = "cuda"
MALL_BYTES = 64 << 20      # gfx1201 Infinity Cache; the weight rotation must EXCEED it to be cold
COLD_TARGET = 2 * MALL_BYTES
MAX_COPIES = 384
MAX_REPS = 512

SHAPES = {                 # (IN, OUT) — what the engine actually routes through minv
    "mlp.gate_up": (2816, 2112),
    "mlp.down": (1056, 2816),
    "router": (2816, 128),
    "lm_head": (2816, 131072),
    "qkv_proj": (2816, 4096),
    "o_proj": (2048, 2816),
}
MS = [64, 128, 192, 256, 512]   # M<=16 peels to the decode GEMV, not dense_gemm


# ---- provenance: record what the selection actually dispatched -----------------------------------
_REC: list[tuple] = []
_RECORDING = False


def _install_recorder(dg):
    # Must cover the _sk arms too. When it did not, the split-K router reported its dispatch as
    # "F.linear" (no record -> assumed fallback) while the timings plainly showed the split arm
    # running — a harness mislabelling the very thing it exists to prove.
    names = ("dense_gemm", "dense_gemm_rd", "dense_gemm_pipe",
             "dense_gemm_rd_sk", "dense_gemm_pipe_sk")
    names = tuple(n for n in names if hasattr(dg, n))
    orig = {n: getattr(dg, n) for n in names}

    def mk(n, f):
        def g(*a, **kw):
            if _RECORDING:
                # a = (x, weight, block_m, BN[, mi, pbk]) -> keep the tile knobs only
                _REC.append((n, a[0].shape[0]) + tuple(a[2:]))
            return f(*a, **kw)
        return g

    for n in names:
        setattr(dg, n, mk(n, orig[n]))
    return orig


def _dispatch_of(fn, x, w) -> str:
    """Run once with recording on and render the (arm, tile) the selection chose."""
    global _RECORDING
    _REC.clear()
    _RECORDING = True
    try:
        fn(x, w)
    finally:
        _RECORDING = False
    if not _REC:
        return "F.linear"          # fell back to rocBLAS inside minv_linear
    n, mpad, *rest = _REC[-1]
    if n == "dense_gemm_rd_sk":
        bm, bn, _sk, grid = (list(rest) + [0, True])[:4]
        return f"rd_sk bm{bm}/bn{bn}/{'grid' if grid else 'slice'}"
    if n == "dense_gemm_pipe_sk":
        return f"pipe_sk bm{rest[0]}/bn{rest[1]}"
    arm = {"dense_gemm": "lds", "dense_gemm_rd": "rd", "dense_gemm_pipe": "pipe"}[n]
    if arm == "pipe":
        bm, bn, mi, pbk = rest
        return f"pipe bm{bm}/bn{bn}/mi{mi}/pbk{pbk}"
    bm, bn = rest[0], rest[1]
    return f"{arm} bm{bm}/bn{bn}"


# ---- timing ---------------------------------------------------------------------------------------
def gbench(mk, ncopy: int, rounds: int = 7):
    """mk(i) -> callable using weight copy i. Median per-call device us via graph replay.

    `reps` is forced to a multiple of `ncopy` and to at least `ncopy`, so every rotation buffer is
    actually touched inside the graph. Capping reps below ncopy would silently shrink the working
    set back under the MALL and re-create the hot measurement this harness exists to avoid.
    """
    try:
        mk(0)()
    except Exception:
        return None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        mk(0)()
    torch.cuda.synchronize()
    est = (time.perf_counter() - t0) * 1e6 / 3
    target = int(max(8, min(400, 8000.0 / max(est, 1.0))))
    reps = ncopy * max(1, -(-target // ncopy))
    if reps > MAX_REPS:
        reps = ncopy * max(1, MAX_REPS // ncopy)
    fns = [mk(i) for i in range(reps)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:4]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for f in fns:
                f()
    except Exception:
        return None
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6 / reps)
    ts.sort()
    del g, fns
    return ts[len(ts) // 2]


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-src", required=True, help="pre-change minv.py (git show <base>:...)")
    ap.add_argument("--new-src", default=os.path.join(os.path.dirname(__file__), "..", "python",
                                                      "minisgl", "layers", "minv.py"))
    ap.add_argument("--hot", action="store_true", help="diagnostic only: reproduce the old bug")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--ms", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--require-local-kernels", action="store_true",
                    help="fail if dense_gemm resolves to the baked /opt/kernels build")
    args = ap.parse_args()

    # PROVENANCE GATE 1: the two legs must be different source.
    def h(p):
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    ho, hn = h(args.old_src), h(args.new_src)
    if ho == hn:
        print(f"FATAL: old and new minv.py are the same file ({ho}) — this A/B would be "
              f"new-vs-itself.", file=sys.stderr)
        return 2

    # The engage ledger logs through the rank-0 logger, which asserts TP info exists; a bare probe
    # never went through engine init.
    from minisgl.distributed.info import set_tp_info
    set_tp_info(0, 1)

    # Neither leg may see the override env vars: they would make BOTH legs pinned.
    for v in ("MINISGL_MINV_BLOCK_M", "MINISGL_MINV_BN"):
        if os.environ.pop(v, None) is not None:
            print(f"# unset {v} (it would have pinned both legs)")

    old = _load("_minv_old", os.path.abspath(args.old_src))
    new = _load("_minv_new", os.path.abspath(args.new_src))
    import dense_gemm as dg
    # PROVENANCE GATE 3: say WHICH kernel build both legs are running. The serve image ships
    # PYTHONPATH=/opt/kernels, so a locally-built dense_gemm only wins if it is ahead of it — and a
    # silent fall-back to the baked build is how a kernel change gets measured as "no effect".
    print(f"# dense_gemm: {dg.__file__}")
    if args.require_local_kernels and "/opt/kernels" in os.path.abspath(dg.__file__):
        print("FATAL: dense_gemm resolved to the BAKED /opt/kernels build, not a local one.",
              file=sys.stderr)
        return 2
    _install_recorder(dg)

    # PROVENANCE GATE 2: the pre-change module must really be pre-change.
    if getattr(old, "_BLOCK_M", None) != 64 or getattr(old, "_BN", None) != 64:
        print(f"FATAL: --old-src does not look like the pinned version "
              f"(_BLOCK_M={getattr(old, '_BLOCK_M', None)}, _BN={getattr(old, '_BN', None)})",
              file=sys.stderr)
        return 2

    names = list(SHAPES) if not args.shapes else args.shapes.split(",")
    ms = MS if not args.ms else [int(x) for x in args.ms.split(",")]
    mode = "HOT (BUGGED, diagnostic)" if args.hot else "COLD (rotating W past the MALL)"
    print(f"# dev={torch.cuda.get_device_name(0)}  mode={mode}")
    print(f"# old={args.old_src} [{ho}]   new={args.new_src} [{hn}]")
    print(f"{'shape':<12}{'M':>5}{'roc':>9}{'old':>9}{'new':>9} | {'old/roc':>8}{'new/roc':>8}"
          f"{'change':>8}  {'old dispatch':<26}{'new dispatch':<26}")
    print("-" * 132)

    torch.manual_seed(0)
    tot_r = tot_o = tot_n = 0.0
    beat_o = beat_n = ncell = ndiff = nbug = 0
    rows = []
    for name in names:
        IN, OUT = SHAPES[name]
        wbytes = OUT * IN * 2
        ncopy = 1 if args.hot else max(1, min(MAX_COPIES, COLD_TARGET // wbytes + 1))
        Ws = [(torch.randn(OUT, IN, device=DEV) * 0.02).to(torch.bfloat16) for _ in range(ncopy)]
        for M in ms:
            x = torch.randn(M, IN, device=DEV).to(torch.bfloat16)
            do = _dispatch_of(old.minv_linear, x, Ws[0])
            dn = _dispatch_of(new.minv_linear, x, Ws[0])
            # the arms are all bit-identical to each other; a nonzero delta is a real bug
            delta = (old.minv_linear(x, Ws[0]).float()
                     - new.minv_linear(x, Ws[0]).float()).abs().max().item()
            roc = gbench(lambda i: (lambda: F.linear(x, Ws[i % ncopy])), ncopy)
            to = gbench(lambda i: (lambda: old.minv_linear(x, Ws[i % ncopy])), ncopy)
            tn = gbench(lambda i: (lambda: new.minv_linear(x, Ws[i % ncopy])), ncopy)
            tot_r += roc; tot_o += to; tot_n += tn; ncell += 1
            beat_o += to < roc; beat_n += tn < roc
            same = do == dn
            ndiff += not same
            # A reduction-ORDER change is an intended bit-move, not a defect: the split-K arms
            # reassociate K on purpose. Any OTHER nonzero delta is a real bug, because every
            # non-split arm is supposed to be bit-identical to every other.
            order_move = ("_sk" in do) != ("_sk" in dn)
            if delta == 0.0:
                flag = ""
            elif order_move:
                flag = f"  [order change, expected: max|d|={delta:.3e}]"
            else:
                flag = f"  !!BITDIFF {delta:.3e}"
                nbug += 1
            if same:
                flag += "  [same dispatch -> expect ~0%]"
            print(f"{name:<12}{M:>5}{roc:>9.1f}{to:>9.1f}{tn:>9.1f} | {to/roc:>7.2f}x{tn/roc:>7.2f}x"
                  f"{(tn-to)/to*100:>+7.1f}%  {do:<26}{dn:<26}{flag}")
            sys.stdout.flush()
            rows.append((name, IN, OUT, M, roc, to, tn, do, dn, delta))
            del x
        del Ws
        torch.cuda.empty_cache()
    print("-" * 132)

    # ---- M-INVARIANCE of the SHIPPED path -------------------------------------------------------
    # Once a shape can change reduction order OR schedule, "old vs new" stops being the load-bearing
    # correctness claim; "new vs itself at every M" becomes it. A token's result must not depend on
    # the batch it arrived in — that is what prefix caching, chunked prefill and spec-verify assume.
    # Chunks stay above _DECODE_GEMV_MAXM=16: minv_linear peels M<=16 to the decode GEMV, a
    # deliberate, pre-existing order crossing that this change neither adds to nor removes.
    print("M-INVARIANCE of the new path (same rows, different batch shapes, must be BIT-IDENTICAL)")
    NROW = 512
    minv_ok = True
    for name in names:
        IN, OUT = SHAPES[name]
        W = (torch.randn(OUT, IN, device=DEV) * 0.02).to(torch.bfloat16)
        h = torch.randn(NROW, IN, device=DEV).to(torch.bfloat16)
        ref = new.minv_linear(h, W)
        bad = []
        for chunk in (32, 64, 128, 192, 256):
            got = torch.cat([new.minv_linear(h[i:i + chunk], W)
                             for i in range(0, NROW, chunk)], dim=0)
            if not torch.equal(ref, got):
                bad.append((chunk, (ref.float() - got.float()).abs().max().item()))
        minv_ok &= not bad
        detail = "OK" if not bad else "  ".join(f"chunk{c}:max|d|={d:.3e}" for c, d in bad)
        print(f"  {name:<13} M={NROW} vs chunks 32/64/128/192/256 -> {detail}")
        del W, h, ref
        torch.cuda.empty_cache()
    print("-" * 132)

    print(f"totals: rocBLAS {tot_r:.1f}  old {tot_o:.1f} ({tot_o/tot_r:.3f}x)  "
          f"new {tot_n:.1f} ({tot_n/tot_r:.3f}x)")
    print(f"cells beating rocBLAS: old {beat_o}/{ncell}   new {beat_n}/{ncell}")
    print(f"cells where the dispatch actually changed: {ndiff}/{ncell}")
    print(f"new vs old: {(tot_n-tot_o)/tot_o*100:+.1f}%")
    print(f"unexpected bit differences (non-split arms disagreeing): {nbug}")
    print(f"M-INVARIANCE: {'PASS' if minv_ok else 'FAIL'}")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("shape,IN,OUT,M,roc_us,old_us,new_us,old_dispatch,new_dispatch,maxabsdiff\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f},"
                        f"{r[7]},{r[8]},{r[9]:.3e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
