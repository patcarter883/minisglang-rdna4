"""Producer-side activation quant: correctness, then the dense-arm re-comparison.

WHAT THIS ANSWERS. `_pick_dense_kernel` records ONE surviving regime for `prefill_wmma`: it FUSES the
activation fp8-quant into its GEMM prologue, while `wmma_tiled_tuned` consumes pre-quantized
activations and therefore eats a separate `compute_act_fp8_and_scales_kernel` launch plus an (M,K)
uint8 HBM round-trip. On clean card-0 data at each arm's own best tile that is worth 1.11-1.23x on
five mid-band cells, all gate_up shapes at M=17..32 -- and it is the entire remaining case for
keeping the arm.

The fix the docstring prescribes is producer-side fusion: quantize in the RMSNorm / residual-add that
already holds the values in registers, NOT a second GEMM and NOT a bare standalone elementwise op.
That is now built (tail_hip `rms_norm_quant` / `rms_norm_add_quant` + an `x_fp8`/`act_scales` policy
on the SAME mmq_fp8_gemm core), so the arm comparison can be re-run honestly:

    ARM prefill    : rms_norm_add(x, res, w)        -> mmq_fp8_gemm(kernel=prefill_wmma)
    ARM tiled      : rms_norm_add(x, res, w)        -> mmq_fp8_gemm(kernel=wmma_tiled_tuned)   [shipped]
    ARM tiled+fused: rms_norm_add_quant(x, res, w)  -> mmq_fp8_gemm(wmma_tiled_tuned, x_fp8=, act_scales=)

All three are timed END TO END -- producer + linear -- because the fusion MOVES work into the
producer rather than deleting it, and an arm comparison that timed only the GEMM would credit the
fused arm with work it still does.

ORDERING IS DELIBERATE: correctness runs BEFORE any timing. Running it the other way round is what
hid a host SIGFPE in the MoE launcher for so long.

METHOD (inherited from tools/w4a8_dense_tile_surface.py, unchanged):
  * graph-replay device timing, never wall time;
  * weights rotated past the 64 MB MALL BY BYTE COUNT;
  * card asserted by DEVICE PROPERTIES (32 WGP = 64 CU, RX 9070 XT), never by ordinal -- note torch
    reports WGPs on RDNA, so the 64-CU card answers multi_processor_count == 32 and the 56-CU
    RX 9070 answers 28;
  * provenance columns on every CSV row, via csv.writer;
  * the probe must REPRODUCE the surface fixture (tools/_fixtures/dense_tile_arms.csv) on the shipped
    arms before any new number from it is trusted -- a probe that silently differs from production
    (the V7_SWIZ=0-vs-1 near miss) produces confident nonsense.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4a8_dense_tile_surface import DEV, pack_uint4_2d, rotation, time_graph  # noqa: E402

# The five cells prefill_wmma still wins, from tools/_fixtures/dense_tile_arms.csv at each arm's OWN
# BEST TILE. (name, K, N, group, M, fixture_prefill_us, fixture_tiled_us).
CELLS = [
    ("q27.gate_up tp1", 5120, 34816, 32, 17, 797.0, 982.3),
    ("q27.gate_up tp1", 5120, 34816, 32, 24, 822.5, 975.6),
    ("q27.gate_up tp1", 5120, 34816, 32, 32, 869.7, 969.7),
    ("glm.gate_up tp2", 2048, 10240, 128, 17, 101.0, 122.5),
    ("glm.gate_up tp2", 2048, 10240, 128, 24, 102.5, 116.9),
]
# The tiled arm's own best tile at each cell, READ OFF the fixture (VLLM_W4A8_V7_CFG; prefill_wmma
# has no tile knob -- its small-M branch IS its tile). Hardcoding one tile for all five cells is what
# broke the first run of this probe: 256x128 is the best only at q27 M=17, and forcing it on the two
# glm cells inflated the tiled arm by ~28% and made the probe fail its own reproduction gate.
TILED_BEST = {(5120, 34816, 17): "256x128", (5120, 34816, 24): "128x128",
              (5120, 34816, 32): "128x128", (2048, 10240, 17): "128x128",
              (2048, 10240, 24): "128x128"}

# Bit-identity through the AUTO-DISPATCH (w4a8_linear), not the raw op.
IDENTITY_MS = [1, 17, 33, 64, 129, 512]


def assert_card() -> str:
    p = torch.cuda.get_device_properties(0)
    # torch reports WGPs on RDNA: 32 WGP == the 64 CU the tile cost model prints.
    assert p.multi_processor_count == 32 and "9070 XT" in p.name, (
        f"expected the 64-CU (32 WGP) RX 9070 XT; got {p.name} / "
        f"{p.multi_processor_count} WGP. Card selected by PROPERTIES, never by ordinal."
    )
    return f"{p.name} ({p.multi_processor_count} WGP = 64 CU) {p.gcnArchName}"


def set_tile(cfg: str | None) -> None:
    os.environ.pop("VLLM_W4A8_V7_CFG", None)
    os.environ.pop("VLLM_W4A8_V10_CFG", None)
    os.environ.pop("VLLM_W4A8_DENSE_SMALLM_OFF", None)
    if cfg:
        os.environ["VLLM_W4A8_V7_CFG"] = cfg


# --------------------------------------------------------------------------------------------------
# PART 1 - CORRECTNESS. Runs first, and a failure here aborts before a single timing number exists.
# --------------------------------------------------------------------------------------------------
def correctness(out) -> int:
    import fp8_wmma as W
    import tail_hip as T
    from minisgl.distributed.info import set_tp_info

    try:
        set_tp_info(0, 1)  # `engaged()` logs rank0-only and resolves TP on its first call per op
    except RuntimeError:
        pass
    from minisgl.quant import kernels as K

    fails = []

    def chk(name, cond, detail=""):
        out(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
        if not cond:
            fails.append(name)

    # ---- PROVENANCE: this must be the fused build, not the baked /opt/kernels one -----------------
    out(f"tail_hip   : {T.__file__}")
    out(f"fp8_wmma   : {W.__file__}")
    chk("tail_hip has the fused producer ops",
        hasattr(T, "rms_norm_quant") and hasattr(T, "rms_norm_add_quant"))
    import inspect
    sig = inspect.signature(W.mmq_fp8_gemm)
    chk("mmq_fp8_gemm takes pre-quantized activations",
        "x_fp8" in sig.parameters and "act_scales" in sig.parameters, str(sig))
    if fails:
        return len(fails)

    torch.manual_seed(0)
    for dt in (torch.bfloat16, torch.float16):
        for (D, g) in ((5120, 32), (2048, 128), (2816, 32)):
            for M in IDENTITY_MS:
                x = (torch.randn(M, D, device=DEV, dtype=dt) * 0.7)
                w = (torch.randn(D, device=DEV, dtype=dt) * 0.1 + 1.0)
                res = (torch.randn(M, D, device=DEV, dtype=dt) * 0.7)
                tag = f"{dt}".split(".")[-1] + f" D={D} M={M}"

                # (a) the fused producer's bf16/fp16 output is BIT-IDENTICAL to the plain producer.
                ref = T.rms_norm(x.contiguous(), w, 1e-6, 0)
                o, xq, sc = T.rms_norm_quant(x.contiguous(), w, 1e-6, 0)
                chk(f"rms_norm_quant out bit-identical to rms_norm  [{tag}]",
                    torch.equal(ref, o))
                r1, r2 = res.clone(), res.clone()
                refa = T.rms_norm_add(x.contiguous(), r1, w, 1e-6, 0)
                oa, xqa, sca = T.rms_norm_add_quant(x.contiguous(), r2, w, 1e-6, 0)
                chk(f"rms_norm_add_quant out bit-identical  [{tag}]", torch.equal(refa, oa))
                chk(f"rms_norm_add_quant residual bit-identical  [{tag}]", torch.equal(r1, r2))

                # (b) THE REAL GATE: the fp8 pair the producer emitted must drive the SAME linear to
                #     a BIT-IDENTICAL answer as the pre-kernel the op would have launched itself. This
                #     goes through w4a8_linear's AUTO-DISPATCH, not a hand-picked raw op, so it also
                #     proves the fused pair is consumable on whichever arm the chooser lands on.
                N = 2048
                wq = pack_uint4_2d(torch.randint(0, 16, (N, D), dtype=torch.int8, device=DEV))
                ws = (torch.randn(N, D // g, device=DEV).abs() * 0.02 + 0.002).to(torch.float16)
                base = K.w4a8_linear(o, wq, ws, None, g)
                fused = K.w4a8_linear(o, wq, ws, None, g, x_fp8=xq, act_scales=sc)
                chk(f"w4a8_linear auto-dispatch BIT-IDENTICAL on the fused pair  [{tag} N={N}]",
                    torch.equal(base, fused),
                    "" if torch.equal(base, fused)
                    else f"max|d|={(base.float()-fused.float()).abs().max().item():.6g}")

    # ---- NO SILENT FALLBACKS: every refusal must be loud -------------------------------------------
    D, g, N, M = 2048, 128, 2048, 33
    x = torch.randn(M, D, device=DEV, dtype=torch.bfloat16)
    wv = torch.randn(D, device=DEV, dtype=torch.bfloat16)
    o, xq, sc = T.rms_norm_quant(x.contiguous(), wv, 1e-6, 0)
    wq = pack_uint4_2d(torch.randint(0, 16, (N, D), dtype=torch.int8, device=DEV))
    ws = (torch.randn(N, D // g, device=DEV).abs() * 0.02 + 0.002).to(torch.float16)
    z = torch.empty(0, dtype=torch.int32, device=DEV)
    for name, kw in (("x_fp8 without act_scales", dict(x_fp8=xq)),
                     ("act_scales without x_fp8", dict(act_scales=sc))):
        try:
            W.mmq_fp8_gemm(o, wq, ws, kernel="wmma_tiled_tuned", w_zeros=z, **kw)
            chk(f"half-pair refused loudly ({name})", False, "it RAN - a silent re-quantize")
        except RuntimeError as e:
            chk(f"half-pair refused loudly ({name})", "TOGETHER" in str(e), str(e)[:110])
    try:
        W.mmq_fp8_gemm(o, wq, ws, kernel="prefill_wmma", w_zeros=z, x_fp8=xq, act_scales=sc)
        chk("prefill_wmma refuses a pre-quantized pair loudly", False, "it RAN - silently ignored")
    except RuntimeError as e:
        chk("prefill_wmma refuses a pre-quantized pair loudly", "in-prologue" in str(e), str(e)[:110])
    # wrong shape / wrong dtype must not be coerced
    try:
        W.mmq_fp8_gemm(o, wq, ws, kernel="wmma_tiled_tuned", w_zeros=z,
                       x_fp8=xq[:, : D // 2].contiguous(), act_scales=sc)
        chk("mis-shaped x_fp8 refused loudly", False, "it RAN")
    except RuntimeError as e:
        chk("mis-shaped x_fp8 refused loudly", "x_fp8 must be" in str(e), str(e)[:110])

    out(f"\n=== CORRECTNESS: {'ALL PASS' if not fails else str(len(fails)) + ' FAIL: ' + str(fails)}")
    return len(fails)


# --------------------------------------------------------------------------------------------------
# PART 2 - the dense-arm re-comparison, END TO END (producer + linear).
# --------------------------------------------------------------------------------------------------
def timing(out, writer, dev_str: str, reps: int) -> None:
    import fp8_wmma as W
    import tail_hip as T

    torch.manual_seed(0)
    z = torch.empty(0, dtype=torch.int32, device=DEV)
    dt = torch.bfloat16
    rows = []

    for (name, Kd, N, g, M, fx_pre, fx_tiled) in CELLS:
        ws, R, wbytes = rotation(N, Kd, g, False, M)
        x = (torch.randn(M, Kd, device=DEV, dtype=dt) * 0.7)
        wn = (torch.randn(Kd, device=DEV, dtype=dt) * 0.1 + 1.0)
        res = (torch.randn(M, Kd, device=DEV, dtype=dt) * 0.7)

        def arm_prefill(wt):
            o = T.rms_norm_add(x, res, wn, 1e-6, 0)
            return W.mmq_fp8_gemm(o, wt[0], wt[1], kernel="prefill_wmma", w_zeros=z)

        def arm_tiled(wt):
            o = T.rms_norm_add(x, res, wn, 1e-6, 0)
            return W.mmq_fp8_gemm(o, wt[0], wt[1], kernel="wmma_tiled_tuned", w_zeros=z)

        def arm_tiled_fused(wt):
            o, xq, sc = T.rms_norm_add_quant(x, res, wn, 1e-6, 0)
            return W.mmq_fp8_gemm(o, wt[0], wt[1], kernel="wmma_tiled_tuned", w_zeros=z,
                                  x_fp8=xq, act_scales=sc)

        cfg = TILED_BEST[(Kd, N, M)]
        res_us = {}
        # `tile=None` on a tiled arm means NO VLLM_W4A8_V7_CFG -- the kernel derives its own tile from
        # the shape (tile_select.h). That is what PRODUCTION dispatches, and it is a different (often
        # better) tile from the fixture's best-of-the-common-set: `_pick_dense_kernel`'s surviving
        # ~1.06x GLM regime is stated against the CHOOSER (100.8 vs 106.9 us), not against 122.5. The
        # retirement question has to be answered on the chooser arm or it is answered on a strawman.
        for arm, fn, tile in (("prefill_wmma", arm_prefill, None),
                              ("tiled@best", arm_tiled, cfg),
                              ("tiled(chooser)", arm_tiled, None),
                              ("tiled(chooser)+producer_fused", arm_tiled_fused, None)):
            set_tile(tile)
            samples = []
            for _ in range(reps):
                us, _ = time_graph(fn, ws)
                samples.append(us)
            med = statistics.median(samples)
            res_us[arm] = med
            writer.writerow([name, Kd, N, g, M, str(dt).split(".")[-1], arm, tile or "-",
                             f"{med:.3f}", f"{min(samples):.3f}", f"{max(samples):.3f}",
                             reps, R, wbytes, dev_str])
        set_tile(None)
        rows.append((name, Kd, N, g, M, fx_pre, fx_tiled, res_us))
        out(f"{name:18s} K={Kd:5d} N={N:6d} g={g:3d} M={M:3d} | "
            f"prefill {res_us['prefill_wmma']:9.2f}  tiled@best {res_us['tiled@best']:9.2f}  "
            f"tiled(chooser) {res_us['tiled(chooser)']:9.2f}  "
            f"chooser+fused {res_us['tiled(chooser)+producer_fused']:9.2f} us")

    # ---- FIXTURE REPRODUCTION GATE ---------------------------------------------------------------
    # The end-to-end arms carry the producer, which the fixture's GEMM-only numbers do not, so the
    # gate is on the RATIO the fixture actually asserts (tiled/prefill), not on absolute us.
    out("\n--- fixture reproduction (tools/_fixtures/dense_tile_arms.csv, ratio tiled/prefill) ---")
    ok = True
    for (name, Kd, N, g, M, fx_pre, fx_tiled, r) in rows:
        fx_ratio = fx_tiled / fx_pre
        me_ratio = r["tiled@best"] / r["prefill_wmma"]
        agree = abs(me_ratio - fx_ratio) / fx_ratio < 0.15
        ok &= agree
        out(f"  {'OK ' if agree else 'DIFF'} {name} M={M}: fixture {fx_ratio:.3f}x  probe {me_ratio:.3f}x")
    out(f"  => probe {'REPRODUCES' if ok else 'DOES NOT REPRODUCE'} the surface fixture")

    out("\n--- the question: can prefill_wmma retire? (end-to-end, producer + linear) ---")
    wins = 0
    for (name, Kd, N, g, M, fx_pre, fx_tiled, r) in rows:
        ratio = r["tiled(chooser)+producer_fused"] / r["prefill_wmma"]
        gain = r["tiled(chooser)"] / r["tiled(chooser)+producer_fused"]
        verdict = "prefill still ahead" if ratio > 1.02 else "prefill RETIRABLE at this cell"
        wins += ratio <= 1.02
        out(f"  {name} M={M:3d}: (chooser+fused)/prefill {ratio:.3f}x  (the fusion bought "
            f"{gain:.3f}x over the shipped chooser arm)  -> {verdict}")
    out(f"  => {wins}/{len(rows)} cells no longer justify prefill_wmma")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-timing", action="store_true")
    a = ap.parse_args()
    fh = open(a.out, "w") if a.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    dev_str = assert_card()
    out(f"device: {dev_str}")
    out(f"lease : ROCR={os.environ.get('ROCR_VISIBLE_DEVICES','?')} "
        f"HIP={os.environ.get('HIP_VISIBLE_DEVICES','?')}")

    nf = correctness(out)
    if nf:
        out("\nABORTING BEFORE TIMING: correctness failed. (Timing a broken kernel is how the MoE "
            "SIGFPE stayed hidden.)")
        return 1
    if a.skip_timing:
        return 0

    cf = open(a.csv, "w", newline="") if a.csv else None
    writer = csv.writer(cf) if cf else csv.writer(open(os.devnull, "w", newline=""))
    writer.writerow(["name", "K", "N", "g", "M", "dtype", "arm", "tile", "us_median",
                     "us_min", "us_max", "reps", "R_rotation", "weight_bytes", "device"])
    timing(out, writer, dev_str, a.reps)
    if cf:
        cf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
