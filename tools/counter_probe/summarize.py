#!/usr/bin/env python3
"""summarize.py — turn rocprofv3 --pmc CSVs into the int4-vs-fp8 utilisation breakdown.

Reads the pN.csv files written by gemv_counters.sh and reports, per kernel variant, the mean of each
counter over the COUNTED dispatches plus the ratios that decide the regime call.

Two things it deliberately does, because getting either wrong silently fakes the answer:
  * Drops `__amd_rocclr_fillBufferAligned`. Those are hipMemset dispatches from buffer setup; they are
    real dispatches and rocprofv3 counts them, so leaving them in dilutes every per-kernel mean.
  * Drops the FIRST dispatch of each variant (the warmup). It carries lazy code-object/first-touch
    cost that is not part of steady-state decode.

Kernel variants are told apart by the WLoad template argument in the mangled-ish Kernel_Name, which is
the only thing that differs between the two launches — same core, same shape, by construction.
"""
import csv, glob, os, sys, collections

d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)) + "/results"


def variant(name: str) -> str | None:
    if "gemv_decode_core" not in name:
        return None
    if "Int4Fp8GemvLoader" in name:
        return "int4 (W4A8, Int4Fp8GemvLoader)"
    if "Fp8DenseGemvLoader" in name:
        return "fp8  (W8A8, Fp8DenseGemvLoader)"
    return "other"


# vals[variant][counter] = [per-dispatch value]; meta[variant] = static launch info
vals = collections.defaultdict(lambda: collections.defaultdict(list))
meta, dur = {}, collections.defaultdict(list)
for f in sorted(glob.glob(f"{d}/p*.csv")):
    seen = collections.defaultdict(set)
    for row in csv.DictReader(open(f)):
        v = variant(row["Kernel_Name"])
        if v is None or v == "other":
            continue
        did = row["Dispatch_Id"]
        # first dispatch of this variant IN THIS PASS is the warmup -> skip
        if did not in seen[v] and len(seen[v]) == 0:
            seen[v].add(did)
            continue
        seen[v].add(did)
        vals[v][row["Counter_Name"]].append(float(row["Counter_Value"]))
        meta[v] = dict(vgpr=row["VGPR_Count"], sgpr=row["SGPR_Count"], lds=row["LDS_Block_Size"],
                       scratch=row["Scratch_Size"], grid=row["Grid_Size"], wg=row["Workgroup_Size"])
        dur[v].append(int(row["End_Timestamp"]) - int(row["Start_Timestamp"]))

if not vals:
    sys.exit(f"no gemv_decode_core rows found in {d}/p*.csv")

mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
order = sorted(vals)
w = 26

print(f"{'':{w}}" + "".join(f"{v.split(' ')[0]:>18}" for v in order))
for v in order:
    m = meta[v]
    print(f"  [{v}] grid={m['grid']} wg={m['wg']} VGPR={m['vgpr']} SGPR={m['sgpr']} "
          f"LDS={m['lds']} scratch={m['scratch']}")
print()

counters = sorted({c for v in order for c in vals[v]})
for c in counters:
    print(f"{c:{w}}" + "".join(f"{mean(vals[v][c]):>18,.0f}" for v in order))
print(f"{'kernel_ns (mean)':{w}}" + "".join(f"{mean(dur[v]):>18,.0f}" for v in order))
print()

# ---- the ratios the regime call actually turns on ----
def g(v, c):
    return mean(vals[v].get(c, [float("nan")]))

print("--- derived ---")
rows = [
    ("VALU insts / wave",      lambda v: g(v, "SQ_INSTS_VALU") / g(v, "SQ_WAVES")),
    ("SALU insts / wave",      lambda v: g(v, "SQ_INSTS_SALU") / g(v, "SQ_WAVES")),
    ("SMEM insts / wave",      lambda v: g(v, "SQ_INSTS_SMEM") / g(v, "SQ_WAVES")),
    ("LDS insts / wave",       lambda v: g(v, "SQ_INSTS_LDS") / g(v, "SQ_WAVES")),
    ("VALU cyc / GUI_ACTIVE",  lambda v: g(v, "SQ_INST_CYCLES_VALU") / g(v, "GRBM_GUI_ACTIVE")),
    ("VMEM cyc / GUI_ACTIVE",  lambda v: g(v, "SQ_INST_CYCLES_VMEM") / g(v, "GRBM_GUI_ACTIVE")),
    ("wait_any / wave_cycles", lambda v: g(v, "SQ_WAIT_ANY") / g(v, "SQ_WAVE_CYCLES")),
    ("VALUBusy %",             lambda v: g(v, "VALUBusy")),
    ("MemUnitBusy %",          lambda v: g(v, "MemUnitBusy")),
    ("ValuPipeIssueUtil %",    lambda v: g(v, "ValuPipeIssueUtil")),
    ("MeanOccupancyPerCU",     lambda v: g(v, "MeanOccupancyPerCU")),
]
for label, fn in rows:
    out = []
    for v in order:
        try:
            out.append(f"{fn(v):>18,.2f}")
        except Exception:
            out.append(f"{'n/a':>18}")
    print(f"{label:{w}}" + "".join(out))
