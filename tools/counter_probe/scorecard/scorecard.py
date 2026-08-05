#!/usr/bin/env python3
"""scorecard.py — combine the counter phases into the per-claim table.

THE TWO NORMALISATIONS THAT DECIDE EVERY VERDICT HERE, both of which have already been got wrong:

1. BYTES come from counters (profile_standard), TIME comes from the kernel trace (auto). Mixing them
   is the perf-level trap: the int4 16384^2 GEMV measures 949,906 ns pinned versus 222,643 ns at
   auto, so a bandwidth computed from the pinned timestamp understates the kernel by 4.3x and looks
   exactly like a catastrophic regression.

2. GL2C_MISS is calibrated to BYTES rather than assumed. Two independent streaming shapes, whose
   working sets are far larger than the 64 MB last-level cache and therefore MUST be read from HBM
   exactly once, both give 256 B per miss:
        fp8  16384^2 : 268.5 MB / 1,049,047 misses = 256.0 B
        int4 16384^2 : 138.4 MB /   543,041 misses = 254.9 B
   So HBM bytes = GL2C_MISS * 256. The derived `FetchSize` metric, which would have answered this
   directly, returns a hard ZERO on gfx1201 even with the perfmon clock ungated -- it is NOT usable,
   despite being listed as working.

ROOFLINE DENOMINATOR = 706.6 GB/s. Not 674, not 644, and never an achieved bench figure: quoting a
previously-measured throughput as the ceiling makes every later kernel look closer to "done" than it
is, which is precisely how a percentage-of-roofline claim closes off work it should not.
"""
import csv, glob, os, re, sys
from collections import defaultdict

HBM_PEAK_GBS = 706.6
BYTES_PER_L2_MISS = 256.0

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "scorecard")


def short(n):
    n = n.strip()
    if "fillBuffer" in n or "copyBuffer" in n:
        return None
    m = re.search(r"(Fp8DenseGemvLoader|Int4Fp8GemvLoader|Bf16GemvLoader)", n)
    ints = re.findall(r"\b(\d+)\b(?=\s*[,>])", n)
    return (m.group(1) if m else n.split("(")[0][-28:]) + "[" + ",".join(ints[:4]) + "]"


def counters(tag):
    """Per-dispatch counter values, keyed BY KERNEL.

    Keying by kernel is load-bearing, not tidiness: the `dense` sweeps launch the fp8 AND the int4
    GEMV in one process. Pooling them summed fp8's bytes onto int4's time and produced 118-129% of
    roofline -- a physically impossible number that is the giveaway for exactly this mistake."""
    d = os.path.join(ROOT, "phase2", tag)
    vals, n = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))
    for f in glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            k = short(r.get("Kernel_Name", ""))
            if not k:
                continue
            try:
                vals[k][r["Counter_Name"]] += float(r["Counter_Value"])
                n[k][r["Counter_Name"]] += 1
            except (KeyError, ValueError):
                pass
    return {k: {c: vals[k][c] / n[k][c] for c in vals[k] if n[k][c]} for k in vals}


def timing(tag):
    """MIN kernel duration at auto clocks. Min, not mean: the first dispatch in a trace carries
    lazy code-object load and one-off cache warming, and with only 4 dispatches it drags the mean
    badly (dense_4096 int4: mean 51,831 ns vs min 15,240 ns). Min is the steady-state estimate."""
    d = os.path.join(ROOT, "phase0", tag)
    per = defaultdict(list)
    for f in glob.glob(os.path.join(d, "**", "*kernel_trace.csv"), recursive=True):
        for r in csv.DictReader(open(f)):
            k = short(r.get("Kernel_Name", ""))
            if not k:
                continue
            try:
                per[k].append(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]))
            except (KeyError, ValueError):
                pass
    return {k: (min(v), sum(v) / len(v), len(v)) for k, v in per.items()}


def rows(tag, label, band):
    cs = counters(tag)
    t = timing(tag)
    out = []
    for k, c in sorted(cs.items()):
        miss = c.get("GL2C_MISS", 0.0)
        hit = c.get("GL2C_HIT", 0.0)
        hbm_bytes = miss * BYTES_PER_L2_MISS
        ns = t.get(k, (0, 0, 0))[0]
        gbs = (hbm_bytes / ns) if ns else 0.0        # bytes/ns == GB/s
        out.append(dict(
            tag=tag, label=label, band=band, kern=k,
            waves=c.get("SQ_WAVES", 0), occ=c.get("OccupancyPercent", float("nan")),
            mem=c.get("MemUnitBusy", float("nan")), valu=c.get("VALUBusy", float("nan")),
            dep=c.get("WAVE_DEP_WAIT", float("nan")), iss=c.get("WAVE_ISSUE_WAIT", float("nan")),
            l2hit=100.0 * hit / (hit + miss) if (hit + miss) else float("nan"),
            mb=hbm_bytes / 1e6, ns=ns, gbs=gbs, pct=100.0 * gbs / HBM_PEAK_GBS,
            valu_per_wave=(c.get("SQ_INSTS_VALU", 0) / c["SQ_WAVES"]) if c.get("SQ_WAVES") else 0,
        ))
    return out


SPEC = [
    ("dense_4096",  "int4/fp8 dense GEMV 4096^2",  "decode M=1, cache-adjacent"),
    ("dense_16384", "int4/fp8 dense GEMV 16384^2", "decode M=1, HBM-STREAMING"),
    ("bf16_gate",   "bf16 shared.gate N=1 K=2048", "decode M=1, launch-floor"),
    ("bf16_down",   "bf16 shared.down N=2048 K=256", "decode M=1, small-N"),
    ("bf16_qkvz",   "bf16 in_proj_qkvz N=6144 K=2048", "decode M=1, large-N"),
    ("bf16_lmhead", "bf16 LM head N=32768 K=2048", "decode M=1, large-N"),
    ("moe1_M1",     "MoE gemm1 M=1",  "decode"), ("moe2_M1", "MoE gemm2(unfused) M=1", "decode"),
    ("moe1_M5",     "MoE gemm1 M=5",  "decode"), ("moe2_M5", "MoE gemm2(unfused) M=5", "decode"),
    ("moe1_M6",     "MoE gemm1 M=6",  "decode"), ("moe2_M6", "MoE gemm2(unfused) M=6", "decode"),
    ("moe1_M30",    "MoE gemm1 M=30", "decode"), ("moe2_M30", "MoE gemm2(unfused) M=30", "decode"),
]

if __name__ == "__main__":
    print(f"roofline denominator = {HBM_PEAK_GBS} GB/s ; HBM bytes = GL2C_MISS x {BYTES_PER_L2_MISS:.0f} B")
    print("counters @ profile_standard (pinned) ; times @ auto (min of 4 dispatches)\n")
    h = ("shape", "kernel", "band", "waves", "Occ%", "Mem%", "ISS%", "L2hit%", "HBM_MB", "min_ns", "GB/s", "%706.6", "VALU/wv")
    print(f"{h[0]:<30}{h[1]:<26}{h[2]:<24}{h[3]:>8}{h[4]:>6}{h[5]:>6}{h[6]:>6}{h[7]:>7}{h[8]:>9}{h[9]:>10}{h[10]:>8}{h[11]:>8}{h[12]:>9}")
    print("-" * 158)
    for tag, label, band in SPEC:
        rs = rows(tag, label, band)
        if not rs:
            print(f"{label:<30}(no data)")
            continue
        for r in rs:
            print(f"{label:<30}{r['kern'][:25]:<26}{band:<24}{r['waves']:>8,.0f}{r['occ']:>6.1f}"
                  f"{r['mem']:>6.1f}{r['iss']:>6.1f}{r['l2hit']:>7.1f}{r['mb']:>9.2f}{r['ns']:>10,.0f}"
                  f"{r['gbs']:>8.1f}{r['pct']:>8.1f}{r['valu_per_wave']:>9,.0f}")
