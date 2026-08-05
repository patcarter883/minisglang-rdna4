#!/usr/bin/env python3
"""parse_counters.py — aggregate rocprofv3 counter CSVs into a per-kernel table.

WHY A REAL CSV PARSER. rocprofv3's `Kernel_Name` is a QUOTED field CONTAINING COMMAS (C++ template
arguments). `awk -F,` with a fixed column index therefore reads a TIMESTAMP for some rows and a
counter value for others, and prints 0 for everything -- that exact mistake once fabricated a
false "all counters are broken on gfx1201" result. This module uses csv.DictReader, which respects
the quoting, so column identity is never positional.

Aggregation: rocprofv3 emits ONE ROW PER (dispatch, counter). A kernel launched `iters` times has
`iters` rows per counter. We SUM over dispatches (counters are extensive) and record the dispatch
count, so a per-dispatch value is always recoverable and a missing/extra dispatch is visible rather
than silently averaged away.
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict


def short_kernel(name: str) -> str:
    """Collapse a C++ template instantiation to something readable but still DISTINGUISHING.

    Deliberately keeps the loader policy and the integer template args -- those are exactly what
    separates the fp8 from the int4 GEMV, and a short name that merged them would silently pool two
    different kernels into one row.
    """
    n = name.strip()
    base = n.split("(")[0]
    m = re.match(r"([A-Za-z_][A-Za-z0-9_:]*)", base)
    head = m.group(1).split("::")[-1] if m else base
    loaders = re.findall(r"([A-Za-z0-9_]*(?:Loader|GemvLoader|WLoad)[A-Za-z0-9_]*)", n)
    ints = re.findall(r"\b(\d+)[ul]*\b(?=\s*[,>])", n)
    tag = head
    if loaders:
        tag += "<" + loaders[0] + ">"
    if ints:
        tag += "[" + ",".join(ints[:6]) + "]"
    return tag


def load_dir(d):
    """Return {kernel_short: {counter: sum}}, {kernel_short: meta}."""
    vals = defaultdict(lambda: defaultdict(float))
    ndisp = defaultdict(lambda: defaultdict(int))
    meta = {}
    files = glob.glob(os.path.join(d, "**", "*counter_collection.csv"), recursive=True)
    files += [f for f in glob.glob(os.path.join(d, "*.csv")) if "counter" in os.path.basename(f)]
    for f in sorted(set(files)):
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                kn = row.get("Kernel_Name")
                cn = row.get("Counter_Name")
                cv = row.get("Counter_Value")
                if not kn or not cn or cv in (None, ""):
                    continue
                k = short_kernel(kn)
                try:
                    v = float(cv)
                except ValueError:
                    continue
                vals[k][cn] += v
                ndisp[k][cn] += 1
                if k not in meta:
                    meta[k] = {
                        "full": kn.strip(),
                        "grid": row.get("Grid_Size"),
                        "wg": row.get("Workgroup_Size"),
                        "vgpr": row.get("VGPR_Count"),
                        "sgpr": row.get("SGPR_Count"),
                        "lds": row.get("LDS_Block_Size"),
                        "scratch": row.get("Scratch_Size"),
                    }
                # kernel duration, from the SAME rows (start/end are per dispatch)
                try:
                    ns = float(row["End_Timestamp"]) - float(row["Start_Timestamp"])
                    vals[k]["_ns_sum"] += ns
                    ndisp[k]["_ns_sum"] += 1
                except (KeyError, TypeError, ValueError):
                    pass
    out = {}
    for k in vals:
        rec = {}
        for c, v in vals[k].items():
            n = ndisp[k][c]
            rec[c] = {"sum": v, "n": n, "per_dispatch": v / n if n else 0.0}
        out[k] = rec
    return out, meta


def main():
    if len(sys.argv) < 2:
        print("usage: parse_counters.py <results_dir> [--json out.json] [--filter substr]")
        sys.exit(2)
    d = sys.argv[1]
    jsout = None
    filt = None
    if "--json" in sys.argv:
        jsout = sys.argv[sys.argv.index("--json") + 1]
    if "--filter" in sys.argv:
        filt = sys.argv[sys.argv.index("--filter") + 1]

    data, meta = load_dir(d)
    if filt:
        data = {k: v for k, v in data.items() if filt in k}
        meta = {k: v for k, v in meta.items() if k in data}
    if not data:
        print(f"(no counter rows found under {d})")
        sys.exit(1)

    kernels = sorted(data)
    counters = sorted({c for k in kernels for c in data[k] if not c.startswith("_")})

    for k in kernels:
        m = meta.get(k, {})
        print(f"\n### {k}")
        print(f"    grid={m.get('grid')} wg={m.get('wg')} VGPR={m.get('vgpr')} SGPR={m.get('sgpr')} "
              f"LDS={m.get('lds')} scratch={m.get('scratch')}")
        nsr = data[k].get("_ns_sum")
        if nsr and nsr["n"]:
            print(f"    dispatch_ns(mean over {nsr['n']} counter-rows) = {nsr['per_dispatch']:,.0f}")
        for c in counters:
            r = data[k].get(c)
            if not r:
                continue
            print(f"    {c:<28} sum={r['sum']:>18,.0f}  n={r['n']:>4}  per_dispatch={r['per_dispatch']:>16,.1f}")

    if jsout:
        with open(jsout, "w") as fh:
            json.dump({"data": data, "meta": meta}, fh, indent=1)
        print(f"\n[json] {jsout}")


if __name__ == "__main__":
    main()
