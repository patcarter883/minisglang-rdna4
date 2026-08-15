#!/usr/bin/env python
"""Static per-kernel GPU resource table for gfx1201, read out of a built .so. CPU-only, no GPU.

Every number here is what the COMPILER decided, so it is the right instrument for a toolchain A/B:
the same source built by two clang versions differs only in these, and a register or scratch change
explains a throughput change that a timing run can only report.

    python tools/kernel_static_resources.py A.so [B.so] [--image ...] [--filter substr]

With two .so files it diffs them and prints only the kernels whose resources moved.

Method (recorded in tools/_fixtures/kernel_static_resources.csv, tool never saved until now):
  llvm-objdump --offloading   -> extract the hipv4-amdgcn-amd-amdhsa--gfx1201 code object bundle
  llvm-readelf --notes        -> NT_AMDGPU_METADATA (msgpack) -> amdhsa.kernels[]
Fields: .vgpr_count, .sgpr_count, .private_segment_fixed_size (scratch/lane),
        .group_segment_fixed_size (STATIC LDS only -- a kernel taking DYNAMIC shared memory reports
        0 here; do not read that as "uses no LDS"), .vgpr_spill_count.

occupancy_by_vgpr = min(16, 1536 // (ceil(vgpr/24)*24))   [gfx1201: register file 1536, granule 24,
cap 16 waves/SIMD]. VGPR-only; the real occupancy is min(this, LDS bound, workgroup bound).
gfx1201 (RDNA4) has no AGPR file, so no .agpr_count is emitted.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

IMAGE = "rocm/dev-ubuntu-24.04:7.14.0-full"
LLVM = "/opt/rocm/lib/llvm/bin"

# Runs inside the container: unbundle each .so's device code and dump the metadata note.
# `llvm-objdump --offloading` writes the extracted code objects NEXT TO ITS INPUT, not into the CWD,
# so the .so is COPIED into the writable scratch first. /work stays read-only deliberately: a tool
# that measures a build artifact must not be able to modify it.
INNER = r'''
set -e
cd /out
for so in "$@"; do
  base=$(basename "$so")
  cp "$so" "/out/$base"
  ''' + LLVM + r'''/llvm-objdump --offloading "/out/$base" >/dev/null 2>&1 || true
  # ONE CODE OBJECT PER TRANSLATION UNIT. A package built from six .hip files unbundles to six
  # gfx1201 objects; taking the first finds whichever TU happened to sort first and silently misses
  # every kernel in the other five.
  n=0; : > "/out/${base}.notes.txt"
  for obj in /out/"$base"*gfx1201*; do
    [ -e "$obj" ] || continue
    ''' + LLVM + r'''/llvm-readelf --notes "$obj" >> "/out/${base}.notes.txt" 2>/dev/null && n=$((n+1))
  done
  rm -f "/out/$base" /out/"$base".*hipv4* /out/"$base".*host-*
  [ "$n" -gt 0 ] || { echo "NOCODEOBJ $base" >&2; continue; }
  echo "OK $base ($n code objects)"
done
'''


def family(sym: str) -> str:
    """Collapse a mangled template instantiation to its kernel name.

    `_ZN13w4a8_fp8_wmma36mmq_fp8_gemm_wmma_tiled_tuned_kernelI14__hip_bfloat16...` -> the
    namespace::kernel, so 600 instantiations of one kernel report as one row. Itanium mangling
    encodes each identifier as <len><chars>, which is enough to walk without a demangler.
    """
    m = re.match(r"^_ZN?(L?)(.*)$", sym)
    if not m:
        return sym
    rest, parts = m.group(2), []
    while rest:
        d = re.match(r"^L?(\d+)", rest)
        if not d:
            break
        n = int(d.group(1))
        start = d.end()
        parts.append(rest[start:start + n])
        rest = rest[start + n:]
    return "::".join(parts) if parts else sym


def occ_by_vgpr(v: int) -> int:
    if v <= 0:
        return 16
    return min(16, 1536 // (math.ceil(v / 24) * 24))


def extract(so: Path, outdir: Path, image: str) -> dict[str, dict]:
    """kernel symbol -> resource dict, via the container's llvm tools."""
    work = so.parent
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{work}:/work:ro", "-v", f"{outdir}:/out",
        "--entrypoint", "bash", image, "-c", INNER, "_", f"/work/{so.name}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if "OK" not in r.stdout:
        print(f"  ! could not unbundle {so.name}: {r.stderr.strip()[:300]}", file=sys.stderr)
        return {}
    return parse_notes(outdir, so.name)


def parse_notes(outdir: Path, name: str) -> dict[str, dict]:
    t = outdir / f"{name}.notes.txt"
    return parse_notes_text(t.read_text()) if t.exists() else {}


FIELDS = {
    ".vgpr_count": "vgpr", ".sgpr_count": "sgpr",
    ".private_segment_fixed_size": "scratch", ".group_segment_fixed_size": "lds",
    ".vgpr_spill_count": "spill", ".sgpr_spill_count": "sgpr_spill",
    ".max_flat_workgroup_size": "max_wg", ".name": "name",
}


def parse_notes_text(txt: str) -> dict[str, dict]:
    """The metadata note prints as YAML-ish msgpack; pull kernel records field by field.

    Records are delimited by REPETITION, not by `.name`: llvm-readelf renders the msgpack map with
    its keys sorted, so `.group_segment_fixed_size` and `.max_flat_workgroup_size` come BEFORE
    `.name` and the rest after. Flushing on `.name` therefore files a kernel's LDS under its
    PREDECESSOR — which reads as "this kernel uses no LDS", quietly and plausibly. So: start a new
    record whenever a key repeats.

    Deliberately a scraper and not a msgpack parser: the note is already rendered by llvm-readelf,
    and adding a msgpack dependency to a tool whose point is 'runs anywhere' is a bad trade.
    """
    out: dict[str, dict] = {}
    cur: dict = {}

    def flush():
        if cur.get("name") is not None:
            out[cur.pop("name")] = dict(cur)

    for line in txt.splitlines():
        key = line.strip().split(":", 1)[0].strip()
        if key not in FIELDS:
            continue
        field = FIELDS[key]
        val = line.split(":", 1)[1].strip().strip(",").strip('"')
        if field in cur:                      # a repeat means the previous record ended
            flush()
            cur = {}
        cur[field] = val if field == "name" else _int(val)
    flush()
    return out


def _int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("so", nargs="+")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--filter", default="", help="only kernels whose symbol contains this")
    ap.add_argument("--label", nargs="*", default=[])
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="ksr-"))
    tables = []
    for i, p in enumerate(args.so):
        lbl = args.label[i] if i < len(args.label) else Path(p).parent.name
        print(f"reading {p}  [{lbl}]", file=sys.stderr)
        tables.append((lbl, extract(Path(p).resolve(), tmp, args.image)))

    if len(tables) == 1:
        lbl, T = tables[0]
        print(f"{'kernel':<70}{'vgpr':>6}{'sgpr':>6}{'spill':>6}{'scratch':>8}{'lds':>8}{'occ':>5}")
        for k, v in sorted(T.items()):
            if args.filter and args.filter not in k:
                continue
            print(f"{k[:70]:<70}{v.get('vgpr',0):6d}{v.get('sgpr',0):6d}{v.get('spill',0):6d}"
                  f"{v.get('scratch',0):8d}{v.get('lds',0):8d}{occ_by_vgpr(v.get('vgpr',0)):5d}")
        return 0

    (la, A), (lb, B) = tables[0], tables[1]
    keys = sorted(set(A) & set(B))
    print(f"\n{len(keys)} kernels in both  ({len(A)} in {la}, {len(B)} in {lb})")
    moved = []
    for k in keys:
        if args.filter and args.filter not in k:
            continue
        a, b = A[k], B[k]
        if any(a.get(f, 0) != b.get(f, 0) for f in ("vgpr", "sgpr", "spill", "scratch", "lds")):
            moved.append(k)
    print(f"{len(moved)} changed resources under {lb}\n")

    # A per-kernel dump of 5,000 template instantiations is unreadable and hides the one fact that
    # matters: did the OCCUPANCY STEP move? VGPR drifting by one is noise unless it crosses a
    # granule boundary; crossing one is worth ~a wave per SIMD.
    fam = defaultdict(lambda: {"n": 0, "up": 0, "down": 0, "spill_up": 0, "spill_down": 0,
                               "best": None, "worst": None})
    for k in keys:
        if args.filter and args.filter not in k:
            continue
        a, b = A[k], B[k]
        f = fam[family(k)]
        f["n"] += 1
        oa, ob = occ_by_vgpr(a.get("vgpr", 0)), occ_by_vgpr(b.get("vgpr", 0))
        if ob > oa:
            f["up"] += 1
            if f["best"] is None or (ob - oa) > f["best"][0]:
                f["best"] = (ob - oa, a.get("vgpr", 0), b.get("vgpr", 0), oa, ob)
        elif ob < oa:
            f["down"] += 1
            if f["worst"] is None or (oa - ob) > f["worst"][0]:
                f["worst"] = (oa - ob, a.get("vgpr", 0), b.get("vgpr", 0), oa, ob)
        sa, sb = a.get("spill", 0), b.get("spill", 0)
        f["spill_up"] += sb > sa
        f["spill_down"] += sb < sa

    print(f"{'kernel family':<44}{'n':>6}{'occ+':>6}{'occ-':>6}{'spill+':>8}{'spill-':>8}"
          f"   {'biggest occupancy move (vgpr, waves/SIMD)':<44}")
    print("-" * 124)
    for name, f in sorted(fam.items(), key=lambda kv: -(kv[1]["up"] - kv[1]["down"])):
        note = ""
        if f["best"]:
            d, va, vb, oa, ob = f["best"]
            note = f"+{d}: vgpr {va}->{vb}, occ {oa}->{ob}"
        elif f["worst"]:
            d, va, vb, oa, ob = f["worst"]
            note = f"-{d}: vgpr {va}->{vb}, occ {oa}->{ob}"
        print(f"{name[:44]:<44}{f['n']:6d}{f['up']:6d}{f['down']:6d}"
              f"{f['spill_up']:8d}{f['spill_down']:8d}   {note}")

    tot_up = sum(f["up"] for f in fam.values())
    tot_dn = sum(f["down"] for f in fam.values())
    tot_su = sum(f["spill_up"] for f in fam.values())
    tot_sd = sum(f["spill_down"] for f in fam.values())
    print(f"\nTOTAL: {tot_up} kernels gain an occupancy step under {lb}, {tot_dn} lose one; "
          f"spill worse in {tot_su}, better in {tot_sd}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
