#!/usr/bin/env python
"""Does clang 23's register pressure EXPLAIN its lost dual-issue? Per-kernel correlation. No GPU.

RDNA's VOPD (`v_dual_*`) issues two VALU ops in one slot, but the pair is only legal under operand
constraints — the two halves must not collide on source register banks, and the ISA restricts which
opcodes may pair. So VOPD formation happens AFTER register allocation and is at its mercy: an
allocator that uses more registers, or assigns them across banks differently, silently makes fewer
pairs legal.

That gives a testable claim rather than a story: if pressure is the cause, kernels whose VGPR count
ROSE under clang 23 should be the same kernels that LOST v_dual_* pairs, and the two deltas should
correlate. If they do not correlate, VOPD formation changed on its own and register pressure is a
coincidence.

    python tools/vopd_pressure_correlation.py A.so B.so --label r72 r714
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kernel_static_resources as KSR   # reuse the metadata reader and the container recipe

DISASM = r'''
set -e
cd /out
base=$(basename "$1")
cp "$1" "/out/$base"
/opt/rocm/lib/llvm/bin/llvm-objdump --offloading "/out/$base" >/dev/null 2>&1 || true
: > "/out/$2.disasm"
for obj in /out/"$base"*gfx1201*; do
  [ -e "$obj" ] || continue
  /opt/rocm/lib/llvm/bin/llvm-objdump -d --mcpu=gfx1201 "$obj" >> "/out/$2.disasm" 2>/dev/null || true
done
rm -f "/out/$base" /out/"$base".*hipv4* /out/"$base".*host-*
echo OK
'''

LABEL = re.compile(r"^([0-9a-f]{16})?\s*<(.+)>:\s*$")
INSN = re.compile(r"^\t([a-z][a-z0-9_]*)\b")


def per_kernel(path: Path, tag: str, outdir: Path, image: str) -> dict[str, Counter]:
    cmd = ["docker", "run", "--rm", "-v", f"{path.parent}:/work:ro", "-v", f"{outdir}:/out",
           "--entrypoint", "bash", image, "-c", DISASM, "_", f"/work/{path.name}", tag]
    subprocess.run(cmd, capture_output=True, text=True)
    f = outdir / f"{tag}.disasm"
    if not f.exists():
        return {}
    out, cur, name = {}, Counter(), None
    for line in f.read_text().splitlines():
        m = LABEL.match(line)
        if m:
            if name:
                out[name] = cur
            name, cur = m.group(2), Counter()
            continue
        if name is None:
            continue
        mi = INSN.match(line)
        if mi:
            cur[mi.group(1)] += 1
    if name:
        out[name] = cur
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--label", nargs=2, default=["A", "B"])
    ap.add_argument("--image", default=KSR.IMAGE)
    args = ap.parse_args()
    la, lb = args.label
    tmp = Path(tempfile.mkdtemp(prefix="vopd-"))

    RA = KSR.extract(Path(args.a).resolve(), tmp, args.image)
    RB = KSR.extract(Path(args.b).resolve(), tmp, args.image)
    DA = per_kernel(Path(args.a).resolve(), "a", tmp, args.image)
    DB = per_kernel(Path(args.b).resolve(), "b", tmp, args.image)

    keys = [k for k in set(RA) & set(RB) if k in DA and k in DB]
    rows = []
    for k in keys:
        dv = RB[k].get("vgpr", 0) - RA[k].get("vgpr", 0)
        da = sum(v for m, v in DA[k].items() if m.startswith("v_dual_"))
        db = sum(v for m, v in DB[k].items() if m.startswith("v_dual_"))
        if da == 0 and db == 0:
            continue                       # kernel never had pairs; carries no signal either way
        rows.append((k, dv, db - da, da))
    if not rows:
        print("no kernels with VOPD pairs in common"); return 1

    print(f"{len(rows)} kernels that use VOPD in at least one build\n")
    # Bucket by what happened to registers, and report what happened to pairing in each bucket.
    def bucket(dv):
        return "vgpr UP" if dv > 0 else ("vgpr DOWN" if dv < 0 else "vgpr same")
    agg = {}
    for k, dv, dd, base in rows:
        b = bucket(dv)
        a = agg.setdefault(b, {"n": 0, "dual_delta": 0, "dual_base": 0, "lost": 0, "gained": 0})
        a["n"] += 1; a["dual_delta"] += dd; a["dual_base"] += base
        a["lost"] += dd < 0; a["gained"] += dd > 0
    print(f"{'register change':<16}{'kernels':>9}{'lost pairs':>12}{'gained':>9}"
          f"{'net v_dual delta':>19}{'% of their pairs':>18}")
    for b in ("vgpr UP", "vgpr same", "vgpr DOWN"):
        if b not in agg:
            continue
        a = agg[b]
        pct = 100.0 * a["dual_delta"] / a["dual_base"] if a["dual_base"] else 0.0
        print(f"{b:<16}{a['n']:>9}{a['lost']:>12}{a['gained']:>9}{a["dual_delta"]:>+19}{pct:>17.1f}%")

    # Pearson r between the two deltas.
    n = len(rows)
    mx = sum(r[1] for r in rows) / n
    my = sum(r[2] for r in rows) / n
    sxy = sum((r[1] - mx) * (r[2] - my) for r in rows)
    sxx = sum((r[1] - mx) ** 2 for r in rows)
    syy = sum((r[2] - my) ** 2 for r in rows)
    r = sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")
    print(f"\nPearson r(delta VGPR, delta v_dual) = {r:+.3f}   over {n} kernels")
    print("negative = kernels that gained registers lost dual-issue pairs, i.e. pressure is the cause")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
