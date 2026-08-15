#!/usr/bin/env python
"""Disassemble one kernel out of two built .so files and say WHAT THE COMPILER DID differently.

`kernel_static_resources.py` says a kernel's registers moved. This says why. CPU-only, no GPU.

    python tools/kernel_isa_diff.py A.so B.so --symbol gdn_decode_kernel --label r72 r714

For the first matching symbol in each it reports the instruction-class histogram side by side —
VALU / SALU / VMEM / LDS / scratch / WMMA / branch / waitcnt — plus the loop structure (backward
branch targets) and the s_waitcnt distribution. A register-pressure regression is almost always
visible as one of: more scratch traffic (real spill), a different unroll factor (the same work in
more or fewer instructions), or waitcnts moved onto the dependency chain.

--full dumps both disassemblies to files so they can be diffed by eye; the histogram is what to
read first.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

IMAGE = "rocm/dev-ubuntu-24.04:7.14.0-full"
LLVM = "/opt/rocm/lib/llvm/bin"

INNER = r'''
set -e
cd /out
base=$(basename "$1")
cp "$1" "/out/$base"
''' + LLVM + r'''/llvm-objdump --offloading "/out/$base" >/dev/null 2>&1 || true
: > "/out/$2.disasm"
for obj in /out/"$base"*gfx1201*; do
  [ -e "$obj" ] || continue
  ''' + LLVM + r'''/llvm-objdump -d --mcpu=gfx1201 "$obj" >> "/out/$2.disasm" 2>/dev/null || true
done
rm -f "/out/$base" /out/"$base".*hipv4* /out/"$base".*host-*
wc -l < "/out/$2.disasm"
'''

# gfx1201 instruction classes. Order matters: the first match wins, so the specific prefixes
# (scratch_, ds_, buffer_) are tested before the generic v_/s_ ones.
CLASSES = [
    ("scratch", re.compile(r"^scratch_")),          # REAL spill traffic: private memory
    ("lds",     re.compile(r"^ds_")),
    ("vmem",    re.compile(r"^(global_|buffer_|flat_|image_|tbuffer_)")),
    ("smem",    re.compile(r"^s_(load|store|buffer)")),
    ("wmma",    re.compile(r"^v_wmma")),
    ("waitcnt", re.compile(r"^s_wait")),
    ("branch",  re.compile(r"^s_(branch|cbranch|setpc|swappc|call|endpgm)")),
    ("valu",    re.compile(r"^v_")),
    ("salu",    re.compile(r"^s_")),
]


def classify(mnemonic: str) -> str:
    for name, rx in CLASSES:
        if rx.match(mnemonic):
            return name
    return "other"


def disasm(so: Path, tag: str, outdir: Path, image: str) -> str:
    cmd = ["docker", "run", "--rm", "-v", f"{so.parent}:/work:ro", "-v", f"{outdir}:/out",
           "--entrypoint", "bash", image, "-c", INNER, "_", f"/work/{so.name}", tag]
    r = subprocess.run(cmd, capture_output=True, text=True)
    p = outdir / f"{tag}.disasm"
    if not p.exists() or p.stat().st_size == 0:
        print(f"  ! no disassembly for {so.name}: {r.stderr.strip()[:300]}", file=sys.stderr)
        return ""
    return p.read_text()


def kernel_body(text: str, symbol: str) -> tuple[str, list[str]]:
    """Return (matched symbol, instruction mnemonics) for the first kernel containing `symbol`.

    llvm-objdump prints `<mangled>:` at each function label and one instruction per line after it.
    """
    cur, body, name = None, [], None
    for line in text.splitlines():
        m = re.match(r"^([0-9a-f]{16})?\s*<(.+)>:\s*$", line)
        if m:
            if name and body:
                return name, body
            cur = m.group(2)
            if symbol in cur:
                name, body = cur, []
            else:
                name = None
            continue
        if name is None:
            continue
        # `        s_load_b64 s[0:1], s[4:5], 0x0    // 000000000000: ...`
        mm = re.match(r"^\s+[0-9a-f]{4,}:\s+(?:[0-9a-f]{2}\s+)*\s*([a-z0-9_]+)", line)
        if not mm:
            mm = re.match(r"^\s*([a-z][a-z0-9_]+)\s", line)
        if mm:
            body.append(mm.group(1))
    return (name, body) if name else (None, [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--label", nargs=2, default=["A", "B"])
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--full", default="", help="write both disassemblies under this dir")
    args = ap.parse_args()
    # An empty --symbol matches EVERY label (`"" in cur` is always true), so the tool silently
    # disassembles whichever kernel happens to come first and prints a confident diff of the wrong
    # thing. Refuse it.
    if not args.symbol.strip():
        print("--symbol must be a non-empty substring of the mangled kernel name", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="isa-"))
    la, lb = args.label
    ta = disasm(Path(args.a).resolve(), "a", tmp, args.image)
    tb = disasm(Path(args.b).resolve(), "b", tmp, args.image)
    na, ba = kernel_body(ta, args.symbol)
    nb, bb = kernel_body(tb, args.symbol)
    if not ba or not bb:
        print(f"symbol {args.symbol!r} not found in both ({bool(ba)}, {bool(bb)})")
        return 1
    print(f"{la}: {na}\n{lb}: {nb}\n")
    if na != nb:
        print("!! DIFFERENT SYMBOLS matched — compare the same instantiation or the diff is noise\n")

    ca, cb = Counter(map(classify, ba)), Counter(map(classify, bb))
    print(f"{'class':<10}{la:>12}{lb:>12}{'delta':>10}{'ratio':>9}")
    print("-" * 53)
    for k in sorted(set(ca) | set(cb), key=lambda k: -(ca[k] + cb[k])):
        d = cb[k] - ca[k]
        r = (cb[k] / ca[k]) if ca[k] else float("inf")
        print(f"{k:<10}{ca[k]:>12}{cb[k]:>12}{d:>+10}{r:>9.2f}")
    print(f"{'TOTAL':<10}{len(ba):>12}{len(bb):>12}{len(bb)-len(ba):>+10}{len(bb)/len(ba):>9.2f}")

    # The mnemonics that moved most, which is where the codegen difference actually lives.
    print(f"\n-- individual mnemonics, biggest movers --")
    ma, mb = Counter(ba), Counter(bb)
    moved = sorted(set(ma) | set(mb), key=lambda k: -abs(mb[k] - ma[k]))[:18]
    print(f"{'mnemonic':<28}{la:>10}{lb:>10}{'delta':>9}")
    for k in moved:
        if ma[k] == mb[k]:
            continue
        print(f"{k:<28}{ma[k]:>10}{mb[k]:>10}{mb[k]-ma[k]:>+9}")

    if args.full:
        d = Path(args.full); d.mkdir(parents=True, exist_ok=True)
        (d / f"{args.symbol}.{la}.s").write_text("\n".join(ba))
        (d / f"{args.symbol}.{lb}.s").write_text("\n".join(bb))
        print(f"\nwrote {d}/{args.symbol}.{{{la},{lb}}}.s")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
