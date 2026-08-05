"""The partial-hit verdict for the block-diffusion SWA-radix gate: leg A vs leg B, cell by cell.

`swa_radix_client.py --mode matrix` cannot decide the PARTIAL-hit cell on its own, and on a canvas
serve it cannot even approximate it. Inside one serve the same prompt cannot be measured cold and
then partially-hit — measuring it cold inserts it into the radix, so the second request is a FULL
hit — and the AR gate's workaround (a cold reference on a different salt) rests on the answer being
salt-independent, which is true of an autoregressive recall tail and false of a canvas: a block
re-samples all 256 positions from noise conditioned on the whole prompt, so a different salt is a
different generation, full stop.

So the partial cell is a TWO-LEG comparison, and this reads it off the transcript:

    leg A   MINISGL_SWA_RADIX=1, prefix WARMED   -> `partial` is a real partial hit (hit > 0)
    leg B   MINISGL_SWA_RADIX=1, --no-warm       -> `partial` is the SAME prompt, cold (hit == 0)

Equal shas => the restored sliding window reproduced a cold prefill byte for byte. Two guards make
that falsifiable rather than decorative: the hit counters (A must be > 0 and B must be 0, or the two
legs did not measure what their names claim), and the `cold` step, which is identical in both legs
and therefore reports whether cross-serve output is reproducible at all — if it moved, the legs are
not comparable and the cell is INADMISSIBLE rather than DIVERGED.
"""
from __future__ import annotations

import re
import sys

STEP = re.compile(
    r"STEP run=(\S+) case=(\S+) max_tokens=(\d+) warm=(\d) step=(\S+) sha=(\S+) hit=(-?\d+)"
)
LEG = re.compile(r"=== LEG (\w+):")


def main() -> int:
    rows: dict = {}
    leg = None
    for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
        m = LEG.search(line)
        if m:
            leg = m.group(1)
        m = STEP.search(line)
        if m and leg:
            _run, case, mt, _warm, step, sha, hit = m.groups()
            rows[(leg, case, mt, step)] = (sha, int(hit))

    cells = sorted({(c, m) for (_l, c, m, _s) in rows})
    print("\n--- PARTIAL-HIT VERDICT (leg A partial hit vs leg B same prompt cold) ---")
    bad = 0
    for case, mt in cells:
        a, b = rows.get(("A", case, mt, "partial")), rows.get(("B", case, mt, "partial"))
        ca, cb = rows.get(("A", case, mt, "cold")), rows.get(("B", case, mt, "cold"))
        if not (a and b):
            continue
        admissible = bool(ca and cb and ca[0] == cb[0])
        if a[1] <= 0 or b[1] != 0:
            verdict, note = "INADMISSIBLE", f"hit counters wrong (A={a[1]} must be >0, B={b[1]} must be 0)"
        elif not admissible:
            verdict, note = "INADMISSIBLE", "the cold control moved across serves"
        else:
            verdict = "IDENTICAL" if a[0] == b[0] else "DIVERGED"
            note = "cold control held"
            bad += verdict == "DIVERGED"
        print(f"  case={case:6s} max_tokens={mt:3s} {verdict:13s} "
              f"A.partial hit={a[1]} sha={a[0]}  B.partial hit={b[1]} sha={b[0]}  [{note}]")
    print(f"{'PASS' if bad == 0 else f'FAIL ({bad} diverged)'}")
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
