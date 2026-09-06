"""Floor-aware two-build comparison.

`compare_parity.py` answers "are these two dumps bit-identical". That is the right question only for
tensors the hardware reproduces run-to-run. `mmq_fp8_moe_gemm_scatter` accumulates through a global
atomicAdd whose order varies, so a cross-build delta there is not evidence of a code change — it is
the same thing the engine's own `quant/kernels.py` documents.

So this reads FOUR dumps: two per build. Per tensor it computes
  floor = max(|base_a - base_b|, |new_a - new_b|)     # what the same code does to itself
  cross = |base_a - new_a|                            # what the change did
and reports a tensor as
  BIT-EXACT      cross == 0
  UNDER FLOOR    cross != 0 but floor != 0 and cross <= floor   (not attributable)
  REGRESSION     cross != 0 and floor == 0                      (the finding)
"""
import sys

import torch

base_a, base_b, new_a, new_b = (torch.load(p, map_location="cpu") for p in sys.argv[1:5])

keys = sorted(set(base_a) | set(new_a))
rows = []
missing = []
for k in keys:
    if k not in base_a or k not in new_a:
        missing.append(k)
        continue
    a, b = base_a[k], new_a[k]
    if a.shape != b.shape:
        rows.append((k, float("nan"), float("nan"), "SHAPE"))
        continue

    def d(x, y):
        na, nb = torch.isnan(x), torch.isnan(y)
        if not torch.equal(na, nb):
            return float("inf")
        fin = ~na
        return (x[fin] - y[fin]).abs().max().item() if fin.any() else 0.0

    floor = max(d(base_a[k], base_b[k]), d(new_a[k], new_b[k]))
    cross = d(a, b)
    if cross == 0.0:
        verdict = "BIT-EXACT"
    elif floor == 0.0:
        verdict = "REGRESSION"
    elif cross <= floor:
        verdict = "UNDER-FLOOR"
    else:
        verdict = "ABOVE-FLOOR"
    rows.append((k, cross, floor, verdict))

from collections import Counter  # noqa: E402

grp: dict = {}
for k, cross, floor, v in rows:
    g = k.split("/")[0]
    grp.setdefault(g, Counter())[v] += 1

print(f"{'op group':26s} {'BIT-EXACT':>10s} {'UNDER-FLOOR':>12s} {'REGRESSION':>11s} "
      f"{'ABOVE-FLOOR':>12s} {'SHAPE':>6s}")
for g in sorted(grp):
    c = grp[g]
    print(f"{g:26s} {c['BIT-EXACT']:10d} {c['UNDER-FLOOR']:12d} {c['REGRESSION']:11d} "
          f"{c['ABOVE-FLOOR']:12d} {c['SHAPE']:6d}")

bad = [r for r in rows if r[3] in ("REGRESSION", "ABOVE-FLOOR", "SHAPE")]
print(f"\n{len(rows)} tensors compared, {len(missing)} present in only one build")
for k in missing:
    print(f"  ONLY-ONE-BUILD {k}")
if bad:
    print(f"\n{len(bad)} NOT attributable to the same-code floor:")
    for k, cross, floor, v in bad:
        print(f"  {v:12s} {k}  cross={cross:.6g} floor={floor:.6g}")
else:
    print("\nEvery tensor is either bit-exact or inside its own same-code floor.")

nonzero_floor = [r for r in rows if r[2] not in (0.0,) and r[2] == r[2]]
print(f"\ntensors with a NON-ZERO same-code floor (i.e. the gate cannot discriminate on them): "
      f"{len(nonzero_floor)}")
for k, cross, floor, v in nonzero_floor[:20]:
    print(f"  floor={floor:.6g} cross={cross:.6g}  {k}")
sys.exit(1 if bad or missing else 0)
