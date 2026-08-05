"""Difference the `[canvas-timing]` cumulative averages into per-window step times.

`_StepTimer` prints a RUNNING MEAN every 10 steps, so the value at n=70 averages in the cold first
steps and is not a step time. The step time is (sum_n - sum_{n-10}) / 10, i.e. the difference of
consecutive cumulative windows — the method docs §D11.4's 86.9 ms was taken with. Reads the
`[canvas-timing]` lines on stdin.
"""
import re
import sys

FIELDS = ("fwd_issue", "fwd_tail", "sampler", "soft_embed", "step")


def main() -> int:
    rows = []
    for line in sys.stdin:
        if "[canvas-timing]" not in line:
            continue
        d = dict(re.findall(r"(\w+)=([0-9.]+)", line))
        if "n" not in d or "step" not in d:
            continue
        rows.append({k: float(v) for k, v in d.items()})
    if len(rows) < 2:
        print(f"  ({len(rows)} 10-step report(s) — need at least two to difference a window)")
        return 0
    print("  window        issue   tail   sampler  soft    STEP")
    steps = []
    for a, b in zip(rows, rows[1:]):
        na, nb = a["n"], b["n"]
        w = nb - na
        if w <= 0:
            continue
        v = {k: (b[k] * nb - a[k] * na) / w for k in FIELDS}
        steps.append(v["step"])
        print(f"  n={na:>3.0f}->{nb:<4.0f}  {v['fwd_issue']:6.1f} {v['fwd_tail']:6.1f} "
              f"{v['sampler']:7.1f} {v['soft_embed']:6.1f}  {v['step']:7.1f}")
    if steps:
        s = sorted(steps)
        med = s[len(s) // 2]
        print(f"  --> median STEP = {med:.1f} ms/step over {len(s)} windows; "
              f"min={s[0]:.1f} max={s[-1]:.1f} spread={100 * (s[-1] - s[0]) / med:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
