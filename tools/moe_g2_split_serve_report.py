"""Median-of-N summary of the MoE gemm2 split serve A/Bs, with the control rows marked.

A ratio is only meaningful next to the spread it came out of, so every row prints min-max for both
legs; a ratio whose legs overlap is reported as inside the noise rather than as a win.

  python3 tools/moe_g2_split_serve_report.py tools/_fixtures/moe_g2_split_serve_ab_*.txt
"""
from __future__ import annotations

import re
import statistics
import sys

# Which bs is the measurement and which is the control, per arm — see the A/B script.
ROLE = {
    "fused": {1: "control", 5: "MEASURED", 6: "MEASURED"},
    "scatter": {1: "MEASURED", 5: "control", 6: "control"},
}


def parse(path):
    leg, rows = None, {}
    for line in open(path):
        m = re.search(r"LEG (\w+)", line)
        if m:
            leg = m.group(1)
        m = re.search(r"bs=(\d+) tokens=\d+ decode_tok_s=([\d.]+)", line)
        if m and leg:
            rows.setdefault((leg, int(m.group(1))), []).append(float(m.group(2)))
    return rows


def main() -> int:
    for path in sys.argv[1:]:
        arm = "scatter" if "scatter" in path else "fused"
        rows = parse(path)
        print(f"\n=== {path}   (arm={arm}) ===")
        print(f"{'bs':>3} {'role':>9} {'base med':>10} {'base range':>16} "
              f"{'new med':>10} {'new range':>16} {'ratio':>8}  verdict")
        for bs in (1, 5, 6):
            b, n = rows.get(("base", bs)), rows.get(("new", bs))
            if not b or not n:
                continue
            mb, mn = statistics.median(b), statistics.median(n)
            overlap = not (min(n) > max(b) or max(n) < min(b))
            role = ROLE[arm][bs]
            verdict = ("legs overlap -> inside the noise" if overlap else
                       ("separated -> real" if mn > mb else "separated -> REGRESSION"))
            if role == "control":
                verdict += " (control: expect no move)"
            print(f"{bs:>3} {role:>9} {mb:>10.2f} {min(b):>7.2f}-{max(b):<8.2f} "
                  f"{mn:>10.2f} {min(n):>7.2f}-{max(n):<8.2f} {mn / mb:>7.4f}x  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
