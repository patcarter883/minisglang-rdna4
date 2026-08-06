"""Median-of-N report for glm_ab.sh, with the noise band stated rather than implied.

Reports MEDIANS, and alongside them the spread of the repeats, because the only honest way to say
"this landed" is to show the effect next to the run-to-run scatter it has to clear. A single-repeat
delta on this box has been wrong more than once.
"""
import argparse
import glob
import json
import os
import statistics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    # The banner used to hardcode "GLM ... SPEC=none". This report is now driven for Qwen
    # under SPEC=mtp too, and a report that names the wrong model is how a measurement gets
    # filed against the wrong config.
    ap.add_argument("--band", default="GLM-4.7-Flash-AWQ TP=2, SPEC=none, pinned --num-pages")
    a = ap.parse_args()

    legs = {}
    for f in sorted(glob.glob(os.path.join(a.results, "*.phases.json"))):
        tag = os.path.basename(f).split("-r")[0]
        try:
            d = json.load(open(f))
        except Exception:                                          # noqa: BLE001
            continue
        for p in d.get("phases", []):
            if p.get("agg_tok_s") is None:
                continue
            legs.setdefault(tag, {}).setdefault(p["label"], []).append(
                (p["agg_tok_s"], p.get("step_ms_from_gaps")))

    if "base" not in legs or "cand" not in legs:
        print(f"\n[report] need both legs; have {sorted(legs)}")
        return

    print(f"\n================ serve A/B: {a.band}  (MODE=base: no profiler attached) ================")
    print(f"{'phase':12s} {'n':>2s} {'base tok/s':>11s} {'cand tok/s':>11s} {'delta':>8s} "
          f"{'base spread':>12s} {'cand spread':>12s} {'verdict':>10s}")
    for label in ("decode_bs1", "decode_bs5", "decode_bs6", "prefill"):
        b = legs["base"].get(label)
        c = legs["cand"].get(label)
        if not b or not c:
            continue
        bt = [x[0] for x in b]
        ct = [x[0] for x in c]
        mb, mc = statistics.median(bt), statistics.median(ct)
        # The noise band: half the total spread of the two legs, relative to the base median. If the
        # effect does not clear it, the honest answer is "inside noise" — say so, do not round it up.
        sb = (max(bt) - min(bt)) / mb * 100 if len(bt) > 1 else float("nan")
        sc = (max(ct) - min(ct)) / mc * 100 if len(ct) > 1 else float("nan")
        delta = (mc / mb - 1) * 100
        band = max(sb if sb == sb else 0.0, sc if sc == sc else 0.0)
        verdict = "NOISE" if abs(delta) <= band else ("WIN" if delta > 0 else "LOSS")
        print(f"{label:12s} {min(len(bt), len(ct)):2d} {mb:11.2f} {mc:11.2f} {delta:+7.2f}% "
              f"{sb:11.2f}% {sc:11.2f}% {verdict:>10s}")
    print(f"\nBAND: served decode, {a.band}, auto clocks.")


if __name__ == "__main__":
    main()
