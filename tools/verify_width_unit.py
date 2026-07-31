"""Host-side unit gate for the pure half of the adaptive verify width. NO GPU, NO torch.

Everything here is reachable only in configurations that are expensive or impossible to stand up on
demand (DP+EP, an int4 checkpoint driven past K=8, a controller that must climb back after
narrowing), which is exactly why the logic was factored into pure functions.

    python3 tools/verify_width_unit.py
"""
import importlib.util
import os
import sys

# Load spec/width.py DIRECTLY, not through `minisgl.spec`, whose __init__ imports torch (which the
# host python cannot load — libmpi_cxx.so.40). width.py itself is torch-free by construction, which
# is what makes this gate runnable without a container or a card.
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_width", os.path.join(_HERE, "..", "python", "minisgl", "spec", "width.py"))
_w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_w)
MAX_VERIFY_ROWS = _w.MAX_VERIFY_ROWS
AdaptiveVerifyWidth = _w.AdaptiveVerifyWidth
max_verify_rows = _w.max_verify_rows
pad_to_captured_width = _w.pad_to_captured_width
verify_width_ladder = _w.verify_width_ladder

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILED.append(name)


class _Q:
    """Stand-in for QuantConfig: only the family predicates matter to max_verify_rows."""

    def __init__(self, **kw):
        self.is_int4 = kw.get("int4", False)
        self.weight_is_e2m1 = kw.get("e2m1", False)
        self.is_nvfp4 = kw.get("nvfp4", False)


print("== 1. the M ceiling is DERIVED, and int4 is the one family that is not 16 ==")
# On the host the kernel modules are unimportable (no torch), so max_verify_rows falls back to its
# own MAX_VERIFY_ROWS; inside the image the imports succeed and the numbers are identical (16/16/8).
check("no quant -> the generic ceiling", max_verify_rows(None) == MAX_VERIFY_ROWS,
      f"({max_verify_rows(None)})")
check("e2m1 -> 16", max_verify_rows(_Q(e2m1=True)) == 16, f"({max_verify_rows(_Q(e2m1=True))})")
check("nvfp4 -> 16", max_verify_rows(_Q(nvfp4=True)) == 16)
_i4 = max_verify_rows(_Q(int4=True))
check("int4 -> 8 (measured cliff, tools/w4a8_int4_m_crossover.py)", _i4 == 8, f"({_i4})")

print("\n== 2. the ladder NEVER crosses its model's ceiling ==")
for q, cap, tag in ((None, MAX_VERIFY_ROWS, "generic"), (_Q(e2m1=True), 16, "e2m1"),
                    (_Q(int4=True), 8, "int4")):
    bad = []
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 16, 32, 64):
        lad = verify_width_ladder(k, q)
        if lad and max(lad) + 1 > cap:
            bad.append((k, lad))
    check(f"{tag}: max(width)+1 <= {cap} for every K", not bad, str(bad))
check("int4 K=16 ladder stays under 8", max(verify_width_ladder(16, _Q(int4=True))) == 7,
      str(verify_width_ladder(16, _Q(int4=True))))
check("shipped configs unaffected: MTP K=4 int4", verify_width_ladder(4, _Q(int4=True))
      == verify_width_ladder(4, None), str(verify_width_ladder(4, _Q(int4=True))))
check("shipped configs unaffected: EAGLE3 K=6 int4", verify_width_ladder(6, _Q(int4=True))
      == verify_width_ladder(6, None), str(verify_width_ladder(6, _Q(int4=True))))
check("Laguna K=16 e2m1 ladder unchanged at [3,7,15]",
      verify_width_ladder(16, _Q(e2m1=True)) == [3, 7, 15])

print("\n== 3. THE INVARIANT: staged rows are never NARROWER than the real drafts ==")
# This is the DP+EP defect. adaptive=False pins the widest captured width WITHOUT the controller
# having truncated first, so a num_draft above the M-cliff cap used to stage fewer rows than the
# accept loop then sliced -> a request reads its neighbour's logits (or trips verify_greedy).
cases = [
    ("DP+EP, K=16 vs ladder max 15", [list(range(16))] * 3, [3, 7, 15], False),
    ("DP+EP, K=16, ragged batch", [list(range(16)), list(range(9)), []], [3, 7, 15], False),
    ("DP+EP, int4 ladder max 7", [list(range(16))] * 2, [2, 4, 7], False),
    ("adaptive, already truncated", [list(range(7))] * 4, [3, 7, 15], True),
    ("adaptive, ragged", [list(range(7)), list(range(2)), list(range(15))], [3, 7, 15], True),
    ("single captured width", [list(range(4))] * 2, [2], False),
    ("no captured widths", [list(range(9))] * 2, [], False),
    ("all empty", [[], []], [3, 7, 15], True),
]
for name, drafts, widths, adaptive in cases:
    d2, staged, w, pad = pad_to_captured_width([list(d) for d in drafts], widths, adaptive)
    ok_inv = all(len(s) >= len(d) for s, d in zip(staged, d2))
    ok_uniform = (not widths) or len({len(s) for s in staged}) == 1
    ok_cap = (not widths) or all(len(d) <= widths[-1] for d in d2)
    ok_prefix = all(s[:len(d)] == d for s, d in zip(staged, d2))
    check(name, ok_inv and ok_uniform and ok_cap and ok_prefix,
          f"w={w} pad={pad} staged={[len(s) for s in staged]} real={[len(d) for d in d2]}")
# The pre-fix behaviour, stated so the regression is unmistakable: the old code padded `d[:w_pad]`
# but left `drafts` at full length.
old_staged = [d[:15] + [0] * (15 - min(len(d), 15)) for d in [list(range(16))] * 2]
check("PRE-FIX form really was broken (kept as the regression witness)",
      any(len(s) < 16 for s in old_staged), f"staged={[len(s) for s in old_staged]} real=[16, 16]")

print("\n== 4. the controller adapts, and can climb back ==")
c = AdaptiveVerifyWidth([3, 7, 15])
for _ in range(40):
    w = c.choose([1])
    c.record([1], [min(2, w)], w)
check("a request accepting ~2 settles at width 3", c.choose([1]) == 3, f"({c.choose([1])})")
c2 = AdaptiveVerifyWidth([3, 7, 15])
for _ in range(40):
    w = c2.choose([1])
    c2.record([1], [w], w)          # saturating: accepts every row it is given
check("a saturating request stays at the max", c2.choose([1]) == 15, f"({c2.choose([1])})")
for _ in range(40):                  # ...and one that improves after narrowing climbs back
    w = c.choose([1])
    c.record([1], [min(15, w)], w)
check("a narrowed request climbs back to the max", c.choose([1]) == 15, f"({c.choose([1])})")
c3 = AdaptiveVerifyWidth([4])
check("a single-width ladder reports NOT adaptive", not c3.adaptive)

print("\n" + ("ALL PASS" if not FAILED else f"FAILED: {FAILED}"))
sys.exit(1 if FAILED else 0)
