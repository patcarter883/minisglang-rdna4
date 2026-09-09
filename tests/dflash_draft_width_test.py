"""CPU unit tests for the DFlash draft WIDTH: what the checkpoint can emit vs what the table asks.

Run:  PYTHONPATH=python python tests/dflash_draft_width_test.py

THE MEASURED DEFECT this pins. A serve of cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit with SPEC=dflash and
serve.sh's shipped `k_dflash=15` (drafter z-lab/Qwen3.6-35B-A3B-DFlash) logged, over 4100 verify
steps:

    [spec] mean accept-len=1.20 over 4100 reqs [accepted-drafts/verify; committed/verify=2.20]
           verify-width[ 0:26(1%)  3:2751(67%)  7:1323(32%)  15:0(0%) ]

i.e. the configured 15 was never the width that ran, and nothing said so. Two separate things were
hiding behind that one line, and the tests below separate them:

  * THE 3/7 IS NOT THE DRAFTER. K sizes the captured verify-width LADDER (spec/width.py
    `verify_width_ladder`: 15 -> [3, 7, 15]) and `AdaptiveVerifyWidth.choose` picks a rung per step
    from censoring-corrected acceptance. At E[accepted] = 1.20 the rule `mean + 1` wants 2.2, which
    rounds up to rung 3 — so 3/7/never-15 is the controller working, and K=15 is right for a drafter
    whose block is 16. `test_measured_histogram_is_the_ladder_not_the_drafter` reproduces it.

  * A K PAST THE BLOCK IS SILENTLY CLIPPED. A DFlash step feeds one block of [anchor, mask x (B-1)],
    so `propose` clips with `min(num_draft, B - 1, ...)` while engine.py still reserves verify
    buffers at K+1 rows. serve.sh shipped `k_dflash=16` on the laguna arm against a drafter that
    declares `block_size: 16` — inert, from the table's first commit, with no warning anywhere.
    `test_num_draft_past_block_warns_naming_both_numbers` and
    `test_serve_table_k_fits_every_drafters_block` are the guards.

Block sizes below are transcribed from the checkpoints' own config.json (fetched 2026-09-10), so the
table guard needs no weights, no HF hub and no card.
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

from minisgl.spec.dflash import dflash_block_size, dflash_check_num_draft
from minisgl.spec.width import AdaptiveVerifyWidth, verify_width_ladder

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# checkpoint -> declared block, and WHERE it is declared. None = the checkpoint declares no block and
# takes `1 + num_draft` by construction (the ZAYA CCA drafters), so no K can exceed it.
_DECLARED_BLOCK = {
    "z-lab/Qwen3.6-35B-A3B-DFlash": 16,        # dflash_config.block_size
    "z-lab/Qwen3.6-27B-DFlash": 16,            # TOP-LEVEL block_size
    "meta-models/Muse-Glimmer-30B-assistant": 16,   # TOP-LEVEL block_size
    "poolside/Laguna-XS-2.1-DFlash-NVFP4": 16,      # dflash_config.block_size
    "RadixArk/Qwen3.8-27B-DSpark": 7,          # dflash_config.block_size — the odd one out
    "/drafts/ZAYA1-8B-DFlash-CCA-5L-minv-ep4": None,
    # Not in the table, but cached on this box and reachable as `DRAFT=` — and it is the checkpoint
    # that STATES the block/width relationship this file asserts, rather than leaving it implied:
    # top-level `block_size: 8` alongside `speculators_config.proposal_methods[0].
    # speculative_tokens: 7`. At serve.sh's k_dflash=15 it would draft 7 and warn.
    "poolside/Laguna-XS.2-speculator.dflash": 8,
}


class _RecordingLogger:
    """Stands in for the module logger so a warning is an assertable value, not a side effect."""

    def __init__(self) -> None:
        self.warnings: list = []

    def warning_rank0(self, msg: str) -> None:
        self.warnings.append(msg)

    def info_rank0(self, msg: str) -> None:
        pass


def _check(name: str, got, want) -> None:
    assert got == want, f"{name}: got {got!r}, want {want!r}"
    print(f"  ok  {name}")


def test_block_size_reads_every_declaration_dialect() -> None:
    print("dflash_block_size dialects:")
    # z-lab / poolside / RadixArk: inside dflash_config.
    cfg = SimpleNamespace(dflash_config={"block_size": 16, "mask_token_id": 248077})
    _check("dflash_config", dflash_block_size(cfg, num_draft=15)[0], 16)
    _check("dspark-block-7", dflash_block_size(
        SimpleNamespace(dflash_config={"block_size": 7}), num_draft=6)[0], 7)

    # Muse-Glimmer / z-lab 27B: TOP-LEVEL, with no block in the sub-dict. This is the case the
    # sub-dict-only lookup missed, and it is why serve.sh's qwen27b comment claimed the checkpoint
    # "states no block_size".
    cfg = SimpleNamespace(dflash_config={"mask_token_id": 248070}, block_size=16)
    _check("top-level", dflash_block_size(cfg, num_draft=15)[0], 16)
    _check("top-level.source", dflash_block_size(cfg, num_draft=15)[1], "top-level block_size")

    # No sub-dict at all (Muse-Glimmer ships none).
    _check("no-dflash-config", dflash_block_size(SimpleNamespace(block_size=16), num_draft=15)[0], 16)

    # ZAYA CCA declares neither -> the trained block is num_spec masks + 1 anchor, so K sets it.
    cfg = SimpleNamespace(dflash_config={})
    _check("derived-from-k", dflash_block_size(cfg, num_draft=4)[0], 5)
    assert "derived" in dflash_block_size(cfg, num_draft=4)[1]
    print("  ok  derived.source names the derivation")


def test_env_override_raises_and_lowers_the_block() -> None:
    print("MINISGL_DFLASH_BLOCK overrides in both directions:")
    cfg = SimpleNamespace(dflash_config={"block_size": 16})
    prev = os.environ.get("MINISGL_DFLASH_BLOCK")
    try:
        os.environ["MINISGL_DFLASH_BLOCK"] = "4"
        _check("lower", dflash_block_size(cfg, num_draft=15)[0], 4)
        os.environ["MINISGL_DFLASH_BLOCK"] = "24"     # past the trained block: allowed, verify gates
        _check("raise", dflash_block_size(cfg, num_draft=15)[0], 24)
        _check("source", dflash_block_size(cfg, num_draft=15)[1], "MINISGL_DFLASH_BLOCK")
        os.environ["MINISGL_DFLASH_BLOCK"] = ""       # compose substitutes the EMPTY STRING
        _check("empty-is-unset", dflash_block_size(cfg, num_draft=15)[0], 16)
    finally:
        os.environ.pop("MINISGL_DFLASH_BLOCK", None)
        if prev is not None:
            os.environ["MINISGL_DFLASH_BLOCK"] = prev


def test_num_draft_past_block_warns_naming_both_numbers() -> None:
    print("dflash_check_num_draft:")
    import minisgl.spec.dflash as dfmod

    real, rec = dfmod.logger, _RecordingLogger()
    dfmod.logger = rec
    try:
        # The laguna arm as it shipped: K=16 against block 16 -> 15 deliverable, 1 silently dropped.
        w = dflash_check_num_draft(16, 16, "dflash_config.block_size",
                                   "poolside/Laguna-XS-2.1-DFlash-NVFP4")
        _check("clamped-width", w, 15)
        _check("warned-once", len(rec.warnings), 1)
        msg = rec.warnings[0]
        # BOTH numbers must appear, or the warning cannot be acted on.
        assert "16" in msg and "15" in msg, msg
        assert "poolside/Laguna-XS-2.1-DFlash-NVFP4" in msg, msg
        assert "SPEC_K=15" in msg, msg          # names the fix in the operator's own vocabulary
        print("  ok  warning names requested K, deliverable width, checkpoint and the fix")

        # A K that the block can carry is SILENT — a warning on every healthy boot is noise, and
        # noise is how the laguna one would have been ignored had it existed.
        rec.warnings.clear()
        _check("exact-fit", dflash_check_num_draft(15, 16, "dflash_config.block_size", "x"), 15)
        _check("exact-fit.silent", rec.warnings, [])
        _check("under-fit", dflash_check_num_draft(4, 16, "dflash_config.block_size", "x"), 15)
        _check("under-fit.silent", rec.warnings, [])

        # DSpark's block of 7: the shipped k_dflash=6 fits exactly, 7 would not.
        _check("dspark-k6", dflash_check_num_draft(6, 7, "dflash_config.block_size", "x"), 6)
        _check("dspark-k6.silent", rec.warnings, [])
        dflash_check_num_draft(7, 7, "dflash_config.block_size", "x")
        _check("dspark-k7.warns", len(rec.warnings), 1)
    finally:
        dfmod.logger = real


def test_serve_table_k_fits_every_drafters_block() -> None:
    """Every `dflash_draft=...; k_dflash=N` pair in tools/serve.sh, against that drafter's block.

    The guard that would have caught laguna's k_dflash=16 the day it was written. It reads the table
    rather than a copy of it, so a new arm is covered the moment it is added — an unknown drafter
    fails loudly here instead of silently skipping.
    """
    print("tools/serve.sh k_dflash vs declared block:")
    src = open(os.path.join(_REPO, "tools", "serve.sh")).read()
    pairs = re.findall(r'dflash_draft="([^"]+)"\s*;\s*k_dflash=(\d+)', src)
    assert pairs, "no `dflash_draft=...; k_dflash=N` pairs found — did the table's shape change?"
    seen = set()
    for ckpt, k in pairs:
        k = int(k)
        assert ckpt in _DECLARED_BLOCK, (
            f"serve.sh arm uses drafter {ckpt!r}, whose block_size is not recorded in this test. "
            "Read its config.json (dflash_config.block_size, else a top-level block_size, else None "
            "for a CCA drafter) and add it — an unrecorded drafter is exactly how a K past the block "
            "gets shipped.")
        block = _DECLARED_BLOCK[ckpt]
        if block is None:
            seen.add(ckpt)
            continue
        assert k <= block - 1, (
            f"serve.sh ships k_dflash={k} for {ckpt}, whose block_size is {block} — propose can "
            f"emit at most {block - 1} drafts, so {k - block + 1} of them are clipped away while "
            f"engine.py reserves verify buffers at {k}+1 rows.")
        seen.add(ckpt)
    print(f"  ok  {len(pairs)} arm(s) over {len(seen)} drafter(s) all fit their block")


def test_measured_histogram_is_the_ladder_not_the_drafter() -> None:
    """Reproduce verify-width[3:67% 7:32% 15:0%] from the MEASURED accept-len of 1.20.

    The point is negative: nothing about the drafter produces 3 and 7. K=15 captures [3, 7, 15] and
    the controller's `mean + 1` rule lands on 3 once it has evidence — so the shipped 15 is a
    CEILING that was never wrong, and lowering it in the table would have removed a rung the run is
    entitled to grow back into.
    """
    print("adaptive ladder from accept-len 1.20:")
    _check("ladder-K15", verify_width_ladder(15), [3, 7, 15])
    _check("ladder-K6", verify_width_ladder(6), [3, 4, 6])     # the DSpark arm

    ctl = AdaptiveVerifyWidth(verify_width_ladder(15))
    # Cold start explores from the MIDDLE rung — which is the 32% at 7 in the measured histogram.
    _check("cold-start", ctl.choose([1]), 7)

    # Feed the measured outcome: 80% of steps accept 1 draft, 20% accept 2 -> E[accepted] = 1.20.
    chosen = []
    for step in range(400):
        w = ctl.choose([1])
        chosen.append(w)
        n = 2 if step % 5 == 0 else 1
        ctl.record([1], [min(n, w)], w, offered=[w])
    assert abs(ctl.expected_run() - 1.20) < 0.05, ctl.expected_run()
    print(f"  ok  censoring-corrected E[accepted] = {ctl.expected_run():.2f} (measured 1.20)")
    _check("settles-at-3", ctl.choose([1]), 3)
    _check("never-picks-the-top-rung", 15 in chosen[50:], False)
    assert set(chosen) <= {3, 7}, sorted(set(chosen))
    print("  ok  only rungs 3 and 7 are ever chosen — exactly the measured histogram")

    # And the cost cap, not acceptance, is what pins a BATCHED step to 3 (bs=8 -> 8*(3+1)=32 rows).
    _check("bs8-cap", ctl.width_cap(8), 3)
    _check("bs1-uncapped", ctl.width_cap(1), 15)


if __name__ == "__main__":
    test_block_size_reads_every_declaration_dialect()
    test_env_override_raises_and_lowers_the_block()
    test_num_draft_past_block_warns_naming_both_numbers()
    test_serve_table_k_fits_every_drafters_block()
    test_measured_histogram_is_the_ladder_not_the_drafter()
    print("\nALL DFLASH DRAFT WIDTH TESTS PASSED")
