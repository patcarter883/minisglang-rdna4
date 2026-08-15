"""The dense W4A8 tile's LDS footprint — the ONE Python statement of it.

`(bm + bn) * (group_size + 8)` appeared in six places across this repo and the kernels repo, and
when the kernel's staging round stopped being one quant group deep, every one of those copies kept
answering for the old kernel. At g=16 the old formula says a 256x256 tile needs 12 KB; the real
footprint is 76 KB. A sweep or a fit using it will both skip tiles that are legal and admit tiles
that cannot launch.

The source of truth is `rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm/tile_select.h` (THE DENSE STAGING
POLICY / `tile_lds`). This module mirrors it for the CSV-analysis tools, which are deliberately
torch-free and so cannot call the package. `assert_matches_package()` closes that gap wherever torch
IS available: it asks the built extension and fails loudly on disagreement, so this copy cannot
drift silently the way its predecessors did.
"""
from __future__ import annotations

LDS_BUDGET = 65536
TARGET_STAGE_K = 128                      # K rows a staging round aims for
MAX_GROUPS_PER_STAGE = TARGET_STAGE_K // 16
LDS_PAD = 8
SCALE_BYTES_N = 4                         # one fp32 per N column PER STAGED GROUP


def groups_per_stage(gs: int) -> int:
    if gs <= 0:
        return 0
    return 1 if gs >= TARGET_STAGE_K else TARGET_STAGE_K // gs


def stage_depth(gs: int) -> int:
    """BK: K rows staged into LDS per round. Whole groups only."""
    return gs * groups_per_stage(gs)


def tile_lds(bm: int, bn: int, gs: int) -> int:
    """Dynamic staging tiles + the kernel's STATIC scale array, which the 64 KB also has to hold.

    The static array is sized at the compile-time MAX_GROUPS_PER_STAGE, not at this shape's
    groups-per-stage, so it costs the same at every group size.
    """
    return (bm + bn) * (stage_depth(gs) + LDS_PAD) + SCALE_BYTES_N * bn * MAX_GROUPS_PER_STAGE


def legal(bm: int, bn: int, gs: int, wn: int = 1) -> bool:
    """LDS fit plus WARPS_N legality: (BM/16)*WN warps, capped at 1024 threads, WN | (BN/16)."""
    if tile_lds(bm, bn, gs) > LDS_BUDGET:
        return False
    return (bm // 16) * wn <= 32 and (bn // 16) % wn == 0 and (bn // 16) >= wn


def assert_matches_package(group_sizes=(16, 32, 48, 64, 80, 96, 112, 128)) -> bool:
    """Cross-check every lattice tile against the built extension. Returns False if torch/the
    package is unavailable (the analysis tools run on the host, where it usually is)."""
    try:
        import fp8_wmma
    except Exception:
        return False
    bad = []
    for gs in group_sizes:
        for bm in (16, 32, 64, 80, 96, 112, 128, 192, 256, 384, 512):
            for bn in (16, 32, 64, 96, 128, 192, 256):
                mine = tile_lds(bm, bn, gs)
                theirs, _ = fp8_wmma.dense_tile_lds(bm, bn, gs)
                if mine != theirs:
                    bad.append((bm, bn, gs, mine, theirs))
    if bad:
        raise AssertionError(
            f"{len(bad)} tiles disagree with the package's tile_lds — this module has drifted from "
            f"tile_select.h. First: bm={bad[0][0]} bn={bad[0][1]} g={bad[0][2]} "
            f"here={bad[0][3]} package={bad[0][4]}")
    return True


if __name__ == "__main__":
    import sys
    ok = assert_matches_package()
    print("matches the built package" if ok else "package not importable here (host); formula unchecked")
    sys.exit(0)
