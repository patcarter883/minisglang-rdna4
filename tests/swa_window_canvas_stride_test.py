"""The SWA window snapshot across a BLOCK-DIFFUSION canvas: stride, aliasing, and round-trip.

CPU-only, no weights, no GPU — it is arithmetic over a stub ring pool. Run inside the serve
image (the host torch install is broken):

    docker run --rm --entrypoint bash -v <worktree>:/wt minisgl-rdna4:gemma4 \
      -lc 'cd /wt && PYTHONPATH=/wt/python:/opt/kernels \
        python tests/swa_window_canvas_stride_test.py'

WHY THIS EXISTS. The block-diffusion serve shipped with SWA-radix forced OFF on the argument that
"the window snapshot addresses the ring at the PRE-CANVAS stride, and the canvas widens that ring".
That argument is wrong, and the wrongness is not visible by reading either file alone: the ring
stride is ONE boot-time constant, `sliding_window + _swa_ring_block(...)` = `W + max(spec, canvas)`,
published on `ctx.swa_ring_stride` and read by every store/gather/decode/snapshot site. There is no
second, narrower stride for a canvas snapshot to disagree with. What was actually missing was the
RESTORE call on the canvas loop's prefill path.

Four properties are pinned, each of which fails SILENTLY if it drifts — a wrong window does not
crash, it answers from scrambled keys:

  1. ONE STRIDE, EVERY PHASE. `_swa_ring_block` is the only source, and it already folds the canvas
     in, so the number a prompt-encoder snapshot uses is the number a mid-canvas decode uses.
  2. A CANVAS CANNOT TOUCH A LIVE WINDOW at that stride. [b-W, b) and [c0, c0+L) span at most
     W+L == R consecutive positions, so every slot is distinct — checked exhaustively over the
     boundaries a block can commit at.
  3. AND IT TOTALLY ALIASES AT R == W: 256 of 256 canvas slots land on window slots. This is the
     failure the widened stride exists to prevent, asserted rather than asserted-about.
  4. CLONE/RESTORE ROUND-TRIPS ACROSS A CANVAS. Snapshot at c0, let a canvas scribble over
     [c0, c0+L), restore into a DIFFERENT ring block: the window comes back byte-identical.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "python")

from minisgl.engine.engine import _swa_ring_block  # noqa: E402
from minisgl.kvcache.swa_window import SWAWindowSnapshotter  # noqa: E402

W = 1024        # DiffusionGemma / gemma-4 sliding window
L = 256         # canvas_length
NUM_LAYERS = 3  # enough to prove the per-layer loop; the real model has 25
HK, D = 2, 8
SEQS = 4


class _StubModelConfig:
    def __init__(self, canvas_length):
        self.canvas_length = canvas_length

    @property
    def is_block_diffusion(self) -> bool:
        return self.canvas_length is not None and self.canvas_length > 0


class _StubSpecConfig:
    def __init__(self, num_draft):
        self.num_draft = num_draft


class _StubRing:
    """The MHAKVCache surface SWAWindowSnapshotter uses: k_cache/v_cache(lid) -> [slots, 1, Hk, D]."""

    def __init__(self, stride: int):
        self.num_layers = NUM_LAYERS
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        n = (SEQS + 2) * stride
        # Distinct per (layer, slot) so any mis-addressed slot shows up as a value mismatch.
        self._k = [torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 1, HK, D)
                   .contiguous() + lid * 1e6 for lid in range(NUM_LAYERS)]
        self._v = [k + 0.5 for k in self._k]

    def k_cache(self, lid):
        return self._k[lid]

    def v_cache(self, lid):
        return self._v[lid]


def _slots(table_idx, positions, R):
    return {table_idx * R + (p % R) for p in positions}


def main() -> int:
    fails = []

    # ---- 1. ONE stride, and it already carries the canvas -------------------------------------
    ar = _StubModelConfig(None)
    dg = _StubModelConfig(L)
    cases = [
        ("AR, no spec", ar, None, 0),
        ("AR, spec K=4", ar, _StubSpecConfig(4), 5),
        ("canvas, no spec", dg, None, L),
        ("canvas + spec K=4", dg, _StubSpecConfig(4), L),  # max(5, 256)
    ]
    for name, mc, sc, want in cases:
        got = _swa_ring_block(mc, sc)
        ok = got == want
        fails += [] if ok else [f"ring block {name}: {got} != {want}"]
        print(f"{'PASS' if ok else 'FAIL'}  ring block {name:20s} = {got} (want {want})")
    R = W + _swa_ring_block(dg, None)
    print(f"      canvas ring stride R = W + canvas = {W} + {L} = {R}")

    # ---- 2. no canvas slot can land on a live window slot ---------------------------------------
    # (a) DURING the block. `_build_swa_canvas_metadata` fixes the window at the CACHED length, so a
    #     denoising step reads [c0-W, c0) while writing [c0, c0+L). Those must not share a slot, or
    #     the decoder overwrites the prefix it is attending to.
    c0 = 5000
    live = len(_slots(1, range(c0 - W, c0), R) & _slots(1, range(c0, c0 + L), R))
    ok = live == 0
    fails += [] if ok else [f"{live} window/canvas slot collisions at stride {R}"]
    print(f"{'PASS' if ok else 'FAIL'}  live window x canvas slot collisions: {live} (want 0)")

    # (b) AFTER a commit of n < L. The tail [c0+n, c0+L) is stale canvas junk nothing re-encodes, and
    #     the next window is [c0+n-W, c0+n). Sweep every commit length a block can produce: the junk
    #     must land strictly BELOW the window, never inside it — otherwise a snapshot taken at the
    #     commit boundary would carry denoising scratch into a reusing request's prefix.
    stale = 0
    for n in range(1, L + 1):
        b = c0 + n
        stale += len(_slots(1, range(b - W, b), R) & _slots(1, range(b, c0 + L), R))
    ok = stale == 0
    fails += [] if ok else [f"{stale} stale-canvas/window collisions at stride {R}"]
    print(f"{'PASS' if ok else 'FAIL'}  post-commit stale canvas x window collisions over all {L} "
          f"commit lengths: {stale} (want 0)")

    # ---- 3. ...and it is total at the un-widened stride ----------------------------------------
    win_w = _slots(1, range(c0 - W, c0), W)
    canvas_w = _slots(1, range(c0, c0 + L), W)
    n_alias = len(win_w & canvas_w)
    ok = n_alias == L
    fails += [] if ok else [f"stride==W aliasing {n_alias} != {L}"]
    print(f"{'PASS' if ok else 'FAIL'}  at stride == W the canvas aliases {n_alias}/{L} window slots")

    # A stride BELOW the window is the one setting that scrambles a snapshot against itself.
    try:
        SWAWindowSnapshotter(_StubRing(W), W, ring_stride=W // 2)
        fails.append("SWAWindowSnapshotter accepted ring_stride < window")
        print("FAIL  SWAWindowSnapshotter accepted ring_stride < window")
    except AssertionError:
        print("PASS  SWAWindowSnapshotter REFUSES ring_stride < window")

    # ---- 4. clone -> canvas scribble -> restore elsewhere, byte-identical ----------------------
    ring = _StubRing(R)
    snap = SWAWindowSnapshotter(ring, W, ring_stride=R)
    src, dst = 1, 3
    boundary = c0  # page-aligned in the serve; the arithmetic does not care
    want_k = ring.k_cache(0)[[src * R + (p % R) for p in range(boundary - W, boundary)], 0].clone()

    s = snap.clone(src, boundary)
    # A canvas denoising step overwrites [c0, c0+L) in the SOURCE block, repeatedly.
    for step in range(3):
        for lid in range(NUM_LAYERS):
            idx = [src * R + (p % R) for p in range(c0, c0 + L)]
            ring.k_cache(lid)[idx, 0] = -float(step + 1)
            ring.v_cache(lid)[idx, 0] = -float(step + 1)
    # The snapshot taken BEFORE the canvas must still describe the window...
    s2 = snap.clone(src, boundary)
    ok = torch.equal(s[1], s2[1]) and torch.equal(s[2], s2[2])
    fails += [] if ok else ["a canvas write disturbed the window a pre-canvas snapshot describes"]
    print(f"{'PASS' if ok else 'FAIL'}  window survives {3 * L} canvas writes to the same ring block")

    # ...and restoring it into a fresh block must reproduce it exactly.
    snap.restore_ring(dst, s)
    got_k = ring.k_cache(0)[[dst * R + (p % R) for p in range(boundary - W, boundary)], 0]
    ok = torch.equal(got_k, want_k)
    fails += [] if ok else ["restore_ring did not reproduce the cloned window"]
    print(f"{'PASS' if ok else 'FAIL'}  clone -> restore round-trip into a different ring block "
          f"(max|delta| = {(got_k - want_k).abs().max().item():.3e})")

    # The restore must also not have spilled into the destination's canvas region.
    canvas_idx = [dst * R + (p % R) for p in range(c0, c0 + L)]
    untouched = ring.k_cache(0)[canvas_idx, 0]
    expect = torch.tensor([float(i) for i in canvas_idx]).view(-1, 1, 1).expand(L, HK, D)
    ok = torch.equal(untouched, expect)
    fails += [] if ok else ["restore_ring wrote into the destination's canvas slots"]
    print(f"{'PASS' if ok else 'FAIL'}  restore_ring left the destination's {L} canvas slots alone")

    print()
    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
