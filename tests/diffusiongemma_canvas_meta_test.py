"""The canvas execution-mode plumbing: ring rows, ring stride, and the block-diffusion predicate.

No weights and no model — this is arithmetic. It still needs a card, because the metadata builders
allocate PINNED host staging buffers (`pin_memory=True`) exactly as the autoregressive ones do, and
that needs a CUDA context; it does not need much of one, so it rides along with the end-to-end run:

    docker run --rm --name dg_meta \
      --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --ipc host \
      -e HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES \
      -v <worktree>:/wt -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 --entrypoint bash minisgl-rdna4:gemma4 \
      -lc "PYTHONPATH=/wt/python:/opt/kernels python /wt/tests/diffusiongemma_canvas_meta_test.py"

Three things are pinned, each of which fails SILENTLY if it drifts — a wrong ring row does not
crash, it attends the wrong keys and produces fluent, wrong text:

  1. `_build_swa_canvas_metadata` lays out `[Wp window keys | L canvas keys]` per request with
     `context_len = Wp + L`, where Wp is computed from the CACHED length. Every canvas position sees
     the SAME window — the one the encoder left behind — not a window that slides per position.
  2. The window slots and the canvas slots are DISJOINT, which is only true because the ring stride
     was widened. At stride == window they collide completely.
  3. `_swa_ring_block` returns the max of the spec block and the canvas, and 0 for a plain
     autoregressive model — so the ring stride of every existing model is byte-unchanged.
"""

from __future__ import annotations

import glob
import sys
import types

import torch

MODEL_ID = "cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4"
AR_MODEL_ID = "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:48s} {detail}")
        return ok


def _snapshot(model_id: str):
    m = glob.glob(f"/root/.cache/huggingface/hub/models--{model_id.replace('/', '--')}/snapshots/*/")
    return m[0] if m else None


def main() -> int:
    torch.set_grad_enabled(False)
    from minisgl.distributed import set_tp_info

    set_tp_info(0, 1)
    from minisgl.utils.hf import load_pretrained_config  # AutoConfig + the 5.17 per-layer opt-in
    from minisgl.attention.rdna4 import RDNA4Backend
    from minisgl.engine.engine import _swa_ring_block
    from minisgl.models.config import ModelConfig

    rep = Report()

    path = _snapshot(MODEL_ID)
    if path is None:
        print(f"SKIP: {MODEL_ID} not cached")
        return 0
    mc = ModelConfig.from_hf(load_pretrained_config(path), spec_algorithm="none")

    print(f"[canvas-meta] canvas_length={mc.canvas_length} window={mc.sliding_window} "
          f"is_block_diffusion={mc.is_block_diffusion}")

    # ---- 1. the predicate keys on the CHECKPOINT, not the model name ------------------------
    print("\n[1] the block-diffusion predicate")
    rep.check(
        "diffusion_gemma is block diffusion",
        mc.is_block_diffusion and mc.canvas_length == 256,
        f"canvas_length={mc.canvas_length} (top-level config.json), model_type={mc.model_type!r}",
    )
    ar_path = _snapshot(AR_MODEL_ID)
    if ar_path is None:
        print(f"  SKIP: {AR_MODEL_ID} not cached, cannot check the negative case")
    else:
        ar_mc = ModelConfig.from_hf(load_pretrained_config(ar_path), spec_algorithm="none")
        rep.check(
            "the autoregressive sibling is NOT",
            not ar_mc.is_block_diffusion and ar_mc.canvas_length is None,
            f"canvas_length={ar_mc.canvas_length}, model_type={ar_mc.model_type!r} — the two share "
            f"a backbone, so the predicate has to key on the head's own field",
        )

    # ---- 2. the ring stride ------------------------------------------------------------------
    print("\n[2] ring stride: window + the largest block written before it is read")
    spec4 = types.SimpleNamespace(num_draft=4)
    rep.check(
        "canvas model, no spec -> canvas_length",
        _swa_ring_block(mc, None) == mc.canvas_length,
        f"{_swa_ring_block(mc, None)} == canvas_length {mc.canvas_length}; ring stride becomes "
        f"{mc.sliding_window + _swa_ring_block(mc, None)}",
    )
    rep.check(
        "canvas dominates a smaller spec block",
        _swa_ring_block(mc, spec4) == mc.canvas_length,
        f"max(spec 5, canvas {mc.canvas_length}) = {_swa_ring_block(mc, spec4)}",
    )
    if ar_path is not None:
        rep.check(
            "autoregressive model, no spec -> 0",
            _swa_ring_block(ar_mc, None) == 0,
            f"{_swa_ring_block(ar_mc, None)} — stride stays == window, so every existing SWA "
            f"model's ring sizing is byte-unchanged",
        )
        rep.check(
            "autoregressive model, spec -> num_draft+1",
            _swa_ring_block(ar_mc, spec4) == 5,
            f"{_swa_ring_block(ar_mc, spec4)} == num_draft+1, the pre-existing behaviour",
        )

    # ---- 3. the ring rows --------------------------------------------------------------------
    print("\n[3] canvas ring rows: [Wp window | L canvas], Wp from the CACHED length")
    W, L = mc.sliding_window, mc.canvas_length
    R = W + L
    backend = types.SimpleNamespace(swa_window=W, swa_ring_stride=R)
    build = RDNA4Backend._build_swa_canvas_metadata

    # Two requests at different prefix lengths, on different table rows: one past the window
    # (Wp == W) and one short of it (Wp == cur_len). Heterogeneous prefixes in one batch is the
    # scheduling case that matters, and it is where a rectangular page table can go wrong.
    reqs = [types.SimpleNamespace(table_idx=0), types.SimpleNamespace(table_idx=3)]
    cached = [2000, 100]
    qlens = [L, L]
    out_loc, page_table, ctx_lens = build(backend, reqs, qlens, cached, torch.device("cpu"))

    ok_rows = True
    detail = []
    for i, (c0, t) in enumerate(zip(cached, [r.table_idx for r in reqs])):
        base = t * R
        Wp = min(c0, W)
        want_win = [base + (p % R) for p in range(c0 - Wp, c0)]
        want_canvas = [base + (p % R) for p in range(c0, c0 + L)]
        row = page_table[i].tolist()
        ok_rows &= row[: Wp + L] == want_win + want_canvas
        ok_rows &= int(ctx_lens[i]) == Wp + L
        ok_rows &= all(v == 0 for v in row[Wp + L :])  # pad is the NULL slot, never attended
        detail.append(f"req{i}(cur_len={c0}): Wp={Wp} ctx={int(ctx_lens[i])} row_len={len(row)}")
    rep.check("rows are [window | canvas], ctx = Wp + L", ok_rows, "; ".join(detail))

    rep.check(
        "out_loc is exactly the canvas slots",
        out_loc.tolist()
        == [r.table_idx * R + (p % R) for r, c0 in zip(reqs, cached) for p in range(c0, c0 + L)],
        f"{out_loc.numel()} slots for {len(reqs)} x {L} canvas positions — the KV scatter target, "
        f"rewritten in place on every denoising step",
    )

    # The disjointness the whole widening exists for.
    for i, c0 in enumerate(cached):
        Wp = min(c0, W)
        row = page_table[i].tolist()
        win, canvas = set(row[:Wp]), set(row[Wp : Wp + L])
        rep.check(
            f"req{i}: window and canvas slots are disjoint",
            not (win & canvas),
            f"{len(win & canvas)} collisions at stride R={R} (Wp={Wp}, L={L})",
        )

    # And the counter-case: at stride == W the row would alias itself. Proven, not asserted, because
    # this is the exact bug the widening prevents and it is invisible in the output otherwise.
    narrow = types.SimpleNamespace(swa_window=W, swa_ring_stride=W)
    try:
        build(narrow, reqs[:1], [L], [2000], torch.device("cpu"))
        raised = False
    except AssertionError:
        raised = True
    collisions = len(
        {p % W for p in range(2000 - W, 2000)} & {p % W for p in range(2000, 2000 + L)}
    )
    rep.check(
        "a window-sized ring is REFUSED, not silently aliased",
        raised,
        f"stride == window would collide on {collisions} of {L} canvas slots; the builder asserts "
        f"rather than producing a row that attends its own overwritten prefix",
    )

    print(f"\n{'PASS' if rep.failures == 0 else f'FAIL ({rep.failures} checks)'}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
