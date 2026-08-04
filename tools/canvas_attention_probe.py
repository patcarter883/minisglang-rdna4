"""Does the paged kernel actually serve a NON-CAUSAL 256-query canvas? Correctness + cost.

Block diffusion's decoder attends 256 canvas queries bidirectionally over `[encoder KV | canvas KV]`
on every denoising step, on both layer geometries. Two of the three pieces this needs are already
shipped and one is not, so before any engine machinery is written this measures, on the card:

  A. SLIDING layers (25 of 30, the ring pool). `_swa_prefill_paged` hardcodes
     `causal=1, sliding_window=W, mask_bias=None` (attention/rdna4.py:590-598) — there is no
     non-causal path on the ring today. But the KERNEL takes both as arguments, so the question is
     whether `causal=0, sliding_window=0` over a `[Wp prefix | canvas]` ring block table with
     `context_len = Wp + canvas` reproduces dense bidirectional attention. If it does, the missing
     path is ~20 lines of Python, not a kernel change.
  B. FULL layers (5 of 30, the main pool). `_hip_prefill_paged` already runs `causal=0`, but ONLY
     when a `custom_mask` is attached (the fused-TiDAR seam), which for a canvas would mean
     materializing a [256, cur_len+256] fp32 bias that says nothing. Measured here: `causal=0` with
     `mask_bias=None` is dense bidirectional attention, so the canvas needs no mask at all.
  C. RING ALIASING. With stride R = W the canvas position `cur_len + j` lands on ring slot
     `(cur_len + j) % W`, which is the slot holding prefix position `cur_len + j - W` — inside the
     very window the canvas must read. Counted, then re-counted at R = W + canvas.
  D. COST at q=256, causal=0 vs causal=1 — the open performance question (a non-causal 256-query
     step computes 2x the score cells of the causal half, and this shape has only ever been
     exercised by fused-TiDAR at small query counts).

GEOMETRY IS A PARAMETER, defaulting to DiffusionGemma at TP=2 (what the 17 GB int4 checkpoint has to
run on): sliding 8 QO / 4 KV at head_dim 256, full 8 QO / 1 KV at head_dim 512, window 1024, canvas
256, softmax scale 1.0 (Gemma4 folds the temperature into the learned k_norm — 1/sqrt(d) here would
make every number below meaningless).

Run (needs ONE card; the arbiter injects the device pair):

    gpu-lease -n 1 -- docker run --rm --name canvas_probe \
      --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable \
      --cap-add SYS_PTRACE --ipc host --shm-size 16gb \
      -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
      -v <worktree>:/wt --entrypoint bash minisgl-rdna4:lean \
      -lc 'PYTHONPATH=/wt/python:/opt/kernels python /wt/tools/canvas_attention_probe.py'
"""

from __future__ import annotations

import os
import sys
import time

import torch

DEV = "cuda"

# DiffusionGemma / Gemma4 at TP=2. Override any of these to probe another rank layout.
SWA_HQ = int(os.environ.get("CANVAS_SWA_HQ", 8))
SWA_HK = int(os.environ.get("CANVAS_SWA_HK", 4))
SWA_D = int(os.environ.get("CANVAS_SWA_D", 256))
FULL_HQ = int(os.environ.get("CANVAS_FULL_HQ", 8))
FULL_HK = int(os.environ.get("CANVAS_FULL_HK", 1))
FULL_D = int(os.environ.get("CANVAS_FULL_D", 512))
WINDOW = int(os.environ.get("CANVAS_WINDOW", 1024))
CANVAS = int(os.environ.get("CANVAS_LEN", 256))
PREFIX = int(os.environ.get("CANVAS_PREFIX", 2000))
SCALE = float(os.environ.get("CANVAS_SCALE", 1.0))


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:46s} {detail}")
        return ok


def bidirectional_reference(q, k, v, scale):
    """Dense non-causal attention over ALL keys, fp32, GQA-expanded. q [n,Hq,D], k/v [m,Hk,D]."""
    n, hq, d = q.shape
    rep = hq // k.shape[1]
    kk = k.repeat_interleave(rep, dim=1).float()
    vv = v.repeat_interleave(rep, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), kk) * scale
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.einsum("hqk,khd->qhd", attn, vv)


def rel_error(got, want):
    d = (got.float() - want.float())
    return d.norm().item() / want.float().norm().clamp_min(1e-30).norm().item(), d.abs().max().item()


def paged_pool(num_slots, heads, dim):
    """A page_size=1 paged cache, the layout both minisgl KV pools use."""
    return (
        torch.zeros(num_slots, 1, heads, dim, device=DEV, dtype=torch.bfloat16),
        torch.zeros(num_slots, 1, heads, dim, device=DEV, dtype=torch.bfloat16),
    )


def run(op, q, kc, vc, rows, ctx, causal, sliding_window, mask=None):
    block_table = torch.tensor([rows], dtype=torch.int32, device=DEV)
    cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=DEV)
    ctx_lens = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    args = [q.contiguous(), kc, vc, block_table, cu_q, ctx_lens,
            SCALE, causal, sliding_window, q.shape[0], 0]
    if mask is not None:
        args.append(mask)
    return op(*args)


def main() -> int:
    torch.manual_seed(0)
    import attn_prefill_paged

    op = attn_prefill_paged.flash_prefill_paged
    # The mask_bias arg is a NEWER addition than some built .so's carry, and the package exposes a
    # plain callable rather than a torch.ops entry, so detect by trial rather than by schema string:
    # a stale kernel package would otherwise fail deep inside check [B] with an arity error.
    schema = str(getattr(op, "_schema", "") or getattr(op, "__doc__", "") or "<opaque callable>")
    probe_q = torch.zeros(1, SWA_HQ, SWA_D, device=DEV, dtype=torch.bfloat16)
    probe_kc, probe_vc = paged_pool(2, SWA_HK, SWA_D)
    try:
        run(op, probe_q, probe_kc, probe_vc, [0], 1, 0, 0,
            mask=torch.zeros(1, 1, dtype=torch.float32, device=DEV))
        has_mask = True
    except (TypeError, RuntimeError):
        has_mask = False
    print(f"[canvas-probe] {torch.cuda.get_device_name(0)}")
    print(f"  kernel: {schema}  mask_bias arg: {has_mask}")
    print(
        f"  geometry: sliding {SWA_HQ}q/{SWA_HK}kv D={SWA_D}, full {FULL_HQ}q/{FULL_HK}kv "
        f"D={FULL_D}, window={WINDOW}, canvas={CANVAS}, prefix={PREFIX}, scale={SCALE}"
    )
    rep = Report()

    # ---------------------------------------------------------------------------------------
    # A. SLIDING layers on the ring pool: [Wp prefix | canvas], causal=0, sliding_window=0
    # ---------------------------------------------------------------------------------------
    print("\n[A] sliding layers — non-causal over a [window | canvas] RING block table")
    ring_stride = WINDOW + CANVAS  # the widening a canvas needs; C below shows why
    n_slots = ring_stride + 8
    kc, vc = paged_pool(n_slots, SWA_HK, SWA_D)
    Wp = min(PREFIX, WINDOW)

    # Ring slots, exactly as _build_swa_metadata / _fill_swa_verify_static compute them:
    # absolute position p -> table_idx*R + p % R, with table_idx 0 here.
    win_pos = list(range(PREFIX - Wp, PREFIX))
    canvas_pos = list(range(PREFIX, PREFIX + CANVAS))
    win_slots = [p % ring_stride for p in win_pos]
    canvas_slots = [p % ring_stride for p in canvas_pos]
    rows = win_slots + canvas_slots

    k_all = torch.randn(len(rows), SWA_HK, SWA_D, device=DEV, dtype=torch.bfloat16)
    v_all = torch.randn(len(rows), SWA_HK, SWA_D, device=DEV, dtype=torch.bfloat16)
    slot_idx = torch.tensor(rows, device=DEV, dtype=torch.long)
    kc[slot_idx, 0] = k_all
    vc[slot_idx, 0] = v_all
    q = torch.randn(CANVAS, SWA_HQ, SWA_D, device=DEV, dtype=torch.bfloat16)

    got = run(op, q, kc, vc, rows, Wp + CANVAS, causal=0, sliding_window=0,
              mask=None if not has_mask else None)
    want = bidirectional_reference(q, k_all, v_all, SCALE)
    rf, ma = rel_error(got, want)
    # bf16 flash vs an fp32 dense reference: the bar is bf16 accumulation noise, not fp32 ULPs.
    rep.check(
        "causal=0 ring extend == dense bidirectional",
        rf < 5e-3,
        f"rel_fro={rf:.3e} max|abs|={ma:.3e} over {Wp + CANVAS} keys x {CANVAS} queries",
    )

    # Sensitivity: the path that EXISTS today (causal=1 + sliding_window=W) is a different answer.
    got_causal = run(op, q, kc, vc, rows, Wp + CANVAS, causal=1, sliding_window=WINDOW)
    rf_c, _ = rel_error(got_causal, want)
    rep.check(
        "the shipped causal=1 ring path is NOT this",
        rf_c > 1e-2,
        f"rel_fro={rf_c:.3e} vs the bidirectional reference — `_swa_prefill_paged` cannot serve a "
        f"canvas as written (attention/rdna4.py:590-598)",
    )
    # And that a canvas query really does see the whole canvas, not just its own past.
    q_probe = q.clone()
    q_probe[0] = q[CANVAS - 1]
    got_probe = run(op, q_probe, kc, vc, rows, Wp + CANVAS, causal=0, sliding_window=0)
    rep.check(
        "query 0 sees the LAST canvas key",
        torch.allclose(got_probe[0].float(), got_probe[CANVAS - 1].float(), atol=2e-2),
        f"identical queries at position 0 and {CANVAS - 1} give the same output "
        f"(max|abs|={(got_probe[0].float() - got_probe[CANVAS - 1].float()).abs().max():.3e}); "
        f"under causal=1 they could not",
    )

    # ---------------------------------------------------------------------------------------
    # B. FULL layers on the main pool: causal=0 with NO mask
    # ---------------------------------------------------------------------------------------
    print("\n[B] full layers — non-causal over the main pool, mask_bias=None")
    total = PREFIX + CANVAS
    kcf, vcf = paged_pool(total + 8, FULL_HK, FULL_D)
    kf = torch.randn(total, FULL_HK, FULL_D, device=DEV, dtype=torch.bfloat16)
    vf = torch.randn(total, FULL_HK, FULL_D, device=DEV, dtype=torch.bfloat16)
    kcf[:total, 0] = kf
    vcf[:total, 0] = vf
    rows_f = list(range(total))
    qf = torch.randn(CANVAS, FULL_HQ, FULL_D, device=DEV, dtype=torch.bfloat16)

    got_f = run(op, qf, kcf, vcf, rows_f, total, causal=0, sliding_window=0)
    want_f = bidirectional_reference(qf, kf, vf, SCALE)
    rf_f, ma_f = rel_error(got_f, want_f)
    rep.check(
        f"head_dim {FULL_D} causal=0 == dense bidirectional",
        rf_f < 5e-3,
        f"rel_fro={rf_f:.3e} max|abs|={ma_f:.3e} over {total} keys x {CANVAS} queries",
    )
    if has_mask:
        zero_mask = torch.zeros(CANVAS, total, dtype=torch.float32, device=DEV)
        got_m = run(op, qf, kcf, vcf, rows_f, total, causal=0, sliding_window=0, mask=zero_mask)
        rep.check(
            "an all-zero mask_bias is redundant",
            torch.equal(got_m, got_f),
            f"max|abs|={(got_m.float() - got_f.float()).abs().max():.3e} — so a canvas needs NO "
            f"[{CANVAS}, {total}] fp32 mask ({CANVAS * total * 4 / 2**20:.1f} MiB/layer/step saved)",
        )
    else:
        print("  (kernel takes no mask_bias arg in this build; skipping the redundancy check)")

    # ---------------------------------------------------------------------------------------
    # C. Ring aliasing: why the stride must widen
    # ---------------------------------------------------------------------------------------
    print("\n[C] ring stride — the canvas must not land on the window it reads")
    for stride, label in ((WINDOW, "R = W (today's serve)"), (WINDOW + CANVAS, "R = W + canvas")):
        w = [p % stride for p in win_pos]
        c = [p % stride for p in canvas_pos]
        collisions = len(set(w) & set(c))
        rep.check(
            f"{label}: collisions",
            (collisions > 0) if stride == WINDOW else (collisions == 0),
            f"{collisions} of {CANVAS} canvas slots alias a window slot"
            + (" — the canvas would overwrite the prefix it must attend to"
               if stride == WINDOW else " (disjoint)"),
        )

    # ---------------------------------------------------------------------------------------
    # D. Cost
    # ---------------------------------------------------------------------------------------
    print("\n[D] cost at q=256 — the open performance question")

    def bench(fn, iters=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # us

    swa_nc = bench(lambda: run(op, q, kc, vc, rows, Wp + CANVAS, 0, 0))
    swa_c = bench(lambda: run(op, q, kc, vc, rows, Wp + CANVAS, 1, WINDOW))
    full_nc = bench(lambda: run(op, qf, kcf, vcf, rows_f, total, 0, 0))
    full_c = bench(lambda: run(op, qf, kcf, vcf, rows_f, total, 1, 0))
    # A denoising step is 25 sliding + 5 full attention calls.
    step_us = 25 * swa_nc + 5 * full_nc
    print(
        f"  sliding ({Wp + CANVAS} keys): causal=0 {swa_nc:8.1f} us   causal=1 {swa_c:8.1f} us   "
        f"ratio {swa_nc / swa_c:.2f}x"
    )
    print(
        f"  full    ({total} keys): causal=0 {full_nc:8.1f} us   causal=1 {full_c:8.1f} us   "
        f"ratio {full_nc / full_c:.2f}x"
    )
    print(
        f"  attention alone per denoising step (25 sliding + 5 full) = {step_us:.0f} us; "
        f"48 steps = {48 * step_us / 1000:.1f} ms for a {CANVAS}-token block "
        f"({48 * step_us / CANVAS:.1f} us/token of attention)"
    )

    print(f"\n{'PASS' if rep.failures == 0 else f'FAIL ({rep.failures} checks)'}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
