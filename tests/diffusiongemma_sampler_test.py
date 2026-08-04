"""The diffusion sampler, replayed against the HuggingFace reference — BIT-EXACT, every step.

CPU-only, no GPU lease, cannot disturb a serve:

    docker run --rm --entrypoint bash -v <worktree>:/wt \
      -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
      minisgl-rdna4:lean -lc 'cd /wt && PYTHONPATH=/wt/python:/opt/kernels \
        python tests/diffusiongemma_sampler_test.py'

The sampler is pure tensor arithmetic driven by a seeded RNG, so there is no excuse for a
tolerance: `minisgl.diffusion` must reproduce `EntropyBoundSampler` +
`LinearTemperatureScheduleLogitsProcessor` + `StableAndConfidentStoppingCriteria` exactly, on the
accepted mask, the sampled canvas, the argmax canvas, the entropies and the early-exit flag, for
every one of the 48 steps. Two of the ports's deliberate divergences are checked as EQUALITIES
rather than assumed:

  * the reference computes the full-vocab entropy TWICE per step and the softmax three times (268
    MiB of fp32 transient each at the shipped 256x262144); `step()` computes each once and shares.
  * the reference's accept-then-renoise is composed into one `where`, which drops the incoming
    canvas entirely — the algebra says `current` cancels, and this test says so too.

Both runs use the GLOBAL RNG from the same seed rather than a shared generator object, because that
is the stronger statement: it requires the two implementations to draw the same random numbers in
the same ORDER (multinomial, then renoise), not merely to agree given identical draws.
"""

from __future__ import annotations

import glob
import sys

import torch

MODEL_ID = "cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4"
MODEL_GLOB = f"/root/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}/snapshots/*/"

# Small enough for CPU, large enough that the entropy ordering is a real sort and the acceptance set
# is a nontrivial subset. The shipped values are canvas 256 / vocab 262144 / 48 steps.
CANVAS, VOCAB, STEPS = 32, 512, 48
SEED = 1234


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str) -> bool:
        self.failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name:48s} {detail}")
        return ok


def shipped_config():
    """The knobs the checkpoint actually ships, read from generation_config.json — not retyped."""
    import json

    matches = glob.glob(MODEL_GLOB)
    if not matches:
        return None
    with open(matches[0] + "generation_config.json") as fh:
        return json.load(fh)


def logit_stream(steps: int):
    """A deterministic logits trajectory that MOVES, in both of the ways that matter.

    SHARPNESS sweeps: early steps are near-uniform (high entropy, so the entropy bound accepts a
    couple of tokens) and late steps are peaked hard enough that mean entropy drops under the 0.005
    confidence threshold, so the early exit actually fires inside the replay. A fixed random tensor
    would exercise one regime of the bound and never the stopping criterion.

    JITTER re-orders: a stream that only rescales one base tensor keeps the entropy ORDER fixed, so
    the acceptance set is nested and the port's non-monotone `where` composition is never
    distinguished from a monotone commit. Per-step jitter breaks that."""
    gen = torch.Generator().manual_seed(99)
    base = torch.randn(CANVAS, VOCAB, generator=gen)
    out = []
    for n in range(steps, 0, -1):
        frac = 1.0 - n / steps
        sharpness = 0.2 + 900.0 * frac**3
        jitter = torch.randn(CANVAS, VOCAB, generator=gen) * (0.6 * (1.0 - frac))
        out.append(base * sharpness + jitter)
    return out


def main() -> int:
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float32)

    from transformers.models.diffusion_gemma.generation_diffusion_gemma import (
        EntropyBoundSampler,
        EntropyBoundSamplerConfig,
        LinearTemperatureScheduleLogitsProcessor,
        StableAndConfidentStoppingCriteria,
    )

    from minisgl.diffusion import (
        CanvasState,
        DiffusionSamplerConfig,
        categorical_entropy,
        normalized_probs,
    )

    rep = Report()
    shipped = shipped_config()
    if shipped is None:
        print(f"SKIP: {MODEL_ID} not cached (needed for the shipped sampler knobs)")
        return 0

    cfg = DiffusionSamplerConfig(
        canvas_length=CANVAS,
        vocab_size=VOCAB,
        max_denoising_steps=int(shipped["max_denoising_steps"]),
        t_min=float(shipped["t_min"]),
        t_max=float(shipped["t_max"]),
        entropy_bound=float(shipped["sampler_config"]["entropy_bound"]),
        confidence_threshold=float(shipped["confidence_threshold"]),
        stability_threshold=int(shipped["stability_threshold"]),
    )
    print(
        f"[sampler] shipped knobs: steps={cfg.max_denoising_steps} t={cfg.t_max}->{cfg.t_min} "
        f"entropy_bound={cfg.entropy_bound} confidence={cfg.confidence_threshold} "
        f"stability={cfg.stability_threshold}  (canvas={CANVAS} vocab={VOCAB})"
    )

    print("\n[1] the temperature schedule counts DOWN")
    temps = [cfg.temperature(n) for n in (cfg.max_denoising_steps, 1)]
    ref_proc = LinearTemperatureScheduleLogitsProcessor(cfg.t_min, cfg.t_max, cfg.max_denoising_steps)
    probe = torch.ones(1, 1, 4)
    ref_temps = [
        1.0 / ref_proc(None, probe, cur_step=n)[0, 0, 0].item()
        for n in (cfg.max_denoising_steps, 1)
    ]
    rep.check(
        "temperature(cur_step) matches the reference",
        max(abs(a - b) for a, b in zip(temps, ref_temps)) < 1e-6 and temps[0] > temps[1],
        f"step {cfg.max_denoising_steps} -> {temps[0]:.6f} (ref {ref_temps[0]:.6f}), "
        f"step 1 -> {temps[1]:.6f} (ref {ref_temps[1]:.6f}); an up-counter would invert the anneal",
    )

    print("\n[2] entropy is bit-identical to torch.distributions.Categorical")
    gen = torch.Generator().manual_seed(5)
    probe = torch.randn(CANVAS, VOCAB, generator=gen) * 3.0
    mine = categorical_entropy(*normalized_probs(probe))
    want = torch.distributions.Categorical(logits=probe).entropy()
    rep.check(
        "categorical_entropy == Categorical.entropy()",
        torch.equal(mine, want),
        f"max|abs|={(mine - want).abs().max().item():.3e} (must be exactly 0) — the entropies drive "
        f"a SORT, so 1 ULP can flip the acceptance set",
    )

    print(f"\n[3] {STEPS}-step replay against the reference sampler, bit-exact at every step")
    stream = logit_stream(STEPS)

    # ---- reference run -------------------------------------------------------------------
    torch.manual_seed(SEED)
    ref_sampler = EntropyBoundSampler(
        EntropyBoundSamplerConfig(entropy_bound=cfg.entropy_bound),
        canvas_length=CANVAS,
        vocab_size=VOCAB,
        max_denoising_steps=cfg.max_denoising_steps,
    )
    ref_stop = StableAndConfidentStoppingCriteria(
        stability_threshold=cfg.stability_threshold, confidence_threshold=cfg.confidence_threshold
    )
    ref_canvas = ref_sampler.initialize_canvas(1, torch.device("cpu"))
    ref_trace = []
    for i, raw in enumerate(stream):
        cur_step = cfg.max_denoising_steps - i
        scaled = ref_proc(None, raw.unsqueeze(0), cur_step=cur_step)
        probs = torch.softmax(scaled, dim=-1, dtype=torch.float32)
        sampled = torch.multinomial(probs.view(-1, VOCAB), num_samples=1).view(1, CANVAS)
        argmax = torch.argmax(scaled, dim=-1)
        accepted_canvas = ref_sampler.accept_canvas(ref_canvas, sampled, scaled, cur_step)
        mask = ref_sampler.accepted_token_mask.clone()
        next_canvas = ref_sampler.renoise_canvas(accepted_canvas, cur_step)
        done = bool(ref_stop(argmax, scaled)[0])
        ref_trace.append(
            dict(
                accepted=mask[0].clone(),
                two_stage=accepted_canvas[0].clone(),
                canvas=next_canvas[0].clone(),
                argmax=argmax[0].clone(),
                entropy=torch.distributions.Categorical(logits=scaled).entropy()[0].clone(),
                done=done,
            )
        )
        ref_canvas = next_canvas
        if done:
            break

    # ---- port run ------------------------------------------------------------------------
    torch.manual_seed(SEED)
    state = CanvasState(cfg, device=torch.device("cpu"))
    bad = []
    accept_counts = []
    for i, ref in enumerate(ref_trace):
        got = state.step(stream[i])
        accept_counts.append(int(got.accepted.sum()))
        for field in ("accepted", "canvas", "argmax", "entropy"):
            if not torch.equal(getattr(got, field), ref[field]):
                bad.append(f"step {i} {field}")
        if got.done != ref["done"]:
            bad.append(f"step {i} done ({got.done} vs {ref['done']})")
        # The composed accept-then-renoise: the incoming canvas must cancel out.
        composed = torch.where(got.accepted, ref["two_stage"], ref["canvas"])
        if not torch.equal(composed, ref["canvas"]):
            bad.append(f"step {i} composition")

    rep.check(
        f"{len(ref_trace)} steps replay bit-exactly",
        not bad,
        f"mismatches={len(bad)}" + (f": {bad[:6]}" if bad else "")
        + f"; accepted per step {accept_counts[:4]}...{accept_counts[-4:]} of {CANVAS}",
    )
    rep.check(
        "at least one token is accepted every step",
        min(accept_counts) >= 1,
        f"min={min(accept_counts)} max={max(accept_counts)} of {CANVAS} — the first sorted element "
        f"always satisfies sum-max <= bound, so the loop cannot stall",
    )
    # Acceptance is not monotone: a position accepted at step n can be dropped (and re-noised) at
    # n+1. If this trajectory never dropped one, the replay would not distinguish the port's
    # composed `where` from a monotone accumulating commit, and the check above would be weak.
    dropped = [
        int((ref_trace[i]["accepted"] & ~ref_trace[i + 1]["accepted"]).sum())
        for i in range(len(ref_trace) - 1)
    ]
    rep.check(
        "acceptance is NOT monotone (positions get dropped)",
        sum(dropped) > 0,
        f"accepted-then-dropped positions: {sum(dropped)} over {len(dropped)} transitions "
        f"(first few: {dropped[:6]}) — a monotone commit would be all zeros, and the replay would "
        f"then not distinguish the two",
    )
    rep.check(
        "the early exit fired within the replay",
        ref_trace[-1]["done"],
        f"ran {len(ref_trace)} of {STEPS} steps; last mean entropy="
        f"{ref_trace[-1]['entropy'].mean().item():.3e} vs threshold {cfg.confidence_threshold}",
    )

    print("\n[4] the first step can never be 'stable'")
    torch.manual_seed(SEED)
    fresh = CanvasState(cfg, device=torch.device("cpu"))
    peaked = torch.zeros(CANVAS, VOCAB)
    peaked[:, 0] = 60.0  # entropy ~0, so 'confident' is satisfied outright
    first = fresh.step(peaked)
    second = fresh.step(peaked)
    rep.check(
        "step 1 not done, step 2 done, on identical peaked logits",
        (not first.done) and second.done,
        f"step1 done={first.done} (mean entropy {first.mean_entropy:.2e}), step2 done={second.done}"
        f" — the history seeds empty, exactly as the reference seeds it with -1",
    )

    print(f"\n{'PASS' if rep.failures == 0 else f'FAIL ({rep.failures} checks)'}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
