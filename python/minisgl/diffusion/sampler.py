"""The entropy-bound diffusion sampler and the per-request canvas state.

This is a port of `DiffusionGemmaGenerationMixin`'s inner denoising loop
(`EntropyBoundSampler`, `LinearTemperatureScheduleLogitsProcessor`,
`StableAndConfidentStoppingCriteria`). Five things about it are counter-intuitive enough that a
reasonable-looking implementation gets them wrong and still produces text:

  1. THERE IS NO MASK TOKEN. The canvas starts as uniform-random ids over the WHOLE vocabulary and
     un-accepted positions are re-randomised every step. This is uniform-state discrete diffusion,
     not the absorbing-mask kind every other diffusion LM uses.
  2. ACCEPTANCE IS NOT MONOTONE, AND THE CURRENT CANVAS IS DISCARDED. The reference computes
     `accepted = where(mask, sampled, current)` and then `next = where(~mask, fresh_noise,
     accepted)`; compose them and `current` cancels out entirely. A token accepted at step n can be
     re-noised at step n+1. Convergence comes from self-conditioning, not from an accumulating
     commit. `step()` implements the composed form, which is why it never reads the incoming canvas.
  3. THE EMITTED TOKENS ARE THE ARGMAX CANVAS, NOT THE SAMPLED ONE. The sampled canvas only ever
     feeds the next forward.
  4. t_max/t_min IS A TEMPERATURE SCHEDULE, NOT A NOISE LEVEL, and `cur_step` counts DOWN. With the
     shipped config it runs 0.8 -> 0.408 over 48 steps.
  5. `confidence_threshold` / `stability_threshold` are an EARLY EXIT, not an acceptance rule. The
     argmax canvas is committed either way.

One deliberate divergence from the reference, for cost: the reference computes the full-vocabulary
entropy TWICE per step (once in `accept_canvas`, once in the stopping criterion) and the softmax a
third time for the multinomial. At the shipped canvas_length 256 and vocab 262144 each of those is a
268 MiB fp32 transient. `step()` computes the softmax and the entropy ONCE and shares them. That is
arithmetically identical, and the test replays it against the reference to prove so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def normalized_probs(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`(log_probs, probs)` exactly as `torch.distributions.Categorical(logits=l)` computes them:
    the logits are shifted by their own logsumexp FIRST, and the probabilities are the softmax of
    THAT, not of the raw logits.

    The shift is mathematically a no-op — softmax already subtracts the row max — but it is not a
    no-op in fp32, and the difference lands in the entropies, which drive a SORT. A 1-ULP
    disagreement there can flip which tokens the entropy bound accepts, so "close enough" is the
    wrong bar and this reproduces the reference's exact op order instead."""
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    return log_probs, torch.softmax(log_probs, dim=-1)


def categorical_entropy(log_probs: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Per-row entropy in nats from `normalized_probs`' outputs. Bit-identical to
    `torch.distributions.Categorical(logits=l).entropy()`."""
    return -(probs * log_probs).sum(dim=-1)


@dataclass(frozen=True)
class DiffusionSamplerConfig:
    """The knobs, all off `generation_config.json` — never a model-name branch."""

    canvas_length: int
    vocab_size: int
    max_denoising_steps: int
    t_min: float
    t_max: float
    entropy_bound: float
    confidence_threshold: float
    stability_threshold: int

    @classmethod
    def from_hf(cls, model_path: str, model_config) -> "DiffusionSamplerConfig":
        """Build from the checkpoint's own `generation_config.json` + `ModelConfig`.

        Every knob is the model author's, never a minisgl default: a wrong `entropy_bound` changes
        how many tokens each step accepts, a wrong `confidence_threshold` changes when a block stops,
        and both produce fluent output, so there is nothing to notice downstream. Missing keys raise
        rather than fall back — a block-diffusion checkpoint shipping no denoising schedule is one
        this engine cannot serve correctly, and saying so at boot is the only honest option."""
        from minisgl.utils.hf import load_generation_config

        gen = load_generation_config(model_path)
        missing = [
            k
            for k in ("max_denoising_steps", "t_min", "t_max", "confidence_threshold",
                      "stability_threshold", "sampler_config")
            if k not in gen
        ]
        if missing or model_config.canvas_length is None:
            raise ValueError(
                f"block-diffusion serving needs the denoising schedule from "
                f"{model_path}/generation_config.json; missing "
                f"{missing or ['canvas_length (top-level config.json)']}. These are model "
                f"properties, not tunables — guessing them yields fluent, wrong text."
            )
        sampler = gen["sampler_config"]
        cls_name = sampler.get("_cls_name")
        if cls_name != "EntropyBoundSamplerConfig":
            raise NotImplementedError(
                f"unsupported diffusion sampler {cls_name!r}; only EntropyBoundSampler is ported"
            )
        return cls(
            canvas_length=int(model_config.canvas_length),
            vocab_size=int(model_config.vocab_size),
            max_denoising_steps=int(gen["max_denoising_steps"]),
            t_min=float(gen["t_min"]),
            t_max=float(gen["t_max"]),
            entropy_bound=float(sampler["entropy_bound"]),
            confidence_threshold=float(gen["confidence_threshold"]),
            stability_threshold=int(gen["stability_threshold"]),
        )

    def temperature(self, cur_step: int) -> float:
        """`cur_step` counts DOWN from max_denoising_steps to 1, so the schedule runs t_max -> ~t_min
        and the LAST step is the coldest. Reading it as an up-counter inverts the anneal, which
        does not crash and produces fluent-but-worse text."""
        return self.t_min + (self.t_max - self.t_min) * (cur_step / self.max_denoising_steps)


@dataclass
class DiffusionStep:
    """One denoising step's outputs. `entropy` and `probs` are returned rather than recomputed by
    the caller because they are the expensive part of the step."""

    canvas: torch.Tensor  # [L] int64 — the ids the NEXT forward consumes
    argmax: torch.Tensor  # [L] int64 — the emit candidate for this step
    accepted: torch.Tensor  # [L] bool — which positions took the sampled token
    entropy: torch.Tensor  # [L] fp32 — per-position entropy of the temperature-scaled logits
    mean_entropy: float
    done: bool  # stable AND confident: the block may stop early
    # The temperature-scaled logits this step consumed — what the NEXT step's self-conditioning soft
    # embedding must be built from (the reference carries `processed_logits`, i.e. post-temperature,
    # not the raw ones). Returned rather than recomputed because it is a [L, vocab] fp32 tensor
    # (268 MiB at the shipped 256 x 262144); the caller is expected to consume it immediately and
    # drop it, which is the whole point of building the soft embedding at the producing step.
    scaled: "torch.Tensor | None" = None


class CanvasState:
    """One request's denoising state for one 256-token block.

    Holds only what must survive a step boundary: the canvas ids, the previous argmax canvases (for
    the stability criterion), the self-conditioning soft embedding, and the step counter. Everything
    else — logits, probabilities, entropies — is a transient of the step that produced it.
    """

    def __init__(
        self,
        config: DiffusionSamplerConfig,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ):
        self.config = config
        self.device = device
        self.generator = generator
        # `step` counts DOWN, matching the reference's `cur_step`; a block ends at 0.
        self.step_index = config.max_denoising_steps
        self.canvas = self._noise()
        self.soft_conditioning: torch.Tensor | None = None
        self.argmax: torch.Tensor | None = None
        self.finished = False
        # Last step's mean entropy — the quantity the confidence criterion tests. Kept so a commit
        # can report how close the block actually got to the threshold rather than only whether it
        # crossed it; a block that always exits on the step CAP is a different story from one that
        # converges, and tok/s cannot tell them apart.
        self.last_mean_entropy = float("nan")
        # The reference seeds the stability history with -1 so the FIRST step can never be "stable"
        # (a fresh canvas would otherwise compare equal to a fresh history and exit at step 1).
        self._history: list[torch.Tensor] = []

    def _noise(self) -> torch.Tensor:
        return torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(self.config.canvas_length,),
            device=self.device,
            generator=self.generator,
        )

    def step(self, logits: torch.Tensor) -> DiffusionStep:
        """Consume this step's RAW (softcapped, fp32) logits and advance the state.

        `logits` is [canvas_length, vocab] for THIS request only — the temperature is a function of
        the request's own step index, so a batch cannot share one scale (see the per-request
        temperature note in the block-diffusion spec)."""
        cfg = self.config
        assert not self.finished, "a finished canvas must not be stepped again"
        scaled = logits / cfg.temperature(self.step_index)

        # ONE softmax and ONE entropy, shared by the multinomial, the acceptance bound and the
        # stopping criterion (the reference computes the entropy twice and the softmax three times,
        # each a 268 MiB fp32 transient at the shipped 256 x 262144).
        #
        # The shared tensor is the NORMALIZED-logit softmax, the one Categorical uses, because the
        # entropy is the consumer that cannot tolerate a rounding difference. The multinomial takes
        # the same tensor; it differs from the reference's raw-logit softmax only by fp32 rounding,
        # which is far below the resolution of an inverse-CDF draw — asserted, not assumed, by the
        # 48-step bit-exact replay in tests/diffusiongemma_sampler_test.py.
        log_probs, probs = normalized_probs(scaled)
        entropy = categorical_entropy(log_probs, probs)
        argmax = torch.argmax(scaled, dim=-1)

        # Draw order matters for reproducibility against the reference: the multinomial is drawn
        # BEFORE the renoise, so a shared generator replays identically.
        sampled = torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(-1)

        # The entropy bound: the longest ascending-entropy prefix whose summed entropy minus its own
        # maximum stays under the bound, i.e. the largest approximately-independent set. The first
        # element always satisfies it (sum == max), so at least one token is accepted every step and
        # the loop cannot stall.
        order = torch.argsort(entropy, dim=-1, descending=False)
        sorted_entropy = entropy[order]
        selected = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy <= cfg.entropy_bound
        accepted = torch.zeros_like(selected)
        accepted.scatter_(dim=-1, index=order, src=selected)

        # The composed accept-then-renoise. The incoming canvas cancels out (see the module note),
        # so it is not read here — every position is either this step's sampled token or new noise.
        canvas = torch.where(accepted, sampled, self._noise())

        stable = False
        if cfg.stability_threshold == 0:
            stable = True
        elif len(self._history) >= cfg.stability_threshold:
            stable = all(bool(torch.equal(h, argmax)) for h in self._history)
        self._history.append(argmax)
        if len(self._history) > cfg.stability_threshold:
            self._history.pop(0)

        mean_entropy = float(entropy.mean())
        done = stable and mean_entropy < cfg.confidence_threshold

        self.canvas = canvas
        self.argmax = argmax
        self.last_mean_entropy = mean_entropy
        self.step_index -= 1
        self.finished = done or self.step_index <= 0
        return DiffusionStep(
            canvas=canvas,
            argmax=argmax,
            accepted=accepted,
            entropy=entropy,
            mean_entropy=mean_entropy,
            done=done,
            scaled=scaled,
        )


__all__ = [
    "CanvasState",
    "DiffusionSamplerConfig",
    "DiffusionStep",
    "categorical_entropy",
    "normalized_probs",
]
