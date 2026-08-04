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
third time for the multinomial — and a FOURTH time for the self-conditioning soft embedding, which
in this engine is built at the producing step (`DiffusionGemmaForBlockDiffusion.soft_embedding`). At
the shipped canvas_length 256 and vocab 262144 each of those is a 268 MiB fp32 transient. `step()`
computes the softmax and the entropy ONCE and shares all four consumers off it (`DiffusionStep.probs`
carries it to the last one). That is arithmetically identical, and the test replays it against the
reference to prove so.
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


# ==================================================================================================
# The VOCAB-PARALLEL canvas tail.
#
# Under TP the LM head produces `[canvas, vocab/tp]` per rank and the engine used to all_gather it
# into a full `[256, 262144]` on both ranks purely so this sampler could run unchanged. Every one of
# this sampler's consumers is a REDUCTION over the vocabulary — logsumexp, softmax, entropy, argmax,
# multinomial, and the soft embedding's `probs @ E` — and a reduction decomposes over disjoint column
# blocks into a per-rank partial plus one `[canvas]`-sized message. So the gather bought nothing and
# cost 67 MB/rank plus a full-tensor `permute().contiguous()` every denoising step.
#
# Four collectives per request per step, each `[k, canvas]` (a few KiB), replace it:
#   A  gather (row max, uniform deviate)          -> the global row max, and rank 0's deviate
#   B  all_reduce (sum exp(x - max))              -> the global logsumexp
#   C  gather (entropy partial, argmax value, probability mass)
#   D  gather (argmax index, renoise ids, this rank's inverse-CDF hit)
#
# TWO THINGS HERE ARE NOT PERFORMANCE, THEY ARE CORRECTNESS, and they were broken before this:
#
#   * THE MULTINOMIAL AND THE RENOISE MUST BE THE SAME DRAW ON EVERY RANK. The schedulers are
#     `mp.set_start_method("spawn")` processes, so their default torch generators are seeded
#     non-deterministically and INDEPENDENTLY; an unseeded canvas request therefore drew a different
#     `sampled` and a different `_noise()` on each rank. Both feed the next forward's `input_ids`, and
#     the vocab-parallel embedding all-reduces a masked gather — so rank 0 contributed rows for ITS
#     ids and rank 1 for ITS ids, and roughly a quarter of the canvas got the sum of two embeddings
#     and another quarter got zero. It still produced fluent text, which is exactly why nothing
#     caught it. Taking rank 0's deviates off the collective makes the draw lockstep BY CONSTRUCTION
#     rather than by trusting two RNGs to agree, and it costs nothing because the deviates ride in a
#     message that already had to be sent. `SamplingParams.seed` requests already agreed (same seed,
#     same generator) and are byte-unaffected.
#   * THE ARGMAX NEEDS VALUE **AND** INDEX. `torch.argmax` returns the FIRST maximal index, so a
#     cross-shard combine that only takes the max value has no tie-break; this one takes the lowest
#     GLOBAL index among the maximal values, which is the same rule because shards are contiguous
#     and ascending.
# ==================================================================================================


class VocabShard:
    """This rank's slice of the vocabulary, plus the two collectives the sharded tail reduces with.

    `start`/`width` mirror `VocabParallelEmbedding.vocab_range` — the same split, because the same
    weight matrix is the LM head and the embedding table, and `soft_embedding` contracts `probs`
    against exactly these columns."""

    __slots__ = ("size", "rank", "start", "width", "_comm")

    def __init__(self, start: int, width: int) -> None:
        from minisgl.distributed import DistributedCommunicator, get_tp_info

        info = get_tp_info()
        self.size, self.rank = info.size, info.rank
        self.start, self.width = start, width
        self._comm = DistributedCommunicator()

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """`[k, n]` per rank -> `[size, k, n]`. all_gather concatenates on dim 0 rank-major, so the
        view is the inverse of the concatenation and not a reinterpretation of anything."""
        if self.size == 1:
            return x.unsqueeze(0)
        return self._comm.all_gather(x.contiguous()).view(self.size, *x.shape)

    def total(self, x: torch.Tensor) -> torch.Tensor:
        """Partial `[n]` -> summed `[n]`. In-place in the torch backend, so `x` must be a tensor the
        caller owns outright (every call site here passes a freshly-reduced temporary)."""
        return x if self.size == 1 else self._comm.all_reduce(x.contiguous())


def _shard_for(width: int, vocab_size: int) -> VocabShard:
    """Locate a `[canvas, width]` logits block in the global vocabulary, or refuse to guess.

    The split is re-derived here rather than read off the LM head because the sampler has no
    reference to it — but it is the SAME arithmetic as `VocabParallelEmbedding.__init__`, and it is
    checked against the width that actually arrived. A block whose width matches neither the whole
    vocabulary nor this rank's shard would otherwise be reduced against the wrong column offsets:
    every collective would still succeed, every entropy would still be finite, and the sampled token
    ids would be shifted by a constant — fluent text from the wrong rows of the vocabulary."""
    from minisgl.distributed import get_tp_info
    from minisgl.utils import div_ceil

    info = get_tp_info()
    per_rank = div_ceil(vocab_size, info.size)
    start = per_rank * info.rank
    count = max(min(start + per_rank, vocab_size) - start, 0)
    if width != count:
        raise ValueError(
            f"canvas logits are {width} columns wide, but TP rank {info.rank}/{info.size} owns "
            f"{count} of the {vocab_size}-token vocabulary (and the whole vocabulary is "
            f"{vocab_size}). The sampler reduces over vocab columns and must know which ones these "
            f"are; it will not assume."
        )
    return VocabShard(start, count)


_INT64_MAX = torch.iinfo(torch.int64).max
# One-shot evidence, not a gate: the first sharded step of the process reports whether the two ranks'
# RNGs had in fact agreed on the deviates. It is the only place the pre-existing unseeded-request
# divergence is observable from inside the engine, and it costs one comparison of two [canvas] rows,
# once per process. A module flag rather than per-CanvasState because the question is about the
# PROCESS's generators, and asking it once is the whole point.
_lockstep_reported = False


def sharded_canvas_tail(
    local: torch.Tensor,
    shard: "VocabShard",
    vocab_size: int,
    generator: Optional[torch.Generator],
) -> tuple:
    """The whole per-step reduction over a VOCAB-SHARDED `[canvas, width]` logits block.

    Returns `(probs, entropy, argmax, sampled, noise)`, all of which are what the full-vocabulary
    path computes, in the same order, from the same numbers: `probs` is this rank's columns of the
    global softmax (which is exactly what `soft_embedding` contracts), and `entropy`/`argmax`/
    `sampled`/`noise` are `[canvas]` global quantities identical on every rank.

    THE MULTINOMIAL IS AN INVERSE-CDF DRAW, not `torch.multinomial`, because `torch.multinomial`
    cannot see the other rank's columns. One uniform per row (rank 0's, so the draw is lockstep),
    scaled by the GLOBAL mass; each rank subtracts the mass of the ranks below it and searches its
    own cumulative distribution; exactly one rank's target lands inside its own mass and reports the
    hit. That is the same algorithm `torch.multinomial` implements, so the draw is distributionally
    exact — it is not bit-identical to it, and it cannot be. Should floating-point rounding leave a
    row unclaimed (`u * total` landing beyond the summed masses), the row falls back to its argmax,
    which is deterministic and identical on every rank rather than a differently-wrong token each."""
    global _lockstep_reported
    L, W = local.shape
    dev = local.device
    part_max, part_idx = local.max(dim=-1)

    # --- A: the global row max, and the deviates every rank must agree on -----------------------
    u = torch.rand(L, device=dev, dtype=torch.float32, generator=generator)
    noise_local = torch.randint(0, vocab_size, (L,), device=dev, generator=generator)
    ga = shard.gather(torch.stack([part_max, u]))                       # [S, 2, L]
    gmax = ga[:, 0, :].amax(dim=0)
    u = ga[0, 1, :]

    # --- B: the global logsumexp, hence log_probs and the softmax --------------------------------
    # `t` IS the un-normalised softmax numerator: exp(x - gmax). Its global sum is exp(lse - gmax),
    # so dividing by that sum gives the same tensor `torch.softmax` would, with no second reduction.
    t = torch.exp(local - gmax.unsqueeze(-1))
    denom = shard.total(t.sum(dim=-1))
    lse = gmax + torch.log(denom)
    log_probs = local - lse.unsqueeze(-1)
    probs = t.div_(denom.unsqueeze(-1))

    # --- C: entropy, the argmax candidates, and each rank's probability mass ----------------------
    ent_part = -(probs * log_probs).sum(dim=-1)
    del log_probs  # [canvas, width] fp32; the cumsum below wants the room
    mass = probs.sum(dim=-1)
    gc = shard.gather(torch.stack([ent_part, part_max, mass]))          # [S, 3, L]
    entropy = gc[:, 0, :].sum(dim=0)
    vals, masses = gc[:, 1, :], gc[:, 2, :]

    # --- D: the inverse-CDF hit, the global argmax index, the lockstep renoise ---------------------
    offsets = (torch.cumsum(masses, dim=0) - masses)[shard.rank]
    target = u * masses.sum(dim=0) - offsets
    cdf = torch.cumsum(probs, dim=-1)
    pos = torch.searchsorted(cdf, target.unsqueeze(-1).contiguous(), right=True)
    pos = pos.squeeze(-1).clamp_(max=W - 1) + shard.start
    owned = (target >= 0) & (target < masses[shard.rank])
    gd = shard.gather(torch.stack([
        part_idx + shard.start,
        noise_local,
        torch.where(owned, pos, torch.full_like(pos, -1)),
    ]))                                                                  # [S, 3, L]
    idxs, noise, hits = gd[:, 0, :], gd[0, 1, :], gd[:, 2, :]
    best = vals.amax(dim=0, keepdim=True)
    argmax = torch.where(vals == best, idxs, torch.full_like(idxs, _INT64_MAX)).amin(dim=0)
    sampled = hits.amax(dim=0)
    sampled = torch.where(sampled < 0, argmax, sampled)

    if not _lockstep_reported and shard.size > 1:
        _lockstep_reported = True
        from minisgl.utils import init_logger

        agreed = bool(torch.equal(noise, noise_local)) and bool(torch.equal(u, ga[shard.rank, 1, :]))
        init_logger(__name__).info(
            f"[canvas] TP rank {shard.rank}: RNG deviates "
            f"{'AGREED with' if agreed else 'DIVERGED from'} rank 0 on the first sharded step "
            f"(rank 0's are used either way — the draw is lockstep by construction, not by luck)"
        )
    return probs, entropy, argmax, sampled, noise


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
    # The full-vocab fp32 softmax of the temperature-scaled logits — the tensor the entropy, the
    # bound and the multinomial were all computed from, handed on so the NEXT step's self-conditioning
    # soft embedding can be `probs @ E` instead of building a SECOND softmax of the same distribution
    # (which is what `soft_embedding(scaled)` did, and it cost a whole extra full-vocab pass per step).
    # It is a [L, vocab] fp32 tensor — 268 MiB at the shipped 256 x 262144 — so the caller is expected
    # to consume it immediately and drop it, which is the whole point of building the soft embedding
    # at the producing step rather than carrying anything vocab-wide across the boundary.
    probs: "torch.Tensor | None" = None


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
        # Built on the first sharded step, from the width of the logits block actually handed over —
        # see `step`. None means "not sharded", which is both the tp_size==1 case and every caller
        # that hands over the full vocabulary.
        self._shard: "VocabShard | None" = None

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

        `logits` is [canvas_length, V] for THIS request only — the temperature is a function of the
        request's own step index, so a batch cannot share one scale (see the per-request temperature
        note in the block-diffusion spec).

        V is EITHER the whole vocabulary OR this TP rank's vocab shard, and which one it is is read
        off the width rather than passed in. That is deliberate: the served path under TP hands over
        a shard (the LM head does not all_gather — see `layers/embedding.py::logits_local_shard`),
        the parity fixtures and every tp_size==1 serve hand over the whole thing, and a width that is
        neither is a shape the reductions below would silently mis-attribute to the wrong columns, so
        it raises."""
        cfg = self.config
        assert not self.finished, "a finished canvas must not be stepped again"
        scaled = logits / cfg.temperature(self.step_index)
        width = scaled.shape[-1]

        if width != cfg.vocab_size:
            # --- the VOCAB-PARALLEL tail ----------------------------------------------------------
            # Same quantities, same order, reduced across the TP group instead of over a tensor that
            # had to be all_gathered first. See `sharded_canvas_tail` for the four messages and for
            # why the deviates come off the wire rather than out of this rank's generator.
            if self._shard is None:
                self._shard = _shard_for(width, cfg.vocab_size)
            probs, entropy, argmax, sampled, fresh = sharded_canvas_tail(
                scaled, self._shard, cfg.vocab_size, self.generator
            )
        else:
            # --- the WHOLE-VOCABULARY tail, which is the reference-replayed one ---------------------
            # ONE softmax and ONE entropy, shared by the multinomial, the acceptance bound, the
            # stopping criterion AND the next step's self-conditioning soft embedding (the reference
            # computes the entropy twice and the softmax three times, each a 268 MiB fp32 transient at
            # the shipped 256 x 262144; `soft_embedding` used to build a fourth). `probs` leaves on
            # the DiffusionStep for that last consumer — see its field comment.
            #
            # The shared tensor is the NORMALIZED-logit softmax, the one Categorical uses, because the
            # entropy is the consumer that cannot tolerate a rounding difference. The multinomial
            # takes the same tensor; it differs from the reference's raw-logit softmax only by fp32
            # rounding, which is far below the resolution of an inverse-CDF draw — asserted, not
            # assumed, by the 48-step bit-exact replay in tests/diffusiongemma_sampler_test.py.
            log_probs, probs = normalized_probs(scaled)
            entropy = categorical_entropy(log_probs, probs)
            argmax = torch.argmax(scaled, dim=-1)
            # Draw order matters for reproducibility against the reference: the multinomial is drawn
            # BEFORE the renoise, so a shared generator replays identically.
            sampled = torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(-1)
            fresh = self._noise()

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
        # `fresh` was drawn ABOVE, in draw order after the multinomial, because on the sharded path
        # it has to ride the same collective as everything else the ranks must agree on.
        canvas = torch.where(accepted, sampled, fresh)

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
            probs=probs,
        )


__all__ = [
    "CanvasState",
    "DiffusionSamplerConfig",
    "DiffusionStep",
    "VocabShard",
    "categorical_entropy",
    "normalized_probs",
    "sharded_canvas_tail",
]
