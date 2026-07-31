"""Adaptive spec-decode VERIFY width.

A speculative step verifies ``width+1`` query rows per sequence: the confirmed token plus `width`
drafted tokens. Every emitted token still comes out of the TARGET's distribution at its own query
row, so the width is a pure COST knob — it decides how many speculative rows we pay for, never what
comes out. Shrinking it when acceptance is poor stops paying for rows that get thrown away;
growing it when acceptance saturates recovers the throughput. That is the whole idea, and the two
invariants that make it safe are:

  1. LOSSLESS. Acceptance statistics choose the WIDTH; they never choose the OUTPUT. Truncating a
     req's drafts from K to W just deletes trailing speculative rows — the remaining rows are
     verified exactly as before and the bonus token at the end of the accepted run is still the
     target's own argmax/sample. `Scheduler._spec_decode_step` slices the target logits by each
     req's REAL draft length, so a shorter list is simply a shorter verify.
  2. IT MUST LAND ON A CAPTURED GRAPH. `GraphRunner.can_use_verify_graph` requires every req in the
     step to share one `extend_len`, and that `extend_len` to be a CAPTURED qlen. A width that is
     not captured falls off the graph onto the eager forward, which costs far more than the rows it
     saves. So the controller only ever returns a member of the captured ladder — see
     `verify_width_ladder`, which is the single place the captured set is defined and is handed
     verbatim to `GraphRunner.capture_verify_graphs`.

THE M<=16 CLIFF (why `MAX_VERIFY_ROWS` exists)
----------------------------------------------
The verify forward is FLAT over tokens: the model sees one ``[M, hidden]`` activation where
``M = padded_bs * (width + 1)`` (`engine/graph.py` `VerifyCaptureBuffer.total`). Several kernels
change FAMILY at M=16 and get much slower above it:

  * `python/minisgl/layers/minv.py:68`       ``_DECODE_GEMV_MAXM = 16`` — bf16/fp16 dense linears
                                             leave `dense_bf16_gemv` for the tiled `dense_gemm_rd`.
  * `python/minisgl/layers/embedding.py:21`  ``_LMHEAD_GEMV_MMAX = 16`` — the LM head (full vocab,
                                             scored for EVERY verify row) leaves the GEMV.
  * `python/minisgl/quant/method.py:28`      ``x.shape[0] <= 16`` — the fused `gate_up + silu_and_mul`
                                             stops firing; an extra launch + a [M, 2*inter] round-trip.
  * `python/minisgl/quant/kernels.py:1227`   ``_W4A8_GEMV_MAX_E2M1 = 16`` (and :1226 int4 = 8) —
                                             quantized dense leaves `decode_gemv` for `prefill_wmma`.

Measured on Laguna: K=16 (qlen 17) -> K=15 (qlen 16) is +20% end-to-end at identical accept-len
(docs/CONTINUANCE_laguna_spec_and_decode_perf.md §2 finding 1). So the ladder is capped so that
``width + 1 <= 16``.

Be honest about the scope of that clamp: because M is ``padded_bs * (width+1)``, the cliff is only
REACHABLE at padded_bs == 1. At bs=2 even width=7 is M=16, and at bs=8 any width >= 1 is already
past it. Clamping the width by ``16 // padded_bs`` would drive the width to 1 at concurrency and
destroy acceptance for a boundary you cannot get back under anyway. So the cap here is a hard
``width <= MAX_VERIFY_ROWS - 1`` (which is exactly right at bs=1 and harmless above it), and the
width at bs>1 is chosen by acceptance alone.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

# Max query rows per sequence in a verify forward. At padded_bs=1 this is the flat M the decode
# GEMV / MoE / fused-SwiGLU kernels see, and every one of them changes family above 16 — see the
# file docstring for the four file:line thresholds.
MAX_VERIFY_ROWS = 16

# How many widths to capture. Every extra width is one more CUDA graph per verify batch size (the
# static I/O buffers are shared — see VerifyCaptureBuffer.view — so the cost is graph-pool memory
# and boot-time capture, not I/O buffers). Three gives a useful dynamic range (e.g. 3/7/15) without
# tripling the verify graph count.
_LADDER_LEN = 3

# Never capture a width below this: at width 1 a step emits at most 2 tokens and the drafter cost is
# no longer amortized — if acceptance is that bad, spec-decode itself is the wrong config.
_LADDER_MIN = 2


def verify_width_ladder(num_draft: int) -> List[int]:
    """The captured verify widths for a run with ``--spec-num-draft num_draft``.

    Ascending, always contains the (clamped) max so a fully-accepting step never has to give up
    rows it would have used. Halving ladder: K=15 -> [3, 7, 15]; K=6 -> [3, 6]; K=4 -> [2, 4];
    K<=2 -> [K] (a single width, i.e. the pre-adaptive behaviour).
    """
    w_max = min(int(num_draft), MAX_VERIFY_ROWS - 1)
    if w_max < 1:
        return []
    ladder = [w_max]
    w = w_max
    while len(ladder) < _LADDER_LEN and w // 2 >= _LADDER_MIN:
        w //= 2
        ladder.append(w)
    return sorted(set(ladder))


class AdaptiveVerifyWidth:
    """Per-request acceptance tracker that sizes the next verify block.

    State is a per-uid EMA of "how many drafts did this request get accepted last step". It is
    updated from the RANK0-AUTHORITATIVE accepted counts (post `_bcast_accept_tp`), from the same
    `reqs` list on every TP rank, so `choose` is a deterministic function of replicated state and
    all ranks pick the same width without an extra collective.

    Sizing rule: ``mean(EMA) + 1``, rounded UP to a captured width. The ``+1`` is not a fudge — the
    statistic is CENSORED: a step that verifies W rows and accepts all W tells you the acceptance
    was AT LEAST W, never how much more. Without one exploratory row past the observed mean the
    controller would latch at whatever width it first narrowed to and could never grow back.

    Cold requests (no history) are seeded at the max width, so a fresh serve starts wide and narrows
    into its measured acceptance rather than starting starved.
    """

    __slots__ = ("_widths", "_max", "_ema", "_alpha", "_hist")

    def __init__(self, widths: Sequence[int], alpha: float = 0.25) -> None:
        self._widths: List[int] = sorted(int(w) for w in widths)
        assert self._widths, "AdaptiveVerifyWidth needs at least one captured width"
        self._max = self._widths[-1]
        self._alpha = float(alpha)
        self._ema: Dict[int, float] = {}
        self._hist: Dict[int, int] = {w: 0 for w in self._widths}

    @property
    def widths(self) -> List[int]:
        return list(self._widths)

    @property
    def max_width(self) -> int:
        return self._max

    @property
    def adaptive(self) -> bool:
        return len(self._widths) > 1

    def choose(self, uids: Iterable[int]) -> int:
        """Width for the next verify block, always a member of the captured ladder."""
        if len(self._widths) == 1:
            return self._max
        total = 0.0
        n = 0
        for uid in uids:
            total += self._ema.get(uid, float(self._max))
            n += 1
        if n == 0:
            return self._max
        want = total / n + 1.0  # one exploratory row past the mean (the censoring argument above)
        for w in self._widths:
            if w >= want:
                return w
        return self._max

    def record(self, uids: Sequence[int], accepted: Sequence[int], width: int) -> None:
        """Fold this step's outcome into the per-uid EMA and count the width that was actually run.

        `accepted[i]` must be the RANK0-AUTHORITATIVE accepted-draft count for `uids[i]` (i.e. taken
        after `_bcast_accept_tp`), or the ranks' EMAs — and hence their next chosen widths — drift."""
        a = self._alpha
        for uid, n in zip(uids, accepted):
            prev = self._ema.get(uid)
            self._ema[uid] = float(n) if prev is None else (1.0 - a) * prev + a * float(n)
        self._hist[width] = self._hist.get(width, 0) + 1

    def free(self, uid: int) -> None:
        self._ema.pop(uid, None)

    def hist_str(self) -> str:
        """Chosen-width histogram, for the log line that proves the width is actually adapting (a
        controller that always returns the max is not adaptive and the distribution says so)."""
        tot = sum(self._hist.values()) or 1
        return " ".join(f"{w}:{c}({100.0 * c / tot:.0f}%)" for w, c in sorted(self._hist.items()))


__all__ = ["MAX_VERIFY_ROWS", "AdaptiveVerifyWidth", "verify_width_ladder"]
