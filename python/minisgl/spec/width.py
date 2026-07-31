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

THE DECODE-KERNEL M CLIFF (why `max_verify_rows` exists)
--------------------------------------------------------
The verify forward is FLAT over tokens: the model sees one ``[M, hidden]`` activation where
``M = padded_bs * (width + 1)`` (`engine/graph.py` `VerifyCaptureBuffer.total`). Several kernels
change FAMILY at an M threshold and get much slower above it. The LIVE constants are IMPORTED from
the modules that own them (`live_m_thresholds`); the values written below are DOCUMENTATION and a
torch-free fallback, and are cross-checked against the live ones at boot — so if a kernel module
moves its threshold, the serve says so out loud instead of this file capping at a stale number:

  * `layers/minv.py`       ``_DECODE_GEMV_MAXM``      bf16/fp16 dense linears leave
                                                      `dense_bf16_gemv` for the tiled `dense_gemm_rd`.
  * `layers/embedding.py`  ``_LMHEAD_GEMV_MMAX``      the LM head (full vocab, scored for EVERY
                                                      verify row) leaves the GEMV.
  * `quant/kernels.py`     ``_W4A8_GEMV_MAX_E2M1``    e2m1/MXFP4/NVFP4 quantized dense leaves
                                                      `decode_gemv` for `prefill_wmma`.
  * `quant/kernels.py`     ``_W4A8_GEMV_MAX_INT4``    the SAME switch for int4 W4A8 — but at 8,
                                                      not 16. See below; this one is model-dependent,
                                                      which is why the cap is a FUNCTION of the
                                                      checkpoint's quant config and not a constant.
  * `quant/method.py`      the fused `gate_up + silu_and_mul` shape gate is also 16; it is a
                           hard-coded literal in a local predicate rather than a named constant, so
                           it is covered by the `_DECODE_GEMV_MAXM`/`_LMHEAD_GEMV_MMAX` = 16 cap
                           rather than imported.

Measured on Laguna (e2m1): K=16 (qlen 17) -> K=15 (qlen 16) is +20% end-to-end at identical
accept-len (docs/CONTINUANCE_laguna_spec_and_decode_perf.md §2 finding 1).

Measured for int4 (tools/w4a8_int4_m_crossover.py, min-of-5x50 on this box, gfx1201): crossing
``_W4A8_GEMV_MAX_INT4 = 8`` is a genuine CLIFF, not a crossover. Timing BOTH kernels at the SAME M
on four dense shapes, `decode_gemv` is faster than `prefill_wmma` at every M from 1 to 16 on three
of them (gemv/wmma 0.25 -> 0.69 at N=K=4096; 0.16 -> 0.41 at N=2048 K=6144), and what the
dispatcher actually costs going from M=8 to M=16 is 1.50x / 1.55x / 2.03x / 3.68x — i.e. worse than
linear in rows on three of four shapes, so past M=8 an int4 model pays MORE per verify row. (The
kernels.py comment claiming "its WMMA tile reclaims M=16 on a dense model" is not what these shapes
measure; raising `_W4A8_GEMV_MAX_INT4` itself would be the better fix, but that changes the kernel —
and hence the numerics — for ordinary int4 decode at M=9..16 too, so it is a separate change.)
So an int4 W4A8 checkpoint gets ``width + 1 <= 8``.

Be honest about the scope of the clamp: because M is ``padded_bs * (width+1)``, the cliff is only
REACHABLE at padded_bs == 1. At bs=2 even width=7 is M=16, and at bs=8 any width >= 1 is already
past it. Clamping the width by ``cap // padded_bs`` would drive the width to 1 at concurrency and
destroy acceptance for a boundary you cannot get back under anyway. So the cap is a hard
``width <= max_verify_rows(quant) - 1`` (exactly right at bs=1, harmless above it), and the width at
bs>1 is chosen by acceptance alone.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

# The ceiling for a model with no quantized dense linears. Also the DOCUMENTED value of the two
# unconditional thresholds; `live_m_thresholds` checks the real ones against these, so if either
# kernel module moves, that shows up as a WARNING rather than as this file being silently stale.
MAX_VERIFY_ROWS = 16

# Documented values of the per-family quantized-dense thresholds. Same contract: the live constants
# win, these exist so the cap is still right when the kernel modules cannot be imported (this module
# is deliberately torch-free so tools/verify_width_unit.py can gate it on the host, with no card).
_M_THRESHOLD_DOC = {"generic": MAX_VERIFY_ROWS, "int4": 8, "e2m1": MAX_VERIFY_ROWS}


def live_m_thresholds() -> "tuple[dict, list]":
    """The kernels' ACTUAL M thresholds, plus any that disagree with `_M_THRESHOLD_DOC`.

    Split out so a serve can report the disagreement once at boot instead of this file quietly
    capping at a stale number."""
    live = dict(_M_THRESHOLD_DOC)
    drift = []
    try:
        from minisgl.layers.embedding import _LMHEAD_GEMV_MMAX
        from minisgl.layers.minv import _DECODE_GEMV_MAXM
        from minisgl.quant.kernels import _W4A8_GEMV_MAX_E2M1, _W4A8_GEMV_MAX_INT4

        live["generic"] = min(int(_DECODE_GEMV_MAXM), int(_LMHEAD_GEMV_MMAX))
        live["int4"] = min(live["generic"], int(_W4A8_GEMV_MAX_INT4))
        live["e2m1"] = min(live["generic"], int(_W4A8_GEMV_MAX_E2M1))
    except Exception as e:                                                  # noqa: BLE001
        return live, [f"kernel M thresholds unreadable ({type(e).__name__}: {e}) — using documented"]
    drift = [f"{k}: doc {v} vs live {live[k]}" for k, v in _M_THRESHOLD_DOC.items() if live[k] != v]
    return live, drift


def max_verify_rows(quant: "Optional[object]" = None) -> int:
    """Max query rows per sequence in one verify forward, for THIS model.

    Taken from the kernel modules' own thresholds rather than restated here, because a constant
    copied out of `layers/minv.py` drifts the day that file changes. `quant` is the served
    checkpoint's `QuantConfig` (``ModelConfig.quant``) or None.
    """
    live, _ = live_m_thresholds()
    if quant is None:
        return max(2, live["generic"])
    if getattr(quant, "is_int4", False):
        # int4 W4A8: the dense GEMV/WMMA switch is at 8, and crossing it MEASURES worse per verify
        # row (module docstring). This is the one family whose ceiling is not 16.
        return max(2, live["int4"])
    if getattr(quant, "weight_is_e2m1", False) or getattr(quant, "is_nvfp4", False):
        return max(2, live["e2m1"])
    return max(2, live["generic"])


# How many widths to capture. Every extra width is one more CUDA graph per verify batch size (the
# static I/O buffers are shared — see VerifyCaptureBuffer.view — so the cost is graph-pool memory
# and boot-time capture, not I/O buffers). Three gives a useful dynamic range (e.g. 3/7/15) without
# tripling the verify graph count.
_LADDER_LEN = 3

# Never capture a width below this: at width 1 a step emits at most 2 tokens and the drafter cost is
# no longer amortized — if acceptance is that bad, spec-decode itself is the wrong config.
_LADDER_MIN = 2


def verify_width_ladder(num_draft: int, quant: "Optional[object]" = None) -> List[int]:
    """The captured verify widths for a run with ``--spec-num-draft num_draft``.

    Ascending, always contains the (clamped) max so a fully-accepting step never has to give up
    rows it would have used. Halving ladder: K=15 -> [3, 7, 15]; K=8 -> [2, 4, 8];
    K<=2 -> [K] (a single width, i.e. the pre-adaptive behaviour).

    At small K halving runs out of room before the ladder is full (K=4 -> [2, 4]), and a two-rung
    ladder that coarse is not usefully adaptive: MTP's measured acceptance is ~2.2, `mean+1` wants 3,
    and 3 rounds up to 4 — so the controller would report 100% at the max and never narrow. When
    there is a spare rung AND a gap of at least 2 to fill, put a rung in the middle: K=4 -> [2, 3, 4],
    K=6 -> [3, 4, 6]. Resolution near the operating point is the whole point of the ladder.

    ``quant`` is the served checkpoint's QuantConfig; it decides the M ceiling (int4 W4A8 caps the
    ladder at 7, everything else at 15 — see `max_verify_rows`). Note this is a NO-OP on every
    config this repo ships: Qwen3.6-35B-AWQ runs MTP at K=4 and GLM-4.7-Flash-AWQ runs EAGLE3 at
    K=6, both already under 7, and Laguna is NVFP4 (e2m1, ceiling 16). It only bites a hand-set
    ``--spec-num-draft > 7`` on an int4 checkpoint.
    """
    w_max = min(int(num_draft), max_verify_rows(quant) - 1)
    if w_max < 1:
        return []
    ladder = [w_max]
    w = w_max
    while len(ladder) < _LADDER_LEN and w // 2 >= _LADDER_MIN:
        w //= 2
        ladder.append(w)
    ladder.sort()
    while len(ladder) < _LADDER_LEN:
        # widen the largest gap by one rung; stop when no gap is big enough to split
        gaps = [(ladder[i + 1] - ladder[i], i) for i in range(len(ladder) - 1)]
        if not gaps or max(gaps)[0] < 2:
            break
        _, i = max(gaps)
        ladder.insert(i + 1, (ladder[i] + ladder[i + 1]) // 2)
    return sorted(set(ladder))


def pad_to_captured_width(
    drafts: "List[List[int]]", widths: Sequence[int], adaptive: bool
) -> "tuple[List[List[int]], List[List[int]], int, bool]":
    """Decide the verify block layout for one step: ``(drafts, staged, width, pad_active)``.

    Pure and host-only, so it is unit-testable without a GPU — which matters, because the invariant
    it enforces is otherwise only reachable in a DP+EP serve.

    `can_use_verify_graph` needs every req to stage the SAME number of query rows, and that count to
    be a CAPTURED width; a ragged step falls to the eager verify. So this pads each req's drafts up
    to one captured width with filler zeros.

    THE TARGET IS THE SMALLEST CAPTURED WIDTH >= max draft length, NOT ``spec.num_draft``. Padding
    back up to num_draft would silently UNDO the adaptive narrowing on every step — the step would
    pay full width while the controller reported a narrow one. Under DP+EP (`adaptive=False`) the
    width must instead be a CONSTANT (the replicas hold different reqs, so a per-step `max(lens)`
    can differ and the in-graph MoE all_gather would see mismatched shapes), so it pins the widest.

    THE INVARIANT: ``len(staged[i]) >= len(drafts[i])`` for every i. The accept loop walks the flat
    verify output with ``offset += len(staged)+1`` but SLICES ``len(drafts)+1`` rows, so a req staged
    NARROWER than it drafted reads into its neighbour's rows (and the last req trips
    `verify_greedy`'s ``len(target) == K+1`` assert, killing the scheduler). The adaptive path cannot
    violate it — the controller already truncated `drafts` to a captured width — but the DP+EP path
    pins ``widths[-1]``, which the M-cliff cap can make NARROWER than num_draft (K=16 -> 15). So the
    truncation happens HERE, for both paths, and `drafts` is returned alongside `staged` because the
    caller must use the truncated list for accept. Truncating (rather than widening the target past
    the captured ladder) keeps the step on a captured graph, and is lossless: dropping trailing
    drafts only removes speculative rows.
    """
    if not widths:
        return drafts, drafts, max((len(d) for d in drafts), default=0), False
    lens = [len(d) for d in drafts]
    w = widths[-1] if not adaptive else next(
        (x for x in widths if x >= max(lens, default=0)), widths[-1])
    if any(L > w for L in lens):
        drafts = [d[:w] for d in drafts]
        lens = [len(d) for d in drafts]
    if all(L == w for L in lens):
        return drafts, drafts, w, False
    staged = [d + [0] * (w - len(d)) for d in drafts]
    # Belt and braces on the invariant above: it is guaranteed by the truncation, and it is also the
    # one thing here whose violation is silent (a request reading its neighbour's logits) rather than
    # loud. One pass over <= max_running_req short lists.
    assert all(len(s) >= len(d) for s, d in zip(staged, drafts)), (
        f"staged verify rows narrower than the real drafts: {[len(s) for s in staged]} vs "
        f"{[len(d) for d in drafts]} (width {w})")
    return drafts, staged, w, True


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


__all__ = ["MAX_VERIFY_ROWS", "AdaptiveVerifyWidth", "live_m_thresholds", "max_verify_rows",
           "pad_to_captured_width", "verify_width_ladder"]
