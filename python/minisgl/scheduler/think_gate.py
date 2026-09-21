from __future__ import annotations

"""The reasoning ("think") gate: when does a request LEAVE its reasoning span?

A reasoning model opens a chain-of-thought before its answer. Two engine behaviours hang off knowing
when that scratch ENDS:

* **structured output** — a JSON schema masked from token 0 would forbid the reasoning markup, so the
  grammar matcher is held off (row left all-ones) until the reasoning span closes;
* **the β budget backstop** — a model that rambles and never closes is FORCED to emit the close
  delimiter after ``budget`` reasoning tokens, so an answer always gets produced.

Both need one fact — "has this request left its reasoning span yet" — and this module owns it.

WHY THIS IS A TOKEN SEQUENCE AND NOT A TOKEN
--------------------------------------------
It used to be a single token id: every mainstream family spells the close as one added token
(``</think>``), and a multi-token delimiter was reduced to its LAST id with the comment "a rare,
tolerable approximation". It is not tolerable. Muse-Glimmer expresses reasoning as a separate TURN
rather than a span inside one, so its delimiters are multi-token AND share a trailing token::

    reasoning open   " to=self<|message|>"                        [328, 19669, 200023]
    reasoning close  "<|eom|><|start|>assistant to=user<|message|>"[200007, 200022, 140680, 328, 76976, 200023]
    answer header    " to=user<|message|>"                        [328, 76976, 200023]

All three end in ``200023`` (``<|message|>``). Gating on the last id therefore opened the gate on the
THIRD token of the OPENING delimiter — the schema engaged at the first token of the reasoning body,
the model emitted its JSON inside the reasoning turn, the close became unemittable (the grammar
forbade it), and the reasoning parser — seeing an opened span with no close — routed the ENTIRE reply
to ``reasoning_content`` and served ``content: ""``. A complete, silent loss of the answer on every
structured-output request, from a one-token shortcut.

So a gate matches a full token SEQUENCE, and it matches several: a request leaves its reasoning span
either by closing it (the close delimiter) or by never opening one and answering directly (the
model's own answer-turn header). Registering only the close would be worse than the bug it fixes on
the PLAIN lane — ``think_close_delim`` is set for every thinking request, so a direct answer would
never release and the backstop would splice a turn header into the middle of a legitimate reply.

WILDCARD PATTERNS
-----------------
A channel-routed template names the RECIPIENT inside the delimiter
(``<|eom|><|start|>assistant to=<recipient><|message|>``), so the closer is not one literal: routing
to a tool rather than the user produces different middle tokens. A release pattern is therefore
``(head, tail, max_gap)`` — "``head``, then at most ``max_gap`` arbitrary tokens, then ``tail``" —
which collapses to a plain "window ends with ``head``" when ``tail`` is empty. Single-token
``</think>`` is ``((id,), (), 0)`` and behaves exactly as it did before this module existed.

MATCHING IS A ROLLING WINDOW, NOT A CURSOR
------------------------------------------
Matching keeps the last ``width`` committed tokens and compares suffixes. An advancing cursor that
resets to 0 on divergence is subtly WRONG for a self-overlapping pattern: against the stream
``4,4,4,5`` a cursor misses ``(4,4,5)`` because the reset discards the ``4`` that should have started
the next attempt. The window is a handful of ints per gated request and needs no failure function to
review.

PURITY / TP LOCKSTEP
--------------------
``commit`` is the ONLY mutator. Everything else — ``forced_next``, ``suppress_eos``, ``scan`` — is a
pure query, so it is safe to call any number of times per step and on any rank. That is what lets
speculative decode ask "where would this candidate chain release?" (``scan``) during its per-rank
verify walk WITHOUT advancing state, and then advance exactly once, on every rank, from the
rank0-authoritative committed tokens. A drafted-then-rejected token can never reach ``commit``.

This module is deliberately free of torch, tokenizers, os.environ and logging: it speaks only ``int``
and ``tuple[int, ...]``. String->id resolution and bitmask writing stay in the scheduler. That is
what makes it unit-testable on a box with no GPU (``tests/think_gate_test.py``).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Ids = Tuple[int, ...]
# (head, tail, max_gap). tail empty -> "the window ends with head". Otherwise -> "the window ends
# with tail, and head ends at most max_gap tokens before tail starts".
Pattern = Tuple[Ids, Ids, int]

# Budget value meaning "never force" — large enough that `count < budget` is always true for any
# reachable token count, so the backstop is disarmed without making `budget` Optional everywhere.
_UNBOUNDED = 1 << 62


def _as_pattern(p) -> Optional[Pattern]:
    """Normalise a caller-supplied release spec. Accepts a bare id sequence (exact match) or an
    explicit ``(head, tail, max_gap)`` triple. Returns None for anything empty — an empty pattern
    would match every window and open the gate instantly."""
    if not p:
        return None
    if isinstance(p, tuple) and len(p) == 3 and isinstance(p[0], (tuple, list)) \
            and isinstance(p[1], (tuple, list)):
        head, tail, gap = tuple(p[0]), tuple(p[1]), int(p[2])
        return (head, tail, max(0, gap)) if head else None
    head = tuple(p)
    return (head, (), 0) if head else None


def _pattern_width(p: Pattern) -> int:
    head, tail, gap = p
    return len(head) + (len(tail) + gap if tail else 0)


def _pattern_tokens(p: Pattern) -> Set[int]:
    head, tail, _ = p
    return set(head) | set(tail)


@dataclass
class _ToolCallState:
    openers: Tuple[Pattern, ...]   # any of these, once committed, means a call block is OPEN
    closers: Tuple[Pattern, ...]   # any of these closes it again
    width: int                     # committed tokens the matcher has to remember
    budget: int                    # tokens to keep suppressing before giving up on this block
    may_suppress_eos: bool         # False when an EOS id appears INSIDE a delimiter (see arm())
    inside: bool = False
    count: int = 0                 # tokens committed since the block opened
    window: List[int] = field(default_factory=list)
    # THINK-GATED (arm(..., think_closers=...)): opener matching is SUSPENDED from arm until one
    # of the think-close patterns commits. Set only for a request the template starts INSIDE its
    # reasoning span; the patterns are the span's close (and answer-header) delimiters.
    think_closers: Tuple[Pattern, ...] = ()
    in_think: bool = False


class ToolCallGate:
    """Stop the model ENDING A TURN while a tool-call block it opened is still unclosed.

    THE DEFECT THIS EXISTS FOR (2026-09-21). Until 3799717d an xgrammar structural tag constrained
    the body of a `<tool_call>` wrapper, and a side effect of that constraint was that the turn-ending
    token could not be sampled mid-structure: measured against the live tokenizer, `<|im_end|>` was
    MASKED inside an open call. Removing the tag fixed a much worse problem (it forced a JSON body on
    a checkpoint whose template mandates XML -- 25% junk arguments, code bodies capped at 369 chars)
    but it also removed that side effect, and the model began ending turns halfway through a call it
    had started. The frontend recovers the fragment rather than dropping it
    (`ToolCallStreamState._parse_unclosed`) and reports finish_reason=length, so the visible symptom
    is a truncated turn with the partial call surfaced as prose -- twice in eleven turns of Hermes
    session e5b8b76e21e8.

    This restores ONLY that property. It constrains no format and reads no schema: the body stays
    whatever the checkpoint's template renders.

    BOUNDED, because suppressing EOS is otherwise a way to hang a request. `budget` caps how long one
    block may suppress; past it the model may end the turn and the frontend's existing runaway
    handling (`MINISGL_TOOLCALL_RUNAWAY_LIMIT`) takes over. A gate that could refuse EOS forever would
    trade a truncated turn for a wedged one.

    Same purity contract as :class:`ThinkGate` -- ints and tuples only, `commit` the sole mutator, so
    it is unit-testable with no GPU and safe to query per rank."""

    def __init__(self, *, enabled: bool = True, budget: int = 8192) -> None:
        self._enabled = bool(enabled)
        self._budget = int(budget)
        self._st: Dict[object, _ToolCallState] = {}

    def arm(self, uid, *, openers: Iterable, closers: Iterable, eos_ids: Iterable[int] = (),
            think_closers: Iterable = ()) -> bool:
        """Arm for one request. No-op (False) when disabled, already armed, or given no usable
        opener/closer pair.

        ``think_closers`` (THINK-GATED mode, SamplingParams.tool_match_gated) suspends opener
        matching until one of these patterns commits: the request's template starts the model
        INSIDE its reasoning span, and on such a checkpoint a tool opener emitted before the
        span's closer is template-register noise, not a call — the degenerating turn of session
        ebbc1dd0903b emitted one mid-think, the gate armed on it, and EOS stayed suppressed
        while the model looped for 15 minutes. Pass the span's close delimiter (and the
        answer-turn header, which also ends the span) resolved to token sequences; an empty
        iterable keeps the ungated behaviour.
        """
        if not self._enabled or uid in self._st:
            return False
        opats = tuple(q for q in (_as_pattern(o) for o in openers) if q is not None)
        cpats = tuple(q for q in (_as_pattern(c) for c in closers) if q is not None)
        if not opats or not cpats:
            return False
        tpats = tuple(q for q in (_as_pattern(t) for t in think_closers) if q is not None)
        # An EOS id that is itself PART of a delimiter cannot be suppressed: masking it would stop the
        # model ever emitting the closer, turning the guard into the hang it is meant to avoid. Mirrors
        # ThinkGate.may_suppress_eos.
        delim_tokens: Set[int] = set()
        for q in opats + cpats + tpats:
            delim_tokens |= _pattern_tokens(q)
        may = not (set(int(e) for e in eos_ids) & delim_tokens)
        self._st[uid] = _ToolCallState(
            openers=opats, closers=cpats,
            width=max(_pattern_width(q) for q in opats + cpats + tpats),
            budget=self._budget, may_suppress_eos=may,
            think_closers=tpats, in_think=bool(tpats),
        )
        return True

    def commit(self, uid, token: int) -> None:
        """Advance by one COMMITTED token. Drafted-then-rejected speculative tokens must not reach
        here, or the window diverges from what the model conditioned on (and across TP ranks)."""
        st = self._st.get(uid)
        if st is None:
            return
        st.window.append(int(token))
        del st.window[: max(0, len(st.window) - st.width)]
        if st.in_think:
            # THINK-GATED: openers committed inside the reasoning span are inert. Only a
            # think-close (or answer-header) pattern commits here ends the suspension — the
            # caller armed with the span's own delimiters, so a genuine post-span opener still
            # latches on the very next token.
            if any(ThinkGate._matches(st.window, q) for q in st.think_closers):
                st.in_think = False
            return
        if st.inside:
            st.count += 1
            if any(ThinkGate._matches(st.window, q) for q in st.closers):
                st.inside, st.count = False, 0
        elif any(ThinkGate._matches(st.window, q) for q in st.openers):
            st.inside, st.count = True, 0

    def commit_many(self, uid, tokens: Sequence[int]) -> None:
        for t in tokens:
            self.commit(uid, t)

    def suppress_eos(self, uid) -> bool:
        st = self._st.get(uid)
        return bool(st and st.inside and st.may_suppress_eos and st.count < st.budget)

    def is_open(self, uid) -> bool:
        st = self._st.get(uid)
        return bool(st and st.inside)

    def any_armed(self) -> bool:
        return bool(self._st)

    def free(self, uid) -> None:
        self._st.pop(uid, None)


@dataclass
class _GateState:
    """Per-request gate state. All of it is derived from COMMITTED tokens, so every TP rank holds an
    identical copy."""

    force: Ids                     # what the β backstop emits, in order (the concrete close delim)
    releases: Tuple[Pattern, ...]  # any of these, once committed, ends the reasoning span
    budget: int                    # reasoning tokens allowed before the backstop forces `force`
    width: int                     # how many committed tokens the matcher has to remember
    may_suppress_eos: bool         # False when an EOS id appears INSIDE a delimiter (see arm())
    count: int = 0                 # reasoning tokens committed; forced tokens are NOT counted
    window: List[int] = field(default_factory=list)  # rolling last `width` committed tokens
    forcing: int = -1              # -1 = no force run in flight; else next index into `force`


class ThinkGate:
    """Owns "is this request still inside its reasoning span" for every live request.

    Armed per request via :meth:`arm`, advanced by :meth:`commit` from committed tokens only, and
    released when a registered pattern completes (or when the β backstop has finished force-emitting
    ``force``). A released uid is remembered in ``_done`` so it cannot re-arm mid-request.
    """

    def __init__(self, *, enabled: bool = True, default_budget: int = 1024) -> None:
        self._enabled = bool(enabled)
        self._default_budget = int(default_budget) if int(default_budget) > 0 else 1024
        self._st: Dict[int, _GateState] = {}
        self._done: Set[int] = set()

    # ---------------------------------------------------------------- lifecycle

    def arm(
        self,
        uid: int,
        *,
        force_seq: Sequence[int],
        release_seqs: Iterable = (),
        budget: Optional[int] = None,
        max_tokens: int = 0,
        eos_ids: Iterable[int] = (),
    ) -> bool:
        """Arm the gate for ``uid``. Idempotent: a no-op returning False when the gate is disabled,
        already armed, already released for this uid, or handed an empty ``force_seq``.

        ``force_seq`` is what the backstop emits at budget — it must be a concrete, emittable id
        sequence (no wildcards). It is ALSO registered as a release pattern, so the model closing the
        span itself is detected by the same matcher that detects the forced close.

        ``release_seqs`` are additional patterns that end the reasoning span but are never forced:
        the answer-turn header (the model answered directly, without reasoning) and the
        wildcard-recipient form of the closer (the model routed to a tool rather than the user).
        """
        if not self._enabled:
            return False
        if uid in self._st or uid in self._done:
            return False
        force = tuple(force_seq or ())
        if not force:
            return False

        pats: List[Pattern] = []
        for cand in (force, *release_seqs):
            p = _as_pattern(cand)
            if p is not None and p not in pats:
                pats.append(p)
        patterns = tuple(pats)

        # Three-way, because "no cap requested" and "cap explicitly disabled" are different asks:
        #   budget > 0  -> that many reasoning tokens
        #   budget < 0  -> UNBOUNDED: the caller has another mechanism and does not want the
        #                  backstop. Used when the chat template consumes the reasoning LEVEL
        #                  (Qwen3.8's `reasoning_effort`), where the model self-regulates from its
        #                  own system-prompt line and a token cap just guillotines it mid-thought.
        #   None / 0    -> nothing requested; the server default applies.
        # `_UNBOUNDED` rather than a None budget so every `count < budget` comparison below stays a
        # plain int compare and no call site needs an Optional guard.
        if isinstance(budget, int) and budget < 0:
            eff = _UNBOUNDED
        elif isinstance(budget, int) and budget > 0:
            eff = int(budget)
        else:
            eff = self._default_budget
        # Reserve room for the ANSWER *and* for the force run itself. Forcing a 6-token delimiter
        # needs 6 decode steps of headroom; without subtracting them a long delimiter can run into
        # `not req.can_decode` and finish `length` mid-delimiter, leaving a half-written turn header
        # that the reasoning parser reads as an unterminated span — the very symptom this fixes.
        # Skipped when unbounded: re-deriving a cap from max_tokens there would be the same bug this
        # sentinel exists to fix — a caller asking for no backstop, silently getting one at 3/4 of
        # max_tokens. The cost is that a model which never closes its span runs to max_tokens and the
        # reply arrives entirely as reasoning_content; that is the caller's trade to make, and it is
        # what upstream engines do by default.
        if eff != _UNBOUNDED and max_tokens > 0 and eff + len(force) >= max_tokens:
            eff = max(1, (max_tokens * 3) // 4 - len(force))

        # If an EOS id sits INSIDE a delimiter, suppressing EOS while gated would make that delimiter
        # unreachable: the model could never close, and the backstop's own forced token would be
        # masked out too — a request that can never finish. Rare, but silent and total, so the
        # suppression is dropped for such a request rather than gambling.
        eos = set(eos_ids or ())
        may_suppress = not any(eos & _pattern_tokens(p) for p in patterns)

        self._st[uid] = _GateState(
            force=force,
            releases=patterns,
            budget=eff,
            width=max(_pattern_width(p) for p in patterns),
            may_suppress_eos=may_suppress,
        )
        return True

    def clear(self, uid: int) -> None:
        """Release the gate for ``uid`` and remember that it released, so a later ``arm`` (the plain
        decode path calls it every step) cannot re-arm a request whose reasoning phase already ended."""
        self._st.pop(uid, None)
        self._done.add(uid)

    def free(self, uid: int) -> None:
        """Forget ``uid`` entirely — request finished and its resources are being reclaimed. Distinct
        from :meth:`clear`, which deliberately REMEMBERS the uid."""
        self._st.pop(uid, None)
        self._done.discard(uid)

    # ------------------------------------------------------------- pure queries

    def is_armed(self, uid) -> bool:
        return uid in self._st

    def any_armed(self) -> bool:
        """Fast bail-out for the per-step batch builders, which must stay free in the common case
        where nothing in the batch is reasoning-gated."""
        return bool(self._st)

    def is_done(self, uid) -> bool:
        return uid in self._done

    def budget_of(self, uid) -> Optional[int]:
        st = self._st.get(uid)
        return None if st is None else st.budget

    def count_of(self, uid) -> Optional[int]:
        st = self._st.get(uid)
        return None if st is None else st.count

    def forced_next(self, uid) -> Optional[int]:
        """The token the β backstop demands RIGHT NOW, or None while the request is still under
        budget (or not gated at all). Emitting it is the caller's job — by masking the bitmask row to
        this id alone, or by overriding the sampled token."""
        st = self._st.get(uid)
        if st is None or st.count < st.budget:
            return None
        return st.force[self._force_base(st)]

    def forced_at(self, uid, ahead: int) -> Optional[int]:
        """:meth:`forced_next` shifted ``ahead`` positions, for the speculative path's per-position
        verify mask. Clamped to the last id: positions past the end of the delimiter are cut off by
        the caller's truncation at the release point, so the clamp is never actually emitted."""
        st = self._st.get(uid)
        if st is None or st.count < st.budget:
            return None
        return st.force[min(self._force_base(st) + max(0, int(ahead)), len(st.force) - 1)]

    def suppress_eos(self, uid) -> bool:
        """Should this request's EOS logits be masked this step? Only while genuinely mid-reasoning:
        gated, under budget, and with no EOS id buried in its delimiters. Once the backstop starts
        forcing, the row is masked to the forced token anyway and EOS suppression is redundant.

        NEVER under an unbounded budget. Holding EOS is only legitimate because the β backstop is
        guaranteed to release it — the request is stopped from ending its turn mid-thought, and a
        few hundred tokens later the closer is forced and it can stop. Take the backstop away and the
        two halves of that bargain come apart: `count < budget` is true forever, EOS stays masked
        forever, `forced_next` never fires, and the request CANNOT terminate. It runs to max_tokens
        sampling ever-less-probable tokens — observed as an answer decaying into `. . . . .` and
        mojibake. Unbounded thinking has to mean the model may also stop on its own."""
        st = self._st.get(uid)
        if st is None or not st.may_suppress_eos:
            return False
        if st.budget >= _UNBOUNDED:
            return False
        return st.count < st.budget

    def scan(self, uid, tokens: Sequence[int]) -> Optional[int]:
        """Index in ``tokens`` at which the gate WOULD release, or None. Pure — the real state is
        untouched, so a speculative chain can be probed before it is known which of its tokens will
        actually be committed."""
        st = self._st.get(uid)
        if st is None:
            return None
        window = list(st.window)
        for i, tok in enumerate(tokens):
            window.append(tok)
            del window[: max(0, len(window) - st.width)]
            if self._released(window, st.releases):
                return i
        return None

    # ------------------------------------------------------------ the mutator

    def commit(self, uid, token: int) -> bool:
        """Advance the gate by one COMMITTED token. Returns True iff the gate released on it.

        Only committed tokens may be passed: a drafted-then-rejected speculative token must never
        reach here, or the match window would diverge from what the model actually conditioned on
        (and, under TP, from the other ranks).
        """
        st = self._st.get(uid)
        if st is None:
            return False

        base = self._force_base(st) if st.count >= st.budget else None
        forced = base is not None and token == st.force[base]

        st.window.append(int(token))
        del st.window[: max(0, len(st.window) - st.width)]

        if forced:
            st.forcing = base + 1
        else:
            # Any token that is not the one being forced invalidates the in-flight force run; the
            # next `forced_next` re-derives its position from what is actually in the window.
            st.forcing = -1
            # Counted EAGERLY, delimiter candidates included. Deferring the count until a partial
            # match resolves would mean a divergent match silently loses those tokens from the
            # budget; the cost of counting them is a handful of tokens against a budget of hundreds.
            st.count += 1

        if self._released(st.window, st.releases):
            self.clear(uid)
            return True
        if st.forcing >= len(st.force):
            # The backstop finished emitting the delimiter. `_released` normally fires first (force
            # is itself a release pattern); this is the belt-and-braces path for a force sequence
            # whose tail was clipped out of the window by a longer wildcard pattern.
            self.clear(uid)
            return True
        return False

    def commit_many(self, uid, tokens: Sequence[int]) -> Optional[int]:
        """:meth:`commit` over a committed run, stopping at the release. Returns the releasing index
        or None. Used by the speculative path, which commits several tokens per step."""
        for i, tok in enumerate(tokens):
            if self.commit(uid, tok):
                return i
        return None

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _suffix_eq(window: List[int], seq: Ids, end: int) -> bool:
        """Does ``window`` carry ``seq`` immediately before offset ``end`` from its right edge?"""
        n = len(seq)
        if n == 0:
            return True
        hi = len(window) - end
        lo = hi - n
        return lo >= 0 and tuple(window[lo:hi]) == seq

    @classmethod
    def _matches(cls, window: List[int], pat: Pattern) -> bool:
        head, tail, gap = pat
        if not tail:
            return cls._suffix_eq(window, head, 0)
        if not cls._suffix_eq(window, tail, 0):
            return False
        # `head` must end somewhere in the `gap` tokens preceding `tail`.
        for g in range(gap + 1):
            if cls._suffix_eq(window, head, len(tail) + g):
                return True
        return False

    @classmethod
    def _released(cls, window: List[int], pats: Tuple[Pattern, ...]) -> bool:
        return any(cls._matches(window, p) for p in pats)

    @staticmethod
    def _force_base(st: _GateState) -> int:
        """Where the force run should resume. An explicit in-flight cursor wins; otherwise align to
        the longest prefix of ``force`` the model has ALREADY emitted on its own. Without that
        alignment, a model that had got as far as ``<|eom|><|start|>`` before the budget expired
        would have a whole second delimiter appended to it."""
        if st.forcing >= 0:
            return st.forcing
        f = st.force
        for k in range(min(len(f) - 1, len(st.window)), 0, -1):
            if tuple(st.window[-k:]) == f[:k]:
                return k
        return 0
