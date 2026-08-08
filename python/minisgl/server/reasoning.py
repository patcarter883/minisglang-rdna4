from __future__ import annotations

"""Reasoning-content extraction for chain-of-thought / "thinking" models.

Reasoning models wrap their scratch reasoning in a delimiter pair and emit the user-facing answer
after the closing one. The pair is NOT universal and it is NOT even symmetric:

* Qwen3 / GLM-4.x / Laguna / ZAYA:  ``<think> … </think>``
* Gemma-4 / diffusiongemma:         ``<|channel>thought … <channel|>`` — a named CHANNEL whose
  close token is the family's mirrored-pipe form (the same convention as its
  ``boi_token``/``eoi_token`` pair ``<|image>``/``<image|>``), so open and close share no stem.

Several templates put the *opening* delimiter in the generation PROMPT
(``…<|im_start|>assistant\n<think>\n``), so the OUTPUT carries only the CLOSING tag + the answer::

    Here's my reasoning: 1. ... 2. ...
    </think>

    The answer is 4.

Others leave the opener to the MODEL (Gemma-4 with thinking on, older Qwen3 templates), so the
output is ``<open> … <close> answer``. This module handles both.

WHERE THE DELIMITERS COME FROM — the model's own artifacts, not a name table
---------------------------------------------------------------------------
``resolve_reasoning_parser`` DERIVES the pair by rendering the checkpoint's chat template twice,
with ``enable_thinking`` true and false, and diffing the generation-prompt tails at TOKEN level.
That diff is definitionally the reasoning markup: "thinking off" is expressed by a template in
exactly one of two ways, and both show up in it —

* it injects an EMPTY PRE-CLOSED span (Qwen3-0.6B ``<think>\n\n</think>\n\n``, Gemma-4
  ``<|channel>thought\n<channel|>``) -> the whole pair is in the thinking-off tail;
* it SWAPS the opener for the closer (GLM, Laguna: ``<|assistant|><think>`` -> ``<|assistant|></think>``)
  -> opener in the thinking-on tail, closer in the thinking-off one.

A name table cannot express this and never could: keyed on family name, it IS a model-name branch,
every new checkpoint needs a new row, and a checkpoint with no row fell through to ``<think>`` —
which matched nothing on Gemma-4 and (with ``thinking_open`` defaulted True) sent every answer to
``reasoning_content`` with ``content=""``. Silent, total output loss at the API boundary. The table
survives only as the LAST resort, below.

Design: SPLIT ON THE CLOSING TAG WHEN PRESENT. When it is ABSENT, the disposition depends on
whether the model was actually inside a reasoning span (``thinking_open``, itself DERIVED from the
rendered prompt by ``ReasoningParser.prompt_state`` — never defaulted):

* **Span not open** (``thinking_open=False``): the prompt carries no unclosed opener — either none
  at all, or a *pre-closed* empty span. The output is a plain answer that never contains the close
  tag. Return it unchanged as ``content`` — a safe no-op, no misclassification.
* **Span open** (``thinking_open=True``): the template injected the opener, so the model is INSIDE
  the reasoning span from token 0. If generation ends before it emits the close tag — truncated at
  ``max_tokens``, or a model that just never closes — the ENTIRE output is scratch reasoning, NOT a
  user answer. Route it all to ``reasoning_content`` (content ``""``) rather than leaking a
  half-finished chain-of-thought into the visible answer.
* **Model-side opener**: when the output ITSELF starts with the open delimiter the span is open even
  though the prompt carried nothing — same treatment, decided from the text rather than assumed.
"""

from typing import List, Optional, Tuple

# LEGACY family -> (open_token, close_token) table. This is the LAST resort of
# `resolve_reasoning_parser`, not its mechanism: it is only consulted for a name a model DECLARED
# (generation_config.json `reasoning_parser`) or an operator passed on `--reasoning-parser`, after
# artifact derivation has already failed. Do NOT add a row for a new checkpoint — if derivation
# cannot see the model's delimiters, the fix is to make the derivation see them, because a row here
# helps exactly one checkpoint and leaves the next one broken the same way.
_REASONING_DELIMITERS = {
    "auto": ("<think>", "</think>"),
    "qwen3": ("<think>", "</think>"),
    "deepseek_r1": ("<think>", "</think>"),
    "deepseek-r1": ("<think>", "</think>"),
    "glm": ("<think>", "</think>"),
    "qwen": ("<think>", "</think>"),
    # Poolside/Laguna declares `reasoning_parser: poolside_v1` in generation_config.json. It IS the
    # <think>/</think> format (the turn is `<assistant><think>REASONING</think>ANSWER</assistant>`),
    # so the declared name maps here. Derivation reaches the same pair from Laguna's template.
    "poolside_v1": ("<think>", "</think>"),
    "poolside": ("<think>", "</think>"),
}

# The pair to fall back on when NOTHING resolved. Keeping it is safe *only because* `thinking_open`
# is now derived rather than defaulted: a parser whose delimiters the model never emits can no
# longer swallow an answer — with no close tag in the output and no open span in the prompt, `parse`
# is a verbatim passthrough. It still buys the split for always-thinking checkpoints whose template
# has no enable_thinking branch to diff and that declare nothing (e.g. DeepSeek-R1 derivatives,
# which emit `<think>` as ORDINARY TEXT — it is not even in their added vocab, so no artifact names
# it). That is the whole remaining job of the table.
_LEGACY_GENERIC = ("<think>", "</think>")

# Probe conversation for the template renders. Content is irrelevant — the reasoning delimiters are
# emitted by the template's `add_generation_prompt` branch, which keys on the thinking kwargs and
# the last turn's ROLE, not on message text.
_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]


class ReasoningParser:
    """Splits a completion into (reasoning_content, content) on a closing think tag."""

    def __init__(self, start_token: str = "<think>", end_token: str = "</think>") -> None:
        self.start_token = start_token
        self.end_token = end_token

    def prompt_state(self, prompt: str) -> Tuple[bool, bool]:
        """Read the RENDERED GENERATION PROMPT and report ``(span_open, reasoning_possible)``.

        The prompt is the ground truth for whether the model starts its completion inside a reasoning
        span — it is the literal text the model conditions on, so nothing has to be assumed about the
        checkpoint. Only the LAST delimiter matters; earlier ones belong to conversation history that
        the template already closed.

        * ``span_open`` — the last delimiter is an OPENER, so the model is mid-reasoning at token 0
          (Qwen3.6 ``…assistant\\n<think>\\n``, GLM ``<|assistant|><think>``). This is what feeds
          ``parse(thinking_open=…)``.
        * ``reasoning_possible`` — the prompt does not END the span, so reasoning may still appear.
          Weaker than ``span_open`` and true in one extra case: NEITHER delimiter present, i.e. the
          template left the opener to the model (Gemma-4 with thinking on, older Qwen3 templates).
          This is what gates the reasoning budget / grammar think-gate, which must stay armed for a
          model that has not opened its span YET.

        Both are false when the template PRE-CLOSED an empty span (``<think>\\n\\n</think>\\n\\n``,
        ``<|channel>thought\\n<channel|>``) — that is how every family spells "thinking off", and
        reading it here is what makes an explicit ``enable_thinking=false`` work without a special
        case: the kwarg reaches the template, the template closes the span, we see it closed.
        """
        open_at = prompt.rfind(self.start_token) if self.start_token else -1
        close_at = prompt.rfind(self.end_token)
        return open_at > close_at, close_at <= open_at

    def opens_span(self, text: str) -> bool:
        """Does this COMPLETION open its own reasoning span? True when the model emitted the opener
        itself because the template did not (Gemma-4 thinking-on renders a bare ``<|turn>model\\n``
        and lets the model write ``<|channel>thought``). Without this, such an output truncated
        before its close tag would be reported as a finished answer that is really raw scratch."""
        return bool(self.start_token) and text.lstrip().startswith(self.start_token)

    def parse(self, text: str, thinking_open: bool = False) -> Tuple[Optional[str], str]:
        """Split ``text`` into ``(reasoning_content, content)``.

        When the closing tag is present: reasoning is everything before it (a leading opening tag,
        if the model echoed one, is stripped), content is everything after.

        When the closing tag is ABSENT the disposition depends on whether the span was open —
        ``thinking_open`` (derived from the rendered prompt by ``prompt_state``) or, failing that,
        ``opens_span(text)`` (the model wrote the opener itself):

        * span NOT open (default): returned unchanged as ``content`` with ``reasoning_content=None``
          — a safe no-op. This is the case that a non-reasoning model, and every thinking-OFF
          request, MUST land in; getting it wrong returns ``content=""`` for the whole reply.
        * span open: the model was inside the reasoning span and generation ended before it emitted
          the closing tag (truncated at ``max_tokens``, or a model that never closes), so the WHOLE
          output is reasoning — returned as ``reasoning_content`` with ``content=""``, never leaking
          a half-finished chain-of-thought into the answer.
        """
        if not text:
            return None, text
        if self.end_token not in text:
            if not thinking_open and not self.opens_span(text):
                return None, text
            # The span was open (prompt-side, or the model opened it itself) and never closed -> it's
            # all reasoning.
            pre = text
            if self.start_token and self.start_token in pre:
                pre = pre.split(self.start_token, 1)[-1]
            return (pre.strip() or None), ""
        # Split on the LAST close tag, not the first: a model (esp. after a β-forced </think> in RSA)
        # may RE-OPEN <think>…</think> before its final answer. rpartition keeps ALL reasoning — the
        # original span plus any re-opened blocks — in reasoning_content and leaves only the final
        # answer as content, so the visible response never looks like leftover thinking. Identical to
        # partition for the normal single-</think> case.
        pre, _, post = text.rpartition(self.end_token)
        # If the model echoed an opening <think> (some do), keep only what follows it.
        if self.start_token and self.start_token in pre:
            pre = pre.split(self.start_token, 1)[-1]
        reasoning = pre.strip()
        content = post.lstrip("\n")
        return (reasoning or None), content

    def stream_state(self, active: bool) -> "ReasoningStreamState":
        return ReasoningStreamState(self.start_token, self.end_token, active)


def _end_overlap_len(text: str, token: str) -> int:
    """Largest k>0 such that ``text`` ends with ``token[:k]`` (a partial closing tag straddling a
    streaming boundary). Returns 0 when no suffix of ``text`` is a prefix of ``token``."""
    max_k = min(len(text), len(token) - 1)
    for k in range(max_k, 0, -1):
        if text.endswith(token[:k]):
            return k
    return 0


class ReasoningStreamState:
    """Incremental splitter for the streaming path. Feed each ``incremental_output`` chunk; get back
    ``(reasoning_delta, content_delta)`` (either may be None). Buffers a possible partial closing tag
    so a close delimiter split across two chunks is still detected.

    ``active`` is the prompt-derived ``span_open``. The head of the stream is WATCHED for a model-side
    opener either way, because both states can meet one and both need it gone:

    * ``active=False`` — Gemma-4 thinking-on writes its own ``<|channel>thought``, and without the
      watch the entire chain of thought streams out as ``content``: the streaming twin of the
      non-stream misroute.
    * ``active=True`` — the model may ECHO the opener even though the prompt already put it inside the
      span. Non-streaming ``parse`` drops such an echo (``pre.split(start_token, 1)[-1]``), so leaving
      it in would put raw markup at the head of every streamed ``reasoning_content`` and disagree with
      the other lane on identical bytes.

    Only as many characters are held back as the opener is long, and text that turns out NOT to be an
    opener is released into whichever channel the state was already in."""

    def __init__(self, start_token: str, end_token: str, active: bool) -> None:
        self.start_token = start_token
        self.end_token = end_token
        self.active = active  # currently inside the reasoning span
        self.pending = ""     # held-back tail that might be a partial end token
        self._content_started = False  # have we emitted any answer text yet
        self._reasoning_started = False  # have we emitted any reasoning text yet
        # Watch the head for an opener in BOTH states (see the class docstring).
        self._probing = bool(start_token)
        self._probe = ""      # head of the stream, held while it could still become the opener

    def _emit_content(self, text: str) -> Optional[str]:
        """Drop the template's leading `\\n\\n` separator on the first answer chunk (it may straddle
        the chunk that closed the reasoning span and the next), then pass content through verbatim."""
        if not self._content_started:
            text = text.lstrip("\n")
            if not text:
                return None
            self._content_started = True
        return text or None

    def _emit_reasoning(self, text: str) -> Optional[str]:
        """Same leading-newline trim for the first REASONING chunk, so streamed reasoning_content
        matches what non-streaming `parse` returns (it .strip()s). The separator after the opener
        usually arrives in the chunk AFTER the opener itself — `<|channel>thought` is one token and
        the `\\n` the next — so trimming only at the opener would miss it."""
        if not self._reasoning_started:
            text = text.lstrip("\n")
            if not text:
                return None
            self._reasoning_started = True
        return text or None

    def push(self, delta: str) -> Tuple[Optional[str], Optional[str]]:
        if self._probing:
            self._probe += delta
            head = self._probe.lstrip("\n")
            if head.startswith(self.start_token):
                # The model opened its own span: switch into reasoning mode and re-feed the rest.
                self._probing = False
                self.active = True
                self._probe = ""
                return self._push_active(head[len(self.start_token):])
            if head and self.start_token.startswith(head):
                return None, None    # still a viable prefix of the opener — keep holding
            if not head:
                return None, None    # nothing but the template's leading newlines so far
            # First character that rules the opener out. It is ordinary text, and which channel it
            # belongs to is the state we were told to start in — inside the span it is reasoning, and
            # only outside it is the answer. Emitting it as content unconditionally would leak a
            # prompt-opened chain of thought that simply did not re-echo its opener.
            self._probing = False
            out, self._probe = self._probe, ""
            if self.active:
                return self._push_active(out)
            return None, self._emit_content(out)
        if not self.active:
            return None, self._emit_content(delta)
        return self._push_active(delta)

    def _push_active(self, delta: str) -> Tuple[Optional[str], Optional[str]]:
        """Inside the reasoning span: emit reasoning until the close delimiter, then content."""
        text = self.pending + delta
        idx = text.find(self.end_token)
        if idx != -1:
            reasoning = self._emit_reasoning(text[:idx])
            self.active = False
            self.pending = ""
            return reasoning, self._emit_content(text[idx + len(self.end_token):])
        keep = _end_overlap_len(text, self.end_token)
        if keep:
            self.pending = text[-keep:]
            emit = text[:-keep]
        else:
            self.pending = ""
            emit = text
        return self._emit_reasoning(emit), None

    def flush(self) -> Tuple[Optional[str], Optional[str]]:
        """Emit any buffered tail at stream end, as ``(reasoning_tail, content_tail)``. Two buffers can
        hold text: the opener probe (the whole completion was shorter than the opener and a prefix of
        it, so it never became one) and the partial-close-tag tail (still inside reasoning). Dropping
        either would silently truncate the reply.

        The probe drains FIRST and through the same routing as ``push`` — it precedes the close-tag
        tail chronologically, and releasing it can itself leave a partial closer pending."""
        reasoning = content = None
        if self._probe:
            out, self._probe = self._probe, ""
            self._probing = False
            if self.active:
                reasoning, content = self._push_active(out)
            else:
                content = self._emit_content(out)
        if self.active and self.pending:
            tail, self.pending = self._emit_reasoning(self.pending), ""
            reasoning = ((reasoning or "") + (tail or "")) or None
        self._probing = False
        return (reasoning or None), content


def get_reasoning_parser(name: Optional[str]) -> Optional[ReasoningParser]:
    """Build a parser from the LEGACY family-name table; None disables reasoning extraction.
    Kept for the explicit ``--reasoning-parser <name>`` override and for a model-declared name —
    both of which are names, not artifacts. Everything else goes through the derivation below."""
    if not name or name == "none":
        return None
    delims = _REASONING_DELIMITERS.get(name.lower())
    if delims is None:
        return None
    return ReasoningParser(*delims)


# ---------------------------------------------------------------------------------------------
# Delimiter DERIVATION from the checkpoint's own artifacts
# ---------------------------------------------------------------------------------------------
# Everything below answers one question without knowing anything about the model: which two strings
# does THIS checkpoint use to open and close a reasoning span? The answer is read out of the chat
# template by rendering it, because the template is what actually produces the bytes the model
# conditions on — a config field is a CLAIM, the rendered prompt is the fact.
#
# Rejected as a source, on evidence: `tokenizer_config.json` `think_token`. Both Gemma checkpoints
# declare `"think_token": "<|think|>"` (next to `boi_token`/`eoi_token`), and it is tempting to read
# it as the span opener. It is NOT — rendering shows `<|think|>` is a MODE FLAG emitted once at the
# top of the first system turn to switch thinking on, while the span itself is
# `<|channel>thought … <channel|>`. Trusting the declaration would have re-broken exactly the model
# it appears on. Only the two Gemma checkpoints declare it at all, so it could not have replaced the
# table anyway.


def _added_vocab(tokenizer) -> set:
    """The checkpoint's special/added tokens. A reasoning delimiter is essentially always one of
    these (models are trained with the tag as a single token), which is what lets the diff below
    tell a delimiter apart from ordinary template punctuation."""
    vocab: set = set()
    try:
        vocab |= set(tokenizer.get_added_vocab().keys())
    except Exception:  # noqa: BLE001 — a tokenizer without added-vocab support is not fatal
        pass
    try:
        vocab |= {t for t in (tokenizer.all_special_tokens or []) if t}
    except Exception:  # noqa: BLE001
        pass
    return vocab


def _render_probe(tokenizer, add_generation_prompt: bool, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        _PROBE_MESSAGES, tokenize=False,
        add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking,
    )


def _generation_tail_ids(tokenizer, with_prompt: str, without_prompt: str) -> List[int]:
    """The token ids that ``add_generation_prompt`` APPENDED — the render WITH it minus its common
    prefix with the render WITHOUT it.

    Anchoring on this tail rather than diffing whole renders is load-bearing: several templates ALSO
    mutate the system block when thinking flips (Gemma-4 injects a whole `<|turn>system\\n<|think|>`
    turn, Laguna opens a system block it otherwise omits), which shifts every following token and
    leaves the two renders with no useful common prefix at all. The generation-prompt tail is where
    the reasoning delimiters live and it stays aligned."""
    with_ids = tokenizer(with_prompt, add_special_tokens=False)["input_ids"]
    without_ids = tokenizer(without_prompt, add_special_tokens=False)["input_ids"]
    i = 0
    while i < min(len(with_ids), len(without_ids)) and with_ids[i] == without_ids[i]:
        i += 1
    return with_ids[i:]


def derive_delimiters(tokenizer) -> Optional[Tuple[str, str, str]]:
    """Derive ``(open, close, how)`` from the checkpoint's chat template, or None if it has no
    reasoning convention. ``how`` is a human-readable provenance string for the boot log.

    Method: render the generation-prompt tail with ``enable_thinking`` true and false and diff them
    at TOKEN level (token level, not character level, so a shared prefix like the ``<`` of
    ``<think>``/``</think>`` cannot split a delimiter in half). Whatever the thinking-off render adds
    IS the closing delimiter, because "thinking off" is universally spelled by ENDING the span in the
    prompt. The opener is then read from one of three positions, in decreasing directness."""
    if not getattr(tokenizer, "chat_template", None):
        return None
    try:
        on = _render_probe(tokenizer, True, True)
        off = _render_probe(tokenizer, True, False)
    except Exception:  # noqa: BLE001 — a template that rejects the probe tells us nothing
        return None
    if on == off:
        # The template has no enable_thinking branch: either a non-reasoning model or an
        # always-thinking one. Nothing to diff — the caller falls through the cascade.
        return None
    try:
        on_ids = _generation_tail_ids(tokenizer, on, _render_probe(tokenizer, False, True))
        off_ids = _generation_tail_ids(tokenizer, off, _render_probe(tokenizer, False, False))
        added = _added_vocab(tokenizer)

        def as_text(toks: List[str]) -> str:
            return tokenizer.convert_tokens_to_string(toks).strip()

        def is_delim(tok: str) -> bool:
            # A delimiter is an added token with real (non-whitespace) surface form. The whitespace
            # test matters: some checkpoints register "\n\n" as an added token, and it sits right
            # next to the real close tag in the diff.
            return tok in added and len(as_text([tok])) >= 2

        i = 0
        while i < min(len(on_ids), len(off_ids)) and on_ids[i] == off_ids[i]:
            i += 1
        on_mid = tokenizer.convert_ids_to_tokens(on_ids[i:])
        off_mid = tokenizer.convert_ids_to_tokens(off_ids[i:])

        # CLOSE: the last delimiter-shaped token the thinking-OFF tail has and the thinking-ON one
        # does not. "Last" because a pre-closed empty span contributes BOTH tokens (Gemma-4's
        # `<|channel>` … `<channel|>`) and the closer is the trailing one.
        closers = [t for t in off_mid if is_delim(t) and t not in on_mid]
        if not closers:
            return None
        close_tok = closers[-1]
        ci = len(off_mid) - 1 - off_mid[::-1].index(close_tok)
        close = as_text([close_tok])

        # OPEN, most direct evidence first:
        # (a) thinking-off injected an EMPTY PRE-CLOSED span, so the opener sits in the same tail
        #     just before the closer (Gemma-4 `<|channel>thought\n<channel|>`, Qwen3-0.6B
        #     `<think>\n\n</think>\n\n`). Note this is the case that proves the pair need not be
        #     symmetric — the opener carries a channel NAME the closer does not.
        pre = as_text(off_mid[:ci])
        if pre and any(is_delim(t) for t in off_mid[:ci]):
            return pre, close, "chat template (thinking-off injects an empty pre-closed span)"
        # (b) thinking-off SWAPPED the opener for the closer, so the opener is what only the
        #     thinking-on tail has (GLM `<|assistant|><think>`, Laguna `<assistant><think>`).
        on_text = as_text(on_mid)
        if on_text and any(is_delim(t) for t in on_mid):
            return on_text, close, "chat template (thinking-off swaps the opener for the closer)"
        # (c) BOTH renders open the span and thinking-off merely closes it immediately after
        #     (Qwen3.6 `…<think>\n` vs `…<think>\n\n</think>\n\n`), so the opener is the last
        #     delimiter in the part the two tails share.
        for tok in reversed(tokenizer.convert_ids_to_tokens(on_ids[:i])):
            if is_delim(tok) and tok != close_tok:
                return as_text([tok]), close, "chat template (opener common to both renders)"
        return None
    except Exception:  # noqa: BLE001 — derivation is best-effort; the cascade has fallbacks
        return None


def resolve_reasoning_parser(
    tokenizer, requested: str = "auto", declared: Optional[str] = None
) -> Tuple[Optional[ReasoningParser], str]:
    """Resolve the served checkpoint's reasoning parser, returning ``(parser, provenance)``.

    The cascade, in order, with the reason for the order:

    1. **``--reasoning-parser <name>``** (anything but ``auto``). An operator who names a parser —
       or ``none`` to disable — outranks any inference. This is the escape hatch for a checkpoint
       whose artifacts say nothing.
    2. **Derivation from the rendered chat template** (``derive_delimiters``). First because it is
       the only step that needs no per-model knowledge and therefore the only one that works on a
       checkpoint nobody has seen yet. Measured to reproduce the legacy table EXACTLY on every
       cached checkpoint the table covered (Qwen3, Qwen3.5/3.6, GLM-4.7, Laguna, ZAYA1), and to
       find Gemma-4's asymmetric `<|channel>thought`/`<channel|>` pair that the table could not
       even represent.
    3. **The model author's declared ``reasoning_parser``** (generation_config.json, e.g. Laguna's
       ``poolside_v1``) resolved through the legacy table. Below derivation because it is a NAME,
       not markup: it can only ever select a row somebody already wrote, and it is a claim about
       the checkpoint rather than the checkpoint itself.
    4. **The legacy generic ``<think>``/``</think>`` pair**. See ``_LEGACY_GENERIC``: with
       ``thinking_open`` derived it can no longer misroute an answer, and it still serves
       always-thinking checkpoints whose template has no thinking branch to diff.

    Cached by the caller — this is a per-model resolution, not a per-request one.
    """
    if requested and requested != "auto":
        return get_reasoning_parser(requested), f"--reasoning-parser {requested}"
    if tokenizer is not None:
        derived = derive_delimiters(tokenizer)
        if derived is not None:
            start, end, how = derived
            return ReasoningParser(start, end), f"derived from {how}"
    if declared:
        parser = get_reasoning_parser(declared)
        if parser is not None:
            return parser, f"generation_config.json reasoning_parser={declared!r} (legacy table)"
    return ReasoningParser(*_LEGACY_GENERIC), "legacy generic <think>/</think> fallback"
