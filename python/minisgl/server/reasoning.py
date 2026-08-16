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


class _EndMatcher:
    """Finds the reasoning CLOSE delimiter in a string.

    Two modes. Normally the closer is one literal (``</think>``) and this is ``str.find`` with extra
    steps. But a channel-routed template names the RECIPIENT inside the delimiter, so the closer is a
    FAMILY of strings: Muse-Glimmer ends reasoning with
    ``<|eom|><|start|>assistant to=<recipient><|message|>``, where the recipient is ``user`` for an
    answer but a TOOL NAME when the model routes to a tool. Deriving the concrete ``to=user`` form and
    matching it literally leaves a tool-routed reply with no closer at all — ``parse`` then reads it
    as an unterminated reasoning span and files the whole reply under ``reasoning_content``.

    So when ``prefix``/``suffix`` are known the match is "``prefix``, then at most ``MAX_GAP``
    characters, then ``suffix``". ``MAX_GAP`` is what stops the pattern spanning half a reply when the
    model emits ``prefix`` and never follows through.
    """

    MAX_GAP = 128  # characters allowed between prefix and suffix (a recipient / tool name)

    def __init__(self, end_token: str, end_prefix: str = "", end_suffix: str = "") -> None:
        self.token = end_token or ""
        self.prefix = end_prefix or ""
        self.suffix = end_suffix or ""
        self.wild = bool(self.prefix and self.suffix)

    def find(self, text: str, start: int = 0) -> Optional[Tuple[int, int]]:
        """``(begin, end)`` of the first closer at or after ``start``, or None."""
        if not self.wild:
            if not self.token:
                return None
            i = text.find(self.token, start)
            return (i, i + len(self.token)) if i >= 0 else None
        i = start
        while True:
            p = text.find(self.prefix, i)
            if p < 0:
                return None
            body = p + len(self.prefix)
            s = text.find(self.suffix, body)
            if s >= 0 and (s - body) <= self.MAX_GAP:
                return (p, s + len(self.suffix))
            i = p + 1

    def rfind(self, text: str) -> Optional[Tuple[int, int]]:
        """``(begin, end)`` of the LAST closer, or None. See ``ReasoningParser.parse`` for why the
        last one and not the first."""
        last = None
        m = self.find(text, 0)
        while m is not None:
            last = m
            m = self.find(text, m[0] + 1)
        return last

    def hold(self, text: str) -> int:
        """How many trailing characters the streaming splitter must hold back so a closer straddling a
        chunk boundary is still detected. For the wildcard form that includes an already-complete
        ``prefix`` still waiting for its ``suffix`` — otherwise the recipient name would stream out as
        visible content before we learn it was markup."""
        if not self.wild:
            return _end_overlap_len(text, self.token)
        p = text.rfind(self.prefix)
        if p >= 0 and self.find(text, p) is None:
            span = len(text) - p
            if span <= len(self.prefix) + self.MAX_GAP + len(self.suffix):
                return span
        return _end_overlap_len(text, self.prefix)


class ReasoningParser:
    """Splits a completion into (reasoning_content, content) on a closing think tag."""

    def __init__(
        self,
        start_token: str = "<think>",
        end_token: str = "</think>",
        turn_header: str = "",
        end_prefix: str = "",
        end_suffix: str = "",
    ) -> None:
        self.start_token = start_token
        self.end_token = end_token
        # The closer with a WILDCARD recipient, when the template varies it by recipient (see
        # `_EndMatcher`). `end_token` stays the CONCRETE form: it is what the engine force-emits at
        # the reasoning budget, so it has to be one emittable literal.
        self.end_prefix = end_prefix or ""
        self.end_suffix = end_suffix or ""
        self._end = _EndMatcher(end_token, self.end_prefix, self.end_suffix)
        # Literal that a completion opens with before its ANSWER, when the template's generation
        # prompt stops MID-HEADER. Empty for every family whose prompt ends at a turn boundary
        # (`<|im_start|>assistant\n`), which is almost all of them.
        #
        # Muse-Glimmer's prompt ends `<|start|>assistant` and the model itself writes the recipient,
        # so a reply with no reasoning begins ` to=user<|message|>` — markup that would otherwise be
        # served as the first characters of the answer. The reasoning path never sees it, because
        # there it is part of the CLOSE delimiter (`<|eom|><|start|>assistant to=user<|message|>`);
        # this covers the case where the model answers directly.
        self.turn_header = turn_header
        # Tool-call block openers (assigned by the server at resolve time — the table lives with the
        # tool parser and this module must not import it). A model that opens a tool call while
        # still inside its reasoning span, WITHOUT emitting the close delimiter, has implicitly
        # finished thinking: both reference engines encode this rule for the Qwen3-family (vLLM
        # qwen3_reasoning_parser `<tool_call>` = implicit reasoning end; SGLang `tool_start_token`).
        # Without it the whole call stays trapped in reasoning_content — the tool subsystem only
        # sees the CONTENT channel — and the client receives an empty answer with no tool_calls.
        # The opener is PRESERVED into content (SGLang behaviour) so the tool parser consumes it.
        self.tool_openers: tuple = ()

    def _strip_turn_header(self, text: str) -> str:
        """Drop a leading answer-turn header. Checked AFTER the opener, which wins: the two share a
        prefix (` to=self…` vs ` to=user…`), and treating a reasoning opener as a header would put
        the chain of thought into `content`."""
        if not self.turn_header or text.startswith(self.start_token):
            return text
        if text.startswith(self.turn_header):
            return text[len(self.turn_header) :]
        return text

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
        _close = self._end.rfind(prompt)
        close_at = _close[0] if _close is not None else -1
        return open_at > close_at, close_at <= open_at

    def opens_span(self, text: str) -> bool:
        """Does this COMPLETION open its own reasoning span? True when the model emitted the opener
        itself because the template did not (Gemma-4 thinking-on renders a bare ``<|turn>model\\n``
        and lets the model write ``<|channel>thought``). Without this, such an output truncated
        before its close tag would be reported as a finished answer that is really raw scratch.

        Both sides are lstripped. The opener is not always a tag: a channel-routed family opens its
        reasoning with a RECIPIENT (Muse-Glimmer's `` to=self<|message|>``, whose leading space is
        part of the delimiter because the generation prompt ends mid-header at ``<|start|>assistant``).
        Stripping only `text` would compare a space-led delimiter against space-stripped text and
        never match — which is precisely the truncated-reasoning case this method exists to catch,
        so the failure would be a full chain-of-thought served as the answer."""
        return bool(self.start_token) and text.lstrip().startswith(self.start_token.lstrip())

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
        text = self._strip_turn_header(text)
        closer = self._end.rfind(text)
        if closer is None:
            if not thinking_open and not self.opens_span(text):
                return None, text
            # The span was open (prompt-side, or the model opened it itself) and never closed. If a
            # tool-call opener appears in the body, the model implicitly ended its reasoning there
            # (see `tool_openers`): reasoning is everything before it, and the opener plus the rest
            # is CONTENT so the tool parser can consume the call. Otherwise it's all reasoning.
            pre = text
            if self.start_token and self.start_token in pre:
                pre = pre.split(self.start_token, 1)[-1]
            t_idx = -1
            for tok in self.tool_openers:
                j = pre.find(tok)
                if j != -1 and (t_idx == -1 or j < t_idx):
                    t_idx = j
            if t_idx != -1:
                return (pre[:t_idx].strip() or None), pre[t_idx:].lstrip("\n")
            return (pre.strip() or None), ""
        # Split on the LAST close tag, not the first: a model (esp. after a β-forced </think> in RSA)
        # may RE-OPEN <think>…</think> before its final answer. rpartition keeps ALL reasoning — the
        # original span plus any re-opened blocks — in reasoning_content and leaves only the final
        # answer as content, so the visible response never looks like leftover thinking. Identical to
        # partition for the normal single-</think> case.
        pre, post = text[: closer[0]], text[closer[1] :]
        # If the model echoed an opening <think> (some do), keep only what follows it.
        if self.start_token and self.start_token in pre:
            pre = pre.split(self.start_token, 1)[-1]
        reasoning = pre.strip()
        content = post.lstrip("\n")
        # A span RE-OPENED after the last closer and never closed again: everything from that opener
        # on is scratch, not answer. rpartition alone cannot see it — the tail has no closer to
        # partition on — so the opener and the thinking behind it went out as visible content. Seen
        # live: an answer ending `…implementation of the Delta-Schema.<|channel>thought`, which the
        # caller then stored and replayed, leaving the NEXT turn's prompt genuinely mid-span; that
        # turn thought for 15k characters and returned an empty answer.
        if self.start_token and self.start_token in content:
            content, _, tail = content.partition(self.start_token)
            tail = tail.strip()
            if tail:
                reasoning = f"{reasoning}\n{tail}" if reasoning else tail
            content = content.rstrip()
        return (reasoning or None), content

    def stream_state(self, active: bool) -> "ReasoningStreamState":
        return ReasoningStreamState(
            self.start_token, self.end_token, active, self.turn_header,
            self.end_prefix, self.end_suffix, tool_openers=self.tool_openers,
        )


def _end_overlap_len(text: str, *tokens: str) -> int:
    """Largest k>0 such that ``text`` ends with ``token[:k]`` for one of ``tokens`` (a delimiter
    straddling a streaming boundary). Returns 0 when no suffix of ``text`` is a prefix of any of
    them. Takes several because the CONTENT phase watches for BOTH delimiters at once — a re-opened
    span and a stale closer can each arrive split across two chunks."""
    best = 0
    for token in tokens:
        if not token:
            continue
        for k in range(min(len(text), len(token) - 1), 0, -1):
            if text.endswith(token[:k]):
                best = max(best, k)
                break
    return best


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

    def __init__(
        self, start_token: str, end_token: str, active: bool, turn_header: str = "",
        end_prefix: str = "", end_suffix: str = "", tool_openers: tuple = (),
    ) -> None:
        # See ReasoningParser.tool_openers: inside the span, one of these ends reasoning implicitly
        # and is PRESERVED into content for the tool parser.
        self.tool_openers = tuple(tool_openers)
        self.start_token = start_token
        self.end_token = end_token
        # Same closer matcher the non-streaming lane uses, so both agree on identical bytes.
        self._end = _EndMatcher(end_token, end_prefix, end_suffix)
        # See `ReasoningParser.turn_header`. The head probe below watches for this AS WELL AS the
        # opener, because the two can share a prefix (` to=self…` vs ` to=user…`) and giving up on
        # the opener must not release a half-matched header into `content`.
        self.turn_header = turn_header
        self.active = active  # currently inside the reasoning span
        self.pending = ""     # held-back tail that might be a partial end token
        self._content_started = False  # have we emitted any answer text yet
        self._reasoning_started = False  # have we emitted any reasoning text yet
        self._closed_once = False  # a close delimiter has been consumed (see _strip_stale_closers)
        # Watch the head for an opener in BOTH states (see the class docstring).
        self._probing = bool(start_token) or bool(turn_header)
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
            if self.start_token and head.startswith(self.start_token):
                # The model opened its own span: switch into reasoning mode and re-feed the rest.
                self._probing = False
                self.active = True
                self._probe = ""
                return self._push_active(head[len(self.start_token):])
            # The ANSWER-turn header (checked after the opener, which wins — they share a prefix).
            # Consume it and carry on in whatever channel we were in; it is markup, not text.
            if self.turn_header and head.startswith(self.turn_header):
                self._probing = False
                self._probe = ""
                rest = head[len(self.turn_header):]
                # Re-feed through the normal lane rather than emitting directly, so the delimiter
                # watch that `_push_inactive`/`_push_active` run still sees everything after it.
                return self._push_active(rest) if self.active else self._push_inactive(rest)
            if head and (
                (self.start_token and self.start_token.startswith(head))
                or (self.turn_header and self.turn_header.startswith(head))
            ):
                return None, None    # still a viable prefix of the opener OR the header — hold
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
            return self._push_inactive(delta)
        return self._push_active(delta)

    def _push_inactive(self, delta: str) -> Tuple[Optional[str], Optional[str]]:
        """Outside the reasoning span: emit content, but keep watching for BOTH delimiters.

        A span is not a once-per-completion event, and the content phase meets each delimiter for its
        own reason:

        * an OPENER means the model RE-OPENED a reasoning span after answering. Observed live: a turn
          whose answer ended `…understand the technical implementation of the Delta-Schema.<|channel>
          thought` — the markup shipped as the visible answer, and because the caller stores content
          and replays it, the NEXT turn's prompt then genuinely ended mid-thought-channel. That turn
          thought for 15,206 characters, never closed the span, and delivered an empty answer. One
          leaked delimiter, one dead turn.
        * a CLOSER means the span already closed once and this is a duplicate — the reasoning-budget
          backstop force-emits the close token at β, the model does not notice, keeps thinking, and
          emits its own closer later. Measured on gemma-4 at `reasoning_effort=low` (β=256):
          `…pointer overhead."<channel|>Radix trees save memory by…` as the answer. Dropped, because a
          second closer closes nothing.

        The opener is watched unconditionally; the closer only once one has been consumed
        (`_closed_once`), so a completion that never opened a span is passed through byte for byte and
        a model writing the delimiter as literal prose on a non-reasoning turn is not silently edited.
        """
        text = self.pending + delta
        self.pending = ""
        out: List[str] = []
        while text:
            i_open = text.find(self.start_token) if self.start_token else -1
            m_close = self._end.find(text) if self._closed_once else None
            i_close = m_close[0] if m_close is not None else -1
            if i_open != -1 and (i_close == -1 or i_open < i_close):
                # Re-entering the span: everything before the opener is answer text, everything after
                # is scratch — hand the remainder to the reasoning half, which owns it from here.
                out.append(text[:i_open])
                self.active = True
                content = self._emit_content("".join(out))
                reasoning, more = self._push_active(text[i_open + len(self.start_token):])
                return reasoning, ((content or "") + (more or "")) or None
            if i_close != -1:
                out.append(text[:i_close])
                text = text[m_close[1]:]
                continue
            break
        # Hold back a suffix that could be the head of either delimiter split across this chunk
        # boundary, exactly as the reasoning phase does — otherwise a delimiter straddling two chunks
        # survives into the answer.
        keep = max(_end_overlap_len(text, self.start_token), self._end.hold(text))
        if keep:
            self.pending = text[-keep:]
            text = text[:-keep]
        out.append(text)
        return None, self._emit_content("".join(out))

    def _push_active(self, delta: str) -> Tuple[Optional[str], Optional[str]]:
        """Inside the reasoning span: emit reasoning until the close delimiter, then content. A
        tool-call opener also ends the span (implicitly — see ReasoningParser.tool_openers), with
        the opener itself routed to CONTENT; whichever of closer/opener appears first wins."""
        text = self.pending + delta
        t_idx, t_tok = -1, None
        for tok in self.tool_openers:
            j = text.find(tok)
            if j != -1 and (t_idx == -1 or j < t_idx):
                t_idx, t_tok = j, tok
        m = self._end.find(text)
        if t_idx != -1 and (m is None or t_idx < m[0]):
            reasoning = self._emit_reasoning(text[:t_idx])
            self.active = False
            self._closed_once = True
            self.pending = ""
            # Opener PRESERVED: content phase re-consumes it so the tool splitter sees the block.
            _r, content = self._push_inactive(text[t_idx:])
            return ((reasoning or "") + (_r or "")) or None, content
        if m is not None:
            reasoning = self._emit_reasoning(text[:m[0]])
            self.active = False
            self._closed_once = True
            self.pending = ""
            # The remainder re-enters the CONTENT phase, which keeps watching for both delimiters —
            # the model can re-open a span, or emit a duplicate closer, in this very chunk.
            _r, content = self._push_inactive(text[m[1]:])
            return ((reasoning or "") + (_r or "")) or None, content
        keep = self._end.hold(text)
        keep = max(keep, _end_overlap_len(text, *self.tool_openers))
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
        it, so it never became one) and the partial-close-tag tail. Dropping either would silently
        truncate the reply.

        The probe drains FIRST and through the same routing as ``push`` — it precedes the close-tag
        tail chronologically, and releasing it can itself leave a partial closer pending.

        The partial-delimiter tail goes to whichever channel is live: reasoning while the span is
        open, and CONTENT once it has closed, because the content phase holds back a possible partial
        delimiter of its own (`_push_inactive`). Only the reasoning half used to be drained, so once
        that second buffer existed a reply ending in `<` would have lost its last character."""
        reasoning = content = None
        if self._probe:
            out, self._probe = self._probe, ""
            self._probing = False
            if self.active:
                reasoning, content = self._push_active(out)
            else:
                reasoning, content = self._push_inactive(out)
        if self.pending:
            tail, self.pending = self.pending, ""
            if self.active:
                reasoning = ((reasoning or "") + (self._emit_reasoning(tail) or "")) or None
            else:
                content = ((content or "") + (self._emit_content(tail) or "")) or None
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


# Probe for `derive_delimiters_from_history`. The marker strings only have to be findable in the
# render and absent from the template's own boilerplate; they are never tokenized.
_RSN_MARK = "ZQREASONINGZQ"
_ANS_MARK = "ZQANSWERZQ"
_HISTORY_PROBE = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "reasoning_content": _RSN_MARK, "content": _ANS_MARK},
]


def derive_delimiters_from_history(tokenizer) -> Optional[Tuple[str, str, str]]:
    """Derive ``(open, close, how)`` by rendering an assistant turn that CARRIES reasoning.

    Complements `derive_delimiters`, which diffs the `enable_thinking` branch of the generation
    prompt. That method sees nothing when a template has no such branch — either because the model
    is always-thinking, or because it spells the reasoning strength some other way (Muse-Glimmer's
    `reasoning_strength`). But such a template still has to RE-RENDER prior reasoning when it
    appears in history, and to do that it must emit the very delimiters we are looking for. So:
    hand it a turn with `reasoning_content` and read the markup straight off the result.

    This is more direct evidence than the thinking-off diff, not less — it is the template stating
    the pair rather than us inferring it from an absence — but it runs SECOND because it depends on
    the `reasoning_content` history convention, which not every template implements.

    It also handles a family the pair-of-tags model otherwise cannot: one where reasoning is a
    separate TURN rather than a span inside one. Muse-Glimmer renders
    ``<|start|>assistant to=self<|message|>R<|eom|><|start|>assistant<|message|>A<|eot|>``, which
    yields ``open=" to=self<|message|>"`` and ``close="<|eom|><|start|>assistant<|message|>"`` —
    delimiters that bracket the reasoning exactly as ``<think>``/``</think>`` do, even though
    neither is a tag.

    The opener has the GENERATION PROMPT stripped off its front, because the model's completion
    starts after that prompt: Muse-Glimmer's prompt already ends ``<|start|>assistant``, so the
    model itself only ever emits the `` to=self<|message|>`` part.
    """
    try:
        rendered = tokenizer.apply_chat_template(
            _HISTORY_PROBE, tokenize=False, add_generation_prompt=False
        )
    except Exception:  # noqa: BLE001 — a template that rejects the probe tells us nothing
        return None
    if not isinstance(rendered, str):
        return None
    ri, ai = rendered.find(_RSN_MARK), rendered.find(_ANS_MARK)
    if ri < 0 or ai <= ri:
        # No reasoning_content support (the marker never appeared), or the template emitted the
        # answer first. Either way there is no bracketing to read.
        return None
    close = rendered[ri + len(_RSN_MARK) : ai]
    head = rendered[:ri]
    try:
        genp = tokenizer.apply_chat_template(
            _PROBE_MESSAGES, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # noqa: BLE001
        genp = ""
    # Strip whatever the generation prompt already supplies. The two renders share the same system
    # block and the same user turn, so their common prefix IS the prompt the model conditions on.
    common = 0
    if isinstance(genp, str):
        limit = min(len(head), len(genp))
        while common < limit and head[common] == genp[common]:
            common += 1
    start = head[common:]
    # A delimiter has to be non-empty and carry real markup. Bail rather than return something
    # degenerate: `parse` treats an absent close tag as "no reasoning", which is the safe outcome,
    # whereas a junk close tag would split an answer in half.
    if not close.strip() or not start.strip():
        return None
    if len(close) > 200 or len(start) > 200:
        return None
    return start, close, "chat template (assistant turn carrying reasoning_content)"


_RECIPIENT_MARK = "ZQRECIPIENTZQ"


def derive_end_wildcard(tokenizer, end_token: str) -> Optional[Tuple[str, str]]:
    """Derive ``(prefix, suffix)`` bracketing a VARIABLE recipient inside the close delimiter, or None
    when the template's closer is a fixed literal (which is every ``<think>`` family).

    ``derive_delimiters_from_history`` renders one assistant turn and reads the closer off it — so on
    a channel-routed template it captures whatever recipient that probe defaulted to. Muse-Glimmer
    renders ``<|eom|><|start|>assistant to=user<|message|>``, and a reply that ends reasoning by
    routing to a TOOL emits ``… to=<toolname><|message|>`` instead: same delimiter, different middle.
    Matched literally, such a reply has no closer at all, so ``parse`` treats it as an unterminated
    reasoning span and files the entire thing — the tool call included — under ``reasoning_content``.

    Method: render the SAME history probe twice, once with the assistant turn's ``recipient`` left to
    the template's default and once with a marker recipient, then diff. The common prefix and common
    suffix bracket the recipient slot. A template that ignores ``recipient`` renders both identically
    and yields None, so this is inert for every family whose closer really is one literal (verified
    against every cached ``<think>`` checkpoint).

    Guarded: the derived bracket must actually reconstruct the concrete closer
    (``end_token.startswith(prefix)`` and ``endswith(suffix)``), or a template that varies something
    OTHER than a recipient could hand back a bracket that splits answers in half.
    """
    if not end_token:
        return None
    probe = [_HISTORY_PROBE[0], dict(_HISTORY_PROBE[1], recipient=_RECIPIENT_MARK)]
    try:
        plain = tokenizer.apply_chat_template(
            _HISTORY_PROBE, tokenize=False, add_generation_prompt=False
        )
        marked = tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
    except Exception:  # noqa: BLE001 — a template that rejects the probe tells us nothing
        return None
    if not isinstance(plain, str) or not isinstance(marked, str) or plain == marked:
        return None

    def _closer(rendered: str) -> Optional[str]:
        ri, ai = rendered.find(_RSN_MARK), rendered.find(_ANS_MARK)
        return rendered[ri + len(_RSN_MARK) : ai] if (ri >= 0 and ai > ri) else None

    a, b = _closer(plain), _closer(marked)
    if not a or not b or a == b or _RECIPIENT_MARK not in b:
        # No closer, or the recipient landed somewhere other than inside it — nothing to bracket.
        return None
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    prefix, suffix = a[:p], a[len(a) - s :]
    if not prefix.strip() or not suffix.strip():
        return None
    if not (end_token.startswith(prefix) and end_token.endswith(suffix)):
        return None
    return prefix, suffix


_ANSWER_PROBE = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": _ANS_MARK},
]


def derive_turn_header(tokenizer, start_token: str = "", end_token: str = "") -> str:
    """The literal a completion opens with before its ANSWER, or "" when there is none.

    Same technique as the opener derivation: render an assistant turn carrying only content, and
    subtract the generation prompt. Whatever is left is what the model has to emit itself before it
    can start answering.

    Almost every template ends its generation prompt at a turn boundary and so yields "". A
    channel-routed one does not: Muse-Glimmer's prompt stops at `<|start|>assistant` and the model
    writes ` to=user<|message|>` before the answer — markup that would otherwise be served as the
    first characters of `content`.

    Guarded two ways, because a false positive here EATS the head of an answer:
      * the result must contain markup (`<`), never bare prose;
      * it must not contain either reasoning delimiter. A template that renders an empty pre-closed
        think span into its assistant turns would otherwise hand back `<think></think>` as a
        "header", and stripping that would defeat the reasoning split entirely.
    """
    try:
        rendered = tokenizer.apply_chat_template(
            _ANSWER_PROBE, tokenize=False, add_generation_prompt=False
        )
        genp = tokenizer.apply_chat_template(
            _PROBE_MESSAGES, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # noqa: BLE001 — a template that rejects the probe tells us nothing
        return ""
    if not isinstance(rendered, str) or not isinstance(genp, str):
        return ""
    ai = rendered.find(_ANS_MARK)
    if ai < 0:
        return ""
    head = rendered[:ai]
    common = 0
    limit = min(len(head), len(genp))
    while common < limit and head[common] == genp[common]:
        common += 1
    header = head[common:]
    if not header or "<" not in header or len(header) > 100:
        return ""
    if (start_token and start_token in header) or (end_token and end_token in header):
        return ""
    return header


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
        derived = derive_delimiters(tokenizer) or derive_delimiters_from_history(tokenizer)
        if derived is not None:
            start, end, how = derived
            header = derive_turn_header(tokenizer, start, end)
            if header:
                how += f"; answer-turn header {header!r} stripped"
            wild = derive_end_wildcard(tokenizer, end)
            prefix, suffix = wild if wild is not None else ("", "")
            if wild is not None:
                how += f"; closer recipient is variable ({prefix!r} … {suffix!r})"
            return (
                ReasoningParser(start, end, header, prefix, suffix),
                f"derived from {how}",
            )
    if declared:
        parser = get_reasoning_parser(declared)
        if parser is not None:
            return parser, f"generation_config.json reasoning_parser={declared!r} (legacy table)"
    return ReasoningParser(*_LEGACY_GENERIC), "legacy generic <think>/</think> fallback"
