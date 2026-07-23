from __future__ import annotations

"""Reasoning-content extraction for chain-of-thought / "thinking" models.

Reasoning models (Qwen3, DeepSeek-R1, GLM-4.x, …) wrap their scratch reasoning in a delimiter
pair — canonically ``<think> … </think>`` — and emit the user-facing answer after the closing
tag. Crucially, several of these templates put the *opening* ``<think>`` in the generation PROMPT
(``…<|im_start|>assistant\n<think>\n``), so the model's OUTPUT contains only the CLOSING
``</think>`` followed by the answer. The raw completion therefore looks like::

    Here's my reasoning: 1. ... 2. ...
    </think>

    The answer is 4.

Without splitting, all of that reasoning prose lands in ``content``. This module splits on the
closing delimiter: everything before -> ``reasoning_content``, everything after -> ``content``.

Design: SPLIT ON THE CLOSING TAG WHEN PRESENT. When it is ABSENT, the disposition depends on
whether thinking was ON for the request (``thinking_open``):

* **Thinking OFF / non-reasoning model** (``thinking_open=False``): the prompt carries no open
  ``<think>`` (or a *closed* ``<think></think>``), so the output is a plain answer that never
  contains ``</think>``. Return it unchanged as ``content`` — a safe no-op, no misclassification.
* **Thinking ON** (``thinking_open=True``): the template injected the *opening* ``<think>`` into the
  prompt, so the model is INSIDE the reasoning span from token 0. If generation ends before it emits
  ``</think>`` — truncated at ``max_tokens``, or a model that just never closes — the ENTIRE output
  is scratch reasoning, NOT a user answer. Route it all to ``reasoning_content`` (content ``""``)
  rather than leaking a half-finished chain-of-thought into the visible answer.

No model-name branch; ``thinking_open`` comes from the request's thinking state. Delimiter sets are
configurable so other families work generically.
"""

from typing import Optional, Tuple

# family -> (open_token, close_token). All the mainstream open-weight reasoning models converge on
# the <think>…</think> pair today; add rows here for families that use different delimiters.
_REASONING_DELIMITERS = {
    "auto": ("<think>", "</think>"),
    "qwen3": ("<think>", "</think>"),
    "deepseek_r1": ("<think>", "</think>"),
    "deepseek-r1": ("<think>", "</think>"),
    "glm": ("<think>", "</think>"),
    "qwen": ("<think>", "</think>"),
    # Poolside/Laguna: the turn is wrapped `<assistant><think>REASONING</think>ANSWER</assistant>`;
    # the generation prompt injects `<assistant><think>` so the OUTPUT is `REASONING</think>ANSWER`
    # (the `<assistant>` opener is prompt-side; `</assistant>` is the eos, trimmed). So the reasoning
    # split is the same `<think>/</think>` pair — register the model's declared parser name so the
    # server selects it explicitly (and treats the format as supported → thinking stays on).
    "poolside_v1": ("<think>", "</think>"),
    "poolside": ("<think>", "</think>"),
}


class ReasoningParser:
    """Splits a completion into (reasoning_content, content) on a closing think tag."""

    def __init__(self, start_token: str = "<think>", end_token: str = "</think>") -> None:
        self.start_token = start_token
        self.end_token = end_token

    def parse(self, text: str, thinking_open: bool = False) -> Tuple[Optional[str], str]:
        """Split ``text`` into ``(reasoning_content, content)``.

        When the closing tag is present: reasoning is everything before it (a leading opening tag,
        if the model echoed one, is stripped), content is everything after.

        When the closing tag is ABSENT the disposition depends on ``thinking_open`` (whether the
        request had thinking enabled, i.e. the prompt injected an unclosed opening ``<think>``):

        * ``thinking_open=False`` (default, thinking off / non-reasoning): returned unchanged as
          ``content`` with ``reasoning_content=None`` — a safe no-op.
        * ``thinking_open=True``: the model was inside the reasoning span from token 0 and generation
          ended before it emitted the closing tag (truncated at ``max_tokens``, or a model that never
          closes), so the WHOLE output is reasoning — returned as ``reasoning_content`` with
          ``content=""``, never leaking a half-finished chain-of-thought into the answer.
        """
        if not text:
            return None, text
        if self.end_token not in text:
            if not thinking_open:
                return None, text
            # Thinking was on and the model never closed </think> -> it's all reasoning.
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
        return ReasoningStreamState(self.end_token, active)


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
    so ``</think>`` split across two chunks is still detected."""

    def __init__(self, end_token: str, active: bool) -> None:
        self.end_token = end_token
        self.active = active  # currently inside the reasoning span
        self.pending = ""     # held-back tail that might be a partial end token
        self._content_started = False  # have we emitted any answer text yet

    def _emit_content(self, text: str) -> Optional[str]:
        """Drop the template's leading `\\n\\n` separator on the first answer chunk (it may straddle
        the chunk that closed </think> and the next), then pass content through verbatim."""
        if not self._content_started:
            text = text.lstrip("\n")
            if not text:
                return None
            self._content_started = True
        return text or None

    def push(self, delta: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.active:
            return None, self._emit_content(delta)
        text = self.pending + delta
        idx = text.find(self.end_token)
        if idx != -1:
            reasoning = text[:idx]
            self.active = False
            self.pending = ""
            return (reasoning or None), self._emit_content(text[idx + len(self.end_token):])
        keep = _end_overlap_len(text, self.end_token)
        if keep:
            self.pending = text[-keep:]
            emit = text[:-keep]
        else:
            self.pending = ""
            emit = text
        return (emit or None), None

    def flush(self) -> Optional[str]:
        """Emit any buffered tail at stream end (still inside reasoning => it was reasoning text)."""
        if self.active and self.pending:
            out, self.pending = self.pending, ""
            return out or None
        return None


def get_reasoning_parser(name: Optional[str]) -> Optional[ReasoningParser]:
    """Build a parser for a ``--reasoning-parser`` value; None disables reasoning extraction."""
    if not name or name == "none":
        return None
    delims = _REASONING_DELIMITERS.get(name.lower())
    if delims is None:
        return None
    return ReasoningParser(*delims)
