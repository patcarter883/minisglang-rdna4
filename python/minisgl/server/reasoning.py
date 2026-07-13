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

Design: SPLIT ONLY WHEN THE CLOSING TAG IS PRESENT. A non-reasoning model (or a thinking-disabled
request, whose prompt already carries a *closed* ``<think></think>``) never emits ``</think>`` in
its output, so the parser is a safe no-op there — no model-name branch, no risk of misclassifying a
plain answer as reasoning. Delimiter sets are configurable so other families work generically.
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
}


class ReasoningParser:
    """Splits a completion into (reasoning_content, content) on a closing think tag."""

    def __init__(self, start_token: str = "<think>", end_token: str = "</think>") -> None:
        self.start_token = start_token
        self.end_token = end_token

    def parse(self, text: str) -> Tuple[Optional[str], str]:
        """Split ``text`` into ``(reasoning_content, content)``.

        When the closing tag is present: reasoning is everything before it (a leading opening tag,
        if the model echoed one, is stripped), content is everything after. When the closing tag is
        absent the text is returned unchanged as ``content`` with ``reasoning_content=None`` — safe
        for non-reasoning models and thinking-disabled requests.
        """
        if not text or self.end_token not in text:
            return None, text
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
