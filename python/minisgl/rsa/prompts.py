"""Aggregation prompt templates for Markovian RSA.

Kept deliberately minimal, following the RSA paper's (arXiv:2509.26626)
note that the method works without extensive prompt engineering.
"""

from __future__ import annotations

import re
from typing import List, Optional

AGGREGATION_SYSTEM = (
    "You are given a problem and several candidate solutions. Some candidates "
    "may be incorrect or contain errors. Examine the candidate solutions and "
    "produce an improved, higher-quality solution to the problem. Reason "
    "carefully; if all candidates are flawed, solve the problem from scratch."
)

AGGREGATION_USER = """\
{query}

Below are {k} candidate solutions:

{candidates}

Examine these candidates and produce an improved, complete solution to the \
problem above. End with your final answer."""

FINAL_SELECTION_SYSTEM = (
    "You are given a problem and several candidate solutions. Select or "
    "synthesize the single best final answer."
)

FINAL_SELECTION_USER = """\
{query}

Below are {k} candidate solutions:

{candidates}

Select or synthesize the single best final answer to the problem above. \
Respond with only the final answer."""

# Tool-calling variant. The plain FINAL_SELECTION prompt ("respond with only the final answer")
# elicits PROSE, so a tool-augmented request would fabricate an answer instead of calling the tool
# (the rollout candidates are generated tools-free, so they can't be trusted for tool-dependent facts).
# This variant reframes the final generation as an ACTION: call the tool when the request needs it.
TOOL_SELECTION_SYSTEM = (
    "You are given a request, a set of available tools, and several candidate solutions. The "
    "candidates were written WITHOUT access to the tools, so any facts they assert that would "
    "require a tool may be fabricated. Determine the single best final response to the request. If "
    "fulfilling it needs information or actions a tool provides, you MUST call the appropriate tool "
    "using the provided interface rather than guessing. Answer directly only when no tool is needed."
)

TOOL_SELECTION_USER = """\
{query}

Below are {k} candidate solutions (written without tool access):

{candidates}

Produce the single best final response to the request above. If a tool is required to answer \
correctly, call it; otherwise respond with only the final answer."""

TRUNCATION_MARKER = "[...truncated...]\n"


def _fill(template: str, **fields: object) -> str:
    """Substitute ``{name}`` placeholders WITHOUT ``str.format``.

    The query/candidate text routinely contains literal braces (LaTeX ``\\boxed{...}``,
    ``\\frac{a}{b}``, JSON, code); ``str.format`` would parse those as format fields and raise
    ``KeyError``/``ValueError`` on essentially every real math/code request. This does a single
    left-to-right pass that replaces ONLY the exact known placeholder tokens and never re-scans
    substituted content, so braces inside the values are emitted verbatim.
    """
    pattern = re.compile(r"\{(" + "|".join(re.escape(k) for k in fields) + r")\}")
    return pattern.sub(lambda m: str(fields[m.group(1)]), template)


def _content_text(message: dict) -> str:
    """Extract plain text from a message's content (str or content parts)."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    # OpenAI content-parts format: [{"type": "text", "text": ...}, ...]
    return "\n".join(
        part.get("text", "") for part in content if part.get("type") == "text"
    )


def render_query(messages: List[dict]) -> str:
    """Render the user-visible conversation into a single query string.

    For the common single-user-message case this is just that message's text.
    Multi-turn conversations render as a ``User:/Assistant:`` transcript.
    System messages are handled separately (see build_aggregation_messages).
    """
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(turns) == 1:
        return _content_text(turns[0])
    parts = []
    for m in turns:
        label = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{label}: {_content_text(m)}")
    return "\n\n".join(parts)


def extract_request_system(messages: List[dict]) -> Optional[str]:
    """Concatenate any system/developer messages from the incoming request."""
    parts = [
        _content_text(m)
        for m in messages
        if m.get("role") in ("system", "developer")
    ]
    joined = "\n\n".join(p for p in parts if p)
    return joined or None


def render_candidates(tails: List[str]) -> str:
    blocks = []
    for i, tail in enumerate(tails, start=1):
        blocks.append(f"=== Candidate {i} ===\n{tail}")
    return "\n\n".join(blocks)


def _build(
    system_template: str,
    user_template: str,
    query: str,
    candidate_tails: List[str],
    request_system: Optional[str],
) -> List[dict]:
    system = system_template
    if request_system:
        system = request_system.rstrip() + "\n\n" + system
    user = _fill(
        user_template,
        query=query,
        k=len(candidate_tails),
        candidates=render_candidates(candidate_tails),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_aggregation_messages(
    query: str,
    candidate_tails: List[str],
    request_system: Optional[str] = None,
) -> List[dict]:
    return _build(
        AGGREGATION_SYSTEM, AGGREGATION_USER, query, candidate_tails, request_system
    )


def build_final_selection_messages(
    query: str,
    candidate_tails: List[str],
    request_system: Optional[str] = None,
    *,
    for_tools: bool = False,
) -> List[dict]:
    # A tool-augmented request needs an ACTION-framed prompt so the model emits a tool call instead of
    # fabricating a prose answer from the (tools-free) candidates; everything else uses the plain
    # "pick the best answer" framing.
    system = TOOL_SELECTION_SYSTEM if for_tools else FINAL_SELECTION_SYSTEM
    user = TOOL_SELECTION_USER if for_tools else FINAL_SELECTION_USER
    return _build(system, user, query, candidate_tails, request_system)
