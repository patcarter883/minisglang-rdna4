"""Structured-output (constrained decoding) backend, built on xgrammar.

A constrained request carries a grammar spec in its ``SamplingParams.grammar``:
  * ``"json"``      -> any syntactically valid JSON value (builtin JSON grammar)
  * a JSON-schema   -> output must conform to the schema (passed as a JSON string)

Per request the scheduler keeps a stateful ``GrammarMatcher``: before each sample it fills a
per-row *token bitmask* of the currently-allowed next tokens; after a token is committed it advances
the matcher. The bitmask is applied to the logits in the Sampler (``engine/sample.py``) — disallowed
tokens are set to ``-inf`` so the constraint holds for greedy AND sampled decoding.

xgrammar is imported lazily (only when the first constrained request arrives), so a plain serve pays
nothing and the dependency is optional. Speculative decoding is bypassed for constrained requests
(the draft chain would have to satisfy the grammar too — a later refinement), so masking happens on
the plain prefill+decode path.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class GrammarBackend:
    """Owns the xgrammar compiler (built once from the tokenizer) and mints per-request matchers."""

    def __init__(self, tokenizer: "PreTrainedTokenizerBase", vocab_size: int) -> None:
        import xgrammar as xgr

        self._xgr = xgr
        self.vocab_size = vocab_size
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        self._compiler = xgr.GrammarCompiler(tokenizer_info)
        # full vocab incl. padding the bitmask is sized to (>= model vocab_size)
        self.full_vocab_size = tokenizer_info.vocab_size

    def make_matcher(self, spec: str):
        """Compile ``spec`` and return a fresh stateful GrammarMatcher. ``spec`` == "json" -> any JSON
        object; a ``{"__structural_tag__": …}`` blob -> an xgrammar structural tag (constrain to a
        schema ONLY after a trigger string, else free text — the `tool_choice:"auto"` case); otherwise
        ``spec`` is a JSON-schema string."""
        xgr = self._xgr
        # Structural tag: free generation until the model emits one of the trigger strings (a tool-call
        # opener), then the wrapped content is constrained to the tool schema. Lets the model choose to
        # call or not, but forces schema-valid arguments when it does.
        if spec.startswith('{"__structural_tag__"'):
            import json as _json

            st = _json.loads(spec)["__structural_tag__"]
            tags = [
                xgr.StructuralTagItem(begin=t["begin"], schema=t["schema"], end=t["end"])
                for t in st["tags"]
            ]
            compiled = self._compiler.compile_structural_tag(tags, st["triggers"])
            return xgr.GrammarMatcher(compiled)
        # any_whitespace=False -> compact JSON: forbids the unbounded-whitespace runs a greedy (temp 0)
        # model otherwise stalls in, so the value completes within max_tokens. json_object maps to "any
        # JSON object"; the builtin grammar is avoided because it has no whitespace bound and loops the
        # same way.
        schema = '{"type": "object"}' if spec == "json" else spec
        compiled = self._compiler.compile_json_schema(schema, any_whitespace=False)
        return xgr.GrammarMatcher(compiled)

    def allocate_bitmask(self, batch_size: int) -> torch.Tensor:
        """CPU int32 bitmask [batch_size, ceil(full_vocab/32)]; bit==1 means the token is ALLOWED."""
        return self._xgr.allocate_token_bitmask(batch_size, self.full_vocab_size)


def apply_token_bitmask(logits: torch.Tensor, bitmask: torch.Tensor) -> torch.Tensor:
    """Mask ``logits`` ([bs, vocab]) with a packed xgrammar ``bitmask`` ([bs, ceil(vocab/32)] int32 on
    any device): tokens whose bit is 0 are set to -inf. Device-agnostic torch unpack (no xgrammar CUDA
    kernel dependency — important on ROCm). Rows meant to be unconstrained must arrive all-ones.
    """
    bs, vocab = logits.shape
    device = logits.device
    bm = bitmask.to(device)
    # The bitmask is built for the PADDED batch (len(batch.padded_reqs) — includes CUDA-graph dummy
    # rows), but `logits` holds only the ACTUAL sampled rows, so bm can have MORE rows than bs (e.g.
    # a graph-padded 128-row bitmask vs a 3-row eager prefill sample). Padding is appended, so align
    # by taking the first bs rows — otherwise bits.reshape(bs, -1) fails when the padded element count
    # isn't divisible by bs (the '[3, -1]' invalid-shape crash on multi-request constrained batches).
    if bm.shape[0] != bs:
        bm = bm[:bs]
    shifts = torch.arange(32, device=device, dtype=torch.int32)
    # [bs, n32, 32] -> [bs, n32*32] -> [bs, vocab]; bit i of word w is token w*32+i.
    bits = (bm.unsqueeze(-1) >> shifts) & 1
    allowed = bits.reshape(bs, -1)[:, :vocab].to(torch.bool)
    return logits.masked_fill(~allowed, float("-inf"))


__all__ = ["GrammarBackend", "apply_token_bitmask"]
