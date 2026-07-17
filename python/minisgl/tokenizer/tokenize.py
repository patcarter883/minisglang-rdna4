from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


import logging

_logger = logging.getLogger("minisgl.tokenize")


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def _render_chat(self, msg) -> str:
        """Render a chat-messages list through the model's template. A malformed message list (e.g. a
        second `system` turn a strict template rejects — "System message must be at the beginning") must
        NEVER crash the tokenizer worker and take the whole serve down. So: try the template; on failure
        retry with every system turn coalesced into one leading turn (fixes the common cause); and if it
        still fails, fall back to a plain role-tagged render so the request degrades instead of the serve."""
        kwargs = dict(
            tools=getattr(msg, "tools", None), tokenize=False, add_generation_prompt=True,
            **(getattr(msg, "chat_template_kwargs", None) or {}),
        )
        try:
            out = self.tokenizer.apply_chat_template(msg.text, **kwargs)
            assert isinstance(out, str)
            return out
        except Exception as e:  # noqa: BLE001 — a bad request must not kill the worker
            try:
                out = self.tokenizer.apply_chat_template(self._coalesce_system(msg.text), **kwargs)
                assert isinstance(out, str)
                _logger.warning("chat template rejected raw messages (%s); recovered by coalescing system turns", e)
                return out
            except Exception:  # noqa: BLE001
                _logger.warning("chat template failed (%s); using plain fallback render", e)
                return self._plain_render(msg.text)

    @staticmethod
    def _coalesce_system(messages):
        """Merge every system message into one leading system turn (templates want at most one, first)."""
        sys = [str(m.get("content") or "") for m in messages if isinstance(m, dict) and m.get("role") == "system"]
        rest = [m for m in messages if not (isinstance(m, dict) and m.get("role") == "system")]
        head = [{"role": "system", "content": "\n\n".join(sys)}] if sys else []
        return [*head, *rest]

    @staticmethod
    def _plain_render(messages):
        lines = [f"{m.get('role', 'user')}: {m.get('content') or ''}" for m in messages if isinstance(m, dict)]
        return "\n".join(lines) + "\nassistant:"

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        # Chat-template RENDERING stays per-msg (tools / chat_template_kwargs differ per request),
        # but the final encode of the rendered prompt STRINGS is batched into one tokenizer call.
        prompts: List[str] = []
        # A chat-templated string ALREADY carries every special token the model needs (BOS, role/turn
        # markers) rendered by the jinja template; a raw `prompt` string does NOT. Track which is which
        # so the encode below doesn't re-add specials to the templated ones (see the add_special note).
        templated: List[bool] = []
        for msg in msgs:
            if isinstance(msg.text, list):
                # `tools` (when set) render the tool specs into the chat template for tool-trained
                # models; None is the transformers default (no tools) and is safe for every template.
                # `chat_template_kwargs` (e.g. {"enable_thinking": False}) is forwarded verbatim so
                # a request can control reasoning-model thinking mode. None -> template defaults
                # (Qwen3 opens `<think>` in the generation prompt, i.e. thinking ON).
                prompt = self._render_chat(msg)
                templated.append(True)
            else:
                prompt = msg.text
                templated.append(False)
            prompts.append(prompt)
        if not prompts:
            return []
        # add_special_tokens MUST be False for chat-templated strings: the template already rendered
        # <bos>/role markers as text, so encode()'s default add_special_tokens=True DOUBLE-adds them.
        # On a Gemma-family tokenizer with a `<bos> $A <|im_end|>` post-processor this prepends a 2nd
        # <bos> (degenerates generation) AND appends a spurious <|im_end|> right after the generation
        # prompt — the model reads its turn as already closed and emits fresh role markup ("template
        # garbage"). Raw `prompt` strings, by contrast, DO want the tokenizer's leading specials.
        # Batch when the whole request set is uniform (the common all-chat / all-raw case); only the
        # rare mixed batch pays a per-prompt encode.
        if all(templated):
            encoded = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        elif not any(templated):
            encoded = self.tokenizer(prompts, add_special_tokens=True)["input_ids"]
        else:
            encoded = [
                self.tokenizer(p, add_special_tokens=not t)["input_ids"]
                for p, t in zip(prompts, templated)
            ]
        return [torch.tensor(ids, dtype=torch.int32) for ids in encoded]
