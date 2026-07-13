from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

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
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tools=getattr(msg, "tools", None),
                    tokenize=False,
                    add_generation_prompt=True,
                    **(getattr(msg, "chat_template_kwargs", None) or {}),
                )
                assert isinstance(prompt, str)
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
