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
            else:
                prompt = msg.text
            prompts.append(prompt)
        if not prompts:
            return []
        # ONE batched encode of all prompt strings (no padding / no return_tensors -> per-prompt
        # id lists, identical to calling tokenizer.encode(prompt) on each; add_special_tokens keeps
        # the encode() default of True). Each result -> the same flat int32 tensor as before.
        encoded = self.tokenizer(prompts)["input_ids"]
        return [torch.tensor(ids, dtype=torch.int32) for ids in encoded]
