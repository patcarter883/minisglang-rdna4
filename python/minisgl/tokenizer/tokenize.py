from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
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
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results
