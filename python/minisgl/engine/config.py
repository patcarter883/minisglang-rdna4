from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config, is_rocm

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    # PyNCCL is a CUDA-only collective: pynccl.cu pulls in NVIDIA NCCL (nccl227.h / -lnccl, not RCCL)
    # and loads via apache-tvm-ffi. Neither is present on ROCm, so default it OFF there — tp>1 then
    # falls back to torch.distributed backend="nccl" (→ RCCL on ROCm) in engine._init_communication.
    # default_factory (not a plain `= not is_rocm()`) so the arch is probed at construct time, not
    # import time. The CLI path sets its own ROCm-aware default in server/args.py (argparse always
    # supplies use_pynccl, so this dataclass default only governs programmatic EngineConfig builds).
    use_pynccl: bool = field(default_factory=lambda: not is_rocm())
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
