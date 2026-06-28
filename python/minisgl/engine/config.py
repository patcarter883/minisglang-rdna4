from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo, DpInfo
from minisgl.utils import cached_load_hf_config, is_rocm

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    # Data-parallel coordinates for this engine replica. Defaults to the inert single-replica DP
    # (dp_rank=0, dp_size=1) so every existing programmatic EngineConfig build is unchanged. The
    # server launcher overrides it per spawned replica. `enable_ep` is reserved for the expert-parallel
    # toggle (shards MoE experts across dp ranks); it stays False / inert in the DP-launcher-only path.
    dp_info: DpInfo = field(default_factory=lambda: DpInfo(0, 1))
    enable_ep: bool = False
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
    # --- speculative decoding (off by default; see SPEC_DECODE.md) ------------------------------
    # "none" disables every spec path (byte-for-byte unchanged serve). "ngram" enables the
    # prompt-lookup MVP. These flat fields mirror the argparse dests; spec_config assembles them.
    spec_algorithm: str = "none"
    spec_num_draft: int = 4
    spec_ngram_max: int = 3
    spec_ngram_min: int = 1
    spec_draft_model_path: str | None = None  # EAGLE3/DFlash: separate draft checkpoint path

    @cached_property
    def spec_config(self):
        from minisgl.spec import SpecConfig

        if self.spec_algorithm == "none":
            return None
        return SpecConfig(
            algorithm=self.spec_algorithm,
            num_draft=self.spec_num_draft,
            ngram_max=self.spec_ngram_max,
            ngram_min=self.spec_ngram_min,
            draft_model_path=self.spec_draft_model_path,
        )

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
    def device_index(self) -> int:
        """Physical card slot for this replica's TP rank, within the lease-visible device set.

        The gpu-lease/HIP_VISIBLE_DEVICES exposes the leased cards as cuda:0..N-1; this picks the
        slot for (dp_rank, tp_rank). With dp_size=1 this collapses to tp_info.rank — the historical
        `cuda:{tp_rank}` mapping — so single-replica runs are unchanged. With dp_size>1 (tp_size=1
        for ZAYA) each replica lands on its own card: dp_rank=0 -> cuda:0, dp_rank=1 -> cuda:1.
        """
        return self.dp_info.dp_rank * self.tp_info.size + self.tp_info.rank

    @property
    def distributed_addr(self) -> str:
        # Each DP replica is an INDEPENDENT TP process group (with EP off there is no cross-replica
        # collective), so give each replica its own rendezvous port to avoid init_method collisions
        # when several replicas come up on one host. dp_size=1 keeps the historical 2333.
        return f"tcp://127.0.0.1:{2333 + self.dp_info.dp_rank}"
