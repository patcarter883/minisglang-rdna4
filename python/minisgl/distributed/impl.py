from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class TorchDistributedImpl(DistributedImpl):
    def _tp_size(self) -> int:
        # Gate on the TP size, NOT dist.get_world_size(): under DP the default WORLD group spans every
        # (dp_rank, tp_rank) process, so get_world_size() would be dp_size*tp_size and wrongly enable a
        # cross-replica reduce when each replica's TP is 1. get_tp_info().size is the per-replica TP
        # degree (== world size in the historical single-replica path, so behaviour is unchanged there).
        from .info import get_tp_info

        return get_tp_info().size

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self._tp_size() == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = self._tp_size()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x)
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info

        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)


@dataclass
class EPCommunicator:
    """Expert-parallel collectives over the DP/EP group (one member per replica's tp-primary).

    EP shards the MoE experts across the dp ranks: each rank owns experts
    ``[dp_rank*E/dp : (dp_rank+1)*E/dp]``. The MoE forward then (1) ``all_gather``s every replica's
    token rows + top-1 route so each rank sees ALL tokens, (2) runs the local-expert GEMM with the
    non-local tokens zeroed (weight 0), (3) ``all_reduce(SUM)``s the partial outputs — each token's
    top-1 expert lives on exactly one rank, so the sum is exact — and slices out its own rows.

    These two collectives are RCCL (CUDA) and run INSIDE the captured decode graph at fixed shapes,
    so the group MUST be an nccl-backed ``ProcessGroup`` (a gloo group is not CUDA-graph-capturable).
    The per-step common-bs agreement (which graph to replay) is a SEPARATE gloo all_reduce(MAX) done
    by the scheduler OUTSIDE the graph — see SchedulerEPMixin."""

    group: "torch.distributed.ProcessGroup"
    dp_rank: int
    dp_size: int
    num_experts: int  # GLOBAL expert count E (every replica owns E/dp_size of them)
    # Per-step common row count for the all_gather. The scheduler agrees it (all_reduce(MAX) of each
    # replica's real token count, on the gloo CPU group, OUTSIDE the graph) and sets it before the
    # forward. For DECODE the batch is already padded to a captured graph bs (equal N on every
    # replica), so this stays None and MoELayer all_gathers as-is. For EAGER PREFILL token counts
    # differ across replicas, so the scheduler sets pad_tokens = common N and MoELayer zero-pads its
    # rows up to it before the all_gather (the collective REQUIRES equal N), then slices back.
    pad_tokens: "int | None" = None

    @property
    def local_num_experts(self) -> int:
        return self.num_experts // self.dp_size

    @property
    def local_expert_offset(self) -> int:
        return self.dp_rank * self.local_num_experts

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate ``x`` (this replica's rows) across all dp ranks along dim 0 -> (dp_size*N, ...).
        Output is ordered by dp_rank, so rank r's own rows are the contiguous slice
        ``[r*N : (r+1)*N]`` (every replica replays the SAME agreed bs N under graph capture)."""
        shape = list(x.shape)
        shape[0] = shape[0] * self.dp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous(), group=self.group)
        return out

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)
        return x


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    if tp_info.size == 1:
        return
    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    DistributedCommunicator.plugins = []
