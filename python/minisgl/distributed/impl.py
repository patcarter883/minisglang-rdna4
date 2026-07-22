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
    # Optional custom_ar one-shot P2P all-reduce state (dp_size==2 + working P2P only). Populated by
    # enable_custom_ar_ep; None -> all_reduce stays on RCCL. See _EPCustomAR / all_reduce below.
    _car: "_EPCustomAR | None" = None

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
        car = self._car
        if car is not None:
            n = x.numel()
            # Custom one-shot P2P path (dp_size==2). Both replicas run the IDENTICAL model in lockstep
            # (the scheduler's per-step all_reduce(MAX) agreed the common bs/graph), so `partial` has the
            # SAME shape + contiguity on both ranks -> both take the SAME branch here every call. That is
            # what keeps the double-buffer slot counter and the fallback decision in sync across the two
            # SEPARATE replica processes (mismatched branches would deadlock: one waits on a peer flag the
            # other never bumps). Oversized (eager long prefill) or non-contiguous -> RCCL on BOTH ranks.
            if x.is_contiguous() and n * x.element_size() <= car.slot_bytes:
                slot = car.ctr & 1
                car.ctr += 1
                sb = car.self_data[slot].view(x.dtype)[:n]  # reinterpret the byte slot as x's dtype
                car.ops.one_shot_ar(x, x, sb, car.peer_data_ptr[slot], car.self_flags, car.peer_flags_ptr)
                return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.group)
        return x


@dataclass
class _EPCustomAR:
    """Per-EPCommunicator custom_ar one-shot all-reduce state (the DP/EP analogue of
    CustomARDistributedImpl's fields). Double-buffered: two byte slots + a per-call-site counter baked
    at graph capture keep back-to-back all_reduces from racing on one buffer (see the kernel comment)."""

    self_data: "torch.Tensor"     # [2, slot_bytes] uint8 fine-grained IPC (2 double-buffer slots)
    self_flags: "torch.Tensor"    # [>=BLOCKS] int32 fine-grained IPC
    peer_data_ptr: "list[int]"    # peer's 2 slot base pointers
    peer_flags_ptr: int
    slot_bytes: int
    ops: "object"
    ctr: int = 0


def enable_custom_ar_ep(
    ep: "EPCommunicator", dp_cpu_group: "torch.distributed.ProcessGroup", max_bytes: int
) -> None:
    """Install the custom_ar one-shot all-reduce on the EP (DP+EP) MoE all_reduce, if usable: exactly
    two DP replicas + working GPU-to-GPU P2P. Plumbing mirrors enable_custom_ar_distributed (the TP=2
    path) — that path is ALSO cross-process (each TP rank is its own process/card), so the only diffs
    are the CPU group carrying the IPC handshake (dp_cpu_group, the 2 DP replicas) and the peer index
    (ep.dp_rank). Falls back silently (keeps RCCL) otherwise. Called AFTER ctx.ep is built."""
    if ep.dp_size != 2:
        return
    try:
        import custom_ar as car
    except Exception:
        return
    dev = torch.cuda.current_device()
    peer_dev = 1 - dev
    if peer_dev < 0 or peer_dev >= torch.cuda.device_count() \
            or not torch.cuda.can_device_access_peer(dev, peer_dev):
        return  # no GPU-to-GPU P2P → keep RCCL (the cross-GPU flag handshake would deadlock)
    try:
        BLOCKS_SLACK = 64
        slot_bytes = ((max_bytes + 255) // 256) * 256
        self_data = car.alloc_shared(2 * slot_bytes, 0).view(2, slot_bytes)   # uint8 [2, slot_bytes]
        self_flags = car.alloc_shared(BLOCKS_SLACK * 4, 3)                     # int32 [64]

        def _exchange(buf):
            h = car.get_ipc_handle(buf)
            gathered = [None, None]
            dist.all_gather_object(gathered, h.numpy().tobytes(), group=dp_cpu_group)
            peer_bytes = torch.frombuffer(bytearray(gathered[1 - ep.dp_rank]), dtype=torch.uint8).clone()
            return car.open_ipc_handle(peer_bytes)

        peer_data_base = _exchange(self_data)
        peer_flags_ptr = _exchange(self_flags)
        peer_data_ptr = [peer_data_base, peer_data_base + slot_bytes]
        dist.barrier(group=dp_cpu_group)
    except Exception as e:  # noqa: BLE001
        from minisgl.utils import init_logger
        init_logger(__name__).info_rank0(f"custom_ar EP all-reduce unavailable ({e!r}) — using RCCL")
        return

    ep._car = _EPCustomAR(
        self_data=self_data, self_flags=self_flags, peer_data_ptr=peer_data_ptr,
        peer_flags_ptr=peer_flags_ptr, slot_bytes=slot_bytes, ops=car,
    )
    from minisgl.utils import init_logger
    init_logger(__name__).info_rank0("custom_ar EP one-shot all-reduce ENABLED (graph-safe, ~1.3x vs RCCL)")


@dataclass
class CustomARDistributedImpl(DistributedImpl):
    """Custom 2-GPU one-shot all-reduce (custom_ar) over PCIe P2P — a low-latency, GRAPH-CAPTURABLE
    drop-in for RCCL on the small TP=2 decode tensors (~1.3x faster; see custom_ar). all_gather is not
    provided here, so it stays on the previous plugin (RCCL); only all_reduce is overridden.

    DOUBLE-BUFFERED: back-to-back all_reduces (62+/forward) would race on a single shared buffer (the
    peer may still be reading slot K's data when call K+1 overwrites it). Two slots + a per-call counter
    alternate them; the intervening call's flag handshake guarantees the peer finished reading the slot
    two calls ago before it is reused. The counter advances only on eager calls / at graph CAPTURE — a
    captured graph bakes a fixed slot per call site, so replays are consistent across both ranks."""

    self_data: "torch.Tensor"      # [2, max_elems] uint8 fine-grained IPC (2 double-buffer slots)
    self_flags: "torch.Tensor"     # [>=BLOCKS] int32 fine-grained IPC
    peer_data_ptr: "list[int]"     # peer's 2 slot base pointers
    peer_flags_ptr: int
    slot_bytes: int
    _ops: "object"
    _ctr: int = 0

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info
        if get_tp_info().size == 1:
            return x
        n = x.numel()
        nb = n * x.element_size()
        if not x.is_contiguous() or nb > self.slot_bytes:
            # Oversized (rare: only long prefills) or non-contiguous → fall back to RCCL for this call.
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            return x
        slot = self._ctr & 1
        self._ctr += 1
        sb = self.self_data[slot].view(x.dtype)[:n]   # reinterpret the byte slot as x's dtype
        self._ops.one_shot_ar(x, x, sb, self.peer_data_ptr[slot], self.self_flags, self.peer_flags_ptr)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        # custom_ar has no all_gather; defer to RCCL (the historical plugin's behaviour).
        tp_size = self._tp_size_or_1()
        if tp_size == 1:
            return x
        shape = list(x.shape)
        shape[0] = shape[0] * tp_size
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous())
        return out

    @staticmethod
    def _tp_size_or_1() -> int:
        from .info import get_tp_info
        return get_tp_info().size


def enable_custom_ar_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """Install the custom_ar one-shot all-reduce as the active all_reduce plugin, if it is usable:
    TP==2 and the two GPUs have working P2P. Falls back silently (keeps RCCL) otherwise. Called by the
    engine AFTER the TP process group is up."""
    if tp_info.size != 2:
        return
    try:
        import custom_ar as car
    except Exception:
        return
    dev = torch.cuda.current_device()
    peer_dev = 1 - dev
    if peer_dev < 0 or peer_dev >= torch.cuda.device_count() \
            or not torch.cuda.can_device_access_peer(dev, peer_dev):
        return  # no GPU-to-GPU P2P → keep RCCL (the cross-GPU flag handshake would deadlock)
    try:
        BLOCKS_SLACK = 64
        slot_bytes = ((max_bytes + 255) // 256) * 256
        self_data = car.alloc_shared(2 * slot_bytes, 0).view(2, slot_bytes)   # uint8 [2, slot_bytes]
        self_flags = car.alloc_shared(BLOCKS_SLACK * 4, 3)                     # int32 [64]

        def _exchange(buf):
            h = car.get_ipc_handle(buf)
            gathered = [None, None]
            dist.all_gather_object(gathered, h.numpy().tobytes(), group=tp_cpu_group)
            peer_bytes = torch.frombuffer(bytearray(gathered[1 - tp_info.rank]), dtype=torch.uint8).clone()
            return car.open_ipc_handle(peer_bytes)

        # peer's two data-slot base pointers (the peer's slot i starts at base + i*slot_bytes)
        peer_data_base = _exchange(self_data)
        peer_flags_ptr = _exchange(self_flags)
        peer_data_ptr = [peer_data_base, peer_data_base + slot_bytes]
        dist.barrier(group=tp_cpu_group)
    except Exception as e:  # noqa: BLE001
        from minisgl.utils import init_logger
        init_logger(__name__).info_rank0(f"custom_ar all-reduce unavailable ({e!r}) — using RCCL")
        return

    DistributedCommunicator.plugins.append(
        CustomARDistributedImpl(
            self_data=self_data, self_flags=self_flags, peer_data_ptr=peer_data_ptr,
            peer_flags_ptr=peer_flags_ptr, slot_bytes=slot_bytes, _ops=car,
        )
    )
    from minisgl.utils import init_logger
    init_logger(__name__).info_rank0("custom_ar one-shot all-reduce ENABLED (graph-safe, ~1.3x vs RCCL)")


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
