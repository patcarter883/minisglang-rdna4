from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


# Narrow float dtypes a checkpoint may legitimately store a WIDE (real-valued) tensor in, and that
# a layer may legitimately declare. A tensor whose dtype is in this set carries an ordinary real
# number, so re-encoding it in another member of the set is a value-preserving-to-rounding cast —
# the same thing `nn.Module.load_state_dict` does via `param.copy_(input_param)`.
#
# Everything NOT in this set is a STORAGE ENCODING, not a number, and must never be cast:
#   * int32 / uint8 packs (AWQ/GPTQ `qweight`/`qzeros`, compressed-tensors `weight_packed`,
#     `weight_zero_point`, MXFP4/NVFP4 `weight_packed`) hold several sub-byte quanta per element —
#     a dtype cast reinterprets the bit pattern as a scalar and destroys the weight.
#   * uint8 E8M0 block scales (MXFP4 `weight_scale`) are raw exponents, not floats.
#   * float8_e4m3fn (ZAYA experts) IS floating point, but upcasting it at load re-inflates ~8 GB to
#     ~16 GB and OOMs a 16 GB card — the storage dtype is a deliberate memory decision (see
#     engine._cast / _GroupedFP8Experts, which dequant at COMPUTE time). Excluded on purpose.
_CASTABLE_FLOAT_DTYPES = frozenset(
    {torch.float32, torch.float64, torch.bfloat16, torch.float16}
)


def _coerce_dtype(key: str, param: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
    """Adapt a checkpoint tensor whose stored dtype differs from the layer's declared dtype.

    A loader must load any checkpoint of an architecture it supports, so a WIDE tensor stored as
    fp16 where the layer declares bf16 (or vice versa) is a cast, not a refusal — checkpoints of the
    same architecture ship both (e.g. `cyankiwi/Agents-A1-AWQ-INT4` is fp16 throughout where
    `Qwen3.6-35B-A3B-AWQ-4bit` is bf16). QUANTIZED tensors are the exception and still hard-fail:
    their dtype is a packing/encoding contract, not a numeric precision (see
    `_CASTABLE_FLOAT_DTYPES`)."""
    if param.dtype in _CASTABLE_FLOAT_DTYPES and item.dtype in _CASTABLE_FLOAT_DTYPES:
        return item.to(param.dtype)
    raise AssertionError(
        f"weight dtype mismatch for {key!r}: model {param.dtype} vs checkpoint {item.dtype}. "
        f"At least one is a packed/quantized storage dtype, which cannot be cast — the layer's "
        f"declared buffer must match the checkpoint's encoding exactly."
    )


def load_nn_bridge_state(
    module: Any,
    state_dict: _STATE_DICT,
    prefix: str,
    *,
    missing_ok: bool = False,
) -> bool:
    """Fill a wrapped `nn.Module` from a slice of a `BaseOP` state dict. Returns True if it filled.

    Three ops wrap a real `nn.Module` and bridge its `_parameters` into the BaseOP state grammar
    (`Qwen3_5LinearAttn._gdn`, `ZayaCCAAttn._cca`, `ZayaMoEBlock.router`) with three byte-identical
    copies of this body. One implementation, because `missing_ok` has to be threaded through all
    three and a bridge that quietly kept the strict behaviour would defeat the chunked load on
    exactly one model family.

    A bridged module is ATOMIC under `missing_ok`: all of its keys are present or none are. A chunk
    that carries half of one is a chunk-boundary bug (the driver's chunks are whole layers or the
    whole non-expert body, and no bridged module straddles those), and filling the half we have
    would leave the rest silently at its `torch.empty` garbage — `strict=True` below cannot see it,
    because the sub-dict it is handed would be complete-looking.
    """
    names = list(module.state_dict())
    keys = [_concat_prefix(prefix, n) for n in names]
    if missing_ok:
        present = [k in state_dict for k in keys]
        if not any(present):
            return False
        if not all(present):
            raise KeyError(
                f"chunked load split a bridged nn.Module at {prefix!r}: "
                f"{sum(present)}/{len(keys)} of its keys are in this chunk. A bridged module is "
                f"loaded atomically (strict=True over the whole sub-dict), so a partial chunk "
                f"would leave the remainder uninitialized with nothing to detect it. Missing: "
                f"{[k for k, p in zip(keys, present) if not p]}"
            )
    sub = {n: state_dict.pop(k) for n, k in zip(names, keys)}
    missing, unexpected = module.load_state_dict(sub, strict=True, assign=True)
    assert not missing and not unexpected, (missing, unexpected)
    return True


class BaseOP:
    # STAGE B (chunked load). Set on an op whose `post_load()` has ALREADY run, so the whole-model
    # `post_load()` that follows a chunked load does not run it a second time. A class attribute, so
    # it stays out of `vars(self)` and is therefore invisible to `state_dict`, `load_state_dict` and
    # the weights/granule walk (same reason `MoELayer._weight_offload` is a class attribute).
    #
    # It is NOT a "post_load has run" bookkeeping flag for general use: a second `post_load()` on a
    # quantized MoE container is not idempotent, it is an AttributeError (`post_load` deletes the
    # checkpoint buffers it read), and on a container that repacks in place it would be silent
    # corruption. The flag exists so the chunked loader — which must finalize each layer the moment
    # its bytes arrive, or it holds the whole checkpoint at once — can hand the model back to the
    # ordinary `model.post_load()` without that call re-entering the layers it already finalized.
    _post_load_done: bool = False

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
        missing_ok: bool = False,
    ) -> None:
        """Fill this op's tensors from `state_dict`, consuming (popping) what it uses.

        `missing_ok=True` is STAGE B's partial fill: a key this op wants but the dict does not carry
        leaves the existing buffer alone instead of raising `KeyError`. That is what lets one
        checkpoint CHUNK be applied to the whole model — the 84 GB target checkpoint cannot be
        materialized as one `state_dict`, and the alternative (walking the module tree here to build
        a path->(owner, attr) index) would be a SECOND implementation of this traversal, which the
        four overrides below already show is not a traversal with one implementation.

        It is deliberately not a "strict=False": nothing is silently tolerated, because the CALLER
        owns the totality ledger. `weights.stage_b.ChunkedWeightLoader` snapshots the model's key set
        BEFORE the first chunk, subtracts what each chunk consumed (whatever the chunk dict no longer
        holds on return), and refuses at the end if any key was never filled or was filled twice. A
        missing key is therefore an error exactly as before — just raised by the driver, once, naming
        every key, instead of by a `KeyError` in the middle of the first chunk.
        """
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                key = _concat_prefix(prefix, name)
                if missing_ok and key not in state_dict:
                    continue
                item = state_dict.pop(key)
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape, (
                    f"weight shape mismatch for {key!r}: model {tuple(param.shape)} "
                    f"vs checkpoint {tuple(item.shape)}"
                )
                if param.dtype != item.dtype:
                    item = _coerce_dtype(key, param, item)
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict,
                    prefix=_concat_prefix(prefix, name),
                    _internal=True,
                    missing_ok=missing_ok,
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def post_load(self) -> None:
        """Recurse after load_state_dict, letting layers finalize weights (e.g. quantized
        layout conversion). Default: descend into sub-ops.

        Skips a sub-op already finalized by the chunked loader — see `_post_load_done`."""
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, BaseOP) and not param._post_load_done:
                param.post_load()


class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
        missing_ok: bool = False,
    ) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
        missing_ok: bool = False,
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(
                state_dict,
                prefix=_concat_prefix(prefix, str(i)),
                _internal=True,
                missing_ok=missing_ok,
            )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def post_load(self) -> None:
        for op in self.op_list:
            if not op._post_load_done:
                op.post_load()
