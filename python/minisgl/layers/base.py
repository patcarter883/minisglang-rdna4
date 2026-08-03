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


class BaseOP:
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
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                key = _concat_prefix(prefix, name)
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
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def post_load(self) -> None:
        """Recurse after load_state_dict, letting layers finalize weights (e.g. quantized
        layout conversion). Default: descend into sub-ops."""
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, BaseOP):
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
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def post_load(self) -> None:
        for op in self.op_list:
            op.post_load()
