from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


# Per-token / hot IPC message types whose fields are all plain msgpack scalars (or, for
# `extra_tokens`, a flat list of ints). These messages cross a ZMQ hop twice per generated token
# (scheduler -> detokenizer as DetokenizeMsg, detokenizer -> frontend as UserReply), so the generic
# recursive `_serialize_any` / `_deserialize_any` object-graph walk is pure per-token overhead. For
# these we pack/unpack the fields DIRECTLY — no per-field function-call recursion, no `__dict__`
# comprehension, no cls_map walk. The wire format is byte-IDENTICAL to the recursive path (a dict
# with `__type__` + the same keys in dataclass-declaration order, with scalar values passed through
# unchanged and `extra_tokens` a plain list of ints), so it round-trips against the recursive
# fallback and composes inside a `Batch*.data` list exactly as before. Field order mirrors each
# dataclass's declaration so the emitted dict is identical to what the generic walk produced.
_FAST_SCALAR_FIELDS: Dict[str, tuple] = {
    "DetokenizeMsg": ("uid", "next_token", "finished", "extra_tokens", "finish_reason"),
    "UserReply": (
        "uid",
        "incremental_output",
        "finished",
        "completion_tokens",
        "prompt_tokens",
        "finish_reason",
    ),
    "StatsMsg": (
        "dp_rank",
        "spec_draft_tokens",
        "spec_accepted_tokens",
        "spec_emitted_tokens",
        "spec_steps",
        "running_requests",
        "waiting_requests",
        "kv_tokens_total",
        "kv_tokens_used",
        "gdn_slots_total",
        "gdn_slots_used",
        "prefix_cache_hit_tokens",
        "prefix_cache_prompt_tokens",
        "cam_facts",
        "cam_namespaces",
        "cam_evicted",
        "cam_max_bank_load",
        "cam_crowded_banks",
        "cam_recovered_from_backup",
        "cam_index_nn_cos_max",
        "cam_last_save_age_s",
    ),
    "StatsFrontendMsg": (
        "dp_rank",
        "spec_draft_tokens",
        "spec_accepted_tokens",
        "spec_emitted_tokens",
        "spec_steps",
        "running_requests",
        "waiting_requests",
        "kv_tokens_total",
        "kv_tokens_used",
        "gdn_slots_total",
        "gdn_slots_used",
        "prefix_cache_hit_tokens",
        "prefix_cache_prompt_tokens",
        "cam_facts",
        "cam_namespaces",
        "cam_evicted",
        "cam_max_bank_load",
        "cam_crowded_banks",
        "cam_recovered_from_backup",
        "cam_index_nn_cos_max",
        "cam_last_save_age_s",
    ),
}


def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = self.numpy().tobytes()
        serialized["dtype"] = str(self.dtype)
        return serialized

    type_name = self.__class__.__name__

    # Hot-path fast lane: flat-scalar messages pack their fields directly (see _FAST_SCALAR_FIELDS).
    # Every field is a plain scalar (or a flat list of ints), so no recursive walk is needed and the
    # emitted dict is byte-identical to the generic path below.
    fast_fields = _FAST_SCALAR_FIELDS.get(type_name)
    if fast_fields is not None:
        serialized["__type__"] = type_name
        for k in fast_fields:
            serialized[k] = getattr(self, k)
        return serialized

    # normal type
    serialized["__type__"] = type_name
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        return torch.from_numpy(np_tensor.copy())

    # Hot-path fast lane (mirror of serialize_type): flat-scalar messages read their fields straight
    # off the unpacked dict — the values are already plain scalars / a flat int list from msgpack, so
    # no per-field recursive rebuild is needed. Equivalent to the generic walk below for these types.
    fast_fields = _FAST_SCALAR_FIELDS.get(type_name)
    if fast_fields is not None:
        cls = cls_map[type_name]
        return cls(**{k: data[k] for k in fast_fields})

    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
