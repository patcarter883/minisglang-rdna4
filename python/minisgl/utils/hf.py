import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

# ZAYA1-base's config.json carries `rope_scaling: false`; huggingface_hub 1.x renames it to
# `rope_parameters` and STRICT-validates it, so the bool raises StrictDataclassFieldValidationError
# in AutoConfig.from_pretrained (transformers 5.x DOES register "zaya", so it builds the real config
# and validates). Fall back to the sanitized generic-config path below when that fires.
_CONFIG_FALLBACK_ERRORS: tuple = (ValueError, KeyError)
try:
    from huggingface_hub.errors import StrictDataclassFieldValidationError as _HFStrictErr
    _CONFIG_FALLBACK_ERRORS = (ValueError, KeyError, _HFStrictErr)
except Exception:
    pass


def _sanitize_zaya_rope(cfg_dict: dict) -> dict:
    """Pop the `rope_scaling: false` wart and write the proper `rope_parameters` dict (mirrors the
    megatron eval sanitize). No-op once already sanitized. Idempotent, in-place-safe on a copy."""
    if isinstance(cfg_dict.get("rope_parameters"), dict) and "rope_scaling" not in cfg_dict:
        return cfg_dict
    if cfg_dict.get("rope_scaling") is not False and cfg_dict.get("rope_parameters") not in (None, False):
        return cfg_dict
    cfg_dict.pop("rope_scaling", None)
    theta = float(cfg_dict.get("rope_theta", 5000000.0))
    cfg_dict["rope_parameters"] = {
        "hybrid": {"rope_type": "default", "rope_theta": theta, "partial_rotary_factor": 0.5},
        "hybrid_sliding": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        "rope_type": "default",
    }
    return cfg_dict

class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    try:
        return AutoConfig.from_pretrained(model_path)
    except _CONFIG_FALLBACK_ERRORS:
        # Models whose `model_type` the installed transformers does not register, OR whose config
        # trips strict validation (ZAYA's `rope_scaling: false` wart), raise here. minisgl reads
        # config fields via getattr, so a generic PretrainedConfig built directly from a sanitized
        # config.json is sufficient — we never need the transformers model class itself.
        cfg_path = (
            os.path.join(model_path, "config.json")
            if os.path.isdir(model_path)
            else hf_hub_download(repo_id=model_path, filename="config.json")
        )
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = _sanitize_zaya_rope(json.load(f))
        return PretrainedConfig.from_dict(cfg_dict)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    fresh = type(config)(**config.to_dict())
    # PretrainedConfig.__init__ does not accept `model_type` as a kwarg (it is a class attribute),
    # so the round-trip above resets it to the base class default ("") for configs loaded via the
    # generic PretrainedConfig fallback (unknown architectures like ZAYA's "zaya"). Restore it so
    # model_type-gated logic (e.g. ModelConfig.is_cca) sees the real type.
    fresh.model_type = config.model_type
    return fresh


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )