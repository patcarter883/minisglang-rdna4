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
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except _CONFIG_FALLBACK_ERRORS:
        # ZAYA rope_scaling:false wart: AutoTokenizer builds the strict model config internally and
        # trips validation. Pass a pre-sanitized config so it uses that instead of re-loading the raw
        # one (cached_load_hf_config restores model_type, so the tokenizer class still resolves).
        tokenizer = AutoTokenizer.from_pretrained(model_path, config=cached_load_hf_config(model_path))
    # Ensure a chat template is present. Recent transformers auto-loads `chat_template.jinja` into
    # `tokenizer.chat_template`, but older transformers and some checkpoints leave it unset while
    # shipping the template in a SIDE FILE — `chat_template.json` (Mistral-style, JSON-wrapped) or
    # `chat_template.jinja` (raw Jinja, e.g. GLM-4.x). Without it, `apply_chat_template` falls back to
    # a wrong/empty prompt and the model degenerates. Load whichever side file exists (JSON first,
    # then raw Jinja), local dir or hub.
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = _load_side_chat_template(model_path)
    return tokenizer


def _resolve_repo_file(model_path: str, filename: str) -> str | None:
    """Path to `filename` for a local dir or a hub repo id, or None if absent."""
    if os.path.isdir(model_path):
        p = os.path.join(model_path, filename)
        return p if os.path.isfile(p) else None
    try:
        return hf_hub_download(repo_id=model_path, filename=filename)
    except Exception:
        return None


def _load_side_chat_template(model_path: str) -> str | None:
    """Load a chat template shipped as a side file: `chat_template.json` (JSON with a
    `chat_template` key) or `chat_template.jinja` (raw Jinja). Returns the template string or None."""
    if (p := _resolve_repo_file(model_path, "chat_template.json")) is not None:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)["chat_template"]
        except Exception:
            pass
    if (p := _resolve_repo_file(model_path, "chat_template.jinja")) is not None:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    return None


@functools.cache
def load_generation_config(model_path: str) -> dict:
    """Load the model's `generation_config.json` as a plain dict (cached; {} if absent/unreadable).
    Carries the model author's serving defaults: `eos_token_id` (often a LIST of stop tokens),
    `pad_token_id`, and sampling defaults (`temperature`/`top_p`/`top_k`). minisgl historically
    ignored this file, so multi-EOS models (e.g. GLM-4.x: [154820,154827,154829]) never stopped on
    their secondary end-of-turn tokens and the recommended sampling was not applied."""
    p = _resolve_repo_file(model_path, "generation_config.json")
    if p is None:
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_stop_token_ids(model_path: str, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """The FULL set of end-of-generation token ids, unioned across every source: the
    `generation_config.json` `eos_token_id` (int OR list — the authoritative serving list), the
    tokenizer's `eos_token_id`, and the model config's `eos_token_id`. Deduped, order-stable.
    A model that ends turns with a token other than the tokenizer's single EOS (GLM-4.x, many
    chat/'thinking' models) will not terminate unless ALL of these are treated as stops."""
    ids: list[int] = []

    def _add(v) -> None:
        for t in v if isinstance(v, (list, tuple)) else [v]:
            if isinstance(t, int) and t not in ids:
                ids.append(t)

    _add(load_generation_config(model_path).get("eos_token_id"))
    _add(getattr(tokenizer, "eos_token_id", None))
    try:
        _add(getattr(cached_load_hf_config(model_path), "eos_token_id", None))
    except Exception:
        pass
    return ids


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