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


def _chat_template_override(model_path: str) -> "str | None":
    """The --chat-template override for the SERVED model, propagated by env so every process that
    loads the served tokenizer (frontend, tokenizer workers — both inherit the launch env) applies
    the same template, while other tokenizers loaded in the same processes (drafters, calibrators)
    are untouched. A path is read here; anything else is treated as a literal jinja string."""
    spec = os.environ.get("MINISGL_CHAT_TEMPLATE")
    if not spec or os.environ.get("MINISGL_CHAT_TEMPLATE_MODEL") != model_path:
        return None
    if os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as f:
            return f.read()
    return spec


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
    # Operator override LAST (it must beat both the baked template and any side file).
    override = _chat_template_override(model_path)
    if override is not None:
        tokenizer.chat_template = override
    return tokenizer


def _resolve_repo_file(model_path: str, filename: str) -> str | None:
    """Path to `filename` for a local dir or a hub repo id, or None if absent."""
    if os.path.isdir(model_path):
        p = os.path.join(model_path, filename)
        return p if os.path.isfile(p) else None
    try:
        return hf_hub_download(repo_id=model_path, filename=filename)
    except Exception:
        pass
    # FALL BACK TO THE LOCAL CACHE, any snapshot. `hf_hub_download` does not merely read — it wants
    # to materialise the file into the current ref's snapshot dir, which fails with PermissionError
    # when that dir is ROOT-OWNED. That is the normal state on this box: the serve containers run as
    # root over the bind-mounted HF cache, so a snapshot fetched inside a container is unwritable by
    # the host user afterwards (observed on RedHatAI/Muse-Glimmer-30B-NVFP4, where the current
    # root-owned snapshot lacks README.md while an older pat-owned one has it). Without this, an
    # optional side file reads as ABSENT rather than unreadable, and a caller silently loses a
    # declared default. Also covers plain offline operation.
    root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    repo_dir = os.path.join(root, "models--" + model_path.replace("/", "--"), "snapshots")
    try:
        for snap in sorted(os.listdir(repo_dir), reverse=True):
            p = os.path.join(repo_dir, snap, filename)
            if os.path.isfile(p):
                return p
    except Exception:
        pass
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


# The sampling triple, and the ONLY keys inherited from a declared base model. Deliberately not
# eos/pad/max_length: a derivative may legitimately change those, whereas losing the author's
# recommended sampling in a repack is a packaging accident, not a decision.
_SAMPLING_KEYS = ("temperature", "top_p", "top_k")


@functools.cache
def declared_base_model(model_path: str) -> str | None:
    """The checkpoint this one DECLARES it was derived from, or None.

    Two declaration sites, both standard, checked in order of specificity: `config.json`
    (`base_model` / `base_model_name_or_path` / `_name_or_path`), then the HF model-card front
    matter, which is where `base_model` actually lives for most quantized repacks. The front-matter
    list mixes the base with method tags — `base_model: [meta-models/Muse-Glimmer-30B, nvfp4,
    llm-compressor]` — so take the first entry shaped like a repo id."""
    p = _resolve_repo_file(model_path, "config.json")
    if p is not None:
        try:
            with open(p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k in ("base_model", "base_model_name_or_path", "_name_or_path"):
                v = cfg.get(k)
                if isinstance(v, str) and "/" in v and v != model_path:
                    return v
        except Exception:
            pass
    p = _resolve_repo_file(model_path, "README.md")
    if p is None:
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    front = text[3 : end if end > 0 else len(text)]
    in_key = False
    for raw in front.splitlines():
        if not raw.strip():
            continue
        if raw.startswith("base_model:"):
            rest = raw.split(":", 1)[1].strip().strip("[]")
            for cand in (c.strip().strip("'\"") for c in rest.split(",")):
                if "/" in cand and cand != model_path:
                    return cand
            in_key = True
            continue
        if in_key:
            stripped = raw.strip()
            if stripped.startswith("- "):
                cand = stripped[2:].strip().strip("'\"")
                if "/" in cand and cand != model_path:
                    return cand
                continue
            if not raw.startswith((" ", "\t")):
                in_key = False               # next top-level key: the list is over
    return None


@functools.cache
def load_generation_config(model_path: str) -> dict:
    """Load the model's `generation_config.json` as a plain dict (cached; {} if absent/unreadable).
    Carries the model author's serving defaults: `eos_token_id` (often a LIST of stop tokens),
    `pad_token_id`, and sampling defaults (`temperature`/`top_p`/`top_k`). minisgl historically
    ignored this file, so multi-EOS models (e.g. GLM-4.x: [154820,154827,154829]) never stopped on
    their secondary end-of-turn tokens and the recommended sampling was not applied."""
    p = _resolve_repo_file(model_path, "generation_config.json")
    data: dict = {}
    if p is not None:
        try:
            with open(p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            data = loaded if isinstance(loaded, dict) else {}
        except Exception:
            data = {}
    # INHERIT the author's sampling from the DECLARED base model when this checkpoint has none.
    # A quantized repack routinely drops `temperature`/`top_p`/`top_k` while the base declares them,
    # and the result is not a neutral default — it is full-distribution sampling (top_p 1.0,
    # top_k -1), the case `_resolve_sampling` was written to prevent. Verified on
    # RedHatAI/Muse-Glimmer-30B-NVFP4 (none of the three) vs meta-models/Muse-Glimmer-30B
    # (1.0 / 0.95 / 64). ONE hop, sampling keys only, and only ones this file does not already set,
    # so a repack that deliberately retunes sampling still wins. Best-effort: fetching the base's
    # config needs the hub (or a warm cache), and if it is unavailable we return what we have rather
    # than fail a boot over a default.
    if not any(k in data for k in _SAMPLING_KEYS):
        base = declared_base_model(model_path)
        if base:
            bp = _resolve_repo_file(base, "generation_config.json")
            if bp is not None:
                try:
                    with open(bp, "r", encoding="utf-8") as f:
                        bdata = json.load(f)
                    if isinstance(bdata, dict):
                        inherited = {k: bdata[k] for k in _SAMPLING_KEYS if k in bdata}
                        if inherited:
                            data = {**data, **inherited}
                            data["_sampling_inherited_from"] = base
                except Exception:
                    pass
    return data


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