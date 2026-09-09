"""Nemotron-H's schedule must parse WITHOUT being squeezed into the GDN/SWA vocabulary.

Phase 0/N3 of docs/NEMOTRON35_LIGHTNING_PLAN.md. The plan originally said "normalise Nemotron's
`layers_block_type` into the existing `layer_types` vocabulary (cheap, preferred)". Reading the
checkpoint falsified that:

  * Every other model here is layer = mixer + MLP. Nemotron-H is ONE mixer per layer — the tensor
    inventory shows 52 layers and 52 `backbone.layers.N.norm.weight`, with `mixer.experts.*` on the
    MoE layers, `mixer.in_proj`/`A_log`/`conv1d` on the mamba layers and `mixer.q_proj` on the six
    attention layers. "moe" is a peer of "attention" in the schedule, not something after it.
  * Mapping "mamba" -> "linear_attention" would flip `is_gdn_hybrid` and route this model into the
    GDN state cache, slot manager and kernels — a different recurrence. It would build, run, and be
    silently wrong.

So the schedule gets its own vocabulary and `layer_types` stays None. These tests pin that, against
the REAL published config (gzipped fixture, not a hand-written stub — a stub cannot catch a key that
the checkpoint spells differently than expected).

    python3 -m pytest tests/nemotron_config_test.py -q -o addopts=""
"""

from __future__ import annotations

import gzip
import json
import os
from types import SimpleNamespace

import pytest

from minisgl.models.config import ModelConfig  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "nemotron", "target_config.json.gz")


def hf_config():
    """The published config as an attribute bag, which is all `from_hf` reads it as."""
    with gzip.open(FIXTURE, "rt") as f:
        d = json.load(f)
    return SimpleNamespace(**d)


@pytest.fixture(scope="module")
def mc():
    return ModelConfig.from_hf(hf_config())


# ---------------------------------------------------------------- the schedule

def test_the_schedule_parses_with_its_own_vocabulary(mc):
    assert mc.block_types is not None and len(mc.block_types) == 52
    assert set(mc.block_types) == {"mamba", "moe", "attention"}
    assert mc.is_mamba_hybrid


def test_the_gdn_and_swa_paths_stay_DEAD(mc):
    """The whole point of a separate field. If either of these flips, this model is routed into a
    recurrence it does not have."""
    assert mc.layer_types is None
    assert not mc.is_gdn_hybrid
    assert not mc.is_cca_hybrid
    assert mc.sliding_window is None


def test_layer_ids_partition_the_stack(mc):
    mamba, moe, attn = mc.mamba_layer_ids, mc.moe_block_layer_ids, mc.full_attn_layer_ids
    assert (len(mamba), len(moe), len(attn)) == (23, 23, 6)
    assert sorted(mamba + moe + attn) == list(range(52))
    assert attn == [5, 12, 19, 26, 33, 42]


# ---------------------------------------------------------------- mamba geometry

def test_mamba_geometry_matches_the_checkpoint(mc):
    assert (mc.mamba_num_heads, mc.mamba_head_dim) == (64, 64)
    assert (mc.mamba_ssm_state, mc.mamba_n_groups) == (128, 8)
    assert (mc.mamba_conv_kernel, mc.mamba_chunk_size) == (4, 128)
    assert (mc.mamba_dt_min, mc.mamba_dt_max) == (0.001, 0.1)
    assert mc.mamba_conv_bias is True and mc.mamba_proj_bias is False


def test_derived_widths_match_the_real_tensor_shapes(mc):
    """These three numbers are the load-time contract. The checkpoint ships
    conv1d.weight (6144, 1, 4) and in_proj.weight (10304, 2688); if either derivation is wrong the
    [z, x, B, C, dt] split is wrong and every number downstream is garbage."""
    assert mc.mamba_inner_dim == 4096                 # 64 heads x 64
    assert mc.mamba_conv_dim == 6144                  # x + B + C  = 4096 + 2*8*128
    assert mc.mamba_in_proj_dim == 10304              # z + (x,B,C) + dt = 4096 + 6144 + 64


# ---------------------------------------------------------------- MoE

def test_the_experts_are_relu2_and_NOT_gated(mc):
    """Load-bearing. Every MoE in this repo before Nemotron-H was gate+up fused with SiLU; the
    checkpoint ships one `up_proj` per expert and `mlp_hidden_act: relu2`. A gated path applied here
    halves the intermediate and multiplies by the wrong thing, silently."""
    assert mc.moe_act == "relu2"


def test_moe_shape_matches_the_checkpoint(mc):
    assert mc.num_experts == 128
    assert mc.num_experts_per_tok == 6
    assert mc.moe_intermediate_size == 1856
    assert mc.moe_shared_intermediate == 3712
    assert mc.routed_scaling_factor == 2.5


# ---------------------------------------------------------------- attention

def test_attention_geometry_forces_TP2(mc):
    """2 kv heads is why TP=2 is the only workable parallelism — it cannot shard four ways."""
    assert mc.num_kv_heads == 2
    assert mc.head_dim == 128
    assert mc.num_qo_heads == 32


# ---------------------------------------------------------------- refusals

@pytest.mark.parametrize("mutate,why", [
    (lambda d: d.pop("layers_block_type"), "no schedule"),
    (lambda d: d.__setitem__("layers_block_type", ["mamba"] * 51), "wrong length"),
    (lambda d: d.__setitem__("layers_block_type", ["mamba", "mlp"] + ["moe"] * 50), "unknown kind"),
])
def test_a_schedule_that_cannot_be_trusted_is_REFUSED_not_guessed(mutate, why):
    """A guessed mixer schedule builds a model that loads and is wrong. Refuse instead."""
    with gzip.open(FIXTURE, "rt") as f:
        d = json.load(f)
    mutate(d)
    with pytest.raises(ValueError):
        ModelConfig.from_hf(SimpleNamespace(**d))


def test_a_non_nemotron_config_gets_none_of_this():
    """The fields must stay None for every other family, or a stray `is_mamba_hybrid` appears on a
    model that has no mamba layer."""
    cfg = SimpleNamespace(model_type="qwen3", num_hidden_layers=4, hidden_size=64,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          vocab_size=32, max_position_embeddings=128, rope_theta=10000.0,
                          intermediate_size=128, rms_norm_eps=1e-6, torch_dtype="bfloat16",
                          tie_word_embeddings=False)
    mc = ModelConfig.from_hf(cfg)
    assert mc.block_types is None and not mc.is_mamba_hybrid
    assert mc.mamba_num_heads is None and mc.moe_act is None
    assert mc.mamba_inner_dim is None and mc.mamba_conv_dim is None
