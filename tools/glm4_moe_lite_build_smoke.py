"""Meta-device build + weight-contract smoke for glm4_moe_lite (GLM-4.7-Flash AWQ).

Builds Glm4MoeLiteForCausalLM on the meta device (no GPU, no real memory) from the REAL HF config,
then proves the loader's name contract against the ACTUAL checkpoint keys (read from the safetensors
index — no tensors loaded): every produced runtime key must land on a model parameter, and every
model parameter must be filled. This is the CPU half of "validate the AWQ checkpoint loads" — it
catches name/merge/stack/MTP-skip mismatches without needing the 19 GB on a GPU (the model exceeds a
single 16 GB card; a real serve needs TP=2).

Checks:
  * is_mla + is_moe + AWQ; MLA attn / router gate / dense layer-0 / lm_head are bf16;
  * routed experts AWQ-stacked (gate_up/down qweight/scales/qzeros over E);
  * shared expert AWQ (QuantTrio quantizes it — see the model docstring / shared-expert flip);
  * loader(checkpoint keys) == model.state_dict() keys exactly (MTP layer skipped).

CPU-only, run in the combined ROCm image:
    PYTHONPATH=/engine/python:/engine python /engine/tools/glm4_moe_lite_build_smoke.py
"""
from __future__ import annotations

import json
import os

import torch
from transformers import AutoConfig

from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.models.weight import (
    _get_expert_stack_info,
    _get_merge_info,
    _is_beyond_decoder,
)
from minisgl.utils import cached_load_hf_config, download_hf_weight, torch_dtype

MODEL = os.environ.get("MINISGL_ORACLE_MODEL", "QuantTrio/GLM-4.7-Flash-AWQ")


def _runtime_key(ckpt_key: str, num_layers: int, is_moe: bool) -> str | None:
    """Replay load_weight's pure name transforms (merge gate/up -> gate_up, then expert stack,
    skip MTP). Returns the runtime state_dict key, or None if the loader drops the tensor."""
    if _is_beyond_decoder(ckpt_key, num_layers):  # MTP / next-token head (layers.>=num_layers)
        return None
    if ckpt_key.startswith(("vision_tower.", "multi_modal_projector.")):
        return None
    if ckpt_key.endswith(".g_idx"):
        return None
    name = ckpt_key.removeprefix("language_model.")
    # e_score_correction_bias is a genuine non-zero bias buffer; only all-zero .bias are dropped
    # (QuantTrio carries no zero biases here, so nothing else to drop).
    info = _get_merge_info(name)
    merged = info[0] if info is not None else name
    if is_moe and (einfo := _get_expert_stack_info(merged)) is not None:
        return einfo[0]
    return merged


def _build(cfg, tp: int):
    set_tp_info(rank=0, size=tp)
    set_rope_device(torch.device("cpu"))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        return create_model(cfg)


def check_tp2(cfg) -> None:
    """TP=2 meta-build: the MLA q/kv up-projections + o_proj are head-parallel (halved), the
    replicated bottlenecks/latent are unchanged, and the AWQ experts shard the intermediate. Also
    shard ONE real AWQ tensor through the loader to prove the col/row AWQ axis-flip is correct."""
    import safetensors
    from minisgl.models.weight import _shard_tensor

    Hf, qk, v, kvl, ql, H = (cfg.num_qo_heads, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
                             cfg.v_head_dim, cfg.kv_lora_rank, cfg.q_lora_rank, cfg.hidden_size)
    inter, g, pf, E = cfg.moe_intermediate_size, cfg.quant.group_size, 8, cfg.num_experts
    m = _build(cfg, 2)
    sd = m.state_dict()
    L1 = "model.layers.1."
    exp = {  # model param -> expected TP=2 LOCAL shape
        L1 + "self_attn.q_b_proj.weight": (Hf * qk // 2, ql),       # col-parallel (heads halved)
        L1 + "self_attn.kv_b_proj.weight": (Hf * (cfg.qk_nope_head_dim + v) // 2, kvl),
        L1 + "self_attn.o_proj.weight": (H, Hf * v // 2),           # row-parallel (input halved)
        L1 + "self_attn.kv_a_proj_with_mqa.weight": (kvl + cfg.qk_rope_head_dim, H),  # replicated
        L1 + "self_attn.q_a_proj.weight": (ql, H),                  # replicated
        L1 + "mlp.experts.gate_up_proj.qweight": (E, H, (2 * inter // 2) // pf),  # routed: AWQ inter
        L1 + "mlp.experts.down_proj.qweight": (E, inter // 2, H // pf),
        L1 + "mlp.shared_experts.gate_up_proj.qweight": (H, (2 * inter) // pf),   # shared: REPLICATED
        L1 + "mlp.shared_experts.down_proj.qweight": (inter, H // pf),
    }
    for k, want in exp.items():
        got = tuple(sd[k].shape)
        assert got == want, f"TP2 shape {k}: got {got}, want {want}"
    print(f"[OK] TP=2 meta shapes: MLA heads halved ({Hf}->{Hf//2}/rank), latent/bottleneck "
          f"replicated, routed experts shard the intermediate, shared expert REPLICATED (K%512).")

    # End-to-end axis-flip proof: load one real ROUTED-expert AWQ row-parallel tensor (down_proj),
    # shard it, and match the per-expert slot of the stacked TP=2 param.
    folder = download_hf_weight(MODEL)
    index = json.load(open(os.path.join(folder, "model.safetensors.index.json")))
    ckpt_key = L1 + "mlp.experts.0.down_proj.qweight"
    fpath = os.path.join(folder, index["weight_map"][ckpt_key])
    with safetensors.safe_open(fpath, framework="pt", device="cpu") as f:
        full = f.get_tensor(ckpt_key)
    sharded = _shard_tensor(ckpt_key, full, r=0, n=2, num_kv_heads=cfg.num_kv_heads)
    want = tuple(sd[L1 + "mlp.experts.down_proj.qweight"].shape[1:])  # per-expert slot
    assert tuple(sharded.shape) == want, \
        f"AWQ row-parallel shard {ckpt_key}: full {tuple(full.shape)} -> {tuple(sharded.shape)} != slot {want}"
    print(f"[OK] AWQ axis-flip: real {ckpt_key} {tuple(full.shape)} --row-parallel(dim0)--> "
          f"{tuple(sharded.shape)} matches the per-expert TP=2 slot.")


def main() -> None:
    # set_tp_info is one-shot per process, so the TP=2 sharding check runs as a separate invocation
    # (MINISGL_SMOKE_TP=2). TP=1 does the full structural + loader-contract pass.
    tp = int(os.environ.get("MINISGL_SMOKE_TP", "1"))
    cfg = ModelConfig.from_hf(cached_load_hf_config(MODEL))
    assert cfg.is_mla and cfg.is_moe and cfg.quant is not None and cfg.quant.is_awq, cfg
    print(f"[cfg] tp={tp} layers={cfg.num_layers} H={cfg.hidden_size} experts={cfg.num_experts} "
          f"top_k={cfg.num_experts_per_tok} kv_lora={cfg.kv_lora_rank} "
          f"qk_nope={cfg.qk_nope_head_dim} qk_rope={cfg.qk_rope_head_dim} v_head={cfg.v_head_dim} "
          f"qo_heads={cfg.num_qo_heads} awq_g={cfg.quant.group_size}")

    if tp == 2:
        check_tp2(cfg)
        print("RESULT: ALL PASS")
        return

    model = _build(cfg, 1)
    sd = model.state_dict()
    sd_keys = set(sd.keys())
    print(f"[build] meta-built; state_dict tensors = {len(sd_keys)}")

    # ---- structural spot-checks ----
    L1 = "model.layers.1."           # first MoE layer (layer 0 is dense)
    # MLA attention bf16 (NOT quantized)
    for suf in ("self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight", "self_attn.kv_b_proj.weight",
                "self_attn.o_proj.weight", "self_attn.q_a_layernorm.weight",
                "self_attn.kv_a_layernorm.weight"):
        assert L1 + suf in sd_keys, suf
    assert sd[L1 + "self_attn.q_b_proj.weight"].dtype == torch.bfloat16
    # router gate bf16 + correction bias
    assert L1 + "mlp.gate.weight" in sd_keys and L1 + "mlp.gate.e_score_correction_bias" in sd_keys
    # routed experts AWQ (stacked over E): gate_up + down qweight/scales/qzeros
    for suf in ("mlp.experts.gate_up_proj.qweight", "mlp.experts.gate_up_proj.scales",
                "mlp.experts.gate_up_proj.qzeros", "mlp.experts.down_proj.qweight",
                "mlp.experts.down_proj.scales", "mlp.experts.down_proj.qzeros"):
        assert L1 + suf in sd_keys, suf
    assert sd[L1 + "mlp.experts.gate_up_proj.qweight"].shape[0] == cfg.num_experts
    # shared expert AWQ (QuantTrio quantizes it -> qweight, NOT a bf16 .weight)
    assert L1 + "mlp.shared_experts.gate_up_proj.qweight" in sd_keys
    assert L1 + "mlp.shared_experts.down_proj.qweight" in sd_keys
    assert L1 + "mlp.shared_experts.gate_up_proj.weight" not in sd_keys, "shared expert should be AWQ"
    # dense layer-0 bf16
    assert "model.layers.0.mlp.gate_up_proj.weight" in sd_keys
    assert "model.layers.0.mlp.down_proj.weight" in sd_keys
    # untied lm_head bf16
    assert "lm_head.weight" in sd_keys and sd["lm_head.weight"].dtype == torch.bfloat16
    print("[OK] structural: MLA/gate/dense0/lm_head bf16; routed + shared experts AWQ-stacked.")

    # ---- loader contract: checkpoint keys -> runtime keys must EXACTLY cover state_dict ----
    folder = download_hf_weight(MODEL)
    index = json.load(open(os.path.join(folder, "model.safetensors.index.json")))
    ckpt_keys = list(index["weight_map"].keys())
    print(f"[ckpt] {len(ckpt_keys)} checkpoint tensors")

    produced, skipped = set(), 0
    for k in ckpt_keys:
        rk = _runtime_key(k, cfg.num_layers, cfg.is_moe)
        if rk is None:
            skipped += 1
        else:
            produced.add(rk)

    missing = sd_keys - produced          # model params with no checkpoint source
    unexpected = produced - sd_keys       # checkpoint keys with no model home
    print(f"[loader] produced {len(produced)} runtime keys; skipped {skipped} ckpt tensors (MTP/etc.)")
    if missing:
        print(f"  MISSING ({len(missing)}): {sorted(missing)[:12]}")
    if unexpected:
        print(f"  UNEXPECTED ({len(unexpected)}): {sorted(unexpected)[:12]}")
    assert not missing and not unexpected, "loader contract MISMATCH"
    print("[OK] loader contract: every checkpoint tensor maps to a model param; all params covered.")
    print("RESULT: ALL PASS  (run with MINISGL_SMOKE_TP=2 for the TP=2 sharding check)")


if __name__ == "__main__":
    main()
