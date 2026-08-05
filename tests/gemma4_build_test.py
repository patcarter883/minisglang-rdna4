"""Gemma4 meta-device build: proves the model assembles and pins the exact parameter set + shapes.

CPU/meta only — no GPU lease, no weights read, cannot disturb a running serve. Run inside the serve
image (the host torch install is broken):

    docker run --rm --entrypoint bash -v <worktree>:/wt \
      -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
      minisgl-rdna4:lean -lc 'cd /wt && PYTHONPATH=/wt/python:/opt/kernels \
        python tests/gemma4_build_test.py [tp_size]'

The key set this prints IS the loader's contract: BaseOP.load_state_dict pops every key it declares
and raises on anything left over, so the weight loader must yield exactly this set. The shape checks
guard the part that is genuinely unusual — the sliding layers are (16 q, 8 kv, head_dim 256) and the
full layers are (16 q, 2 kv, head_dim 512), so q_proj/k_proj/o_proj widths DIFFER BY LAYER, and the
5 full layers carry no v_proj at all.
"""

from __future__ import annotations

import collections
import glob
import re
import sys

import torch

MODEL_GLOB = (
    "/root/.cache/huggingface/hub/"
    "models--cyankiwi--gemma-4-26B-A4B-it-qat-AWQ-INT4/snapshots/*/"
)

FULL_LAYERS = (5, 11, 17, 23, 29)


def main() -> int:
    tp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    from minisgl.distributed import set_tp_info

    set_tp_info(0, tp_size)

    import minisgl.layers.rotary as rotary_mod
    from transformers import AutoConfig

    from minisgl.models import create_model
    from minisgl.models.config import ModelConfig

    rotary_mod.set_rope_device(torch.device("cpu"))

    matches = glob.glob(MODEL_GLOB)
    if not matches:
        print(f"SKIP: checkpoint not cached under {MODEL_GLOB}")
        return 0
    mc = ModelConfig.from_hf(AutoConfig.from_pretrained(matches[0]), spec_algorithm="none")

    torch.set_default_dtype(torch.float16)
    with torch.device("meta"):
        model = create_model(mc)
    sd = model.state_dict()

    print(f"[build] tp_size={tp_size}  parameters={len(sd)}")
    groups = collections.Counter(re.sub(r"\.\d+\.", ".{N}.", k) for k in sd)
    for group, count in sorted(groups.items()):
        print(f"  {group:64s} x{count}")

    failures = 0

    def check(name: str, got, want) -> None:
        nonlocal failures
        ok = got == want
        failures += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {name:52s} got={got!r:<22} want={want!r}")

    print("\n[shapes] per-layer attention geometry (the split-head_dim guard)")
    # The attention projections ARE int4-quantized here (only the dense mlp, the router and lm_head
    # are in the checkpoint's ignore list), so they materialize as compressed-tensors
    # `weight_packed` (out, in//8) int32 + `weight_scale` (out, in//group_size).
    #
    # q heads are 16 on BOTH types; only kv heads and head_dim change. Column-parallel projections
    # shard the output dim, row-parallel (o_proj) shards the input dim.
    pack = 8  # int4 nibbles per int32 word

    def proj_shape(key: str) -> tuple:
        return tuple(sd[f"{key}.weight_packed"].shape)

    for layer_id, kind, n_kv, head_dim in (
        (4, "sliding", 8, 256),
        (5, "full", 2, 512),
        (28, "sliding", 8, 256),
        (29, "full", 2, 512),
    ):
        p = f"model.layers.{layer_id}.self_attn"
        kv_local = max(n_kv // tp_size, 1)  # replicated rather than split when nkv < tp_size
        check(f"L{layer_id} {kind} q_proj", proj_shape(f"{p}.q_proj"),
              (16 * head_dim // tp_size, mc.hidden_size // pack))
        check(f"L{layer_id} {kind} k_proj", proj_shape(f"{p}.k_proj"),
              (kv_local * head_dim, mc.hidden_size // pack))
        check(f"L{layer_id} {kind} o_proj", proj_shape(f"{p}.o_proj"),
              (mc.hidden_size, 16 * head_dim // tp_size // pack))
        check(f"L{layer_id} {kind} q_norm", tuple(sd[f"{p}.q_norm.weight"].shape), (head_dim,))
        # v_proj must be ABSENT on the full layers and PRESENT on the sliding ones — building one
        # the checkpoint does not ship would fail the loader's exact-key check.
        check(f"L{layer_id} {kind} has v_proj",
              f"{p}.v_proj.weight_packed" in sd, kind == "sliding")

    print("\n[shapes] the easily-missed per-layer scalars and router rescalings")
    check("layer_scalar", tuple(sd["model.layers.0.layer_scalar"].shape), (1,))
    check("router.weight", tuple(sd["model.layers.0.router.weight"].shape),
          (mc.num_experts, mc.hidden_size))
    check("router.scale", tuple(sd["model.layers.0.router.scale"].shape), (mc.hidden_size,))
    check("router.per_expert_scale", tuple(sd["model.layers.0.router.per_expert_scale"].shape),
          (mc.num_experts,))

    print("\n[counts] every layer carries the full parallel-FFN norm set")
    for norm in (
        "input_layernorm", "post_attention_layernorm",
        "pre_feedforward_layernorm", "post_feedforward_layernorm_1",
        "pre_feedforward_layernorm_2", "post_feedforward_layernorm_2",
        "post_feedforward_layernorm",
    ):
        check(norm, sum(1 for k in sd if k.endswith(f".{norm}.weight")), mc.num_layers)

    check("v_proj count (sliding only)",
          sum(1 for k in sd if k.endswith(".v_proj.weight_packed")),
          mc.num_layers - len(FULL_LAYERS))
    check("lm_head tied (no own weight)",
          "lm_head.weight" in sd, False)

    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} checks)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
