"""Gemma4 weight loader: the emitted key set must EXACTLY match what the model declares.

CPU-only (streams the checkpoint to CPU, never touches a GPU) — no lease, cannot disturb a serve:

    docker run --rm --entrypoint bash -v <worktree>:/wt \
      -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
      minisgl-rdna4:lean -lc 'cd /wt && PYTHONPATH=/wt/python:/opt/kernels \
        python tests/gemma4_loader_test.py [tp_size]'

Why an exact set and not "close enough": BaseOP.load_state_dict pops each declared key and raises on
leftovers, so a MISSING key is a KeyError at load and an EXTRA key is a RuntimeError — both loud. The
dangerous case is the third one this test exists to catch: a key that is present with the RIGHT name
but the WRONG shape or the wrong TP shard, which loads cleanly and produces garbage. So every tensor
is compared on shape and dtype too.
"""

from __future__ import annotations

import glob
import sys

import torch

MODEL_ID = "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4"
MODEL_GLOB = f"/root/.cache/huggingface/hub/models--{MODEL_ID.replace('/', '--')}/snapshots/*/"


def main() -> int:
    tp_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL_ID

    from minisgl.distributed import set_tp_info

    set_tp_info(0, tp_size)

    import minisgl.layers.rotary as rotary_mod
    from transformers import AutoConfig

    from minisgl.models import create_model, load_weight
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
    declared = {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}

    emitted: dict[str, tuple] = {}
    duplicates: list[str] = []
    for name, tensor in load_weight(model_path, torch.device("cpu"), spec_algorithm="none"):
        if name in emitted:
            duplicates.append(name)
        emitted[name] = (tuple(tensor.shape), tensor.dtype)
        del tensor

    print(f"[loader] tp_size={tp_size}  declared={len(declared)}  emitted={len(emitted)}")

    missing = sorted(set(declared) - set(emitted))
    extra = sorted(set(emitted) - set(declared))
    # dtype is compared separately: _coerce_dtype legitimately casts between wide float types
    # (a checkpoint may ship fp16 where the layer declares bf16), but a packed/quantized dtype
    # mismatch is a hard error because the dtype IS the encoding contract.
    wrong_shape = sorted(
        k for k in set(declared) & set(emitted) if declared[k][0] != emitted[k][0]
    )
    castable = {torch.float32, torch.float64, torch.bfloat16, torch.float16}
    wrong_dtype = sorted(
        k
        for k in set(declared) & set(emitted)
        if declared[k][1] != emitted[k][1]
        and not (declared[k][1] in castable and emitted[k][1] in castable)
    )

    def report(label: str, keys: list[str], detail=None) -> None:
        print(f"\n{label}: {len(keys)}")
        for k in keys[:15]:
            print(f"    {k}{'' if detail is None else detail(k)}")
        if len(keys) > 15:
            print(f"    ... and {len(keys) - 15} more")

    report("MISSING (declared but never emitted)", missing)
    report("EXTRA (emitted but not declared)", extra)
    report(
        "WRONG SHAPE", wrong_shape,
        lambda k: f"  model={declared[k][0]} loader={emitted[k][0]}",
    )
    report(
        "WRONG DTYPE (non-castable)", wrong_dtype,
        lambda k: f"  model={declared[k][1]} loader={emitted[k][1]}",
    )
    report("DUPLICATE emissions", sorted(set(duplicates)))

    failures = len(missing) + len(extra) + len(wrong_shape) + len(wrong_dtype) + len(duplicates)
    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} problems)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
