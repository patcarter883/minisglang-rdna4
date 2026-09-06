"""MEASURE the two numbers that decide whether a full-48-layer qwen4_exp forward is reachable.

Everything about "does this model fit" in `docs/QWEN4EXP_BRINGUP_PLAN.md` §2 is ARITHMETIC on
shapes read out of safetensors headers. It has never been checked against the model this repo
actually builds, and the plan says so. This probe replaces both halves with measurements:

  [A] RESIDENT BYTES of the real 48-layer model, split into the ROUTED-EXPERT tier and everything
      else, taken from the meta-device build's own `state_dict()` — i.e. from the containers the
      engine really allocates, after `post_load`'s repack shapes are accounted. If "everything
      else" fits a 16 GB card with room for a KV pool, then a full-depth forward is reachable by
      streaming ONLY the expert tier, and the layer-subset restriction is not fundamental.

  [B] WALL TIME to bring one layer's 512 NVFP4 experts from its four on-disk shards through the
      real loader (fold + gate/up merge + stack over E). x48 x (1 + n_decode) is the cost of a
      streamed full-depth run, so this number decides whether the streaming design is a prefill-only
      probe or a real decode loop.

CPU + one device for the fold. No claims, only numbers.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

import torch

MODEL = os.environ.get("Q4E_MODEL", "/model")


def _fmt(nbytes: float) -> str:
    return f"{nbytes / 2**30:8.3f} GiB"


def main() -> int:
    from minisgl.distributed import set_tp_info

    set_tp_info(0, 1)
    import minisgl.layers.rotary as rotary_mod

    rotary_mod.set_rope_device(torch.device("cpu"))

    from minisgl.models import create_model
    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config

    mc = ModelConfig.from_hf(cached_load_hf_config(MODEL), spec_algorithm="none")
    print(f"[A] layers={mc.num_layers} experts={mc.num_experts} top_k={mc.num_experts_per_tok}")

    # NOT optional, and the failure is silent in the direction that makes the model look infeasible.
    # Every bf16 parameter is built at `torch.get_default_dtype()`, which is fp32 unless set — the
    # engine sets it from the config (`Engine.__init__`), a bare probe does not. Omitting this
    # reported a 18.432 GiB non-expert body instead of 9.216 GiB, i.e. exactly 2x on every bf16
    # tensor, which is enough to conclude "the body alone does not fit a 16 GB card" and abandon a
    # full-depth run that in fact fits with 4.5 GiB to spare. The NVFP4 expert tier is unaffected
    # (uint8/fp16 are declared explicitly), which is why the two halves disagreed and caught it.
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        model = create_model(mc)

    # `state_dict()` on the meta build is the pre-`post_load` (checkpoint-shaped) layout. The NVFP4
    # containers repack in `post_load`: weight_packed (E,N,K//2) uint8 -> _w_op (E,N,K//8) int32 is
    # byte-IDENTICAL, and weight_scale (E,N,K//16) fp16 -> _scales_op (E,K//16,N) fp16 is a
    # transpose, also byte-identical. So the two layouts have the same resident bytes and this sum
    # is the real one. (The e4m3->fp16 scale inflation already happened at the LEAF, before this.)
    expert_bytes = 0
    other_bytes = 0
    per_kind: "dict[str, int]" = {}
    for name, p in model.state_dict().items():
        nb = p.numel() * p.element_size()
        if ".experts." in name or "gate_up_proj.weight" in name or "down_proj.weight" in name:
            # Routed-expert containers only: they are the ones stacked over E.
            if p.dim() >= 3 and p.shape[0] == mc.num_experts:
                expert_bytes += nb
                per_kind.setdefault("routed_experts", 0)
                per_kind["routed_experts"] += nb
                continue
        other_bytes += nb
        key = "embed/lm_head" if ("embed" in name or "lm_head" in name) else (
            "hyper_connection" if "hyper_connection" in name or "hc_" in name else (
                "ple" if ".ple." in name else (
                    "linear_attn" if "linear_attn" in name else (
                        "self_attn" if "self_attn" in name else "misc"))))
        per_kind[key] = per_kind.get(key, 0) + nb

    print(f"[A] routed experts      {_fmt(expert_bytes)}")
    print(f"[A] everything else     {_fmt(other_bytes)}")
    for k, v in sorted(per_kind.items(), key=lambda kv: -kv[1]):
        print(f"[A]     {k:20s} {_fmt(v)}")
    print(f"[A] per-layer expert    {_fmt(expert_bytes / mc.num_layers)}")

    # ---- [B] one layer's experts through the real loader -------------------------------------
    if not torch.cuda.is_available():
        print("[B] SKIP: no device (the NVFP4 fold runs on device)")
        return 0

    layer = int(os.environ.get("Q4E_PROBE_LAYER", "0"))
    tmp = tempfile.mkdtemp(prefix="q4e-onelayer-")
    import json

    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    tc = cfg["text_config"]
    tc["num_hidden_layers"] = 1
    tc["layer_types"] = tc["layer_types"][:1]
    tc["ple_layer_ids"] = []
    with open(os.path.join(tmp, "config.json"), "w") as f:
        json.dump(cfg, f)
    shards = [
        f"{MODEL}/layer-{layer:05d}-experts-{lo:04d}-{lo + 127:04d}.safetensors"
        for lo in (0, 128, 256, 384)
    ]
    for s in shards:
        assert os.path.exists(s), s
        # Symlink, so the loader's own glob sees EXACTLY this layer and nothing else.
        os.symlink(s, os.path.join(tmp, os.path.basename(s)))

    from minisgl.models.weight import _load_qwen4_exp_weight

    one = ModelConfig.from_hf(cached_load_hf_config(tmp), spec_algorithm="none")
    dev = torch.device("cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    got = {}
    for k, v in _load_qwen4_exp_weight(tmp, dev, one):
        got[k] = tuple(v.shape) + (str(v.dtype),)
    torch.cuda.synchronize()
    dt = time.time() - t0
    on_disk = sum(os.path.getsize(s) for s in shards)
    print(f"[B] layer {layer}: {len(got)} stacked tensors in {dt:.1f} s "
          f"({on_disk / 2**30:.2f} GiB on disk, {on_disk / dt / 2**20:.0f} MiB/s)")
    for k, v in sorted(got.items()):
        print(f"[B]     {k:70s} {v}")
    print(f"[B] PROJECTION full depth = 48 x {dt:.1f}s = {48 * dt / 60:.1f} min per forward pass")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
