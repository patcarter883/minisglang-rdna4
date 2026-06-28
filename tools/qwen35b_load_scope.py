"""CPU-only load scope for cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit (qwen3_5_moe).

Meta-builds the model (no GPU alloc) to enumerate the params it EXPECTS, then walks the checkpoint
and the qwen3_5 weight loader to find exactly what doesn't line up. No GPU lease needed (meta tensors
+ CPU safetensors header reads). Prints: model-expected param families, checkpoint suffix families,
and where load_weight first diverges. This is a SCOPING probe, not a fix.
"""
import os, sys, traceback
from collections import Counter

import torch

sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine")
MODEL = os.environ["MODEL_PATH"]


def fam(name: str) -> str:
    # collapse layer indices / expert indices to a family key
    import re
    n = re.sub(r"\.\d+\.", ".N.", name)
    return n


def main() -> None:
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29561")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from minisgl.distributed import set_tp_info; set_tp_info(rank=0, size=1)
    from minisgl.layers import set_rope_device; set_rope_device(torch.device("cpu"))
    from minisgl.layers.base import BaseOP
    from minisgl.models import create_model, load_weight
    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config

    hf = cached_load_hf_config(MODEL)
    cfg = ModelConfig.from_hf(hf)
    print(f"[cfg] model_type={cfg.model_type} arch={cfg.architectures} layers={cfg.num_layers} "
          f"hidden={cfg.hidden_size} experts={getattr(cfg,'num_experts',None)} "
          f"quant={cfg.quant.method if cfg.quant else None} grp={cfg.quant.group_size if cfg.quant else None} "
          f"mtp={cfg.mtp_num_hidden_layers} is_moe={cfg.is_moe}")

    with torch.device("meta"):
        model = create_model(cfg)

    exp = {}
    def collect(op, prefix=""):
        # OPList exposes its members as `<name>.<i>.` (the state_dict key form), NOT `<name>.op_list.<i>`.
        ol = getattr(op, "op_list", None)
        if isinstance(ol, (list, tuple)):
            for i, e in enumerate(ol):
                if isinstance(e, BaseOP): collect(e, prefix + f"{i}.")
                elif isinstance(e, torch.Tensor): exp[prefix + f"{i}"] = (tuple(e.shape), e.dtype)
            return
        for nm, p in op.__dict__.items():
            if nm.startswith("_"): continue
            if isinstance(p, torch.Tensor):
                exp[prefix + nm] = (tuple(p.shape), p.dtype)
            elif isinstance(p, BaseOP):
                collect(p, prefix + nm + ".")
            elif isinstance(p, (list, tuple)):
                for i, e in enumerate(p):
                    if isinstance(e, BaseOP): collect(e, prefix + nm + f".{i}.")
    collect(model)
    print(f"\n[model] expects {len(exp)} params. Families (suffix of last 2 tokens):")
    ef = Counter(".".join(k.split(".")[-2:]) for k in exp)
    for k, v in sorted(ef.items(), key=lambda x: -x[1])[:25]:
        sample = next(n for n in exp if ".".join(n.split(".")[-2:]) == k)
        print(f"   {k:38s} x{v:<5d} e.g. {sample} {exp[sample][0]} {exp[sample][1]}")

    # Walk the loader; capture first error + compare shapes for what it yields.
    print("\n[load_weight] streaming + comparing to model-expected ...")
    seen = set(); mismatch = []; extra = []; n_ok = 0
    try:
        for k, t in load_weight(MODEL, torch.device("cpu")):
            seen.add(k)
            if k not in exp:
                extra.append((k, tuple(t.shape), str(t.dtype))); continue
            es, ed = exp[k]
            if tuple(t.shape) != es:
                mismatch.append((k, tuple(t.shape), es));
            else:
                n_ok += 1
    except Exception as e:
        print(f"   load_weight RAISED: {type(e).__name__}: {e}")
        traceback.print_exc()

    missing = [k for k in exp if k not in seen]
    print(f"\n[result] shape-OK={n_ok}  shape-MISMATCH={len(mismatch)}  "
          f"EXTRA(ckpt key not in model)={len(extra)}  MISSING(model param unfilled)~={len(missing)}")
    print("\n=== shape MISMATCH families (model vs ckpt) ===")
    for k, got, want in mismatch[:12]:
        print(f"   {fam(k)}\n       ckpt {got}  vs model {want}")
    print("\n=== EXTRA ckpt keys (loader produced, model has no such param) ===")
    for k, sh, dt in extra[:12]:
        print(f"   {fam(k)}  {sh} {dt}")
    print("\n=== MISSING model params (never filled) — families ===")
    for k, v in sorted(Counter(fam(m) for m in missing).items(), key=lambda x: -x[1])[:15]:
        print(f"   x{v:<5d} {k}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
