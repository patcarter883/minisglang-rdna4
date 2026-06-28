"""CPU-only REAL load_state_dict probe — builds the model on meta (bf16, like the engine) and runs
the actual BaseOP.load_state_dict with the cast checkpoint, so the informative assert pinpoints the
exact mismatched key. No GPU (meta params carry shape/dtype only)."""
import os, sys, traceback
sys.path.insert(0, "/engine/python"); sys.path.insert(0, "/engine")
import torch
MODEL = os.environ["MODEL_PATH"]


def cast(k, v):  # mirror Engine._load_weight_state_dict._cast
    if not v.is_floating_point() or k.endswith(".scales"):
        return v
    if v.dtype == torch.float8_e4m3fn or k.endswith(".weight_scale"):
        return v
    if k.endswith((".A_log", ".dt_bias")):
        return v.to(torch.float32)
    if k.endswith((".balancing_biases", ".hidden_states_scale", ".hidden_states_bias",
                   ".residual_scale", ".residual_bias")):
        return v.to(torch.float32)
    return v.to(torch.bfloat16)


def main():
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29562")
    dist.init_process_group("gloo", rank=0, world_size=1)
    TP = int(os.environ.get("TP", "1"))
    from minisgl.distributed import set_tp_info; set_tp_info(rank=0, size=TP)
    print(f"[probe] simulating TP=0/{TP} (sharding math only; no collectives during load)")
    from minisgl.layers import set_rope_device; set_rope_device(torch.device("cpu"))
    from minisgl.models import create_model, load_weight
    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config, torch_dtype

    cfg = ModelConfig.from_hf(cached_load_hf_config(MODEL))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(cfg)
    sd = {k: cast(k, v) for k, v in load_weight(MODEL, torch.device("cpu"))}
    print(f"[probe] built meta model + {len(sd)} cast checkpoint tensors; running load_state_dict ...")
    try:
        model.load_state_dict(sd)
        print("[probe] load_state_dict PASS — no shape/dtype mismatch, no leftover keys")
    except Exception as e:
        print(f"[probe] load_state_dict FAILED: {type(e).__name__}: {e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
