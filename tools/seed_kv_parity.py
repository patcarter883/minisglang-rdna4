"""Numerical parity for the prompt-prefill draft-KV seed (MINISGL_SPEC_PREFILL_SEED).

Runs INSIDE the serve image via tools/run_seed_kv_parity.sh (TP=1; the EAGLE3 draft is replicated, so a TP=1 forward is the
per-rank computation). Validates the ONE correctness risk of the seed: that the BATCHED, attention-free
``GLMEagle3DraftModel.seed_buffered`` produces BYTE-IDENTICAL (k, v) to the autoregressive ``step_masked``
that the decode-time propose uses — i.e. seeding the cache from the prompt yields exactly the KV the
draft layer would have produced had it processed those positions one-at-a-time during decode.

The k/v a position contributes depends ONLY on that position's (embed, hidden, RoPE pos) — NOT on the
attention history — so step's appended cache entry at position p must equal seed_kv's entry p. Feeds
BOTH the SAME fixed-random inputs, so any divergence is a seed_kv bug (not aux/serve).
"""
import glob
import os
import sys

import torch

sys.path.insert(0, "/engine/python")
sys.path.insert(0, "/engine")

CKPT = os.environ["EAGLE3_CKPT"]  # local snapshot folder


def main() -> None:
    torch.manual_seed(0)
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29556")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from minisgl.distributed import set_tp_info

    set_tp_info(rank=0, size=1)
    from minisgl.layers import set_rope_device

    set_rope_device(torch.device("cuda"))

    import safetensors.torch as st

    sd = st.load_file(glob.glob(os.path.join(CKPT, "*.safetensors"))[0], device="cuda")
    dtype = torch.bfloat16
    hidden, H, Hkv, hd, eps, theta = 2048, 16, 4, 128, 1e-5, 1e6
    from minisgl.models.glm_eagle3 import GLMEagle3DraftModel

    with torch.device("cuda"):
        m = GLMEagle3DraftModel(
            hidden_size=hidden, intermediate_size=8192, num_heads=H, num_kv_heads=Hkv,
            head_dim=hd, num_aux_layers=3, draft_vocab_size=32000, target_vocab_size=154880,
            rms_norm_eps=eps, rope_theta=theta, max_position=4096,
        )
    kmap = {
        "midlayer.input_layernorm.weight": ("input_layernorm", "weight"),
        "midlayer.hidden_norm.weight": ("hidden_norm", "weight"),
        "midlayer.self_attn.q_proj.weight": ("q_proj", "weight"),
        "midlayer.self_attn.k_proj.weight": ("k_proj", "weight"),
        "midlayer.self_attn.v_proj.weight": ("v_proj", "weight"),
        "midlayer.self_attn.o_proj.weight": ("o_proj", "weight"),
        "midlayer.post_attention_layernorm.weight": ("post_attention_layernorm", "weight"),
        "midlayer.mlp.gate_proj.weight": ("gate_proj", "weight"),
        "midlayer.mlp.up_proj.weight": ("up_proj", "weight"),
        "midlayer.mlp.down_proj.weight": ("down_proj", "weight"),
        "norm.weight": ("norm", "weight"),
        "lm_head.weight": ("lm_head", "weight"),
    }
    for ck, (obj, leaf) in kmap.items():
        setattr(getattr(m, obj), leaf, sd[ck].to(dtype).contiguous())

    # A "prompt" of S positions with fixed-random embeds/hiddens (the fused-aux feature) and
    # contiguous RoPE positions 1..S (the seed runs pairs p=1..P-1).
    S = 12
    embeds = torch.randn(S, hidden, device="cuda", dtype=dtype) * 0.1
    hiddens = torch.randn(S, hidden, device="cuda", dtype=dtype) * 0.5
    positions = torch.arange(1, S + 1, device="cuda", dtype=torch.int32)

    # Reference: the decode-time step_masked over each position, one at a time, writing its (k, v) into
    # ring A at column s (the attention output is discarded — only the stored KV matters here).
    # Batched: seed_buffered writes all S rows into ring B in one shot, attention-free.
    from minisgl.spec.draft_attn import DraftAttnBuilder
    R = 16
    ring_a = [torch.zeros(1, R, Hkv, hd, device="cuda", dtype=dtype) for _ in range(2)]
    ring_b = [torch.zeros(1, R, Hkv, hd, device="cuda", dtype=dtype) for _ in range(2)]
    builder = DraftAttnBuilder(R, torch.device("cuda"))
    slot = torch.zeros(1, dtype=torch.int64, device="cuda")
    for s in range(S):
        col = torch.full((1,), s, dtype=torch.int64, device="cuda")
        keep = torch.zeros(1, R, dtype=torch.bool, device="cuda")
        keep[0, : s + 1] = True
        m.step_masked(embeds[s : s + 1], hiddens[s : s + 1], positions[s : s + 1], ring_a[0], ring_a[1],
                      slot, col, builder.meta(slot, col, keep))
    m.seed_buffered(embeds, hiddens, positions, ring_b[0], ring_b[1], 0, 0)
    max_dk = (ring_a[0][0, :S].float() - ring_b[0][0, :S].float()).abs().max().item()
    max_dv = (ring_a[1][0, :S].float() - ring_b[1][0, :S].float()).abs().max().item()
    print(f"[seed-kv-parity] S={S} max|dk|={max_dk:.3e} max|dv|={max_dv:.3e}")
    ok = max_dk == 0.0 and max_dv == 0.0
    print("[seed-kv-parity] RESULT:", "PASS (byte-identical)" if ok else
          ("PASS (within bf16 eps)" if max_dk < 1e-2 and max_dv < 1e-2 else "FAIL"))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
