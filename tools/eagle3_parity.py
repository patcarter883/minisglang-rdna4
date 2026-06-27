"""Offline numerical parity for the EAGLE3 draft forward.

Runs INSIDE vllm22-w4a8:combined (torch works there). Compares minisgl's GLMEagle3DraftModel.step
against a self-contained reference forward built directly from the raw safetensors weights with
explicit matmuls (the canonical llama_eagle3 midlayer math). Feeds BOTH the SAME fixed-random aux +
token, so any divergence is a bug in the minisgl draft forward (not the aux capture or the serve).

Single rank (TP=1) — the draft is replicated, so a TP=1 forward is the per-rank computation.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/engine/python")
sys.path.insert(0, "/engine")

CKPT = os.environ["EAGLE3_CKPT"]  # local snapshot folder


def load_sd():
    import safetensors.torch as st
    import glob
    f = glob.glob(os.path.join(CKPT, "*.safetensors"))[0]
    return st.load_file(f, device="cuda")


def reference_step(sd, embed_e, hidden, positions, head_dim, num_heads, num_kv_heads, eps, theta):
    """Canonical llama_eagle3 midlayer single-token step (one token, no KV history beyond itself)."""
    def rms(x, w):
        xf = x.float()
        v = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(v + eps)).to(x.dtype) * w

    H, Hkv, hd = num_heads, num_kv_heads, head_dim
    a = rms(embed_e, sd["midlayer.input_layernorm.weight"])
    b = rms(hidden, sd["midlayer.hidden_norm.weight"])
    widened = torch.cat([a, b], dim=-1)  # [B, 2*hidden]
    q = F.linear(widened, sd["midlayer.self_attn.q_proj.weight"]).view(-1, H, hd)
    k = F.linear(widened, sd["midlayer.self_attn.k_proj.weight"]).view(-1, Hkv, hd)
    v = F.linear(widened, sd["midlayer.self_attn.v_proj.weight"]).view(-1, Hkv, hd)

    # NeoX RoPE (full rotary over head_dim).
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device="cuda").float() / hd))
    ang = positions.float().unsqueeze(-1) * inv.unsqueeze(0)  # [B, hd/2]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1).unsqueeze(1)  # [B,1,hd]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1).unsqueeze(1)

    def rope(x):
        xf = x.float()
        d = hd // 2
        x1, x2 = xf[..., :d], xf[..., d:]
        rot = torch.cat([-x2, x1], dim=-1)
        return (xf * cos + rot * sin).to(x.dtype)

    q = rope(q); k = rope(k)
    group = H // Hkv
    kx = k.repeat_interleave(group, dim=1)  # [B,H,hd]
    vx = v.repeat_interleave(group, dim=1)
    # single token attends only itself -> softmax over 1 == identity -> attn out = v.
    scale = hd ** -0.5
    score = (q * kx).sum(-1, keepdim=True) * scale  # [B,H,1]
    prob = score.softmax(dim=-1)
    attn = (prob * vx)  # [B,H,hd]
    attn_out = F.linear(attn.reshape(attn.shape[0], H * hd), sd["midlayer.self_attn.o_proj.weight"])
    residual = hidden + attn_out
    normed = rms(residual, sd["midlayer.post_attention_layernorm.weight"])
    gate = F.linear(normed, sd["midlayer.mlp.gate_proj.weight"])
    up = F.linear(normed, sd["midlayer.mlp.up_proj.weight"])
    mlp = F.linear(F.silu(gate.float()).to(up.dtype) * up, sd["midlayer.mlp.down_proj.weight"])
    out_hidden = residual + mlp
    logits = F.linear(rms(out_hidden, sd["norm.weight"]), sd["lm_head.weight"])
    return logits, out_hidden


def main():
    torch.manual_seed(0)
    # minimal TP setup for minisgl layers
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from minisgl.distributed import set_tp_info
    set_tp_info(rank=0, size=1)
    from minisgl.layers import set_rope_device
    set_rope_device(torch.device("cuda"))

    sd = load_sd()
    dtype = torch.bfloat16
    hidden = 2048; H = 16; Hkv = 4; hd = 128; eps = 1e-5; theta = 1e6
    from minisgl.models.glm_eagle3 import GLMEagle3DraftModel
    with torch.device("cuda"):
        m = GLMEagle3DraftModel(
            hidden_size=hidden, intermediate_size=8192, num_heads=H, num_kv_heads=Hkv,
            head_dim=hd, num_aux_layers=3, draft_vocab_size=32000, target_vocab_size=154880,
            rms_norm_eps=eps, rope_theta=theta, max_position=4096,
        )
    # load weights into minisgl model (same mapping as the proposer)
    kmap = {
        "fc.weight": ("fc", "weight"),
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

    B = 2
    aux = torch.randn(B, 3, hidden, device="cuda", dtype=dtype) * 0.5
    embed_e = torch.randn(B, hidden, device="cuda", dtype=dtype) * 0.1
    positions = torch.tensor([20, 21], device="cuda", dtype=torch.int32)

    fused = m.fuse_aux(aux)  # [B, hidden]
    cache = []
    mlogits, mhid = m.step(embed_e, fused, positions, cache)

    # reference: fc then step
    rfused = F.linear(aux.reshape(B, -1), sd["fc.weight"].to(dtype))
    rlogits, rhid = reference_step(sd, embed_e, rfused, positions, hd, H, Hkv, eps, theta)

    dl = (mlogits.float() - rlogits.float()).abs().max().item()
    dh = (mhid.float() - rhid.float()).abs().max().item()
    df = (fused.float() - rfused.float()).abs().max().item()
    cos = F.cosine_similarity(mlogits.float().flatten(), rlogits.float().flatten(), dim=0).item()
    print(f"[parity] fc max|d|={df:.4e} hidden max|d|={dh:.4e} logits max|d|={dl:.4e} cos={cos:.6f}")
    print(f"[parity] minisgl argmax={mlogits.argmax(-1).tolist()} ref argmax={rlogits.argmax(-1).tolist()}")
    ok = cos > 0.999 and mlogits.argmax(-1).tolist() == rlogits.argmax(-1).tolist()
    print("[parity] RESULT:", "PASS" if ok else "FAIL")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
