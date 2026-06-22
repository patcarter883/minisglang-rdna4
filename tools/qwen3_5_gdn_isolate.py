"""Phase 3d-4 — isolate ONE GDN layer: minisgl QwenGatedDeltaNet vs HF Qwen3_5GatedDeltaNet.

Builds HF's layer-0 GDN module (random-init is fine — we copy its weights into minisgl's module),
feeds an identical input through both for a single fresh prefill sequence, and compares the mixer
output (cos-sim / rel-err). No checkpoint needed; this tests the COMPUTE wiring, not loading.

GPU via the lease (both use GPU kernels). Run in the combined image.
"""
from __future__ import annotations

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from minisgl.gdn.layer import QwenGatedDeltaNet

MODEL = "Qwen/Qwen3.5-4B"
DEV = torch.device("cuda")


def main() -> None:
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL)
    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    torch.set_default_dtype(torch.bfloat16)

    hf = Qwen3_5GatedDeltaNet(tc, layer_idx=0).to(DEV).eval()
    # random but realistic gating/decay params
    with torch.no_grad():
        hf.A_log.copy_(torch.log(torch.empty(tc.linear_num_value_heads, device=DEV).uniform_(1, 16)))
        hf.dt_bias.uniform_(-2, 2)

    ms = QwenGatedDeltaNet(
        hidden_size=tc.hidden_size,
        num_k_heads=tc.linear_num_key_heads,
        num_v_heads=tc.linear_num_value_heads,
        head_k_dim=tc.linear_key_head_dim,
        head_v_dim=tc.linear_value_head_dim,
        conv_kernel_size=tc.linear_conv_kernel_dim,
        eps=tc.rms_norm_eps,
        dtype=torch.bfloat16,
        device=DEV,
    )
    # ---- copy HF weights into minisgl module ----
    with torch.no_grad():
        ms.in_proj_qkvz.weight.copy_(torch.cat([hf.in_proj_qkv.weight, hf.in_proj_z.weight], dim=0))
        ms.in_proj_ba.weight.copy_(torch.cat([hf.in_proj_b.weight, hf.in_proj_a.weight], dim=0))
        ms.conv1d_weight.copy_(hf.conv1d.weight)  # (conv_dim,1,kernel)
        ms.A_log.copy_(hf.A_log.float())
        ms.dt_bias.copy_(hf.dt_bias.float())
        ms.norm.weight.copy_(hf.norm.weight)
        ms.out_proj.weight.copy_(hf.out_proj.weight)

    T = 6
    x = torch.randn(1, T, tc.hidden_size, device=DEV, dtype=torch.bfloat16)

    with torch.no_grad():
        hf_out = hf(x)[0]  # (T, hidden) after squeeze? returns (batch, seq, hidden)
    if hf_out.dim() == 3:
        hf_out = hf_out[0]

    # minisgl fresh-prefill plumbing: slot 1 (slot 0 is NULL), zeroed state
    num_slots = 4
    conv_dim = ms.conv_dim
    conv_state = torch.zeros(num_slots, conv_dim, tc.linear_conv_kernel_dim - 1, device=DEV, dtype=torch.bfloat16)
    ssm_state = torch.zeros(
        num_slots, tc.linear_num_value_heads, tc.linear_value_head_dim, tc.linear_key_head_dim,
        device=DEV, dtype=torch.bfloat16,
    )
    qsl = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    state_idx = torch.tensor([1], dtype=torch.int32, device=DEV)
    has_init = torch.tensor([False], device=DEV)
    ms.warmup_conv(8)
    with torch.no_grad():
        ms_out = ms.forward_prefill(x[0], conv_state, ssm_state, qsl, state_idx, has_init)

    a, b = ms_out.float(), hf_out.float()
    print(f"ms_out {tuple(ms_out.shape)} hf_out {tuple(hf_out.shape)}")
    for t in range(T):
        cos = torch.nn.functional.cosine_similarity(a[t], b[t], dim=0).item()
        rel = ((a[t] - b[t]).norm() / (b[t].norm() + 1e-9)).item()
        print(f"  t={t}: cos={cos:.5f} rel={rel:.3e} |ms|={a[t].norm():.3f} |hf|={b[t].norm():.3f}")
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    print(f"OVERALL cos={cos:.5f} rel={((a-b).norm()/b.norm()).item():.3e}")


if __name__ == "__main__":
    main()
