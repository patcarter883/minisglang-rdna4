"""Replay the ENGINE's actual layer-0 GDN input through HF's real GDN layer 0, compare to the
engine's saved output. Decides whether the engine's GDN call is faithful (bug is plumbing) or not.
GPU via the lease. Needs /engine/tmp/gdn_l0.pt from a MINISGL_GDN_SAVE_L0 run."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-4B"
DEV = torch.device("cuda")

d = torch.load("/engine/tmp/gdn_l0.pt")
x = d["x"].to(DEV, torch.bfloat16)  # (T, hidden) engine's normed input to GDN L0
ms_out = d["out"].to(DEV).float()  # (T, hidden) engine's GDN L0 output
print(f"x {tuple(x.shape)} ms_out {tuple(ms_out.shape)}")

from transformers import AutoTokenizer

hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
h = hf.model.language_model if hasattr(hf.model, "language_model") else hf.model
L0 = h.layers[0].linear_attn

# --- check: is the engine's GDN input x == HF's true layer-0 input (input_layernorm(embed))? ---
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(DEV)
print(f"HF token_ids={ids[0].tolist()}")
with torch.no_grad():
    emb = h.embed_tokens(ids)
    true_in = h.layers[0].input_layernorm(emb)[0].float()  # (T, hidden)
c_in = torch.nn.functional.cosine_similarity(x.float().flatten(), true_in.flatten(), dim=0).item()
r_in = ((x.float() - true_in).norm() / true_in.norm()).item()
print(f"engine GDN-input x vs HF input_layernorm(embed): cos={c_in:.5f} rel={r_in:.3e} "
      f"|x|={x.float().norm():.3f} |hf_in|={true_in.norm():.3f}")

with torch.no_grad():
    hf_out = L0(x.unsqueeze(0))  # (1, T, hidden), fresh (cache_params=None)
if isinstance(hf_out, tuple):
    hf_out = hf_out[0]
hf_out = hf_out[0].float()

a, b = ms_out, hf_out
cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
rel = ((a - b).norm() / b.norm()).item()
print(f"GDN L0 engine-out vs HF-on-engine-input: cos={cos:.5f} rel={rel:.3e} "
      f"|ms|={a.norm():.3f} |hf|={b.norm():.3f}")
for t in range(a.shape[0]):
    c = torch.nn.functional.cosine_similarity(a[t], b[t], dim=0).item()
    print(f"  t={t}: cos={c:.5f} |ms|={a[t].norm():.3f} |hf|={b[t].norm():.3f}")
