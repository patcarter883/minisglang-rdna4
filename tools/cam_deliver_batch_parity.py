"""CPU parity: the vectorised _subj_keys_batch + GPU-argmax deliver path must be DECISION-IDENTICAL to
the original per-candidate loop (per-row _subj_key, argmax().item(), tau gate). No GPU / no checkpoint —
constructs a bare CAMMemory via __new__ with a random embed table and a random key matrix.
"""
import sys
sys.path.insert(0, "/engine/python")
import torch, torch.nn.functional as F
from minisgl.cam.memory import CAMMemory

torch.manual_seed(0)
V, H, M, C = 4000, 128, 40, 300
embed = torch.randn(V, H)
Kmat = F.normalize(torch.randn(M, H), dim=-1)          # stored unit-norm keys
tau = 0.70

m = CAMMemory.__new__(CAMMemory)
m._embed_w = embed
m._gte = None
m._decode = None
from collections import OrderedDict
m._key_cache = OrderedDict()
m._key_cache_cap = 20000

# reference per-candidate _subj_key (base-embed): normalize(mean(embed[ids]))
def ref_key(ids):
    e = F.embedding(torch.tensor([ids]), embed).float()
    return F.normalize(e.mean(1), dim=-1)[0]

subs = [[int(x) for x in torch.randint(0, V, (int(torch.randint(1, 6, (1,))),))] for _ in range(C)]

# --- key-build parity ---
Q_ref = torch.stack([ref_key(s) for s in subs])
Q_bat = m._subj_keys_batch(subs)                        # miss path (cache cold)
Q_cache = m._subj_keys_batch(subs, texts=[str(s) for s in subs])  # populate cache
Q_hit = m._subj_keys_batch(subs, texts=[str(s) for s in subs])    # all cache hits
kd = (Q_ref - Q_bat).abs().max().item()
kd_hit = (Q_bat - Q_hit).abs().max().item()
print(f"key-build max|Δ| vs per-candidate ref = {kd:.2e}")
print(f"cache-hit  max|Δ| vs miss             = {kd_hit:.2e}")

# --- decision parity: argmax + tau gate ---
sims_ref = Q_ref @ Kmat.t()
ref_idx, ref_ok = [], []
for i in range(C):
    j = int(sims_ref[i].argmax().item())
    ref_ok.append(sims_ref[i, j].item() >= tau); ref_idx.append(j)

sims_bat = Q_bat @ Kmat.t()
vals, idx = sims_bat.max(dim=1)
bat_ok = (vals >= tau).tolist(); bat_idx = idx.tolist()

idx_mismatch = sum(1 for i in range(C) if ref_ok[i] and bat_ok[i] and ref_idx[i] != bat_idx[i])
gate_mismatch = sum(1 for i in range(C) if ref_ok[i] != bat_ok[i])
print(f"tau-gate decision mismatches = {gate_mismatch}/{C}")
print(f"argmax mismatches (both pass)= {idx_mismatch}/{C}")
print("PARITY", "OK" if (gate_mismatch == 0 and idx_mismatch == 0 and kd < 1e-4 and kd_hit == 0) else "FAIL")
