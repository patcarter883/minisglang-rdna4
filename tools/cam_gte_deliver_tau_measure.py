"""Measure the whitened-GTE DELIVERY precision/recall curve, so MINISGL_CAM_DELIVER_TAU is set from
data rather than guessed. Delivery must fire when a query paraphrases a STORED subject (recall) and stay
silent when the query is a DIFFERENT subject (precision) — including a distinct-but-related one
("Leopold Mozart" when only "Mozart" is stored), which is the false-fire hazard the tau guards.

Two probes, both CPU (the GTE encoder runs on CPU; no GPU lease):
  1. RECALL   — hand-crafted paraphrase groups: query each paraphrase against its stored subject.
  2. PRECISION — at scale: store N real CounterFact subjects, query M held-out distinct subjects, and
                 also the crafted related-entity pairs, measuring how often the nearest stored key
                 exceeds tau (a false delivery).

    docker run --rm -v "$PWD":/engine -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 --entrypoint bash minisgl-rdna4:lean -lc \
      'PYTHONPATH=/engine/python /opt/venv/bin/python /engine/tools/cam_gte_deliver_tau_measure.py'
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from minisgl.cam.gte_encoder import GTEEncoder  # noqa: E402

ART = os.environ.get("MINISGL_CAM_GTE_WHITEN", "/engine/.cam_gte/gte_whiten.pkl")
CF = os.environ.get("CF_JSON", "/engine/data/counterfact.json")
N_STORE = int(os.environ.get("N_STORE", "1000"))     # distinct subjects held in the store for the precision probe
M_QUERY = int(os.environ.get("M_QUERY", "1000"))     # held-out distinct subjects queried (must NOT deliver)

# Recall probe: within a group all refer to the SAME subject (the first is the STORED form, the rest are
# query paraphrases). Cross-group are distinct. A couple of related-entity pairs probe the false-fire edge.
GROUPS = {
    "mozart":   ["Wolfgang Amadeus Mozart", "the composer Mozart", "Mozart the Austrian musician",
                 "Amadeus Mozart", "Mozart, the classical composer"],
    "einstein": ["Albert Einstein", "the physicist Einstein", "Einstein, the relativity theorist", "A. Einstein"],
    "darrieux": ["Danielle Darrieux", "the actress Danielle Darrieux", "Ms. Darrieux",
                 "the French film star Darrieux"],
    "everest":  ["Mount Everest", "the peak Everest", "Everest, the world's highest mountain"],
    "beethoven":["Ludwig van Beethoven", "the composer Beethoven", "Beethoven the German composer"],
    "tokyo":    ["Tokyo", "the city of Tokyo", "Tokyo, capital of Japan"],
}
# Distinct-but-RELATED (query, stored-subject-it-must-NOT-hit): the precision hazard.
RELATED = [("Leopold Mozart", "Wolfgang Amadeus Mozart"), ("Paris", "Tokyo"),
           ("Mount Kilimanjaro", "Mount Everest")]

import pickle  # noqa: E402
art = pickle.load(open(ART, "rb"))
mu = torch.tensor(np.asarray(art["mu"], np.float32))
W = torch.tensor(np.asarray(art["W"], np.float32))
enc = GTEEncoder(art.get("model", "lightonai/GTE-ModernColBERT-v1"))


def keys(texts):
    G = torch.stack([enc.encode([t])[0] for t in texts])
    return F.normalize((G - mu) @ W, dim=-1)


# ---- RECALL: query paraphrase vs its own stored subject -------------------------------------------
recall_cos = []
for ts in GROUPS.values():
    K = keys(ts)
    stored = K[0]
    for q in K[1:]:
        recall_cos.append(float(q @ stored))
recall_cos = np.array(recall_cos)

# ---- PRECISION at scale: store N real subjects, query M held-out distinct ones --------------------
d = json.load(open(CF))
subs = sorted({(r.get("requested_rewrite") or {}).get("subject") or r.get("subject") for r in d} - {None})
rng = np.random.default_rng(1); rng.shuffle(subs)
store_subs = subs[:N_STORE]
query_subs = subs[N_STORE:N_STORE + M_QUERY]
print(f"[deliver-tau] store N={len(store_subs)}  query M={len(query_subs)} (held-out distinct)", flush=True)

BS = 64
def encode_all(lst):
    return torch.cat([keys(lst[i:i + BS]) for i in range(0, len(lst), BS)])

Kstore = encode_all(store_subs)                       # [N,d]
Kquery = encode_all(query_subs)                       # [M,d]
# nearest stored key for each held-out distinct query (these must NOT deliver)
nn_distinct = (Kquery @ Kstore.t()).max(1).values.numpy()

# related-entity probe against the crafted stored subjects
stored_map = {g: keys([ts[0]])[0] for g, ts in GROUPS.items()}
gname_by_sub = {ts[0]: g for g, ts in GROUPS.items()}
related_cos = []
for q, stored_sub in RELATED:
    kq = keys([q])[0]
    ks = stored_map[gname_by_sub[stored_sub]]
    related_cos.append((q, stored_sub, float(kq @ ks)))


def stats(a):
    return (f"min {a.min():.3f}  p10 {np.percentile(a,10):.3f}  med {np.median(a):.3f}  "
            f"p90 {np.percentile(a,90):.3f}  p99 {np.percentile(a,99):.3f}  max {a.max():.3f}")


print(f"\nRECALL   query-vs-stored paraphrase (n={len(recall_cos)}): {stats(recall_cos)}")
print(f"PRECISION nearest-stored for held-out DISTINCT (n={len(nn_distinct)}): {stats(nn_distinct)}")
print("\nrelated-entity false-fire probe (cos to the WRONG stored subject):")
for q, s, c in related_cos:
    print(f"  {c:.3f}  {q!r} -> nearest stored {s!r}")

print("\ntau    recall%(deliver paraphrase)   falsefire%(distinct@scale)   related-fires")
for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.85]:
    rf = sum(1 for _, _, c in related_cos if c >= tau)
    print(f"{tau:.2f}   {100*(recall_cos>=tau).mean():5.1f}%                     "
          f"{100*(nn_distinct>=tau).mean():6.2f}%                    {rf}/{len(related_cos)}")
