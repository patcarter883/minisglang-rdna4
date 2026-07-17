"""Measure the whitened-GTE cosine gap between PARAPHRASES of the same subject and DISTINCT subjects,
so the write-side semantic-dedup threshold (MINISGL_CAM_WRITE_DEDUP_TAU) can be set in the gap:
high enough that distinct facts never silently merge (data loss), low enough that paraphrase
re-remembers collapse onto one entry. CPU-only; run in the serve image, no GPU lease.

    docker run --rm -v "$PWD":/engine -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 --entrypoint bash minisgl-rdna4:lean -lc \
      'source /app/.venv/bin/activate && PYTHONPATH=/engine/python python /engine/tools/cam_gte_dedup_tau_measure.py'
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from minisgl.cam.gte_encoder import GTEEncoder  # noqa: E402

ART = os.environ.get("MINISGL_CAM_GTE_WHITEN", "/engine/.cam_gte/gte_whiten.pkl")

# Paraphrase groups: within a group, all refer to the SAME subject (should be high-cosine = merge);
# across groups, subjects are distinct (should be low-cosine = keep separate). Includes a couple of
# deliberately-related-but-distinct pairs (Mozart vs Leopold Mozart; France vs Paris) to probe the
# wrong-merge risk at the boundary.
GROUPS = {
    "mozart":    ["Wolfgang Amadeus Mozart", "the composer Mozart", "Mozart the Austrian musician",
                  "Amadeus Mozart", "Mozart, the classical composer"],
    "leopold":   ["Leopold Mozart", "Mozart's father Leopold", "the violinist Leopold Mozart"],
    "einstein":  ["Albert Einstein", "the physicist Einstein", "Einstein, the relativity theorist",
                  "A. Einstein"],
    "darrieux":  ["Danielle Darrieux", "the actress Danielle Darrieux", "Ms. Darrieux",
                  "the French film star Darrieux"],
    "france":    ["France", "the French Republic", "the country France"],
    "paris":     ["Paris", "the city of Paris", "Paris, capital of France"],
    "kilimanjaro": ["Mount Kilimanjaro", "Kilimanjaro, the mountain", "the Kilimanjaro volcano"],
    "everest":   ["Mount Everest", "the peak Everest", "Everest, the world's highest mountain"],
    "violin":    ["the violin", "a violin", "the violin instrument"],
    "beethoven": ["Ludwig van Beethoven", "the composer Beethoven", "Beethoven the German composer"],
}

art = pickle.load(open(ART, "rb"))
mu = torch.tensor(np.asarray(art["mu"], np.float32))
W = torch.tensor(np.asarray(art["W"], np.float32))
enc = GTEEncoder(art.get("model", "lightonai/GTE-ModernColBERT-v1"))
print(f"artifact: model={art.get('model')} dim={art.get('dim')} nn {art.get('nn_raw')}->{art.get('nn_whitened')}", flush=True)


def key(text):
    g = enc.encode([text])[0]
    return F.normalize((g - mu) @ W, dim=-1)


keys = {g: torch.stack([key(t) for t in ts]) for g, ts in GROUPS.items()}
names = list(GROUPS)

para, cross = [], []
for g in names:
    K = keys[g]                                            # [n,d]
    S = (K @ K.t())                                        # within-group cosines
    iu = torch.triu_indices(len(K), len(K), offset=1)
    para += S[iu[0], iu[1]].tolist()                       # paraphrase pairs (same subject)
allkeys = torch.cat([keys[g] for g in names])
labels = [g for g in names for _ in GROUPS[g]]
Sall = allkeys @ allkeys.t()
for i in range(len(allkeys)):
    for j in range(i + 1, len(allkeys)):
        if labels[i] != labels[j]:
            cross.append(float(Sall[i, j]))               # distinct-subject pairs

para, cross = np.array(para), np.array(cross)


def stats(a):
    return f"min {a.min():.3f}  p10 {np.percentile(a,10):.3f}  med {np.median(a):.3f}  p90 {np.percentile(a,90):.3f}  max {a.max():.3f}"


print(f"\nPARAPHRASE (same subject, n={len(para)}):  {stats(para)}")
print(f"DISTINCT   (diff subject, n={len(cross)}):  {stats(cross)}")

# The safe threshold sits above the distinct-subject tail and below the paraphrase body. Report the
# separation and how each candidate tau would score (merge-recall on paraphrases / wrong-merge on distinct).
print("\ntau   merge%(para, want high)   wrongmerge%(distinct, want 0)")
for tau in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92]:
    print(f"{tau:.2f}   {100*(para>=tau).mean():5.1f}%                {100*(cross>=tau).mean():5.1f}%")

# nearest distinct pair (the wrong-merge hazard) at a few taus
order = np.argsort(cross)[::-1][:5]
print("\ntop distinct-subject collisions (wrong-merge hazards):")
pairs = [(labels[i], labels[j], float(Sall[i, j])) for i in range(len(allkeys))
         for j in range(i + 1, len(allkeys)) if labels[i] != labels[j]]
for a, b, c in sorted(pairs, key=lambda x: -x[2])[:6]:
    print(f"  {c:.3f}  {a} <> {b}")
