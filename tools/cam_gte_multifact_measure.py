"""Scope multi-fact-per-entity: measure whitened-GTE cosine for
  PARAPHRASE  — same entity + same relation, different wording  (must DELIVER + may dedup-merge)
  SIBLING     — same entity, DIFFERENT relation                 (must NOT dedup-merge; must deliver the
                                                                  RIGHT relation, not a sibling)
  CROSS       — different entity                                 (must NOT deliver)
under two key constructions:
  (A) key = bare entity phrase ("Mozart")                 -> siblings identical (multi-fact impossible)
  (B) key = relation phrase   ("Mozart's birthplace")     -> siblings separate iff this holds

If (B) sibling cosine sits comfortably below the dedup threshold (0.82) AND below the paraphrase band,
multi-fact already works via relation-phrase keys (dedup won't clobber siblings, delivery disambiguates).
Otherwise an explicit (entity, relation) store key is needed. CPU only; no GPU lease.

    docker run --rm -v "$PWD":/engine -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 --entrypoint bash minisgl-rdna4:lean -lc \
      'PYTHONPATH=/engine/python /opt/venv/bin/python /engine/tools/cam_gte_multifact_measure.py'
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from minisgl.cam.gte_encoder import GTEEncoder  # noqa: E402

ART = os.environ.get("MINISGL_CAM_GTE_WHITEN", "/engine/.cam_gte/gte_whiten.pkl")

# entity -> relation -> [paraphrases of (entity, relation)]. First phrasing is the "stored" key form.
DATA = {
    "Mozart": {
        "birthplace": ["Mozart's birthplace", "where Mozart was born", "the city Mozart was born in"],
        "birth year": ["Mozart's birth year", "the year Mozart was born", "when Mozart was born"],
        "instrument": ["Mozart's instrument", "the instrument Mozart played", "what Mozart played"],
    },
    "Einstein": {
        "field": ["Einstein's field of work", "the field Einstein worked in", "what Einstein studied"],
        "birthplace": ["Einstein's birthplace", "where Einstein was born", "the city Einstein was born in"],
        "prize": ["Einstein's Nobel prize", "the prize Einstein won", "what award Einstein received"],
    },
    "Tokyo": {
        "country": ["Tokyo's country", "the country Tokyo is in", "which nation Tokyo belongs to"],
        "population": ["Tokyo's population", "how many people live in Tokyo", "the number of Tokyo residents"],
        "river": ["Tokyo's main river", "the river running through Tokyo", "which river is in Tokyo"],
    },
}

art = pickle.load(open(ART, "rb"))
mu = torch.tensor(np.asarray(art["mu"], np.float32))
W = torch.tensor(np.asarray(art["W"], np.float32))
enc = GTEEncoder(art.get("model", "lightonai/GTE-ModernColBERT-v1"))


def key(text):
    g = enc.encode([text])[0]
    return F.normalize((g - mu) @ W, dim=-1)


def stats(a):
    a = np.array(a)
    return f"n={len(a):3d}  min {a.min():.3f}  p10 {np.percentile(a,10):.3f}  med {np.median(a):.3f}  p90 {np.percentile(a,90):.3f}  max {a.max():.3f}"


for mode in ("A", "B"):
    para, sib, cross = [], [], []
    # build keys per mode
    ekeys = {}   # entity -> relation -> [keys]
    for ent, rels in DATA.items():
        ekeys[ent] = {}
        for rel, paras in rels.items():
            if mode == "A":
                ekeys[ent][rel] = [key(ent) for _ in paras]                 # bare entity
            else:
                ekeys[ent][rel] = [key(f"{ent}'s {rel}")] + [key(p) for p in paras[1:]]  # relation phrase + para queries
    for ent, rels in ekeys.items():
        for rel, ks in rels.items():
            stored = ks[0]
            for q in ks[1:]:
                para.append(float(q @ stored))                              # same entity+relation
        rel_list = list(rels)
        for i in range(len(rel_list)):
            for j in range(len(rel_list)):
                if i != j:
                    sib.append(float(rels[rel_list[i]][0] @ rels[rel_list[j]][0]))  # same entity, diff relation
    ents = list(ekeys)
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            for r1 in ekeys[ents[i]].values():
                for r2 in ekeys[ents[j]].values():
                    cross.append(float(r1[0] @ r2[0]))                      # different entity
    label = "A: key = bare entity" if mode == "A" else "B: key = relation phrase"
    print(f"\n==== {label} ====")
    print(f"PARAPHRASE (same entity+relation): {stats(para)}")
    print(f"SIBLING    (same entity, diff rel): {stats(sib)}")
    print(f"CROSS      (different entity):      {stats(cross)}")
    if mode == "B":
        sib = np.array(sib)
        print(f"  sibling >= dedup_tau 0.82 (would WRONGLY merge): {100*(sib>=0.82).mean():.1f}%")

        # DELIVERY argmax-among-siblings: store every entity's relations (canonical), query with each
        # paraphrase, check the nearest stored key is the RIGHT (entity,relation) AND clears deliver_tau.
        TAU = 0.70
        rows, hit, fired, wrong = [], 0, 0, 0
        for ent, rels in ekeys.items():
            for r in DATA[ent]:
                rows.append((ent, r, key(f"{ent}'s {r}")))              # canonical stored keys
        Kall = torch.stack([r[2] for r in rows])
        n = 0
        for ent, rels in DATA.items():
            for rel, paras in rels.items():
                for q in paras[1:]:                                     # paraphrase queries (not the stored form)
                    n += 1
                    sims = Kall @ key(q)
                    j = int(sims.argmax()); top = float(sims[j])
                    right = (rows[j][0] == ent and rows[j][1] == rel)
                    if top >= TAU:
                        fired += 1
                        if right: hit += 1
                        else:
                            wrong += 1
                            print(f"    WRONG-RELATION: {q!r} -> {rows[j][0]}'s {rows[j][1]} (cos {top:.3f}), want {ent}'s {rel}")
        print(f"\n  DELIVERY among siblings (n={n} paraphrase queries, deliver_tau {TAU}):")
        print(f"    right relation delivered: {hit}/{n} ({100*hit/n:.0f}%)   "
              f"wrong-relation fires: {wrong}   silent (below tau): {n-fired}")
