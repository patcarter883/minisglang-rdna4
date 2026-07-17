"""Does TRANSPARENT read (auto-augment on /v1/chat) already surface the right fact for a multi-fact
entity? The scheduler extracts candidate spans from the prompt — proper-noun spans + 2..6-word content
windows — and cosine-matches each against the store's composite subject keys ("Mozart birthplace"). A
natural question ("Where was Mozart born?") yields windows like "Mozart born" that may match the RIGHT
composite. This replicates the scheduler's candidate extraction faithfully and measures, per question:
whether the top match above deliver_tau is the intended fact, and any cross-fire. CPU only; no GPU lease.

    docker run --rm -v "$PWD":/engine -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 --entrypoint bash minisgl-rdna4:lean -lc \
      'PYTHONPATH=/engine/python /opt/venv/bin/python /engine/tools/cam_gte_transparent_multifact_measure.py'
"""
import os
import pickle
import re

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from minisgl.cam.gte_encoder import GTEEncoder  # noqa: E402

ART = os.environ.get("MINISGL_CAM_GTE_WHITEN", "/engine/.cam_gte/gte_whiten.pkl")
TAU = float(os.environ.get("TAU", "0.70"))

# Store: (entity, relation, object). Key text = "entity relation" (matches _compose_addr).
STORE = [
    ("Wolfgang Amadeus Mozart", "birthplace", "Salzburg"),
    ("Wolfgang Amadeus Mozart", "birth year", "1756"),
    ("Wolfgang Amadeus Mozart", "instrument", "Piano"),
    ("Marie Curie", "field", "Radioactivity"),
    ("Marie Curie", "birthplace", "Warsaw"),
]
# Natural chat prompts -> the (entity, relation) they SHOULD surface (or None = should stay quiet).
QUERIES = [
    ("Where was Mozart born?", ("Wolfgang Amadeus Mozart", "birthplace")),
    ("What year was Mozart born?", ("Wolfgang Amadeus Mozart", "birth year")),
    ("What instrument did Mozart play?", ("Wolfgang Amadeus Mozart", "instrument")),
    ("Tell me about Marie Curie's field of research.", ("Marie Curie", "field")),
    ("Which city was Marie Curie born in?", ("Marie Curie", "birthplace")),
    ("What is the capital of France?", None),                 # nothing stored -> must stay quiet
    ("How tall is Mount Everest?", None),
]

# Interrogatives carry the RELATION signal for retrieval ("WHERE was X born" vs "WHEN was X born").
# The current _STOP trims them off span boundaries, collapsing both to "X born" -> location/date
# ambiguity. KEEP_INTERROGATIVES=1 drops them from the trim set so the disambiguator survives.
_INTERROG = {"what", "which", "who", "whom", "whose", "where", "when", "why", "how"}
_STOP_BASE = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was",
              "were", "be", "does", "do", "did", "that", "this", "these", "those", "it", "its", "his",
              "her", "their", "your", "my", "you", "he", "she", "they", "we", "as", "by", "with",
              "from", "about", "tell", "me", "please", "can", "could", "would", "will", "s"}
# SURGICAL fix (KEEP_INTERROGATIVES=1): interrogatives stay in _STOP for the TRAILING trim and the
# all-stop check, but are allowed at the LEADING boundary so "Where was Mozart born" survives (relation
# signal) while "Mozart born where" still trims to "Mozart born" (trailing noise).
_STOP = _STOP_BASE | _INTERROG
_LEAD_STOP = _STOP_BASE if os.environ.get("KEEP_INTERROGATIVES") == "1" else _STOP


def candidates(text):
    """Faithful copy of scheduler._cam_retrieve candidate extraction (with the surgical leading rule)."""
    cands = set()
    for m in re.finditer(r"[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,4}", text):
        cands.add(m.group(0))
    budget = 160
    for clause in re.split(r"[.?!,;:\n]+", text):
        if budget <= 0:
            break
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.\-]*", clause)[:budget]
        budget -= len(words)
        n = len(words)
        for i in range(n):
            for L in range(2, 7):
                if i + L > n:
                    break
                span = words[i:i + L]
                if span[0].lower() in _LEAD_STOP or span[-1].lower() in _STOP:
                    continue
                if all(w.lower() in _STOP for w in span):
                    continue
                cands.add(" ".join(span))
    return sorted(cands, key=len, reverse=True)


art = pickle.load(open(ART, "rb"))
mu = torch.tensor(np.asarray(art["mu"], np.float32))
W = torch.tensor(np.asarray(art["W"], np.float32))
enc = GTEEncoder(art.get("model", "lightonai/GTE-ModernColBERT-v1"))


def key(text):
    g = enc.encode([text])[0]
    return F.normalize((g - mu) @ W, dim=-1)


Kstore = torch.stack([key(f"{e} {r}") for e, r, _ in STORE])   # composite keys

ok = 0
for q, want in QUERIES:
    cands = candidates(q)
    best = None  # (cos, store_idx, cand)
    for c in cands:
        sims = Kstore @ key(c)
        j = int(sims.argmax()); s = float(sims[j])
        if best is None or s > best[0]:
            best = (s, j, c)
    fired = best[0] >= TAU
    hit_ent, hit_rel = (STORE[best[1]][0], STORE[best[1]][1]) if fired else (None, None)
    if want is None:
        good = not fired
        verdict = "quiet" if good else f"FALSE-FIRE -> {hit_ent} {hit_rel}"
    else:
        good = fired and (hit_ent, hit_rel) == want
        verdict = (f"{hit_ent} {hit_rel} ({STORE[best[1]][2]})" if fired else "SILENT (below tau)")
    ok += good
    print(f"  [{'PASS' if good else 'FAIL'}] {q!r}")
    print(f"        want={want}  got={verdict}  via span {best[2]!r} @ cos {best[0]:.3f}")

print(f"\nTRANSPARENT MULTI-FACT: {ok}/{len(QUERIES)}")
