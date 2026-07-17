"""Fit the whitened-GTE subject-key artifact the CAM serve loads (MINISGL_CAM_GTE_WHITEN).

Uses the SAME encoder the serve runs at request time (minisgl.cam.gte_encoder.GTEEncoder — the
pylate-free transformers path) so the whitening fit and the live keys are self-consistent. Soft-ZCA
whitening pulls distinct subject keys apart (NN-other-cos drops ~0.94 -> ~0.50), which is what makes
paraphrased subjects address the right stored fact. CPU-only; no GPU lease needed.

Run inside the serve image (so transformers/safetensors + the HF cache are present), e.g.:

    docker run --rm -v "$PWD":/engine -v /home/pat/.cache/huggingface:/root/.cache/huggingface \
      -e HF_HUB_OFFLINE=1 -e CF_JSON=/engine/data/counterfact.json -e OUT=/engine/.cam_gte/gte_whiten.pkl \
      --entrypoint bash minisgl-rdna4:lean -lc \
      'source /app/.venv/bin/activate && PYTHONPATH=/engine/python python /engine/tools/cam_gte_whiten_precompute.py'

CF_JSON is any JSON list of records with a subject field (CounterFact's requested_rewrite.subject or a
top-level subject); a larger, more diverse subject set gives a better whitening basis. Saves
{mu, W, model, eps, dim, n_fit, nn_raw, nn_whitened, encoder}.
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import numpy.linalg as la

os.environ.setdefault("HF_HUB_OFFLINE", "1")
DATA = os.environ.get("CF_JSON", "/engine/data/counterfact.json")
OUT = os.environ.get("OUT", "/engine/.cam_gte/gte_whiten.pkl")
N_FIT = int(os.environ.get("N_FIT", "6000"))
BS = int(os.environ.get("BS", "64"))
MODEL = os.environ.get("MODEL", "lightonai/GTE-ModernColBERT-v1")
EPS = float(os.environ.get("EPS", "0.05"))


def fit_whiten(X, eps=EPS):
    mu = X.mean(0)
    Xc = X - mu
    C = (Xc.T @ Xc) / len(X)
    U, S, _ = np.linalg.svd(C)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    return mu, W


def l2n(X):
    return X / (la.norm(X, axis=-1, keepdims=True) + 1e-8)


def nn_other_cos(K):
    Kn = l2n(K)
    S = Kn @ Kn.T
    np.fill_diagonal(S, -1.0)
    return float(S.max(1).mean())


d = json.load(open(DATA))
subs = sorted({(r.get("requested_rewrite") or {}).get("subject") or r.get("subject") for r in d} - {None})
rng = np.random.default_rng(0)
rng.shuffle(subs)
subs = subs[:N_FIT]
print(f"[gte-white] {len(subs)} distinct subjects for the fit", flush=True)

# Import the exact encoder the serve uses (works whether run from /engine or an installed package).
try:
    from minisgl.cam.gte_encoder import GTEEncoder
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
    from minisgl.cam.gte_encoder import GTEEncoder

t0 = time.time()
enc = GTEEncoder(MODEL)
print(f"[gte-white] encoder loaded ({time.time()-t0:.0f}s, dim {enc.dim}); encoding...", flush=True)
G = np.concatenate([enc.encode(subs[i:i + BS]).numpy() for i in range(0, len(subs), BS)]).astype(np.float32)
print(f"[gte-white] encoded {G.shape} in {time.time()-t0:.0f}s", flush=True)

mu, W = fit_whiten(G)
Gw = (G - mu) @ W
sl = slice(0, min(2000, len(G)))
nn_raw = nn_other_cos(G[sl])
nn_wht = nn_other_cos(Gw[sl])
print(f"[gte-white] NN-other-cos (lower=better): raw {nn_raw:.3f} -> whitened {nn_wht:.3f}", flush=True)

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
pickle.dump({"mu": mu.astype(np.float32), "W": W.astype(np.float32),
             "dim": int(G.shape[1]), "eps": EPS, "model": MODEL, "n_fit": len(subs),
             "nn_raw": nn_raw, "nn_whitened": nn_wht, "encoder": "transformers-direct"},
            open(OUT, "wb"))
print(f"[gte-white] saved -> {OUT} (dim {G.shape[1]}); DONE", flush=True)
