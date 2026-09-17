# CAM delivery-index scaling — measured, and where ANN fits

The CAM pointer delivery is a cosine nearest-neighbour over the per-namespace subject-key index.
Two hot-path wins are **shipped** (see `python/minisgl/cam/memory.py`, `_key_matrix`):

| Win | Effect | Status |
|---|---|---|
| Cached key matrix | reuse a pre-stacked `[N,d]` instead of re-stacking every lookup: **13× (N=1k) → 28× (N=100k)** | shipped |
| fp16 index | half the cached-matrix memory + **1.8–6×** faster matmul, **decision-lossless** (0/1600 argmax+gate flips vs fp32 on real GTE keys) | shipped |

Net: cached fp16 brute-force is **sub-millisecond through the tens of thousands** and **~1.4 ms at
100k** facts per namespace — which covers realistic per-namespace scale.

## When ANN (IVF) is worth building — measured, not yet built

Brute-force is O(N·d)/lookup, so it grows: fp16 ~4 ms at 500k, ~9 ms at 1M. An IVF (coarse
k-means quantizer + inverted lists, search only `nprobe` nearest centroids) was prototyped at
N=1M, D=128, K=2000 centroids, with **realistic queries** (a paraphrase strongly matching one stored
subject, cos ~0.85 — the actual delivery case; equidistant random queries are pathological and
misleading here):

| nprobe | recall@1 | speedup vs brute | keys searched |
|---|---|---|---|
| 8  | 74.7% | 32× | 4k |
| 16 | 86.3% | 23× | 8k |
| 32 | 93.7% | 12.6× | 16k |
| 64 | **99.0%** | **6.7×** (9 ms → 1.35 ms) | 32k |

**Verdict:** at 1M facts IVF buys a **single-digit (≈6.7×) approximate win at 99% recall**. It is
*approximate* — ~1% of deliverable facts miss (graceful base fallback, but a real recall loss) — and
carries real complexity: centroids, inverted lists, incremental insert/delete, periodic re-cluster as
N grows, `nprobe` tuning, recall monitoring. In 128-dim the recall/speedup tradeoff is fundamentally
modest without a *graph* index (HNSW ~10–50× at 99%), which needs a library the lean image ships
without.

**Recommendation:** do NOT build IVF speculatively. Build it only for a deployment that genuinely
holds **millions of facts in a single namespace**. Build sketch when that day comes:
- `_NsState`: add `centroids [K,d]`, `assign [N]`, inverted lists; build lazily above an N threshold
  (e.g. `MINISGL_CAM_ANN_MIN_N`), else keep the exact cached-matrix path.
- Query: `topk(nprobe)` centroids → gather their rows → exact cosine within → tau gate. Fall back to
  full brute-force if the probed set is empty.
- Maintain on write (assign new key to nearest centroid) / delete (drop from list); re-cluster when
  the max/median list-size skew crosses a threshold.
- Default `nprobe` for ≥99% recall (≈64 at K≈√N·2); expose it + recall via `/cam/stats`.
- Repro: `scratchpad/cam_ivf_probe.py`, `scratchpad/cam_scale_measure.py`.
