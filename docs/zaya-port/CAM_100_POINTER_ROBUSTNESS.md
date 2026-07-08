# CAM #100 pointer-serving robustness — N-scaling + paraphrased subjects

Measured the pointer id-bank serving delivery (`/cam/ask` via `CAMMemory.deliver_object_ids`) at scale
and under subject variation. Real Qwen3.5-4B embeddings (addressing is content-based), CPU, no base
forward — pure embed + store ops. Each subject gets a DISTINCT multi-token object so a store collision
delivers the WRONG object (detected), not a shared one (masked). Checkpoint: `cam_ckpt` (n_banks=32,
mem_dim=512, mt_cap=16). Harness: `scratchpad/robust.py`.

## (A) Exact-subject delivery vs N — collision-limited
| N | span-exact | per-token |
|---|---|---|
| 25 | 0.840 | 0.940 |
| 50 | 0.820 | 0.909 |
| 100 | 0.750 | 0.837 |
| 200 | 0.505 | 0.655 |
| 400 | 0.367 | 0.493 |

Delivery degrades with N. The offline "4/4 @ N=137" held because that run used **512** value banks; this
serving checkpoint ships **32**. Two subjects collide when they route to the same bucket AND address the
same top-1 store slot — the later write overwrites the earlier id.

## (C) Root cause = bucket count (subjects-per-bank). Same sweep at n_banks=512:
| N | n_banks=32 | **n_banks=512** |
|---|---|---|
| 100 | 0.750 | **0.970** |
| 200 | 0.505 | **0.935** |
| 400 | 0.367 | **0.912** |

More banks → fewer subjects per bank → far less collision. **Fix (trivial): export the serving
checkpoint with more banks (scale n_banks to the expected fact count; 512 holds ~400 facts at 0.91).**
A stronger fix is per-position DISJOINT id-stores (each position its own codebook), but n_banks alone
recovers most of it.

## (B) Paraphrased subjects — NO generalization (exact-match retrieval)
N=50 stored (n_banks=32), ask with subject variants:
| variant | span-exact |
|---|---|
| exact | 0.820 |
| first_only | 0.000 |
| last_only | 0.000 |
| with_title ("Ms. …") | 0.000 |
| reordered ("Last First") | 0.000 |
| lowercased | 0.020 |

The pointer is an EXACT-match retrieval: `_subject_bank` is an md5 hash of the exact token ids (any
variation → different bucket), and the product-key addressing keys on the exact subject embedding. So
any subject paraphrase misses entirely.

## (D) Paraphrase ceiling — n_banks=1 (single bucket, pure store content-addressing)
| variant | span-exact |
|---|---|
| exact | 0.160 |
| first_only | 0.020 |
| last_only | 0.020 |
| with_title | 0.100 |
| reordered | 0.160 |
| lowercased | 0.100 |

Removing the exact-hash bucket does NOT recover paraphrase generalization: a paraphrase still doesn't
address the same product-key slot as the original (the addressing is sensitive to the exact subject
embedding, not paraphrase-invariant), and a single bucket tanks even exact delivery via collision
(0.82 → 0.16 at N=50). So paraphrase robustness needs a **semantic key**, not base embeddings:
memory-organ's `CAM_GTE_KEYS` (a GTE-ModernColBERT table decoupling addressing from the base embed) or
subject canonicalization/normalization at the API layer. This is a real feature, not a knob.

## RESOLUTION — a cosine-NN subject index fixes BOTH (implemented + validated)
Both problems were the product-key id-bank's, not fundamentals. Two fixes landed:

1. **`MINISGL_CAM_NBANKS` serving knob** (commit 4cade96) — n_banks is a pure serving parameter (banks
   are just states; codebooks are shared), so scaling it cuts id-bank collision with no re-export. But it
   is superseded by:
2. **Cosine-NN subject index** (commit c3ad0ed) — the delivery mechanism is now nearest stored subject by
   cosine over `key = L2-norm(mean(base input embeds over subject tokens))`, ≥ `deliver_tau` (0.7,
   `MINISGL_CAM_DELIVER_TAU`). It replaces the product-key id-bank for delivery and fixes both:

| metric | id-bank (before) | **cosine-NN index (now)** |
|---|---|---|
| exact @ N=400 | 0.37 (n_banks=32) | **1.00** (exact retrieval, no slot collision to N=500) |
| reordered / with-title / trailing-punct | 0.00 | **1.00** |
| lowercased | 0.00 | ~1.0 (weakest cases fall back) |
| unknown false-deliver | — | **0** (unknown max-cos ≤0.51 « tau 0.7) |

Cosine-value calibration (N=60): exact/reorder 1.00, trailing 0.79, title 0.72, last-only 0.68 — all
clean of unknowns (max 0.51); lowercase-with-retokenization (min 0.43) and bare first/last-name overlap
the unknown band, so they conservatively fall back (no false delivery). **tau=0.7 gives a ~0.19 margin.**

Validated end-to-end over HTTP (Qwen3.5-4B): remember Klingon/Sindarin, then ask **"Quillsworth
Zephyrina"** (reordered) → "Klingon. ==History==…" and **"Ms. Cornelius Blackwood"** (title) →
"Sindarin, a language of the Lord of the Rings…" — both paraphrases deliver the right object + coherent
continuation. No new model — uses the base embeddings already held.

## Bottom line
- **Exact + structural paraphrases (reorder, title, trailing punct, case): delivered, N-scale-free** via
  the cosine-NN index. Production-viable, no tuning.
- **Aggressive variations (bare first/last name, heavy re-tokenisation): conservatively fall back** (no
  wrong delivery). API-side subject canonicalisation would extend coverage; a semantic encoder is NOT
  needed (the base pooled embedding already generalises, and Qwen3-Embedding-0.6B tested WORSE).
