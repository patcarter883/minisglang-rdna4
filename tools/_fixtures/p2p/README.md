# Cross-card P2P on this box — the first actual measurement

2026-08-12, both gfx1201 cards under one `-n 2` lease. Tool: `p2p.hip` (torch-free, no engine).
The repo had asserted things about this link for months and never measured it.

## Topology

The two compute cards sit on **separate PCIe root ports** — `0000:03:00.0` under `00:01.1`,
`0000:07:00.0` under `00:01.3` — so peer traffic crosses the CPU root complex. There is no XGMI.
Both links negotiate PCIe 5.0 x16 at the card.

## Numbers

`hipDeviceCanAccessPeer` is **YES both directions** — the copy does not bounce through system RAM.

| transfer | us | GB/s | via host (D2H+H2D) |
|---|---|---|---|
| 4 KB (hidden, bs=1) | 5.9 | 0.7 | 155.2 |
| 24 KB (hidden, bs=6) | 7.1 | 3.5 | 79.9 |
| 256 KB (hidden, bs=64) | 23.1 | 11.4 | 101.6 |
| 8 MB (prefill chunk) | 604.2 | 13.9 | 996.2 |
| 64 MB | 4823.3 | 13.9 | 7901.1 |

**Latency floor 5.9 us; bulk bandwidth 13.9 GB/s.** Staging through the host instead costs 6-26x.

## What this settles

1. **`docs/ZAYA_SERVING_NORTH_STAR.md:44,53-55` is STALE.** It says custom-AR is "a dead end on
   RDNA4 (no XGMI)" and "architecturally impossible on consumer RDNA4 … every fine-grained x-GPU
   sync primitive fails". P2P works, custom_ar has been the shipped default since `impl.py` (2026-08-05),
   and it measures 1.15-1.18x faster than RCCL. That file was last touched 2026-07-04.

2. **The all-reduce tax is NOT bandwidth — it is a spin barrier.** The recorded in-serve cost is
   3.514 ms/step over 81 dispatches = **43.4 us per call**, against a 5.9 us link floor for the same
   payload class: **7x what the link can explain**. `DIFFUSIONGEMMA_BLOCK_DIFFUSION.md:1859-1861`
   independently puts the AR kernel at 1.44 MB in 124 us = 11.6 GB/s, i.e. 83% of the 13.9 GB/s this
   measurement says the link delivers. The time is rank skew, not payload.

   That is the load-bearing fact for any parallelism change: a faster link buys nothing, and
   anything that replaces a symmetric barrier with an asymmetric dependency inherits the skew.
