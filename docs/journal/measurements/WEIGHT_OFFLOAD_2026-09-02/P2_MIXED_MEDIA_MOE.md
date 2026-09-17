# P2 — mixed-media grouped MoE GEMM: **NOT MEASURED**

*Probe spec:* `docs/WEIGHT_OFFLOAD_PLAN.md` §3 row **P2** (+ §2 placement-only architecture, §11 unknown #2).
*Run:* 2026-09-02, two attempts, both aborted on a precondition.
*Raw:* `p2.aborted.json` (attempt 2 state) and `p2_driver_defects.json` (the findings).
*Diagnostics:* `p2_diagnostics/`.
*Worktree:* `/home/pat/code/minisgl-rdna4-offload` @ `8bcc7035` (`feat/weight-offload`).
*Timed on:* **ROCR device 0 — AMD Radeon RX 9070 XT, `gfx1201`, PCI `0000:03:00.0`, 64 CU.** Card 1 (RX 9070) idle throughout.

## Verdict: NOT_MEASURED — the probe's mechanism does not exist on this box

**The kill criterion was never evaluated.** No correctness arm ran, no miss-count curve was taken,
`cliff_index` was never computed. Nothing below is a P2 result; the sections P2 would have filled
are empty in the JSON rather than zero.

But the run is not a null. It establishes, with a control and a cross-check, that **the plan's §2
placement-only architecture cannot be built on this hardware** — and that lands the *same*
architectural conclusion the KILL was written to force (placement must be per-**layer**, not
per-expert), by capability rather than by a timing cliff.

## Box state (the box is not idle — recorded, per the rule)

| MemTotal | MemAvailable | MemFree | SwapFree | pswpout Δ over run | loadavg | GPU0 / GPU1 |
|---|---|---|---|---|---|---|
| 91.8 GiB | 67.8 GiB | 30.6 GiB | 33.9 GiB | **0** (no swapping) | 2.68 2.52 3.35 | 0% use, 57 MiB VRAM each, 6 W |

Both compute cards were confirmed idle before the run (`rocm-smi`, no `llama-server`/`vllm`/stray
python). GPU lease waived by instruction; probes run serially. `ROCR_VISIBLE_DEVICES=0,1`,
`HIP_VISIBLE_DEVICES` unset, so the Ryzen iGPU (ROCm device 2, 47 GB GTT) never entered enumeration.

## What was attempted

E=512 experts, top_k=10, hidden=2048, inter=768, group=128, bf16 activations. One
`hipMemAddressReserve` of 1.70 GiB holding the w4a8 op-layout stack (`w13` 1572864 B/row,
`w13_scales` 49152, `w2` 786432, `w2_scales` 24576 — granule 2.32 MiB/expert, stack 1.16 GiB) plus
two 256 MiB scratch regions, backed by **1026** `hipMemCreate` handles, 256 device / 256 host
experts in a random placement, VMM granularity **4096 B** (confirming the plan's measured figure,
not the 64 KiB one).

## Defect D1 — `hipMemSetAccess` rejects adjacent mixed-size mappings (worked around)

Attempt 1 aborted at `hipMemSetAccess(...) -> hipError 1` partway through building the arena.

Per-chunk `hipMemSetAccess` returns `hipErrorInvalidValue` **non-deterministically** whenever two
*adjacent* mappings in one reservation have *different sizes*. `hipMemCreate` and `hipMemMap`
return `hipSuccess` for the very same chunk; only `SetAccess` rejects, and a retry at the same VA —
even over a single 4096 B sub-range — rejects again. It hits device- and host-located handles about
equally, so it is not a media property.

| pattern | chunks | SetAccess failures |
|---|---|---|
| adjacent **equal**-sized, per-chunk SetAccess (control) | 3072 | **0** |
| adjacent **mixed**-sized, per-chunk SetAccess | 256/trial | **33–34%** (host ≈ dev) |
| minimal pair (dev 1.5 MiB @0, host 7.5 MiB @1.5 MiB), fresh reservations | 100 | **50–75%** |
| one SetAccess per **component** region | 16 calls | **7–10** (≈60%) |
| **one SetAccess over the whole mapped span** | 4096 | **0** |

Fix landed in `Arena._set_access_once()`: map everything first, then issue **one**
`hipMemSetAccess` over the whole hole-free span. Semantically identical (nothing touches the arena
until it returns) and the only pattern measured reliable. It aborts rather than flaking if a future
layout introduces holes, since two adjacent SetAccess calls re-enter the defect.

Repro: `p2_diagnostics/_p2_diag_flake.out.txt`, `_p2_diag_fix.out.txt`, `_p2_diag_layout.out.txt`.

## Defect D2 — `hipMemCreate(location = Host)` is silently ignored (the blocker)

With D1 worked around, the arena built cleanly: 1026 handles, all four components fingerprinted
per-row (head *and* tail of every row, 2048 markers per dtype, all distinct — no aliasing, no
unmapped tails). Attempt 2 then aborted on the media-separation precondition:

> device-backed read **117.9 GB/s** vs host-backed **118.2 GB/s** — **ratio 1.00**

The "host" pages are not host pages. Every call succeeds, the VA works, the data is correct — and
the pages consume VRAM. This is precisely the trap the plan names: *a capability probe that passes
while the operation fails*.

P1 had already found this independently, by a different method, on **both** cards:
per-BDF `amdgpu` sysfs `mem_info_vram_used` moves **1.75–1.78×** the region size,
`mem_info_gtt_used` moves **0.0×**, `MemAvailable` moves **0.04×**, the VA is not CPU-accessible,
and kernel reads run at **689–692 GB/s** — full HBM. `HostNuma` and `HostNumaCurrent` are
unsupported. Two probes, two methods, same answer.

## Defect D3 — managed memory closes the last route

`hipMallocManaged(512 MiB)` lands **entirely in host RAM** (`MemAvailable` −406 MiB, VRAM +68 KiB,
GTT +0) and stays there: `hipMemAdvise(SetPreferredLocation=CPU)`, `SetAccessedBy`, and
`hipMemPrefetchAsync(device 0)` all return `hipSuccess` and all do nothing. Both halves read at
**28.8 GB/s**, ratio 1.00. An all-host contiguous VA with no device tier — the mirror image of D2,
equally unusable for a mixed stack. (`_p2_diag_managed.out.txt`.)

## Mechanism inventory — is there *any* route to a mixed-media contiguous VA?

**No.**

| route | contiguous VA | interleavable | real host pages | measured kernel read | verdict |
|---|---|---|---|---|---|
| VMM `location=Device` | yes | yes | — | 689–692 GB/s | device tier only |
| VMM `location=Host` | yes | yes | **no** (D2) | 689–692 GB/s | a device allocation wearing a host label |
| `hipHostMalloc(Mapped)` + `hipHostGetDevicePointer` | no | **no** | yes | 28.9 GB/s (card 0) | real host tier, wrong VA |
| `hipMallocManaged` + `hipMemAdvise` | yes | **no** (D3) | yes (all of it) | 28.8 GB/s | host tier only |

The two working host mechanisms give ~**28.8–28.9 GB/s** kernel-read on card 0, consistent with the
26.8–28.7 GB/s copy-engine figures and an x8 @ 32 GT/s link. **Card 1's root port was observed
down-trained to x8 @ 16.0 GT/s under load — 14.5 GB/s, half of card 0.** A host-tier design must
not assume the cards are symmetric.

## Consequences

1. **§2 placement-only is dead on this box.** Expert rows cannot straddle media in one VA. The
   blocker is capability, not performance — no amount of tuning reaches it.
2. **T2 (per-expert device tier) is unavailable by placement.** Getting it would require explicit
   copies into a device-resident arena — which is the layer-group streaming design, not placement.
3. **Placement must be per-layer / per-layer-group.** Same conclusion as the P2 KILL, different and
   stronger route.
4. **T1 (all-host) is constructible**, two ways, both measured at ~28.8 GB/s on card 0.
5. **P2's actual question is still open.** Whether per-layer time is linear or cliffed in miss count
   is now unanswerable as specified; it needs a probe built on explicit copies. Do not read this
   document as evidence either way.

## Confidence: high

D2 is two independent probes, two methods, both cards. D1 is quantified over 3072+ chunks against a
0/3072 control, with a 4/4-clean workaround. D3 has sysfs, `MemAvailable` and bandwidth all
agreeing. And no P2 measurement is reported, because none was taken.
