# P5 — torch over a foreign device pointer, in the serve image, under graph capture

**Result: GREEN (exit 0). The ~30-line ctypes arena is available; no C++ `from_blob` extension is
needed.**

Raw: [`p5.json`](p5.json) + per-leg `p5_leg_<name>.json` / `.log`, machine-rendered
[`p5.md`](p5.md), console [`p5.console.log`](p5.console.log), diagnostics in
[`p5_diagnostics/`](p5_diagnostics/). Earlier runs preserved in `p5_run1_slot8MiB/` and
`p5_run2_slot16MiB_devicecompare/`.

| | |
|---|---|
| Image | `minisgl-rdna4:lean` (`sha256:9748d5f9ec1d`), in-container, worktree `/engine` |
| Card | **physical card 0** — RX 9070 XT, `0000:03:00.0`, 32 WGPs, 17.1 GB (all four legs) |
| torch | 2.14.0.dev20260803+rocm7.2, HIP 7.2.53211, allocator backend `native` |
| Canonical run | 4 legs × 16 MiB slots, 12 measured reps + 1 warmup, 24 replays + 1 warmup |
| Lease | waived for this development work (user instruction); box otherwise idle, GPUs at 60 MB |

## Answers

| Question | Answer |
|---|---|
| `t.data_ptr() == reserved_base` under `use_mem_pool` | **YES**, first ptr == base in all 4 legs |
| every allocation inside the reservation, zero `hipMalloc` fallbacks | **YES** / **0** |
| coexists with `expandable_segments:True` (the compose default) | **YES**, verified live via `memory_snapshot().is_expandable` |
| host-located `hipMemCreate` pages behind a device VA (**bytes** verified) | **YES** |
| kernel read + kernel write + exact D2H roundtrip through the foreign VA | **YES**, max abs err 0.0 |
| graph capture + 24 replays over the foreign VA numerically correct | **YES**, 24/24 |
| arena survives `torch.cuda.empty_cache()` (`engine/graph.py:314`) | **YES**, 12/12, pointer stable |
| `empty_cache()` invokes the custom **free** cb for a **live** pool block | **NO** (0 calls) |
| … for a **cached (dropped)** pool block, on `del` or on `empty_cache` | **NO** (0 calls, either) |

The free callback never fired **at all** — not on `del`, not on `empty_cache()`, not on capture's
internal `empty_cache()`. The no-op free trampoline is therefore never exercised in this shape, and
the arena is structurally safe from being handed back.

## Numbers

VMM allocation granularity **4096 B** (min == recommended), confirming the measured figure and
contradicting the 64 KiB number quoted elsewhere. Reservation 720 MiB/leg, 208 MiB consumed by 13
× 16 MiB slots. Every leg ran on card 0.

| leg | alloc conf | backing | alloc µs via our allocator (n=12) | `empty_cache()` µs (n=12) | replay wall µs (n=24) | replay event ms |
|---|---|---|---|---|---|---|
| expandable_host | `expandable_segments:True` | host | median **7.48**, mean 8.58, sd 2.41, min 6.73, max 14.98, p90 10.70 | median 45.9 | median 91.5 | 0.0347 |
| expandable_device | `expandable_segments:True` | device | median **7.22**, mean 8.49, sd 2.51, min 6.63, max 14.54, p90 10.84 | median 41.7 | median 92.3 | 0.0351 |
| plain_host | unset | host | median **7.03**, mean 8.15, sd 2.16, min 6.34, max 13.67, p90 10.09 | median 142.5 | median 88.1 | 0.0350 |
| plain_device | unset | device | median **7.77**, mean 8.85, sd 2.38, min 6.82, max 14.44, p90 11.22 | median 142.3 | median 85.9 | 0.0346 |

Allocation cost is ~7 µs regardless of medium — it is python/ctypes trampoline overhead, not a
memory operation (the arena allocator is a bump pointer). **Replay times are a functional latency
over a 4 MiB L2/MALL-resident working set and are NOT a bandwidth measurement** — that is P1's job,
on a ≥256 MB working set.

## COLLATERAL FINDING — this one is not about P5, and it is worse

> **`expandable_segments:True` + `torch.cuda.empty_cache()` on this stack silently returns
> stale/zeroed device memory.** A freshly allocated tensor whose `fill_` has completed and been
> synchronised reads back as **all zeros**, at a VA torch had just unmapped and re-mapped.

Measured three ways, all host-verified (ctypes `hipMemcpy` D2H + CPU byte compare — no device
allocation participates in the check):

- **Plain torch, no MemPool, no custom allocator, no reservation** — `_p5_diag_pure_torch.py`:
  **7 of 12 reps corrupt**, 16 MiB float32 tensor, 4194304/4194304 elements wrong, value seen
  `0.0`, VA reuse rate **1.00**. With `expandable_segments` off: **0 of 12**.
  (`p5_diagnostics/p5_diag_pure_torch_{expandable,plain}.json`)
- **Inside P5** — the naive device-vs-device check disagreed with host truth in **6/12**
  (expandable_host) and **5/12** (expandable_device) reps; **0/12** in both plain legs.
- **Discriminator** — `_p5_diag_emptycache.py` verified both operands independently against the
  host: `n_reps_arena_host_wrong = 0` in every configuration, `n_reps_ref_tensor_host_wrong = 2/8`.
  The corrupted operand was always torch's own fresh allocation, never the arena.

This is the signature of the box's known `hipMemUnmap` → `hipMemMap`-at-an-already-used-VA defect
(the one P6 exists to document), surfacing **inside torch's expandable-segment allocator**, which
unmaps physical handles on `empty_cache()` and re-maps at the same VA on the next allocation.

Scope, stated precisely: **this is a demonstrated defect in the primitive, not a demonstrated wrong
serve output.** `engine/graph.py:314` calls `empty_cache()` and the compose default
(`docker-compose.yml:68`) sets `expandable_segments:True`, so the ingredients are both present in
production; whether a live weight/state tensor lands on a poisoned re-map has not been measured.
It deserves its own investigation and does not gate P5.

## Why the first 16 MiB run reported a false KILL — and what was fixed

Run 2 (`p5_run2_slot16MiB_devicecompare/`) exited **2** with
`tensor_survives_empty_cache: false` in both gating legs, in a perfectly alternating per-rep
pattern, pointer stable, zero free callbacks. That was **the checker, not the arena**: the arm
compared two *device* tensors,

```python
torch.equal(arena, torch.full_like(arena, v))
```

and the comparison operand — plus `torch.equal`'s own reduction output — is allocated from torch's
default expandable-segment allocator, i.e. exactly the allocation the defect above corrupts. A
false negative on the arm wired to the KILL criterion.

Fixed by adding `HostVerify` (one host buffer allocated once, blocking `hipMemcpy` D2H, full-range
CPU byte compare, **no device allocation inside any verification**) and rewiring both
`empty_cache_live` and `graph_capture` to it. The torch-side comparison is still *recorded*, as
`torch_side_comparison.n_reps_torch_equal_disagreed_with_host`, because the disagreement is the
collateral finding. Run 3 — identical parameters to the failing run 2 — is green in all four legs.

Run 1 (`p5_run1_slot8MiB/`, 8 MiB slots) passed with the old checker only by luck: torch served the
comparison tensors from cached blocks that were never unmapped, so no re-map happened.

## Caveats

- The three runs used one process per leg, on one card (card 0). No multi-card or TP=2 arena was
  exercised; the allocator explicitly refuses to serve a device index other than the arena's.
- 720 MiB reservations, 13 slots. Nothing here is evidence about a multi-GB arena, about
  fragmentation over a long serve, or about a real model's allocation pattern.
- The `correctness` arm still uses device-side temporaries for its `max_abs_err` term; it passed in
  every leg and its exact-D2H-roundtrip term is host-verified, but it was left unmodified.
- `empty_cache()` is ~3× faster with expandable_segments (42–46 µs) than without (142 µs) — noted,
  not investigated.
