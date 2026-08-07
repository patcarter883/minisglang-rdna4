# CONTINUE — the W4A16 MoE scale gather is uncoalesced; transpose the scale/zero layout

**Session ended:** 2026-08-07. Serve is DOWN, both GPUs FREE, both cards restored to `auto`.
Everything below is measured and committed. The fix is **scoped and not started**.

Predecessor: [`CONTINUE_stride_conflict_and_moe.md`](CONTINUE_stride_conflict_and_moe.md). Read its
trap list first — this task came out of falsifying its headline prediction.

| repo | branch | SHA |
|---|---|---|
| `rdna4-hip-kernels` | `main` | `fe3d3c7` (the stride probe + fixture) |
| `rdna4-hip-kernels` | `task/moe-w4a16-counters` (worktree `rdna4-hip-kernels-moeprof`) | `a505815` — **not merged** |
| `minisgl-rdna4` | `rdna4` | `328e99c9` + this doc |

Serve image: **`minisgl-rdna4:sk6`** (verified to carry the six-arg `dense_gemm_rd_sk`). The old
`:lean` tag will `TypeError` on the first router call.

---

## THE TASK

`mmq_regdirect_w4a16_moe_kernel` reads its per-group weight scale like this
(`fp8_wmma/fp8_wmma_rocm/w4a8_fp8_wmma_kernel.hip:1765`):

```c
const int abs_n = block_n + f * 16 + frag_col;          // frag_col = lane & 15
const float wsc = __half2float(ws_e[abs_n * num_k_groups + g]);
```

`abs_n` carries the lane, so **16 lanes read 16 addresses `num_k_groups * 2` bytes apart** — a
fully scattered load, 16 separate requests per fragment, once per k-group. The packed zeros have
the same defect one line up (`wz_e[(nc / 8) * num_k_groups + g]`), but `nc / 8` collapses 16 lanes
onto 2 addresses, so it is minor.

**Fix: store the scales (and zeros) transposed** — `(E, K/group, N)` instead of `(E, N, K/group)` —
so those 16 lanes read 16 *contiguous* fp16 and the load coalesces into one request. It is a
`post_load` repack plus two index expressions. No restructuring of the kernel.

### Why this is the right target, and how big it is

The served AWQ checkpoint (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, and `FenomAI`/`palmfuture` variants)
is **`group_size=32`**, and small groups are what make this expensive: the gather happens once per
group, so g=32 pays it 4× more often than g=128 while having only `k_sub = 2` k-tiles of work to
hide the latency behind.

Measured, same shape (T=2048, E=256, top_k=8, N=2·inter=1024, K=hidden=2048), only `group_size`
changing — **g=32 costs 1.40× g=128**, and `wide` swept independently at fixed group moves <1%, so
it is not the weight-load width. Fixture:
`rdna4-hip-kernels/tools/_fixtures/moe_w4a16_stride_cumode_card0.txt`.

Crediting g=32 with its 4× larger scale/zero *bytes* still leaves it at 63.1% of HBM against 79.5%,
so **~1.26× is inefficiency rather than traffic** — that is the headroom this fix goes after. It
applies to gemm1 **and** gemm2, at every batch size. GLM-4.7-Flash-AWQ is already g=128 and gets
little; this is a Qwen-AWQ-shaped win.

### The counters, which corrected the diagnosis

Card 0, `profile_standard`. `SQ_WAVES` identical at 73,088 and occupancy flat at 90–91% across all
three legs, so they are matched by construction; `SQ_BUSY_CYCLES` reproduces the 1.40×. Fixture:
`rdna4-hip-kernels/tools/_fixtures/moe_w4a16_groupsize_counters_card0.txt`.

| per wave | g=32 | g=64 | g=128 | g32/g128 |
|---|---|---|---|---|
| **wait / wave_cycles** | **78.2%** | 49.6% | **41.4%** | — |
| `SQ_WAIT_ANY` | 13.0e9 | 6.3e9 | 4.9e9 | **2.64** |
| `SQ_INSTS_SALU` | 1,378 | 931 | 675 | **2.04** |
| `TCP_REQ` | 3,108 | 2,276 | 2,164 | **1.44** |
| `SQ_INSTS_VALU` | 8,807 | 7,528 | 6,920 | 1.27 |
| HBM MB (`GL2C_MISS`×256) | 403 | 381 | 370 | 1.09 |
| L2 hit % | 99.1 | 98.1 | 97.3 | — |

**It is a stall, not extra throughput.** Wave cycles rise 4.75e9 while *wait* cycles rise 8.09e9.
HBM traffic is flat and L2 hit is 99%, so it is not bandwidth; occupancy and `MemUnitBusy` are
unchanged, so it is not occupancy.

The prediction going in was the **zero-point** gather. The counters say **scale**: `TCP_REQ` rises
944/wave over 48 extra groups ≈ **20 requests per extra group**, far more than one zp load, and
48 × 16 ≈ 768 plus the zp and overhead ≈ 944. That is what makes the fix a layout change rather
than an instruction-count optimisation — and it is why "fewer instructions" would have been the
wrong thing to chase.

### Do this in order

1. **Transpose in the loader.** `python/minisgl/layers/moe.py` builds `self._scales_op` at
   `:71`, `:119`, `:234`, `:318`, `:351` — five call sites, one per checkpoint format, and they
   must move together or a format silently feeds the kernel the wrong layout. The W4A16 path
   already drops the op-layout weight in favour of `_w_rep` at `:81/:129/:257`; do the same for
   scales so only one layout exists per path. `fp8_wmma.repack_w_rep_wide_moe` is the precedent for
   where such a repack lives.
2. **Two index expressions in the kernel** (`:1765` and the zp line above it), plus the same pair
   in the `_scatter` twin and in `mmq_regdirect_w4a16` if it shares the layout — **grep before
   editing; a fix that lands on one sibling only is a documented failure mode here.**
3. **Parity first**: `fp8_wmma/tests/test_w4a16_regdirect.py` covers all 7 ops in both act dtypes
   and asserts fp16 is materially tighter than bf16 — that is the check a silent fallback cannot
   pass. The transpose is a pure reindex, so it must be **bit-identical**, not merely close.
4. **Then the kernel bench**, then a **sampled serve A/B**. Do not stop at the bench.

---

## The tooling this session built, and how to use it

**Counters on gfx1201 require ROCm 7.14** (7.2.1's `rocprofv3 --pmc` hangs, confirmed again this
session). **But the kernels we ship are built by 7.2.1's hipcc**, and that is not a detail:

> Recompiling `w4a8_fp8_wmma_kernel.hip` with 7.14's clang 23, same flags, leaves only **8 of 2074**
> kernels identical. The worst instantiation moves **VGPR 256 → 87 with 288 B of scratch spill
> going to zero**.

So `tools/counter_probe/probe_counters.sh`'s standing claim — *"the kernel source is shared, so a
counter measurement taken in 7.14 transfers"* — is **false for this TU**, and every existing
torch-free probe in this tree is a clang-23 build of a clang-22 kernel. That line is now corrected
in place.

**The working pattern is COMPILE HERE, RUN THERE:**

```
bash fp8_wmma/local/build_moe_w4a16_probe.sh          # host, CPU only: extract + compile + gate
GPU_LEASE_WEDGE_WATCH=0 gpu-lease -n 2 -- \
  bash /home/pat/code/minisgl-rdna4/tools/counter_probe/perf_level_guard.sh \
    bash fp8_wmma/local/counters_moe_groupsize.sh
```

* Compiled in `minisgl-rdna4:lean` (7.2.1 hipcc), run in **`rdna4-rocm7.14`**. A 7.2.1 HIP binary
  executes correctly there. **Not** `rocm/dev-ubuntu-24.04:7.14.0-full` — no `libamdhip64.so.7`.
* The kernel body is **extracted from the source at build time**, not hand-transcribed the way
  `moe_g2fuse_probe.hip` and friends are, so it cannot drift from what it claims to measure.
* `isa_equivalence.py` is the gate: register footprint must be **identical** (it is, on all three
  instantiations), instruction-class counts within a printed residual (0/1/2 instructions, all
  address-setup in the once-per-block prologue). Bit-identity does *not* hold — a 3-kernel probe TU
  does not reproduce a 2074-kernel one exactly, because AMDGPU attribute inference is
  module-influenced. Registers are the hard gate because registers are what the compiler swap
  wrecked.

If the fix changes the kernel, **re-run the gate** — it compares against the real build object, so
it stays valid as the source moves.

---

## Traps this session paid for

1. **A `.so` cannot be unbundled naively.** Its `.hip_fatbin` is a **concatenation of one bundle per
   TU** and `clang-offload-bundler --unbundle` silently returns only the **first**. Dumping the
   `.so` yields a device object with zero matching kernels, which reads as "the probe does not
   match the shipped build". Compare against the TU's `.o`.
2. **`pmc_have_counter` reported OK when NOTHING was collected.** It was
   `find … | xargs -r grep -l`, and with `-r` xargs never runs grep on empty input, so the pipeline
   exits 0. A collection dying `rc=127` printed a full column of `OK`. Fixed in `pmc_lib.sh` to
   count matches; the harness also asserts a non-zero CSV count now.
3. **`rocprofv3` resolves differently under `bash -c` vs `bash -lc`.** Without the login profile,
   `/app/.venv/bin` is off PATH and `rocprofv3` resolves to `/opt/rocm/bin/rocprofv3`, whose shebang
   is `#!/usr/bin/env python3` — and python3 is not on PATH. `rc=127`, silently, per trap 2.
4. **Keeping the shipped `.so` and swapping only the ROCm runtime underneath does not work.**
   torch 2.14+rocm7.2 imports fine on the 7.14 runtime, sees the RX 9070 XT, and **SIGSEGVs on the
   first kernel launch**. That route is closed; the image built for it has been deleted.
5. **Compile flags must match the real build exactly.** Without `-fPIC`/`-fno-gpu-rdc` and the
   `__HIP_NO_HALF_*` defines that the TU `#undef`s, codegen packing shifted and the gate failed for
   a reason unrelated to the compiler.
6. **`VALUBusy` and `VALU cyc / busy cyc` read >100%** (115–344%) on these kernels — the
   normalisation is wrong, as `COUNTER_SCORECARD.md` already records. Relative only, never absolute.
7. `profile_standard` **pins clocks non-boost**. The `ms/iter` the probe prints inside the window is
   not comparable to the wall-clock fixture. Counters and times come from separate runs.

---

## Also still open

* Two docs assert counters are permanently dead on this box —
  `rdna4-hip-kernels/opt_loop/roles/profile_engineer.md` §2 and
  `opt_loop/knowledge/gfx1201.md:223`. Overturned 2026-08-05, still not updated.
* `mmq_regdirect_w4a16_moe_gemv*` (3 ops) is **unreachable from the engine** — `kernels.w4a16_moe`
  uses the WMMA op at every M. Either wire it or delete it.
* `mmq_regdirect_w8a8_moe_kernel` (7 engine refs) and `flash_prefill_paged` (9 refs) were never
  measured for the same scale-gather pattern. Check the indexing before assuming it is there —
  the predecessor doc's headline prediction failed exactly that way.
* Everything under "Still open from the predecessor doc" in
  [`CONTINUE_stride_conflict_and_moe.md`](CONTINUE_stride_conflict_and_moe.md).
