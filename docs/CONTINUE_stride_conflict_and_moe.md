# CONTINUE — the 4 KB stride conflict, and taking it to the served MoE path

**Session ended:** 2026-08-07. Serve is DOWN, both GPUs FREE. Everything described as "landed" is
**committed and merged**. The next task is scoped, verified, and not started.

Predecessor: [`CONTINUE_bf16_kernel_parity.md`](CONTINUE_bf16_kernel_parity.md) — still accurate,
now corrected. Read its "FIVE errors, one root cause" section before trusting any sweep.

---

## START HERE

Two things, in this order.

### 1. The serve image MUST be rebuilt before anything starts

The merged `minv.py` calls `dense_gemm_rd_sk(..., 0, M < 448)` — **six** args. The baked
`/opt/kernels` in `minisgl-rdna4:lean` has the **five**-arg op:

```
baked rd_sk grid_split:  False
baked rd  mode_override: False
```

Python is mounted (updates on restart), kernels are baked (do not). So a restart without a rebuild
is a hard `TypeError` on the first router call in prefill — not a silent slow path. Deliberate: a
`hasattr` fallback would serve the slow path while looking healthy, which is the `_tail_hip`
re-export failure mode.

```
docker build -t minisgl-rdna4:<tag> \
  --build-context kernels=<clean rdna4-hip-kernels worktree at 479c918> \
  --build-arg KERNELS_REF=479c918 .
```

### 2. The next task: `mmq_regdirect_w4a16_moe_kernel`

Verified, scoped, **not started**. Details in "The next task" below. Do not start with
`moe_bf16act_regdirect_kernel` — a survey ranked it #1 and it is **dead code from the engine's
side** (see the trap list).

---

## What landed and is merged

| repo | branch → merged into | SHA |
|---|---|---|
| `rdna4-hip-kernels` | `feat/bf16-structural-parity` → `main` | `479c918` (feat `1d5c633`) |
| `minisgl-rdna4` | `feat/minv-unpin-tiles` → `rdna4` | `29778066` (feat `32779c09`) |
| `minisgl-rdna4` | docs | `b946b2fb` |

Base for the whole stream was `356bfdaa`. Worktrees `minisgl-rdna4-minv` and
`rdna4-hip-kernels-bf16tune` are **still present** (kept for the weight-storage question); remove
with `git worktree remove` when done.

**End-to-end, cold, card 0, real old code vs new: 1.223× → 1.102× vs rocBLAS, 6/30 → 12/30 cells
beating rocBLAS, M-invariance PASS, 0 unexpected bit differences.**

### `SLICED` — a reduction-order policy that makes an M-gated split-K legal

Split-K wins hugely on the router below M≈448 (0.52× rd at M=16) and loses just as hugely above it
(2.21× at M=2048). It could not be gated on M, because a split arm and an unsplit arm reassociate K
differently — switching at an M boundary makes a token's logits a function of its batch size.

`SLICED` runs the **same** `kslice` partition inside one block with a running fp32 total advanced
once per slice — bit-for-bit what `dense_gemm_reduce_kernel` does to the partials. So the two are
the same *number*, and choosing between them becomes pure scheduling that M is free to make.
`dense_gemm_rd_sk` gained `grid_split`; cost is `NFRAG*8` VGPRs, ±1% vs plain rd.

The doc dismissed this route as "far more invasive". It was ~20 lines. **Router: 1.38× → 0.66–0.76×
rocBLAS**, where the non-split lattice's own per-cell oracle never got below 1.19×.

Split into two decisions that had been conflated:
1. **Which shapes take the split-K ORDER** — per-shape, never a function of M (`OUT<=256 and
   IN>=1024` in `minv.py`). The `IN>=1024` guard is load-bearing: below it `split_k_slices()`
   returns ≤1 and the op silently reverts to plain-rd order, re-mixing two orders on one weight.
2. **How it is SCHEDULED** — `grid_split = M < 448`. Bit-identical, so M may gate it.

### The 4 KB stride conflict

A warp's fragment load reads 16 consecutive **rows** at one k-offset — 16 addresses exactly
`IN*sizeof(T)` bytes apart. When that stride is a multiple of **4096 B**, all 16 land in one cache
set and every fragment load is a 16-way conflict. (IN=1024, a 2 KB stride, is clean, so the critical
stride is 4 KB, not "any power of two". The 4096 B period implies `line × nsets = 4096`, consistent
with a 32 KB / 128 B-line / 8-way L0 — inferred from the period, **not** read from counters.)

Cost per K-element, M=128 OUT=2816, `rd bm128/bn32`:

| IN | 2016 | 2032 | **2048** | 2064 | 2080 | 2112 | 4080 | **4096** | 4112 |
|---|---|---|---|---|---|---|---|---|---|
| ns/K | 14.12 | 17.67 | **29.57** | 17.63 | 14.01 | 13.37 | 16.99 | **25.99** | 16.97 |

**Two mitigations, and they are complementary, not alternatives.**

**(a) Per-kernel CU mode — LANDED.** `__attribute__((target("cumode")))` sets `WGP_MODE` in that
kernel's descriptor alone. `dense_gemm_rd` now has two thin `__global__` wrappers over one shared
`__device__` body — same template instantiation, hence bit-identical — and the launcher elects CU
mode only where it pays. A **global** `-mcumode` is a net loss (+5–14% on every non-conflicting
stride); per-kernel it is free. Shipped `.so`: 136 WGP + 20 CU descriptors.

**o_proj M=128: 2.20× → 1.13× rocBLAS.**

**(b) Zero-padding K — measured, bit-exact, NOT integrated.** Widen K with extra **weight** columns
set to zero; the extra k-steps add exact `0.0` to an fp32 accumulator, so it is bit-identical
(`torch.equal` True on all 24 cells), and the stride moves off the boundary. This is the *complete*
fix where CU mode is partial. Rule: smallest `PAD` (multiple of 16) with `2*(IN+PAD) mod 4096 >= 128`
— `PAD=16` on IN=2048 lands on 4128 B, still in the shoulder, and **hurts**.

o_proj vs rocBLAS, M=16/64/128/192/256/512:
`1.02 / 1.02 / 2.20 / 1.53 / 1.20 / 1.14` → **`0.82 / 0.85 / 0.94 / 1.12 / 1.00 / 1.02`**.
Pipe benefits too (1.12–1.36×), less than rd (up to 2.11×), because it stages B through LDS.

### `minv.py` tile derivation

`_BLOCK_M`/`_BN` were 64 for every shape and every M. Now derived; env vars became overrides
(`0` = derive), which is what makes a same-build A/B possible. The ARM rule is unchanged.

* **rd `block_m` must not force an M-pad** — minimise padded rows, then largest tile.
* **rd `BN` is not a function of OUT alone** — A-traffic `(OUT/BN)*M*IN` (halved by doubling BN,
  grows with M) against grid parallelism `(OUT/BN)*ceil(M/bm)` (halved by doubling BN, flat in M).
  Right on every dispatchable rd cell except `gate_up M=128`.
* The wide-OUT pipe staircase gained a step at M≥128. The existing `OUT>=65536 && M>=192 -> 256`
  clause is **load-bearing** (dropping it regressed lm_head M=256 by 43%).

---

## THE NEXT TASK — `mmq_regdirect_w4a16_moe_kernel`

`hidden = 2048` is near-universal across the cached models — **including production**
(`z-lab/Qwen3.6-35B-A3B-DFlash`: hidden 2048 → 4096 B, q_stride 4096 → 8192 B). Also GLM-4.7-Flash-AWQ,
Laguna-XS, ZAYA1, Instella, Qwen3-30B-A3B, DeepSeek-V2-Lite, Agents-A1. So `o_proj IN=2048` was never
a quirk — it is *the hidden dimension*, and every kernel reading activations with a K=hidden row
stride hits the same well.

**Target** (`fp8_wmma/fp8_wmma_rocm/w4a8_fp8_wmma_kernel.hip`, the production AWQ MoE path,
**8 engine refs**):

```c
:1637  __global__ ... mmq_regdirect_w4a16_moe_kernel(...)
:1712  const int ar = lane >> 1;                                 // 16 distinct rows per warp
:1741  av = *reinterpret_cast<const Frag*>(&x_h[arow * K + gk]); // stride = K*2 = 4096 B
```

LDS is three `__shared__ int[16]` = **192 B**, so CU mode is fully available.

### Do this in order

1. **MEASURE FIRST, do not port the gate.** `block_m == 16` here ("one expert per block"), far below
   the M≈96 latency-floor threshold fitted on dense. CU mode may not pay at all. Add
   `mode_override` (0=auto/1=WGP/2=CU) to this launcher — same precedent as `sk_override` — and
   measure both modes on the real MoE shapes before any kernel edit. This is exactly what let two
   wrong gates be falsified on the dense side without rebuilding.
2. If it pays, add the CU wrapper the same way: one shared `__device__` body, two `__global__`
   wrappers, launcher elects. Verify at the artifact (see "Verifying a mode change" below).
3. **The bar here is a real serve A/B, not a kernel bench** — this is the production quantized path.
   Sampled, not greedy.

### Also wired, lower priority

* `mmq_regdirect_w8a8_moe_kernel` — 7 engine refs, same `lane>>1` pattern, stride `K` **bytes**
  (fp8), so K=4096 conflicts and K=2048 is 8-way. ~192 B LDS.
* `flash_prefill_paged` — 9 engine refs, `q_stride = num_q_heads*HEAD_DIM` = 4096 elem = **8192 B**
  for standard 32-head MHA. But only 2–4 rows/wave and global→LDS staged, so lower yield. LDS
  36–43 KB, still under the CU cap.

### Disqualified / cleared — do not re-walk

* **`moe_bf16act_regdirect_kernel` — DEAD CODE from the engine.** A survey ranked it #1 (IN=2048 is
  literally in its dispatch table at `moe_bf16_ops.hip:133`). But
  `engine → moe_bf16_gemm_out → bf16_launch_moe_gemm → moe_bf16act_tiled_kernel`, which stages A
  through **LDS**. The regdirect twin is reachable only via ops at `torch_binding.cpp:1352-1396`
  that `fused.py` never calls.
* **`dense_gemm_pipe_kernel` — CU mode is off the table.** It requests
  `MaxDynamicSharedMemorySize, 65536` at lines 497 and 661, i.e. it already sits on the 64 KB
  workgroup cap CU mode imposes. For o_proj at M≥192 (which routes to pipe) **zero-padding is the
  only lever**. This is why the landed fix touches `rd` only.
* CCA `dot_row` — highest theoretical degree (32-way, `cca_kernel.hip:154`, lanes index weight rows
  directly) but no evidence its `hidden` is 2048/4096 in served CCA configs. Check the config first.
* `flash_decode_*`, `mla_split/combine`, `gemv_decode_core`, `moe_gemm_splitk`, GDN WMMA — all
  cleared. Lanes index *within* a row (contiguous), or the loads are LDS-resident. Note GDN's LDS
  arena is ~62 KB, so GDN is itself at the CU-mode ceiling if that ever comes up.

---

## The open decision: weight storage for the zero-pad

The zero-pad is the complete fix but the weight must be **stored** padded — padding per call would
copy the whole weight. Options:

* **Duplicate**: cache a padded copy. ~346 MB for o_proj across 30 layers. Real money against the KV
  pool after the host-tier work.
* **Single storage, strided view**: allocate `[OUT, IN+PAD]`, zero the tail, expose the logical
  weight as a non-contiguous view `[:, :IN]` for the `F.linear` fallback. Collides with the
  `is_contiguous()` check in `minv_supported()`, and anything calling `.contiguous()` silently
  copies.
* **Load-time**: pad in the weight loader so only the padded tensor exists (+3% for that layer).
  Cleanest on memory, most invasive on the loading path.

Deferred by the user — "we'll revisit the weight storage question".

---

## Traps — these cost real time this session

All of the same family as the predecessor doc's five: **the harness and the thing it measures
silently disagreeing.**

1. **An A/B baseline that EMULATES the old behaviour is not the old code.** The prior harness set the
   *new* `_BLOCK_M_OVERRIDE` to 64 to reproduce "pinned", but the old code never fed `_BLOCK_M` to
   the pipe arm — pipe ran its own `pbm` staircase. That pinned pipe at a config the engine would
   never launch, and fabricated the headline "lm_head M=512, 3.02× → 1.18×". Against the real old
   code that cell is **−0.4%**. The claimed −50% total was **−9.9%**. Load the pre-change file as its
   own module and call both.
2. **`CUDAGraph` bakes the weight pointer at capture.** A rotation counter inside the timed lambda
   still measures HOT. The rotation must be unrolled *into* the graph — `reps` distinct captured
   calls over `ncopy` buffers.
3. **`sys.path.append` loses to the image's `PYTHONPATH=/opt/kernels`**, so a probe measures the
   BAKED kernels and a fresh kernel change reads as "no effect". Use `insert(0, …)` **and assert**
   `dense_gemm.__file__`. `sweep_policy.py` and `splitk_router_flips.py` **still have this bug**.
4. **`sweep_policy.py` hoists the M-pad out of the timed region**, under-pricing every tile whose
   `block_m` does not divide M. It scored a +19.1% regression as a win.
5. **Optimising what isn't dispatched.** Check the engine call path before touching a kernel — twice
   now (`dense_w4a8_gemm`, `moe_bf16act_regdirect`).
6. **Fitting a gate on one slice of the space.** The CU gate was fitted and refuted twice:
   `blocks < 64` dies on M=128/bm128/bn32 (CU wins at 88 blocks); `block_m >= 96` dies on
   M=128/bm64/* (CU wins at bm64, which loses at M=64). Only M separated all eight forced-both-ways
   points. **`M >= 96` is fitted on ONE shape** and is a proxy for "past the latency floor", not the
   cause — re-measure for any new shape.

---

## Tools and fixtures now available

**Engine** (`minisgl-rdna4/tools/`)
* `minv_tile_ab.py` — the trustworthy A/B. Loads the pre-change source as its own module, refuses to
  run if the two hash equal, records what each leg **actually dispatched**, asserts M-invariance of
  the shipped path, distinguishes an intended reduction-order bit-move from a bug, and has
  `--require-local-kernels`. 13/30 cells dispatch identically in both legs and land within ±1.5% —
  a free per-run noise floor, and what exposed trap #1.
* `_fixtures/minv_tile_ab_cold_card0.csv`

**Kernels** (`rdna4-hip-kernels/`)
* `dense_gemm/local/splitk_router_ladder.py` — M ladder + schedule bit-identity + M-invariance gate.
* `dense_gemm/local/probe_stride_aliasing.py` — sweeps IN around the 4 KB boundary, normalised per
  K-element.
* `dense_gemm/local/probe_zeropad_stride_fix.py` — the zero-pad test, `--arm rd|pipe`.
* `dense_gemm/local/splitk_router_flips.py` — **the** criterion for an MoE router: top-k index
  mismatches on real captured inputs. A "max rel delta 3e-7" proves nothing there; top-k is a step
  function and one ULP reroutes a token. Fixture `/home/pat/code/_fixtures/gemma4_router.pt`
  (real, 5 layers, 2800 rows). Result: **0/22400**.
* `fp8_wmma/local/run_tile_select_bitident.sh` — new; the gate `#include`d two headers nothing
  generated, so it was unrunnable. Now dumps them from git. Host-only (git + g++, no GPU, no lease).
* `tools/_fixtures/`: `dense_gemm_surface_card0.csv`, `bf16_gemv_surface_card0.csv`,
  `splitk_router_ladder_card0.txt`, `stride_aliasing_card0.txt`, `cumode_vs_wgp_card0.txt`.

**Gates that must stay green:** `splitk_check` GREEN, `ktail_check` GREEN, tile_select bitident
0/124080, expert-flip 0/22400, schedule bit-identity at every M 16..8192, M-invariance 0.000e+00.

---

## Recipes that took iterations to get right

**Container run.** The lean image has **no `/app/.venv`** — python is `/opt/venv/bin/python` on PATH,
and `PYTHONPATH=/opt/kernels:/opt/minisgl/python` is already set. To test a LOCAL kernel build, put
its `torch-ext` **first**:

```
gpu-lease -n 1 -- bash -c 'docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
  --ipc host --shm-size 16gb \
  -e HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES -e ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
  -e PYTHONPATH=/k/dense_gemm/torch-ext:/engine/python:/engine:/opt/kernels \
  -v /home/pat/code/minisgl-rdna4-minv:/engine \
  -v /home/pat/code/rdna4-hip-kernels-bf16tune:/k \
  --entrypoint bash minisgl-rdna4:lean -lc "python /engine/tools/minv_tile_ab.py ..."'
```

Single-quote the whole `bash -c` so `$HIP_VISIBLE_DEVICES` expands **inside** the lease shell.
Mount flags must come **before** the image name (an easy silent failure: the mount is passed to bash
instead and the probe falls back to synthetic data).

**Kernel build:** `cd <pkg> && rm -rf build && bash local/build_local.sh` inside the image, no GPU
lease needed (compile is CPU-only). `EXTRA_HIPCC=-mcumode` is a build hook in
`dense_gemm/local/setup.py` for whole-package experiments.

**A git worktree's `.git` points outside the mount**, so anything needing `git` must run on the host,
not in the container (`run_tile_select_bitident.sh` hit this).

### Verifying a mode change at the artifact

Do not trust that a flag or attribute landed — parse the shipped kernel descriptors.
`COMPUTE_PGM_RSRC1` is at byte 48 of each 64-byte descriptor; **bit 29 is `WGP_MODE`** (1 = WGP,
0 = CU):

```
L=/opt/rocm-7.2.1/lib/llvm/bin
$L/llvm-objcopy --dump-section=.hip_fatbin=/tmp/fb.bin <pkg>_C*.so
$L/clang-offload-bundler --type=o --input=/tmp/fb.bin \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 --output=/tmp/dev.o --unbundle
$L/llvm-objcopy --dump-section=.rodata=/tmp/kd.bin /tmp/dev.o
python3 -c "
import struct,collections
d=open('/tmp/kd.bin','rb').read(); c=collections.Counter()
for i in range(0,len(d)//64*64,64):
    c['WGP' if (struct.unpack_from('<I',d,i+48)[0]>>29)&1 else 'CU']+=1
print(dict(c))"
```

Also: **rocBLAS is a free control** in any of these probes. It is unaffected by our compile flags, so
if its column moves between two runs, the comparison is drift and not a result. It matched within
0.5% across the WGP/CU builds, which is what made those deltas trustworthy.

---

## Still open from the predecessor doc

* `dense_gemm_pipe_sk` did **not** get `SLICED` — pipe's split arm is still all-or-nothing, so no
  pipe-class shape can take the split-K order without the prefill penalty.
* `FMT_BF16`'s `vgpr_fixed`/`vgpr_per_frag` in `tile_select.h` are carried over from W4A8 and remain
  **unfitted**. The bf16 probe pins `bk` to the lattice minimum on most shapes — the classic sign of
  an unmodelled per-k-step cost.
* `_RD_BN64_MIN_TILES = 40` in `minv.py` is the weakest constant in the file: two shapes in the
  regime, they disagree, anything in (33, 44] fits.
* `gate_up M=128` is the one rd cell the BN rule gets wrong (~10%).
* The two documentation errors the predecessor lists (the "5.4× over rocBLAS" LM-head
  misattribution, and dead Claim 7 in `COUNTER_SCORECARD.md:64`) are **still unfixed**.
* `64 of 106` `dense_gemm_pipe` instantiations spill (max 2676 B/lane, 1376 VGPR).
