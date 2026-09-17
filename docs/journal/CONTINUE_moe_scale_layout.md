# CONTINUE — the W4A16/W4A8 MoE scale layout flip is DONE and MEASURED; what is left

**Session ended:** 2026-08-07. Serve is DOWN, both GPUs FREE. Everything below is measured and
committed. Predecessor: [`CONTINUE_stride_conflict_and_moe.md`](CONTINUE_stride_conflict_and_moe.md).

| repo | branch | SHA |
|---|---|---|
| `rdna4-hip-kernels` | `perf/moe-scale-layout` (worktree `…-moescale`) | `c3f815c` — **not merged to main** |
| `minisgl-rdna4` | `perf/moe-scale-layout` (worktree `…-moescale`) | `9ad7726f` — **not merged to rdna4** |

Baseline worktrees kept for re-measurement: `rdna4-hip-kernels-moebase` @ `fe3d3c7`,
`minisgl-rdna4-moebase` @ `3e7add85`. Images `minisgl-rdna4:sclbase` / `:sclnew`.

---

## WHAT LANDED

Per-group weight scales are now **`(E, K/group, N)`** and packed zeros **`(E, K/group, N/8)`** —
group-major, output channel CONTIGUOUS. Previously `ws_e[abs_n * num_k_groups + g]` put the lane in
`abs_n`, so 16 lanes of a fragment read 16 addresses `num_k_groups*2` bytes apart: 16 requests per
fragment, once per k-group. Now it is one request.

**Measured, card 0, old CODE vs new CODE (not emulated), 4 reps, spread <1%:**

| | old | new | |
|---|---|---|---|
| g=32 wide=2 (served AWQ) | 3.088 ms, 63.5% HBM | 2.598 ms, 75.4% HBM | **1.189x** |
| g=128 wide=8 (control) | 2.172 ms, 81.1% | 2.155 ms, 81.7% | 1.008x |

**Sampled serve A/B** (Qwen3.6-35B-A3B-AWQ TP=2, 3 reps, provenance asserted, ledgers identical):

| | speedup |
|---|---|
| prefill, all four rungs | **1.054 – 1.070x** |
| decode M=1 / 4 / 16 | 1.004 / 1.008 / 1.005x — **a wash** |

Fixture: `rdna4-hip-kernels/tools/_fixtures/moe_scale_layout_ab_card0.txt` (carries both, plus the
failed first run).

Correctness: **16/16 tensors bit-identical** old-vs-new compiled kernels (a pure reindex must be
bit-identical, not close), `tests/test_w4a16_regdirect.py` 13/13, and
`minisgl tools/test_scale_layout_reindex.py` (dependency-free CPU proof of the loader half).

### Three things the predecessor doc got wrong, for calibration
1. **Scope.** It scoped "five `_scales_op` sites in `layers/moe.py`". The same expression was in
   seven kernel files, and it is **all-or-nothing** — the same `_scales_op` tensor feeds the tiled
   GEMM at prefill and the decode GEMV at M<=2.
2. **Which kernel.** The measured `mmq_regdirect_w4a16_moe` is **not the default served path**;
   `MINISGL_MOE_W4A16` defaults to `0`, so Qwen-AWQ runs the W4A8 tiled MoE. Same defect, never
   measured. Third recorded instance of optimising what isn't dispatched.
3. **The `_hip.h` "siblings" are a non-issue** — gitignored hipify artifacts nothing includes.

---

## THE ONE THAT MATTERS: the first serve A/B FAILED at 0.56x decode

Decode came out **0.56x / 0.64x / 0.66x** with spreads of 0.1–2.4%. Cause: four engine shape reads
still assumed `(E, N, G)`. The damaging one is

```python
_grp = (w13.shape[-1] * 8) // w13_scales.shape[-1]   # quant/kernels.py
```

which returns `N` instead of `G`, makes `_gemv_ok` False, and **silently** drops decode gemm1 from
the scalar GEMV onto the prefill WMMA arm (~8.7x slower on that GEMM). It does not raise.

**The kernel bench and the bit-identity gate are blind to this by construction** — both pass tensor
shapes explicitly, so neither exercises the engine's dispatch. Both were green.

What localised it in one step: **diffing the `engaged()` ledgers between legs.** The AFTER leg never
dispatched `mmq_fp8_moe_gemm1_silu(gemv+prequant)` at all. Do this on every serve A/B.

Fixed in `9ad7726f`, which also asserts the derived `group_size` so a wrong axis fails loudly.

---

## STILL OPEN

1. **`vllm-gfx1201/w4a8_vllm` is BROKEN and this was a deliberate choice.** It builds these kernels
   separately and constructs the old `(N, G)` layout itself — `moe_experts.py`
   `_awq_to_op_layout_single` (`scales_op = sc_e.t()…`, ~:358), `_ct_to_op_layout_single` (~:433),
   and `_wna16_moe_to_op_layout` (~:473). The host bindings now `TORCH_CHECK` the new shape, so it
   fails LOUDLY rather than computing wrong numbers (the two layouts have different shapes, which is
   what makes that assert possible). Flip it the same way: drop the `.t()`, pack zeros along N.
2. **Merge both branches.** They MUST land together — a mismatched pair fails at the first MoE call.
3. **Decode is untouched by this work.** The GEMV holds `nc` warp-uniform, so its scale read was
   already a broadcast. If decode is the target, this layout is not the lever.
4. **Prefill is now ~6% better; the remaining g=32 penalty is 1.21x** (was 1.42x). Whether the rest
   is irreducible traffic (g=32 has 4x the scale/zero bytes) or more inefficiency is unmeasured.
   Re-run the counter probe from the predecessor doc against `abcbb09` to find out.
5. `opt_loop/harvest/R1_moe_splitk_bench.py` still builds the old layout — left alone deliberately:
   it targets the deleted `moe_splitk_hip` package and is marked IMMUTABLE by `COMMANDMENT.md`.
6. Two docs still assert counters are permanently dead on this box
   (`opt_loop/roles/profile_engineer.md` §2, `opt_loop/knowledge/gfx1201.md:223`). Overturned
   2026-08-05, still not updated.
7. `mmq_regdirect_w4a16_moe_gemv*` (3 ops) remains unreachable from the engine — `kernels.w4a16_moe`
   uses the WMMA op at every M. Wire it or delete it.

---

## Traps paid for this session

1. **A missing `uprof` build context** makes `docker build` die with
   `pull access denied … insufficient_scope`, which reads as an auth failure. Pass
   `--build-context uprof=/home/pat/pkgs`. The Dockerfile documents this; I still hit it.
2. **`docker build … | tail` masks the exit code** — a failed build printed `EXIT=0`. Same class as
   the `pmc_have_counter` trap in the predecessor doc. Use `pipefail`/`PIPESTATUS`.
3. **A wrong mount path silently falls back to the image's baked `/opt/kernels`**, which would make
   an A/B a comparison of one build against itself. The dump script asserts the imported package
   path; that assert is what caught it. Keep it.
4. **`hash(str)` is salted per process** — using it to seed two processes' "identical" random data
   silently gives them different data. Use literal seeds in any cross-process comparison.
5. **`replace_all` skips near-identical siblings whose trailing comments differ.** It silently left
   `moe.py:666` on the old axis while fixing its twin at `:646`. Grep after every `replace_all`.
6. A parity **negative control** that slices by axis position (`sc_flat[:, :, 0::2]`) becomes
   vacuous after a transpose — it still prints PASS while testing nothing.
