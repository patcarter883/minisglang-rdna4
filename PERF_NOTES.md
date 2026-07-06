# PERF_NOTES — optimization backlog

Status legend: ✅ implemented + validated · 🔨 implemented, GPU-validation pending · ❌ open

## 2026-07-06 — backlog cleared (branch `perf-backlog` + canonical `rdna4-hip-kernels`)

Both original gating preconditions are met (the ecosystem serves end-to-end; the logit oracle
exists), so the whole deferred list below has been implemented. Code + HIP kernels compile clean for
gfx1201 and register/import in the lean image; each item is marked 🔨 until its GPU logit-oracle /
parity run is green (see `tools/` + `rdna4-hip-kernels/*/tests`), then ✅.

| item | what | state |
|---|---|---|
| [V1] | logit oracle (`tools/oracle_cmp.py`, `qwen3_5_decode_oracle_*`) | ✅ (pre-existing) |
| [A1] | Triton seg=64 / autotuner — mooted: native HIP is the prod path | ✅ n/a |
| [A2] | persistent attention output buffer (no per-forward `empty_like`) | ✅ live GDN capture coherent |
| [A3] | vectorized page-table gather (was per-req list-comp + `torch.stack`) | ✅ live GDN capture coherent |
| [A4] | fused `store_kv` kernel (cast+scale+scatter; bf16/fp16/fp8) | ✅ parity byte-exact (incl. fp8) + live |
| [A5] | fp8-KV per-tensor scale calibration + descale plumbing (opt-in) | ✅ store byte-exact; calib opt-in |
| [S1] | RMSNorm → native `tail_hip.rms_norm(_add)` | ✅ (prior) |
| [S2] | RoPE → native `tail_hip.rope` | ✅ (prior) |
| [S3] | silu (prior ✅) + **gelu** → native `tail_hip.gelu_and_mul` | ✅ parity cos 0.999999 |
| [S4] | fused `sampler_hip` wired into `sample_impl` (no full-vocab sort) | ✅ distributional+cudagraph green + live |
| [S5] | embedding: drop the `zeros_like` masking temporary | ✅ (provably equal) |
| [B2] | reserve GDN/CCA recurrent-state bytes before sizing KV pages | ✅ fail-fast + co-size + boot |
| [GC] | graph capture runs grad-free → **GDN cudagraph now works** | ✅ Qwen3.5-4B captures + coherent |

## [GC] Graph capture grad-free — the real "GDN is eager" fix

`GraphRunner._capture_graphs` (and `capture_verify_graphs`) ran the warmup + captured `model.forward()`
with **autograd active** (no `inference_mode`) — while the serve forward is `@torch.inference_mode()`
(Scheduler). With grad on, the models' in-place-on-view ops (`q_norm`/`k_norm` `forward_inplace` on a
qkv-`split` view) trip the autograd view-guard, breaking capture. That, not any real limitation, is
why GDN "had to run eager": `GDNGraphCapture`/`CCAGraphCapture`/MLA decode capture were all already
wired. Wrapping both capture forwards in `torch.inference_mode()` fixes it: Qwen3.5-4B (GDN hybrid)
now captures all decode sizes and generates coherently (France→Paris, primes 2 3 5 7 11). The stale
"GDN/CCA cudagraph out of scope / eager only" notes are lifted; the only remaining eager paths are
legitimate — dynamic-shape prefill and the opt-in atomicAdd MoE scatter (gated to the capturable
`gather_reduce` under graphs, `MINISGL_MOE_SCATTER=0`).

## Detail

- **[A2] Output buffer.** `RDNA4Backend._get_out_buf` — one persistent `[tokens, Hq, D]` buffer,
  grown to the largest token count, reused across eager forwards (rdna4 Triton path + both prefill
  helpers). Safe: attention output is consumed by o_proj before the next attention call, and these
  eager paths are never cudagraph-captured (decode-capture returns kernel-owned output).
- **[A3] Page table.** `prepare_metadata` and `HIPAttnBackend._fill_decode_static` now gather all
  rows in one advanced-index (`page_table[table_idx, :max:page_size]`) instead of a per-req Python
  loop + `torch.stack`. Provably identical output; removes host-side per-req work every decode step.
- **[A4] Fused store.** `tail_hip.store_kv(k, v, k_cache, v_cache, out_loc, k_scale, v_scale)` — one
  kernel does cast-to-cache-dtype + per-tensor `1/scale` + scatter, replacing the torch
  scatter+cast in `MHAKVCache.store_kv`. Covers the **default bf16** path (scale 1.0), fp16, and fp8
  (e4m3 via `__builtin_amdgcn_cvt_pk_fp8_f32`, RNE). Parity test: `rdna4-hip-kernels/tail/tests`.
- **[A5] fp8-KV scale.** `MHAKVCache` accumulates per-layer `|k|/|v|` amax and
  `finalize_kv_calibration()` freezes a static per-tensor scale = amax/448; the attention fp8 ops now
  receive that descale instead of a hardcoded 1.0. **OFF by default** (`MINISGL_KV_FP8_CALIBRATE=1`):
  a correct static scale needs representative data + a freeze before real requests store, so it is
  meant to be driven by a calibration harness. Default fp8-KV behaviour is byte-identical to before.
- **[S3] gelu.** `layers/activation.py::gelu_and_mul` → `tail_hip.gelu_and_mul` (exact/erf gelu to
  match `F.gelu(approximate="none")`), fp32 torch fallback kept for off-dtype / disabled.
- **[S4] sampler.** `engine/_sampler_hip.py` gates the fused kernel into `sample_impl`
  (`MINISGL_FUSED_SAMPLER=1`, soft fallback to the torch sort path). Greedy (argmax) and the grammar
  bitmask path are unaffected. Added `sampler:sampler_hip` to the Dockerfile build list. Validate
  with `rdna4-hip-kernels/sampler/tests/sampler_parity.py` before trusting the default-on.
- **[S5] embedding.** `y.mul_(mask.unsqueeze(-1))` in place on the fresh gather instead of
  `torch.where(mask, y, zeros_like(y))` — same result, no full zeros temporary. TP>1 only.
- **[B2] memory defaults.** `_recurrent_state_bytes()` computes the GDN/CCA state-cache size from
  arch + `max_running_req`; `_determine_num_pages` subtracts it before dividing the budget into KV
  pages. Dense/MHA/MLA reserve 0 (unchanged); GDN/CCA now co-size the two caches automatically
  instead of OOMing at the stock `--max-running-requests` (the old manual-cap workaround).

## GPU validation gate (before flipping any 🔨 → ✅)
1. `tail/tests/test_tail.py` green (rms/silu/**gelu**/**store_kv**/rope) under a lease.
2. `sampler/tests/sampler_parity.py` green (greedy/determinism/distributional/cudagraph).
3. Engine logit oracle cos-sim unchanged with A2/A3/S3/S4/S5/A4-bf16 active (bf16 serve path).
4. GDN 35B boots at the default `--max-running-requests` (no OOM) — [B2].
5. fp8-KV oracle (`MINISGL_KV_FP8=1`, calibration on) recovers cos-sim vs the scale-1.0 baseline — A5.
