# CONTINUANCE — vhip (vLLM 0.24 HIP) spec-decode + GLM serving

Session of 2026-07-27/28. Everything below is MEASURED unless labelled a hypothesis.

## Where to work

Worktree `/home/pat/code/minisgl-rdna4-vhipspec` (branch `task/vhip-spec`), **not** the shared tree —
per CLAUDE.md source isolation. Nothing is committed yet; `git status` shows the modified
`docker-compose.yml` plus untracked `tools/vhip_launch.sh`, `tools/profiling/`, `tools/vhip_patches/`.

Launch anything with `tools/vhip_launch.sh <preset>` (books both cards via `gpu-lease`, points every
mount at this worktree):

    tools/vhip_launch.sh qwen-mtp     # Qwen3.6-35B-A3B + MTP K=2   <- the spec work
    tools/vhip_launch.sh qwen         # same, no spec (baseline)
    tools/vhip_launch.sh glm          # GLM-4.7-Flash-AWQ on mla_hip
    tools/vhip_launch.sh glm-mtp / glm-eagle3 / laguna-dflash
    docker compose -p lease-vhip-<preset> down     # stop + release the lease

Env knobs worth knowing: `VHIP_MAXLEN VHIP_MNS VHIP_MEM VHIP_PREFIX_CACHE VHIP_EXTRA
VLLM_GDN_HIP_SPEC VLLM_GDN_HIP_SPEC_DEBUG VLLM_GDN_HIP_SPEC_NO_SCATTER VLLM_GDN_HIP_SPEC_AB
VLLM_GDN_HIP_SPEC_GATHER`. **`VLLM_GDN_HIP_SPEC` defaults to `0` in `docker-compose.yml`** — pass it
explicitly or you are measuring the fla-Triton path.

Serve-side numbers: `python tools/vhip_patches/bench_spec_serve.py --port 8000` (coherence + bs=1
tok/s + concurrency-4, same three metrics as the table below).

**MIND THE METRIC.** That bench reports **end-to-end** tok/s: `completion_tokens / wall clock`, so it
charges TTFT, HTTP and detokenize against decode. Measured 2026-07-28 that is ~4% below the true
decode rate (TTFT ~0.11 s of a 2.5 s / 200-token request). The **decode** rate is
`(completion_tokens - 1) / delta(vllm:request_decode_time_seconds_sum)` across one request, which
agrees with vLLM's own logged "Avg generation throughput" to ~1%. Both are quoted below. When
comparing against ANY older number, establish which metric it was first — the two differ by ~4% and
that is enough to invert a close call.

**Diagnose standalone, not by booting serves** (user directive, and it is what finally cracked the
spec bug after many wasted boot cycles). The repros in `tools/vhip_patches/test_*.py` run in seconds
in a plain container with `PYTHONPATH=/patches/gdn/torch-ext`; a serve boot (~4 min) is only for the
final "does it actually serve" and for real throughput numbers.

## DONE 2026-07-28 — SSM state is stride-aware; gather/scatter deleted

`gdn_prefill` / `gdn_prefill_verify` / `gdn_prefill_wmma` / `gdn_prefill_chunked` now take
`ss_slot/ss_hv/ss_v/ss_k` and index the recurrent state in place, matching what the decode kernels
always did. Launchers pass `ssm_state.stride()` via a new `GDN_SS_STRIDES(SSM)` macro next to
`GDN_DISPATCH_SSM`. Bindings are unchanged (strides are read inside the launcher, not plumbed
through the op schema). The compact 1-based gather/scatter is gone from
`_forward_core_gdn_hip_spec_impl`, and so is the whole-cache `.contiguous()` shadow + copy-back that
the NON-spec prefill branch was still doing for the same reason.

**What the layout actually is** (read from vLLM, not inferred): `gpu_model_runner.py`, `MambaSpec`
branch, carves one raw buffer per layer with `torch.as_strided`, stride
`(num_element_per_page, *inner_contiguous_strides)`. So the inner `(HV,V,K)` dims **are** contiguous —
the only wrong term was the SLOT stride, which is the padded page size (conv and ssm share a page,
ssm at a storage offset). The old packed `((slot*HV+hv)*V+r)*K` therefore walked into a neighbouring
slot's conv half. That is the whole bug.

**Validated standalone** — `tools/vhip_patches/test_stride_aware_state.py` rebuilds that exact
`as_strided` layout (NaN-filled page gaps so a mis-strided read can't look plausible) and runs all
four kernels against both a contiguous and a paged state: **bit-exact** (`torch.equal`) on core,
scratches and persisted state, for fp32 and bf16 state dtype. `gdn/tests/test_gdn.py` ALL GREEN.
`test_prefill_then_verify.py` STEP 3 (strided state) now matches the contiguous run exactly.

One caveat that surfaced and is NOT a striding issue: `gdn_prefill_chunked` returns NaN under
raw-randn `A_log` because it forms explicit `gamma_C/gamma_j` ratios and the cumulative gamma
underflows to 0 over a 32-token chunk (0/0). Pre-existing, not on the serve path — the WMMA kernel
exists precisely because of it (it keeps the ratios in log space). The test damps decay for that
case only.

Also fixed as a side effect: `_forward_core_gdn_hip_tokens` (the non-spec half of a MIXED spec step)
was already handing the raw paged state to `gdn_prefill_wmma` — a live silent-corruption path.

Rebuild: `docker run --rm --entrypoint bash -v "$PWD":/build -w /build vllm24-hip:combined -lc
'GPU_ARCHS=gfx1201 bash local/build_local.sh'` from `tools/vhip_patches/gdn/`.

### The perf expectation did NOT hold — measure, don't assume

The previous entry predicted this would close the gap. It did not. Measured A/B in the same config,
`VLLM_GDN_HIP_SPEC_GATHER=1` restores the old gather/scatter so the cost is attributable rather than
assumed (the knob only exists because the kernels are now stride-aware — a contiguous compact buffer
is just another stride set to them):

| gdn_hip spec | bs=1 | conc-4 mean | conc-4 peak |
|---|---|---|---|
| legacy gather/scatter | 49.7 | 134.7 | 141.1 |
| raw paged (this fix)  | **53.8** | **143.4** | **151.9** |
|  | +8.2% | +6.5% | +7.7% |

So the gather/scatter was worth ~8%, not the gap. (49.7 reproduces the old 49.6 almost exactly,
which is what makes the comparison trustworthy.)

### Why fla-Triton is faster — READ FROM ITS SOURCE, not inferred

`vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`,
`fused_sigmoid_gating_delta_rule_update_kernel`. The spec branch of vLLM's `_forward_core`
(`qwen_gdn_linear_attn.py:1457`) calls it **once per layer** and hands it the raw paged `ssm_state`,
the **2-D slot table**, and `num_accepted_tokens`. The kernel then:

* selects the load slot **in-kernel**: `i_t = num_accepted_tokens[i_n] - 1`, then
  `state_idx = ssm_state_indices[i_n, i_t]`, with the `state_idx <= 0` NULL check inline;
* runs the **same scalar recurrence we do** — `b_h *= exp(g)`, `b_v -= sum(b_h*b_k)`,
  `b_h += v k^T`, `b_o = sum(b_h*b_q)`. Confirms the earlier falsification: the non-WMMA verify is
  NOT the difference, fla is scalar too;
* **publishes each token's state in-place inside the token loop** (`INPLACE_FINAL_STATE`):
  `p_ht = ht + ssm_state_indices[i_n, i_t] * stride_final_state_token`. No scratch buffer, no
  Python scatter. It is also stride-aware (`stride_final_state_token`), same as us now.

So the difference is not the math — it is that fla fuses the per-position state publish into the
recurrence, while we materialise a `[max_qlen, N, HV, V, K]` scratch and then scatter it from Python.

**Measured per-layer breakdown** (`tools/vhip_patches/bench_spec_layer_breakdown.py`, standalone,
bs=1, qlen=3, real paged layout, 30 GDN layers):

| phase | us/layer | ms/step x30 |
|---|---|---|
| fp32 pre-casts (mixed_qkv, a, b) | 10.9 | 0.33 |
| causal_conv1d_fwd_verify | 5.7 | 0.17 |
| split + 3x reshape.contiguous | 19.1 | 0.57 |
| gdn_prefill_verify (the recurrence) | 104.4 | 3.13 |
| **per-position publish loop** | **141.4** | **4.24** |
| whole layer | 242.0 | 7.26 |

**The Python publish loop costs MORE than the recurrence kernel it exists to serve** — 141 us to move
3 MiB, i.e. ~21 GB/s, so it is op-overhead/`.to()`-copy bound, not bandwidth bound. That is the one
thing fla does not do at all.

Honest limit: this accounts for ~4 ms of the 15.3 ms/step gap (and the microbench is a warm, N=1
lower bound). **The other ~11 ms is NOT yet attributed** — that needs an in-serve rocprof, not
another guess.

**The old `_SPEC_NO_SCATTER` bisect did not falsify this.** It was read as "disabling the scatter
measures *slower* (42.5 vs 49.6), so the scatter is not the cost". But dropping the publish leaves
the next step loading a slot that was never written -> wrong state -> drafts rejected -> accept_len
collapses. tok/s falls for a CORRECTNESS reason while the step itself got cheaper. The knob confounds
speed with acceptance and cannot answer this question; only a fused-publish kernel can.

### DONE — fused in-kernel publish (2026-07-28), +37.7% bs=1

Built it. `gdn_prefill_verify_kernel` and `causal_conv1d_fwd_verify_kernel` gained a templated
`INPLACE` publish policy — **ONE core, two store policies**, per `KERNEL_CORE_POLICY.md`, NOT a forked
kernel. With `slot_table [N, max_qlen]` + `num_accepted [N]` they resolve the load slot in-kernel
(`slots[n, num_accepted[n]-1]`) and store each token's state straight into
conv_state/ssm_state[slots[n, t]] inside the token loop — exactly fla's `INPLACE_FINAL_STATE`. Deleted
from the glue: the per-position scatter loop, both scratch buffers, and the host-side load-slot
gather. `VLLM_GDN_HIP_SPEC_FUSED=0` restores the old path.

Both kernels are templated on the INDEX dtype: vLLM's `spec_state_indices_tensor` /
`num_accepted_tokens` are **int32** (`gdn_attn.py:127,162`), and a caller-side `.long()` would have
reintroduced exactly the per-layer op the fusion removes. (First boot failed with
`expected scalar type Long but found Int` — the standalone test had used int64.)

Validated bit-exact vs the scratch+scatter path across fp32/bf16 state, int32/int64 slot tables,
num_accepted in {1,2,3}, and ragged query lengths, with the scratch confirmed elided
(`tools/vhip_patches/test_fused_publish_parity.py`). `gdn/tests/test_gdn.py` ALL GREEN — the scratch
contract minisgl depends on is untouched.

| gdn_hip spec (same config) | bs=1 | conc-4 mean | conc-4 peak | accept_len |
|---|---|---|---|---|
| scratch + Python scatter | 53.8 | 143.4 | 151.9 | 2.51/3 |
| **fused in-kernel publish** | **74.1** | **194.0** | **208.8** | **2.511/3** |
|  | **+37.7%** | **+35.3%** | **+37.5%** | unchanged |

accept_len is unchanged to three decimals, so this is pure step-time, not an acceptance artefact.

**The standalone microbench under-predicted this badly and that is worth remembering.** It measured
the publish loop at 141 us/layer = 4.24 ms/step and I wrote that it should be worth ~4 ms of the
15.3 ms gap. In-serve it was worth ~12.8 ms (step time 46.7 -> 33.9 ms). A warm, N=1, tight-loop
microbench is a LOWER BOUND on host-op cost in a real step — the same ops cost ~3x more once they are
competing with a full model forward. So most of what I labelled "unattributed" was the same publish
loop after all. Trust the serve number over the microbench for host-op overhead.

Remaining gap to fla: **77.1 vs 82.5 decode (-6.5%)**, 74.3 vs 79.2 e2e (-6.2%) — the same gap on
both metrics, so it is real and not a measurement artefact. no-spec (83.3 decode) still wins.

## State of play

### MTP on Qwen3.6-35B-A3B (TP=2, K=2, graph capture ON)

**Re-measured 2026-07-28 at ONE config** — 32768 ctx, max_num_seqs=8, fp8 KV,
`tools/vhip_patches/bench_spec_serve.py`, 3 trials each:

| config | bs=1 **decode** | bs=1 e2e | conc-4 mean (e2e) | stable | accept_len |
|---|---|---|---|---|---|
| no-spec | **83.3** | 81.8 | **241.8** | 3/3 | — |
| fla-Triton spec (`VLLM_GDN_HIP_SPEC=0`) | 82.5 | 79.2 | 199.4 | 3/3 | 2.48/3 |
| gdn_hip spec (`=1`, **fused publish**, current) | 77.1 | 74.3 | 194.0 | 3/3 | 2.511/3 |
| gdn_hip spec — scratch+scatter (superseded) | not measured | 53.8 | 143.4 | 3/3 | 2.51/3 |
| gdn_hip spec — compact gather (superseded) | not measured | 49.7 | 134.7 | 3/3 | — |

decode = server-side `(tokens-1)/delta(request_decode_time_seconds)`; e2e = the bench's
`tokens/wall`. The two superseded rows were only ever taken on e2e — the +37.7% fused win is an
e2e-vs-e2e comparison, so it stands, but do not mix those rows with the decode column.

**On the older "fla-Triton = 98.0 tok/s" figure: NOT REPRODUCIBLE AS RECORDED.** Chased 2026-07-28
because it is a big claim and the current number is far below it. At this config fla-Triton measures
**79.2 e2e / 82.5 decode**, and vLLM's own logged "Avg generation throughput" independently says
**83.2** — three methods agreeing inside ~5%, none near 98. The metric difference is worth ~3 tok/s,
not the ~19 required, so this is NOT the e2e-vs-decode confusion. It cannot be called *wrong* either:
the config it claims (60k ctx, max_num_seqs=4) **refuses to boot today** under MTP (60k exceeds the
KV ceiling once a drafter is resident; MNS=4 fails the hybrid allocator at 0.42 vs 0.43 GiB), so
there is no way to re-run it as written. Treat 98.0 as unverified until someone reproduces it on a
config that boots, and quote the numbers above instead.

**RETRACTED — "MTP is break-even/net-negative on this model" was WRONG.** It compared spec against
*our own degraded GDN baseline*, not the engine's. Every row in the table above runs the gdn_hip
plugin on the plain-decode path (`VLLM_GDN_HIP_SPEC=0` only re-routes the SPEC branch to Triton), so
"no-spec 83.3" is our stack's no-spec.

**The real engine baseline is 92 tok/s** — vLLM 0.24 ALL-STOCK (fla-Triton GDN + Triton attention),
already measured and recorded in memory `vllm24-hip-gdn-decomp-and-fused-decode-lever` /
`qwen-decode-88pct-busy-elementwise-dominates` (also "matches tcclaviger stock image 92-95"). Our HIP
trajectory on the identical engine: **46 → 69 (slot-fix) → 83.3 (today, stride-aware + fused decode)**
vs stock 92. So we are still ~9% BELOW stock at no-spec, and MTP has been judged against a handicapped
baseline the whole time. Against stock 92, the old 98.0 spec figure is +6.5% — an ordinary MTP win,
not an outlier. That also resolves the "no-spec down, Triton up" contradiction above: the old table
almost certainly mixed a stock-GDN spec row with an our-stack no-spec row.

Do NOT re-derive the stock number by booting all-stock — it pays a ~30-minute GDN Triton compile
(vendored `vllm/model_executor/layers/fla/ops/`, compile-bound not autotune-bound, cache does not
persist on ROCm). It is 92; use it.

All three are coherent, and gdn_hip spec is *correct* — accept_len tracks fla-Triton to within noise,
which a subtly-wrong state could not do — but it is still the slowest of the three.

Four corrections to the older table above, all from measurement:
* **the older table's absolute numbers do not reproduce** — see the 98.0 note above. Its no-spec
  (75.1) is *below* today's (81.8 e2e) while its fla-Triton (98.0) is far *above* today's (79.2 e2e).
  A pure config shift would move both the same way; it doesn't, so something other than config
  differed. Unresolved — do not build on those figures.
* **The earlier numbers were NOT all one config.** `60k ctx` does not boot under MTP at all (the
  launcher already caps spec presets at 32768 for this reason) and `max_num_seqs=4` does not boot
  either — it leaves *less* KV headroom than 8 (0.42 vs 0.43 GiB needed) because of hybrid
  attn+mamba page quantization. So the older figures cannot be re-run as recorded at all, which is
  why the whole table was rebuilt at a config that boots rather than patched.
* **`docker-compose.yml` defaults `VLLM_GDN_HIP_SPEC` to `0`, not `1`.** The old entry called `=1`
  "the current default" — it is not. Any qwen-mtp run without an explicit `VLLM_GDN_HIP_SPEC=1`
  measures the **fla-Triton** path. Check
  `docker inspect <ctr> --format '{{.Config.Env}}' | grep GDN_HIP_SPEC` before attributing a number.
* **The fla-Triton concurrency HANG did not reproduce** (3/3 clean at conc-4). Expected: the hang is
  a COLD `chunk_scaled_dot_kkt` autotune, and the mounted Triton cache is warm for these shapes. It
  is a cold-start property, so it can return on a fresh shape — the diagnosis stands, the "HANGS"
  cell does not, unconditionally.

Given no-spec wins on every axis here, the default `VLLM_GDN_HIP_SPEC=0` is fine and the real
question is whether MTP is worth running on this model at all.

**The concurrency hang was never a TP deadlock.** It is a cold Triton autotune of
`chunk_scaled_dot_kkt` (the ONLY `@triton.autotune` kernel in the GDN path), reached via
`chunk_gated_delta_rule` on any step that mixes spec-decode with a prefill. py-spy showed both ranks
in `do_bench`/`make_amdgcn` 28/28 samples; a standalone run sat in one tune **19+ minutes** against a
300 s RPC timeout. bs=1 never hit it because a prefill step there has no in-flight spec sequences.

**The corruption bug** (one correct token then token 0, `" Paris!!!!!!"`): the verify kernels were
handed a non-contiguous paged state. `core` is computed in registers so it stayed correct; the state
written back was garbage. Worked around first with a whole-cache `.contiguous()` shadow (a 2.75 GB
copy worth ~64% of decode, 39.1 tok/s), then with a compact 1-based gather of only the touched slots
(49.6). **FIXED properly** — see the stride-aware section above; both workarounds are deleted.

### GLM-4.7-Flash — done

Was a hard segfault in CK FlashAttention MLA prefill on gfx1201. Now serves on our kernels:
* new `mla_vllm` plugin (`tools/vhip_patches/mla_vllm/`) registering MLA **prefill** (`mla_prefill`)
  and **decode** (`mla_decode`/`mla_verify`) backends. Decode 56.9 tok/s vs 53.0 on TRITON_MLA.
* added `mla_prefill_lse` to the kernel for vLLM's chunked-context merge; parity-tested vs fp32.
* the decode backend declares `QueryLenSupport.UNIFORM`, which is what makes MLA spec decode
  possible at all (TRITON_MLA is SINGLE_ONLY and asserts out).
* `awqdense_vllm` plugin routes `auto_awq` **dense** linears to our W4A8 kernel — vLLM only consults
  `_POSSIBLE_KERNELS` for AWQ on CUDA/CPU, so on ROCm our kernel was dead weight for every AWQ model.
* GLM must run `--dtype bfloat16` (mla_hip is bf16-typed) — see backlog.

### Other findings

* **EAGLE3 blocked upstream**: `glm4_moe_lite` doesn't implement `SupportsEagle3`. The cached
  `thoughtworks/GLM-4.7-Flash-Eagle3` drafter is otherwise valid (`fc [2048,6144]`, `d2t`/`t2d`).
  Needs `EagleModelMixin` + aux-hidden-state plumbing on that model class.
* **DFlash**: z-lab's Qwen drafters are rejected by vLLM (mixed sliding/full `layer_types`,
  vllm#40898). Laguna is the supported pairing (`laguna-dflash` preset) — OOM'd on KV sizing, not
  retried.
* GLM MTP boots on the new MLA decode backend but output degrades — open, untouched since.
* `glm47` reasoning parser drops `reasoning_content` (thinking generated but returned empty; with
  `max_tokens` truncation the whole response comes back empty).

## Do NOT re-derive — already eliminated by measurement

For the GDN spec corruption: the slot contract (instrumented dump matches vLLM's kernel exactly:
load `spec_state_indices[s, num_accepted[s]-1]`, store positions 0..T-1, no NULL slots); graph
capture (reproduces under `--enforce-eager`); `use_l2norm=0` on pre-normalised q/k (bit-identical to
`=1` on raw); state transpose; `has_init`; conv-state layout and striding (all three layouts
give identical finite results standalone).

**RETRACTED 2026-07-28**: "the per-position scatter is not the cost (disabling it measures *slower*,
42.5 vs 49.6)". That bisect is confounded — no publish means the next step loads an unwritten slot,
so accept_len collapses and tok/s drops for a correctness reason. It never measured the scatter's
speed. Standalone the publish loop is 141 us/layer, MORE than the recurrence kernel (see above).

Two things I asserted and later **falsified** — do not repeat them as causes: "no WMMA in
`gdn_prefill_verify`" (T=3 tokens, and vLLM's spec kernel is also a scalar recurrence; the
non-WMMA choice is a deliberate bit-exactness decision by minisgl's author), and "~300 extra
launches from the per-position scatter". The perf cause is *not yet proven* — the microbenchmark
meant to attribute it (`bench_spec_kernels.py`) timed out and was killed. Prove it before claiming.

Also: my hand-written PyTorch gated-delta-rule reference (`test_gdn_ref_parity.py`) is **wrong** — it
fails the production-validated config. Don't trust its output; compare against the real Triton path
or `mla_decode`-style known-good ops instead.

And, added 2026-07-28: the **gather/scatter was not the perf cause** either — measured +8% in a
same-config A/B (above), against a prediction that it would close the gap. That makes THREE
falsified explanations for this gap (no-WMMA verify, ~300 scatter launches, gather/scatter copies).
Do not assert a fourth without a profile.

## Reference: minisgl's caller

`/home/pat/code/minisgl-rdna4/python/minisgl/gdn/layer.py:405-431` — the working consumer of
`causal_conv1d_fwd_verify` + `gdn_prefill_verify`. It owns plain contiguous state
`conv_state (slots, C, W-1)` / `ssm_state (slots, HV, V, K)`, which is why it never hit any of this.
vLLM instead allocates `conv (W-1)+num_spec` wide (mamba_utils.gated_delta_net_state_shape) and hands
over possibly-strided views.

## Backlog (tasks carried)

* ~~close the gdn_hip spec-verify perf gap via a fused publish~~ **DONE** — +37.7% bs=1, gap to fla
  down from 15.3 to 2.5 ms/step. See the fused-publish section above.
* the LAST 2.5 ms/step vs fla is unattributed. Candidates, all unproven: the fp32 pre-casts of
  mixed_qkv/a/b (fla loads bf16 and converts in-register — measured 10.9 us/layer standalone, so
  likely ~1 ms in-serve by the 3x rule), the 3x `reshape().contiguous()` after the conv split
  (19.1 us/layer), and conv being a separate launch from the recurrence (fla fuses gating+recurrence
  but still runs conv separately, so this one is probably NOT it). rocprof before building.
  A WMMA verify path is NOT the lever — fla is a scalar recurrence too and is faster anyway.
* **CLOSE THE 83.3 -> 92 no-spec GAP vs stock. This is the real headline lever, and it is NOT GDN.**
  Measured profile (memory `qwen-decode-88pct-busy-elementwise-dominates`, Qwen35B-AWQ TP=2 bs=1
  decode, torch profiler): decode is **88.6% GPU-BUSY** (so NOT launch/gap-bound — that belief came
  from Laguna/MLA and was wrongly generalised), and
  - `at::native::elementwise_kernel_manual_unroll<128,4>` = **64.4%** (852 ms/1.32 s, 3690 calls,
    **231 us avg**, ~2.5x/layer). 231 us at M=1 is WEIGHT-sized, and every other elementwise entry is
    1-2 us -> signature of an **AWQ int4->fp16 weight DEQUANT run as a standalone elementwise op
    before an fp16 GEMV instead of being fused into an int4-native GEMV** (dequant ~10x its own GEMV).
    Paired with `wvSplitK` at only 6.9%, which fits.
  - **`gdn_decode_kernel` = 1.4%** (18.5 ms). nccl comms 8.9%, fused_moe_gptq_awq 8.5%, copy_ 2.4%.
  So GDN cannot be the residual: making the whole GDN decode kernel FREE is worth ~1% e2e. The
  original 92->46 collapse WAS GDN (launches + the whole-cache shadow copy) and that is now fixed
  (46 -> 83.3); what remains is the dequant/GEMV/comms stack.
  **CHEAPEST NEXT TEST (no Triton compile, one env flip):** `VLLM_ROCM_W4A8_AWQ_DENSE=0` and/or drop
  `w4a8_fp8_wmma_register`/`awq_dense_hip` from `VLLM_PLUGINS`, keeping gdn_hip on. If the 231 us
  elementwise is ours, no-spec should move toward 92 and the exact op is identified. The memory notes
  the exact op ID is "pending an eager+record_shapes+with_stack re-profile" — do that
  (`tools/profiling/`, PROF_EAGER/PROF_SHAPES/PROF_STACK) rather than guessing.
* re-judge MTP ONLY against stock 92, not against our own no-spec — see the retraction above.
* GLM MTP output degradation on the new MLA decode backend
* `glm47` reasoning_content dropped
* template `mla_hip` for fp16 so GLM need not be pinned to bf16 (bf16 53.0 vs fp16-Triton-FA 55.5)
* EAGLE3 `SupportsEagle3` on `glm4_moe_lite`; Laguna DFlash KV sizing
* land the compose presets + docs (nothing committed yet)
