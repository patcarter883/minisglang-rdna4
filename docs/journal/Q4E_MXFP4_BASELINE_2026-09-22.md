# A baseline arm for the q4e degeneration: the author's own engine — 2026-09-22

Third thread of the day. [`Q4E_DEGENERATION_2026-09-22.md`](Q4E_DEGENERATION_2026-09-22.md) chased
answer DELIVERY, [`Q4E_RETRIEVAL_2026-09-22.md`](Q4E_RETRIEVAL_2026-09-22.md) narrowed the defect to
CONTEXT RETRIEVAL in q4e-exclusive machinery (QSA or PLE, GDN exonerated at p=0.002). Both ran out of
the same thing: **there is no second implementation to compare against.** Every reproduction attempt
has been our engine versus our engine.

`tcclaviger/vllm:dev` serving `tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8-GPTQ` is that second
implementation — the same weights on the engine they were packed for, on the same two cards, same
ROCm stack. This records what it took to boot on 16 GiB cards and what it measured.

## 1. It serves, and it answers the failing cases correctly

    K  arithmetic  2.2s   ->  43
    R  verbatim    2.6s   ->  @docs/ui-plan @ui core/src/mc_velocity.c mc_velocity_update  (exact)
    R  needle      2.8s   ->  2311-XON-8919                                               (exact)

**~26 tok/s decode**, against our own engine's best measured arm on this model (20.31 tok/s, expert
cache at low_water 25). That is with 19% of the routing mass held VRAM-resident.

A clean pass on a fresh serve proves nothing about degeneration — §5 of the retrieval journal records
four such passes on our own engine. Its value is as an instrument: it is registered under the SAME
model id Hermes already uses, on the same port, so the failing prompt can be replayed with no client
change.

## 2. Four --expert-* flags it cannot boot without, and they STACK

Fixing one reveals the next, so they have to be fixed together. Each failure arrives minutes apart.

    --expert-resident-layers=      default "0,1,2" = 3 x 512 experts fully resident = a 1584-slot
                                   FLOOR. The whole cache is ~3351 slots, so the floor alone
                                   overran it. Worth +3% decode on 32 GiB cards.
    --expert-offload-mem 57        LATENT behind the first. Expert set 29.94 GiB/rank; the VRAM
                                   cache holds ~4, so ~57 GiB MUST be host-backed. The plan's auto
                                   bound is 44.87 GiB and cannot hold it, and it then tries to pin
                                   the excess VRAM-only with nowhere near enough slots.
    --expert-precise-memory        the default keeps ~1 GiB/GPU for the compile phase: 2.35 GiB of
                                   cache, 1419 slots vs 3351 -- 9% vs 19% of routing mass resident,
                                   which on a PCIe-bound arm IS the decode rate.
    --expert-direct-load           the default stages each layer's raw experts in PAGEABLE RAM
                                   first; at 57 GiB of pinned rows there is no room for a copy.

## 3. The KV cap: why max-model-len and --enforce-eager are both useless here

This one cost the most and is the most transferable. With expert offload on, the fork **hard-caps the
KV cache at the plan's own reserve** (`gpu_worker.py:756` — "the plan's overhead headroom is for
compile, capture and request-time allocations, not KV"). That reserve is built from a registry
estimate that is **13.3% low** on this checkpoint. The worker says so itself:

    expert offload plan check: runtime overhead planned 5.25 GiB, measured 4.09 GiB (-1.16);
    KV planned 8937 B/token, measured 10125 B/token (+1188)

So one 32768-token request needs 0.31 GiB, the cap hands it 0.28, and the boot dies in
`_check_enough_kv_cache_memory` — *after* weights load and graphs capture, ~5 minutes in — while
`1.77 GiB of overhead headroom stays free` beside it, because the overhead figure is itself 1.2 GiB
too HIGH.

Two levers that look like they should fix it and provably cannot:

1. **Lowering `--max-model-len`.** The cap scales with it, so the shortfall is PROPORTIONAL:
   1.08 x 8937 = 9652 < 10125 at *every* length. vLLM's own error message suggests max_model_len
   28800; that advice assumes available KV stays fixed while the length drops, which this fork's cap
   makes false. It fails again at 28800, and at 16384.
2. **`--enforce-eager`.** The 0.55 GiB the plan budgets for graphs is subtracted from `room`, so
   releasing it grows the EXPERT CACHE and leaves the KV reserve byte-identical. Measured across two
   plan runs: cache 4.07 -> 4.63 GiB, KV reserve 0.29 GiB both ways.

`--kv-cache-memory-bytes` works, because it returns from `determine_available_memory` at
`gpu_worker.py:659`, before the cap at :756. Cost: it skips memory profiling, so the plan-vs-measured
drift report above stops printing.

### 3a. Graph capture is not the context tax here; the expert cache is

Pat's concern was that graph capture cost >150,000 tokens of context on minisgl. On this engine it
measured **0.43 GiB — about 46,000 tokens** — cheap only because `--max-num-seqs 2` captures 6 graphs.
Priced in KV tokens at the measured 10125 B/token, the card actually goes:

    non-expert weights   4.95 GiB  = 525,000 tokens
    runtime overhead     4.03 GiB  = 427,000 tokens   (measured, not the planned 5.25)
    expert cache         4.07 GiB  = 432,000 tokens   <- the real context tax, and it has no flag
    CUDA graphs          0.43 GiB  =  46,000 tokens

`max_model_len` does double duty: it caps a request AND it is the only knob that moves VRAM from the
expert cache to KV (the plan sizes its reserve from it and hands the remainder to experts).

    max_model_len   expert slots   routing share VRAM-held   KV we can set   = tokens
        32,768         3,359              0.189                2.08 GiB       220,128
        65,536         3,117              0.176                2.37 GiB       251,333
       131,072         2,633              0.149                2.50 GiB       262,000
       262,144         1,664              0.094                4.14 GiB       438,692

Set to 131,072 to match our own engine, which passes no context flag at all — it serves the model's
native length and lets the KV pool be the limit, so an apples-to-apples baseline must too. The ~21%
of resident expert share it costs is a real decode cost on a PCIe-bound arm, taken deliberately: a
baseline that cannot hold a long agent session cannot be compared against the serve under debug.

### 3b. Host RAM is the hard wall, and it moves the OPPOSITE way

88 of 93 GiB used once up (57 GiB of mlocked expert rows plus two rank processes). Shrinking the VRAM
cache to buy KV GROWS the host-backed set, so `--expert-offload-mem` has to come DOWN as
`max-model-len` goes UP — 57, not 58, at 131,072. The two pressures are opposed.

## 4. Pre-flight the plan in 15 seconds, not 5 minutes

The expert plan runs in the API server at the end of `create_engine_config`, before a single weight
loads. So parsing the real argv through `AsyncEngineArgs.add_cli_args` + `create_engine_config`
reproduces any plan failure in ~15 s. This is what turned a guess-and-reboot loop into a sweep; it
caught both stacked failures and an option-name error, and it is why the final launch was validated
before it ran. Scripts kept at `scratchpad/{argvprobe,planprobe}.py`.

It does NOT catch worker-stage failures — the KV cap is applied after profiling, so the KV death was
only visible in a real boot. Pre-flight bounds the plan, not the serve.

## 5. Two served names, on purpose

`--served-model-name Qwen3.8-Flash-Next Qwen3.8-Flash-Next-MXFP4-tcvllm`. The first is the id Hermes
already uses for our minisgl q4e serve (17 sessions in `~/.hermes/state.db` under that exact string,
5d26424a4be3 among them) and this arm listens on the same port 1919, so the failing prompt replays
with no client change. The second exists because the first destroys provenance on its own: the census
that exonerated GDN keys on `sessions.model`, so two ENGINES under one id would pool into one
population. vLLM accepts either name and advertises both.

## 6. Also settled in passing

* **No GDN Triton autotune to wait for.** `[clav_gdn] CLAV GDN ENABLED -- QwenGatedDeltaNetAttention
  executing on torch.ops.gdn_hip.* (no Triton JIT on the non-spec GDN path)`. Warm boot ~5 min,
  weights 85 s. Use a FRESH Triton cache dir — this image is triton 3.6.0 / torch 2.11.0+rocm10.0 /
  HIP 7.15, not our toolchain, so ours would be invalid to reuse.
* **The checkpoint ships a routing profile.** `model-expertprofile.safetensors`,
  `expert_routing_counts` F32 [48, 512], tagged `expert-routing-profile/1` — a measured expert usage
  ranking. This engine plans its VRAM-only set from it (`PINNED_FRAC` 0.33, hottest first); ours has
  to learn the same thing at runtime, and our expert cache's measured hit rate (h=0.28-0.35) sits far
  below its own oracle (0.864). That sidecar is worth reading, not just skipping.

## 7. Next

1. Pat replays the failing Hermes prompt against this arm. A degeneration that reproduces here is NOT
   in our QSA/PLE implementation; one that does not is evidence it is.
2. minisgl's own MXFP4 loader is committed (acdc30d5, tests 7a783d66) and validated on headers, but
   NOT yet boot-verified — the cards have been held by this arm. That is the next GPU window.
3. Pat wants this engine as the perf and MTP/spec-decode comparison for the OTHER Qwen models. Not
   Flash-Next: spec cannot pay on a host-offloaded MoE (cost scales with query tokens, ~100% accept
   to break even, 1.57x loss measured). The decisive cell is MTP, where we measured stock vLLM at a
   wash (+1%) against our engine's much larger wins.
