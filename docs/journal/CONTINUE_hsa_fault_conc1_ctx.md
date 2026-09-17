# OPEN BUG — HSA 0x1016 queue abort at `CONC=1` + `CTX=8192` (qwen35b-awq + DFlash)

**Status: characterised, NOT root-caused. Parked 2026-08-14.**
**Not in the production path** — see "Scope" before deciding priority.

## Symptom

Mid-generation, the scheduler worker dies and the server shuts down to release the lease:

```
:0:rocdevice.cpp :3586: Callback: Queue 0x… aborting with error :
HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
worker 'minisgl-DP0-TP0-scheduler' died unexpectedly (exitcode=-6)
```

An in-kernel GPU memory fault, reported asynchronously. `dmesg` was not checked this time; the
2026-07-05 instance of the same code (see "Prior art") had a clean dmesg, i.e. a userspace-caught
hardware exception rather than a driver/ring reset.

Free VRAM at the time of death was healthy (1.82 GiB) — **this is not the OOM** that also occurs in
this area (that one is a separate, understood issue: the propose ring competing with a long
generation's workspace).

## Reproduction

Unmodified `HEAD` = `4b6ff698`, worktree `/home/pat/code/minisgl-rdna4-dfbase`.

```
REPO=/home/pat/code/minisgl-rdna4-dfbase LEG=<label> PAIRS=qwen35b-awq CONC=1 \
  PAIR_ENV_EXTRA="CONC=1 CTX=8192" tools/dflash_matrix_ab.sh
```

Pair: `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` + `z-lab/Qwen3.6-35B-A3B-DFlash`, TP=2,
`SPEC=dflash`, `MEM_RATIO=0.86`, `MINISGL_DFLASH_QUANT=fp8`. Workload: the harness's long
mathematical/code probe (3 prompts, `max_tokens=12000`, thinking on). It appears **thousands of
decode steps in**, not at boot — logged step counts at death ranged ~1500–4000.

## The matrix — BOTH knobs are required

| leg | CONC | CTX | serialize | runs | faults | fixture |
|---|---|---|---|---|---|---|
| `base-c1`, `rate-base-1/2/3` | **1** | **8192** | no | 4 | **3** | `base-c1_20260813T132322Z*`, `rate-base-{1,2,3}_*` |
| `iso-c4ctx` | 4 | **8192** | no | 1 | 0 | `iso-c4ctx_20260813T234904Z*` |
| `iso-c1noctx` | **1** | checkpoint | no | 1 | 0 | `iso-c1noctx_20260813T235307Z*` |
| `fault-ser2` | **1** | **8192** | `AMD_SERIALIZE_KERNEL=3` | 1 | 0 | `fault-ser2_20260813T233103Z*` |

Fixtures (durable): `/home/pat/fixtures/minisgl-dflash-matrix/`.

Three things follow:

1. **It is an interaction.** Neither `CONC=1` nor `CTX=8192` reproduces it alone. Only together.
2. **It is intermittent — 3 of 4, not 4 of 4.** `rate-base-3` completed clean at 118.68 tok/s on the
   identical config. Any future "fix" must therefore be validated against a RATE over repeats; a
   single clean run proves nothing. This is the same discipline `serve-acceptance-code-check-is-flaky`
   already demands for accept-len.
3. **Kernel serialization suppresses it.** `AMD_SERIALIZE_KERNEL=3` ran clean. That points at a
   timing/concurrency fault — a host-write vs kernel-read race, stream ordering, or a
   use-after-free — and **away from** a deterministic out-of-bounds index.

## Prior art — and why it is NOT the same bug

`memory/reboot-continuance-mes-firmware.md` records an earlier HSA 0x1016 on this box
(2026-07-05). That one was root-caused to a **CAM data bug**: `cam/realedit.py` set
`new_tid = -1` for multi-token objects and the −1 reached an embedding lookup. Kernels were
exonerated.

The distinguishing evidence: that instance **"Reproduced identically under `AMD_SERIALIZE_KERNEL=3`
⇒ not transient."** This one is *suppressed* by serialization. So the "bad token id into an
embedding" mechanism does not transfer, and the method that cracked it (serialize, then instrument
ops with sync+markers to catch the faulting op) will not work here unmodified — serializing makes
the bug disappear.

Also checked and **ruled out** as a static cause: drafter vocab 248320 == target vocab 248320; no
`draft_vocab_size`, so no d2t remap; `mask_token_id=248077` in range. Draft ids come from an argmax
over the borrowed target head and are in range by construction.

## Scope — read before prioritising

**Production is not affected as configured.** The serve runs `CONC=4` with the checkpoint's own
context; `iso-c4ctx` shows `CONC=4` + `CTX=8192` is clean, and the long-generation probe at `CONC=4`
(`mlv2-base`, 109.88 tok/s, 15803 tokens) is clean. The faulting corner was constructed for a
capture experiment. An earlier note in this investigation claimed "it reproduces on unmodified HEAD,
so it is in what you are serving now" — that overstated it and is withdrawn.

## Consequence for the DFlash propose-capture work

Measurements taken at `CONC=1 / CTX=8192` are **inside an independently-faulting config** and must
not be used as evidence. Specifically INVALIDATED:

* the propose-ring capacity bisect (4112 clean / 6160 clean / **8208 faulting**) — the faults there
  are not attributable to ring capacity, because baseline faults 3/4 in the same config with no ring
  in the code at all;
* the conclusion that chunking `_rebuild_ring` "fixed the fault". The two rebuild defects fixed
  there are real on their own merits (a duplicate-column scatter that could desynchronise `_pk` from
  `_ppos`, and an unbounded-M projection on the cold path) and should be kept — but the fault-fix
  claim is unproven.

Redo the DFlash A/B at **`CONC=4`**, which is both production-like and clean, with repeats.

## Next steps when we cycle back

1. Bisect **what `CONC=1` actually changes** against `max_seq_len=8192`: the CUDA-graph bucket list
   collapses to `[1]`, `max_running_req=1`, the page table drops to 2 rows, GDN state to 1 slot.
   Vary each independently at `CTX=8192` (e.g. `CONC=1` with `--cuda-graph-max-bs 2`) to find which
   one carries it.
2. Get the rate up first if possible — a 75% reproducer is workable, a 100% one is far better. Try
   longer generations / more prompts to see if the rate is a function of step count.
3. `dmesg` during a faulting run, to confirm (as in 2026-07-05) that there is no ring reset and this
   is a userspace-caught exception.
4. Since serialization suppresses it, the localisation lever is NOT `AMD_SERIALIZE_KERNEL`. Try
   `--graph 0` (fully eager) at the faulting config: if it survives, the fault is inside a captured
   replay and the search narrows to the graph pools; if it still faults, graphs are exonerated.
