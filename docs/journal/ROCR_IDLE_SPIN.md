# The ROCr idle busy-wait: upstream status and the options

Researched 2026-09-16. Local characterisation is in `tools/serve.sh` (search "THE IDLE BUSY-WAIT");
this file is the *external* picture — whether anyone else has this, and what can be done.

## It is a known ROCm bug, not something about this box

The symptom is reported repeatedly and independently: one ROCr thread pinned at 100% of a core for
the life of any process that has touched the GPU, with the GPU itself idle.

* [ROCm/ROCm#6522](https://github.com/rocm/rocm/issues/6522) — "HSA runtime AsyncEventsLoop livelocks
  (100% CPU spin) … GPU itself confirmed idle, no hang/reset".
* [ROCm/TheRock#7051](https://github.com/ROCm/TheRock/issues/7051) — bundled ROCR 1.21 busy-spins a
  full core permanently after *any* GPU op.
* [ROCm/TheRock#8213](https://github.com/ROCm/TheRock/issues/8213) /
  [Comfy-Org/ComfyUI#16339](https://github.com/Comfy-Org/ComfyUI/issues/16339) — same, via ComfyUI,
  with "high idle temperature" called out exactly as the operator did here.

Our own measurements match the upstream description precisely: it starts at the first kernel, the
thread is pure user time with no syscall, and the GPU is doing nothing.

## Root cause and the upstream fix

`Runtime::AsyncEventsLoop` falls back to **polling** when interrupt-backed signal events are not
available, and that fallback has no backoff — it spins.

[ROCm/rocm-systems#7898](https://github.com/ROCm/rocm-systems/pull/7898), **merged 2026-08-18**, adds
an escalating nap to that fallback: 20 µs doubling to a ceiling (200 µs on interrupt-capable systems,
2 ms when globally polling), reset whenever events are processed. New `core/util/poll_backoff.h`,
integrated in `core/runtime/runtime.cpp` (~line 2117), with unit tests. The PR reports **182 % idle
CPU → 1.07 %**. A related follow-up is
[#10463](https://github.com/ROCm/rocm-systems/pull/10463) (notify the wake signal on async-events
wakeup paths).

**It is not in our stack.** We run ROCm 7.2.1 / ROCR 1.18.70201; the
[7.2.4 release notes](https://rocm.docs.amd.com/en/docs-7.2.4/about/release-notes.html) still ship
ROCR 1.18.0 and carry no such entry. The fix went to `develop` after the 7.2.x line.

## Precedent on OUR GPU

[FA85/r9700-vllm-rocm72-fixes](https://github.com/FA85/r9700-vllm-rocm72-fixes) targets the Radeon AI
PRO **R9700 — gfx1201, the same architecture as this box** — on ROCm 7.2.3 / libhsa 1.18.0 / vLLM
0.29.0, and carries two ROCr patches:

1. **AsyncEventsLoop backoff** — a straight backport of #7898. Upstream-blessed.
2. **Null-event backoff** — a *local* workaround for "a blocked `InterruptSignal` without a KFD
   event", replacing repeated `hsaKmtWaitOnEvent_Ext(nullptr, …)` calls with a 20–200 µs exponential
   sleep. **This one cost 20–29 % of decode throughput (55–58 → 41–44 tok/s).**

Both are *source* patches requiring a ROCr rebuild, not binary patches.

## Options, in order of blast radius

1. **Backport #7898 into ROCR 1.18, rebuild `libhsa-runtime64.so`, substitute it into
   `torch/lib/`.** ~20 lines against a known anchor, and the substitution half is already proven
   safe here: swapping the system build in (one runtime loaded, verified in `/proc/<pid>/maps`) left
   prefill at 183.12 tok/s vs 182.21. Same move FA85 made on the same GPU.
   **Caveat, and it is the important one:** #7898 only adds backoff to the *polling fallback*. If our
   spin is actually the **null-event** path, #7898 alone will not fix it, and the patch that does
   costs 20–29 % of decode. Which path we are on is not yet established — that is the thing to
   determine before building anything.
2. **Upgrade the stack.** [#8213](https://github.com/ROCm/TheRock/issues/8213) reports the spin
   resolved on a ROCm 10.1 nightly (build 20260822, i.e. after #7898 merged) with matching torch —
   "spending most of its time sleeping". That is a ROCm 7.2 → 10.1 jump plus a torch rebuild and full
   revalidation of this serve.
3. **Do nothing.** ~10 W and two cores.

## What NOT to do

`LD_PRELOAD` of the system runtime is the workaround suggested in several of those threads. **It is a
trap on this stack:** it does not replace torch's runtime (both end up mapped — torch's libamdhip64
pulls its bundled copy in by RPATH) and it cost **13–51x on prefill** here. See `tools/serve.sh`.
