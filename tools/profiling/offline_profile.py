#!/usr/bin/env python3
"""Reusable OFFLINE decode profiler for vLLM (TP-aware) — runs INSIDE the ROCm container.

Boots the model offline (spawn-safe under ``if __name__ == '__main__'``), warms up, then profiles
exactly ONE decode burst via vLLM's official ``llm.start_profile()`` / ``llm.stop_profile()``. That
path propagates the profile RPC to EVERY TP worker, so each rank writes its own TraceLens-native
``*.pt.trace.json.gz`` to ``$VLLM_TORCH_PROFILER_DIR`` — which is why this works for TP>=2 where
rocprofv3-wrapping (traces the launcher, not the spawned workers) produced empty traces.

Everything is env-driven so the same script profiles any model/shape (see run_profile.sh):
  PROF_MODEL (req), PROF_TP=2, PROF_MAXLEN=24000, PROF_KV=fp8, PROF_DTYPE=float16,
  PROF_MNBT=2048, PROF_MNS=8, PROF_MEM=0.92, PROF_EAGER=0, PROF_PREFILL=300, PROF_DECODE=30.
Only the marked decode burst is inside the profiler window (warmup runs before start_profile).
"""
import os, time


def _env(k, d): return os.environ.get(k, d)


def main():
    from vllm import LLM, SamplingParams
    prof_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR")
    assert prof_dir, "VLLM_TORCH_PROFILER_DIR must be set"
    model = os.environ["PROF_MODEL"]
    # vLLM 0.24 replaced the VLLM_TORCH_PROFILER_DIR env trigger with a structured profiler_config
    # (dict accepted by offline LLM). start_profile()/stop_profile() propagate to ALL TP workers.
    llm = LLM(
        model=model,
        profiler_config={"profiler": "torch", "torch_profiler_dir": prof_dir,
                         "torch_profiler_use_gzip": True,
                         # record_shapes + with_stack pin an op's tensor sizes and Python origin;
                         # combine with PROF_EAGER=1 so cudagraph capture doesn't hide the op->kernel
                         # linkage (needed to identify the dominant elementwise kernel).
                         "torch_profiler_record_shapes": _env("PROF_SHAPES", "0") == "1",
                         "torch_profiler_with_stack": _env("PROF_STACK", "0") == "1"},
        # text-only decode tests: pin multimodal to 0 so the vision tower's encoder-cache budget
        # isn't reserved (it OOMs KV on the 16 GB cards). PROF_MM=1 to re-enable.
        limit_mm_per_prompt=({} if _env("PROF_MM", "0") == "1" else {"image": 0, "video": 0}),
        tensor_parallel_size=int(_env("PROF_TP", "2")),
        max_model_len=int(_env("PROF_MAXLEN", "24000")),
        kv_cache_dtype=_env("PROF_KV", "fp8"),
        dtype=_env("PROF_DTYPE", "float16"),
        max_num_batched_tokens=int(_env("PROF_MNBT", "2048")),
        max_num_seqs=int(_env("PROF_MNS", "8")),
        gpu_memory_utilization=float(_env("PROF_MEM", "0.92")),
        enforce_eager=_env("PROF_EAGER", "0") == "1",
        trust_remote_code=True,
    )
    pre = int(_env("PROF_PREFILL", "300"))
    dec = int(_env("PROF_DECODE", "30"))
    # PROF_PROMPT=real => a coherent long prompt (checks the state path is bit-correct, not just fast);
    # else the fixed-length filler prompt (stable profiling shape).
    if _env("PROF_PROMPT", "") == "real":
        prompt = ("The following is a detailed technical explanation of how modern distributed "
                  "database systems achieve consistency and fault tolerance. " * max(1, pre // 12))
    else:
        prompt = "Write a detailed technical essay on distributed systems. " + ("filler token " * pre)

    llm.generate([prompt], SamplingParams(max_tokens=20, temperature=0.0))  # warmup (NOT profiled)
    print("WARMUP_DONE", flush=True)

    # UN-profiled timed pass = the REAL decode tok/s (profiler overhead inflates ~2.4x) + coherence.
    t = time.time()
    o0 = llm.generate([prompt], SamplingParams(max_tokens=dec, temperature=0.0))
    dt0 = time.time() - t
    real_txt = o0[0].outputs[0].text if o0 and o0[0].outputs else "<none>"
    print(f"REAL_TOK_S tokens={dec} wall={dt0:.3f}s tok_s={dec/dt0:.1f}", flush=True)
    print("REAL_COHERENCE_TEXT:", repr(real_txt[:220]), flush=True)

    if _env("PROF_TRACE", "1") != "1":
        print("PROFILE_FLUSHED", flush=True); return
    llm.start_profile()
    t = time.time()
    outs = llm.generate([prompt], SamplingParams(max_tokens=dec, temperature=0.0))  # THE profiled burst
    dt = time.time() - t
    llm.stop_profile()
    # coherence check — a broken state path yields garbage while the profiler still 'succeeds'
    txt = outs[0].outputs[0].text if outs and outs[0].outputs else "<none>"
    print("COHERENCE_TEXT:", repr(txt[:200]), flush=True)
    # profiling adds overhead; tok/s here is not the real serve number — the trace breakdown is.
    print(f"DECODE_BURST_DONE tokens={dec} wall={dt:.3f}s prof_tok_s={dec/dt:.1f}", flush=True)
    time.sleep(5)  # let each rank flush its trace file
    print("PROFILE_FLUSHED", flush=True)


if __name__ == "__main__":
    main()
