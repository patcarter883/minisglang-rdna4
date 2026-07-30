"""LM-head + sampling isolation microbench (decode M=1, Qwen3.6-35B geometry).

Measurement (1) of the 35B decode-latency localization. Times, in the serve dtypes, the pieces that
run AFTER the backbone in a decode step:
  (a) LM-head GEMV [1,2048] x [2048,248320] via the real minv_linear (dense_gemm) kernel — full vocab
      AND the TP=2 per-rank shard (vocab/2 = 124160); plus torch F.linear for reference.
  (b) softmax over 248320 (fp32) — the classic full-vocab reduction.
  (c) greedy argmax over 248320 (fp32) — the Sampler greedy branch.
  (d) the fused sampler_hip path (temperature+softmax+top-k+top-p+multinomial), if built.
Each reported in microseconds. Compare against the ~5-24 ms/token decode budget (bs=1 ~42 tok/s =>
~24 ms; conc=6 ~167 tok/s => ~6 ms/token-slot). Single card: gpu-lease -n 1.

Run inside the serve image:
  PYTHONPATH=/engine/python:/engine python /engine/tools/lmhead_sampling_microbench.py
"""
from __future__ import annotations

import torch

HIDDEN = 2048
VOCAB = 248320
TP = 2


def perf_us(f, iters=100, warmup=20) -> float:
    """Eager wall time per call in microseconds (event-timed, no graph wrap so alloc-heavy paths
    like the fused sampler are measured honestly)."""
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        f()
    torch.cuda.synchronize()
    tic.record()
    for _ in range(iters):
        f()
    toc.record()
    toc.synchronize()
    return tic.elapsed_time(toc) / iters * 1e3  # ms/iter -> us


def main() -> int:
    dev = torch.device("cuda")
    print(f"torch={torch.__version__} hip={torch.cuda.is_available()} dev={torch.cuda.get_device_name()}",
          flush=True)

    from minisgl.distributed.info import set_tp_info, try_get_tp_info
    if try_get_tp_info() is None:
        set_tp_info(0, 1)  # single-rank stub so the hip-engage logger works outside the engine

    from minisgl.layers.minv import minv_linear, minv_supported
    from minisgl.engine import _sampler_hip

    budget_bs1_us = 1e6 / 42.0      # ~23810 us/token at 42 tok/s
    budget_conc6_us = 1e6 / 167.0   # ~5988 us/token-slot at conc=6

    def frac(us):
        return f"{us/budget_bs1_us*100:5.1f}% of bs=1 budget | {us/budget_conc6_us*100:5.1f}% of conc6 budget"

    results = {}

    # ---- (a) LM-head GEMV ----------------------------------------------------------------
    for dt, name in [(torch.bfloat16, "bf16"), (torch.float16, "fp16")]:
        x = torch.randn(1, HIDDEN, device=dev, dtype=dt)
        for vocab, tag in [(VOCAB, "full-vocab"), (VOCAB // TP, "TP2-shard")]:
            w = torch.randn(vocab, HIDDEN, device=dev, dtype=dt)
            sup = minv_supported(x, w)
            us_minv = perf_us(lambda: minv_linear(x, w))
            us_torch = perf_us(lambda: torch.nn.functional.linear(x, w))
            key = f"lmhead_{name}_{tag}"
            results[key] = us_minv
            print(f"[a] LMhead {name:4} {tag:10} [1,{HIDDEN}]x[{vocab},{HIDDEN}]  "
                  f"minv={us_minv:8.1f}us (supported={sup})  Flinear={us_torch:8.1f}us  | {frac(us_minv)}",
                  flush=True)

    # logits are produced fp32-ish; sampler consumes fp32 [1, VOCAB]. Build one for b/c/d.
    logits = torch.randn(1, VOCAB, device=dev, dtype=torch.float32)

    # ---- (b) softmax over full vocab ----------------------------------------------------
    us = perf_us(lambda: torch.softmax(logits, dim=-1))
    results["softmax"] = us
    print(f"[b] softmax    [1,{VOCAB}] fp32                    {us:8.1f}us  | {frac(us)}", flush=True)

    # ---- (c) greedy argmax (the Sampler greedy branch) ----------------------------------
    us = perf_us(lambda: torch.argmax(logits, dim=-1))
    results["argmax_greedy"] = us
    print(f"[c] argmax     [1,{VOCAB}] fp32 (greedy sample)    {us:8.1f}us  | {frac(us)}", flush=True)

    # ---- (d) fused sampler_hip path -----------------------------------------------------
    print(f"[d] sampler_hip available={_sampler_hip.available()}", flush=True)
    temps = torch.full((1,), 0.7, device=dev, dtype=torch.float32)
    top_k = torch.full((1,), 20, device=dev, dtype=torch.int32)
    top_p = torch.full((1,), 0.8, device=dev, dtype=torch.float32)
    if _sampler_hip.available():
        us = perf_us(lambda: _sampler_hip.sample(logits, temps, top_k, top_p))
        results["sampler_hip_fused"] = us
        print(f"[d] sampler_hip fused (T+softmax+topk+topp+multinomial)  {us:8.1f}us  | {frac(us)}",
              flush=True)
    # torch reference sampler path (sort-based top-k/top-p) for contrast
    from minisgl.engine.sample import sample_impl
    import os
    os.environ["MINISGL_FUSED_SAMPLER"] = "0"  # not honored post-import; time torch path directly below
    def torch_sample():
        probs = torch.softmax(logits / temps.unsqueeze(-1).clamp_min(1e-6), dim=-1)
        # top-k + top-p via sort (the two full-vocab sorts the fused kernel replaces)
        sp, si = torch.sort(probs, dim=-1, descending=True)
        return torch.multinomial(probs, num_samples=1)
    us = perf_us(torch_sample)
    results["sampler_torch_sort"] = us
    print(f"[d'] torch sort-based sampler (softmax+sort+multinomial) {us:8.1f}us  | {frac(us)}", flush=True)

    # ---- summary -------------------------------------------------------------------------
    print("\n==== SUMMARY (decode-tail cost, M=1) ====", flush=True)
    head = results.get("lmhead_bf16_TP2-shard", results.get("lmhead_bf16_full-vocab", 0.0))
    greedy = results["argmax_greedy"]
    fused = results.get("sampler_hip_fused", results["sampler_torch_sort"])
    print(f"LM-head (TP2 per-rank shard, bf16):  {head:8.1f}us  {frac(head)}", flush=True)
    print(f"greedy argmax:                       {greedy:8.1f}us  {frac(greedy)}", flush=True)
    print(f"fused sampler (T=0.7,k20,p0.8):      {fused:8.1f}us  {frac(fused)}", flush=True)
    print(f"head+greedy tail:                    {head+greedy:8.1f}us  {frac(head+greedy)}", flush=True)
    print(f"head+fused-sample tail:              {head+fused:8.1f}us  {frac(head+fused)}", flush=True)
    print("NOTE: penalties (rep/presence/freq) are NOT applied in minisgl sample path -> 0us.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
