#!/usr/bin/env python3
"""Capture driver for ZAYA DFlash drafter RE-distillation, on-policy to minisgl.

Mirrors the vLLM capture (zaya/dflash/capture_drafter_data.py) but points at the minisgl
OpenAI server (default :1919). Two phases per prompt against a minisgl-ZAYA server booted with
MINISGL_ZAYA_CAPTURE_DIR set (scheduler._dump_capture dumps aux for every prefill position) and
--cache-type none (so the re-feed is a full prefill, not a prefix-cache hit):

  1. Greedy rollout (temperature 0) -> the continuation = minisgl-ZAYA's OWN tokens (the labels).
  2. Teacher-forcing re-feed -> POST (prompt+continuation) as max_tokens=1: one full prefill, so the
     capture hook dumps aux for every position aligned with the token ids.

The point: the drafter distills against minisgl-ZAYA's exact aux + tokens, fixing the cross-engine
OOD that pinned the vLLM-trained drafter at 0.26 accept-len (see zaya-cca-dflash-port memory).

Host-side, stdlib only (no torch). seedbuf_*.pt records land in MINISGL_ZAYA_CAPTURE_DIR; then:
  python /home/pat/code/vllm-gfx1201-zaya-dflash/zaya/dflash/train_cca_drafter.py \
      --init /home/pat/code/_models/ZAYA1-8B-DFlash-CCA-5L-init \
      --seed-dir <capture_dir> --out /home/pat/code/_models/ZAYA1-8B-DFlash-CCA-5L-minisgl --epochs 14

  python tools/capture_dflash_data.py --ports 1919 --workers 16 \
      --prompts-file /home/pat/code/_capture_prompts.txt --gen-tokens 256
"""
import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SEED_PROMPTS = [
    "Explain how a hash map works and why lookups are O(1) on average.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Summarize the causes of the French Revolution in a short paragraph.",
    "What is the difference between TCP and UDP? Give one example use of each.",
    "Describe the water cycle step by step.",
    "Q: What are the first ten prime numbers?\nA:",
]

_lock = threading.Lock()


def generate(base, prompt, max_tokens, timeout=300.0):
    """POST minisgl's /generate (RAW text in — no chat template, exactly what teacher-forcing needs) and
    concatenate its SSE delta stream (`data: <incremental text>\\n` ... `data: [DONE]`)."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base, data=body, headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            s = raw.decode("utf-8", "replace")
            if not s.startswith("data: "):
                continue
            payload = s[6:]
            if payload.rstrip("\n") == "[DONE]":
                break
            out.append(payload.rstrip("\n"))
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--ports", default="1919", help="comma-separated minisgl server ports")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default="model")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--prompts-file", default=None)
    args = ap.parse_args()

    ports = [int(p) for p in args.ports.split(",") if p.strip()]
    bases = [f"http://{args.host}:{p}/generate" for p in ports]
    prompts = SEED_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [ln.rstrip("\n") for ln in f if ln.strip()]

    print(f"capture: {len(prompts)} prompts, servers {ports}, {args.workers} workers", flush=True)
    t0 = time.time()
    done = [0]
    errs = [0]

    def process(i, prompt):
        base = bases[i % len(bases)]
        try:
            # 1. greedy rollout -> minisgl-RXF's own continuation (the labels)
            cont = generate(base, prompt, args.gen_tokens)
            full = prompt + cont
            # 2. teacher-forcing re-feed -> one full prefill; the capture hook dumps aux per position
            generate(base, full, 1)
        except Exception as e:  # noqa: BLE001
            with _lock:
                errs[0] += 1
                print(f"  ERR [{i}] {type(e).__name__}: {e}", flush=True)
            return
        with _lock:
            done[0] += 1
            if done[0] % 20 == 0 or done[0] == len(prompts):
                print(f"  {done[0]}/{len(prompts)} captured "
                      f"({time.time() - t0:.0f}s, {errs[0]} errs)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, i, p) for i, p in enumerate(prompts)]
        for _ in as_completed(futs):
            pass

    print(f"done: {done[0]} captured, {errs[0]} errs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
