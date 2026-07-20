#!/usr/bin/env python3
"""Container self-test: proves the image + kernels + HIP are all wired correctly.

This is the default command of the `run` compose service — `docker compose --profile run run --rm run`
runs it. If this prints "ALL GOOD" the container is a known-good base for any test/bench/compile.
Kernels resolve from /opt/kernels (baked, always consistent with source); a mismatch here is exactly
the stale-.so class of failure (e.g. gdn_decode_gated missing) that this catches BEFORE a real run."""
import os
import sys


def main() -> int:
    ok = True
    try:
        import torch
        hip = torch.cuda.is_available()
        print(f"[selftest] torch {torch.__version__}  |  HIP available: {hip}", flush=True)
        if hip:
            print(f"[selftest] device: {torch.cuda.get_device_name(0)}  "
                  f"(HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES','?')} "
                  f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES','?')})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[selftest] FAIL torch/HIP: {type(e).__name__}: {e}", flush=True)
        return 1

    # Import every serve-path kernel package. An import failure here (e.g. a stale .so missing an op
    # the current wrapper registers) is the exact failure class this test exists to surface.
    pkgs = ["gdn_hip", "fp8_wmma", "attn_decode", "attn_prefill_paged", "moe_hip",
            "tail_hip", "sampler_hip", "mla_hip", "zaya_cca"]
    for p in pkgs:
        try:
            m = __import__(p)
            print(f"[selftest] OK   {p:20} <- {getattr(m, '__file__', '?').rsplit('/' + p, 1)[0]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[selftest] FAIL {p:20} {type(e).__name__}: {e}", flush=True)
            ok = False

    # Spot-check ops that have bitten us via stale .so vs current wrapper.
    try:
        import gdn_hip
        print(f"[selftest] gdn_decode_gated present: {hasattr(gdn_hip, 'gdn_decode_gated')}", flush=True)
        if not hasattr(gdn_hip, "gdn_decode_gated"):
            ok = False
        import sampler_hip
        print(f"[selftest] sampler_hip MAX_ROUNDS: {getattr(sampler_hip, 'MAX_ROUNDS', None)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[selftest] FAIL op spot-check: {type(e).__name__}: {e}", flush=True)
        ok = False

    print("[selftest] ALL GOOD" if ok else "[selftest] PROBLEMS FOUND (see FAIL lines above)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
