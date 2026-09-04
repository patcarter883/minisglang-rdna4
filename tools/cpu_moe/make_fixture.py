#!/usr/bin/env python3
"""Build the CPU-MoE fixture from the REAL Qwen3.8-Flash-Next-NVFP4 checkpoint.

Emits three things into --out:

  plan.txt   -- a byte-offset plan the C++ harness uses to mmap the safetensors shards
                and gather each expert's (gate, up, down) codes+scales into one packed,
                64B-aligned resident slab. No large temp file is written.
  ref/*.bin  -- a SMALL correctness fixture: the real x vector, a real top-k routing,
                and the float64 dequant-and-GEMM reference output for that routing.
  meta.json  -- shapes / sizes so the harness never guesses.

Format (compressed-tensors `nvfp4-pack-quantized`), pinned against
python/minisgl/quant/nvfp4.py + mxfp4.py:

  weight        uint8   (N, K/2)    two E2M1 nibbles per byte, LOW nibble = lower K index
  weight_scale  e4m3    (N, K/16)   per-16 block scale
  weight_scale_2 f32    scalar      per-tensor global, a MULTIPLIER = amax / (FP4_MAX*FP8_MAX)

  W[n,k] = E2M1_LUT[code(n,k)] * e4m3(weight_scale[n, k//16]) * weight_scale_2

PINNED EMPIRICALLY, not assumed.  minisgl's quant/nvfp4.py documents the OTHER (llm-compressor)
convention -- `weight_global_scale = 448*6/amax`, a DIVISOR -- and its loader keys on the name
`.weight_global_scale`.  This checkpoint spells it `.weight_scale_2` and stores the RECIPROCAL.
Evidence that settles it:
  * dividing gives |W| ~ 4.2e6 (absurd); multiplying gives rms 0.0135 (a textbook weight matrix)
  * down_proj's largest block scale byte is 126 (= e4m3 448 = FP8_E4M3_MAX) for EVERY expert,
    which only the two-level formula scale = block_amax/FP4_MAX/weight_scale_2 can produce
  * gate_proj and up_proj share ONE weight_scale_2 across all 512 experts (they were quantized
    as one stacked/fused tensor), so their per-expert block maxima sit below 448 -- consistent
  * input_scale * 2688 = 5.3, a plausible activation amax; its reciprocal would be 1.4e6
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import numpy as np

E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float64,
)

PROJ = ("gate_proj", "up_proj", "down_proj")


def e4m3_table() -> np.ndarray:
    """256-entry float64 table for float8_e4m3fn (bias 7, no inf, 0x7F/0xFF = NaN)."""
    t = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * (2.0 ** -6)
        elif e == 15 and m == 7:
            v = np.nan
        else:
            v = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        t[b] = s * v
    return t


E4M3 = e4m3_table()


def read_header(path: str):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def dequant(codes_u8: np.ndarray, scale_u8: np.ndarray, gscale: float, n: int, k: int) -> np.ndarray:
    """RAW checkpoint bytes -> (N, K) float64.  The golden reference. gscale is a MULTIPLIER."""
    b = codes_u8.reshape(n, k // 2)
    codes = np.empty((n, k), dtype=np.uint8)
    codes[:, 0::2] = b & 0x0F
    codes[:, 1::2] = (b >> 4) & 0x0F
    w = E2M1_LUT[codes]
    s = E4M3[scale_u8.reshape(n, k // 16)]
    return w * np.repeat(s, 16, axis=1) * gscale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--experts", type=int, default=512, help="how many experts to make resident")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "ref"), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    shards = sorted(
        os.path.join(args.model, f)
        for f in os.listdir(args.model)
        if f.startswith(f"layer-{args.layer:05d}-experts-")
    )
    assert shards, f"no expert shards for layer {args.layer} in {args.model}"

    # ---- locate every (expert, proj) tensor -------------------------------------------------
    loc = {}  # (expert, proj) -> dict
    for si, sp in enumerate(shards):
        hdr, base = read_header(sp)
        for name, rec in hdr.items():
            if name == "__metadata__":
                continue
            parts = name.split(".")
            # ...layers.<L>.mlp.experts.<E>.<proj>.<field>
            try:
                ei = parts.index("experts")
            except ValueError:
                continue
            e = int(parts[ei + 1])
            proj = parts[ei + 2]
            field = parts[ei + 3]
            if proj not in PROJ:
                continue
            d = loc.setdefault((e, proj), {"shard": si})
            o0, o1 = rec["data_offsets"]
            d[field] = (base + o0, o1 - o0, rec["shape"], rec["dtype"])

    n_experts = args.experts
    for e in range(n_experts):
        for p in PROJ:
            d = loc[(e, p)]
            assert "weight" in d and "weight_scale" in d and "weight_scale_2" in d, (e, p, d)

    # gather the per-tensor global scales (tiny f32 reads)
    mm = [np.memmap(sp, dtype=np.uint8, mode="r") for sp in shards]

    def rd(d, field):
        off, ln, shape, dt = d[field]
        return mm[d["shard"]][off : off + ln], shape, dt

    plan_lines = [f"SHARDS {len(shards)}"]
    plan_lines += list(shards)
    plan_lines.append(f"EXPERTS {n_experts}")
    for e in range(n_experts):
        for pi, p in enumerate(PROJ):
            d = loc[(e, p)]
            wo, wl, wshape, _ = d["weight"]
            so, sl, sshape, _ = d["weight_scale"]
            g2, _, _, _ = d["weight_scale_2"]
            g = float(np.frombuffer(mm[d["shard"]][g2 : g2 + 4].tobytes(), dtype="<f4")[0])
            plan_lines.append(
                f"E {e} {pi} {d['shard']} {wo} {wl} {so} {sl} {wshape[0]} {wshape[1] * 2} {g!r}"
            )
    with open(os.path.join(args.out, "plan.txt"), "w") as f:
        f.write("\n".join(plan_lines) + "\n")

    # ---- correctness fixture ----------------------------------------------------------------
    hidden = 2560
    inter = 640
    sel = np.sort(rng.choice(n_experts, size=args.topk, replace=False)).astype(np.int32)
    # Activation magnitudes in the ballpark of a real residual stream.
    x = (rng.standard_normal(hidden) * 0.5).astype(np.float32)
    rw = rng.random(args.topk).astype(np.float32)
    rw = (rw / rw.sum()).astype(np.float32)

    xd = x.astype(np.float64)
    y = np.zeros(hidden, dtype=np.float64)
    for i, e in enumerate(sel):
        e = int(e)
        dg = loc[(e, "gate_proj")]
        du = loc[(e, "up_proj")]
        dd = loc[(e, "down_proj")]

        def W(d, n, k):
            wo, wl, _, _ = d["weight"]
            so, sl, _, _ = d["weight_scale"]
            g2, _, _, _ = d["weight_scale_2"]
            gs = float(np.frombuffer(mm[d["shard"]][g2 : g2 + 4].tobytes(), dtype="<f4")[0])
            return dequant(
                np.asarray(mm[d["shard"]][wo : wo + wl]),
                np.asarray(mm[d["shard"]][so : so + sl]),
                gs,
                n,
                k,
            )

        gp = W(dg, inter, hidden) @ xd
        up = W(du, inter, hidden) @ xd
        h = (gp / (1.0 + np.exp(-gp))) * up
        y += float(rw[i]) * (W(dd, hidden, inter) @ h)

    ref = os.path.join(args.out, "ref")
    x.tofile(os.path.join(ref, "x.bin"))
    sel.tofile(os.path.join(ref, "sel.bin"))
    rw.tofile(os.path.join(ref, "rw.bin"))
    y.astype(np.float64).tofile(os.path.join(ref, "y_f64.bin"))

    meta = {
        "layer": args.layer,
        "n_experts": n_experts,
        "topk": args.topk,
        "hidden": hidden,
        "moe_intermediate": inter,
        "shards": shards,
        "bytes_per_expert_e4m3": 3 * (640 * 1280 + 640 * 160),  # gate/up; down is the same size
        "ref_rms": float(np.sqrt(np.mean(y**2))),
        "ref_absmax": float(np.max(np.abs(y))),
    }
    # exact: gate 640*1280 + 640*160, up same, down 2560*320 + 2560*40
    meta["bytes_per_expert_e4m3"] = (
        640 * 1280 + 640 * 160 + 640 * 1280 + 640 * 160 + 2560 * 320 + 2560 * 40
    )
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
