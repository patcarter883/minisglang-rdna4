#!/usr/bin/env python3
"""How much does the ACTIVATION format cost, on the real checkpoint, at full-layer scale?

The int8-activation CPU kernel measures 8.3e-03 rel_rms against make_fixture's float64
reference, ~35,000x the fp32-activation kernel's 2.4e-07.  Read alone that looks
disqualifying.  It is not interpretable alone, because the GPU path this would offload FROM
does not use fp32 activations either: quant/method.py:440 serves the MoE experts through the
W4A8 fp8-WMMA kernel with PER-TOKEN FP8 E4M3 activations.

So this script holds the WEIGHTS exact (float64 dequant of the real checkpoint bytes, same code
as make_fixture.py) and varies ONLY the activation format, for the same layer, the same routing
and the same x.  Both activation sites are quantized -- the layer input x AND the h that feeds
down_proj -- exactly as each kernel does.

  exact            fp32 activations                 -- what make_fixture's reference uses
  int8_g16         symmetric int8, group of 16      -- the CPU VNNI kernel in moe_core.hpp
  int8_token       symmetric int8, one scale/vector -- a cheaper CPU variant
  fp8e4m3_token    fp8 e4m3, one scale/vector       -- WHAT THE GPU ALREADY SERVES
  fp8e4m3_g16      fp8 e4m3, group of 16            -- for reference

Run:  python3 act_format_error.py --fixture fixture/L0 [--layer 0]
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from make_fixture import E2M1_LUT, E4M3, PROJ, dequant, read_header


# ---------------------------------------------------------------------------- activation formats
def q_exact(v):
    return v.astype(np.float64)


def _int8(v, group):
    v = v.astype(np.float64).reshape(-1, group) if group else v.astype(np.float64).reshape(1, -1)
    amax = np.abs(v).max(axis=1, keepdims=True)
    s = np.where(amax > 0, amax / 127.0, 0.0)
    inv = np.where(amax > 0, 127.0 / np.maximum(amax, 1e-300), 0.0)
    q = np.clip(np.rint(v * inv), -127, 127)
    return (q * s).reshape(-1)


def _fp8e4m3(v, group):
    """Round to the nearest representable float8_e4m3fn, after a per-group amax scale.

    E4M3 tops out at 448, so the scale is amax/448 -- the same convention the served path uses
    (a per-tensor/per-token input_scale).  Rounding is done by snapping to the exact 256-entry
    value set, which is what the hardware convert does, not by a mantissa-truncation shortcut.
    """
    v = v.astype(np.float64).reshape(-1, group) if group else v.astype(np.float64).reshape(1, -1)
    amax = np.abs(v).max(axis=1, keepdims=True)
    s = np.where(amax > 0, amax / 448.0, 1.0)
    x = v / np.where(s > 0, s, 1.0)
    grid = np.unique(E4M3[np.isfinite(E4M3)])
    idx = np.abs(x.reshape(-1, 1) - grid.reshape(1, -1)).argmin(axis=1)
    return (grid[idx].reshape(x.shape) * s).reshape(-1)


FORMATS = {
    "exact": q_exact,
    "int8_g16": lambda v: _int8(v, 16),
    "int8_token": lambda v: _int8(v, 0),
    "fp8e4m3_token": lambda v: _fp8e4m3(v, 0),
    "fp8e4m3_g16": lambda v: _fp8e4m3(v, 16),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/pat/.cache/hf-q4e")
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.fixture, "meta.json")))
    layer = meta.get("layer", args.layer)
    hidden, inter = meta["hidden"], meta["moe_intermediate"]

    ref = os.path.join(args.fixture, "ref")
    x = np.fromfile(os.path.join(ref, "x.bin"), dtype=np.float32)
    sel = np.fromfile(os.path.join(ref, "sel.bin"), dtype=np.int32)
    rw = np.fromfile(os.path.join(ref, "rw.bin"), dtype=np.float32)
    y_ref = np.fromfile(os.path.join(ref, "y_f64.bin"), dtype=np.float64)

    shards = sorted(
        os.path.join(args.model, f)
        for f in os.listdir(args.model)
        if f.startswith(f"layer-{layer:05d}-experts-")
    )
    loc = {}
    for si, sp in enumerate(shards):
        hdr, base = read_header(sp)
        for name, rec in hdr.items():
            if name == "__metadata__":
                continue
            parts = name.split(".")
            try:
                ei = parts.index("experts")
            except ValueError:
                continue
            e, proj, field = int(parts[ei + 1]), parts[ei + 2], parts[ei + 3]
            if proj not in PROJ:
                continue
            d = loc.setdefault((e, proj), {"shard": si})
            o0, o1 = rec["data_offsets"]
            d[field] = (base + o0, o1 - o0)
    mm = [np.memmap(sp, dtype=np.uint8, mode="r") for sp in shards]

    def W(e, proj, n, k):
        d = loc[(e, proj)]
        wo, wl = d["weight"]
        so, sl = d["weight_scale"]
        g2, _ = d["weight_scale_2"]
        gs = float(np.frombuffer(mm[d["shard"]][g2 : g2 + 4].tobytes(), dtype="<f4")[0])
        return dequant(np.asarray(mm[d["shard"]][wo : wo + wl]),
                       np.asarray(mm[d["shard"]][so : so + sl]), gs, n, k)

    # cache the 30 dequantized matrices once; only the activation format varies
    mats = {}
    for e in sel:
        e = int(e)
        mats[(e, "gate_proj")] = W(e, "gate_proj", inter, hidden)
        mats[(e, "up_proj")] = W(e, "up_proj", inter, hidden)
        mats[(e, "down_proj")] = W(e, "down_proj", hidden, inter)

    print(f"layer={layer}  topk={len(sel)}  ref_rms={np.sqrt(np.mean(y_ref**2)):.6f}")
    print(f"{'activation format':<16} {'rel_rms':>11} {'max_abs':>11}   note")
    rows = {}
    for name, q in FORMATS.items():
        xq = q(x)
        y = np.zeros(hidden, dtype=np.float64)
        for i, e in enumerate(sel):
            e = int(e)
            gp = mats[(e, "gate_proj")] @ xq
            up = mats[(e, "up_proj")] @ xq
            h = (gp / (1.0 + np.exp(-gp))) * up
            y += float(rw[i]) * (mats[(e, "down_proj")] @ q(h))
        d = y - y_ref
        rel = float(np.sqrt((d @ d) / (y_ref @ y_ref)))
        rows[name] = rel
        note = {
            "exact": "fp32 activations = the CPU fp32 core",
            "int8_g16": "the CPU VNNI core built here",
            "fp8e4m3_token": "<-- WHAT THE GPU SERVES TODAY",
        }.get(name, "")
        print(f"{name:<16} {rel:11.4e} {np.abs(d).max():11.4e}   {note}")
    if rows["int8_g16"] > 0:
        print(f"\nserved fp8-per-token / CPU int8-g16 = "
              f"{rows['fp8e4m3_token'] / rows['int8_g16']:.2f}x")


if __name__ == "__main__":
    main()
