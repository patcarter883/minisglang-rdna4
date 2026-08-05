#!/usr/bin/env python
"""Reproduce, and then GATE, the two defects in the fused MoE gemm1+silu BN plumbing.

DEFECT 1 -- SIGFPE (exit 136), host-side integer divide by zero.
    `make_moe_tile_config` defaults `cfg.BN = 0` ("0 = ask the chooser"). The UNFUSED grouped
    launcher guards it (`int bn = cfg.BN > 0 ? cfg.BN : MV5_BN`); the FUSED gemm1+silu launcher
    did not (`int bn = cfg.BN;`) and then computed `dim3 grid((inter + bn - 1) / bn, ...)`.
    bn == 0 -> integer division by zero ON THE HOST -> SIGFPE, which takes the whole process down.

    It shows up at "M > 32" only because that is where the ENGINE first routes gemm1 to WMMA at
    all: minisgl `_MOE_GEMM1_GEMV_MAX = 32`, so M <= 32 runs the gemv gemm1, which returns from the
    launcher BEFORE the unguarded divide. The threshold is an engine dispatch boundary, not a
    kernel one -- which is why bisecting on M found "32" and nothing in the kernel explained it.

DEFECT 2 -- a SILENT mis-launch (the fifth silent fallback).
    `if (bn == 128) G1SILU_ASHUF(128); else G1SILU_ASHUF(MV5_BN);` selects the kernel by TEMPLATE
    BN, but the grid was already sized with the RUNTIME bn. The kernel indexes
    `block_n = blockIdx.x * BN<template>`. So a requested BN=96 launches ceil(inter/96) blocks that
    each cover only 64 columns: ceil(inter/96)*64 < inter, and the tail output columns are NEVER
    WRITTEN. No error, no fallback message -- a wrong answer AND an under-reported time.

Both arms below run in SUBPROCESSES, because defect 1 is fatal by construction and a gate that
dies with the thing it is gating is not a gate.

    gpu-lease -n 1 -- bash tools/w4a8_moe_bn_fault_run.sh
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# q35.moe tp2 -- the shape the fault was first seen on.
SHAPE = dict(E=256, top_k=8, hidden=2048, inter=768, g=128)

CHILD = r'''
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4a8_moe_tile_surface import expert_stack
from minisgl.quant.kernels import w4a8_moe

M   = int(sys.argv[1])
bn  = sys.argv[2]            # "" = leave VLLM_W4A8_MOE_BN unset (the crashing configuration)
mode = sys.argv[3]           # "run" | "cover"

E, top_k, hidden, inter, g = %d, %d, %d, %d, %d
dev = torch.device("cuda:0")
torch.manual_seed(0)
if bn:
    os.environ["VLLM_W4A8_MOE_BN"] = bn
else:
    os.environ.pop("VLLM_W4A8_MOE_BN", None)

w13p, w13s = expert_stack(E, 2 * inter, hidden, g)
w2p,  w2s  = expert_stack(E, hidden, inter, g)
x    = torch.randn(M, hidden, device=dev, dtype=torch.bfloat16) * 0.05
gate = torch.randn(M, E, device=dev, dtype=torch.bfloat16)

if mode == "cover":
    # Probe the fused gemm1+silu DIRECTLY and read it against the ONE BN that is genuinely
    # compiled (64). The tile is a LAUNCH parameter, so every BN must return the SAME bytes.
    # A column the mis-launched grid never reached holds whatever was in the fresh allocation,
    # which is NOT reliably zero -- so "all-zero column" is not the test. "differs from BN=64"
    # is, and it also catches a partially-covered column.
    import fp8_wmma, moe_hip
    import torch.nn.functional as F
    block_m = 16
    tw = F.softmax(gate.float(), dim=-1)
    tw, tid = torch.topk(tw, top_k, dim=-1)
    tid = tid.to(torch.int32).contiguous()
    sti, eid, ntp = moe_hip.moe_align(tid, E, block_m)

    def g1(b):
        if b:
            os.environ["VLLM_W4A8_MOE_BN"] = str(b)
        else:
            os.environ.pop("VLLM_W4A8_MOE_BN", None)
        return fp8_wmma.mmq_fp8_moe_gemm1_silu(
            x.contiguous(), w13p, w13s, sti, eid, ntp, top_k, block_m, kernel="wmma")

    live = int(ntp[0].item())
    # Only the REAL rows are defined. `moe_align` pads each expert's block, and the kernel
    # skips a padded slot (`if (offs_token >= num_valid_tokens) continue`), so those rows keep
    # whatever was in the fresh allocation -- frequently NaN. Comparing them compares garbage
    # to garbage and makes every delta NaN, which then silently passes a `> 0` test.
    valid = (sti[:live] < M * top_k)
    ref = g1(64)[:live].float()[valid]
    got = g1(int(bn))[:live].float()[valid]
    # NaN-aware: `!=` is False for NaN vs NaN, so a NaN column would otherwise read as clean.
    diff = (got != ref) | (got.isnan() ^ ref.isnan())
    bad_cols = int(diff.any(dim=0).sum().item())
    fin = torch.isfinite(got - ref)
    md = (got - ref).abs()[fin].max().item() if fin.any() else float("nan")
    print(f"RESULT inter={inter} rows={int(valid.sum())} maxdelta={md:.3e} "
          f"bad_cols={bad_cols}/{inter}")
    sys.exit(0)

out = w4a8_moe(x, w13p, w13s, None, w2p, w2s, None, gate, top_k, True, kernel="wmma")
torch.cuda.synchronize()
print(f"RESULT ok finite={bool(torch.isfinite(out).all().item())} "
      f"absmax={out.abs().max().item():.6g}")
''' % (SHAPE["E"], SHAPE["top_k"], SHAPE["hidden"], SHAPE["inter"], SHAPE["g"])


def child(M: int, bn: str, mode: str) -> tuple[int, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, "_bn_fault_child.py")
    with open(p, "w") as fh:
        fh.write(CHILD)
    r = subprocess.run([sys.executable, p, str(M), bn, mode],
                       capture_output=True, text=True, cwd=here)
    tail = (r.stdout + r.stderr).strip().splitlines()
    msg = next((l for l in tail if l.startswith("RESULT")), (tail[-1] if tail else ""))
    return r.returncode, msg[:160]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    fh = open(args.out, "w") if args.out else None

    def out(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    import torch
    p = torch.cuda.get_device_properties(0)
    out(f"device: {p.name}  WGPs={p.multi_processor_count} -> {2 * p.multi_processor_count} CUs")
    out(f"shape: q35.moe tp2 {SHAPE}")
    out("")

    bad = 0

    # --- DEFECT 1: the M ladder, with BN UNSET (the served/default configuration) ---
    out("A. VLLM_W4A8_MOE_BN UNSET (cfg.BN = 0 -> the chooser sentinel)")
    out(f"   {'M':>6}  {'exit':>5}  note")
    for M in (1, 2, 8, 32, 33, 64, 129, 512):
        rc, msg = child(M, "", "run")
        flag = ""
        if rc == 136:
            flag = "  <-- SIGFPE (host integer divide by zero)"
            bad += 1
        elif rc != 0:
            flag = f"  <-- FAILED"
            bad += 1
        out(f"   {M:>6}  {rc:>5}  {msg}{flag}")

    # --- control: the same M ladder with BN forced to the compiled default ---
    out("")
    out("B. CONTROL -- same M ladder, VLLM_W4A8_MOE_BN=64 (cfg.BN non-zero)")
    out(f"   {'M':>6}  {'exit':>5}  note")
    for M in (33, 64, 129, 512):
        rc, msg = child(M, "64", "run")
        out(f"   {M:>6}  {rc:>5}  {msg}{'' if rc == 0 else '  <-- FAILED'}")
        if rc != 0:
            bad += 1

    # --- DEFECT 2: template/grid BN mismatch leaves output columns unwritten ---
    out("")
    out("C. FUSED gemm1+silu at M=64, every requested BN read against BN=64")
    out("   The tile is a LAUNCH parameter, so every BN must return the SAME bytes.")
    out("   bad_cols > 0 = the grid was sized for the requested BN but the kernel was")
    out("   instantiated at another one -> columns the grid never reached.")
    out(f"   {'BN':>6}  {'exit':>5}  note")
    for bn in (16, 32, 64, 96, 128, 192, 256):
        rc, msg = child(64, str(bn), "cover")
        nbad = -1
        if "bad_cols=" in msg:
            nbad = int(msg.split("bad_cols=")[1].split("/")[0])
        flag = ""
        if rc != 0:
            # A LOUD, EXPLAINED REFUSAL IS A PASS. BN=256 genuinely does not fit: the fused
            # gemm1+silu stages BOTH the gate and the up slab, so its LDS bill is 2x the unfused
            # path's and 2*256*(128+8) = 69632 B is over the 65536 budget at group_size=128. The
            # defect being gated here is the SILENT substitution that used to happen instead; a
            # TORCH_CHECK naming the tile and the byte count is the fix, not another fault.
            flag = "  <-- refused (expected: names the tile and the budget)"
            if "of LDS" not in msg:
                flag = "  <-- REFUSED WITHOUT EXPLAINING WHY"
                bad += 1
        elif nbad > 0:
            flag = "  <-- SILENT MIS-LAUNCH"
            bad += 1
        out(f"   {bn:>6}  {rc:>5}  {msg}{flag}")

    out("")
    out(f"VERDICT: {'FAULTS PRESENT' if bad else 'CLEAN'} ({bad} bad cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
