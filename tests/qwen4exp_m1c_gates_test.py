"""M1-C correctness gates — A1.0 / A1.1 / A1.2 / A1.3 / A1.4 — ON REAL PINNED PAGES.

WHY THIS FILE EXISTS
--------------------
Every other offload test asserts about the arena, the plan, the seam or the bake. NONE of them
asserts that the weights the kernels read out of the pinned host arena are the SAME WEIGHTS the
un-offloaded model read. That is the only question that matters here, because Phase 0 produced four
independent cases of this driver returning `hipSuccess` over wrong state (P1/P2/P3 `location=Host`
silently allocating VRAM; P3 VMM over-commit faulting only at first touch; P6 `hipMemUnmap`→
`hipMemMap` serving the stale page with no non-zero return code; P5 `expandable_segments` handing
back zeroed memory). **A capability probe can PASS while the operation FAILS. Assert on the data.**

WHAT EACH GATE IS, AND WHAT MAKES IT FAIL
-----------------------------------------
  A1.0 eps0  — the noise floor. The SAME binary, the SAME fixed trajectory, N repeats, offload OFF.
               `eps0 = max|Δlogits|` over every pair. Nothing below it is meaningful.
               Non-vacuity is proved, not assumed: `--inject bitflip` flips ONE bit in ONE routed
               expert's packed weight and the same comparison must move far above eps0.
               Reported separately for the PREFILL step (M=8, the bit-exact WMMA gemm2 +
               gather_reduce) and the DECODE steps (M=1, `kernels.py`'s unconditional `M<=2`
               `mmq_fp8_moe_gemm_scatter`, whose atomic reduction order varies by its own admission).
               `MINISGL_MOE_G2FUSE=0` now also bypasses that scatter (a two-line change this gate
               required, plan §5.4), so a deterministic decode reference exists at all.

  A1.4 populate self-test — the Phase-0 trap detector, on real pages. Two independent checks:
               (a) `PinnedWeightArena.selftest_light()` — every chunk's fingerprint re-read THROUGH
                   THE DEVICE POINTER after all chunks were written. `--inject arena-fingerprint`
                   scribbles one word through that same device pointer and the self-test must go red.
               (b) the bake's own bitwise read-back sample (`moe_interpose._bake` step 4), counted in
                   `SeamBindReport.verified_components/_bytes`. `--inject bake-readback` corrupts an
                   arena row immediately before the comparison and the bake must REFUSE.

  A1.1 identity — offload ON (every MoE layer host-resident) vs offload OFF, same fixed trajectory,
               SAME PROCESS, same KV pool, same state: `max|Δ| <= eps0`.
               `--inject zero-component` zeroes one whole component in the arena; must blow up.

  A1.2 permuted mirror (MERGE GATE) — a second process builds the SAME model, applies a fixed random
               permutation to every per-expert component row AND to the router gate rows (so slot i
               of the permuted model IS old expert perm[i]), and only THEN bakes. Its host logits
               must agree with the identity process's host logits to eps0.
               The permutation is applied to the SOURCE, before the bake, on purpose: permuting the
               arena after the bake would be mirrored by any bake-side offset bug and detect nothing.
               `--no-route-remap` is the must-fail control — permute the experts, leave the router
               alone — and it must diverge catastrophically.

  A1.3 desync matrix (must-fail) — after an identity bake, each per-expert component of each
               container is independently rolled by one expert row IN THE ARENA and the trajectory
               re-run. Every arm must move `max|Δ|` ABOVE eps0. An arm that does not is only
               acceptable with a proof of expert-invariance (`t[0]` bitwise-equal to `t[1:]`), which
               this file computes rather than assumes.

RUN (card 0, in the serve image; PYTHONPATH must APPEND /opt/kernels):

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
      -v <worktree>:/engine -v /home/pat/.cache/hf-q4e:/model:ro -v /home/pat/.cache/hf-ple:/ple:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_m1c_gates_test.py \
         --arm identity --json /engine/out/ident.json --save-logits /engine/out/ident.pt'
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODEL = os.environ.get("Q4E_MODEL", "/model")
PLE_DIR = os.environ.get("Q4E_PLE", "/ple")
GIB = 1 << 30

_failures = 0
_results: "list[dict]" = []


def check_true(name: str, cond: bool, detail: str = "") -> bool:
    global _failures
    _failures += not cond
    print(f"  {'ok  ' if cond else 'FAIL'} {name:56s} {detail}", flush=True)
    return bool(cond)


def _f(x) -> float:
    return float(x)


# ======================================================================================
# the fixed trajectory
# ======================================================================================


class Harness:
    """Builds the model + a hand-made `Context`, and runs ONE fixed trajectory on demand.

    The trajectory is fixed in the strongest sense available: a fixed prompt AND fixed
    "decode" tokens (never argmax-fed, which would let a 1-ulp logit difference fork the
    two arms into different token sequences and report a numerics failure that is really a
    trajectory failure). Every repeat re-zeroes the GDN recurrent state and rebuilds the PLE
    state cache, so repeat k is not reading repeat k-1's residue.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.dev = torch.device("cuda:0")
        torch.cuda.set_device(self.dev)

        from minisgl.distributed import set_tp_info, try_get_tp_info

        if try_get_tp_info() is None:
            set_tp_info(0, 1)
        from minisgl.layers.rotary import set_rope_device

        set_rope_device(self.dev)

        from qwen4exp_gpu_forward_test import (
            _load_ple_multipliers,
            _load_real_subset,
            _fill_random,
            _make_ple_runtime,
            _subset_config,
        )

        from minisgl import core
        from minisgl.attention import create_attention_backend
        from minisgl.kvcache import create_kvcache_pool
        from minisgl.kvcache.gdn_state import GDNStateCache
        from minisgl.models import create_model
        from minisgl.models.config import ModelConfig
        from minisgl.moe import create_moe_backend
        from minisgl.utils import cached_load_hf_config

        import tempfile

        self._core = core
        tmp = tempfile.mkdtemp(prefix="q4e-m1c-")
        _subset_config(args.model, tmp, args.layers, args.experts, ple_1based=2)
        self.mc = mc = ModelConfig.from_hf(cached_load_hf_config(tmp), spec_algorithm="none")
        print(
            f"[config] layers={mc.num_layers} experts={mc.num_experts} top_k={mc.num_experts_per_tok} "
            f"hidden={mc.hidden_size} quant={getattr(mc.quant, 'ct_format', None)}",
            flush=True,
        )
        torch.set_default_dtype(torch.bfloat16)

        t0 = time.perf_counter()
        if args.weights == "random":
            with torch.device(self.dev):
                model = create_model(mc)
            gen = torch.Generator(device=self.dev).manual_seed(20260903)
            _fill_random(model, gen)
            _load_ple_multipliers(model, mc)
        else:
            with torch.device("meta"):
                model = create_model(mc)
            _load_real_subset(model, mc, self.dev, model_dir=args.model)
        model.post_load()
        torch.cuda.synchronize()
        self.model = model
        self.build_seconds = round(time.perf_counter() - t0, 1)

        # ---- context ----
        self._saved_ctx = core._GLOBAL_CTX
        core._GLOBAL_CTX = None
        ctx = core.Context(page_size=16)
        core.set_global_ctx(ctx)
        self.ctx = ctx
        max_running, max_seq = 2, 64
        page_size = ctx.page_size
        ctx.page_table = self.page_table = torch.zeros(
            (max_running + 1, max_seq), dtype=torch.int32, device=self.dev
        )
        ctx.kv_cache = create_kvcache_pool(
            model_config=mc,
            num_pages=1 + max_running * (max_seq // page_size),
            page_size=page_size,
            dtype=torch.bfloat16,
            device=self.dev,
        )
        ctx.attn_backend = create_attention_backend(args.attn_backend, mc)
        if mc.is_moe:
            ctx.moe_backend = create_moe_backend("fused")
        self.gdn = ctx.gdn_state = GDNStateCache(
            num_gdn_layers=mc.num_gdn_layers,
            num_slots=max_running + 2,
            conv_dim=mc.gdn_conv_dim,
            conv_kernel=mc.linear_conv_kernel_dim,
            num_v_heads=mc.linear_num_value_heads,
            head_v_dim=mc.linear_value_head_dim,
            head_k_dim=mc.linear_key_head_dim,
            dtype=torch.float32,
            ssm_dtype=torch.bfloat16,
            device=self.dev,
        )
        for g in model.iter_gdn_layers():
            g.warmup_conv(8)
        self.ple_rt = _make_ple_runtime(
            model, mc, self.dev, real=args.real_ple, max_seqs=max_running + 1,
            max_tokens=max(64, args.prompt_len),
        )
        ctx.ple = self.ple_rt
        self.page_table[0, :].copy_(
            torch.arange(page_size, page_size + max_seq, dtype=torch.int32, device=self.dev)
        )

        # ---- the fixed trajectory ----
        rng = np.random.default_rng(args.seed)
        self.prompt = rng.integers(0, mc.vocab_size, size=args.prompt_len, dtype=np.int64)
        self.decode_tokens = rng.integers(
            0, mc.vocab_size, size=args.decode_steps, dtype=np.int64
        )

    # -- trajectory ------------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.gdn.conv_state.zero_()
        self.gdn.ssm_state.zero_()
        if self.ple_rt is not None:
            blk = self.model.ple_block()
            self.ple_rt.state = blk.make_state_cache(
                num_slots=self.ple_rt.max_seqs + 1,
                eos_token_id=self.mc.ngram_eos_token_id,
                device=self.dev,
                dtype=torch.bfloat16,
            )

    @torch.inference_mode()
    def trajectory(self) -> torch.Tensor:
        """One prefill + `decode_steps` fixed decode steps. Returns (1+D, V) fp32 on CPU."""
        from qwen4exp_gpu_forward_test import _make_batch, _make_req
        from minisgl.gdn.metadata import build_gdn_metadata

        self._reset_state()
        out = []
        req = _make_req(self.prompt, table_idx=0)
        batch = _make_batch([req], "prefill", self.page_table, self.dev)
        self.ctx.attn_backend.prepare_metadata(batch)
        batch.gdn_metadata = build_gdn_metadata(
            batch, torch.tensor([1], dtype=torch.int32, device=self.dev), self.dev
        )
        if self.ple_rt is not None:
            self.ple_rt.prepare([1], [self.prompt])
        with self.ctx.forward_batch(batch):
            logits = self.model.forward()
        torch.cuda.synchronize()
        if self.ple_rt is not None:
            self.ple_rt.commit([1], [self.prompt])
        out.append(logits.float().cpu().clone())

        for nxt in self.decode_tokens:
            req.append_host(torch.tensor([int(nxt)], dtype=torch.int64))
            req.complete_one()
            batch = _make_batch([req], "decode", self.page_table, self.dev)
            self.ctx.attn_backend.prepare_metadata(batch)
            batch.gdn_metadata = build_gdn_metadata(
                batch, torch.tensor([1], dtype=torch.int32, device=self.dev), self.dev
            )
            tok = np.array([int(nxt)], dtype=np.int64)
            if self.ple_rt is not None:
                self.ple_rt.prepare([1], [tok])
            with self.ctx.forward_batch(batch):
                logits = self.model.forward()
            torch.cuda.synchronize()
            if self.ple_rt is not None:
                self.ple_rt.commit([1], [tok])
            out.append(logits.float().cpu().clone())
        return torch.cat(out, dim=0)

    # -- model surgery ---------------------------------------------------------------------
    def moe_layers(self):
        from minisgl.weights.moe_interpose import discover_moe_layers

        return discover_moe_layers(self.model)

    def sparse_blocks(self):
        """`(layer_index, block, moe_layer)` for every decoder layer's sparse MLP block.

        Found by walking the decoder and ASSERTING `block.experts is <the MoELayer the seam
        walker found>`, rather than by string-parsing the seam path — a gate whose router and
        whose experts came from different objects would permute one and not the other.
        """
        found = {id(l): p for p, l in self.moe_layers()}
        out = []
        for i, layer in enumerate(self.model.model.layers.op_list):
            blk = getattr(layer, "mlp", None)
            experts = getattr(blk, "experts", None)
            if experts is not None and id(experts) in found:
                out.append((i, blk, experts))
        if len(out) != len(found):
            raise RuntimeError(
                f"found {len(out)} sparse blocks for {len(found)} MoE layers — the router gate "
                f"cannot be paired with its experts, so a permuted mirror would remap one and not "
                f"the other."
            )
        return out

    @staticmethod
    def gate_weight(block, num_experts: int):
        """The (E, H) router projection of a sparse block, located by SHAPE not by name."""
        gate = getattr(block, "gate", None)
        if gate is None:
            raise RuntimeError("sparse block has no `gate` — cannot remap the route")
        cands = [
            (n, t)
            for n, t in vars(gate).items()
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == num_experts
        ]
        if len(cands) != 1:
            raise RuntimeError(f"router gate has {len(cands)} (E,*) tensors: {[n for n, _ in cands]}")
        return cands[0]

    def component_tensors(self):
        """`(path, attr, comp_name, tensor)` for every PER-EXPERT component of every MoE layer.

        Off `granule_specs()`, i.e. the same walk the bake and the residency proof use — so the
        set permuted/desynced here is exactly the set that moves into the arena, never a
        hand-listed one that could go stale against a format.
        """
        out = []
        for path, layer in self.moe_layers():
            specs = layer.granule_specs()
            for attr, spec in specs.items():
                container = getattr(layer, attr)
                for name, t in spec.stacked_tensors(container).items():
                    out.append((path, attr, name, t))
        return out

    def permute_experts(self, perm: torch.Tensor, *, route_remap: bool) -> dict:
        """Permute expert ROWS of every per-expert component, and (unless suppressed) the router.

        `perm[new] = old`: row `new` of the permuted stack holds what was expert `old`, and the
        router's row `new` holds what was expert `old`'s logit — so slot `new` of the permuted
        model IS old expert `old`, end to end. `route_remap=False` is the must-fail control.
        """
        E = int(self.mc.num_experts)
        moved = []
        for path, attr, name, t in self.component_tensors():
            if t.shape[0] != E:
                raise RuntimeError(f"{path}.{attr}.{name} has dim0={t.shape[0]}, expected E={E}")
            t.copy_(t[perm])
            moved.append(f"{path}.{attr}.{name}")
        gates = []
        if route_remap:
            for _i, blk, experts in self.sparse_blocks():
                gname, gw = self.gate_weight(blk, E)
                gw.copy_(gw[perm])
                gates.append(gname)
        torch.cuda.synchronize()
        return {"components_permuted": len(moved), "gates_permuted": len(gates)}

    def close(self) -> None:
        self._core._GLOBAL_CTX = self._saved_ctx


# ======================================================================================
# comparisons
# ======================================================================================


def max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


def per_step_delta(a: torch.Tensor, b: torch.Tensor) -> "list[float]":
    return [float((a[i] - b[i]).abs().max()) for i in range(a.shape[0])]


def eps0_from_repeats(runs: "list[torch.Tensor]") -> dict:
    """Pairwise max|Δ| over N repeats of the identical forward, split prefill vs decode."""
    n = len(runs)
    steps = runs[0].shape[0]
    per_step = [0.0] * steps
    overall = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = per_step_delta(runs[i], runs[j])
            per_step = [max(a, b) for a, b in zip(per_step, d)]
            overall = max(overall, max(d))
    return {
        "repeats": n,
        "eps0": overall,
        "eps0_prefill": per_step[0],
        "eps0_decode": max(per_step[1:]) if steps > 1 else 0.0,
        "per_step": per_step,
    }


# ======================================================================================
# fault injection
# ======================================================================================


def _flip_lsb(t: torch.Tensor, rows) -> None:
    """XOR 1 into element 0 of the given expert rows, in whatever storage dtype they are."""
    v = t.reshape(t.shape[0], -1)
    if t.dtype in (torch.int32, torch.int64, torch.uint8, torch.int8):
        v[rows, 0] = v[rows, 0] ^ 1
    elif t.dtype == torch.float16:
        w = v[rows, 0].view(torch.int16) ^ 1
        v[rows, 0] = w.view(torch.float16)
    else:
        raise RuntimeError(f"no bit-flip recipe for dtype {t.dtype}")
    torch.cuda.synchronize()


def inject_bitflip(h: Harness, mode: str) -> dict:
    """Flip ONE bit per expert of ONE component — the smallest per-expert weight change there is.

    Proves the whole comparison apparatus is non-vacuous: if one flipped bit inside a routed
    expert cannot be seen above eps0, neither can any desync arm, and every "pass" below would
    be a pass by insensitivity rather than by correctness.

    `mode="one"` flips expert 0 only, and is expected to be INVISIBLE on a top-10-of-512 router
    over a short trajectory — expert 0 is simply not routed. That is recorded rather than hidden,
    because it is the reason the default flips one bit in EVERY expert row: the gate has to
    perturb a weight the forward actually dereferences, not merely a weight that exists.
    """
    comps = h.component_tensors()
    path, attr, name, t = comps[0]
    rows = torch.tensor([0], device=t.device) if mode == "one" else torch.arange(
        t.shape[0], device=t.device
    )
    _flip_lsb(t, rows)
    return {
        "component": f"{path}.{attr}.{name}",
        "dtype": str(t.dtype),
        "experts_flipped": int(rows.numel()),
        "bits_flipped_per_expert": 1,
        "mode": mode,
    }


def inject_arena_fingerprint(arena) -> str:
    """Scribble ONE fingerprint word through the arena's own DEVICE pointer.

    Uses the same `hip.memcpy(device_ptr + off, ...)` path `_write_fingerprints` uses, so what
    is being tested is the read-back over real pinned pages, not a python-side mock.
    """
    from minisgl.weights import hipmem
    from minisgl.weights.pinned_arena import verify_offsets

    hip = arena._hip_or_bind()
    c = arena.chunks[-1]
    off = verify_offsets(c.nbytes)[len(verify_offsets(c.nbytes)) // 2]
    word = ctypes.c_uint32(0xDEADBEEF)
    hip.memcpy(c.device_ptr + off, ctypes.addressof(word), 4, hipmem.hipMemcpyHostToDevice)
    hip.sync()
    return f"chunk {c.index} offset {off}: fingerprint 0x{c.fingerprint:x} -> 0xdeadbeef"


def arm_bake_readback_injection() -> "list[str]":
    """Corrupt each arena row IMMEDIATELY BEFORE the bake compares it against its source.

    The comparison itself is untouched — `moe_interpose._bitwise_equal` still runs, on real
    arena bytes, and must return False. What is injected is the corruption, not the verdict.
    """
    from minisgl.weights import moe_interpose

    fired: "list[str]" = []
    real = moe_interpose._bitwise_equal

    def wrapper(dst, src):
        if not fired:
            v = dst.reshape(-1).view(torch.uint8)
            v[0] = (int(v[0].item()) ^ 0xFF) & 0xFF
            torch.cuda.synchronize()
            fired.append(f"scribbled byte 0 of a {dst.numel() * dst.element_size()} B arena row")
        return real(dst, src)

    moe_interpose._bitwise_equal = wrapper
    return fired


def inject_zero_component(h: Harness) -> str:
    """Zero one whole per-expert component IN THE ARENA (post-bake). A1.1's must-fail."""
    comps = h.component_tensors()
    path, attr, name, t = comps[0]
    t.zero_()
    torch.cuda.synchronize()
    return f"{path}.{attr}.{name} ({t.numel() * t.element_size()} B) zeroed in the arena"


# ======================================================================================
# main
# ======================================================================================


@torch.inference_mode()
def main() -> int:
    global _failures
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--weights", choices=("real", "random"), default="real")
    ap.add_argument("--real-ple", action="store_true", default=True)
    ap.add_argument("--no-real-ple", dest="real_ple", action="store_false")
    ap.add_argument("--attn-backend", default="rdna4")
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--perm-seed", type=int, default=20260903)
    ap.add_argument(
        "--arm",
        choices=("control", "identity", "permuted"),
        default="identity",
        help="control = no offload at all (the ledger-diff baseline and the eps0 reference); "
             "identity = all-host bake, A1.1 + A1.3 + A1.4; permuted = A1.2's mirror leg",
    )
    ap.add_argument("--no-route-remap", action="store_true",
                    help="A1.2 MUST-FAIL control: permute the experts, leave the router alone")
    ap.add_argument(
        "--inject",
        choices=("none", "bitflip", "arena-fingerprint", "bake-readback", "zero-component"),
        default="none",
    )
    ap.add_argument("--desync", action="store_true", help="run the A1.3 must-fail matrix")
    ap.add_argument("--host-gb", type=float, default=24.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--save-logits", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible")
        return 1

    from minisgl._hip_engage import _seen as ledger
    from minisgl.quant import kernels as qkernels

    out: dict = {
        "arm": args.arm,
        "inject": args.inject,
        "layers": args.layers,
        "experts": args.experts,
        "weights": args.weights,
        "prompt_len": args.prompt_len,
        "decode_steps": args.decode_steps,
        "repeats": args.repeats,
        "route_remap": not args.no_route_remap,
        "moe_g2fuse": bool(getattr(qkernels, "_MOE_G2FUSE", True)),
        "env_g2fuse": os.environ.get("MINISGL_MOE_G2FUSE", "(unset)"),
    }

    h = Harness(args)
    out["card"] = torch.cuda.get_device_name(0)
    out["build_seconds"] = h.build_seconds
    print(f"[gpu] {out['card']}  build {h.build_seconds}s", flush=True)

    # ---------------------------------------------------------------- A1.0
    print("\n[A1.0] eps0 — the same binary, the same trajectory, offload OFF", flush=True)
    runs = [h.trajectory() for _ in range(args.repeats)]
    e0 = eps0_from_repeats(runs)
    out["A1.0"] = e0
    ref = runs[0]
    out["logit_scale"] = {"absmax": float(ref.abs().max()), "std": float(ref.std())}
    print(
        f"  eps0={e0['eps0']:.6g}  prefill={e0['eps0_prefill']:.6g} decode={e0['eps0_decode']:.6g}"
        f"  |logits|max={out['logit_scale']['absmax']:.4g}",
        flush=True,
    )
    check_true("A1.0: logits are non-degenerate", out["logit_scale"]["absmax"] > 1e-3)
    check_true(
        "A1.0: prefill (M>2, bit-exact path) is EXACTLY deterministic",
        e0["eps0_prefill"] == 0.0,
        f"{e0['eps0_prefill']:.6g}",
    )
    eps0 = e0["eps0"]

    if args.inject == "bitflip":
        # Expert 0 alone FIRST, so the "a single unrouted expert is invisible" fact is measured
        # rather than assumed, and the all-experts arm is not mistaken for a coarse perturbation.
        one = inject_bitflip(h, "one")
        d_one = max_abs_delta(h.trajectory(), ref)
        _flip_lsb(h.component_tensors()[0][3], torch.tensor([0], device=h.dev))  # restore
        d_restore = max_abs_delta(h.trajectory(), ref)
        allx = inject_bitflip(h, "all")
        d_all = max_abs_delta(h.trajectory(), ref)
        out["inject_bitflip"] = {
            "one_expert": {**one, "max_delta": d_one},
            "restore_delta": d_restore,
            "every_expert": {**allx, "max_delta": d_all},
            "eps0": eps0,
        }
        check_true("A1.0 restore after the single-expert flip is exact", d_restore <= eps0,
                   f"Δ={d_restore:.6g}")
        check_true(
            "A1.0 non-vacuity: 1 flipped bit per expert moves max|Δ| above eps0",
            d_all > eps0,
            f"Δ={d_all:.6g} vs eps0={eps0:.6g}  [{allx['component']}, {allx['experts_flipped']} "
            f"experts x 1 bit];  expert-0-only Δ={d_one:.6g} (unrouted on this trajectory)",
        )
        _finish(out, args, ledger, None)
        return 1 if _failures else 0

    if args.arm == "control":
        out["engaged"] = sorted(ledger)
        _finish(out, args, ledger, ref)
        return 1 if _failures else 0

    # ---------------------------------------------------------------- A1.2 setup
    perm = None
    if args.arm == "permuted":
        print("\n[A1.2] permute the SOURCE expert rows (and the router) BEFORE the bake",
              flush=True)
        g = torch.Generator().manual_seed(args.perm_seed)
        perm = torch.randperm(int(h.mc.num_experts), generator=g).to(h.dev)
        out["perm_seed"] = args.perm_seed
        out["perm_digest"] = int(perm.sum().item()), int(perm[:8].cpu().tolist()[0])
        info = h.permute_experts(perm, route_remap=not args.no_route_remap)
        out["permute"] = info
        print(f"  {info}", flush=True)
        dev_perm = h.trajectory()
        d_dev = max_abs_delta(dev_perm, ref)
        out["A1.2_device_mirror"] = {"max_delta": d_dev, "eps0": eps0}
        print(f"  device-resident mirror: max|Δ| = {d_dev:.6g}  (eps0 {eps0:.6g})", flush=True)
        if args.no_route_remap:
            check_true(
                "A1.2 must-fail control: permuted experts + UNremapped route diverges",
                d_dev > eps0,
                f"Δ={d_dev:.6g} vs eps0={eps0:.6g}",
            )
        else:
            check_true(
                "A1.2 pre-bake control: the mirror is exact while still device-resident",
                d_dev <= eps0,
                f"Δ={d_dev:.6g} vs eps0={eps0:.6g} — a failure here is a ROUTE artefact "
                f"(top-k tie order), not an arena bug",
            )
        ref_for_host = dev_perm
    else:
        ref_for_host = ref

    # ---------------------------------------------------------------- the bake
    print("\n[bake] pin the arena and move every MoE layer host-resident", flush=True)
    from minisgl.weights.bake import StageASession
    from minisgl.weights.stacks import StackKind

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.model_config = h.mc
    cfg.weight_offload_device_gb = 0.0
    cfg.weight_offload_gb = args.host_gb
    cfg.dtype = torch.bfloat16
    cfg.enable_ep = False

    if args.inject == "bake-readback":
        fired = arm_bake_readback_injection()
        out["inject_bake_readback"] = {"armed": True}

    def _probe():
        """`(free, allocated, reserved)` — the same three the engine's `_woff_mem_probe` returns.

        Mandatory, not optional: `seal()` refuses an enabled session with no probe, because every
        one of its gates is a DIFFERENCE between two samples and a missing sample reads as a delta
        of zero — i.e. "the host arena cost no VRAM" would pass vacuously, which is precisely the
        Phase-0 assertion that must not be waved through.
        """
        torch.cuda.synchronize(h.dev)
        return (
            torch.cuda.mem_get_info(h.dev)[0],
            torch.cuda.memory_allocated(h.dev),
            torch.cuda.memory_reserved(h.dev),
        )

    t0 = time.perf_counter()
    sess = StageASession.begin(
        cfg, model=h.model, probe=_probe, device_budget_bytes=0, budget_is_derived=False, log=print
    )
    check_true(
        "session ENABLED (a non-empty, all-host plan)",
        sess.enabled,
        f"host={sess.accounting.host_bytes / GIB:.3f} GiB "
        f"device={sess.accounting.device_bytes / GIB:.3f} GiB",
    )
    out["plan_host_bytes"] = int(sess.accounting.host_bytes)
    out["plan_device_bytes"] = int(sess.accounting.device_bytes)

    bake_error = None
    try:
        sess.attach()
        arena = sess.driver.arena
        st = arena._selftest
        out["A1.4_arena_selftest"] = st.summary() if st is not None else None
        print(f"  arena selftest: {out['A1.4_arena_selftest']}", flush=True)
        check_true(
            "A1.4(a): arena fingerprint self-test PASSED on real pages",
            bool(st is not None and st.passed),
            str(out["A1.4_arena_selftest"]),
        )
        check_true(
            "A1.4(a): the self-test was NON-VACUOUS (chunks x offsets probed)",
            bool(st is not None and st.chunks_checked > 0 and st.offsets_per_chunk > 0),
            f"{getattr(st, 'chunks_checked', 0)} chunks x "
            f"{getattr(st, 'offsets_per_chunk', 0)} offsets",
        )

        if args.inject == "arena-fingerprint":
            what = inject_arena_fingerprint(arena)
            red = arena.selftest_light()
            out["inject_arena_fingerprint"] = {"what": what, "result": red.summary()}
            check_true(
                "A1.4(a) MUST-FAIL: a scribbled fingerprint word turns the self-test RED",
                not red.passed and len(red.failures) > 0,
                f"{what} -> passed={red.passed} failures={len(red.failures)}",
            )
            _finish(out, args, ledger, None)
            return 1 if _failures else 0

        sess.note_loaded()
        sess.bind(h.model)
        sess.seal()
    except BaseException as e:  # noqa: BLE001 - reported, never swallowed
        bake_error = f"{type(e).__name__}: {e}"
        out["bake_error"] = bake_error
        if args.inject == "bake-readback":
            check_true(
                "A1.4(b) MUST-FAIL: a corrupted arena row makes the bake REFUSE",
                "read-back does not match" in str(e),
                bake_error[:200],
            )
            out["inject_bake_readback"]["fired"] = fired
            _finish(out, args, ledger, None)
            return 1 if _failures else 0
        traceback.print_exc()
        _finish(out, args, ledger, None)
        return 1

    if args.inject == "bake-readback":
        check_true(
            "A1.4(b) MUST-FAIL: a corrupted arena row makes the bake REFUSE",
            False,
            "the bake COMPLETED over a corrupted row — the read-back gate did not fire",
        )
        _finish(out, args, ledger, None)
        return 1

    out["bake_seconds"] = round(time.perf_counter() - t0, 1)
    out["copied_bytes"] = int(sess.accounting.copied_bytes)
    out["arena_pinned_bytes"] = int(arena.pinned_bytes)
    out["arena_torch_fallbacks"] = int(arena.torch_fallbacks)
    verified = sum(r.verified_components for r in sess.outcome.reports)
    verified_b = sum(r.verified_bytes for r in sess.outcome.reports)
    comps = sum(r.components for r in sess.outcome.reports)
    out["A1.4_bake_readback"] = {
        "components": comps,
        "verified_components": verified,
        "verified_bytes": verified_b,
    }
    print(
        f"  bake: {comps} components, {verified} bitwise-read-back-verified "
        f"({verified_b / GIB:.3f} GiB), {out['copied_bytes'] / GIB:.3f} GiB copied",
        flush=True,
    )
    check_true("A1.4(b): the bake read-back sample ran and verified bytes",
               verified > 0 and verified_b > 0, f"{verified} components / {verified_b} B")
    check_true("arena hipMalloc fallbacks == 0", out["arena_torch_fallbacks"] == 0,
               str(out["arena_torch_fallbacks"]))

    # residency, from outside the code under test
    host_owned = dev_owned = 0
    seams = list(sess.driver.seams)
    for s in seams:
        for _n, t in s.live_tensors():
            owned = arena.owns_pointer(t.data_ptr(), t.numel() * t.element_size())
            if s.kind is StackKind.HOST:
                host_owned += owned
            else:
                dev_owned += owned
    out["seam_host_tensors_in_arena"] = host_owned
    out["seam_device_tensors_in_arena"] = dev_owned
    check_true("every MoE layer is HOST-placed", all(s.kind is StackKind.HOST for s in seams),
               f"{sum(s.kind is StackKind.HOST for s in seams)}/{len(seams)}")
    check_true("host tensors live inside a pinned arena chunk", host_owned > 0, f"{host_owned}")

    if args.inject == "zero-component":
        what = inject_zero_component(h)
        d = max_abs_delta(h.trajectory(), ref_for_host)
        out["inject_zero_component"] = {"what": what, "max_delta": d, "eps0": eps0}
        check_true(
            "A1.1 MUST-FAIL: a zeroed arena component diverges above eps0",
            d > eps0, f"Δ={d:.6g} vs eps0={eps0:.6g}  [{what}]",
        )
        _finish(out, args, ledger, None)
        return 1 if _failures else 0

    # ---------------------------------------------------------------- A1.1 / A1.2
    print("\n[forward] the same fixed trajectory, now reading the pinned host arena", flush=True)
    host_logits = h.trajectory()
    d_host = max_abs_delta(host_logits, ref_for_host)
    steps = per_step_delta(host_logits, ref_for_host)
    out["host_vs_device_same_process"] = {"max_delta": d_host, "per_step": steps, "eps0": eps0}
    print(f"  max|Δ| host-arena vs device = {d_host:.6g}   per-step {['%.3g' % s for s in steps]}",
          flush=True)
    if args.arm == "identity":
        check_true("A1.1 IDENTITY: host-resident experts == device-resident, to eps0",
                   d_host <= eps0, f"Δ={d_host:.6g} vs eps0={eps0:.6g}")
    else:
        check_true("A1.2: the permuted model is unchanged by moving it into the arena",
                   d_host <= eps0, f"Δ={d_host:.6g} vs eps0={eps0:.6g}")

    # ---------------------------------------------------------------- A1.3
    if args.desync:
        print("\n[A1.3] desync matrix — every per-expert component, independently, IN THE ARENA",
              flush=True)
        arms = []
        for path, attr, name, t in h.component_tensors():
            key = f"{attr}.{name}"
            t.copy_(t.roll(1, 0))
            torch.cuda.synchronize()
            d = max_abs_delta(h.trajectory(), host_logits)
            t.copy_(t.roll(-1, 0))
            torch.cuda.synchronize()
            invariant = None
            if d <= eps0:
                invariant = bool((t[1:] == t[0]).all())
            arms.append(
                {"layer": path, "component": key, "dtype": str(t.dtype),
                 "bytes": int(t.numel() * t.element_size()), "max_delta": d,
                 "moved_above_eps0": d > eps0, "expert_invariant": invariant}
            )
            print(f"  {path}.{key:22s} roll-1 -> max|Δ| = {d:.6g}"
                  f"{'' if d > eps0 else '   <-- NOT ABOVE eps0'}", flush=True)
        out["A1.3"] = arms
        bad = [a for a in arms if not a["moved_above_eps0"] and not a["expert_invariant"]]
        check_true(
            "A1.3: every desynced component moved max|Δ| above eps0",
            not bad,
            f"{len(bad)} arm(s) did not move and are not expert-invariant: "
            f"{[a['component'] for a in bad][:4]}",
        )
        # And the restore must be exact, or every arm after the first is measuring residue.
        d_restored = max_abs_delta(h.trajectory(), host_logits)
        out["A1.3_restore_delta"] = d_restored
        check_true("A1.3: the arena was restored exactly after the matrix",
                   d_restored <= eps0, f"Δ={d_restored:.6g}")

    out["engaged"] = sorted(ledger)
    _finish(out, args, ledger, host_logits)
    return 1 if _failures else 0


def _finish(out: dict, args, ledger, logits) -> None:
    out["engaged"] = sorted(ledger)
    out["failures"] = _failures
    print("\n[ledger] engaged() names", flush=True)
    for n in out["engaged"]:
        print(f"  {n}", flush=True)
    print(f"\n{'PASS' if not _failures else 'FAIL'}: {_failures} failure(s)", flush=True)
    print(json.dumps({k: v for k, v in out.items() if k != "engaged"}, indent=2, default=str),
          flush=True)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
    if args.save_logits and logits is not None:
        os.makedirs(os.path.dirname(args.save_logits) or ".", exist_ok=True)
        torch.save(logits, args.save_logits)
        print(f"[saved] {args.save_logits}  {tuple(logits.shape)}", flush=True)


def compare_mode(argv) -> int:
    """`--compare A.pt B.pt` — the cross-process half of A1.2 and the ledger diff."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, required=True)
    ap.add_argument("--eps0", type=float, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    x, y = torch.load(a.compare[0]), torch.load(a.compare[1])
    d = max_abs_delta(x, y)
    steps = per_step_delta(x, y)
    print(json.dumps({"label": a.label, "max_delta": d, "per_step": steps, "eps0": a.eps0,
                      "pass": d <= a.eps0}, indent=2))
    return 0 if d <= a.eps0 else 1


if __name__ == "__main__":
    try:
        if "--compare" in sys.argv:
            raise SystemExit(compare_mode(sys.argv[1:]))
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
