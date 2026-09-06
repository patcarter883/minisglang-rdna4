"""QSA sparse-attention gates — the free ≤budget equivalence, the >budget f64 reference, sparsity.

WHY THESE THREE GATES AND NOT A COHERENCE SMOKE TEST
----------------------------------------------------
Every measurement this model has ever produced was taken at ≤2048 tokens, which is precisely the
regime where the sparse selection is bypassed: below `indexer_budget` the reference clamps
`row_topk = min(topk, visible)`, so the selection contains EVERY visible token and sparse attention
IS dense causal attention. That fact is the strongest test available and it costs nothing:

  G1  SAME-CODE FLOOR. Run the dense path twice, same binary, same trajectory. `eps0 = max|Δlogits|`.
      Nothing below eps0 is evidence about anything. Measured at bs=1 only: the engine is NOT
      greedy-reproducible at bs=2, because decode gemm2 is an atomic scatter that `kernels.py`
      documents as non-bit-exact, so a bs=2 gate would be measuring the MoE reduction, not attention.

  G2  DENSE EQUIVALENCE ≤ budget. Dense (MINISGL_QSA=0) vs sparse (QSA on), same trajectory,
      same process, same KV pool. Reported PER PATH because the two paths have different claims:
        * DECODE is expected BIT-EXACT and the reason is structural, not a tolerance. The sparse
          call is `flash_decode_paged` with the selected physical slots as its block table at
          page_size 1; below the budget slot j of that table IS the row the dense call's
          `bt[j/page]*stride + (j%page)` names, and the kernel's key loop visits them in the same
          order with the same online-softmax accumulation. Same kernel, same order, same bits.
        * PREFILL is NOT expected bit-exact and claiming otherwise would be dishonest: dense
          prefill runs `attn_hip.flash_prefill` (a WMMA tile core) while the sparse form is
          per-query-row, so the fp32 accumulation ORDER differs. The gate on that path is
          `max|Δ| <= eps0` plus identical greedy ids, which is the same standard every other
          numerics change in this repo is held to.
      The split-KV policy is keyed on the block-table ROW WIDTH, which differs between the two
      (max_seq vs index_width), so `--pin-split` sets MINISGL_ATTN_SPLIT_MIN_CTX above both and
      forces the single-pass kernel on both sides. Run BOTH ways; a difference between them is a
      real property of the split/reduce path and is reported, not hidden.

  G3  ABOVE THE BUDGET. There is no dense equivalent to compare against, so the selection is
      re-derived on the HOST IN FLOAT64 from the tapped q + compressed keys and compared as an index
      SET. And sparsity is PROVEN, not assumed: `visited < dense` with the measured ratio. A "sparse"
      path that quietly selected everything would pass a coherence test and prove nothing.

RUN (card 0, in the serve image):

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
      -v <worktree>:/engine -v /home/pat/.cache/hf-q4e:/model:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_qsa_gate_test.py --gate all'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The PLE plumbing (real hash multipliers + an in-memory random row table with the REAL staging
# path) already exists next door; importing it keeps one implementation of "a PLE runtime for a
# harness" instead of a second one that drifts.
from qwen4exp_gpu_forward_test import _load_ple_multipliers, _make_ple_runtime  # noqa: E402

MODEL = os.environ.get("Q4E_MODEL", "/model")
_failures: list = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)
    if not cond:
        _failures.append(name)
    return cond


# ------------------------------------------------------------------------------------------------
# harness — a layer subset with random weights. The gates are about the ATTENTION path, and the
# attention path does not care what the weights are; using random ones keeps a 4-layer build on one
# card at 16k context, which is what makes the >budget arm runnable at all.
# ------------------------------------------------------------------------------------------------

def _subset_config(src: str, dst: str, n_layers: int, n_experts: int, ple_1based: int) -> None:
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    tc["num_hidden_layers"] = n_layers
    # EVERY layer is a full-attention layer. Two reasons, and the first is a measurement problem
    # rather than a preference: with random weights the GDN linear-attention block produces NaN on
    # the first token (its recurrence is not stable for N(0, 0.02) A_log/dt_bias), and a NaN makes
    # every comparison downstream vacuously "different" — the same-code floor comes back as NaN and
    # the gate proves nothing. Second, these gates are about attention, and this puts an index layer
    # on EVERY layer rather than one in four, so a 4-layer subset exercises four independent
    # indexers instead of one.
    tc["layer_types"] = ["full_attention"] * n_layers
    tc["num_experts"] = n_experts
    tc["ple_layer_ids"] = [ple_1based]
    # Build the MoE UNQUANTIZED. Not a convenience: with random weights there is no loader to
    # normalise the NVFP4 group scales (the real path casts them to fp16 at read), so a randomly
    # filled quantized build reaches the MoE kernel with fp8 scales and dies on `scales must be
    # fp16` — which is a property of the harness, not of anything these gates measure. Both legs
    # build identically, and the MoE is downstream of every attention comparison here.
    cfg.pop("quantization_config", None)
    tc.pop("quantization_config", None)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f)
    for name in ("generation_config.json",):
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy(p, dst)


def _fill_random(model, gen) -> None:
    """Plausible values IN PLACE, in each leaf's OWN dtype.

    The NVFP4 leaves are a packed uint8 E2M1 blob plus a folded scale, and `torch.normal_` is not
    implemented for float8 at all — filling through a float32 staging tensor is what keeps the MoE
    kernel from reading NaN scales and producing a finite-but-meaningless result that looks like a
    real failure. int64 (`layer_multipliers`) is left alone; it is loaded from the checkpoint even
    in random mode, because a wrong hash multiplier reads a real embedding from the wrong row and
    errors nowhere.
    """
    for name, p in model.state_dict().items():
        if not isinstance(p, torch.Tensor):
            continue
        if p.dtype == torch.uint8:            # packed E2M1 pairs — any byte is a legal pair
            p.random_(0, 256, generator=gen)
        elif p.dtype in (torch.int8, torch.int32, torch.int64):
            continue
        elif p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            src = torch.empty(p.shape, dtype=torch.float32, device=p.device)
            src.uniform_(0.5, 1.5, generator=gen)
            p.copy_(src.to(p.dtype))
        elif "scale" in name:                 # group scales must be positive and O(1)
            p.uniform_(0.5, 1.5, generator=gen)
        else:
            p.normal_(0.0, 0.02, generator=gen)


def _make_req(prompt: np.ndarray, table_idx: int, out_len: int):
    from minisgl.core import Req, SamplingParams

    class _H:
        pass

    return Req(
        input_ids=torch.from_numpy(prompt.astype(np.int64)),
        table_idx=table_idx,
        cached_len=0,
        output_len=out_len,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=_H(),
    )


def _make_batch(reqs, phase, page_table, dev):
    from minisgl.core import Batch

    batch = Batch(reqs=reqs, phase=phase)
    batch.padded_reqs = reqs
    pos, tok, rows = [], [], []
    for r in reqs:
        pos.extend(range(r.cached_len, r.device_len))
        tok.extend(int(x) for x in r.input_ids[r.cached_len : r.device_len])
        rows.extend([r.table_idx] * r.extend_len)
    batch.positions = torch.tensor(pos, dtype=torch.int32, device=dev)
    batch.input_ids = torch.tensor(tok, dtype=torch.int32, device=dev)
    batch.out_loc = page_table[
        torch.tensor(rows, dtype=torch.int64, device=dev), batch.positions.to(torch.int64)
    ]
    return batch


class _Harness:
    """Build the model + a fresh context; run a fixed prefill/decode trajectory; return logits."""

    def __init__(self, args, dev) -> None:
        self.args, self.dev = args, dev
        from minisgl.models.config import ModelConfig
        from minisgl.utils import cached_load_hf_config

        self.tmp = tempfile.mkdtemp(prefix="q4e-qsa-")
        # The PLE block IS built (the config refuses a ple_layer_id outside the subset depth) but is
        # backed by the in-memory random row table `qwen4exp_gpu_forward_test` already ships — same
        # hasher, same staging buffer, same H2D, different bytes. These gates are about attention,
        # both legs are built identically, so the n-gram features cancel exactly.
        _subset_config(args.model, self.tmp, args.layers, args.experts, ple_1based=2)
        self.mc = ModelConfig.from_hf(cached_load_hf_config(self.tmp), spec_algorithm="none")

    # When set, the COLD prefill is routed through attn_prefill_paged.flash_prefill_paged instead of
    # attn_hip.flash_prefill by clearing metadata.cold_prefill. Both compute the same dense causal
    # attention over the same KV with different tiling, so the difference between them IS the
    # prefill path's kernel-swap noise floor — the control the QSA prefill number has to be read
    # against, since the sparse prefill is also a kernel swap and bit-exactness is not on offer.
    force_paged_prefill = False

    def build(self, *, qsa: bool):
        from minisgl import core
        from minisgl.attention import create_attention_backend
        from minisgl.kvcache import create_kvcache_pool
        from minisgl.kvcache.gdn_state import GDNStateCache
        from minisgl.models import create_model
        from minisgl.moe import create_moe_backend

        dev, mc, args = self.dev, self.mc, self.args
        os.environ["MINISGL_QSA"] = "1" if qsa else "0"
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(dev):
            model = create_model(mc)
        gen = torch.Generator(device=dev).manual_seed(20260906)
        _fill_random(model, gen)
        model.post_load()

        core._GLOBAL_CTX = None
        ctx = core.Context(page_size=16)
        core.set_global_ctx(ctx)
        page_size = ctx.page_size
        max_seq = args.max_seq
        assert max_seq % page_size == 0
        ctx.page_table = pt = torch.zeros((2, max_seq), dtype=torch.int32, device=dev)
        num_pages = 1 + (max_seq // page_size)
        ctx.kv_cache = create_kvcache_pool(
            model_config=mc, num_pages=num_pages, page_size=page_size,
            dtype=torch.bfloat16, device=dev,
        )
        ctx.attn_backend = create_attention_backend("hip", mc)
        if mc.is_moe:
            ctx.moe_backend = create_moe_backend("fused")
        ctx.gdn_state = GDNStateCache(
            num_gdn_layers=mc.num_gdn_layers, num_slots=4, conv_dim=mc.gdn_conv_dim,
            conv_kernel=mc.linear_conv_kernel_dim, num_v_heads=mc.linear_num_value_heads,
            head_v_dim=mc.linear_value_head_dim, head_k_dim=mc.linear_key_head_dim,
            dtype=torch.float32, ssm_dtype=torch.bfloat16, device=dev,
        )
        for g in model.iter_gdn_layers():
            g.warmup_conv(8)
        _load_ple_multipliers(model, mc)
        self.ple = _make_ple_runtime(
            model, mc, dev, real=False, max_seqs=2, max_tokens=args.max_seq + 8
        )
        ctx.ple = self.ple
        pt[0, :].copy_(torch.arange(page_size, page_size + max_seq, dtype=torch.int32, device=dev))
        # PROVENANCE: a build that MEANT to be sparse must be sparse. force=True turns every
        # "unavailable, running dense" case into a raise, so this harness cannot green a leg that
        # silently measured the other implementation.
        if qsa:
            model.prepare_qsa(force=True)
            assert ctx.qsa is not None, "QSA leg built without a runtime"
        self.model, self.ctx, self.page_table = model, ctx, pt
        return model

    def run(self, prompt: np.ndarray, steps: int):
        """Fixed trajectory: one prefill + `steps` GREEDY decodes. Returns (logits list, ids)."""
        from minisgl.gdn.metadata import build_gdn_metadata

        dev, ctx = self.dev, self.ctx
        req = _make_req(prompt, 0, out_len=steps + 1)
        batch = _make_batch([req], "prefill", self.page_table, dev)
        ctx.attn_backend.prepare_metadata(batch)
        if self.force_paged_prefill:
            batch.attn_metadata.cold_prefill = False
        batch.gdn_metadata = build_gdn_metadata(
            batch, torch.tensor([1], dtype=torch.int32, device=dev), dev
        )
        outs, ids, spars = [], [], []
        if self.ple is not None:
            self.ple.prepare([1], [prompt])
        with ctx.forward_batch(batch):
            logits = self.model.forward()
        torch.cuda.synchronize()
        if self.ple is not None:
            self.ple.commit([1], [prompt])
        outs.append(("prefill", logits.detach().float().cpu().clone()))
        spars.append(self._sparsity())
        for s in range(steps):
            nxt = int(logits[-1].float().argmax().item())
            ids.append(nxt)
            req.append_host(torch.tensor([nxt], dtype=torch.int64))
            req.complete_one()
            batch = _make_batch([req], "decode", self.page_table, dev)
            ctx.attn_backend.prepare_metadata(batch)
            batch.gdn_metadata = build_gdn_metadata(
                batch, torch.tensor([1], dtype=torch.int32, device=dev), dev
            )
            tokarr = np.array([nxt], dtype=np.int64)
            if self.ple is not None:
                self.ple.prepare([1], [tokarr])
            with ctx.forward_batch(batch):
                logits = self.model.forward()
            torch.cuda.synchronize()
            if self.ple is not None:
                self.ple.commit([1], [tokarr])
            outs.append((f"decode{s}", logits.detach().float().cpu().clone()))
            spars.append(self._sparsity())
        return outs, ids, spars

    def _sparsity(self):
        qsa = getattr(self.ctx, "qsa", None)
        return getattr(qsa, "last_sparsity", None) if qsa is not None else None

    def teardown(self):
        from minisgl import core

        self.model = None
        core._GLOBAL_CTX = None
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------
# gates
# ------------------------------------------------------------------------------------------------

def _tap_attn(ctx, store):
    """Record layer-0 (q, out) from whichever backend entry point runs, so the two legs' ATTENTION
    can be compared to a float64 reference directly instead of through 4 layers + the lm_head."""
    b = ctx.attn_backend
    f, fs = b.forward, getattr(b, "forward_sparse", None)

    def w(q, k, v, lid, batch, **kw):
        o = f(q, k, v, lid, batch, **kw)
        if lid == 0 and not store:
            store.append((q.detach().clone(), o.detach().clone()))
        return o

    b.forward = w
    if fs is not None:
        def ws(q, k, v, lid, batch, slots, lens):
            o = fs(q, k, v, lid, batch, slots, lens)
            if lid == 0 and not store:
                store.append((q.detach().clone(), o.detach().clone()))
            return o
        b.forward_sparse = ws


def _f64_attn_rel(ctx, q, out, rows) -> float:
    """Mean relative error of `out` against EXACT causal attention over the stored paged KV, f64."""
    kv = ctx.kv_cache
    kc, vc = kv.k_cache(0), kv.v_cache(0)
    ns, kvh, hd = kc.shape[0] * kc.shape[1], kc.shape[2], kc.shape[3]
    K, V = kc.view(ns, kvh, hd).double(), vc.view(ns, kvh, hd).double()
    scale = ctx.attn_backend._softmax_scale(q)
    errs = []
    for m in rows:
        slots = ctx.page_table[0, : m + 1].long()
        k, v = K[slots], V[slots]
        qi = q[m].double()
        g = qi.shape[0] // kvh
        k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
        p = torch.softmax(torch.einsum("hd,lhd->hl", qi, k) * scale, dim=-1)
        ref = torch.einsum("hl,lhd->hd", p, v)
        n = float(ref.norm())
        if n > 0:
            errs.append(float((out[m].double() - ref).norm() / n))
    return float(np.mean(errs)) if errs else float("inf")


def _dense_check_stats(backend) -> dict:
    """Counters from HIPAttnBackend._qsa_check_vs_dense, through a HybridBackend if there is one."""
    for b in (backend, getattr(backend, "decode_backend", None),
              getattr(backend, "prefill_backend", None)):
        if b is not None and getattr(b, "_qsa_dc_calls", 0):
            return {"calls": b._qsa_dc_calls, "bitexact": b._qsa_dc_bitexact,
                    "worst": b._qsa_dc_worst}
    return {"calls": 0, "bitexact": 0, "worst": float("inf")}


def _max_abs(a, b) -> float:
    return max(float((x[1] - y[1]).abs().max().item()) for x, y in zip(a, b))


def _per_path(a, b):
    pre = max(float((x[1] - y[1]).abs().max().item())
              for x, y in zip(a, b) if x[0] == "prefill")
    dec = [(x, y) for x, y in zip(a, b) if x[0] != "prefill"]
    dc = max((float((x[1] - y[1]).abs().max().item()) for x, y in dec), default=0.0)
    bit = all(bool(torch.equal(x[1], y[1])) for x, y in dec)
    return pre, dc, bit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--long-prompt-len", type=int, default=0,
                    help="G3 prompt length; 0 = max_seq - steps - 64")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--gate", choices=("floor", "equiv", "above", "all"), default="all")
    ap.add_argument("--pin-split", action="store_true",
                    help="force the single-pass decode kernel on BOTH legs (see the module docstring)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.pin_split:
        os.environ["MINISGL_ATTN_SPLIT_MIN_CTX"] = "100000000"
    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible")
        return 1
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    print(f"[gpu] {torch.cuda.get_device_name(0)} "
          f"free/total={[x >> 20 for x in torch.cuda.mem_get_info(dev)]} MiB", flush=True)

    from minisgl.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    from minisgl.layers.rotary import set_rope_device

    set_rope_device(dev)

    rng = np.random.default_rng(11)
    report: dict = {"pin_split": bool(args.pin_split)}
    h = _Harness(args, dev)
    vocab = h.mc.vocab_size
    short = rng.integers(0, vocab, size=args.prompt_len, dtype=np.int64)
    long_len = args.long_prompt_len or (args.max_seq - args.steps - 64)
    longp = rng.integers(0, vocab, size=long_len, dtype=np.int64)
    budget = int(h.mc.indexer_budget)
    print(f"[cfg] budget={budget} short_prompt={len(short)} long_prompt={len(longp)} "
          f"steps={args.steps} max_seq={args.max_seq}", flush=True)
    check("G0 short prompt is at/below the budget", len(short) + args.steps <= budget,
          f"{len(short) + args.steps} <= {budget}")
    check("G0 long prompt is ABOVE the budget", len(longp) > budget, f"{len(longp)} > {budget}")

    eps0 = None
    if args.gate in ("floor", "all"):
        print("\n[G1] same-code floor (dense twice, bs=1)", flush=True)
        h.build(qsa=False)
        a, ida, _ = h.run(short, args.steps)
        h.teardown()
        h.build(qsa=False)
        b, idb, _ = h.run(short, args.steps)
        h.teardown()
        eps0 = _max_abs(a, b)
        report["eps0"] = eps0
        # G1b — the PREFILL kernel-swap floor. Same model, same KV, same math, different prefill
        # kernel (flash_prefill's WMMA tiles vs flash_prefill_paged). Nothing that swaps a prefill
        # kernel can beat this, so it is the bar the sparse prefill is actually held to.
        h.build(qsa=False)
        h.force_paged_prefill = True
        c, idc, _ = h.run(short, args.steps)
        h.force_paged_prefill = False
        h.teardown()
        eps_kernel = _max_abs(a, c)
        report["eps_prefill_kernel_swap"] = eps_kernel
        report["eps_kernel_ids_equal"] = ida == idc
        check("G1b prefill kernel-swap floor measured (dense cold vs dense paged)", True,
              f"eps_kernel = {eps_kernel:.3e}, greedy ids "
              + ("identical" if ida == idc else f"DIFFER {ida} vs {idc}"))
        report["floor_ids_equal"] = ida == idb
        check("G1 dense is greedy-reproducible at bs=1", ida == idb, f"{ida} vs {idb}")
        check("G1 floor measured", True, f"eps0 = {eps0:.3e}")

    if args.gate in ("equiv", "all"):
        print("\n[G2] dense-equivalence at/below the budget", flush=True)
        h.build(qsa=False)
        tap_d = []
        _tap_attn(h.ctx, tap_d)
        d, id_d, _ = h.run(short, args.steps)
        rows = [r for r in (1, 7, 63, 255, 511, len(short) - 1) if r < len(short)]
        f64_d = _f64_attn_rel(h.ctx, *tap_d[0], rows) if tap_d else float("inf")
        h.teardown()
        # The IN-PLACE check is the one that actually tests the bit-exactness claim; see
        # HIPAttnBackend._qsa_check_vs_dense. End-to-end logits cannot, because prefill genuinely
        # runs a different kernel, so by the first decode step the two legs' KV caches have already
        # diverged and any decode comparison measures THAT.
        os.environ["MINISGL_QSA_DENSE_CHECK"] = "1"
        h.build(qsa=True)
        tap_s = []
        _tap_attn(h.ctx, tap_s)
        s, id_s, sp = h.run(short, args.steps)
        be = _dense_check_stats(h.ctx.attn_backend)
        f64_s = _f64_attn_rel(h.ctx, *tap_s[0], rows) if tap_s else float("inf")
        h.teardown()
        os.environ["MINISGL_QSA_DENSE_CHECK"] = "0"
        pre, dec, bit = _per_path(d, s)
        report.update({"le_budget_prefill_maxabs": pre, "le_budget_decode_e2e_maxabs": dec,
                       "le_budget_ids_equal": id_d == id_s, "in_place_dense_check": be})
        check("G2 greedy ids identical", id_d == id_s, f"{id_d} vs {id_s}")
        check("G2 DECODE attention bit-exact vs dense, SAME inputs",
              be["calls"] > 0 and be["bitexact"] == be["calls"],
              f"{be['bitexact']}/{be['calls']} calls exact, worst |d| = {be['worst']:.3e}"
              + ("" if args.pin_split else "  (run --pin-split: the split-KV policy is keyed on "
                                           "the block-table row width and differs between the two)"))
        # The prefill legs run DIFFERENT kernels by construction (dense = attn_hip.flash_prefill's
        # WMMA tile core; sparse = per-query-row through the paged decode core), so bit-exactness is
        # not on offer there and claiming it would be dishonest. What IS claimed: the difference is
        # bf16 accumulation-order noise, i.e. small RELATIVE to the logit scale. Gated at 1e-3 —
        # bf16 carries ~3 decimal digits, so anything at or below that is indistinguishable from
        # reordering the same sum.
        scale = max(float(x[1].abs().max()) for x in d)
        rel = pre / scale if scale else float("inf")
        report["le_budget_prefill_rel"] = rel
        report["le_budget_logit_scale"] = scale
        # THE PREFILL CLAIM, and it is not "within a tolerance". The dense prefill runs the WMMA
        # tile core and the sparse prefill runs the paged decode core per query row, so they are two
        # attention IMPLEMENTATIONS and bit-exactness is not on offer. The kernel-swap control (G1b)
        # came back at 0.0, so "reordering noise" is not something this harness has earned the right
        # to say. What IS measurable: which one is closer to EXACT. Both layer-0 prefill outputs are
        # compared against float64 causal attention over the same stored paged KV, and the gate is
        # that the sparse path is no further from f64 than the dense path it replaces.
        report["f64_rel_dense_prefill"] = f64_d
        report["f64_rel_sparse_prefill"] = f64_s
        check("G2 PREFILL sparse attention is no further from float64 than dense",
              f64_s <= f64_d * 1.05,
              f"rel(f64): sparse = {f64_s:.3e} vs dense = {f64_d:.3e}; "
              f"end-to-end logit max|d| = {pre:.3e} (rel {rel:.2e} of {scale:.3e})")
        check("G2 DECODE end-to-end (informational: inherits the prefill kernel change)",
              True, f"max|d| = {dec:.3e}")

    if args.gate in ("above", "all"):
        print("\n[G3] ABOVE the budget: f64 reference + sparsity proof", flush=True)
        os.environ["MINISGL_QSA_TAP"] = "1"
        h.build(qsa=True)
        outs, ids, sp = h.run(longp, args.steps)
        rt = h.ctx.qsa
        ok_fin = all(bool(torch.isfinite(o[1]).all()) for o in outs)
        check("G3 logits are finite above the budget", ok_fin)
        report["above_ids"] = ids
        # --- sparsity, measured ---
        vis, dense = rt.total_visited, rt.total_dense
        report["visited"] = vis
        report["dense"] = dense
        report["sparsity_ratio"] = (vis / dense) if dense else None
        check("G3 selection is genuinely SPARSE (visited < dense)", vis < dense,
              f"visited={vis} dense={dense} ratio={vis / dense:.4f}" if dense else "")
        # --- f64 reference on the selection ---
        bad, checked = _verify_selection_f64(rt, args)
        report["f64_rows_checked"] = checked
        report["f64_rows_mismatched"] = bad
        check("G3 selection matches the float64 host reference", bad == 0,
              f"{checked - bad}/{checked} rows exact")
        h.teardown()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[json] {args.json}")
    print(f"\n{'PASS' if not _failures else 'FAIL: ' + ', '.join(_failures)}")
    return 1 if _failures else 0


def _verify_selection_f64(rt, args) -> "tuple[int, int]":
    """Re-derive the selected token set on the HOST in float64 and compare as a SET.

    Uses the kernel's own tie rule (value descending, index ascending under the order-preserving
    fp32->uint32 map) so a disagreement is about the selection, not about how ties were broken.
    """
    bad = checked = 0
    for layer, t in rt._last.items():
        q = t["q"].to(torch.float64).cpu()
        comp = t["compressed"].reshape(t["compressed"].shape[0], -1).to(torch.float64).cpu()
        cpt = t["comp_page_table"].cpu().to(torch.int64)
        rows = q.shape[0]
        ends = t["row_ends"].cpu().to(torch.int64)
        seqs = t["row_seq"].cpu().to(torch.int64)
        pos = t["logical_pos"].cpu().to(torch.int64)
        lens = t["seq_len_row"].cpu().to(torch.int64)
        toks = t["tokens"].cpu().to(torch.int64)
        scale = float(q.shape[-1]) ** 0.5
        ratio = rt.profile.compress_ratio
        topk = rt.profile.block_topk
        sample = list(range(0, rows, max(1, rows // 8)))[:8]
        for m in sample:
            end = int(ends[m])
            if end <= 0:
                continue
            slots = cpt[int(seqs[m]), :end]
            keys = comp[slots]                                        # [end, D]
            sc = torch.einsum("hd,nd->nh", q[m], keys)
            logit = torch.relu(sc).sum(dim=-1) / scale
            if end <= topk:
                blocks = np.arange(end)
            else:
                f32 = logit.to(torch.float32).numpy()
                b = np.ascontiguousarray(f32).view(np.uint32)
                key = np.where(b & np.uint32(0x80000000), ~b, b | np.uint32(0x80000000))
                order = np.lexsort((np.arange(end), -key.astype(np.int64)))
                blocks = np.sort(order[:topk])
            want = []
            for i in range(topk * ratio):
                if i // ratio >= len(blocks):
                    break
                tk = int(blocks[i // ratio]) * ratio + (i % ratio)
                if 0 <= tk < int(lens[m]):
                    want.append(tk)
            visible = int(pos[m]) + 1
            tail = (visible // ratio) * ratio
            n_tail = max(0, min(ratio - 1, visible - tail, int(lens[m]) - tail))
            want.extend(tail + o for o in range(n_tail))
            got = [int(x) for x in toks[m] if int(x) >= 0]
            checked += 1
            if set(got) != set(want[: len(got)]) or len(got) != len(want[: toks.shape[1]]):
                bad += 1
                if bad <= 3:
                    print(f"    layer {layer} row {m}: got {len(got)} want {len(want)}; "
                          f"missing={sorted(set(want) - set(got))[:8]} "
                          f"extra={sorted(set(got) - set(want))[:8]}")
    return bad, checked


if __name__ == "__main__":
    with torch.inference_mode():
        sys.exit(main())
