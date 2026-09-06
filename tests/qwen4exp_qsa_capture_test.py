"""QSA under CUDAGRAPH CAPTURE — identity gates, not a "capture did not throw" smoke test.

THE THREE THINGS THAT HAVE TO BE TRUE, and why each is a separate gate
----------------------------------------------------------------------
Capture on this path is not "it ran". `QSARuntime.prepare` used to raise a NAMED
NotImplementedError, so a QSA build decoded EAGER and the shipped arm carried GRAPH_BS=0. Making it
capture is easy; making it capture the SAME COMPUTATION is the work, and this repo has already
shipped the failure once (61d96cf / 0972e387: the capture width silently selected a different
kernel). So:

  C0  SAME-CODE FLOOR, first. Two identical eager builds, same seed, same trajectory, bs=1 with
      MINISGL_MOE_G2FUSE=0 (the decode gemm2 atomic scatter is documented non-bit-exact and a
      bs>1 or fused leg would be measuring THAT). eps0 must be 0.0. Nothing below the floor is
      evidence about anything, and this exact discipline already caught a harness returning
      eps0=3.4 on this work.

  C1  CAPTURED == EAGER, BIT FOR BIT. Same build procedure, same weights, same prompt, same greedy
      trajectory; one leg replays a captured bs=1 decode graph, the other runs the identical static
      path eagerly. `torch.equal` on every decode step's logits, and identical greedy ids.

  C2  THE SAME KERNEL, AND THE SAME GRID. Every launch policy the selection can pick is recorded on
      both legs and compared as a tuple:
        * the scorer's grid is `(ceil(rows/ROWS), ceil(num_key_cols/256))` — `num_key_cols` is the
          logits tensor's STATIC width, so it must be identical, and identical to `max_blocks`;
        * the split top-k's `num_splits` comes from `qsa_index_topk_num_splits(static_width, rows)`,
          the kernel's OWN host-side introspection entry point (not a re-derivation), so a leg that
          took the one-CTA path and a leg that took the split path cannot be confused for each other;
        * the sparse attention's block-table width is `index_width` (2051) on both legs, which is
          what dissolves the split-K width-inference landmine on this path.
      And the SELECTION ITSELF is compared: the static `blocks`/`tokens`/`slots` buffers are the
      exact device tensors the captured graph writes through its baked pointers, so reading them
      after a replay and after an eager step compares the selected KV sets directly rather than
      inferring equality from the logits.

  C3  THE LEDGER SURVIVED. `visited < dense` with a real ratio on the >budget leg. A "sparse" path
      that quietly selected everything passes C1 and C2 perfectly.

Both a <=budget prompt (where the selection provably degenerates to dense causal attention) and a
>budget prompt (where it genuinely selects) are run, because they exercise different regimes of the
top-k and of the expand.

RUN (one card, 4-layer subset, ~3 min):
    gpu-lease -n 1 -- docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES -e HIP_VISIBLE_DEVICES \
      -v <worktree>:/engine -v <kernels>:/kern:ro -v /home/pat/.cache/hf-q4e:/model:ro \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'MINISGL_MOE_G2FUSE=0 PYTHONPATH=/engine/python:/kern/qsa_index/torch-ext:/opt/kernels \
       python /engine/tests/qwen4exp_qsa_capture_test.py'
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen4exp_qsa_gate_test import _Harness, _make_batch, _make_req, check  # noqa: E402
import qwen4exp_qsa_gate_test as _gate  # noqa: E402


def _policy_probe(ctx, rows: int) -> dict:
    """Every LAUNCH POLICY the selection can pick, read from the live static buffers + the kernel's
    own introspection op. Nothing here is re-derived from the engine's idea of the shapes."""
    qsa = ctx.qsa
    s = qsa._s
    cols = int(s["logits"].shape[1])
    out = {
        "max_blocks": int(qsa.static_max_blocks),
        "logits_cols": cols,
        "logits_rows": int(s["logits"].shape[0]),
        "score_grid": [rows, (cols + 255) // 256],
        "blocks_width": int(s["blocks"].shape[1]),
        "tokens_width": int(s["tokens"].shape[1]),
        "slots_width": int(s["slots"].shape[1]),
        "ops_backend": __import__("minisgl.attention.qsa.ops", fromlist=["backend"]).backend(),
    }
    try:
        import qsa_index

        out["topk_num_splits"] = int(qsa_index.qsa_index_topk_num_splits(cols, rows))
        out["kernel_so"] = getattr(qsa_index, "__file__", "?")
    except Exception as exc:  # noqa: BLE001
        out["topk_num_splits"] = f"unavailable: {exc}"
    return out


def _sel_snapshot(ctx):
    """The LAST index layer's selection, straight out of the static buffers the captured graph
    writes through. Cloned so the next step cannot overwrite it under us."""
    s = ctx.qsa._s
    return {
        "blocks": s["blocks"].detach().clone().cpu(),
        "tokens": s["tokens"].detach().clone().cpu(),
        "slots": s["slots"].detach().clone().cpu(),
        "lens": s["lens"].detach().clone().cpu(),
    }


class _CaptureHarness(_Harness):
    """`_Harness` plus: arm the static plan, and (optionally) run decode through a REAL GraphRunner.

    The captured leg deliberately uses `minisgl.engine.graph.GraphRunner` rather than a hand-rolled
    `torch.cuda.graph` block, because the thing under test is the engine's capture path — the
    attention/GDN/PLE static metadata, the model's `prepare_for_replay` hook, and the padded dummy
    row — not the CUDA API.
    """

    def arm(self, max_bs: int) -> None:
        self.ctx.qsa.init_capture(max_bs, self.args.max_seq)

    def make_graph_runner(self, max_bs: int):
        from minisgl.core import Req, SamplingParams
        from minisgl.engine.graph import GraphRunner

        dev = self.dev
        # The dummy padding row lives in page-table row 1 and points at a page no request owns.
        dummy_page = self.page_table.shape[1] + self.args.max_seq  # past every real slot
        self.page_table[1].fill_(0)
        dummy = Req(
            input_ids=torch.zeros(1, dtype=torch.int32, device="cpu"),
            table_idx=1, cached_len=0, output_len=1, uid=-1,
            sampling_params=SamplingParams(), cache_handle=None,  # type: ignore
        )
        del dummy_page
        return GraphRunner(
            stream=torch.cuda.Stream(device=dev), device=dev, model=self.model,
            attn_backend=self.ctx.attn_backend, cuda_graph_bs=[max_bs], cuda_graph_max_bs=None,
            free_memory=torch.cuda.mem_get_info(dev)[0], max_seq_len=self.args.max_seq,
            vocab_size=self.mc.vocab_size, dummy_req=dummy, gdn_state=self.ctx.gdn_state,
            ple=self.ple,
        )

    def run_traj(self, prompt: np.ndarray, steps: int, *, captured: bool):
        """PREFILL eager (always — prefill is eager everywhere in this engine), then `steps` greedy
        decodes either through a graph replay or through the identical static eager path."""
        from minisgl.gdn.metadata import build_gdn_metadata

        dev, ctx = self.dev, self.ctx
        req = _make_req(prompt, 0, out_len=steps + 1)
        batch = _make_batch([req], "prefill", self.page_table, dev)
        ctx.attn_backend.prepare_metadata(batch)
        batch.gdn_metadata = build_gdn_metadata(
            batch, torch.tensor([1], dtype=torch.int32, device=dev), dev
        )
        if self.ple is not None:
            self.ple.prepare([1], [prompt])
        with ctx.forward_batch(batch):
            logits = self.model.forward()
        torch.cuda.synchronize()
        if self.ple is not None:
            self.ple.commit([1], [prompt])

        gr = self.make_graph_runner(1) if captured else None
        if not captured:
            self.arm(1)
        outs, ids, sels, pol = [], [], [], None
        for s in range(steps):
            nxt = int(logits[-1].float().argmax().item())
            ids.append(nxt)
            req.append_host(torch.tensor([nxt], dtype=torch.int64))
            req.complete_one()
            batch = _make_batch([req], "decode", self.page_table, dev)
            batch.padded_reqs = [req]
            tokarr = np.array([nxt], dtype=np.int64)
            if self.ple is not None:
                self.ple.prepare([1], [tokarr])
            ctx.attn_backend.prepare_metadata(batch)
            batch.gdn_metadata = build_gdn_metadata(
                batch, torch.tensor([1], dtype=torch.int32, device=dev), dev
            )
            if gr is not None:
                # Exactly the scheduler's sequence: pad, stage the I/O into the static buffers,
                # repoint the batch at them, replay. `gr.replay` runs attn/GDN/PLE
                # `prepare_for_replay` and `model.prepare_for_replay` (which is where QSA's static
                # decode plan is refreshed) before it launches the graph.
                gr.pad_batch(batch)
                gr.buffer.copy_from(batch)
                gr.buffer.set_batch(batch)
                with ctx.forward_batch(batch):
                    logits = gr.replay(batch)
            else:
                with ctx.forward_batch(batch):
                    logits = self.model.forward()
            torch.cuda.synchronize()
            if self.ple is not None:
                self.ple.commit([1], [tokarr])
            if pol is None:
                pol = _policy_probe(ctx, rows=1)
            outs.append((f"decode{s}", logits.detach().float().cpu().clone()))
            sels.append(_sel_snapshot(ctx))
        prov = {
            "static_steps": int(ctx.qsa.static_steps),
            "captured_steps": int(ctx.qsa.captured_steps),
            "graph_replays": int(gr.replays) if gr is not None else 0,
            "visited_dense": list(ctx.qsa.sparsity_totals()),
        }
        self._gr = gr
        return outs, ids, sels, pol, prov


def _bit_equal(a, b) -> bool:
    return all(bool(torch.equal(x[1], y[1])) for x, y in zip(a, b))


def _max_abs(a, b) -> float:
    return max(float((x[1] - y[1]).abs().max().item()) for x, y in zip(a, b))


def _sel_equal(sa, sb) -> "tuple[bool, str]":
    for i, (x, y) in enumerate(zip(sa, sb)):
        for k in ("blocks", "tokens", "slots", "lens"):
            if not torch.equal(x[k], y[k]):
                d = int((x[k] != y[k]).sum())
                return False, f"step {i} field {k}: {d} differing entries"
    return True, ""


def _leg(args, dev, prompt, *, captured: bool):
    """One leg, plus its ENGAGED LEDGER AS COUNTS.

    The ledger is a SET and a set saturates: after the first leg of a two-leg A/B in one process a
    set-diff is empty by construction for BOTH legs, so it cannot tell "the two legs dispatched the
    same arms" from "leg B dispatched nothing at all". `counts_delta` is the same ledger as a tally,
    which is differenceable.

    The two legs' counts are NOT expected to be equal and comparing the numbers would be the
    mistake: a captured leg runs host python once (the capture pass) and then replays, so its tally
    is ~one forward's worth against the eager leg's `steps`. What must match is the arm SET — an arm
    that dispatches on eager and vanishes under capture is a silent fallback (the HIP
    `qsa_index.topk` quietly becoming `qsa_index.topk(torch)`, say), and every logit-identity gate
    in this file would pass while it happened.
    """
    from minisgl._hip_engage import counts, counts_delta

    h = _CaptureHarness(args, dev)
    h.build(qsa=True)
    before = counts()
    out = h.run_traj(prompt, args.steps, captured=captured)
    delta = counts_delta(before)
    h.teardown()
    return (*out, delta)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_gate.MODEL)
    ap.add_argument("--layers", type=int, default=4)
    # >= num_experts_per_tok (10 in this checkpoint) or the router's torch.topk is out of range.
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--max-seq", type=int, default=16384)
    ap.add_argument("--short-prompt-len", type=int, default=1024)
    ap.add_argument("--long-prompt-len", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible")
        return 1
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    print(f"[gpu] {torch.cuda.get_device_name(0)} "
          f"free/total={[x >> 20 for x in torch.cuda.mem_get_info(dev)]} MiB", flush=True)
    print(f"[env] MINISGL_MOE_G2FUSE={os.environ.get('MINISGL_MOE_G2FUSE', '<unset>')} "
          f"(bit-identity requires 0 — the decode gemm2 atomic scatter is not bit-exact)", flush=True)

    from minisgl.distributed import set_tp_info, try_get_tp_info
    from minisgl.layers.rotary import set_rope_device

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    set_rope_device(dev)

    rng = np.random.default_rng(11)
    report: dict = {"args": vars(args)}
    h0 = _Harness(args, dev)
    vocab = h0.mc.vocab_size
    budget = int(h0.mc.indexer_budget)
    prompts = {
        "le_budget": rng.integers(0, vocab, size=args.short_prompt_len, dtype=np.int64),
        "gt_budget": rng.integers(0, vocab, size=args.long_prompt_len, dtype=np.int64),
    }
    check("prompt <=budget really is", args.short_prompt_len + args.steps <= budget,
          f"{args.short_prompt_len + args.steps} <= {budget}")
    check("prompt >budget really is", args.long_prompt_len > budget,
          f"{args.long_prompt_len} > {budget}")

    for tag, prompt in prompts.items():
        print(f"\n================ {tag} (prompt_len={len(prompt)}) ================", flush=True)
        print("[C0] same-code floor — two identical EAGER builds", flush=True)
        a, ida, sela, pola, prova, enga = _leg(args, dev, prompt, captured=False)
        b, idb, selb, polb, provb, engb = _leg(args, dev, prompt, captured=False)
        eps0 = _max_abs(a, b)
        floor_ok = check(f"{tag} C0 eps0 == 0.0", eps0 == 0.0, f"eps0 = {eps0:.3e}")
        check(f"{tag} C0 greedy ids identical", ida == idb, f"{ida} vs {idb}")
        check(f"{tag} C0 selection identical", _sel_equal(sela, selb)[0])

        print("[C1] captured vs eager, same build procedure", flush=True)
        c, idc, selc, polc, provc, engc = _leg(args, dev, prompt, captured=True)
        eps = _max_abs(a, c)
        # PROVENANCE FIRST: a green comparison of two legs that took the same path is worthless.
        check(f"{tag} PROVENANCE eager leg replayed NO graph",
              prova["graph_replays"] == 0 and prova["captured_steps"] == 0, json.dumps(prova))
        check(f"{tag} PROVENANCE captured leg replayed EVERY decode step",
              provc["graph_replays"] == args.steps and provc["captured_steps"] >= 1,
              json.dumps(provc))
        check(f"{tag} C1 captured == eager BIT FOR BIT", _bit_equal(a, c), f"max|d| = {eps:.3e}")
        check(f"{tag} C1 greedy ids identical", ida == idc, f"{ida} vs {idc}")

        print("[C2] same kernel, same grid, same selection", flush=True)
        check(f"{tag} C2 launch policies identical", pola == polc,
              f"\n      eager   {json.dumps(pola)}\n      captured{json.dumps(polc)}")
        ok, why = _sel_equal(sela, selc)
        check(f"{tag} C2 selected blocks/tokens/slots identical", ok, why)
        check(f"{tag} C2 max_blocks is the STATIC ceil(max_seq/r), not this step's window",
              pola["max_blocks"] == (args.max_seq + 3) // 4,
              f"{pola['max_blocks']} vs {(args.max_seq + 3) // 4}")
        check(f"{tag} C2 sparse attention width is index_width (fixed at every context length)",
              pola["slots_width"] == budget + 3, f"{pola['slots_width']} vs {budget + 3}")

        print("[C4] the engaged ledger, per leg, as COUNTS", flush=True)
        # An arm that fired on the eager leg and fired ZERO times under capture is a silent dispatch
        # regression: the selection would have fallen back to `MINISGL_QSA_OPS=torch`, or an arm
        # would have been gated off inside the capture, and C1/C2 would both still be green because
        # the torch reference computes the same values. The NUMBERS are deliberately not compared
        # (see `_leg`): a replayed step re-enters no python at all.
        missing = sorted(set(enga) - set(engc))
        added = sorted(set(engc) - set(enga))
        check(f"{tag} C4 no arm vanished under capture", not missing,
              f"eager-only arms: {missing}")
        check(f"{tag} C4 no NEW arm appeared under capture (a fallback would show up here)",
              not added, f"captured-only arms: {added}")
        check(f"{tag} C4 the three qsa_index HIP arms dispatched on BOTH legs",
              all(f"qsa_index.{k}" in enga and f"qsa_index.{k}" in engc
                  for k in ("score_paged", "topk", "expand")),
              f"eager={sorted(k for k in enga if k.startswith('qsa_index'))} "
              f"captured={sorted(k for k in engc if k.startswith('qsa_index'))}")
        check(f"{tag} C4 the TORCH selection fallback never dispatched on either leg",
              not any(k.endswith("(torch)") for k in (*enga, *engc)),
              f"{[k for k in (*enga, *engc) if k.endswith('(torch)')]}")

        print("[C3] the sparsity ledger survived capture", flush=True)
        v, d = provc["visited_dense"]
        ratio = (v / d) if d else float("nan")
        check(f"{tag} C3 ledger is live", v > 0 and d > 0, f"visited={v} dense={d} ratio={ratio:.4f}")
        if tag == "gt_budget":
            check(f"{tag} C3 the path is genuinely SPARSE", v < d, f"ratio = {ratio:.4f}")
        else:
            check(f"{tag} C3 at/below budget the selection is DENSE (visited == dense)", v == d,
                  f"visited={v} dense={d}")
        report[tag] = {
            "eps0": eps0, "eps_capture": eps, "bit_equal": _bit_equal(a, c),
            "ids_eager": ida, "ids_captured": idc,
            "policy_eager": pola, "policy_captured": polc,
            "prov_eager": prova, "prov_captured": provc,
            "selection_equal": ok, "floor_ok": floor_ok,
            "visited": v, "dense": d, "sparsity": ratio,
            # Per-leg dispatch tallies. Kept in full rather than summarised: the arms are the
            # evidence, and a future reader diffing a regression needs the names, not a verdict.
            "engaged_eager": enga, "engaged_eager2": engb, "engaged_captured": engc,
        }

    print()
    report["failures"] = _gate._failures
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[json] {args.json}")
    if _gate._failures:
        print(f"FAILED: {len(_gate._failures)} — {_gate._failures}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
