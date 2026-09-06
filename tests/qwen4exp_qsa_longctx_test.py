"""LONG-CONTEXT gate for the qwen4_exp QSA sparse-attention path (T5 stage 5).

WHAT THIS FILE IS FOR, and why it is not the dense-equivalence gate.
`tests/qwen4exp_qsa_gate_test.py` proves the sparse path reproduces dense attention BIT-FOR-BIT at
or below `indexer_budget` (2048), on a 4-layer random-weight subset. That gate is exact and free,
and it says nothing at all about the regime the feature exists for: a REAL 48-layer checkpoint,
TP=2, at a context length where the selection actually throws keys away. This file is that half.

The claims it is built to support, each with the counter-evidence it would produce if false:

  * REACH. What context does the engine ACCEPT? The bound is not the checkpoint's 262,144 — it is
    `Engine.__init__`'s `max_seq_len = min(config.max_seq_len, num_tokens)`, i.e. the KV POOL. The
    ladder walks up until the engine refuses and the refusal is RECORDED VERBATIM, because "we
    served 16k" and "16k is the ceiling" are different statements and only the second one needs a
    failure next to it.
  * COHERENCE, SAMPLED. temperature 1.0 / top_k 20 / top_p 0.95 — the checkpoint's own
    generation_config, never greedy (CLAUDE.md: temperature 0 fakes degeneration that mimics a
    quant bug). Judged against the four degeneration signatures, with the mechanical part of that
    judgement computed here (loop rate, 1-char-token run, whitespace-free run) rather than left to
    a human reading a paragraph.
  * RETRIEVAL, which is the signature a sparse-attention bug actually produces. A wrong selection
    does not emit noise — it emits fluent text that has FORGOTTEN the early half of the prompt. So
    a needle is planted at a known FRACTION of the prompt and the answer is checked literally. Two
    needles: one at the very start (the block QSA is most likely to drop) and one near the end (the
    control — if the late needle also fails, the failure is the model or the harness, not the
    selection).
  * SPARSITY IS MEASURED PER RUN. `QSARuntime.total_visited / total_dense` is snapshotted around
    each generate. A "sparse" path that quietly selected everything passes every coherence and
    retrieval test in this file; the ratio is the only thing that catches it, and it is reported
    even on the runs that pass.
  * PREFILL AND DECODE ARE SEPARATE NUMBERS. `STEP_LOG` tags every forward `prefill` /
    `decode_eager` / `decode_graph`, so prefill tok/s comes from the prefill steps' own wall time
    and decode tok/s from the median decode step. A `len(tokens)/wall` figure would tax every token
    with a 64k prefill and is not a decode number.

CAPTURE IS NOW A LEG, NOT A REFUSAL (2026-09-06). `QSARuntime.prepare` used to raise inside a
cudagraph capture, so this harness hard-wired `cuda_graph_max_bs=0` and ASSERTED the decode steps
were eager. The selection now has a static decode plan, so `--graph-bs` selects the leg and the
assert became a CHECK OF WHAT WAS ASKED FOR: `--graph-bs 0` still demands `decode_eager` on every
decode step, `--graph-bs N` demands `decode_graph`. A leg that silently ran the other dispatch is
the one thing an A/B of the two must not be able to do, so the per-rung phase set is recorded and
gated either way, alongside the engine's own replay/eager COUNTERS (never the engaged() set, which
saturates).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from typing import List

import torch

MODEL = "/model"

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    if not ok:
        _failures += 1
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f" != {want!r}"),
          flush=True)


def check_true(name: str, cond, detail: str = "") -> None:
    global _failures
    if not cond:
        _failures += 1
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


# ---------------------------------------------------------------------------------------------
# The haystack.
# ---------------------------------------------------------------------------------------------
# Filler that is REAL PROSE, not repeated tokens. A haystack of one repeated sentence is a
# pathological input for a top-k selector: every compressed block scores identically, ties dominate
# the selection, and the result measures the tie rule rather than the retrieval. Numbered, varied
# sentences give the selector genuinely distinguishable blocks — and the numbering makes it possible
# to say WHERE in the prompt an answer came from.
_FILLER = [
    "The survey team recorded the soil moisture at station {i} and noted the reading was within tolerance.",
    "Log entry {i}: the northern conveyor ran for the full shift without an unscheduled stop.",
    "In paragraph {i} the auditor observes that the inventory count matched the ledger to the unit.",
    "Item {i} of the maintenance schedule concerns the quarterly inspection of the coolant lines.",
    "Field note {i} describes a light frost overnight, clearing before the first collection round.",
    "Section {i} restates the calibration procedure for the handheld conductivity meter.",
    "Observation {i}: the flow rate held steady through the afternoon despite the change in pressure.",
    "Appendix line {i} lists the serial numbers issued to the eastern depot during the period.",
]

_NEEDLE_EARLY = ("The Meridian vault access code is 47-ZEBRA-9183, and it must never be written "
                 "in the shared log.")
_NEEDLE_LATE = ("The night supervisor on the eastern depot roster is Dr. Imogen Halloway, "
                "reachable on extension 6120.")
_QUESTION = ("\n\nQuestion: Read the document above and answer with the facts it contains. "
             "What is the Meridian vault access code, and who is the night supervisor on the "
             "eastern depot roster? Answer in one short sentence.")


def _needle_lines(n_sent: int, early_frac: float, late_frac: float):
    e = max(1, min(n_sent - 2, int(n_sent * early_frac)))
    l = max(e + 1, min(n_sent - 1, int(n_sent * late_frac)))
    return e, l


def build_haystack(tokenizer, target_tokens: int, early_frac: float, late_frac: float,
                   chat: bool = True, seed_offset: int = 0):
    """A prompt of ~`target_tokens` tokens with two needles at known fractional depths.

    THE PROMPT IS CHAT-TEMPLATED and the length is measured on the TEMPLATED string, not on the raw
    document. Two reasons, both of which would otherwise silently change what is being tested:
    the checkpoint is an instruct model, so a raw document ending in "Question:" is answered by
    CONTINUING the document rather than by answering, and a retrieval verdict taken off a
    continuation measures the prompt format; and the template's own wrapper is worth ~30 tokens,
    which is noise at 16k but not at the length where the engine refuses.

    `enable_thinking=False`: the answer has to be a short factual sentence for the literal needle
    check to mean anything, and a thinking budget spent inside `--gen-tokens` produces an empty
    answer that reads as a retrieval failure. This is a HARNESS choice, not a serving one.

    Length is hit by BINARY SEARCH over the filler-sentence count. The ids are NEVER truncated —
    truncation would cut the generation prompt off the end — so the achieved length lands within
    one filler sentence of the target and `actual_tokens` is what every number is reported against.
    """
    # `seed_offset` renumbers every filler line. It exists ONLY for the concurrency leg: two
    # IDENTICAL prompts share a radix prefix, so a width-2 leg of copies measures the prefix cache
    # rather than two concurrent long contexts — and it would also halve the KV the leg is supposed
    # to be stressing. Renumbering makes the two token streams differ at line 1.
    def doc(n_sent: int) -> str:
        lines = [_FILLER[i % len(_FILLER)].format(i=i + 1 + seed_offset * 1_000_000)
                 for i in range(n_sent)]
        e, l = _needle_lines(n_sent, early_frac, late_frac)
        lines[e] = _NEEDLE_EARLY
        lines[l] = _NEEDLE_LATE
        return "\n".join(lines)

    def render(n_sent: int) -> str:
        body = doc(n_sent) + _QUESTION
        if not chat:
            return body
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": body}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)

    def ntok(n_sent: int) -> int:
        return len(tokenizer.encode(render(n_sent), add_special_tokens=False))

    lo, hi = 8, 16
    while ntok(hi) < target_tokens:
        lo, hi = hi, hi * 2
        if hi > 4_000_000:
            raise RuntimeError("haystack search diverged")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ntok(mid) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    ids = tokenizer.encode(render(lo), add_special_tokens=False)
    e_line, l_line = _needle_lines(lo, early_frac, late_frac)
    return ids, {
        "target_tokens": target_tokens,
        "actual_tokens": len(ids),
        "filler_sentences": lo,
        "chat_templated": bool(chat),
        "early_needle_line": e_line,
        "late_needle_line": l_line,
        "early_needle_depth_frac": round(e_line / lo, 4),
        "late_needle_depth_frac": round(l_line / lo, 4),
    }


# ---------------------------------------------------------------------------------------------
# Degeneration signatures (docs: degeneration-signatures-triage-table).
# ---------------------------------------------------------------------------------------------
def degeneration_report(text: str, token_ids: List[int]) -> dict:
    """The mechanical half of the coherence verdict. Four signatures, four different bugs.

    Deliberately NOT a pass/fail on its own — it is evidence printed next to the text, because a
    short correct answer can trip a naive repetition threshold and a fluent hallucination trips
    none of them. The retrieval check is what catches the sparse-attention failure mode; this
    catches the ones that look like a quant or sampler bug.
    """
    words = re.findall(r"[^\W\d_]+", text.lower())
    n = len(words)
    uniq = len(set(words)) if n else 0
    # LOOP: the longest immediately-repeating n-gram run.
    loop = 0
    for k in (1, 2, 3, 4, 5, 8):
        if n < 2 * k:
            continue
        run = 0
        best = 0
        for i in range(k, n - k + 1):
            if words[i:i + k] == words[i - k:i]:
                run += 1
                best = max(best, run)
            else:
                run = 0
        loop = max(loop, best)
    # LETTER-SPELL: a run of single-character whitespace-separated pieces ("t h e").
    pieces = text.split()
    spell = cur = 0
    for p in pieces:
        cur = cur + 1 if len(p) == 1 else 0
        spell = max(spell, cur)
    # TOKEN NOISE: fraction of characters outside a plausible latin/punct set.
    bad = sum(1 for c in text if not (c.isascii() and (c.isprintable() or c in "\n\t")))
    return {
        "chars": len(text),
        "words": n,
        "distinct_word_ratio": round(uniq / n, 4) if n else 0.0,
        "longest_immediate_repeat_run": loop,
        "longest_single_char_run": spell,
        "non_ascii_char_frac": round(bad / len(text), 4) if text else 0.0,
        "tokens": len(token_ids),
    }


def _steps_since(mark: int):
    from minisgl.scheduler.scheduler import STEP_LOG
    return list(STEP_LOG)[mark:]


def _median(xs):
    s = sorted(xs)
    if not s:
        return 0.0
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def run_batch(llm, tokenizer, args, target: int, rank: int, width: int = 1) -> dict:
    """One context length at one batch width: prompt(s), generate sampled, all numbers off one run.

    `width > 1` is the CONCURRENCY leg. It is the same code path deliberately: the question "what
    context does the pool support at CONC=2" is not answered by halving a bs=1 number, because the
    scheduler may simply QUEUE the second request instead of refusing, and queueing is a latency
    result while refusing is a capacity result. Running them together is the only way to see which.
    """
    from minisgl.core import SamplingParams
    from minisgl.scheduler.scheduler import STEP_LOG, STEP_LOG_SYNC as _STEP_LOG_SYNC

    prompts, metas = [], []
    for w in range(width):
        ids, meta = build_haystack(tokenizer, target, args.early_frac, args.late_frac,
                                   chat=not args.no_chat_template, seed_offset=w)
        prompts.append(ids)
        metas.append(meta)
    meta = metas[0]
    row: dict = dict(meta)
    row["rank"] = rank
    row["batch_width"] = width
    row["batch_actual_tokens"] = [m["actual_tokens"] for m in metas]

    qsa = getattr(llm.engine.ctx, "qsa", None)
    # `sparsity_totals()` and not `total_visited`: the static (capturable) decode path accumulates
    # the ledger in DEVICE counters, because a per-layer `.item()` is a host sync a graph cannot
    # record. Reading the python attributes alone would report 0 visited on every captured leg —
    # i.e. the one instrument that proves the path is sparse would read zero exactly when capture
    # is on.
    v0, d0 = qsa.sparsity_totals() if qsa is not None else (0, 0)
    gr0 = int(getattr(getattr(llm.engine, "graph_runner", None), "replays", 0) or 0)
    eg0 = int(getattr(llm.engine, "eager_decode_forwards", 0) or 0)
    # THE ENGAGED LEDGER, PER RUNG, AS COUNTS AND NOT AS THE SET. The set saturates on the first
    # forward of the first rung, so a set-diff across rungs — or across the two legs of the
    # eager-vs-captured A/B — is empty by construction and can only ever report "nothing changed".
    # The tally says which arms actually dispatched HERE, which is what makes a vanished arm
    # visible. Read it with the caveat in `_hip_engage`: a graph replay re-enters no host python,
    # so on a captured leg these counts move at CAPTURE time and then stand still.
    from minisgl._hip_engage import counts as _eng_counts, counts_delta as _eng_delta
    eng0 = _eng_counts()
    mark = len(STEP_LOG)

    sp = SamplingParams(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                        max_tokens=args.gen_tokens)
    t0 = time.perf_counter()
    try:
        results = llm.generate(prompts, sp)
        res = results[0]
    except Exception as exc:  # noqa: BLE001 — a refusal IS the measurement here
        row["ok"] = False
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)[:600]
        # TERMINAL vs CAPACITY. A clean capacity refusal (the scheduler declining an over-length
        # request) leaves the engine usable and the ladder should continue. An OOM inside a
        # collective does NOT: the process group is poisoned and every later rung reports the same
        # NCCL error, which reads in the artifact as five independent failures at five lengths
        # instead of one failure at the first. Distinguish them and stop on the terminal kind.
        blob = f"{type(exc).__name__}: {exc}"
        row["terminal"] = any(k in blob for k in
                              ("DistBackendError", "ncclUnhandled", "out of memory", "OutOfMemory",
                               "HIP error", "Memory access fault"))
        print(f"  ctx={target} w={width}: REFUSED{' (TERMINAL)' if row['terminal'] else ''} "
              f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
        return row
    row["wall_seconds"] = round(time.perf_counter() - t0, 3)

    # A REJECTION IS NOT A RUN. The scheduler refuses an over-length request with a `finished`
    # message carrying `error` and NO tokens; `LLM.generate` now surfaces that. Scoring it as `ok`
    # is how the 131,056-token rung first came back as a passing row with 1 token, 0 prefill steps
    # and both needles False — a "result" that was really a refusal wearing a completion's clothes.
    if any(r.get("error") for r in results):
        row["ok"] = False
        row["error_type"] = "Rejected"
        row["error"] = next(r["error"] for r in results if r.get("error"))[:600]
        row["terminal"] = False          # the engine is fine; this request was declined
        print(f"  ctx={target} w={width}: REFUSED by the scheduler — {row['error'][:160]}",
              flush=True)
        return row

    steps = _steps_since(mark)
    pf = [d for (ph, _bs, d) in steps if ph == "prefill"]
    dec = [(ph, d) for (ph, _bs, d) in steps if ph != "prefill"]
    row["ok"] = True
    row["prefill_steps"] = len(pf)
    row["prefill_seconds"] = round(sum(pf), 3)
    row["prefill_tok_per_s"] = round(meta["actual_tokens"] / sum(pf), 1) if sum(pf) else None
    row["decode_steps"] = len(dec)
    row["decode_phases"] = sorted({p for p, _ in dec})
    if dec:
        med = _median([d for _, d in dec])
        row["decode_ms_per_step_median"] = round(med * 1e3, 2)
        row["decode_tok_per_s_median"] = round(1.0 / med, 2) if med else None
        # WHAT THAT NUMBER IS, recorded beside it rather than left to the reader. Without
        # MINISGL_STEP_LOG_SYNC the STEP_LOG window closes when the HOST finished enqueuing, so on a
        # captured leg it is a `hipGraphLaunch` return time and not a forward at all (MEASURED: 0.54
        # ms/step captured against 60.8 eager, a 112x that is an instrument artefact). Carried per
        # row so a future reader of the artifact cannot mistake one for the other.
        row["decode_ms_is_device_time"] = _STEP_LOG_SYNC

    def _needles(t: str) -> tuple:
        low = t.lower()
        early = ("47-zebra-9183" in low or "47 zebra 9183" in low
                 or ("zebra" in low and "9183" in low))
        late = ("halloway" in low or "6120" in low)
        return early, late

    text = res["text"]
    row["text"] = text
    row["gen_tokens"] = len(res["token_ids"])
    # WALL tok/s as well as the forward-only median, because the two answer different questions and
    # this model's gap between them is enormous: 37 tok/s of forward against 14 tok/s of wall, the
    # difference being the ~66 ms/token the host expert tier spends on PCIe OUTSIDE the timed
    # forward. A decode number quoted without saying which one it is is not a number.
    row["wall_tok_per_s"] = (round(sum(len(r["token_ids"]) for r in results) / row["wall_seconds"], 3)
                             if row["wall_seconds"] else None)
    row["degeneration"] = degeneration_report(text, res["token_ids"])
    row["retrieved_early_needle"], row["retrieved_late_needle"] = _needles(text)
    if width > 1:
        row["batch_texts"] = [r["text"] for r in results]
        row["batch_needles"] = [{"early": e, "late": l}
                                for e, l in (_needles(r["text"]) for r in results)]
        row["batch_gen_tokens"] = [len(r["token_ids"]) for r in results]

    row["graph_replays"] = int(getattr(getattr(llm.engine, "graph_runner", None),
                                       "replays", 0) or 0) - gr0
    row["eager_decode_forwards"] = int(getattr(llm.engine, "eager_decode_forwards", 0) or 0) - eg0
    row["engaged_counts"] = _eng_delta(eng0)
    if qsa is not None:
        v1, d1 = qsa.sparsity_totals()
        dv, dd = v1 - v0, d1 - d0
        row["qsa_static_steps"] = int(qsa.static_steps)
        row["qsa_captured_steps"] = int(qsa.captured_steps)
        row["visited"] = dv
        row["dense"] = dd
        row["sparsity_visited_over_dense"] = round(dv / dd, 6) if dd else None
    print(f"  ctx={target:7d} w={width} ({meta['actual_tokens']} tok) "
          f"prefill {row['prefill_seconds']}s ({row['prefill_tok_per_s']} tok/s) "
          f"decode {row.get('decode_tok_per_s_median')} tok/s "
          f"sparsity {row.get('sparsity_visited_over_dense')} "
          f"needles early={row['retrieved_early_needle']} late={row['retrieved_late_needle']}",
          flush=True)
    print(f"    -> {text[:400]!r}", flush=True)
    return row


def rank_main(rank: int, tp: int, args, model_dir: str) -> dict:
    global _failures
    from minisgl.distributed import DistributedInfo
    from minisgl.llm import LLM

    out: dict = {"rank": rank, "tp": tp, "layers": args.layers}
    kw = {}
    if args.device_gb > 0:
        kw = dict(weight_offload_device_gb=args.device_gb, weight_offload_gb=args.host_gb)

    print(f"\n[1] boot rank {rank}/{tp}", flush=True)
    t0 = time.perf_counter()
    llm = LLM(
        model_path=model_dir,
        dtype=torch.bfloat16,
        tp_info=DistributedInfo(rank, tp),
        # WAS hard-wired to 0 ("QSARuntime.prepare raises inside a capture"). The selection is now
        # capturable (static decode plan), so this is the A/B knob: 0 = eager decode, N = capture
        # buckets up to N. Whichever is asked for is GATED below, per rung.
        cuda_graph_max_bs=args.graph_bs,
        page_size=args.page_size,
        memory_ratio=args.memory_ratio,
        attention_backend="hip",
        max_running_req=args.max_running_req,
        max_extend_tokens=args.max_extend_tokens,
        **kw,
    )
    out["boot_seconds"] = round(time.perf_counter() - t0, 1)
    out["card"] = torch.cuda.get_device_name(llm.engine.device)
    out["device_index"] = int(getattr(llm.engine.device, "index", 0) or 0)

    # ---- the reach ceiling, stated as arithmetic ------------------------------------------------
    # `Engine.__init__` sets max_seq_len = min(checkpoint max_position, KV pool tokens). Both terms
    # are recorded so a ceiling can be attributed to the pool rather than to the model.
    eng = llm.engine
    kv = eng.ctx.kv_cache
    k0 = kv.k_cache(0)
    # `num_pages`, not the BUFFER's page count: `KVCache` is allocated with `num_pages + 1` so a
    # dummy request has somewhere to write, and billing that page as capacity overstates the pool by
    # one page on every serve.
    pool_tokens = int(eng.num_pages * args.page_size)
    out["kv_pages"] = int(eng.num_pages)
    out["page_size"] = int(args.page_size)
    out["kv_pool_tokens"] = pool_tokens
    out["engine_max_seq_len"] = int(eng.max_seq_len)
    # The checkpoint's own native context, read off the LIVE model rather than the file — the two
    # can differ (a rope-scaling override, an `--context-length` clamp), and it is the live one that
    # bounds `min(config.max_seq_len, num_tokens)`.
    mc = getattr(eng.model, "_config", None) or getattr(eng.model, "config", None)
    rc = getattr(mc, "rotary_config", None)
    out["checkpoint_max_position"] = int(getattr(rc, "max_position", 0) or 0)
    # KV BYTES PER TOKEN PER RANK, summed over the KV-bearing layers. This is the number that turns
    # "the pool holds N tokens" into "the pool costs M GiB", and it is what makes the reach ceiling
    # attributable: at TP=2 the kv heads are sharded, so a per-rank figure is the honest one.
    n_kv_layers = int(k0.shape[0] and len(getattr(kv, "_k_buffer", [k0])))
    out["kv_layers"] = n_kv_layers
    out["kv_bytes_per_token_per_rank"] = int(
        sum(int(kv.k_cache(i)[0, 0].numel()) * kv.k_cache(i).element_size() * 2
            for i in range(n_kv_layers))
    )
    out["kv_pool_bytes_per_rank"] = out["kv_bytes_per_token_per_rank"] * pool_tokens

    # THE WARMUP IS LOAD-BEARING, not politeness. `ctx.qsa` is installed by
    # `Qwen4ExpForConditionalGeneration.forward` on its FIRST call (the index-key cache has to be
    # allocated outside graph capture), so reading it straight after `LLM(...)` reads None and the
    # ACTIVE gate below fails on a perfectly good boot — which is exactly what run r1 did.
    from minisgl.core import SamplingParams as _SP
    llm.generate(["The capital of France is"],
                 _SP(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                     max_tokens=1))
    qsa = getattr(eng.ctx, "qsa", None)
    out["qsa_active"] = qsa is not None
    check_true("QSA runtime is ACTIVE (this file measures nothing without it)", qsa is not None,
               "MINISGL_QSA=0, a bad page size, or the qsa_index kernels are missing")
    if qsa is not None:
        out["qsa_profile"] = {
            "budget": qsa.profile.budget,
            "compress_ratio": qsa.profile.compress_ratio,
            "block_topk": qsa.profile.block_topk,
            "index_width": qsa.profile.index_width,
            "head_dim": qsa.profile.head_dim,
        }
        out["qsa_cache"] = repr(qsa.cache)
        out["qsa_cache_bytes"] = int(qsa.cache.bytes())
    print(f"  kv_pages={out['kv_pages']} pool_tokens={pool_tokens} "
          f"engine_max_seq_len={out['engine_max_seq_len']} "
          f"checkpoint_max_position={out['checkpoint_max_position']}", flush=True)

    tok = llm.tokenizer
    hf_tok = getattr(tok, "tokenizer", tok)

    # ---- the ladder ----------------------------------------------------------------------------
    print("\n[2] the long-context ladder (sampled: "
          f"temp={args.temperature} top_k={args.top_k} top_p={args.top_p})", flush=True)
    rows = []
    out["ladder"] = rows
    for target in [int(x) for x in args.lens.split(",") if x.strip()]:
        rows.append(run_batch(llm, hf_tok, args, target, rank))
        # FLUSHED AFTER EVERY RUNG, not at the end. A rung that takes down the rank (an OOM inside
        # the forward, a GPU fault) would otherwise destroy the rungs that already succeeded — and
        # on a 250 s boot those are the expensive part. The partial file IS the artifact if the run
        # dies; the parent's file is written only if it survives.
        if args.rank_json:
            with open(f"{args.rank_json}.rank{rank}.json", "w") as fh:
                json.dump(out, fh, indent=2)
        if rows[-1].get("terminal"):
            out["ladder_aborted_after"] = rows[-1]["target_tokens"]
            print("  ladder ABORTED: the engine is no longer usable after that failure", flush=True)
            break

    # ---- the CONC=2 capacity legs ---------------------------------------------------------------
    # The operating point admits 2 running requests, and the KV pool is SHARED between them. This is
    # the only leg that can distinguish the three things "CONC=2 at 16k" could mean: both resident
    # at once, the second QUEUED behind the first (a latency result), or a refusal (a capacity one).
    #
    # MORE THAN ONE LENGTH, because the group-alignment defect this leg exists to catch is CHUNK
    # PACKING ARITHMETIC and each prompt length is its own instance of it. The two recorded
    # reproductions are different leftovers, not the same bug seen twice:
    #     16382 = 15*1024 + 1022 -> leftover 2 -> the next request's chunk is 2  (48L TP=2, r2b.log)
    #      4087 =  3*1024 + 1015 -> leftover 9 -> ...its chunk is 9             (4L  TP=1, r3.log)
    # A fix verified at only one of them is verified against one arithmetic instance.
    conc_lens = [int(x) for x in str(args.conc_len).split(",") if x.strip() and int(x) > 0]
    conc_rows = []
    for cl in conc_lens:
        if out.get("ladder_aborted_after"):
            break
        print(f"\n[2b] CONC={args.conc_width} at ctx={cl}", flush=True)
        cr = run_batch(llm, hf_tok, args, cl, rank, width=args.conc_width)
        cr["leg"] = "concurrency"
        rows.append(cr)
        conc_rows.append(cr)
        if args.rank_json:
            with open(f"{args.rank_json}.rank{rank}.json", "w") as fh:
                json.dump(out, fh, indent=2)
    if conc_rows:
        out["concurrency_legs"] = conc_rows
        out["concurrency"] = conc_rows[0]  # back-compat with the r2b-era artifacts and the report

    ok_rows = [r for r in rows if r.get("ok")]
    out["max_context_served_tokens"] = max((r["actual_tokens"] for r in ok_rows), default=0)
    out["first_refusal"] = next(({"target": r["target_tokens"], "error_type": r.get("error_type"),
                                  "error": r.get("error")} for r in rows if not r.get("ok")), None)

    # ---- the gates -----------------------------------------------------------------------------
    print("\n[3] gates", flush=True)
    # THE INSTRUMENT, BEFORE ANY NUMBER TAKEN WITH IT. `decode_ms_per_step_median` comes from
    # STEP_LOG, which by default closes its window when the HOST finished enqueuing the step. A
    # captured decode step is a single graph launch that returns immediately, so without
    # MINISGL_STEP_LOG_SYNC=1 that column is launch latency and not a forward — MEASURED on this
    # exact configuration as 0.54 ms/step against the eager leg's 60.8, which would read as a 112x
    # "win" and is an instrument artefact. A captured leg may therefore not publish a forward time
    # unless the device-synced bracket was on. (The eager leg happens to read near-true either way,
    # because the dynamic QSA plan syncs on `row_ends.max().item()` every forward — which is exactly
    # why the artefact is so easy to miss: only one of the two legs is wrong.)
    from minisgl.scheduler.scheduler import STEP_LOG_SYNC as _SYNC_ON
    out["step_log_sync"] = bool(_SYNC_ON)
    if args.graph_bs > 0:
        check_true("a CAPTURED leg's ms/forward was taken with the device-synced bracket "
                   "(MINISGL_STEP_LOG_SYNC=1)", bool(_SYNC_ON),
                   "without it decode_ms_per_step_median is a hipGraphLaunch return time")
    above = [r for r in ok_rows if r["actual_tokens"] > (qsa.profile.budget if qsa else 2048)]
    check_true("at least one run finished ABOVE indexer_budget (the old refusal is gone)",
               bool(above), f"{len(ok_rows)} runs ok, none above budget")
    for r in above:
        # SPARSITY, per run. The one check a wrong-but-fluent selection cannot pass.
        sr = r.get("sparsity_visited_over_dense")
        check_true(f"ctx={r['actual_tokens']}: the selection actually skipped keys (ratio<1)",
                   sr is not None and sr < 0.999, f"ratio={sr}")
        # THE SELECTION ARM IS THE ONE THAT DISPATCHED, per rung, from the COUNTS. The `engaged()`
        # SET saturates at the first forward of the first rung and would report every later rung as
        # unchanged whatever it ran. What this catches and nothing else here does: a build where
        # `qsa_index` failed to import falls back to `MINISGL_QSA_OPS=torch`, which computes the
        # same values — so sparsity, coherence and retrieval all stay green while the HIP kernels
        # under test never ran. Prefill is eager on every leg, so these arms must appear on a
        # captured leg too.
        ec = r.get("engaged_counts", {})
        check_true(f"ctx={r['actual_tokens']}: the three qsa_index HIP arms dispatched",
                   all(ec.get(f"qsa_index.{k}", 0) > 0
                       for k in ("score_paged", "topk", "expand")),
                   f"{ {k: v for k, v in ec.items() if k.startswith('qsa_index')} }")
        check_true(f"ctx={r['actual_tokens']}: the TORCH selection fallback never dispatched",
                   not any(k.endswith("(torch)") for k in ec),
                   f"{[k for k in ec if k.endswith('(torch)')]}")
        # THE DISPATCH THIS LEG ASKED FOR IS THE DISPATCH IT GOT. Both directions are gated: an
        # eager leg that quietly replayed graphs and a captured leg that quietly fell back to eager
        # would both make the A/B "new vs itself". The phase tags come from STEP_LOG; the counters
        # come from the engine and the GraphRunner (an int add per step, not a saturating set).
        want = ["decode_graph"] if args.graph_bs > 0 else ["decode_eager"]
        check("ctx=%d: decode dispatch is what --graph-bs %d asked for"
              % (r["actual_tokens"], args.graph_bs), r["decode_phases"], want)
        if args.graph_bs > 0:
            check_true(f"ctx={r['actual_tokens']}: every decode step REPLAYED a graph",
                       r.get("graph_replays", 0) == r["decode_steps"]
                       and r.get("eager_decode_forwards", 0) == 0,
                       f"replays={r.get('graph_replays')} eager={r.get('eager_decode_forwards')} "
                       f"decode_steps={r['decode_steps']}")
        else:
            check_true(f"ctx={r['actual_tokens']}: NO graph was replayed",
                       r.get("graph_replays", 0) == 0,
                       f"replays={r.get('graph_replays')}")
        dg = r["degeneration"]
        check_true(f"ctx={r['actual_tokens']}: no loop signature",
                   dg["longest_immediate_repeat_run"] <= args.max_repeat_run,
                   f"run={dg['longest_immediate_repeat_run']}")
        check_true(f"ctx={r['actual_tokens']}: no letter-spell signature",
                   dg["longest_single_char_run"] <= 4, f"run={dg['longest_single_char_run']}")
        check_true(f"ctx={r['actual_tokens']}: no token-noise signature",
                   dg["non_ascii_char_frac"] <= 0.05, f"frac={dg['non_ascii_char_frac']}")
    # ---- CONCURRENCY, GATED EXPLICITLY --------------------------------------------------------
    # NOT covered by the `above` loop, and the reason is the whole hazard: `above` is filtered to
    # rows with `ok`, so a concurrency leg that RAISED — which is exactly the failure this leg was
    # added to catch (`QSA prefill chunk is not group-aligned: cached_len=2`) — drops silently out
    # of every check above it and the run still reports PASS. The refusal has to be gated where it
    # cannot be filtered away.
    for cr in conc_rows:
        w, n = args.conc_width, cr["target_tokens"]
        check_true(f"CONC={w} at ctx={n}: the batch ran (no group-alignment refusal)",
                   bool(cr.get("ok")),
                   f"{cr.get('error_type')}: {str(cr.get('error'))[:200]}")
        if not cr.get("ok"):
            continue
        # AND IT RAN AS A BATCH, not as two sequential requests. `chunk_gran` is about what the
        # PREFILL PACKER does when a second request shares a step's token budget, so a leg where the
        # scheduler happened to serialise the two proves nothing about it.
        check_true(f"CONC={w} at ctx={n}: {w} requests actually completed",
                   len(cr.get("batch_gen_tokens", [])) == w,
                   f"gen_tokens={cr.get('batch_gen_tokens')}")
        check_true(f"CONC={w} at ctx={n}: every request generated tokens",
                   all(t > 0 for t in cr.get("batch_gen_tokens", [0])),
                   f"gen_tokens={cr.get('batch_gen_tokens')}")

    # RETRIEVAL is reported for every run and gated only when asked, because a sampled single
    # generation at temperature 1.0 is a noisy retrieval test — the ladder's shape across lengths
    # is the signal, not any one row.
    out["retrieval"] = {str(r["actual_tokens"]): {"early": r.get("retrieved_early_needle"),
                                                  "late": r.get("retrieved_late_needle")}
                        for r in ok_rows}
    if args.gate_retrieval:
        for r in above:
            check_true(f"ctx={r['actual_tokens']}: EARLY needle retrieved "
                       f"(depth {r['early_needle_depth_frac']})", r["retrieved_early_needle"])

    out["failures"] = _failures
    print(f"\n[rank {rank}] {'PASS' if not _failures else 'FAIL'}: {_failures} failure(s)",
          flush=True)
    return out


def _spawn_target(rank, tp, args, model_dir, q):
    try:
        q.put(rank_main(rank, tp, args, model_dir))
    except BaseException:
        traceback.print_exc()
        q.put({"rank": rank, "failures": 1, "error": traceback.format_exc()[-2000:]})
        raise


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=512)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--device-gb", type=float, default=8.1)
    ap.add_argument("--host-gb", type=float, default=25.0)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--memory-ratio", type=float, default=0.90)
    ap.add_argument("--max-running-req", type=int, default=2)
    # 0 = eager decode; N = capture decode buckets up to N. THE A/B KNOB — see the module docstring.
    ap.add_argument("--graph-bs", type=int, default=0)
    # The PREFILL CHUNK. QSA needs every chunk boundary group-aligned (cached_len % r == 0); the
    # scheduler chunks on page boundaries and page_size 16 is a multiple of r=4, so any multiple of
    # the page size is legal. It is also the row count the [rows, compressed_blocks] scoring
    # workspace is tiled out of, so it is a memory term, not just a latency one.
    ap.add_argument("--max-extend-tokens", type=int, default=2048)
    ap.add_argument("--lens", default="4096,16384,32768")
    ap.add_argument("--gen-tokens", type=int, default=64)
    # The checkpoint's own generation_config, NOT greedy. See the module docstring.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--early-frac", type=float, default=0.02)
    ap.add_argument("--late-frac", type=float, default=0.90)
    ap.add_argument("--max-repeat-run", type=int, default=6)
    ap.add_argument("--no-chat-template", action="store_true")
    # A COMMA LIST, not one length: the chunk-packing arithmetic that produced the group-alignment
    # refusal is per prompt length (16382 -> leftover 2, 4087 -> leftover 9), so one length verifies
    # one instance. "" or 0 skips the concurrency legs entirely.
    ap.add_argument("--conc-len", type=str, default="0",
                    help="comma list of prompt lengths to drive at --conc-width; 0 = skip")
    ap.add_argument("--conc-width", type=int, default=2)
    ap.add_argument("--gate-retrieval", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--rank-json", default="", help="per-rank path PREFIX, flushed after each rung")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible")
        return 1
    if args.tp > torch.cuda.device_count():
        print(f"FAIL: --tp {args.tp} but {torch.cuda.device_count()} device(s) visible")
        return 1

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qwen4exp_offload_serve_test import subset_dir

    model_dir = subset_dir(args.model, args.layers, args.experts)
    print(f"[subset] {model_dir} ({args.layers} layers, {args.experts} experts, tp={args.tp})",
          flush=True)

    if args.tp == 1:
        results = [rank_main(0, 1, args, model_dir)]
    else:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
        q: "mp.Queue" = mp.Queue()
        procs = []
        for rank in range(args.tp):
            p = mp.Process(target=_spawn_target, args=(rank, args.tp, args, model_dir, q),
                           name=f"q4e-qsa-TP{rank}")
            p.start()
            procs.append(p)
        results = []
        alive = list(procs)
        while alive:
            while not q.empty():
                results.append(q.get())
            alive = [p for p in alive if p.is_alive()]
            if alive:
                time.sleep(0.5)
        for p in procs:
            p.join()
        while not q.empty():
            results.append(q.get())
        codes = [p.exitcode for p in procs]
        print(f"\n[parent] rank exit codes {codes}", flush=True)
        if any(c != 0 for c in codes):
            results.append({"rank": -1, "failures": 1, "error": f"rank exit codes {codes}"})

    results.sort(key=lambda r: r.get("rank", 0))
    failures = sum(int(r.get("failures", 0)) for r in results)
    out = {"tp": args.tp, "layers": args.layers, "lens": args.lens,
           "sampler": {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p},
           "failures": failures, "ranks": results}
    r0 = next((r for r in results if r.get("rank") == 0), None)
    if r0:
        out["max_context_served_tokens"] = r0.get("max_context_served_tokens")
        out["engine_max_seq_len"] = r0.get("engine_max_seq_len")
        out["first_refusal"] = r0.get("first_refusal")
    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} failure(s)", flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1)
