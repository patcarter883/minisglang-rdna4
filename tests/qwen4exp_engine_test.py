"""qwen4_exp through the REAL `Engine`/`Scheduler`: boot, prefill, sample, decode, detokenize.

WHY THIS EXISTS
---------------
`qwen4exp_gpu_forward_test.py` proves the MODEL runs: it builds a `Context` by hand and calls
`model.forward()`. That is deliberately not a serve. Everything between a prompt string and a token
string — pool sizing, the prefix-cache choice, slot lifecycle across prefill->decode, chunked
prefill, the sampler, the scheduler loop — is untested by it, and one of those gaps was total: the
PLE block reads its n-gram embeddings out of `Context.ple`, nothing in the engine ever set it, and
so EVERY qwen4_exp forward through `Engine` raised. This is the test that would have caught that,
and it is the one that now holds the wiring in place.

WHAT IT ASSERTS, in increasing strength
---------------------------------------
  [1] the engine boots, builds a PLE runtime, and picks the prefix cache this model can actually
      run (naive — the recurrent-radix snapshot store does not cover PLE state);
  [2] a prompt generates the requested number of tokens and detokenizes;
  [3] two CONCURRENT requests get distinct state slots and are staged in one batch;
  [4] **the token stream the PLE runtime was staged with, per sequence, is exactly the sequence the
      model saw — every token once, in order.** This is the load-bearing one. It fails if the
      n-gram context lags a step (the overlap-scheduling hazard), if a batch is committed twice, if
      a chunked prefill drops or repeats a chunk, or if a recycled slot keeps a dead request's
      history. None of those crashes; all of them just make the features quietly wrong.
  [5] the QSA indexer-budget refusal is still live (a request past it must raise, not run dense).

SUBSET. `--model` should point at a layer-subset checkpoint (see the report): the full 48 layers do
not fit on a 16 GB card, and NOTHING here is about output quality — with a 4-layer prefix of a
48-layer model the text is meaningless by construction, so this only ever checks shapes, counts,
lifecycle and finiteness. `max_extend_tokens` is set small on purpose to force chunked prefill.

Run (card 0, in the serve image):

    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
      --security-opt seccomp=unconfined --security-opt label=disable --cap-add SYS_PTRACE \
      --ipc host --shm-size 16gb -e ROCR_VISIBLE_DEVICES=0 \
      -v <worktree>:/engine -v <subset>:/subset:ro -v /home/pat/.cache/hf-ple:/ple:ro \
      -e MINISGL_PLE_FILES=... -e MINISGL_PLE_META_FILES=... \
      --entrypoint bash minisgl-rdna4:m1b-20260903 -lc \
      'PYTHONPATH=/engine/python:/opt/kernels python /engine/tests/qwen4exp_engine_test.py'
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_failures = 0


def check(name: str, got, want) -> None:
    global _failures
    ok = got == want
    _failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name:50s} got={got!r:<24} want={want!r}", flush=True)


def check_true(name: str, cond, detail: str = "") -> None:
    global _failures
    _failures += not cond
    print(f"  {'ok  ' if cond else 'FAIL'} {name:50s} {detail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("Q4E_SUBSET", "/subset"))
    ap.add_argument("--max-tokens", type=int, default=6)
    ap.add_argument("--max-running-req", type=int, default=4)
    ap.add_argument("--memory-ratio", type=float, default=0.90)
    # Small on purpose: a prompt longer than this prefills in several passes, which is the only way
    # to reach the PLE continuation path (cached_len > 0) without an 8192-token prompt.
    ap.add_argument("--max-extend-tokens", type=int, default=8)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no HIP device visible (is_rocm false?) — check device passthrough")
        return 1

    from minisgl.core import SamplingParams
    from minisgl.llm import LLM

    print("\n[1] boot the engine", flush=True)
    llm = LLM(
        model_path=args.model,
        dtype=torch.bfloat16,
        cuda_graph_max_bs=0,  # graph capture for this model is bring-up plan T6
        page_size=16,  # the native-HIP attention kernels require a multiple of 16
        memory_ratio=args.memory_ratio,
        attention_backend="rdna4",
        max_running_req=args.max_running_req,
        max_extend_tokens=args.max_extend_tokens,
    )
    ple = llm.engine.ple_runtime
    check_true("engine built a PLE runtime", ple is not None)
    check_true("scheduler bound it", llm._ple is ple)
    # The snapshot radix would restore GDN state at a prefix boundary and leave the PLE conv window
    # and n-gram history at their zero/EOS seed — silent, and exactly what the store exists to stop.
    check("prefix cache", llm.cache_manager.__class__.__name__.lower().find("radix") < 0, True)
    check_true("overlap scheduling is off", not _overlap_possible(llm),
               "PLE hashes host token ids BEFORE the forward; overlap commits them after")

    # Record every batch the runtime is staged with, so [4] can replay the per-sequence stream.
    staged: "list[tuple[list[int], list[np.ndarray], bool]]" = []
    _prepare = ple.prepare

    def _spy(slots, token_lists, **kw):
        b = _prepare(slots, token_lists, **kw)
        staged.append((list(b.slots), [t.copy() for t in b.tokens], b.is_decode))
        return b

    ple.prepare = _spy  # type: ignore[method-assign]

    print("\n[2/3] two concurrent requests, one long enough to prefill in chunks", flush=True)
    prompts = [
        "The capital of France is",
        "Write a short paragraph about the history of the bicycle and its uses in daily life today",
    ]
    # SAMPLED, not greedy — repo rule. Nothing here reads the token VALUES (a layer subset makes
    # them meaningless); sampling is what keeps the run on the same sampler path a serve uses.
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.max_tokens)
    out = llm.generate(prompts, sp)
    # The prompt ids are not in `generate`'s result — read them from the request records it kept, so
    # [4] compares against the tokenization the engine actually ran, not a re-tokenization.
    prompt_ids = {uid: list(st.input_ids) for uid, st in llm.status_map.items()}
    for p, o in zip(prompts, out):
        print(f"      {p[:38]!r:42s} -> {o['text']!r}", flush=True)
        check("tokens generated", len(o["token_ids"]), args.max_tokens)
        check_true("detokenized to a string", isinstance(o["text"], str))

    extend = [s for s in staged if not s[2]]
    check_true("prefill staged in >1 chunk", len(extend) > 2, f"{len(extend)} extend batches")
    multi = [s for s in staged if len(s[0]) > 1]
    check_true("a batch carried both sequences", len(multi) > 0, f"{len(multi)} multi-seq batches")
    if multi:
        slots = multi[0][0]
        check("concurrent sequences hold distinct slots", len(set(slots)), len(slots))

    print("\n[4] the staged n-gram stream == the sequence the model saw", flush=True)
    # Replay per slot, in staging order. Padding rows carry the reserved NULL slot 0 and are skipped.
    seen: "dict[int, list[int]]" = {}
    for slots, tokens, _ in staged:
        for slot, toks in zip(slots, tokens):
            if slot == 0:
                continue
            seen.setdefault(int(slot), []).extend(int(t) for t in toks)
    check("one slot per request", len(seen), len(prompts))
    # The LAST sampled token is emitted but never fed back (generation stops there), so each staged
    # stream is the full sequence minus that one token. Anything else — a repeat, a gap, a one-step
    # lag — means the n-gram context is not the model's context, and nothing anywhere would say so.
    # Compared as SETS: which slot id a request draws is a scheduling detail, not an invariant.
    want = {tuple((prompt_ids[uid] + list(o["token_ids"]))[:-1]) for uid, o in enumerate(out)}
    got = {tuple(v) for v in seen.values()}
    check_true("staged streams == the sequences the model saw", got == want,
               f"lengths staged={sorted(len(g) for g in got)} "
               f"expected={sorted(len(w) for w in want)}")
    if got != want:
        for g in sorted(got):
            near = min(want, key=lambda w: abs(len(w) - len(g)))
            print(f"      slot stream {len(g)} tok: {_first_diff(list(g), list(near))}", flush=True)

    print("\n[5] context past indexer_budget refuses rather than running dense", flush=True)
    # Neither Engine nor Scheduler keeps its EngineConfig, so read the budget from the checkpoint.
    from minisgl.models.config import ModelConfig
    from minisgl.utils import cached_load_hf_config

    budget = int(ModelConfig.from_hf(
        cached_load_hf_config(args.model), spec_algorithm="none").indexer_budget or 0)
    try:
        llm.generate([list(range(budget + 512))], SamplingParams(temperature=0.0, max_tokens=1))
        check_true(f"raised past indexer_budget={budget}", False, "it did NOT raise")
    except NotImplementedError as e:
        check_true(f"raised past indexer_budget={budget}", "budget" in str(e), str(e)[:90])

    print(f"\n{'PASS' if not _failures else f'FAIL ({_failures} checks)'}")
    return 1 if _failures else 0


def _first_diff(got, want) -> str:
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return f"index {i}: staged {a} vs sequence {b}"
    return "none (a length difference only)"


def _overlap_possible(llm) -> bool:
    """True if `run_forever` would pick the overlap loop for this scheduler. Read from the same
    attributes the loop selection reads, so it cannot drift from the decision it is checking."""
    from minisgl.env import ENV

    return not (ENV.DISABLE_OVERLAP_SCHEDULING or llm._rec_radix or llm._swa_radix
                or llm._ple is not None)


if __name__ == "__main__":
    sys.exit(main())
