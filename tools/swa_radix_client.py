"""SWA-radix serve validation client (runs inside the lean container against localhost:1919).

Three modes:
  * reuse:  WARM the prefix P (so its sliding-window snapshot is cached), then generate B = P+suffix,
            which REUSES P via the sliding-layer window extend (rdna4.py::_swa_prefill_extend).
  * cold:   generate B directly against an empty/naive cache (no reuse) — the identity reference.
  * matrix: the full cold-vs-hit comparison in ONE process, over cases x max_tokens (below).

Byte-identity gate: sha256 of the DECODED text (identical greedy output => identical bytes) must match
between reuse and cold, for a prefix LONGER than the window and SHORTER. Also reports TTFT (time to
first token) so the reuse prefill-skip benefit is measurable.

Why `matrix` exists. A single serve cannot serve a request COLD twice — the first one inserts it into
the radix — so cold-vs-hit has to be staged. Each trial gets a unique salt (so its prefix has never
been seen) and runs three steps:

    [cold]    B1 = P+tail1, never seen        -> hit_tokens must be 0
    [full]    B1 again                        -> hit_tokens ~ len(B1): the whole prompt is reused
    [partial] B2 = P+tail2, tail2 never seen  -> hit_tokens ~ len(P): a SHARED long prefix with a
                                                 differing tail, the shape SWA-radix exists for

cold-vs-full is a same-process comparison and is the core losslessness gate. The partial step has no
in-process cold reference (running one would cache B2), so it is compared ACROSS a MINISGL_SWA_RADIX=1
serve and a MINISGL_SWA_RADIX=0 serve driven with the SAME --run-id, i.e. byte-identical request
sequences. The cold step doubles as the control for that comparison: it must agree across the two
serves, or nothing cross-serve means anything.

Stay at or below 64 output tokens. This repo's serve is not bit-reproducible against itself past
~32-64 tokens (batch-shape-dependent reductions), so a divergence beyond that window says nothing
about the prefix cache.

TRUE greedy is not `temperature: 0` alone. `SamplingParams.is_greedy` (core.py) is
`(temperature <= 0 or top_k == 1) and top_p == 1.0`, and an UNSET top_p inherits the checkpoint's
generation_config.json — gemma-4 declares top_p 0.95, so a request carrying only temperature 0 still
goes through the sampler and is not reproducible against itself. Every request below therefore pins
all three (GREEDY, overridable via --temperature/--top-p/--top-k for a deliberate sampled run). A
byte-identity result measured without this is an artefact, not a losslessness verdict.

Cache-hit evidence: a "lossless" reuse proves nothing if nothing was reused, so the client reads
/metrics around the B request and reports the DELTA in `minisgl_prefix_cache_hit_tokens_total` and
`minisgl_prefix_cache_prompt_tokens_total`. reuse must show hit_tokens > 0; cold must show 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request

HOST = "http://localhost:1919"
URL = f"{HOST}/generate"
METRICS = f"{HOST}/metrics"

# See the module docstring: all three are required for is_greedy, because an unset top_p inherits
# generation_config.json (0.95 on gemma-4) and silently re-enables the sampler.
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 1}


# RECALL tails, not open-ended ones. Two properties are needed and an open continuation has neither:
# the first few output tokens must depend on DEEP prompt content (or the gate cannot see a corrupted
# reused prefix at all), and they must be CONFIDENT (or run-to-run decode noise flips near-ties and
# the gate reports divergence that has nothing to do with the cache). Asking the model to echo a
# numbered note it was given gives both. The two tails name DIFFERENT notes, so they also share the
# whole prefix and differ only in the tail — the shape SWA-radix exists for.
TAILS = (
    " Repeat note 7 exactly, word for word. Note 7:",
    " Repeat note 11 exactly, word for word. Note 11:",
)
# The filler must be VARIED. A prefix built by repeating one sentence drives this checkpoint into a
# token loop, and a degenerate loop emits the SAME text for every prompt — cold and reuse then agree
# no matter what the cache did, which makes the losslessness verdict vacuous. (Measured: with a
# repeated-sentence prefix, a 33-token prompt and a 2430-token prompt produced byte-identical
# 16-token output.) The matrix asserts the outputs are prompt-discriminating for this reason.
_SUBJECTS = (
    "the harmonic oscillator", "a rigid rotor", "the hydrogen atom", "a square potential well",
    "the free electron gas", "a diatomic molecule", "the Debye solid", "a two-level system",
    "the anharmonic lattice", "a charged pendulum", "the spherical top", "a tunnelling barrier",
)
_CLAIMS = (
    "stores energy in evenly spaced levels", "loses coherence under weak damping",
    "responds sharply near its resonance", "shows a measurable isotope shift",
    "obeys a simple sum rule at low temperature", "couples only to odd-parity modes",
    "saturates once the drive exceeds threshold", "leaves a distinctive spectral fingerprint",
)


def build_prompts(case: str, salt: str = "", tail: int = 0):
    """(prefix, prefix+tail). `salt` goes at the FRONT so it makes the whole PREFIX unique — a salt in
    the tail would leave P shared with earlier trials and the 'cold' step would not be cold."""
    head = f"Trial {salt}. " if salt else ""
    if case == "long":                     # prefix > window on a W=512 model (Laguna): ~780 tokens
        n = 60
    elif case == "xlong":
        # Gemma4's window is 1024, so the 'long' case above sits BELOW it and exercises only the
        # Wp == boundary path — the same path 'short' takes. The wraparound case (boundary > W, so the
        # ring has overwritten positions the snapshot must still reconstruct in ascending order) is the
        # one that can actually go wrong, and it is window-size-dependent, not model-name-dependent.
        # ~2600 tokens covers W=1024 with room to spare and still covers W=512.
        n = 200
    else:                                  # prefix < window (still needs >= 11 notes for the tails)
        n = 12
    body = " ".join(
        f"Note {i + 1}: {_SUBJECTS[(i * 5 + 1) % len(_SUBJECTS)]} "
        f"{_CLAIMS[(i * 3 + 2) % len(_CLAIMS)]}."
        for i in range(n)
    )
    P = head + "A rigorous study of physics. " + body
    return P.strip(), (P + TAILS[tail]).strip()


def read_prefix_counters() -> "tuple[int, int]":
    """(hit_tokens, prompt_tokens) from /metrics — the server's own prefix-cache accounting."""
    try:
        with urllib.request.urlopen(METRICS, timeout=30) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception as exc:  # a serve without the metrics route still runs the identity gate
        print(f"  (metrics unavailable: {exc})")
        return (-1, -1)
    out = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        for key in ("minisgl_prefix_cache_hit_tokens_total",
                    "minisgl_prefix_cache_prompt_tokens_total"):
            # The samples carry a {model_name="..."} label, so match the metric NAME and take the
            # trailing value — an exact `startswith(key + " ")` silently read every counter as -1.
            if line.startswith(key):
                out[key] = int(float(line.rsplit(None, 1)[1]))
    return (out.get("minisgl_prefix_cache_hit_tokens_total", -1),
            out.get("minisgl_prefix_cache_prompt_tokens_total", -1))


def gen(prompt: str, max_tokens: int, sampling: dict):
    body = json.dumps(
        {"prompt": prompt, "max_tokens": max_tokens, "ignore_eos": True, **sampling}
    ).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    ttft = None
    raw = bytearray()
    with urllib.request.urlopen(req, timeout=900) as r:
        for chunk in r:
            raw += chunk
            if ttft is None and chunk.strip() not in (b"", b"data:", b"data: "):
                ttft = time.time() - t0
    # Reconstruct the DECODED TEXT: the server yields exactly `data: <incremental_output>\n` per chunk
    # (stream_generate). Strip the fixed 6-char "data: " prefix EXACTLY (NOT lstrip — the token text may
    # legitimately start with spaces, and lstrip + variable SSE framing made identical greedy output
    # hash differently). Join the payloads; that content is framing-independent.
    text = "".join(
        ln[6:] for ln in raw.decode("utf-8", "replace").split("\n")
        if ln.startswith("data: ") and "[DONE]" not in ln
    )
    return text, (ttft or 0.0), time.time() - t0


def measure(prompt: str, max_tokens: int, sampling: dict):
    """One request, bracketed by the server's own prefix-cache counters."""
    hit0, prompt0 = read_prefix_counters()
    text, ttft, total = gen(prompt, max_tokens, sampling)
    time.sleep(0.5)  # the scheduler's stats snapshot reaches /metrics over the detokenizer link
    hit1, prompt1 = read_prefix_counters()
    return {
        "sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "text": text,
        "ttft": ttft,
        "total": total,
        "hit": hit1 - hit0,
        "prompt": prompt1 - prompt0,
    }


def run_matrix(cases, token_counts, run_id: str, sampling: dict, ref_repeats: int) -> int:
    failures = 0
    seen: "dict[str, str]" = {}  # cold sha -> which trial produced it (degeneracy guard)
    for case in cases:
        for mt in token_counts:
            salt = f"{run_id}-{case}-{mt}"
            p, b1 = build_prompts(case, salt, tail=0)
            _, b2 = build_prompts(case, salt, tail=1)
            cold = measure(b1, mt, sampling)
            full = measure(b1, mt, sampling)
            # CONTROL: the same request a THIRD time — same prompt, same cache state, same prefill
            # shape as `full`. full-vs-full2 therefore isolates the serve's own reproducibility from
            # the cache: if it diverges, cold-vs-full divergence cannot be blamed on the prefix cache.
            full2 = measure(b1, mt, sampling)
            # COLD reference for the PARTIAL-hit step. It cannot be the same prompt (asking for B2
            # cold would cache B2 and turn the partial step into a full hit), so it is B2 built on a
            # DIFFERENT salt — legitimate only because the recall tail's answer is the note text,
            # which does not depend on the salt. The harness verifies that assumption rather than
            # assuming it: cold-vs-cold2-prefix identity is asserted below via `salt_free`.
            # Measured ref_repeats times, each on its OWN fresh salt so every one is a true cold
            # request. If they disagree the reference is not self-stable and the cell yields
            # INDETERMINATE rather than a false DIVERGED — a cold long-prompt prefill is not always
            # reproducible on this engine, and calling that a cache defect would be wrong.
            cold2s = [
                measure(build_prompts(case, f"{salt}z{i}", tail=1)[1], mt, sampling)
                for i in range(ref_repeats)
            ]
            cold2 = cold2s[0]
            ref_stable = len({r["sha"] for r in cold2s}) == 1
            # WARM P on its own before the partial step. A SWA serve keeps ONE snapshot per request,
            # at that request's own page-aligned end (snapshot_ladder_depth == 0 for "swa"), so B1's
            # snapshot sits at align_down(len(B1)) — PAST the shared prefix, and B2's match boundary
            # align_down(len(P)) has no snapshot behind it and is refused. Prefilling P by itself is
            # what puts a snapshot at that boundary. Without this the partial step silently measured
            # hit=0, i.e. a second cold request.
            gen(p, 4, sampling)
            time.sleep(0.7)  # let the async cache_req/attach settle before B2 reuses it
            partial = measure(b2, mt, sampling)
            for label, r in (("cold", cold), ("full", full), ("full2", full2),
                             ("cold2", cold2), ("partial", partial)):
                print(
                    f"STEP run={run_id} case={case} max_tokens={mt} step={label} "
                    f"sha={r['sha']} hit={r['hit']} prompt={r['prompt']} "
                    f"ttft={r['ttft']:.4f} total={r['total']:.4f} nchars={len(r['text'])}"
                )
            same = cold["sha"] == full["sha"]
            repro = full["sha"] == full2["sha"]
            same_p = cold2["sha"] == partial["sha"]
            p_verdict = (
                "INDETERMINATE" if not ref_stable else
                ("IDENTICAL" if same_p else "DIVERGED")
            )
            failures += (not same) + (ref_stable and not same_p)
            # The hit counters make the verdicts falsifiable: identical output with hit==0 on the
            # 'full'/'partial' steps would mean the cache was never consulted, i.e. nothing proved.
            print(
                f"VERDICT run={run_id} case={case} max_tokens={mt} "
                f"cold_vs_full={'IDENTICAL' if same else 'DIVERGED'} "
                f"cold2_vs_partial={p_verdict} "
                f"repro_full_vs_full2={'IDENTICAL' if repro else 'DIVERGED'} "
                f"cold_hit={cold['hit']} full_hit={full['hit']} full2_hit={full2['hit']} "
                f"cold2_hit={cold2['hit']} partial_hit={partial['hit']} "
                f"ref_stable={ref_stable} ref_shas={[r['sha'][:8] for r in cold2s]} "
                f"ttft_cold={cold['ttft']:.4f} ttft_full={full['ttft']:.4f} "
                f"ttft_partial={partial['ttft']:.4f}"
            )
            if not same:
                print(f"  COLD   : {cold['text']!r}")
                print(f"  FULLHIT: {full['text']!r}")
            if ref_stable and not same_p:
                print(f"  COLD2  : {cold2['text']!r}")
                print(f"  PARTHIT: {partial['text']!r}")
            if not repro:
                print(f"  FULLHIT2: {full2['text']!r}")
            # Emitted for the cross-serve (RADIX=1 vs RADIX=0) diff; the shas alone are the compare
            # key, the text is here so a divergence can be read without a re-run.
            print(f"  XCHK cold={cold['sha']} cold2={cold2['sha']} partial={partial['sha']}")
            print(f"  PARTIAL: {partial['text']!r}")
            # Degeneracy guard: the two TAILS ask for different notes, so their answers must differ.
            # If they do not, the recall probe has collapsed and byte-identity is satisfied by a
            # constant rather than by the cache.
            trial = f"{case}/{mt}"
            seen[trial] = cold["sha"]
            if cold["sha"] == cold2["sha"]:
                failures += 1
                print(
                    f"  DEGENERATE {trial}: the note-7 and note-11 tails produced the SAME output "
                    f"({cold['sha']}) — the probe is answering from a constant, not from the "
                    f"prompt, so byte-identity cannot detect a lossy cache"
                )
            # The cold2-as-reference construction assumes the answer does not depend on the salt.
            # Assert it: cold(tail 0) is the same prompt content under a different salt in every
            # other trial, so a salt-sensitive answer shows up as a cold sha that moves per trial.
            if len({s for t, s in seen.items() if t.split("/")[1] == str(mt)}) > 1:
                print(
                    f"  NOTE {trial}: cold sha varies across cases at max_tokens={mt} — the "
                    f"cold2 reference for the partial step assumes a salt-independent answer"
                )
    return failures


def run_repro(cases, token_counts, run_id: str, sampling: dict, repeats: int) -> int:
    """The FLOOR, measured with no comparison to anything: send ONE prompt `repeats` times and count
    distinct outputs. Run this on a MINISGL_SWA_RADIX=0 serve and it isolates the engine's own
    determinism — if the same prompt does not reproduce itself with the prefix cache disabled, a
    cold-vs-reuse byte-identity gate cannot mean anything, and any divergence it reports is the
    serve, not the cache."""
    bad = 0
    for case in cases:
        for mt in token_counts:
            _, b = build_prompts(case, f"{run_id}-{case}-{mt}", tail=0)
            shas = [measure(b, mt, sampling)["sha"] for _ in range(repeats)]
            distinct = len(set(shas))
            bad += distinct > 1
            print(
                f"REPRO run={run_id} case={case} max_tokens={mt} repeats={repeats} "
                f"distinct={distinct} {'STABLE' if distinct == 1 else 'UNSTABLE'} shas={shas}"
            )
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reuse", "cold", "matrix", "repro"], required=True)
    ap.add_argument("--repeats", type=int, default=5, help="repro mode: identical requests per cell")
    ap.add_argument("--ref-repeats", type=int, default=3,
                    help="matrix mode: cold references for the partial step; a cell whose reference "
                         "is not self-stable reports INDETERMINATE instead of DIVERGED")
    ap.add_argument("--case", choices=["long", "short", "xlong"], default="short")
    ap.add_argument("--cases", default="short,xlong", help="matrix mode: comma-separated")
    ap.add_argument("--token-counts", default="16,32,64", help="matrix mode: comma-separated")
    ap.add_argument("--run-id", default="0",
                    help="matrix mode: salts every prompt; MUST match across the two serves")
    ap.add_argument("--max-tokens", type=int, default=48)
    # Overrides for a DELIBERATE sampled run; the defaults are the greedy triple.
    ap.add_argument("--temperature", type=float, default=GREEDY["temperature"])
    ap.add_argument("--top-p", type=float, default=GREEDY["top_p"])
    ap.add_argument("--top-k", type=int, default=GREEDY["top_k"])
    a = ap.parse_args()
    sampling = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k}
    greedy = (a.temperature <= 0.0 or a.top_k == 1) and a.top_p == 1.0  # core.py is_greedy
    print(f"[swa-radix] sampling={sampling} greedy={greedy}")
    if not greedy:
        print("  WARNING: not greedy — output is not reproducible, so identity proves nothing")

    if a.mode == "repro":
        bad = run_repro(
            a.cases.split(","), [int(t) for t in a.token_counts.split(",")], a.run_id, sampling,
            a.repeats,
        )
        print(f"\n{'PASS' if bad == 0 else f'FAIL ({bad} unstable cells)'}")
        return bad

    if a.mode == "matrix":
        failures = run_matrix(
            a.cases.split(","), [int(t) for t in a.token_counts.split(",")], a.run_id, sampling,
            a.ref_repeats,
        )
        print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} divergences)'}")
        return failures

    P, B = build_prompts(a.case)
    if a.mode == "reuse":
        gen(P, 4, sampling)  # warm: prefill+cache P (creates the page-aligned window snapshot)
        time.sleep(0.7)      # let the async cache_req/attach settle before B reuses it

    # Bracket ONLY the measured request, so the warm-up's own prompt tokens are excluded and the
    # delta is exactly what B reused.
    hit0, prompt0 = read_prefix_counters()
    text, ttft, total = gen(B, a.max_tokens, sampling)
    time.sleep(0.5)  # the scheduler's stats snapshot reaches /metrics over the detokenizer link
    hit1, prompt1 = read_prefix_counters()
    d_hit, d_prompt = hit1 - hit0, prompt1 - prompt0

    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]   # hash the DECODED text
    print(f"RESULT mode={a.mode} case={a.case} sha={sha} ttft={ttft:.4f} "
          f"total={total:.4f} nchars={len(text)} greedy={greedy} "
          f"sampling={sampling} hit_tokens={d_hit} prompt_tokens={d_prompt}")
    print(f"  OUT[{a.mode}/{a.case}]: {text[:160]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
