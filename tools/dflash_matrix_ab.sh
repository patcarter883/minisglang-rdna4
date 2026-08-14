#!/usr/bin/env bash
# Measure EVERY DFlash (target, drafter) pair on ONE code leg, and record the evidence that says
# which leg it was. Run it twice — REPO=<baseline worktree> then REPO=<fix worktree> — and diff.
#
# Why this exists rather than tools/dflash_graph_ab.sh: that one is TP=1, single-pair, and compares
# VERIFY capture. The DFlash defects under test here are per-layer mask correctness and PROPOSE
# capture, across four pairs that only exist at TP=2. It also has to survive the failure mode that
# started this whole investigation — a leg that silently runs eager and reports a plausible number —
# so the propose/verify ENGAGEMENT LEDGER is scraped for every pair and printed next to the score.
# A pair whose ledger is missing is reported as MISSING, never folded into a mean.
#
# The harness deliberately lives in ONE place and takes the tree as a parameter: if each leg ran its
# own copy of the script, a harness difference would be indistinguishable from a code difference.
#
#   REPO=/home/pat/code/minisgl-rdna4-dfbase  LEG=baseline  tools/dflash_matrix_ab.sh
#   REPO=/home/pat/code/minisgl-rdna4-musecap LEG=fixed     tools/dflash_matrix_ab.sh
set -uo pipefail

REPO="${REPO:?set REPO=<worktree to measure>}"
LEG="${LEG:?set LEG=<label for this code leg, e.g. baseline|fixed>}"
OUTDIR="${OUTDIR:-/home/pat/fixtures/minisgl-dflash-matrix}"   # DURABLE host path, never /tmp
PAIRS="${PAIRS:-muse qwen35b-awq qwen27b laguna}"
# 1919, the port Prometheus actually scrapes (`minisgl` job -> host.docker.internal:1919). An
# off-port harness is invisible to Grafana for its whole run, which is backwards: a measurement run
# is exactly when the dashboard is worth having. Nothing else can hold the cards at the same time —
# the lease serialises that — so reusing the production port is safe rather than contended.
PORT="${PORT:-1919}"
TP="${TP:-2}"
CONC="${CONC:-4}"
MAXTOK="${MAXTOK:-12000}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"

mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SUMMARY="$OUTDIR/${LEG}_${STAMP}.summary.txt"

# ---- provenance ---------------------------------------------------------------------------------
# A green A/B that turns out to have run the SAME code twice is worse than no A/B, so the tree's
# identity is captured BEFORE anything boots and asserted against the container's mount after.
REV="$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
DIRTY="$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l)"
DIFFHASH="$(git -C "$REPO" diff HEAD -- python/ 2>/dev/null | sha256sum | cut -c1-16)"
{
  echo "leg=$LEG"
  echo "repo=$REPO"
  echo "head=$REV"
  echo "dirty_files=$DIRTY"
  echo "python_diff_sha=$DIFFHASH   # DIFFERS between legs or the A/B is measuring one tree twice"
  echo "stamp=$STAMP"
  echo "pairs=$PAIRS  tp=$TP conc=$CONC max_tokens=$MAXTOK"
  echo
} | tee "$SUMMARY"

probe() {  # $1=port  $2=outfile -> prints "tok/s completion_tokens wall_s"
  PORT="$1" OUT="$2" MAXTOK="$MAXTOK" PROBE_TEMP="${PROBE_TEMP:-}" PROBE_TOP_P="${PROBE_TOP_P:-0.95}" PROBE_TOP_K="${PROBE_TOP_K:-}" PROBE_CTK="${PROBE_CTK:-}" python3 - <<'PY'
import json, os, time, urllib.request
port, out, maxtok = os.environ["PORT"], os.environ["OUT"], int(os.environ["MAXTOK"])
# LONG MATHEMATICAL / CODE GENERATION. The previous probe — eight short prose prompts at 256 tokens
# with ignore_eos — measured the drafter at the weakest point of its own curve. `spec/dflash.py`
# records accept-len as a monotone function of generated position, "real code 3.5 at P<32 rising to
# 7.4 at P=512-1024", so a 256-token prose sample cannot reach the regime this drafter is for, and
# ignore_eos padded every sample with post-stop drift nothing was trained to predict. Structured,
# repetitive, long-horizon mathematical code is where a block-diffusion drafter should look best, and
# it is what the pair is actually served for.
#
# EXPLICIT, ENUMERATED deliverables plus a developer system prompt (thinking stays ON). A Qwen3.6
# target given an open-ended "write a library" prompt will spend the budget reasoning about what to
# write instead of writing it — which measures the drafter on chain-of-thought prose, the exact
# regime the old probe already over-sampled, and can burn the whole 12k window before a line of code
# appears. Naming the signatures makes the output long AND structured, which is the workload the
# pair is served for.
SYSTEM = (
    "You are a senior Python library engineer. Your ANSWER is production-quality Python code and "
    "nothing else: no preamble, no restatement of the task, no questions back. Start the answer at "
    "the first line of code. Implement every item on the required list, in the order given. Every "
    "implementation must be COMPLETE — never `...`, never `# TODO`, never 'rest omitted for "
    "brevity'. Include full docstrings and inline comments explaining the mathematics. Keep any "
    "planning brief and get to the code."
)
prompts = [
    "Write a module `rational.py` implementing exact arbitrary-precision rational arithmetic.\n"
    "Required: class Rational with __init__(self, num:int, den:int=1) normalising via gcd and "
    "keeping the sign on the numerator; __add__, __sub__, __mul__, __truediv__, __neg__, __abs__, "
    "__pow__(int); __eq__, __lt__, __le__, __gt__, __ge__, __hash__; __repr__, __str__; "
    "classmethod from_float(cls, x:float); classmethod from_decimal_string(cls, s:str) handling "
    "'-12.345' and repeating forms like '0.(3)'; to_float(self); "
    "continued_fraction(self)->list[int]; classmethod from_continued_fraction(cls, terms); "
    "best_approximation(self, max_den:int) via the Stern-Brocot mediant walk.\n"
    "Give a docstring with doctests for every method, then a table of the time complexity of each "
    "operation in terms of the bit-lengths of numerator and denominator.",

    "Write a module `tridiag_eig.py` computing eigenvalues of a real symmetric tridiagonal matrix "
    "by implicit QL iteration with Wilkinson shifts.\n"
    "Required functions: wilkinson_shift(d_n1, d_n, e_n1) -> float; givens(a, b) -> (c, s); "
    "ql_implicit(d: list[float], e: list[float], max_iter: int = 30) -> list[float]; "
    "sturm_count(d, e, x) -> int; bisection_eigenvalues(d, e, tol) -> list[float]; "
    "gershgorin_bounds(d, e) -> (lo, hi).\n"
    "Before each function, write a comment block deriving it: the shift formula from the trailing "
    "2x2 block, the chase of the bulge, and why the implicit form avoids forming T - sigma*I. "
    "Then write tests covering a matrix with degenerate eigenvalues, one with tightly clustered "
    "eigenvalues, a 1x1 case and a zero off-diagonal split.",

    "Write a module `fft.py` implementing the DFT in pure Python (complex built-in only, no numpy).\n"
    "Required functions: dft_naive(x) -> list[complex]; fft_radix2(x) -> list[complex] for "
    "len(x) a power of two, iterative with bit-reversal permutation; bit_reverse_indices(n) -> "
    "list[int]; next_pow2(n) -> int; bluestein(x) -> list[complex] for arbitrary n via the "
    "chirp-z transform; fft(x) dispatching between them; ifft(X); convolve(a, b) using fft.\n"
    "Before each function write a comment block with the derivation: the Danielson-Lanczos "
    "splitting, the chirp identity nk = (n^2 + k^2 - (k-n)^2)/2, and the O(n log n) argument. "
    "Then write tests comparing every path against dft_naive on random inputs of length "
    "1, 2, 3, 5, 8, 12 and 16, and report the maximum absolute error.",
]
tot_tok = 0.0; tot_s = 0.0; texts = []
for p in prompts:
    # No ignore_eos: let the model stop where it means to. The prompts are what make the samples
    # long, so length comes from real content rather than from forced post-EOS drift.
    # Thinking stays ON — it is what the serve actually does, so turning it off would measure a
    # configuration nobody runs. The system prompt and the enumerated deliverables are what keep the
    # reasoning span bounded; an open-ended prompt is what makes it run away.
    # Sampling is a MEASURED VARIABLE, not a harness constant. A checkpoint that ships no
    # temperature/top_p/top_k (Muse-Glimmer) is served at the neutral 1.0/1.0/-1, which is full-
    # distribution sampling and the worst case for draft acceptance — so a probe that hardcodes its
    # own values characterises neither the served default nor the model card's recommendation.
    body = {"model": "m", "temperature": float(os.environ.get("PROBE_TEMP") or 0.7),
            "seed": 1234, "max_tokens": maxtok,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]}
    if os.environ.get("PROBE_TOP_P"): body["top_p"] = float(os.environ["PROBE_TOP_P"])
    if os.environ.get("PROBE_TOP_K"): body["top_k"] = int(os.environ["PROBE_TOP_K"])
    if os.environ.get("PROBE_CTK"):   body["chat_template_kwargs"] = json.loads(os.environ["PROBE_CTK"])
    body = json.dumps(body).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=1800))
    el = time.time() - t0
    ct = d["usage"]["completion_tokens"]
    tot_tok += ct; tot_s += el
    texts.append(d["choices"][0]["message"]["content"])
json.dump(texts, open(out, "w"))
print(f"{tot_tok/tot_s:.2f} {int(tot_tok)} {tot_s:.1f}")
PY
}

# ---- per-pair OPERATING POINT -------------------------------------------------------------------
# serve.sh's table is NOT the whole launch configuration. The Muse spec point needs env that lives
# only in the control panel's state.json (the panel SHADOWS the table), and at MEM_RATIO=0.96 the
# table's own comment says the KV pool clears by 0.47 GiB — so a single reserve left on-card is the
# difference between booting and `num_pages > 1`. Launching with serve.sh defaults reproduced exactly
# that crash. Spell the point out here and PRINT it, so a leg can never be compared against a
# differently-configured leg without it showing up in the summary.
#
# EVERY pair is pinned, none left to "serve.sh's default". That is not belt-and-braces: the two
# worktrees under test carry DIFFERENT serve.sh tables (the operating points for muse 0.93->0.96 and
# for qwen27b, which HEAD lacks entirely, are uncommitted work in the shared tree), so leaving the
# ratio to the table would make the A/B depend on which launch config each leg happened to check
# out. The launch configuration is not the code under test; the `python/` diff is. Pinning here
# holds the launch line identical across legs and makes it visible in the summary.
pair_env() {
  case "$1" in
    muse|muse-glimmer)
      echo "MEM_RATIO=0.96 CONC=4 MINISGL_SPEC_MHA_PAGED=1 MINISGL_GHOST_ORACLE=0" \
           "MINISGL_REC_SNAP_HOST=1 MINISGL_DFLASH_QUANT=nvfp4" ;;
    qwen27b)     echo "MEM_RATIO=0.90 CONC=2 MINISGL_DFLASH_QUANT=fp8" ;;
    qwen35b-awq) echo "MEM_RATIO=0.86 CONC=4 MINISGL_DFLASH_QUANT=fp8" ;;
    laguna)      echo "MEM_RATIO=0.93 CONC=4" ;;
    *) echo "" ;;
  esac
}

for ALIAS in $PAIRS; do
  # LOWERCASED: docker compose rejects a project name with uppercase ("invalid project name ...
  # must consist only of lowercase alphanumeric characters, hyphens, and underscores"), and the
  # lease derives the project from this name — so a capital letter in a leg label kills the run at
  # launch rather than at validation.
  NAME="$(echo "dfm-${LEG}-${ALIAS}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '-')"
  LOG="$OUTDIR/${LEG}_${STAMP}_${ALIAS}.log"
  PENV="$(pair_env "$ALIAS")"
  echo "===== [$LEG] $ALIAS =====" | tee -a "$SUMMARY"
  echo "  env: MODEL=$ALIAS SPEC=dflash TP=$TP ${PENV:-<serve.sh defaults>} ${PAIR_ENV_EXTRA:-}" | tee -a "$SUMMARY"

  ( cd "$REPO" && \
    env MODEL="$ALIAS" SPEC=dflash TP="$TP" CONC="$CONC" \
    MINISGL_HOST_PORT="$PORT" \
    MINISGL_SPEC_TIMING=1 \
    $PENV ${PAIR_ENV_EXTRA:-} \
    gpu-lease -n 2 --detach --name "$NAME" -- docker compose --profile serve up -d ) \
    >>"$LOG" 2>&1
  if [ $? -ne 0 ]; then echo "  BOOT_LAUNCH_FAILED (see $LOG)" | tee -a "$SUMMARY"; continue; fi

  # gpu-lease PREFIXES the compose project (and so the container_name) with `lease-`; it prints
  # `project=lease-<name>`. Deriving the container from $NAME alone silently watches a container that
  # never exists -> instant "not ready" AND a teardown that misses, leaving the cards leased.
  PROJ="lease-${NAME}"
  CID="${PROJ}-serve"
  ready=0
  gone=0
  # A CRASHED SCHEDULER DOES NOT STOP THE CONTAINER. `Process minisgl-DP0-TP0-scheduler:` dies on
  # its own traceback and the parent stays `Up` forever, so polling only for container-liveness sat
  # on BOTH CARDS for the full BOOT_TIMEOUT after the serve was already dead. Watch for the worker's
  # own death notice as well, and fail in seconds instead of a quarter of an hour.
  crashed=0
  for _ in $(seq 1 $((BOOT_TIMEOUT/5))); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    docker ps --format '{{.Names}}' | grep -q "^${CID}$" || { gone=1; break; }
    if docker logs "$CID" 2>&1 | grep -qE "^Process minisgl-[A-Za-z0-9-]+:|AssertionError:|torch.OutOfMemoryError"; then
      crashed=1; break
    fi
    sleep 5
  done

  if [ "$ready" = 1 ]; then
    # PROVENANCE ASSERT: the container must be serving the tree we think we measured.
    MOUNT="$(docker inspect "$CID" --format '{{range .Mounts}}{{if eq .Destination "/engine"}}{{.Source}}{{end}}{{end}}' 2>/dev/null)"
    if [ "$MOUNT" != "$REPO" ]; then
      echo "  PROVENANCE_MISMATCH mounted=$MOUNT expected=$REPO" | tee -a "$SUMMARY"
    else
      read -r TPS CT WALL <<<"$(probe "$PORT" "$OUTDIR/${LEG}_${STAMP}_${ALIAS}.gen.json")"
      docker logs "$CID" >>"$LOG" 2>&1
      ACC="$(grep -oE "mean accept-len=[0-9.]+ over [0-9]+ reqs" "$LOG" | tail -1)"
      CMT="$(grep -oE "committed/verify=[0-9.]+" "$LOG" | tail -1)"
      WID="$(grep -oE "verify-width\[[^]]*\]" "$LOG" | tail -1)"
      # THE LEDGER. `propose-graph ...` comes from ProposeCaptureStats.line(); ALWAYS-EAGER means the
      # proposer declined capture, which is the exact thing this A/B exists to detect.
      PG="$(grep -oE "propose-graph [^ ]*(\([^)]*\))?( replay=[0-9]+ eager=[0-9]+)?" "$LOG" | tail -1)"
      VG="$(grep -oE "verify-graph replay=[0-9]+ eager=[0-9]+" "$LOG" | tail -1)"
      RING="$(grep -ao "DFlash propose ring.*" "$LOG" | tail -1)"
      {
        echo "  tok/s=$TPS  (tokens=$CT wall=${WALL}s)"
        echo "  ${ACC:-accept-len=MISSING}  ${CMT:-}"
        echo "  ${WID:-verify-width=MISSING}"
        echo "  propose: ${PG:-MISSING}"
        echo "  verify : ${VG:-MISSING}"
        echo "  ring   : ${RING:-none (uncaptured propose)}"
      } | tee -a "$SUMMARY"
    fi
  else
    if   [ "$gone" = 1 ];    then echo "  CONTAINER_EXITED during boot (see $LOG)" | tee -a "$SUMMARY"
    elif [ "$crashed" = 1 ]; then echo "  SCHEDULER_CRASHED during boot (see $LOG)" | tee -a "$SUMMARY"
    else echo "  NOT_READY after ${BOOT_TIMEOUT}s (see $LOG)" | tee -a "$SUMMARY"; fi
    docker logs "$CID" >>"$LOG" 2>&1
    # The LAUNCH LINE first — a boot failure is far more often a launch-config difference than a
    # code fault, and it is the one line that distinguishes them.
    grep -aoE "\[serve\] python -m minisgl.*" "$LOG" | tail -1 | sed 's/^/    /' | tee -a "$SUMMARY"
    grep -aoE "Reserved [0-9.]+ GiB for [^;]*" "$LOG" | sort -u | sed 's/^/    /' | tee -a "$SUMMARY"
    grep -aE "Error|assert" "$LOG" | tail -4 | sed 's/^/    /' | tee -a "$SUMMARY"
  fi

  ( cd "$REPO" && docker compose -p "$PROJ" --profile serve down ) >>"$LOG" 2>&1
  # Wait on the EXIT CONDITION, not a fixed sleep — the next boot needs both cards back.
  for _ in $(seq 1 60); do
    docker ps --format '{{.Names}}' | grep -q "^${CID}$" || break; sleep 2
  done
  echo | tee -a "$SUMMARY"
done

echo "summary -> $SUMMARY"
