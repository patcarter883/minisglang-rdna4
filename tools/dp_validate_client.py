"""DP launcher validation client (runs INSIDE the container).

Starts the minisgl server (python -m minisgl) with the given --data-parallel-size, waits for the
HTTP endpoint to come up, then:
  1. Coherence: drives the CHAT endpoint (/v1/chat/completions with `messages`) so the instruct
     model's chat template is applied server-side -> real, judge-able text (raw /generate emits
     scaffold tokens on an instruct checkpoint and is NOT a coherence signal).
  2. Throughput: N concurrent DECODE-DOMINATED chat requests (short prompt, max_tokens>=256) ->
     aggregate output tok/s. Decode is where the ~2x DP win lives; a short-gen workload is
     prefill-bound and hides it.

Every HTTP request carries a hard per-request timeout: a GPU-wedged replica makes its in-flight
requests fail fast (counted as errors) instead of the whole phase riding the host's phase timeout.

Prints a single machine-parseable RESULT line. Server is torn down on exit. The host wraps THIS in
a hard timeout as a backstop.

Usage (inside container):
    PYTHONPATH=/engine/python:/engine python /engine/tools/dp_validate_client.py \
        --dp 2 --model /models/ZAYA1-8B-fp8 --conc 16 --max-tokens 256 --port 1919 --graph 16
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def _post_chat(
    base: str, messages: list[dict], max_tokens: int, timeout: float, model: str = "default"
) -> tuple[str, int]:
    """POST /v1/chat/completions (non-streaming) with `messages` so the server applies the chat
    template. Returns (assistant_text, completion_tokens_est). Raises on HTTP/timeout error so the
    caller can count it as a fail-fast failure.

    `model` is a REQUIRED field on the server's OpenAICompletionRequest schema (no default) — omit it
    and FastAPI rejects the body with HTTP 422 before any GPU work, which looks like a server failure
    but is purely a missing-field client bug. The value is informational (the server serves a single
    loaded checkpoint regardless), so any non-empty string satisfies validation."""
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    text = data["choices"][0]["message"]["content"]
    return text, max_tokens  # ignore_eos -> exactly max_tokens generated


def _wait_health(base: str, timeout: float) -> bool:
    """Poll /v1/models until the server answers (proxy for 'all replicas ready')."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument(
        "--max-running",
        type=int,
        default=0,
        help="Per-replica --max-running-requests ceiling forwarded to the server. 0 (default) -> "
        "use the graph cap (--graph), so EACH replica is capped at one captured decode step and CONC "
        ">> max-running genuinely SATURATES a replica (the prerequisite for measuring DP's ~2x). "
        "Do NOT slave this to CONC: if the per-replica ceiling tracks concurrency, a single dp=1 "
        "replica just batches all CONC requests into one (possibly eager-fallback) step and the "
        "second card adds nothing -> the 2x gate spuriously fails.",
    )
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--attn", default="hip")
    ap.add_argument(
        "--graph",
        type=int,
        default=16,
        help="cuda-graph max bs forwarded to the server; 0 DISABLES graph capture (isolation).",
    )
    ap.add_argument("--boot-timeout", type=float, default=600.0)
    ap.add_argument(
        "--req-timeout",
        type=float,
        default=120.0,
        help="Hard per-request HTTP timeout (s); a wedged replica fails its requests fast.",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="Free-form label echoed into the RESULT line (e.g. isolation-nograph).",
    )
    ap.add_argument(
        "--enable-ep",
        action="store_true",
        help="Pass --enable-ep to the server: shard the MoE experts across the dp replicas (EP) "
        "instead of replicating them. Requires --dp > 1.",
    )
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "/engine/python:/engine")
    env.setdefault("MINISGL_MOE_SCATTER", "0")  # graph-capture-friendly (prod config)
    # MINISGL_ATTN_HIP / MINISGL_TAIL_HIP are inherited from the parent env so the harness can flip
    # the native-HIP attention/tail kernels off for crash isolation without touching this file.

    # Per-replica running ceiling. Default (0) -> the graph cap, so each replica is capped at exactly
    # one captured decode step. This is INTENTIONALLY decoupled from CONC: to expose DP's ~2x, CONC
    # must exceed what ONE replica can batch (see --max-running help). When graph is OFF (0) there is
    # no captured step to align to, so fall back to a fixed 16.
    max_running = args.max_running if args.max_running > 0 else (args.graph or 16)
    cmd = [
        sys.executable, "-m", "minisgl",
        "--model", args.model,
        "--data-parallel-size", str(args.dp),
        "--attn", args.attn,
        "--graph", str(args.graph),
        "--port", str(args.port),
        "--max-running-requests", str(max_running),
    ]
    if args.enable_ep:
        cmd.append("--enable-ep")
    tag = f"[{args.tag}] " if args.tag else ""
    print(f"{tag}[dp_validate] launching: {' '.join(cmd)}", flush=True)
    print(
        f"{tag}[dp_validate] env knobs: MINISGL_ATTN_HIP={env.get('MINISGL_ATTN_HIP','1')} "
        f"MINISGL_TAIL_HIP={env.get('MINISGL_TAIL_HIP','1')} graph={args.graph}",
        flush=True,
    )
    log_path = f"/engine/tools/dp_serve_dp{args.dp}{('_'+args.tag) if args.tag else ''}.log"
    logf = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    rc = 1
    try:
        print(f"{tag}[dp_validate] waiting for endpoint (<= {args.boot_timeout}s)...", flush=True)
        if not _wait_health(base, args.boot_timeout):
            print(f"{tag}[dp_validate] FAIL: server did not become healthy", flush=True)
            print(f"RESULT dp={args.dp} tag={args.tag} status=BOOT_FAIL toks_per_s=0", flush=True)
            return 1
        print(f"{tag}[dp_validate] endpoint healthy", flush=True)

        # --- 1. coherence via the CHAT endpoint (chat template applied server-side) ---
        coh_msgs = [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}]
        try:
            txt, _ = _post_chat(base, coh_msgs, max_tokens=48, timeout=args.req_timeout)
            print(f"{tag}[dp_validate] COHERENCE single -> {txt!r}", flush=True)
        except Exception as e:
            print(f"{tag}[dp_validate] COHERENCE single FAILED: {e!r}", flush=True)

        # --- 1b. coherence: several concurrent distinct chat prompts (hits >1 replica via RR) ---
        coh_prompts = [
            "What is the capital of France? Answer in one short sentence.",
            "What is two plus two? Answer with just the number.",
            "What is the opposite of hot? Answer with one word.",
            "What are the two elements that make up water? Answer briefly.",
        ]
        coh_results: dict[int, str] = {}
        def _coh(i: int, p: str) -> None:
            try:
                coh_results[i] = _post_chat(
                    base, [{"role": "user", "content": p}], max_tokens=40, timeout=args.req_timeout
                )[0]
            except Exception as e:  # noqa: BLE001
                coh_results[i] = f"<ERROR {e!r}>"
        ths = [threading.Thread(target=_coh, args=(i, p)) for i, p in enumerate(coh_prompts)]
        for t in ths: t.start()
        for t in ths: t.join()
        for i, p in enumerate(coh_prompts):
            print(f"{tag}[dp_validate] CONC[{i}] {p!r} -> {coh_results.get(i,'')!r}", flush=True)

        # --- 2. throughput: N concurrent DECODE-DOMINATED chat requests ---
        # Short prompt + long max_tokens so DECODE dominates (the prefill is a one-shot cost; the
        # ~2x DP win is in the steady-state decode loop). Each request fails fast on req-timeout.
        tput_msgs = [{"role": "user", "content": "Write a long, detailed story about a robot who learns to paint."}]
        N = args.conc
        MT = args.max_tokens
        counts = [0] * N
        errs: list[str | None] = [None] * N
        def _run(i: int) -> None:
            try:
                _txt, ntok = _post_chat(base, tput_msgs, MT, timeout=args.req_timeout)
                counts[i] = ntok
            except Exception as e:  # noqa: BLE001
                errs[i] = repr(e)
        t0 = time.perf_counter()
        ths = [threading.Thread(target=_run, args=(i,)) for i in range(N)]
        for t in ths: t.start()
        for t in ths: t.join()
        dt = time.perf_counter() - t0
        total_tok = sum(counts)
        nerr = sum(1 for e in errs if e is not None)
        tps = total_tok / dt if dt > 0 else 0.0
        print(
            f"{tag}[dp_validate] THROUGHPUT N={N} max_tokens={MT} max_running={max_running} "
            f"total_tok={total_tok} elapsed={dt:.2f}s errors={nerr} "
            f"(saturated={'yes' if N > max_running else 'NO -- CONC<=max_running, 2x cannot show'})",
            flush=True,
        )
        if nerr:
            print(f"{tag}[dp_validate] errors sample: {[e for e in errs if e][:2]}", flush=True)
        status = "OK" if nerr == 0 and total_tok > 0 else "REQ_FAIL"
        print(
            f"RESULT dp={args.dp} tag={args.tag} status={status} toks_per_s={tps:.2f} "
            f"total_tok={total_tok} elapsed_s={dt:.2f} errors={nerr} "
            f"conc={N} max_running={max_running}",
            flush=True,
        )
        rc = 0 if status == "OK" else 1
    finally:
        print(f"{tag}[dp_validate] tearing down server (log: {log_path})", flush=True)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        logf.close()
        # echo last lines of the serve log for diagnostics (the 0x1016 abort lands here)
        try:
            with open(log_path) as f:
                tail = f.readlines()[-30:]
            print(f"{tag}[dp_validate] --- serve log tail ---", flush=True)
            sys.stdout.write("".join(tail))
            print(f"{tag}[dp_validate] --- end serve log ---", flush=True)
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
