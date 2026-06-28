"""In-engine RSA validation client (runs INSIDE the container).

Launches `python -m minisgl` on the normal port, then exercises ONE endpoint
(`/v1/chat/completions`) two ways:
  1. PLAIN  — no `rsa` field -> ordinary single completion (regression check).
  2. RSA    — `rsa: {n,k,t,...}` -> the WHOLE Markovian-RSA loop runs server-side
              (expand N -> aggregate K-subsets over T rounds -> select), returning the
              final answer plus an `rsa` metadata block. Confirms the loop ran in-engine
              (rounds == t, n_requests >= n) and the answer is coherent.

Prints a single machine-parseable RESULT line. Server torn down on exit.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _post(base: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _wait_health(base: str, timeout: float) -> bool:
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--graph", type=int, default=16)
    ap.add_argument("--boot-timeout", type=float, default=360.0)
    ap.add_argument("--req-timeout", type=float, default=180.0)
    ap.add_argument("--rsa-n", type=int, default=4)
    ap.add_argument("--rsa-k", type=int, default=2)
    ap.add_argument("--rsa-t", type=int, default=2)
    ap.add_argument("--rsa-max-tokens", type=int, default=192)
    ap.add_argument("--rsa-tail-tokens", type=int, default=256)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "/engine/python:/engine")
    env.setdefault("MINISGL_MOE_SCATTER", "0")
    cmd = [
        sys.executable, "-m", "minisgl",
        "--model", args.model, "--attn", "hip", "--graph", str(args.graph),
        "--port", str(args.port), "--max-running-requests", "16",
    ]
    print(f"[rsa-val] launching: {' '.join(cmd)}", flush=True)
    log_path = "/engine/tools/rsa_serve.log"
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    rc = 1
    try:
        print(f"[rsa-val] waiting for endpoint (<= {args.boot_timeout}s)...", flush=True)
        if not _wait_health(base, args.boot_timeout):
            print("RESULT status=BOOT_FAIL", flush=True)
            return 1
        print("[rsa-val] endpoint healthy", flush=True)

        msg = [{"role": "user",
                "content": "What is 17 + 26? Reason briefly, then give the final answer "
                           "inside \\boxed{}."}]

        # 1) PLAIN single completion (no rsa field) — regression check
        plain = _post(base, {"model": "zaya", "messages": msg, "max_tokens": 64,
                             "temperature": 0.0}, args.req_timeout)
        plain_txt = plain["choices"][0]["message"]["content"]
        has_rsa_meta_plain = "rsa" in plain
        print(f"[rsa-val] PLAIN -> {plain_txt[:200]!r}", flush=True)
        print(f"[rsa-val] PLAIN has rsa-meta? {has_rsa_meta_plain} (expect False)", flush=True)

        # 2) RSA call — the whole loop runs server-side
        t0 = time.perf_counter()
        rsa = _post(base, {
            "model": "zaya", "messages": msg,
            "rsa": {"n": args.rsa_n, "k": args.rsa_k, "t": args.rsa_t,
                    "max_tokens": args.rsa_max_tokens, "tail_tokens": args.rsa_tail_tokens,
                    "temperature": 0.7},
        }, args.req_timeout)
        dt = time.perf_counter() - t0
        rsa_txt = rsa["choices"][0]["message"]["content"]
        meta = rsa.get("rsa", {})
        print(f"[rsa-val] RSA -> {rsa_txt[:300]!r}", flush=True)
        print(f"[rsa-val] RSA meta = {json.dumps(meta)}  elapsed={dt:.1f}s", flush=True)

        ok = (
            bool(plain_txt) and not has_rsa_meta_plain          # plain path intact, no rsa meta
            and bool(rsa_txt)                                    # rsa returned an answer
            and meta.get("rounds") == args.rsa_t                 # ran T rounds server-side
            and meta.get("n_requests", 0) >= args.rsa_n          # at least N rollouts issued
        )
        print(
            f"RESULT status={'OK' if ok else 'FAIL'} "
            f"plain_len={len(plain_txt)} rsa_len={len(rsa_txt)} "
            f"rounds={meta.get('rounds')} n_requests={meta.get('n_requests')} "
            f"selection={meta.get('selection_method')} elapsed_s={dt:.1f}",
            flush=True,
        )
        rc = 0 if ok else 1
    except Exception as e:  # noqa: BLE001
        print(f"[rsa-val] EXCEPTION: {e!r}", flush=True)
        print("RESULT status=EXC", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        logf.close()
        try:
            with open(log_path) as f:
                tail = f.readlines()[-25:]
            print("[rsa-val] --- serve log tail ---", flush=True)
            sys.stdout.write("".join(tail))
            print("[rsa-val] --- end serve log ---", flush=True)
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
