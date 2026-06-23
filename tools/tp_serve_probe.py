"""Phase 4-0/4-3/4-4 — TP=2 serve bring-up + greedy-text parity probe (RUNS ON GPU).

The in-process `LLM` is TP=1-only (`DistributedInfo(0,1)`), so TP=2 must go through the real
server (mp.spawn rank schedulers + ZMQ + FastAPI). This driver launches the server per config,
greedy-generates the fixed prompts via the NON-streaming /v1/chat/completions endpoint (one JSON
per prompt — robust, no SSE text-newline parsing), dumps the text, tears the server down, then
diffs TP=1 vs TP=2 generations (greedy text equality is the practical proxy for token equality,
matching the established "PASS not bit-identical" posture).

Escalation (fail-fast: cheapest first, so a harness/flag bug dies on the 0.6B, not the 35B):
  s0 dense 0.6B  TP1 + TP2   -> 4-0: RCCL/torch.distributed collectives work on the 2-card box
  s1 GDN 4B      TP1 + TP2   -> 4-3: GDN head-parallel sharding numerics (TP1 4B is the oracle)
  s2 MoE 35B     TP2         -> 4-4: the 35B boots+coheres across 2 cards (~9-10 GB/card)

TP>1 forces --disable-pynccl => torch.distributed "nccl" backend == RCCL on ROCm (the plan's
path; the custom pynccl kernel is deferred). Results -> tools/tp2_results/. This script is GPU
work: it is launched INSIDE the leased container (see the run command in the commit / PORT.md).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "The three primary colors are red, blue, and",
    "Q: What is 2 + 2? A:",
]

OUTDIR = "/engine/tools/tp2_results"

# Each config launches one server. Distinct ports => distinct torch.distributed init addr
# (tcp://127.0.0.1:port+1), so sequential servers never collide. GDN models bound the recurrent
# state slots (max_running_requests small) — the default 256 is GiB-scale and OOMs a 16 GB card.
CONFIGS = [
    dict(name="s0_dense_0p6b_tp1", model="Qwen/Qwen3-0.6B", tp=1, port=21001),
    dict(name="s0_dense_0p6b_tp2", model="Qwen/Qwen3-0.6B", tp=2, port=21003),
    dict(name="s1_gdn_4b_tp1", model="Qwen/Qwen3.5-4B", tp=1, port=21005, max_running=16),
    dict(name="s1_gdn_4b_tp2", model="Qwen/Qwen3.5-4B", tp=2, port=21007, max_running=16),
    dict(name="s2_moe_35b_tp2", model="cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", tp=2, port=21009,
         max_running=16, memory_ratio=0.85, ready_timeout=1800),
]
MAX_TOKENS = 48


def _get_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_config(cfg: dict) -> dict:
    port = cfg["port"]
    name = cfg["name"]
    os.makedirs(OUTDIR, exist_ok=True)
    logpath = f"{OUTDIR}/{name}.server.log"
    cmd = [
        sys.executable, "-m", "minisgl",
        "--model", cfg["model"],
        "--tensor-parallel-size", str(cfg["tp"]),
        "--port", str(port),
        "--graph", "0",  # eager (GDN forces eager anyway; keep dense comparable + simple)
    ]
    if cfg["tp"] > 1:
        cmd += ["--disable-pynccl"]  # torch.distributed nccl == RCCL on ROCm (deferred: pynccl)
    if "max_running" in cfg:
        cmd += ["--max-running-requests", str(cfg["max_running"])]
    if "memory_ratio" in cfg:
        cmd += ["--memory-ratio", str(cfg["memory_ratio"])]

    print(f"\n[{name}] launching: {' '.join(cmd)}", flush=True)
    result = {"name": name, "model": cfg["model"], "tp": cfg["tp"], "ok": False,
              "error": None, "load_secs": None, "generations": []}
    t0 = time.time()
    logf = open(logpath, "wb")
    env = {**os.environ, "PYTHONPATH": "/engine/python"}
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    try:
        deadline = time.time() + cfg.get("ready_timeout", 900)
        ready = False
        while time.time() < deadline:
            if _get_ok(f"http://127.0.0.1:{port}/v1"):
                ready = True
                break
            if proc.poll() is not None:
                result["error"] = f"server exited during load (rc={proc.returncode}); see {logpath}"
                return result
            time.sleep(3)
        if not ready:
            result["error"] = f"server not ready within {cfg.get('ready_timeout', 900)}s; see {logpath}"
            return result
        result["load_secs"] = round(time.time() - t0, 1)
        print(f"[{name}] ready in {result['load_secs']}s; generating ...", flush=True)
        for p in PROMPTS:
            try:
                resp = _post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"model": "", "prompt": p, "max_tokens": MAX_TOKENS,
                     "temperature": 0.0, "ignore_eos": False, "stream": False},
                    timeout=300,
                )
                text = resp["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001 - record + continue
                text = None
                result.setdefault("gen_errors", []).append(repr(e))
            result["generations"].append({"prompt": p, "text": text})
            print(f"  [{name}] {p!r} -> {text!r}", flush=True)
        result["ok"] = bool(result["generations"]) and all(g["text"] for g in result["generations"])
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except Exception:
                pass
            time.sleep(4)
        logf.close()
    return result


def _diff(a: dict, b: dict) -> str:
    ga = {g["prompt"]: g["text"] for g in a.get("generations", [])}
    gb = {g["prompt"]: g["text"] for g in b.get("generations", [])}
    if not a.get("ok") or not b.get("ok"):
        return f"SKIP ({a['name']} ok={a.get('ok')}, {b['name']} ok={b.get('ok')})"
    n = sum(1 for p in PROMPTS if ga.get(p) == gb.get(p))
    return f"{n}/{len(PROMPTS)} greedy-text identical"


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    results = {}
    for cfg in CONFIGS:
        r = run_config(cfg)
        results[cfg["name"]] = r
        with open(f"{OUTDIR}/{cfg['name']}.json", "w") as f:
            json.dump(r, f, indent=2)
        status = "OK" if r["ok"] else f"FAIL ({r['error'] or r.get('gen_errors')})"
        print(f"[{cfg['name']}] {status}", flush=True)

    print("\n==================== SUMMARY ====================", flush=True)
    for name, r in results.items():
        load = f"{r['load_secs']}s" if r["load_secs"] else "-"
        print(f"  {name:24s} tp={r['tp']} boot={'OK' if r['ok'] else 'FAIL'} load={load}", flush=True)
    print("  --- parity (TP1 vs TP2, greedy text) ---", flush=True)
    if "s0_dense_0p6b_tp1" in results and "s0_dense_0p6b_tp2" in results:
        print(f"  4-0 dense 0.6B : {_diff(results['s0_dense_0p6b_tp1'], results['s0_dense_0p6b_tp2'])}", flush=True)
    if "s1_gdn_4b_tp1" in results and "s1_gdn_4b_tp2" in results:
        print(f"  4-3 GDN 4B     : {_diff(results['s1_gdn_4b_tp1'], results['s1_gdn_4b_tp2'])}", flush=True)
    s2 = results.get("s2_moe_35b_tp2", {})
    print(f"  4-4 MoE 35B TP2: boot={'OK' if s2.get('ok') else 'FAIL'} (vLLM parity = follow-up)", flush=True)
    with open(f"{OUTDIR}/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUTDIR}/ (per-config .json + .server.log + summary.json)", flush=True)


if __name__ == "__main__":
    main()
