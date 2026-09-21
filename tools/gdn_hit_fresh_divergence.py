"""GDN recurrent-state restore: fresh-prefill vs radix-HIT divergence probe.

THE LEAD IT TESTS (skill: minisgl-serving-stack, "Degeneration — prevention leads" #2).
The ebbc1dd0903b collapse (2026-09-21) began mid-generation after a `recurrent-radix HIT
restored recurrent state at cached_len=43616`. GDN state restore is delicate; a subtly wrong
restore corrupts hidden state WITHOUT crashing — which is exactly what mid-stream quality
collapse looks like. This probe is the discriminating experiment:

    fresh  : cold-cache full prefill of prompt X        -> text F
    hit    : same X again, radix restores at len(X)     -> text H
    partial: X + one more user turn (restores at len(X), prefills the delta — the
             production seam)                           -> text P

Run `fresh` on a cold boot, then `hit` and `partial` in the same boot; then a SECOND boot
repeats all three. Six samples, five comparisons:

    div(F1,F2)  fresh-vs-fresh   the engine's intrinsic determinism FLOOR
    div(F1,H1), div(F2,H2)       fresh-vs-HIT — THE test: 0 floor + nonzero divergence = restore bug
    div(H1,H2)  hit-vs-hit       restore determinism across boots
    div(P1,P2)  partial-vs-partial  the restore+continue-prefill seam

DETERMINISM: top_k=1 forces argmax on every step, so any divergence is a logit/state
difference, not sampling noise. Argmax also AMPLIFIES detection — a perturbed logit flips
the argmax on near-ties — which is what we want here. No other sampling params are sent;
the arm's own defaults apply (the box-wide rule).

PROVENANCE: the request is the REAL context of the degenerated session (WebUI
ebbc1dd0903b, messages 0..28 = everything the collapsing request saw), with the real
Hermes toolset from toolcall_degen_probe, padded with deterministic filler to
production prefill depth (~45k tokens — the live request HIT at 43,616).

READ THE SERVE LOG, not just the outputs: each phase must be checked against its
`recurrent-radix HIT: uid=N restored recurrent state at cached_len=C` line — a "hit" run
that silently prefilled fresh (or a "fresh" run that silently hit) voids the comparison,
exactly the way the probe's provenance check guards arms. The driver greps
`docker logs lease-<name>-serve` between phases.

USAGE (driven from the shell, one phase per call; the boot sequencing lives outside):
    python3 tools/gdn_hit_fresh_divergence.py --phase fresh  --boot 1 --out /home/pat/fixtures/minisgl-hit-divergence
    python3 tools/gdn_hit_fresh_divergence.py --phase hit    --boot 1 ...
    python3 tools/gdn_hit_fresh_divergence.py --phase partial --boot 1 ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import toolcall_degen_probe as degen  # SYSTEM, FILLER, HERMES_TOOLS — one source of truth

SESSION_JSON = "/home/pat/.hermes/webui/sessions/ebbc1dd0903b.json"
BASE = "http://127.0.0.1:1919"
# The session's final context the collapsing request saw: messages 0..28 (29 was its own
# degenerate output, 30/31 the injected retry + clean turn — excluded).
CTX_END = 29
# Padding so the prefill lands at production depth: the live request cached 43,616 tokens.
# Transcript is ~24k tok + tools ~5k; this filler (~64k chars ~ 16k tok) brings X to ~45k.
PAD_LINES = 820
EXTRA_USER_TURN = ("New information: the review deadline moved up. Continue from where the "
                   "previous turn left off and report your current verdict briefly.")


def build_pad() -> str:
    return ("Reference material (do not summarise; it is background only).\n" + "\n".join(
        f"  P{i:04d}. Peripheral 0x{i:03X} — bit {i % 8} gates sub-block {i % 7}; "
        f"read-returns-zero on this silicon revision; errata {i} applies after reset."
        for i in range(PAD_LINES)))


def build_messages(session: dict, phase: str) -> list:
    msgs = []
    src = session["context_messages"][:CTX_END]
    for i, m in enumerate(src):
        role, content = m.get("role"), m.get("content") or ""
        if i == 0 and role == "user":
            content = build_pad() + "\n\n" + content  # depth padding, production-faithful tail
        if role == "assistant":
            out = {"role": "assistant"}
            if content:
                out["content"] = content
            if m.get("reasoning"):
                out["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                # normalise: the WebUI stores extra fields (call_id, response_item_id) that a
                # strict request model may reject — keep the canonical OpenAI triple only.
                out["tool_calls"] = [
                    {"id": c.get("id"), "type": "function",
                     "function": {"name": (c.get("function") or {}).get("name"),
                                  "arguments": (c.get("function") or {}).get("arguments")}}
                    for c in m["tool_calls"]]
            if "content" not in out and "tool_calls" not in out:
                continue
            msgs.append(out)
        elif role == "tool":
            msgs.append({"role": "tool", "tool_call_id": m.get("tool_call_id"),
                         "content": content})
        else:  # user
            msgs.append({"role": "user", "content": content})
    if phase == "partial":
        msgs.append({"role": "user", "content": EXTRA_USER_TURN})
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("fresh", "hit", "partial"))
    ap.add_argument("--boot", required=True, help="boot number (1 or 2) — labels the output")
    ap.add_argument("--out", default="/home/pat/fixtures/minisgl-hit-divergence")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    with open(SESSION_JSON) as fh:
        session = json.load(fh)
    messages = [{"role": "system", "content": degen.SYSTEM}] + build_messages(session, args.phase)
    payload = {"model": args.model, "messages": messages, "tools": degen.HERMES_TOOLS,
               "max_tokens": args.max_tokens, "top_k": 1}  # top_k=1 = argmax, the only param sent
    blob = json.dumps(payload, sort_keys=True).encode()
    sha = hashlib.sha256(blob).hexdigest()[:16]

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=blob,
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=args.timeout).read())
    dt = round(time.time() - t0, 1)

    choice = resp["choices"][0]
    msg = choice["message"]
    rec = {
        "phase": args.phase, "boot": args.boot, "request_sha": sha,
        "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"), "elapsed_s": dt,
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": [{"name": (c.get("function") or {}).get("name"),
                        "arguments": (c.get("function") or {}).get("arguments")}
                       for c in (msg.get("tool_calls") or [])],
    }
    out = os.path.join(args.out, f"boot{args.boot}-{args.phase}.json")
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in
                      ("phase", "boot", "prompt_tokens", "completion_tokens",
                       "finish_reason", "elapsed_s")}))
    print(f"request_sha={sha} (phases must match across boots)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
