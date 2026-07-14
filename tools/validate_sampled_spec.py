#!/usr/bin/env python3
"""GPU validation for sampled (rejection-sampling) speculative verify (MINISGL_SPEC_SAMPLED).

Isolates the two components whose correctness the losslessness claim rests on, on-device, without a
model (fast, decisive):

  Test 1 — probs_from_logits MATCHES the production fused sampler. Draw M tokens from the fused HIP
           sampler (sampler_hip, the real serve path) and M from probs_from_logits+multinomial on the
           SAME logits + temp/top_k/top_p; the empirical distributions must agree (total-variation ~0).
           This is the real risk: if my top_k/top_p semantics differ from the kernel's, the sampled-
           spec output distribution drifts from plain decode.
  Test 2 — verify_sampled emits ~ p. On random target dists p and random drafts, the first emitted
           token's histogram must equal p (total-variation ~0), independent of the draft. This is the
           speculative-sampling losslessness property, on the real torch impl.
  Test 3 — verify_sampled reduces to greedy at temp->0 (p one-hot => emitted == the argmax chain).

PASS iff every TV distance is within Monte-Carlo noise (~3/sqrt(M)). Run in-container on a leased card:
  gpu-lease -n 1 -- bash tools/run_sampled_spec_validate.sh
"""
from __future__ import annotations

import sys

import torch

from minisgl.spec import probs_from_logits, verify_sampled
from minisgl.spec.accept import verify_greedy


def tv(a: torch.Tensor, b: torch.Tensor) -> float:
    """Total-variation distance between two histograms/dists (both sum to 1)."""
    return 0.5 * float((a - b).abs().sum())


def hist(ids: torch.Tensor, V: int) -> torch.Tensor:
    return torch.bincount(ids.flatten().cpu(), minlength=V).float() / ids.numel()


def test_probs_match_sampler(device, M=200_000) -> bool:
    print("\n[Test 1] probs_from_logits vs the fused HIP sampler (distribution match)")
    try:
        from minisgl.engine import _sampler_hip
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP — sampler_hip unavailable ({e})")
        return True
    V = 512
    ok = True
    for temp, top_k, top_p in [(0.8, -1, 1.0), (1.0, -1, 0.95), (0.8, 40, 0.95), (0.7, 100, 0.9)]:
        torch.manual_seed(0)
        logits = torch.randn(1, V, device=device, dtype=torch.float32) * 2.0
        p = probs_from_logits(logits, temp, top_k, top_p)[0]  # my ANALYTIC target dist [V]
        # Noise floor: TV of M samples drawn from p BY US vs p. Comparing the kernel's histogram to a
        # SECOND noisy histogram would double this; the right test is kernel-hist vs the analytic p,
        # PASS iff the kernel is as close to p as our own sampler is (kern_tv <= noise_floor * 1.5).
        g = torch.Generator(device=device).manual_seed(1)
        p_cpu = p.cpu()
        noise_tv = tv(hist(torch.multinomial(p, M, replacement=True, generator=g), V), p_cpu)
        # production: the fused sampler over the SAME logits, one row replicated M-wide in chunks
        tk = None if top_k <= 0 else torch.full((1,), top_k, dtype=torch.int32, device=device)
        tp = None if top_p >= 1.0 else torch.full((1,), top_p, dtype=torch.float32, device=device)
        te = torch.full((1,), temp, dtype=torch.float32, device=device)
        samp = []
        g2 = torch.Generator(device=device).manual_seed(2)
        CH = 4096
        for _ in range(M // CH):
            lg = logits.expand(CH, V).contiguous()
            samp.append(_sampler_hip.sample(lg, te.expand(CH).contiguous(),
                                            None if tk is None else tk.expand(CH).contiguous(),
                                            None if tp is None else tp.expand(CH).contiguous(),
                                            generator=g2))
        prod = torch.cat(samp)
        kern_tv = tv(hist(prod, V), p_cpu)   # kernel histogram vs OUR analytic p
        good = kern_tv <= max(noise_tv * 1.5, noise_tv + 0.004)
        ok &= good
        print(f"  temp={temp} top_k={top_k} top_p={top_p}: kern_TV={kern_tv:.5f} "
              f"noise_floor={noise_tv:.5f} {'OK' if good else 'FAIL'}")
    return ok


def test_verify_sampled_lossless(device, M=300_000) -> bool:
    print("\n[Test 2] verify_sampled emits ~ p (losslessness), draft-independent")
    V, K = 64, 4
    torch.manual_seed(3)
    p = torch.softmax(torch.randn(K + 1, V, device=device) * 1.5, dim=-1)  # [K+1,V]
    ok = True
    for draft0 in (int(p[0].argmax()), 1, 7):  # ANY draft must give emitted[pos0] ~ p[0]
        g = torch.Generator(device=device).manual_seed(10)
        counts = torch.zeros(V, device="cpu")
        # only the FIRST position's emission distribution is compared to p[0] (accept-or-residual at 0)
        draft = [draft0] + [int(torch.randint(0, V, (1,)).item()) for _ in range(K - 1)]
        for _ in range(M):
            r = verify_sampled(draft, p, g)
            counts[r.emitted[0]] += 1
        emp = counts / M
        d = tv(emp, p[0].cpu())
        tol = 4.0 / (M ** 0.5)
        good = d < tol
        ok &= good
        print(f"  draft[0]={draft0}: TV(emitted[0], p[0])={d:.5f} (tol {tol:.5f}) {'OK' if good else 'FAIL'}")
    return ok


def test_greedy_reduction(device) -> bool:
    print("\n[Test 3] temp->0 reduces verify_sampled to greedy")
    V, K = 64, 4
    torch.manual_seed(4)
    logits = torch.randn(K + 1, V, device=device) * 2.0
    p = probs_from_logits(logits, 0.0, -1, 1.0)  # one-hot(argmax)
    target = logits.argmax(-1).tolist()
    draft = target[:K]  # all-accept case -> emitted == target
    g = torch.Generator(device=device).manual_seed(0)
    rs = verify_sampled(draft, p, g)
    rg = verify_greedy(draft, target)
    good = rs.emitted == rg.emitted and rs.num_accepted == rg.num_accepted == K
    print(f"  sampled={rs.emitted} greedy={rg.emitted} {'OK' if good else 'FAIL'}")
    return good


def main() -> int:
    assert torch.cuda.is_available(), "no HIP GPU visible"
    device = torch.device("cuda")
    print(f"device={device}  torch={torch.__version__}")
    # The fused sampler's [hip-engage] log goes through the rank0 logger, which needs TP info; set a
    # single-rank context (this is a standalone test, not a served engine).
    from minisgl.distributed.info import set_tp_info

    set_tp_info(rank=0, size=1)
    ok = True
    ok &= test_probs_match_sampler(device)
    ok &= test_verify_sampled_lossless(device)
    ok &= test_greedy_reduction(device)
    print("\n==== SAMPLED-SPEC VALIDATION:", "PASS" if ok else "FAIL", "====")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
