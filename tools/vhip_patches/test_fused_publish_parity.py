"""Fused in-kernel per-position publish == scratch + Python scatter, bit for bit.

The verify kernels now support two publish contracts (one templated core, `INPLACE`):
  * scratch  — write every token's state to [max_qlen,N,...] scratch, caller scatters it into
               conv_state/ssm_state[slots[n,t]] with a Python loop (the old path);
  * fused    — kernel resolves the load slot from slots[n, num_accepted[n]-1] and stores each
               token's state straight into the paged cache (what vLLM's fla kernel does).

Both must leave IDENTICAL cache contents and identical core output. This checks that against a
realistic paged (as_strided, page-padded slot stride) cache, over several num_accepted values.

One deliberate difference is asserted rather than ignored: when a sequence is SHORTER than max_qlen,
the old Python loop clamps and re-writes the final state into the surplus slots, while the fused
kernel simply doesn't touch them. Those slots are never read back (the next step loads
num_accepted-1, and num_accepted <= qlen), so the test checks the reachable slots match and reports
the surplus ones separately.

Run: one GPU, seconds, no serve.
"""

import sys

import torch

import gdn_hip

torch.manual_seed(0)
DEV = "cuda"
NK, NV, HK, HV = 8, 16, 128, 128
C = HK * NK * 2 + HV * NV
W, NUM_SPEC = 4, 2
MAX_QLEN = NUM_SPEC + 1
CONV_W = W - 1
SLOTS = 64
SCALE = HK ** -0.5

WT = (torch.randn(C, W, device=DEV) * 0.1).float()
A_LOG = torch.randn(NV, device=DEV).float()
DT_BIAS = torch.randn(NV, device=DEV).float()
fails = []


def paged(dtype, elems_shape, off, page, n_slots=SLOTS, raw=None):
    return raw, page


def make_cache(ssm_dtype):
    """conv_state + ssm_state as vLLM carves them: one raw buffer, page-padded slot stride."""
    conv_elems, ssm_elems = C * (CONV_W + NUM_SPEC), NV * HV * HK
    page = conv_elems + ssm_elems + 137
    rawc = torch.zeros(SLOTS * page, device=DEV, dtype=torch.float32)
    raws = torch.zeros(SLOTS * page, device=DEV, dtype=ssm_dtype)
    cs = torch.as_strided(rawc, (SLOTS, C, CONV_W + NUM_SPEC), (page, CONV_W + NUM_SPEC, 1), 0)
    ss = torch.as_strided(raws, (SLOTS, NV, HV, HK), (page, HV * HK, HK, 1), 0)
    cs.copy_(torch.randn(SLOTS, C, CONV_W + NUM_SPEC, device=DEV) * 0.1)
    ss.copy_((torch.randn(SLOTS, NV, HV, HK, device=DEV) * 0.1).to(ssm_dtype))
    return cs, ss


def split(conv_out):
    q, k, v = conv_out.split([HK * NK, HK * NK, HV * NV], dim=-1)
    return (q.reshape(-1, NK, HK).contiguous(), k.reshape(-1, NK, HK).contiguous(),
            v.reshape(-1, NV, HV).contiguous())


def run_case(label, qlens, num_acc, ssm_dtype, idx_dtype=torch.int32):
    n = len(qlens)
    T = sum(qlens)
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32, device=DEV)
    x = torch.randn(T, C, device=DEV).float()
    a = torch.randn(T, NV, device=DEV).float()
    b = torch.randn(T, NV, device=DEV).float()
    # disjoint slot rows per sequence, all > 0
    # vLLM's spec_state_indices_tensor / num_accepted_tokens are INT32 (gdn_attn.py:127,162) and the
    # kernels take that natively; minisgl passes int64. Exercise the dtype the serve actually uses.
    slots = (torch.arange(1, n * MAX_QLEN + 1, device=DEV, dtype=idx_dtype).view(n, MAX_QLEN))
    acc = torch.tensor(num_acc, device=DEV, dtype=idx_dtype)
    hi = torch.ones(n, dtype=torch.uint8, device=DEV)

    # ---- A: scratch contract + Python publish loop (the old path) ----
    csA, ssA = make_cache(ssm_dtype)
    ref_c, ref_s = csA.clone(), ssA.clone()
    load = slots.long().gather(1, (acc.long().clamp(min=1) - 1)[:, None]).squeeze(1)
    convA, cscrA = gdn_hip.causal_conv1d_fwd_verify(x, WT, None, cu, load, hi, csA, MAX_QLEN, 1)
    qA, kA, vA = split(convA)
    coreA, sscrA = gdn_hip.gdn_prefill_verify(qA, kA, vA, a, b, A_LOG, DT_BIAS, cu, load, hi,
                                              ssA, MAX_QLEN, SCALE, 1)
    ql = torch.tensor(qlens, device=DEV, dtype=torch.long)
    rows = torch.arange(n, device=DEV)
    last = (ql - 1).clamp(min=0)
    for t in range(MAX_QLEN):
        src = torch.minimum(torch.full_like(last, t), last)
        tgt = slots[:, t].long()
        ssA[tgt] = sscrA[src, rows].to(ssA.dtype)
        csA[tgt, :, :CONV_W] = cscrA[src, rows].to(csA.dtype)

    # ---- B: fused in-kernel publish ----
    csB, ssB = make_cache(ssm_dtype)
    csB.copy_(ref_c); ssB.copy_(ref_s)          # same starting cache as A
    convB, cscrB = gdn_hip.causal_conv1d_fwd_verify(x, WT, None, cu, None, hi, csB, MAX_QLEN, 1,
                                                    slots, acc)
    qB, kB, vB = split(convB)
    coreB, sscrB = gdn_hip.gdn_prefill_verify(qB, kB, vB, a, b, A_LOG, DT_BIAS, cu, None, hi,
                                              ssB, MAX_QLEN, SCALE, 1, slots, acc)

    print(f"\n[{label}]  qlens={qlens} num_accepted={num_acc} ssm={ssm_dtype} idx={idx_dtype}")
    if cscrB.numel() or sscrB.numel():
        fails.append(f"{label}: fused path still allocated a scratch "
                     f"(conv {cscrB.numel()}, ssm {sscrB.numel()})")
    print(f"    fused scratch elided: conv={cscrB.numel()} ssm={sscrB.numel()} (want 0/0)")

    for nm, ra, rb in (("core", coreA, coreB), ("conv_out", convA, convB)):
        same = torch.equal(ra.float(), rb.float())
        print(f"    {nm:10s} bitexact={same}")
        if not same:
            fails.append(f"{label}: {nm} differs")

    # reachable slots = positions 0..qlen-1 of each sequence; surplus = the rest
    reach = torch.zeros(n, MAX_QLEN, dtype=torch.bool, device=DEV)
    for i, q_ in enumerate(qlens):
        reach[i, :q_] = True
    r_slots, s_slots = slots[reach].long(), slots[~reach].long()
    for nm, ca, cb, sel in (("ssm_state", ssA, ssB, r_slots), ("conv_state", csA, csB, r_slots)):
        if sel.numel() == 0:
            continue
        A_, B_ = ca[sel].float(), cb[sel].float()
        same = torch.equal(A_, B_)
        print(f"    {nm:10s} reachable slots bitexact={same} "
              f"maxdiff={(A_ - B_).abs().max().item():.3e}")
        if not same:
            fails.append(f"{label}: {nm} differs on reachable slots")
    if s_slots.numel():
        d = (ssA[s_slots].float() - ssB[s_slots].float()).abs().max().item()
        print(f"    (surplus slots {s_slots.tolist()}: ssm maxdiff={d:.3e} — expected nonzero, "
              f"never read back)")


for dt, ix in ((torch.float32, torch.int32), (torch.bfloat16, torch.int32),
               (torch.float32, torch.int64)):
    # all sequences full length -> no surplus slots, must be identical everywhere
    run_case("full-length, acc=1", [MAX_QLEN], [1], dt, ix)
    run_case("full-length, acc=2", [MAX_QLEN], [2], dt, ix)
    run_case("full-length, acc=3", [MAX_QLEN], [3], dt, ix)
    run_case("batch of 3, mixed acc", [MAX_QLEN] * 3, [1, 2, 3], dt, ix)
    # a short sequence -> surplus slots exist; reachable slots must still match
    run_case("ragged qlens", [MAX_QLEN, 2, 1], [3, 2, 1], dt, ix)

print("\n" + ("FAIL: " + "; ".join(fails) if fails else
             "ALL BIT-EXACT: fused in-kernel publish == scratch + Python scatter"))
sys.exit(1 if fails else 0)
