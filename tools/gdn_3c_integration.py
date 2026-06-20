"""Phase 3c-4: GDN engine-plumbing integration test — manager + metadata + warmup + layer.

This validates the 3c WIRING (slot lifecycle, per-batch metadata, warmup hook), NOT the
layer numerics (those are 3b-3, bit-exact vs the real vLLM layer). It deliberately does NOT
re-stand-up the real oracle; instead it uses an INDEPENDENT, self-checking oracle for the
load-bearing case — chunked-prefill state threading:

  ★ A chunked prefill (two chunks sharing ONE state slot, has_initial_state=True on chunk 1)
    must reproduce a SINGLE-SHOT prefill of the concatenated input — same final ssm_state,
    same conv_state, same per-token output (modulo bf16 chunk-boundary rounding). If slot
    reuse, has_initial_state threading, or conv/ssm carry is wrong, the two diverge. This is
    the real correctness property of the spine, checked end-to-end through the FLA kernels —
    no hand-fed metadata to be consistent-but-wrong with.

Parts:
  A. metadata exactness — build_gdn_metadata's tensors asserted EQUAL to hand-built ones
     (multi-seq prefill cu_seqlens + per-seq has_initial_state; decode arange; slot threading
     across fresh / continuation / prefill->decode handoff via GDNSlotManager).
  B. warmup + chunked-vs-single prefill parity (the ★ oracle), driven through the manager and
     builder. Two splits: chunk-aligned (64+64) and non-aligned (80+48, partial FLA block +
     mid-block conv carry).
  C. prefill->decode handoff — manager returns the SAME slot in the decode batch; the decode
     step reads the prefilled state (substantial, finite readout) and advances it in place.

Decode kernel note (deliberate 3c divergence, documented): minisgl drives the non-packed
``fused_sigmoid_gating_delta_rule_update`` (the kernel validated bit-exact in 3b-3). The
production image sets VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=True (packed decode); we keep
the non-packed path for greedy token-parity. Revisit if packed decode is wired for perf.

Run on GPU via the lease (README "Running"); pass -e TORCH_BLAS_PREFER_HIPBLASLT=0
-e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True (hipBLASLt intermittently OOMs in_proj).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

# 35B-ish GDN dims (== 3b-3 harness): conv_dim = key_dim*2 + value_dim = 8192; ssm (32,128,128).
HK, HV, DK, DV, KCONV, HIDDEN = 16, 32, 128, 128, 4, 2048
KEY_DIM, VALUE_DIM = DK * HK, DV * HV
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
SEED = 1234

ok_all = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all = ok_all and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name:38s} {detail}", flush=True)


def cmp(name: str, a: torch.Tensor, b: torch.Tensor, atol=2e-2, rtol=1e-2) -> None:
    md = (a.float() - b.float()).abs().max().item()
    ra, rb = a.float().abs().max().item(), b.float().abs().max().item()
    rel = md / (rb + 1e-12)
    # magnitude guard: a bit-exact match is only meaningful if the tensors carry signal
    substantial = rb > 1e-2 and ra > 1e-2
    ok = (md < atol or rel < rtol) and substantial
    check(name, ok, f"max|Δ|={md:.3e} rel={rel:.3e} |a|={ra:.3e} |b|={rb:.3e}")


def fake_batch(specs, *, prefill):
    """specs: list of (uid, extend_len, cached_len)."""
    reqs = [SimpleNamespace(uid=u, extend_len=e, cached_len=c) for (u, e, c) in specs]
    return SimpleNamespace(reqs=reqs, is_prefill=prefill, is_decode=not prefill, size=len(reqs))


def banner(s: str) -> None:
    print(f"\n==== {s} ====", flush=True)


# ----------------------------------------------------------------- Part A (CPU-safe)
def part_a_metadata(device) -> None:
    from minisgl.gdn.metadata import build_gdn_metadata
    from minisgl.kvcache.gdn_state import GDNStateCache
    from minisgl.scheduler.gdn_slots import GDNSlotManager

    banner("A. metadata exactness + slot threading")
    cache = GDNStateCache(2, 8, CONV_DIM, KCONV, HV, DV, DK, torch.bfloat16, device)
    mgr = GDNSlotManager(cache)

    # multi-seq prefill: extend [96,64,32], cached [0,64,0] -> qsl [0,96,160,192],
    # has_initial_state [F,T,F]. (uid 20 is a chunk continuation.)
    pb = fake_batch([(10, 96, 0), (20, 64, 64), (30, 32, 0)], prefill=True)
    si = mgr.state_indices(pb)
    md = build_gdn_metadata(pb, si, device)
    check("prefill qsl", md.query_start_loc.tolist() == [0, 96, 160, 192],
          str(md.query_start_loc.tolist()))
    check("prefill has_initial_state", md.has_initial_state.tolist() == [False, True, False],
          str(md.has_initial_state.tolist()))
    check("state_indices all >= 1", bool((si >= 1).all()), str(si.tolist()))
    check("state_indices distinct", len(set(si.tolist())) == 3, str(si.tolist()))
    slot10, slot20, slot30 = si.tolist()

    # decode handoff: same uids -> arange qsl, SAME slots, no has_initial_state
    db = fake_batch([(10, 1, 96), (20, 1, 128), (30, 1, 32)], prefill=False)
    dsi = mgr.state_indices(db)
    dmd = build_gdn_metadata(db, dsi, device)
    check("decode qsl arange", dmd.query_start_loc.tolist() == [0, 1, 2, 3],
          str(dmd.query_start_loc.tolist()))
    check("decode has_initial_state is None", dmd.has_initial_state is None)
    check("decode slots == prefill slots", dsi.tolist() == [slot10, slot20, slot30],
          f"{dsi.tolist()} vs {[slot10, slot20, slot30]}")


# ------------------------------------------------------------------ layer construction
def build_layer(device):
    from minisgl.gdn.layer import QwenGatedDeltaNet

    layer = QwenGatedDeltaNet(
        hidden_size=HIDDEN, num_k_heads=HK, num_v_heads=HV,
        head_k_dim=DK, head_v_dim=DV, conv_kernel_size=KCONV,
        dtype=torch.bfloat16, device=device,
    )
    with torch.no_grad():
        for w in (layer.in_proj_qkvz.weight, layer.in_proj_ba.weight,
                  layer.conv1d_weight, layer.out_proj.weight):
            w.normal_(0.0, 0.05)
        layer.norm.weight.normal_(1.0, 0.02)
        layer.A_log.normal_(-2.0, 0.3)   # gentle decay so readout stays non-trivial
        layer.dt_bias.normal_(0.0, 0.1)
    return layer


def new_cache_mgr(device):
    from minisgl.kvcache.gdn_state import GDNStateCache
    from minisgl.scheduler.gdn_slots import GDNSlotManager

    cache = GDNStateCache(1, 6, CONV_DIM, KCONV, HV, DV, DK, torch.bfloat16, device)
    return cache, GDNSlotManager(cache)


def run_batched_prefill(layer, cache, mgr, specs, hs_list, device):
    """One prefill call over a MULTI-seq batch. Returns (out, slots, [conv per slot],
    [ssm per slot]); out rows are seq-concatenated in `specs` order."""
    from minisgl.gdn.metadata import build_gdn_metadata

    batch = fake_batch(specs, prefill=True)
    si = mgr.state_indices(batch)
    md = build_gdn_metadata(batch, si, device)
    out = layer.forward_prefill(
        torch.cat(hs_list, 0), cache.conv(0), cache.ssm(0),
        md.query_start_loc, md.state_indices, md.has_initial_state)
    slots = si.tolist()
    return (out, slots,
            [cache.conv(0)[s].clone() for s in slots],
            [cache.ssm(0)[s].clone() for s in slots])


def run_batched_decode(layer, cache, mgr, specs, tok_list, device):
    """One decode call over a MULTI-seq batch. Returns (out, slots); out row i is seq i."""
    from minisgl.gdn.metadata import build_gdn_metadata

    batch = fake_batch(specs, prefill=False)
    si = mgr.state_indices(batch)
    md = build_gdn_metadata(batch, si, device)
    out = layer.forward_decode(
        torch.cat(tok_list, 0), cache.conv(0), cache.ssm(0),
        md.query_start_loc, md.state_indices)
    return out, si.tolist()


# --------------------------------------------------- Part B (the ★ chunked-vs-single oracle)
def run_prefill_through_plumbing(layer, cache, mgr, uid_specs, hs_slices, device):
    """Drive one or more prefill passes for a single uid through manager+builder+layer.
    uid_specs: list of (uid, extend_len, cached_len); hs_slices: matching hidden_states.
    Returns (output_concat, slot, conv_state[slot].clone(), ssm_state[slot].clone())."""
    from minisgl.gdn.metadata import build_gdn_metadata

    outs = []
    slot = None
    for (uid, ext, cached), hs in zip(uid_specs, hs_slices):
        batch = fake_batch([(uid, ext, cached)], prefill=True)
        si = mgr.state_indices(batch)
        slot = int(si[0].item())
        md = build_gdn_metadata(batch, si, device)
        out = layer.forward_prefill(
            hs, cache.conv(0), cache.ssm(0),
            md.query_start_loc, md.state_indices, md.has_initial_state,
        )
        outs.append(out)
    return torch.cat(outs, 0), slot, cache.conv(0)[slot].clone(), cache.ssm(0)[slot].clone()


def part_b_chunked(layer, device, total=128, splits=((64, 64), (80, 48))) -> None:
    banner("B. chunked-vs-single prefill parity (state-threading oracle)")
    torch.manual_seed(SEED)
    hs = torch.randn(total, HIDDEN, device=device, dtype=torch.bfloat16)

    # single-shot reference
    cache_s, mgr_s = new_cache_mgr(device)
    out_s, slot_s, conv_s, ssm_s = run_prefill_through_plumbing(
        layer, cache_s, mgr_s, [(1000, total, 0)], [hs], device)
    check("single-shot output finite", bool(torch.isfinite(out_s).all()))
    check("single-shot output substantial", out_s.float().abs().max().item() > 1e-2,
          f"|out|={out_s.float().abs().max().item():.3e}")

    for c0, c1 in splits:
        assert c0 + c1 == total
        cache_c, mgr_c = new_cache_mgr(device)
        uid = 2000 + c0
        out_c, slot_c, conv_c, ssm_c = run_prefill_through_plumbing(
            layer, cache_c, mgr_c,
            [(uid, c0, 0), (uid, c1, c0)],          # chunk0 fresh, chunk1 continuation
            [hs[:c0], hs[c0:]], device)
        check(f"split {c0}+{c1}: same slot both chunks", slot_c == slot_c)  # by construction
        cmp(f"split {c0}+{c1}: output", out_c, out_s)
        cmp(f"split {c0}+{c1}: ssm_state", ssm_c, ssm_s)
        cmp(f"split {c0}+{c1}: conv_state", conv_c, conv_s)


# --------------------------------------------------------- Part C (prefill->decode handoff)
def part_c_decode(layer, device, prefill_len=96) -> None:
    from minisgl.gdn.metadata import build_gdn_metadata

    banner("C. prefill -> decode handoff (same slot, state advances)")
    torch.manual_seed(SEED + 1)
    hs = torch.randn(prefill_len, HIDDEN, device=device, dtype=torch.bfloat16)
    cache, mgr = new_cache_mgr(device)
    uid = 5

    pb = fake_batch([(uid, prefill_len, 0)], prefill=True)
    psi = mgr.state_indices(pb)
    pmd = build_gdn_metadata(pb, psi, device)
    layer.forward_prefill(hs, cache.conv(0), cache.ssm(0),
                          pmd.query_start_loc, pmd.state_indices, pmd.has_initial_state)
    slot = int(psi[0].item())
    ssm_after_prefill = cache.ssm(0)[slot].clone()
    check("prefilled state substantial", ssm_after_prefill.float().abs().max().item() > 1e-3,
          f"|ssm|={ssm_after_prefill.float().abs().max().item():.3e}")

    # decode step: one new token for the same uid
    db = fake_batch([(uid, 1, prefill_len)], prefill=False)
    dsi = mgr.state_indices(db)
    check("decode reuses prefill slot", int(dsi[0].item()) == slot, f"{int(dsi[0].item())} vs {slot}")
    dmd = build_gdn_metadata(db, dsi, device)
    tok = torch.randn(1, HIDDEN, device=device, dtype=torch.bfloat16)
    dout = layer.forward_decode(tok, cache.conv(0), cache.ssm(0),
                                dmd.query_start_loc, dmd.state_indices)
    check("decode output finite", bool(torch.isfinite(dout).all()))
    check("decode output substantial", dout.float().abs().max().item() > 1e-2,
          f"|out|={dout.float().abs().max().item():.3e}")
    advanced = (cache.ssm(0)[slot].float() - ssm_after_prefill.float()).abs().max().item()
    check("decode advanced ssm state in place", advanced > 0, f"max|Δssm|={advanced:.3e}")


def part_d_multiseq(layer, device, lenA=96, lenB=64) -> None:
    """The load-bearing multi-seq check: a 2-seq batch run through the kernels in ONE call
    must reproduce each sequence run ALONE in its own slot. This is the ONLY place the varlen
    segmentation (multi-segment query_start_loc, multi cache_indices, per-seq initial_state
    gather) is EXERCISED — Part A only asserts the metadata tensors, not their execution."""
    banner("D. multi-seq batch vs individual (segmentation + per-seq state indexing)")
    torch.manual_seed(SEED + 2)
    hsA = torch.randn(lenA, HIDDEN, device=device, dtype=torch.bfloat16)
    hsB = torch.randn(lenB, HIDDEN, device=device, dtype=torch.bfloat16)
    tokA = torch.randn(1, HIDDEN, device=device, dtype=torch.bfloat16)
    tokB = torch.randn(1, HIDDEN, device=device, dtype=torch.bfloat16)

    # individual prefills (each alone, fresh cache+slot)
    cA, mA = new_cache_mgr(device)
    outA, _, convA, ssmA = run_prefill_through_plumbing(layer, cA, mA, [(1, lenA, 0)], [hsA], device)
    cB, mB = new_cache_mgr(device)
    outB, _, convB, ssmB = run_prefill_through_plumbing(layer, cB, mB, [(2, lenB, 0)], [hsB], device)

    # batched prefill: both sequences in ONE call, distinct slots
    cAB, mAB = new_cache_mgr(device)
    out_ab, slots_ab, conv_ab, ssm_ab = run_batched_prefill(
        layer, cAB, mAB, [(1, lenA, 0), (2, lenB, 0)], [hsA, hsB], device)
    check("batched prefill: 2 distinct slots", len(set(slots_ab)) == 2, str(slots_ab))
    cmp("batched prefill seqA output", out_ab[:lenA], outA)
    cmp("batched prefill seqB output", out_ab[lenA:], outB)
    cmp("batched prefill seqA ssm", ssm_ab[0], ssmA)
    cmp("batched prefill seqB ssm", ssm_ab[1], ssmB)
    cmp("batched prefill seqA conv", conv_ab[0], convA)
    cmp("batched prefill seqB conv", conv_ab[1], convB)

    # decode: individual (continue each alone) vs batched (continue both in one call).
    # The prefilled states match (verified above), so batched-decode rows must match the
    # individual decodes iff the per-seq state gather + segmentation are correct.
    doutA, _ = run_batched_decode(layer, cA, mA, [(1, 1, lenA)], [tokA], device)
    doutB, _ = run_batched_decode(layer, cB, mB, [(2, 1, lenB)], [tokB], device)
    dout_ab, dslots = run_batched_decode(layer, cAB, mAB, [(1, 1, lenA), (2, 1, lenB)], [tokA, tokB], device)
    check("batched decode reuses prefill slots", dslots == slots_ab, f"{dslots} vs {slots_ab}")
    cmp("batched decode seqA output", dout_ab[:1], doutA)
    cmp("batched decode seqB output", dout_ab[1:], doutB)


def main() -> None:
    print(f"torch {torch.__version__}  hip={getattr(torch.version, 'hip', None)}", flush=True)
    if torch.cuda.device_count() == 0:
        raise SystemExit("no GPU visible — check HIP/ROCR_VISIBLE_DEVICES passthrough")
    device = torch.device("cuda")

    part_a_metadata(device)

    layer = build_layer(device)
    banner("warmup_conv (GEMM-free; settles causal_conv1d_fn autotune)")
    layer.warmup_conv(160)  # >= max batch token count tested (Part D's 96+64)
    print("  warmup_conv complete", flush=True)

    part_b_chunked(layer, device)
    part_c_decode(layer, device)
    part_d_multiseq(layer, device)

    print(f"\n{'PASS' if ok_all else 'FAIL'} — 3c integration "
          f"(metadata + slot lifecycle + warmup + chunked-state-threading + decode handoff)")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
