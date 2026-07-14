"""Localize the CCA chunk-boundary conv seam: is it the STORED window (init_states) or the MATH?

Hooks zaya_cca.cca_prefill_qk for CCA layer 0. For every prefill call it records:
  - qk_new  [P, C]           the conv INPUT for this pass's tokens
  - init_states [R, C, TP]   the cached window handed in (chunk continuations only)
  - seg_pos, req_id, is a continuation?

single mode: one pass, all tokens. chunked mode: several passes.
--compare loads both and, for each chunk boundary b in the chunked run, checks whether
init_states[req, :, :] equals the single-pass qk_new at the two positions just BEFORE b
([b-2],[b-1]). If they match, the recurrent STORAGE is faithful and the seam is in the conv MATH
(the boundary token's kernel arithmetic). If they differ, the store/gather of conv_states is lossy.
"""
import argparse, torch


def build(model, max_extend):
    from minisgl.llm import LLM
    return LLM(model_path=model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
               memory_ratio=0.85, attention_backend="hip", max_running_req=2,
               gdn_radix=False, cache_type="radix", max_extend_tokens=max_extend)


def run(a):
    from minisgl.core import SamplingParams
    import zaya_cca
    llm = build(a.model, a.max_extend)
    tok = llm.tokenizer
    base = ("The quick brown fox jumps over the lazy dog near the riverbank while the sun sets. "
            "In distant mountains, snow falls softly over ancient pines and quiet valleys. ") * 4
    ids = tok.encode(base, add_special_tokens=True)[:a.ntok]

    calls = []  # list of dicts, per prefill pass (layer 0 only)
    orig = zaya_cca.cca_prefill_qk
    layer0_seen = {"n": 0}

    def hook(qk_new, conv, init_states, seg_pos, req_id, slot, is_last, *rest):
        # Only capture CCA layer 0: it is the FIRST cca_prefill_qk call of each forward pass.
        sp = seg_pos.detach().cpu().clone()
        idx = layer0_seen["n"]
        layer0_seen["n"] += 1
        out = orig(qk_new, conv, init_states, seg_pos, req_id, slot, is_last, *rest)
        if idx % 40 == 0:  # 40 CCA layers -> layer 0 starts each pass
            calls.append({
                "qk_new": qk_new.detach().float().cpu().clone(),
                "init": init_states.detach().float().cpu().clone(),
                "qk_out": out.detach().float().cpu().clone(),  # kernel OUTPUT (conv result)
                "seg_pos": sp,
                "req_id": req_id.detach().cpu().clone(),
            })
        return out
    zaya_cca.cca_prefill_qk = hook
    llm.generate([list(ids)], SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True))
    torch.save({"mode": a.mode, "calls": calls, "ntok": len(ids)}, a.out)
    print(f"saved {a.out} passes={len(calls)} "
          f"seg_starts={[int(c['seg_pos'][0]) for c in calls]} "
          f"lens={[c['qk_new'].shape[0] for c in calls]}", flush=True)


def cmp(single_p, chunk_p):
    s = torch.load(single_p); c = torch.load(chunk_p)
    sc = s["calls"][0]  # single pass: one call, qk_new = all tokens
    sqk = sc["qk_new"]  # [N, C]  conv INPUT
    sout = sc["qk_out"]  # [N, C]  conv OUTPUT (single, ground truth)
    N, C = sqk.shape
    TP = c["calls"][0]["init"].shape[2]
    lens = [call["qk_new"].shape[0] for call in c["calls"]]
    print(f"single tokens={N} C={C} TP={TP}  chunk passes={len(c['calls'])} lens={lens}")
    boundary = 0
    print(f"\nAt each chunk boundary: INPUT faithfulness (init vs single window) AND OUTPUT divergence")
    print(f"(conv_out of boundary token, chunk vs single):")
    for ci, call in enumerate(c["calls"]):
        L = call["qk_new"].shape[0]
        seg0 = int(call["seg_pos"][0])
        cont = seg0 == 0 and boundary > 0
        if cont:
            b = boundary
            init = call["init"][0]                        # [C, TP]
            want = sqk[b - TP:b].transpose(0, 1)          # [C, TP]  what init SHOULD be
            din = (init - want).abs().max().item()
            # OUTPUT: boundary token is chunk-local row 0; compare to single row b
            out_b = call["qk_out"][0]                     # [C]  chunk boundary-token conv output
            single_b = sout[b]                            # [C]  single same token
            dout = (out_b - single_b).abs().max().item()
            # also: is the boundary token's INPUT qk_new row identical? (the current token col)
            din_cur = (call["qk_new"][0] - sqk[b]).abs().max().item()
            print(f"  boundary @ {b:3d} (page_aligned={b % 16 == 0}): "
                  f"|init-window|={din:.3e}  |cur_tok_in Δ|={din_cur:.3e}  ->  |conv_out Δ|={dout:.3e}")
        boundary += L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "chunked"])
    ap.add_argument("--model", default="/root/.cache/huggingface/ZAYA1-8B-RXF-h32")
    ap.add_argument("--ntok", type=int, default=160)
    ap.add_argument("--max-extend", type=int, default=8192)
    ap.add_argument("--out", default="/engine/tools/_seamp.pt")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        cmp(*a.compare)
    else:
        run(a)


if __name__ == "__main__":
    main()
