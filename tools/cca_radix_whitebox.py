"""White-box CCA recurrent-radix snapshot fidelity probe.

Directly compares the recurrent-state SNAPSHOT the radix captured for a prefix boundary against a
FRESH reference state produced by re-forwarding the same prefix — with NO KV/attention/fp8 confound.
This answers deliverable-1's core question: is the captured (conv_states, prev_hs) snapshot at a
page-aligned boundary bit-equal to what a fresh forward to that boundary produces?

Method (single process, recurrent radix ON):
  1. Reference: forward prefix P (len == boundary) via generate(max_tokens=1); read the LIVE cca_state
     slot for that uid right after prefill (state@boundary), snapshot it. This is the ground truth.
  2. The same generate also inserted node@boundary with a captured rec_state (fallback path). Walk the
     radix tree, pull the node's rec_state (conv, prev), and diff vs the reference.
  3. Repeat for a chunked prefix (STASH path) by lowering max_extend so P chunks.

Prints max abs diff per (conv, prev). ~0 => snapshot faithful (bug is downstream: KV/attn/fp8).
Nonzero => the snapshot itself is corrupt (root cause is capture/restore of the recurrent state).
"""
import argparse
import torch


def build_llm(model, max_extend):
    from minisgl.llm import LLM
    return LLM(
        model_path=model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
        memory_ratio=0.85, attention_backend="hip", max_running_req=4,
        gdn_radix=True, cache_type="radix", max_extend_tokens=max_extend,
    )


def read_live_slot(llm, uid):
    """conv_states+prev_hs for uid's live slot, cloned (all CCA layers)."""
    slot = llm.cca_slots.slot_for(uid)
    st = llm.engine.ctx.cca_state
    return slot, st.conv_states[:, slot].clone().float().cpu(), st.prev_hs[:, slot].clone().float().cpu()


def walk_rec_states(llm):
    """All (boundary_len, conv, prev) snapshots stored on radix nodes."""
    pc = llm.cache_manager.prefix_cache
    out = []

    def cumlen(node):
        n, L = node, 0
        while not n.is_root():
            L += n.length
            n = n.parent
        return L

    stack = [pc.root_node]
    while stack:
        node = stack.pop()
        if getattr(node, "rec_state", None) is not None:
            conv, prev = node.rec_state
            out.append((cumlen(node), conv.float().cpu(), prev.float().cpu()))
        stack.extend(node.children.values())
    return out


def probe(llm, ids, boundary, tag):
    from minisgl.core import SamplingParams
    assert len(ids) == boundary, "prefix length must equal boundary (single-pass fresh forward)"
    # Read the LIVE slot inside the capturing generate's sampler hook. The model forward (which writes
    # conv/prev to the slot) runs BEFORE sampler.sample, so at hook time the slot holds the genuine
    # FRESH state@boundary — the ground-truth reference. This generate has no prior radix, so it is a
    # true from-zero forward. The capture (_maybe_capture_rec_state -> clone_slot) fires later in
    # _process_last_data of the SAME iteration; comparing it to this read tells us if capture is faithful.
    ref = {}
    orig = llm.engine.sampler.sample

    def hook(logits, args):
        if "conv" not in ref:
            uid = next(iter(llm.status_map))
            s, c, p = read_live_slot(llm, uid)
            ref["slot"], ref["conv"], ref["prev"] = s, c, p
        return orig(logits, args)

    llm.engine.sampler.sample = hook
    # max_tokens=2 so the req is NOT finished after prefill -> capture goes through the PREFILL branch
    # (_process_last_data cache_req + _maybe_capture_rec_state), exactly the path the black-box uses.
    llm.generate([list(ids)], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True))
    llm.engine.sampler.sample = orig

    snaps = walk_rec_states(llm)
    print(f"\n[{tag}] boundary={boundary} rec_snapshots@={[s[0] for s in snaps]} ref_slot={ref.get('slot')}", flush=True)
    match = [s for s in snaps if s[0] == boundary]
    if not match:
        print(f"[{tag}] NO snapshot at boundary {boundary}")
        return
    _, cap_conv, cap_prev = match[0]
    # clone_slot keeps the s:s+1 dim -> cap_conv [L,1,C,TP], cap_prev [L,1,hidden]; read_live_slot
    # indexed [:,slot] -> ref [L,C,TP]/[L,hidden]. Squeeze the singleton so the diff is elementwise.
    cap_conv = cap_conv.squeeze(1)
    cap_prev = cap_prev.squeeze(1)
    assert cap_conv.shape == ref["conv"].shape, (cap_conv.shape, ref["conv"].shape)
    assert cap_prev.shape == ref["prev"].shape, (cap_prev.shape, ref["prev"].shape)
    dc = (cap_conv - ref["conv"]).abs().max().item()
    dp = (cap_prev - ref["prev"]).abs().max().item()
    print(f"[{tag}] max|snap.conv - fresh.conv|={dc:.6e}   max|snap.prev - fresh.prev|={dp:.6e}")
    print(f"[{tag}] VERDICT: {'SNAPSHOT FAITHFUL (bug downstream: KV/attn/fp8)' if (dc<1e-3 and dp<1e-3) else 'SNAPSHOT CORRUPT (recurrent capture bug)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/.cache/huggingface/ZAYA1-8B-RXF-h32")
    ap.add_argument("--max-extend", type=int, default=256)
    ap.add_argument("--boundary", type=int, default=32)
    args = ap.parse_args()
    # ONE boundary per fresh process (fresh radix tree) -> no cross-probe restore contamination.
    llm = build_llm(args.model, args.max_extend)
    tok = llm.tokenizer
    base = ("The quick brown fox jumps over the lazy dog near the riverbank while the sun sets. "
            "In distant mountains, snow falls softly over ancient pines and quiet valleys. ") * 6
    ids = tok.encode(base, add_special_tokens=True)
    probe(llm, ids[:args.boundary], args.boundary, f"FALLBACK@{args.boundary}")


if __name__ == "__main__":
    main()
