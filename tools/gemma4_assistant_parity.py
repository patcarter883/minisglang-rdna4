"""Gemma-4 assistant drafter vs the transformers reference, on synthetic target K/V (1 GPU, TP=1).

    gpu-lease -n 1 -- docker run ... python tools/gemma4_assistant_parity.py

Both sides get the same (embedding, seed hidden) input and the same target K/V: the reference as
`shared_kv_states` tensors, ours through a paged main pool (page 16) and a ring-style SWA pool
(page 1) read by the HIP decode kernel. Ground truth is the reference in fp32; ours (bf16) must be
no further from it than the reference's own bf16 run, per context length (below and above the
sliding window): the mean relative error over 8 chained steps (each fed the fp32 reference's token
and seed), for the logits and the projected next seed.
"""
import os
import sys

import torch

DEV = torch.device("cuda")
DRAFT = os.environ.get("DRAFT", "google/gemma-4-26B-A4B-it-assistant")
TARGET = os.environ.get("TARGET", "cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4")
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILED.append(name)


def init_dist():
    import torch.distributed as dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29563")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from minisgl.distributed import set_tp_info
    from minisgl.layers import set_rope_device
    set_tp_info(rank=0, size=1)
    set_rope_device(DEV)


class _Pool:
    def __init__(self, k, v):
        self._k, self._v = k, v
        self.dtype = k.dtype

    def k_cache(self, i):
        return self._k

    def v_cache(self, i):
        return self._v


def rope_for(rc, head_dim):
    from minisgl.layers import get_rope
    with torch.device(DEV):   # the cos/sin cache must live where the rope kernel reads it
        return get_rope(head_dim=head_dim, rotary_dim=rc.rotary_dim, max_position=rc.max_position,
                        base=rc.base, rope_scaling=tuple(rc.scaling.items()) if rc.scaling else None,
                        interleave=rc.interleave)


def main():
    init_dist()
    import attn_decode
    import safetensors.torch as st
    from transformers import AutoModelForCausalLM

    from minisgl.models.config import ModelConfig
    from minisgl.models.gemma4_assistant import Gemma4AssistantDraft, _AttnTarget
    from minisgl.utils import cached_load_hf_config, download_hf_weight

    tcfg = ModelConfig.from_hf(cached_load_hf_config(TARGET))
    hf = cached_load_hf_config(DRAFT)
    folder = download_hf_weight(DRAFT)
    ref = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.bfloat16).to(DEV).eval()
    ref32 = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32).to(DEV).eval()
    print("reference:", type(ref).__name__, flush=True)

    types = list(hf.text_config.layer_types)
    with torch.device(DEV):
        ours = Gemma4AssistantDraft(hf, types)
    sd = {}
    for f in sorted(os.listdir(folder)):
        if f.endswith(".safetensors"):
            sd.update(st.load_file(os.path.join(folder, f), device="cpu"))
    ours.load(sd, DEV, torch.bfloat16)

    sw_hd, sw_kv = tcfg.swa_head_dim, tcfg.swa_num_kv_heads
    fu_hd, fu_kv = tcfg.head_dim, tcfg.num_kv_heads
    W = int(tcfg.sliding_window)
    sw_rope = rope_for(tcfg.sliding_rotary_config, sw_hd)
    fu_rope = rope_for(tcfg.rotary_config, fu_hd)
    B = int(hf.backbone_hidden_size)
    g = torch.Generator(device="cpu").manual_seed(5)
    decode = (attn_decode.flash_decode_paged, attn_decode.flash_decode_paged_fp8)

    for L in (700, 1500):
        print(f"== context L={L} (window {W})", flush=True)
        # Target-like K/V: K is k_norm'd (small gain), V is unit-RMS.
        ks = (torch.randn(L, sw_kv, sw_hd, generator=g) * 0.126).bfloat16().to(DEV)
        vs = torch.randn(L, sw_kv, sw_hd, generator=g).bfloat16().to(DEV)
        kf = (torch.randn(L, fu_kv, fu_hd, generator=g) * 0.062).bfloat16().to(DEV)
        vf = torch.randn(L, fu_kv, fu_hd, generator=g).bfloat16().to(DEV)
        # The reference's flipped bidirectional mask admits W+1 sliding keys; the served convention
        # (vLLM's, and the target's own sliding layers) is exactly the last W. Hand the reference
        # only those W so both sides attend the same set.
        lo = max(0, L - W)
        shared = {"sliding_attention": (ks[lo:].transpose(0, 1)[None], vs[lo:].transpose(0, 1)[None]),
                  "full_attention": (kf.transpose(0, 1)[None], vf.transpose(0, 1)[None])}

        ps = 16
        npages = (L + ps - 1) // ps
        kfp = torch.zeros(npages * ps, fu_kv, fu_hd, dtype=torch.bfloat16, device=DEV)
        vfp = torch.zeros_like(kfp)
        kfp[:L], vfp[:L] = kf, vf
        # pages in reverse order, so the block table actually indirects
        perm = torch.arange(npages - 1, -1, -1, device=DEV)
        kpool = kfp.view(npages, ps, fu_kv, fu_hd)[perm]
        vpool = vfp.view(npages, ps, fu_kv, fu_hd)[perm]
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(npages, device=DEV)
        full_bt = inv.to(torch.int32)[None]
        full_len = torch.tensor([L], dtype=torch.int32, device=DEV)
        cnt = min(L, W)
        swa_bt = torch.arange(L - cnt, L, dtype=torch.int32, device=DEV)[None]
        swa_len = torch.tensor([cnt], dtype=torch.int32, device=DEV)
        sw_t = _AttnTarget(_Pool(ks[:, None], vs[:, None]), 0, sw_rope, sw_hd, False)
        fu_t = _AttnTarget(_Pool(kpool, vpool), 0, fu_rope, fu_hd, False)
        attn = ((sw_t, swa_bt, swa_len), (fu_t, full_bt, full_len))

        pos = torch.tensor([L - 1], dtype=torch.int64, device=DEV)
        shared32 = {k: (a.float(), b.float()) for k, (a, b) in shared.items()}
        table = (torch.randn(hf.text_config.vocab_size, B, generator=g) * 0.02).bfloat16().to(DEV)
        emb = torch.randn(1, B, generator=g).bfloat16().to(DEV)
        seed = torch.randn(1, B, generator=g).bfloat16().to(DEV)

        def rel(a, b):
            return ((a.float() - b.float()).norm() / b.float().norm()).item()

        # Each step: all three sides get the SAME input (the fp32 reference's token and seed), so
        # every step is compared on its own rather than after a divergence.
        errs = {"ours_l": [], "ref_l": [], "ours_s": [], "ref_s": []}
        with torch.inference_mode():
            for step in range(8):
                x_in = torch.cat([emb, seed], -1)
                r32 = ref32(inputs_embeds=x_in.float()[None], position_ids=pos[None],
                            shared_kv_states=shared32, attention_mask=None)
                r16 = ref(inputs_embeds=x_in[None], position_ids=pos[None],
                          shared_kv_states=shared, attention_mask=None)
                x = ours.pre_projection.forward(x_in)
                for t, layer in zip(types, ours.layers):
                    tg, bt, ln = attn[0] if t == "sliding_attention" else attn[1]
                    x = layer.forward(x, pos, tg, bt, ln, decode, 1.0)
                d = ours.norm.forward(x)
                o_logits = ours.lm_head.logits_all_rows(d)[0]
                o_seed = ours.post_projection.forward(d)[0]
                gt_l, gt_s = r32.logits[0, -1], r32.last_hidden_state[0, -1]
                errs["ours_l"].append(rel(o_logits, gt_l))
                errs["ref_l"].append(rel(r16.logits[0, -1], gt_l))
                errs["ours_s"].append(rel(o_seed, gt_s))
                errs["ref_s"].append(rel(r16.last_hidden_state[0, -1], gt_s))
                tok = int(gt_l.argmax())
                emb, seed = table[tok][None], gt_s[None].to(torch.bfloat16)
            mean = {k: sum(v) / len(v) for k, v in errs.items()}
            check(f"L={L} logits vs fp32 (mean of 8 steps)", mean["ours_l"] <= 1.25 * mean["ref_l"],
                  f"ours {mean['ours_l']:.2e}, reference bf16 {mean['ref_l']:.2e}")
            check(f"L={L} seed vs fp32 (mean of 8 steps)", mean["ours_s"] <= 1.25 * mean["ref_s"],
                  f"ours {mean['ours_s']:.2e}, reference bf16 {mean['ref_s']:.2e}")
            # The same chain through the drafter's own step(): tokens it drafts from the fp32
            # reference's first input must match the fp32 reference's greedy chain.
            emb0 = torch.randn(1, B, generator=g).bfloat16().to(DEV)
            seed0 = torch.randn(1, B, generator=g).bfloat16().to(DEV)
            rchain, ochain = [], []
            e, s_ = emb0, seed0
            for _ in range(4):
                r32 = ref32(inputs_embeds=torch.cat([e, s_], -1).float()[None], position_ids=pos[None],
                            shared_kv_states=shared32, attention_mask=None)
                t_ = int(r32.logits[0, -1].argmax())
                rchain.append(t_)
                e, s_ = table[t_][None], r32.last_hidden_state[0, -1:].to(torch.bfloat16)
            e, s_ = emb0, seed0
            for _ in range(4):
                tok, s_ = ours.step(e, s_, pos, attn, decode, 1.0)
                ochain.append(int(tok[0]))
                e = table[tok]
            agree = sum(a == b for a, b in zip(rchain, ochain))
            print(f"  info  L={L} greedy chain vs fp32: {agree}/4 agree  ours={ochain} fp32={rchain}")

    print()
    print("FAILED: " + "; ".join(FAILED) if FAILED else "ALL CHECKS PASS")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
