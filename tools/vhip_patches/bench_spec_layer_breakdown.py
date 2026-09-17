"""Where does the gdn_hip spec-verify step actually spend its time? Per-GDN-layer breakdown.

In-serve arithmetic (2026-07-28, Qwen3.6-35B-A3B TP=2, K=2, 32768 ctx, MNS=8):
    plain decode step   12.2 ms      (no-spec, 82.2 tok/s, 1 tok/step)
    fla-Triton spec     31.4 ms      (79.2 tok/s, accept 2.49)
    gdn_hip  spec       46.7 ms      (53.8 tok/s, accept 2.51)
=> gdn_hip is +15.3 ms/step over fla, across 30 GDN layers = ~0.51 ms per layer to explain.

fla's spec path is ONE kernel per layer (fused_sigmoid_gating_delta_rule_update): it takes the raw
paged ssm_state, the 2-D slot table and num_accepted_tokens, selects the load slot IN-KERNEL, runs
the same scalar recurrence we do, and publishes each token's state straight into
ssm_state[slots[n, t]] inside the token loop (INPLACE_FINAL_STATE). No scratch, no Python scatter.

Ours splits that into: fp32 pre-casts + conv verify (writes a conv scratch) + splits/contiguous +
gdn_prefill_verify (writes a [max_qlen,N,HV,V,K] ssm scratch) + a Python publish loop of
max_qlen x 2 index_puts. This times each phase so the 0.51 ms is attributed rather than guessed.

Run: one GPU, seconds, no serve.
    PYTHONPATH=<worktree>/tools/vhip_patches/gdn/torch-ext python bench_spec_layer_breakdown.py
"""

import torch

import gdn_hip

torch.manual_seed(0)
DEV = "cuda"
# Qwen3.6-35B-A3B GDN, per TP=2 rank
NK, NV, HK, HV = 8, 16, 128, 128
C = HK * NK * 2 + HV * NV          # 4096
W, NUM_SPEC = 4, 2
MAX_QLEN = NUM_SPEC + 1            # 3 query positions per spec sequence
N = 1                              # bs=1 -> one spec sequence
SLOTS = 512
N_GDN_LAYERS = 30
SCALE = HK ** -0.5
ITERS, WARMUP = 50, 10

# vLLM's paged mamba cache: as_strided, page-padded slot stride (see test_stride_aware_state.py)
conv_elems, ssm_elems = C * (W - 1 + NUM_SPEC), NV * HV * HK
page = conv_elems + ssm_elems + 137
raw = torch.zeros(SLOTS * page, device=DEV, dtype=torch.float32)
conv_state = torch.as_strided(raw, (SLOTS, C, W - 1 + NUM_SPEC), (page, W - 1 + NUM_SPEC, 1), 0)
ssm_state = torch.as_strided(raw, (SLOTS, NV, HV, HK), (page, HV * HK, HK, 1), conv_elems)

T = N * MAX_QLEN
mixed_qkv = torch.randn(T, C, device=DEV, dtype=torch.bfloat16)
a = torch.randn(T, NV, device=DEV, dtype=torch.bfloat16)
b = torch.randn(T, NV, device=DEV, dtype=torch.bfloat16)
conv_weights = (torch.randn(C, W, device=DEV) * 0.1).float()
A_log = torch.randn(NV, device=DEV).float()
dt_bias = torch.randn(NV, device=DEV).float()
sqsl = torch.arange(0, N * MAX_QLEN + 1, MAX_QLEN, dtype=torch.int32, device=DEV)
slots = torch.arange(1, N * MAX_QLEN + 1, device=DEV, dtype=torch.long).view(N, MAX_QLEN)
load_slots = slots[:, 1].contiguous()
has_init = torch.ones(N, dtype=torch.uint8, device=DEV)
qlens = torch.full((N,), MAX_QLEN, device=DEV, dtype=torch.long)
rows = torch.arange(N, device=DEV)
last = qlens - 1


def timed(fn, label, store):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / ITERS
    store[label] = ms
    print(f"  {label:44s} {ms * 1000:8.1f} us/layer   {ms * N_GDN_LAYERS:7.3f} ms x{N_GDN_LAYERS}")
    return ms


# ---- the phases of _forward_core_gdn_hip_spec_impl, in order -------------------------------
def p_casts():
    return mixed_qkv.float().contiguous(), a.float().contiguous(), b.float().contiguous()


def p_conv(x):
    return gdn_hip.causal_conv1d_fwd_verify(
        x, conv_weights, None, sqsl, load_slots, has_init, conv_state, MAX_QLEN, 1)


def p_split(conv_out):
    q, k, v = conv_out.split([HK * NK, HK * NK, HV * NV], dim=-1)
    return (q.reshape(-1, NK, HK).contiguous(), k.reshape(-1, NK, HK).contiguous(),
            v.reshape(-1, NV, HV).contiguous())


def p_verify(q, k, v, af, bf):
    return gdn_hip.gdn_prefill_verify(
        q, k, v, af, bf, A_log, dt_bias, sqsl, load_slots, has_init, ssm_state,
        MAX_QLEN, SCALE, 1)


def p_publish(ssm_scratch, conv_scratch):
    cw = W - 1
    for t in range(MAX_QLEN):
        src = torch.minimum(torch.full_like(last, t), last)
        tgt = slots[:, t]
        ssm_state[tgt] = ssm_scratch[src, rows].to(ssm_state.dtype)
        conv_state[tgt, :, :cw] = conv_scratch[src, rows].to(conv_state.dtype)


res = {}
print(f"\nper-GDN-layer cost of one spec-verify step (N={N} seq, qlen={MAX_QLEN}, {N_GDN_LAYERS} layers)\n")

timed(lambda: p_casts(), "1. fp32 pre-casts (mixed_qkv, a, b)", res)
xf, af, bf = p_casts()
timed(lambda: p_conv(xf), "2. causal_conv1d_fwd_verify (+conv scratch)", res)
conv_out, conv_scratch = p_conv(xf)
timed(lambda: p_split(conv_out), "3. split + 3x reshape.contiguous", res)
q, k, v = p_split(conv_out)
timed(lambda: p_verify(q, k, v, af, bf), "4. gdn_prefill_verify (+ssm scratch)  <- THE KERNEL", res)
core, ssm_scratch = p_verify(q, k, v, af, bf)
timed(lambda: p_publish(ssm_scratch, conv_scratch), "5. per-position publish loop (Python scatter)", res)


def whole():
    x2, a2, b2 = p_casts()
    co, cs = p_conv(x2)
    q2, k2, v2 = p_split(co)
    _, ss = p_verify(q2, k2, v2, a2, b2)
    p_publish(ss, cs)


print()
tot = timed(lambda: whole(), "WHOLE spec-verify glue (end to end)", res)
kern = res["4. gdn_prefill_verify (+ssm scratch)  <- THE KERNEL"]
print(f"\n  recurrence kernel is {kern / tot * 100:.0f}% of the layer; "
      f"the other {100 - kern / tot * 100:.0f}% is glue.")
print(f"  glue overhead x{N_GDN_LAYERS} layers = {(tot - kern) * N_GDN_LAYERS:.2f} ms/step "
      f"(budget to explain: ~15.3 ms)")
