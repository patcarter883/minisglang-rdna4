"""Standalone: do the GDN prefill/verify kernels honour a PAGED (strided) ssm/conv state?

Reproduces vLLM's real mamba-cache allocation (gpu_model_runner, MambaSpec branch): one raw buffer
per layer, carved with ``torch.as_strided`` into

    conv_state : (num_blocks, C, W-1+num_spec)  stride (num_element_per_page, W-1+num_spec, 1)
    ssm_state  : (num_blocks, HV, V, K)         stride (num_element_per_page, V*K, K, 1)
                 + a storage offset (the ssm half sits AFTER the conv half inside the page)

i.e. the INNER dims stay contiguous but the SLOT stride is the padded page size, not HV*V*K — which
is precisely what the hardcoded `((slot*HV+hv)*V+r)*K` addressing got wrong (it walked into the
neighbouring slot's conv half). Each kernel is run twice on identical inputs: once against a plain
contiguous state, once against the paged view seeded with the same values. Outputs AND the persisted
state must match bit-for-bit — only the addressing differs, the arithmetic is untouched.

Run inside the vllm24-hip image with PYTHONPATH=/patches/gdn/torch-ext (seconds, one GPU, no serve).
"""

import sys

import torch

import gdn_hip

torch.manual_seed(0)
DEV = "cuda"
# Qwen3.6-35B-A3B GDN at TP=2, per rank
NK, NV, HK, HV = 8, 16, 128, 128
C = HK * NK * 2 + HV * NV          # packed conv channels (q|k|v)
W, NUM_SPEC = 4, 2
CONV_WIDTH = W - 1 + NUM_SPEC      # vLLM widens the conv row under spec
SLOTS = 8
SCALE = HK ** -0.5

failures = []


def paged_pair(dtype):
    """(conv_state, ssm_state) as vLLM carves them: shared page-padded raw buffer, as_strided views.

    Returns the views plus the contiguous references to seed them from.
    """
    esz = torch.empty((), dtype=dtype).element_size()
    conv_elems = C * CONV_WIDTH
    ssm_elems = NV * HV * HK
    # vLLM pads the page; mimic with a non-trivial pad so the slot stride can't accidentally equal
    # either half's element count.
    page_elems = conv_elems + ssm_elems + 137
    # NaN-fill the whole buffer: only the view's own elements get seeded, so any mis-strided read
    # lands in a gap (or a neighbouring slot's other half) and shows up as NaN rather than a
    # plausible-looking zero.
    raw = torch.full((SLOTS * page_elems,), float("nan"), device=DEV, dtype=dtype)
    conv_v = torch.as_strided(raw, (SLOTS, C, CONV_WIDTH), (page_elems, CONV_WIDTH, 1), 0)
    ssm_v = torch.as_strided(raw, (SLOTS, NV, HV, HK), (page_elems, HV * HK, HK, 1), conv_elems)
    return conv_v, ssm_v


def seed_states(dtype, init_scale=0.1):
    """Contiguous reference states + the paged views holding identical values."""
    conv_ref = (torch.randn(SLOTS, C, CONV_WIDTH, device=DEV) * init_scale).to(torch.float32)
    ssm_ref = (torch.randn(SLOTS, NV, HV, HK, device=DEV) * init_scale).to(dtype)
    conv_v, ssm_v = paged_pair(torch.float32)[0], paged_pair(dtype)[1]
    conv_v.copy_(conv_ref)
    ssm_v.copy_(ssm_ref)
    assert not ssm_v.is_contiguous() and not conv_v.is_contiguous()
    return conv_ref.clone(), ssm_ref.clone(), conv_v, ssm_v


def check(name, ref, got):
    ref, got = ref.float(), got.float()
    if not torch.isfinite(ref).all():
        failures.append(f"{name}: contiguous reference itself is non-finite")
        print(f"    {name:16s} REFERENCE NON-FINITE")
        return
    same = torch.equal(ref, got)
    md = (ref - got).abs().max().item()
    print(f"    {name:16s} bitexact={same} maxdiff={md:.3e} |ref|={ref.abs().max().item():.4f}")
    if not same:
        failures.append(f"{name}: paged != contiguous (maxdiff {md:.3e})")


def run_case(label, dtype, plen, fn, alog_shift=0.0):
    """fn(conv_state, ssm_state, cu, idx, has_init, x, a, b) -> dict of output tensors.

    ``alog_shift`` damps the decay (A_log -> A_log + shift). gdn_prefill_chunked forms explicit
    gamma_C/gamma_j ratios, so with raw-randn A_log (per-token decay as steep as exp(-36)) the
    cumulative gamma underflows to 0 over a 32-token chunk and the ratio goes 0/0 -> NaN. That is a
    PRE-EXISTING property of that kernel's math, unrelated to striding (the WMMA kernel exists
    precisely because of it — it keeps the ratios in log space), and it is not on the serve path.
    Shift it into the realistic mild-decay regime so the striding question can actually be asked.
    """
    print(f"\n[{label}]  ssm dtype={dtype}")
    x = torch.randn(plen, C, device=DEV).float()
    a = torch.randn(plen, NV, device=DEV).float()
    b = torch.randn(plen, NV, device=DEV).float()
    global A_LOG
    A_LOG = A_LOG_BASE + alog_shift
    cu = torch.tensor([0, plen], dtype=torch.int32, device=DEV)
    idx = torch.tensor([3], dtype=torch.long, device=DEV)          # a mid-buffer slot, not 0/1
    has_init = torch.ones(1, dtype=torch.uint8, device=DEV)        # exercise the state LOAD path

    conv_ref, ssm_ref, conv_v, ssm_v = seed_states(dtype)
    out_c = fn(conv_ref, ssm_ref, cu, idx, has_init, x, a, b)      # contiguous
    out_p = fn(conv_v, ssm_v, cu, idx, has_init, x, a, b)          # paged views
    for k in out_c:
        check(k, out_c[k], out_p[k])
    check("ssm_state", ssm_ref, ssm_v)
    check("conv_state", conv_ref, conv_v)


def _conv_fwd(conv_state, cu, idx, has_init, x):
    return gdn_hip.causal_conv1d_fwd(x, WT, None, cu, idx, has_init, conv_state, 1)


def _split(conv_out):
    q, k, v = conv_out.split([HK * NK, HK * NK, HV * NV], dim=-1)
    return (q.reshape(-1, NK, HK).contiguous(),
            k.reshape(-1, NK, HK).contiguous(),
            v.reshape(-1, NV, HV).contiguous())


WT = torch.randn(C, W, device=DEV).float() * 0.1
A_LOG_BASE = torch.randn(NV, device=DEV).float()
A_LOG = A_LOG_BASE
DT_BIAS = torch.randn(NV, device=DEV).float()


def make_prefill(op):
    def fn(conv_state, ssm_state, cu, idx, has_init, x, a, b):
        q, k, v = _split(_conv_fwd(conv_state, cu, idx, has_init, x))
        core = op(q, k, v, a, b, A_LOG, DT_BIAS, cu, idx, has_init, ssm_state, SCALE, 1)
        return {"core": core}
    return fn


def make_verify(max_qlen):
    def fn(conv_state, ssm_state, cu, idx, has_init, x, a, b):
        conv_out, conv_scr = gdn_hip.causal_conv1d_fwd_verify(
            x, WT, None, cu, idx, has_init, conv_state, max_qlen, 1)
        q, k, v = _split(conv_out)
        core, ssm_scr = gdn_hip.gdn_prefill_verify(
            q, k, v, a, b, A_LOG, DT_BIAS, cu, idx, has_init, ssm_state, max_qlen, SCALE, 1)
        return {"core": core, "conv_scratch": conv_scr, "ssm_scratch": ssm_scr}
    return fn


for dt in (torch.float32, torch.bfloat16):
    # verify: the spec-decode shape (1 verified + num_spec drafts)
    run_case("gdn_prefill_verify  (spec, qlen=3)", dt, 3, make_verify(3))
    # recurrent prefill oracle
    run_case("gdn_prefill (recurrent)", dt, 40, make_prefill(gdn_hip.gdn_prefill))
    # WMMA chunked prefill — the default serve prefill; spans >1 chunk (C=16) and a ragged tail
    run_case("gdn_prefill_wmma", dt, 40, make_prefill(gdn_hip.gdn_prefill_wmma))
    # scalar chunked prefill (GDN_CHUNK=32): spans 2 chunks incl. a partial one. Mild decay only —
    # see run_case's alog_shift note (its gamma ratios underflow, pre-existing, not a striding issue).
    run_case("gdn_prefill_chunked", dt, 40, make_prefill(gdn_hip.gdn_prefill_chunked), alog_shift=-3.0)

print("\n" + ("FAIL: " + "; ".join(failures) if failures else "ALL BIT-EXACT: paged == contiguous"))
sys.exit(1 if failures else 0)
