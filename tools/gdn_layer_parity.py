"""Phase 3b-3: single-layer GDN parity — minisgl `QwenGatedDeltaNet` vs the REAL
vLLM `QwenGatedDeltaNetAttention` (the independent capture/replay oracle).

Oracle seam (see PORT.md "Phase 3 — GDN", 3b-3):
  * Stand up the real layer (Qwen3NextConfig + minimal VllmConfig, TP=1, quant=None,
    gqa_interleaved_layout=False / Qwen3.5).
  * Force `enable_packed_recurrent_decode=False` so the real decode uses the SAME
    kernel (`fused_sigmoid_gating_delta_rule_update`) minisgl implements. The live
    env default is PRINTED — if it is True, minisgl's decode is a deliberate
    divergence to revisit in 3c (recorded, not silently passed).
  * Monkeypatch `qwen_gdn_linear_attn.get_forward_context` -> stub whose
    `.attn_metadata` is a dict {prefix: GDNAttentionMetadata(...)}; set
    `real.kv_cache=[conv,ssm]`; call `_forward_core_rocm(qkvz, ba, z, core_attn_out)`
    directly, THEN `real._output_projection(...)` (the core op does NOT project).

Layout: `is_conv_state_dim_first()` is read LIVE and drives both the buffer fed to
the real layer and the frame compared:
  * DS (dim-first, True): real uses kv_cache[0] as (slots, conv_dim, k-1) directly.
  * SD (False, the RDNA4 default): real transposes kv_cache[0] internally, so we
    store real's conv buffer as (slots, k-1, conv_dim) and compare
    real.kv_cache[0].transpose(-1,-2) against minisgl's DS conv_state.
minisgl + GDNStateCache assume DS; if the live value is SD, 3c's GDNStateCache
wiring needs the transpose (recorded in PORT.md).

Checks (advisor-mandated):
  1. (run gdn_layer_parity_cpu.py first — split/reshape pre-check, CPU-only.)
  2. weight copy real->minisgl with per-param shape assert.
  3. prefill: compare OUTPUT + conv_state + ssm_state (validates the 3a write-path).
  4. decode: gated on prefill-state parity; PLUS an independent decode test that
     injects one shared random (conv,ssm) into both layers. The independent test is
     the load-bearing nonzero-readout decode check (core_attn_out ~3e-2, output ~17);
     the continue-from-prefill readout happens to decay to ~0 for these dims, so it
     only confirms real and minisgl read the prefill state identically (real==mini).

Tolerances: abs OR relative (atol 2e-2 / rtol 5e-3). The only non-zero diffs are
bf16-magnitude (rel < 4e-3): real reads a NON-contiguous SD-transposed conv view while
minisgl reads a contiguous DS buffer, so causal_conv1d_update's conv_out differs at the
bf16 ulp and propagates to q/output. minisgl's contiguous DS path is the cleaner one.

Run on GPU via the lease (README "Running"). On this box hipBLASLt intermittently throws
HIPBLAS_STATUS_INTERNAL_ERROR -> ALLOC_FAILED on the in_proj GEMM under load; pass
`-e TORCH_BLAS_PREFER_HIPBLASLT=0` (route GEMMs through rocBLAS) for a reliable run.
"""

from __future__ import annotations

import os
import traceback

import torch

# ---- 35B-ish GDN dims: conv_dim = key_dim*2 + value_dim = 8192; ssm (32,128,128) ----
HK, HV, DK, DV, KCONV, HIDDEN = 16, 32, 128, 128, 4, 2048
KEY_DIM, VALUE_DIM = DK * HK, DV * HV
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
T_PREFILL = 128  # two 64-chunks
SEED = 1234

report: list[tuple[str, bool, float, str]] = []


def banner(s: str) -> None:
    print(f"\n==== {s} ====", flush=True)


def cmp(name: str, a: torch.Tensor, b: torch.Tensor, tol: float = 2e-2,
        rtol: float = 5e-3, gated: bool = True) -> bool:
    """Pass if abs OR relative error is within tolerance. Relative gating matters
    for decode: real reads a non-contiguous SD-transposed conv view while minisgl
    reads a contiguous DS buffer, so conv_out (and hence output) carries a benign
    bf16-magnitude difference that scales with signal magnitude."""
    md = (a.float() - b.float()).abs().max().item()
    rel = md / (b.float().abs().max().item() + 1e-12)
    ok = (md < tol) or (rel < rtol)
    report.append((name, ok if gated else True, md, "" if gated else "info"))
    tag = ("PASS" if ok else "FAIL") if gated else "INFO"
    print(f"  [{tag}] {name:30s} max|Δ|={md:.3e} rel={rel:.3e}  "
          f"(atol={tol:.0e} rtol={rtol:.0e})", flush=True)
    return ok


# --------------------------------------------------------------------------- config
def build_vllm_config():
    from vllm.config import VllmConfig, set_current_vllm_config  # noqa: F401

    last = None
    try:
        from vllm.config import ModelConfig

        mc = ModelConfig(model="Qwen/Qwen3-0.6B", dtype="bfloat16",
                         tokenizer="Qwen/Qwen3-0.6B", trust_remote_code=True)
        return VllmConfig(model_config=mc)
    except Exception as e:  # noqa: BLE001
        last = e
        print(f"  ModelConfig(Qwen3-0.6B) strategy failed: {type(e).__name__}: {e}")
    try:
        return VllmConfig()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"no VllmConfig strategy worked; last={last!r}, then {e!r}")


def build_qwen3next_config():
    from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig

    cfg = Qwen3NextConfig()
    cfg.hidden_size = HIDDEN
    cfg.hidden_act = "silu"
    cfg.rms_norm_eps = 1e-6
    cfg.linear_num_key_heads = HK
    cfg.linear_num_value_heads = HV
    cfg.linear_key_head_dim = DK
    cfg.linear_value_head_dim = DV
    cfg.linear_conv_kernel_dim = KCONV
    return cfg


# ----------------------------------------------------------------------- metadata
def md_prefill(slot: int, device, GDNAttentionMetadata):
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    qsl = torch.tensor([0, T_PREFILL], device=device, dtype=torch.int32)
    qsl_cpu = torch.tensor([0, T_PREFILL], dtype=torch.int32)
    ci = prepare_chunk_indices(qsl_cpu, FLA_CHUNK_SIZE).to(device)
    co = prepare_chunk_offsets(qsl_cpu, FLA_CHUNK_SIZE).to(device)
    try:
        nums_dict, batch_ptr, tco = compute_causal_conv1d_metadata(qsl_cpu, device=device)
    except Exception as e:  # noqa: BLE001
        print(f"  (conv metadata fallback to None: {type(e).__name__}: {e})")
        nums_dict = batch_ptr = tco = None
    return GDNAttentionMetadata(
        num_prefills=1, num_prefill_tokens=T_PREFILL, num_decodes=0, num_decode_tokens=0,
        num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=T_PREFILL,
        has_initial_state=torch.tensor([False], device=device),
        non_spec_query_start_loc=qsl,
        non_spec_state_indices_tensor=torch.tensor([slot], device=device, dtype=torch.int32),
        chunk_indices=ci, chunk_offsets=co,
        nums_dict=nums_dict, batch_ptr=batch_ptr, token_chunk_offset_ptr=tco,
    ), qsl


def md_decode(slots: list[int], device, GDNAttentionMetadata):
    b = len(slots)
    qsl = torch.arange(b + 1, device=device, dtype=torch.int32)
    return GDNAttentionMetadata(
        num_prefills=0, num_prefill_tokens=0, num_decodes=b, num_decode_tokens=b,
        num_spec_decodes=0, num_spec_decode_tokens=0, num_actual_tokens=b,
        has_initial_state=None,
        non_spec_query_start_loc=qsl,
        non_spec_state_indices_tensor=torch.tensor(slots, device=device, dtype=torch.int32),
    ), qsl


# ------------------------------------------------------------------------- drivers
def drive_real(real, Q, md, hs, conv_dtype, ssm_dtype, device):
    """Run the real layer's core (conv + recurrent) + output projection. Returns
    (output, conv_state_DS_view, ssm_state). real.kv_cache must be pre-set."""
    n = hs.shape[0]
    qkvz, _ = real.in_proj_qkvz(hs)
    ba, _ = real.in_proj_ba(hs)
    qkvz = qkvz.contiguous().view(n, -1)
    ba = ba.contiguous().view(n, -1)
    core = torch.zeros(n, HV, DV, dtype=hs.dtype, device=device)
    z_out = torch.empty(n, HV, DV, dtype=hs.dtype, device=device)
    Q.get_forward_context = lambda: type("C", (), {"attn_metadata": {real.prefix: md}})()
    real._forward_core_rocm(qkvz=qkvz, ba=ba, z_out=z_out, core_attn_out=core)
    out = torch.empty(n, HIDDEN, dtype=hs.dtype, device=device)
    real._output_projection(core, z_out, out, n)
    return out, core, z_out


def main() -> None:
    print(f"torch {torch.__version__}  hip={torch.version.hip}", flush=True)
    import vllm

    print(f"vllm {vllm.__version__}", flush=True)
    print(f"  cuda.is_available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()}  "
          f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')!r} "
          f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')!r}", flush=True)
    if torch.cuda.device_count() == 0:
        raise SystemExit("no GPU visible to torch — check HIP/ROCR_VISIBLE_DEVICES passthrough")
    device = torch.device("cuda")

    banner("live env / layout facts")
    from vllm import envs
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as Q
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    env_packed = getattr(envs, "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "<absent>")
    env_layout = getattr(envs, "VLLM_SSM_CONV_STATE_LAYOUT", "<absent>")
    dim_first = is_conv_state_dim_first()
    print(f"  VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE = {env_packed!r}")
    print(f"  VLLM_SSM_CONV_STATE_LAYOUT              = {env_layout!r}")
    print(f"  is_conv_state_dim_first()              = {dim_first}  "
          f"({'DS / dim-first' if dim_first else 'SD / state-first (real transposes)'})")
    print(f"  GDN_AITER_TRITON_AVAILABLE             = {Q.GDN_AITER_TRITON_AVAILABLE}")

    banner("distributed init (TP=1) + configs")
    from vllm.distributed import init_distributed_environment, initialize_model_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12377")
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method="tcp://127.0.0.1:12377", backend="nccl",
    )
    from vllm.config import set_current_vllm_config

    vllm_config = build_vllm_config()
    cfg = build_qwen3next_config()
    # Keep the vLLM config active for the WHOLE run: initialize_model_parallel,
    # layer construction, AND the torch.compiled core split all call
    # get_current_vllm_config(). Enter the context manually so the rest of main()
    # need not be re-indented; the process exit tears it down.
    _vc = set_current_vllm_config(vllm_config)
    _vc.__enter__()
    initialize_model_parallel(tensor_model_parallel_size=1)
    print(f"  model dtype={vllm_config.model_config.dtype}  "
          f"conv_dim={CONV_DIM} value_dim={VALUE_DIM} key_dim={KEY_DIM}")

    banner("stand up REAL QwenGatedDeltaNetAttention")
    real = Q.QwenGatedDeltaNetAttention(
        config=cfg, vllm_config=vllm_config,
        prefix="model.layers.0.linear_attn", gqa_interleaved_layout=False,
    )
    real = real.to(device)
    real.enable_packed_recurrent_decode = False  # match minisgl's decode kernel
    conv_dtype, ssm_dtype = real.get_state_dtype()
    print(f"  forward_method={real._forward_method.__name__}  "
          f"gdn_prefill_backend={real.gdn_prefill_backend}")
    print(f"  enable_packed_recurrent_decode -> forced {real.enable_packed_recurrent_decode}")
    print(f"  state dtypes: conv={conv_dtype}  ssm={ssm_dtype}")
    if env_packed not in (False, "<absent>"):
        print("  *** NOTE: packed-recurrent-decode env is truthy in this image — "
              "minisgl decode kernel choice is a 3c divergence to revisit ***")

    # Standalone (no model-load context) the parallel-linear weights default to
    # float32; coerce to minisgl's dtypes so the real layer runs bf16 activations
    # with bf16 weights + fp32 gating params (A_log/dt_bias), exactly like minisgl.
    # Then randomize the (empty / uninitialized) params to finite values.
    with torch.no_grad():
        for w in (real.in_proj_qkvz.weight, real.in_proj_ba.weight,
                  real.conv1d.weight, real.out_proj.weight, real.norm.weight):
            w.data = w.data.to(torch.bfloat16)
            w.normal_(0.0, 0.05)
        real.norm.weight.normal_(1.0, 0.02)
        real.A_log.data = real.A_log.data.to(torch.float32)
        real.dt_bias.data = real.dt_bias.data.to(torch.float32)
        # Gentle decay (A = -exp(A_log) ~ -0.14) so a decode step's recurrent readout
        # off a 128-token prefill state stays non-trivial — otherwise core_attn_out
        # decays to ~0 and continue-decode parity becomes a vacuous 0≈0 check.
        real.A_log.normal_(-2.0, 0.3)
        real.dt_bias.normal_(0.0, 0.1)
    print(f"  real param dtypes: qkvz={real.in_proj_qkvz.weight.dtype} "
          f"conv={real.conv1d.weight.dtype} norm={real.norm.weight.dtype} "
          f"A_log={real.A_log.dtype} dt_bias={real.dt_bias.dtype}")

    banner("build minisgl layer + copy weights (shape-asserted)")
    from minisgl.gdn.layer import QwenGatedDeltaNet

    mini = QwenGatedDeltaNet(
        hidden_size=HIDDEN, num_k_heads=HK, num_v_heads=HV,
        head_k_dim=DK, head_v_dim=DV, conv_kernel_size=KCONV,
        dtype=torch.bfloat16, device=device,
    )
    pairs = [
        ("in_proj_qkvz", mini.in_proj_qkvz.weight, real.in_proj_qkvz.weight),
        ("in_proj_ba", mini.in_proj_ba.weight, real.in_proj_ba.weight),
        ("conv1d", mini.conv1d_weight, real.conv1d.weight),
        ("out_proj", mini.out_proj.weight, real.out_proj.weight),
        ("A_log", mini.A_log, real.A_log),
        ("dt_bias", mini.dt_bias, real.dt_bias),
        ("norm", mini.norm.weight, real.norm.weight),
    ]
    with torch.no_grad():
        for name, dst, src in pairs:
            assert tuple(dst.shape) == tuple(src.shape), (
                f"{name}: minisgl {tuple(dst.shape)} != real {tuple(src.shape)}")
            dst.copy_(src.to(dst.dtype))
            print(f"  copied {name:14s} {tuple(dst.shape)}")

    n_slots = 4

    # NOTE on determinism: the FLA/conv kernels autotune on their FIRST call by
    # running candidate configs on the LIVE input buffer. With a COLD on-disk triton
    # cache that search corrupted one side once (real reads a non-contiguous SD conv
    # view, minisgl a contiguous DS buffer -> divergent cold autotune; a one-shot
    # Δ≈27 seen only on the very first cold run). With the warm mounted
    # .triton-cache-combined the cached config is reused without re-searching, so the
    # first measured call is clean and the result is stable across runs.

    # ------------------------------------------------------------------ PREFILL
    banner("PREFILL parity (output + conv_state + ssm_state)")
    slot = 0
    torch.manual_seed(SEED)
    hs = torch.randn(T_PREFILL, HIDDEN, dtype=torch.bfloat16, device=device)
    md_p, qsl_p = md_prefill(slot, device, GDNAttentionMetadata)
    state_idx = torch.tensor([slot], device=device, dtype=torch.int32)
    has_init = torch.tensor([False], device=device)

    # real conv buffer: DS (slots, conv_dim, k-1) if dim_first else SD (slots, k-1, conv_dim)
    if dim_first:
        real_conv = torch.zeros(n_slots, CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device)
    else:
        real_conv = torch.zeros(n_slots, KCONV - 1, CONV_DIM, dtype=conv_dtype, device=device)
    real_ssm = torch.zeros(n_slots, HV, DV, DK, dtype=ssm_dtype, device=device)
    real.kv_cache = [real_conv, real_ssm]

    mini_conv = torch.zeros(n_slots, CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device)
    mini_ssm = torch.zeros(n_slots, HV, DV, DK, dtype=ssm_dtype, device=device)

    real_out, _, _ = drive_real(real, Q, md_p, hs, conv_dtype, ssm_dtype, device)
    mini_out = mini.forward_prefill(hs, mini_conv, mini_ssm, qsl_p, state_idx, has_init)

    real_conv_ds = real.kv_cache[0] if dim_first else real.kv_cache[0].transpose(-1, -2)
    cmp("prefill.output", real_out, mini_out, tol=2e-2)
    cmp("prefill.conv_state", real_conv_ds[slot], mini_conv[slot], tol=2e-2)
    cmp("prefill.ssm_state", real.kv_cache[1][slot], mini_ssm[slot], tol=2e-2)

    prefill_state_ok = all(
        ok for name, ok, _, _ in report
        if name in ("prefill.conv_state", "prefill.ssm_state"))

    # ---- per-stage capture: localize any decode divergence (conv_out / q / core) ----
    import minisgl.gdn.layer as ML

    cap: dict = {}

    def _wrap_conv(ns, tag):
        orig = ns.causal_conv1d_update

        def w(*a, **k):
            r = orig(*a, **k)
            cs = a[1] if len(a) > 1 else k.get("conv_state")
            cap[f"{tag}.conv_out"] = r.detach().clone()
            cap[f"{tag}.conv_contig"] = bool(cs.is_contiguous())
            return r

        ns.causal_conv1d_update = w

    def _wrap_fused(ns, tag):
        orig = ns.fused_sigmoid_gating_delta_rule_update

        def w(*a, **k):
            core, last = orig(*a, **k)
            cap[f"{tag}.core"] = core.detach().clone()
            if k.get("q") is not None:
                cap[f"{tag}.q"] = k["q"].detach().clone()
            return core, last

        ns.fused_sigmoid_gating_delta_rule_update = w

    _wrap_conv(Q, "real"); _wrap_conv(ML, "mini")
    _wrap_fused(Q, "real"); _wrap_fused(ML, "mini")

    def relrow(name, rt, mt):
        a, b = rt.float().flatten(), mt.float().flatten()
        if a.shape != b.shape:
            print(f"  {name:20s} SHAPE real{tuple(rt.shape)} mini{tuple(mt.shape)}")
            return
        num = (a - b).abs().max().item()
        den = b.abs().max().item() + 1e-12
        print(f"  {name:20s} max|Δ|={num:.3e} rel={num / den:.3e} "
              f"|real|={a.abs().max():.3e} |mini|={b.abs().max():.3e}")

    def localize(hs_x, mini_conv_x, mini_ssm_x, sidx_x, tag):
        print(f"  -- stage localization ({tag}) --")
        print(f"  conv_state_in contiguous: real={cap.get('real.conv_contig')} "
              f"mini={cap.get('mini.conv_contig')}")
        relrow("conv_out", cap["real.conv_out"], cap["mini.conv_out"])
        if "real.q" in cap and "mini.q" in cap:
            relrow("q (post-l2norm?)", cap["real.q"], cap["mini.q"])
        relrow("core_attn_out", cap["real.core"], cap["mini.core"])
        mqkvz = mini.in_proj_qkvz(hs_x)
        _, mz, _, _ = mini._split_qkvz_ba(mqkvz, mini.in_proj_ba(hs_x), hs_x.shape[0])
        return mz

    # --------------------------------------------------- DECODE (continue prefill)
    banner("DECODE parity (continues prefill state)" if prefill_state_ok
           else "DECODE parity SKIPPED — prefill state diverged")
    if prefill_state_ok:
        torch.manual_seed(SEED + 1)
        hs_d = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)
        md_d, qsl_d = md_decode([slot], device, GDNAttentionMetadata)
        real_out_d, _, real_z_d = drive_real(real, Q, md_d, hs_d, conv_dtype, ssm_dtype, device)
        mini_out_d = mini.forward_decode(hs_d, mini_conv, mini_ssm, qsl_d, state_idx)
        mz_d = localize(hs_d, mini_conv, mini_ssm, slot, "continue")
        relrow("z", real_z_d, mz_d)
        relrow("output", real_out_d, mini_out_d)
        real_conv_ds_d = real.kv_cache[0] if dim_first else real.kv_cache[0].transpose(-1, -2)
        cmp("decode.output", real_out_d, mini_out_d, tol=2e-2)
        cmp("decode.conv_state", real_conv_ds_d[slot], mini_conv[slot], tol=2e-2)
        cmp("decode.ssm_state", real.kv_cache[1][slot], mini_ssm[slot], tol=2e-2)

    # ---------------------------------- INDEPENDENT DECODE (shared random state) --
    banner("INDEPENDENT DECODE parity (shared random conv+ssm injected into both)")
    sidx = 1
    torch.manual_seed(SEED + 7)
    rconv = torch.randn(CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device) * 0.1
    rssm = torch.randn(HV, DV, DK, dtype=ssm_dtype, device=device) * 0.1
    hs_i = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)

    real_conv2 = torch.zeros_like(real_conv)
    real_ssm2 = torch.zeros(n_slots, HV, DV, DK, dtype=ssm_dtype, device=device)
    # write the shared state into real (respecting its conv layout) and mini (DS)
    if dim_first:
        real_conv2[sidx] = rconv
    else:
        real_conv2[sidx] = rconv.transpose(-1, -2)
    real_ssm2[sidx] = rssm
    real.kv_cache = [real_conv2, real_ssm2]
    mini_conv2 = torch.zeros(n_slots, CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device)
    mini_ssm2 = torch.zeros(n_slots, HV, DV, DK, dtype=ssm_dtype, device=device)
    mini_conv2[sidx] = rconv
    mini_ssm2[sidx] = rssm

    md_i, qsl_i = md_decode([sidx], device, GDNAttentionMetadata)
    state_idx_i = torch.tensor([sidx], device=device, dtype=torch.int32)
    real_out_i, _, real_z_i = drive_real(real, Q, md_i, hs_i, conv_dtype, ssm_dtype, device)
    mini_out_i = mini.forward_decode(hs_i, mini_conv2, mini_ssm2, qsl_i, state_idx_i)
    mz_i = localize(hs_i, mini_conv2, mini_ssm2, sidx, "independent")
    relrow("z", real_z_i, mz_i)
    relrow("output", real_out_i, mini_out_i)
    real_conv2_ds = real.kv_cache[0] if dim_first else real.kv_cache[0].transpose(-1, -2)
    cmp("indep_decode.output", real_out_i, mini_out_i, tol=2e-2)
    cmp("indep_decode.conv_state", real_conv2_ds[sidx], mini_conv2[sidx], tol=2e-2)
    cmp("indep_decode.ssm_state", real.kv_cache[1][sidx], mini_ssm2[sidx], tol=2e-2)

    # ----------------------------------------------------------------- verdict
    banner("VERDICT")
    all_ok = all(ok for _, ok, _, _ in report)
    print(f"  conv layout: {'DS' if dim_first else 'SD'}  "
          f"(minisgl/GDNStateCache assume DS; "
          f"{'matches' if dim_first else '3c GDNStateCache wiring needs the transpose'})")
    print("RESULT:", "PASS — minisgl layer matches the real vLLM oracle"
          if all_ok else "FAIL — investigate divergence above")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
