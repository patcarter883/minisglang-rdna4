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

★ CRITICAL — cache slot MUST be >= 1. Cache index 0 == NULL_BLOCK_ID, which
causal_conv1d_fn treats as a null/padding block and SKIPS (returns its output buffer
UNWRITTEN -> reads back 0 / NaN / aliased garbage). The premature 2026-06-19 "PREFILL
bit-exact" PASS was VACUOUS: it ran at slot 0, so BOTH real and minisgl conv were skipped
and the harness compared garbage-to-garbage. 3c's GDNStateCache must never allocate slot 0
to a real sequence. (Proven via a pure-torch CPU reference conv: slot 0 -> WRONG, slot 1 ->
CORRECT at rel ~7e-3 bf16. See `GDN_SINGLE=probe`.)

Run modes via the GDN_SINGLE env var:
  * (unset) — the full default harness: warmup, prefill cmp, decode + independent decode.
  * both  — the canonical IN-PROCESS prefill parity (shared weights+in_proj+hs at slot 1);
            q/k/v/g/beta/core/ssm/conv all bit-exact. A cross-PROCESS compare is INVALID
            (each process random-inits its own real layer incl. A_log ~ N(-2,0.3)).
  * probe — kernel matrix (namespace x layout x slot x metadata x packed) vs a CPU
            ground-truth conv; this is how the slot-0 root cause was localized.
  * real/mini — single-layer capture/print (debug only).

Checks (default path):
  1. (run gdn_layer_parity_cpu.py first — split/reshape pre-check, CPU-only: 9/9 bit-exact.)
  2. weight copy real->minisgl with per-param shape assert (ALL 7 tensors).
  3. prefill (slot 1): compare OUTPUT + conv_state + ssm_state. conv_state/ssm bit-exact and
     SUBSTANTIAL (non-vacuous); validates the GDN ORCHESTRATION (split order, DS-vs-SD
     conv_state layout, state write/index, call sequence) — the kernels themselves are
     validated separately in 3b-1.
  4. decode (slot 1): continue-from-prefill PLUS an independent shared-random-state decode
     (load-bearing nonzero readout). slot 1 also avoids the NULL_BLOCK_ID skip in
     causal_conv1d_update.

Tolerances: abs OR relative (atol 2e-2 / rtol 5e-3). At slot>=1 the only non-zero diff is
the z-dominated `output` (rel ~2e-3) from real's vs minisgl's out_proj GEMM backend; conv,
conv_state, ssm_state and core are bit-exact (Δ=0).

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
    """Pass if abs OR relative error is within tolerance. At slot>=1 conv_state, ssm_state
    and core are bit-exact (Δ=0) between real and minisgl (shared in_proj + weights + the
    byte-identical kernels); the only non-zero diff is the z-dominated `output`, which
    carries a benign bf16 rounding from real's vs minisgl's out_proj GEMM backend (rel~2e-3).
    SD-transposed (real) vs DS (minisgl) conv_state layout gives identical results here."""
    md = (a.float() - b.float()).abs().max().item()
    ra, rb = a.float().abs().max().item(), b.float().abs().max().item()
    rel = md / (rb + 1e-12)
    ok = (md < tol) or (rel < rtol)
    report.append((name, ok if gated else True, md, "" if gated else "info"))
    tag = ("PASS" if ok else "FAIL") if gated else "INFO"
    # |real|/|mini| guard against a VACUOUS 0==0 pass: a bit-exact match is only
    # meaningful when the tensors carry substantial signal.
    print(f"  [{tag}] {name:30s} max|Δ|={md:.3e} rel={rel:.3e} "
          f"|real|={ra:.3e} |mini|={rb:.3e}  (atol={tol:.0e} rtol={rtol:.0e})", flush=True)
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
    # Localize a corrupted real-second: is real's GEMM input already degraded, or is
    # the conv kernel the corruptor? Print the projected qkvz magnitude pre-conv.
    print(f"  [drive_real] qkvz nan={torch.isnan(qkvz).any().item()} "
          f"|max|={qkvz.float().abs().max().item():.3e}", flush=True)
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

    # Isolate the GDN COMPUTE: route minisgl's input projection through the SAME module
    # (and thus the same GEMM kernel) as the real layer. minisgl's in_proj is a plain
    # nn.Linear (torch F.linear); the real layer's is vLLM MergedColumnParallelLinear
    # (rocm_unquantized_gemm). With identical weights they agree bit-exact at n=1 (decode)
    # but the two GEMM backends round differently at n=128 (prefill), and the 128-step
    # chunk recurrence amplifies that tiny input delta into a large output divergence —
    # an in_proj GEMM-backend artifact, NOT a GDN-compute bug (the split is already
    # validated bit-exact by the CPU pre-check). Sharing the projection removes the
    # confound so prefill parity tests the conv+chunk+state-write path on identical inputs.
    class _SharedProj(torch.nn.Module):
        def __init__(self, real_linear: torch.nn.Module) -> None:
            super().__init__()
            self._r = real_linear

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self._r(x)[0]

    mini.in_proj_qkvz = _SharedProj(real.in_proj_qkvz)
    mini.in_proj_ba = _SharedProj(real.in_proj_ba)

    n_slots = 4

    # ============================ CLEAN-ROOM SINGLE-PASS MODE ============================
    # GDN_SINGLE={mini|real}: run ONE prefill for ONE layer as the first-and-only heavy
    # causal_conv1d_fn forward in the process (mirrors real-engine usage), warmed once with
    # the EXACT measured args, then save {hs, out, conv_state, ssm_state} for a CPU-side A/B.
    # This sidesteps the in-process observer effect: prior conv calls with varying
    # strides/metadata destabilise the autotune-on-live-buffer kernel. Run twice (separate
    # processes) with an ISOLATED triton cache + identical CPU-seeded hs, then compare.
    single = os.environ.get("GDN_SINGLE")
    if single in ("mini", "real", "probe", "both"):
        import minisgl.gdn.layer as ML  # noqa: F401  (parity: same module as decode capture)
        banner(f"SINGLE-PASS prefill: {single} (warm-once, then measured, saved to disk)")
        # slot MUST be >= 1: cache index 0 == NULL_BLOCK_ID, which causal_conv1d_fn treats
        # as a padding/null block and SKIPS (returns out unwritten -> 0/NaN/aliased garbage).
        # Proven against a CPU reference conv: slot 0 -> WRONG, slot 1 -> CORRECT (rel 7e-3).
        # 3c: GDNStateCache must never allocate slot 0 to a real sequence.
        slot = int(os.environ.get("GDN_SLOT", "1"))
        torch.manual_seed(SEED)
        hs = torch.randn(T_PREFILL, HIDDEN).to(device=device, dtype=torch.bfloat16)
        print(f"  hs nan={torch.isnan(hs).any().item()} |max|={hs.float().abs().max().item():.3e} "
              f"(CPU-seeded fp32->device bf16; identical across processes)")
        md_p, qsl_p = md_prefill(slot, device, GDNAttentionMetadata)
        state_idx = torch.tensor([slot], device=device, dtype=torch.int32)
        has_init = torch.tensor([False], device=device)

        def _fresh_real_buffers():
            if dim_first:
                cv = torch.zeros(n_slots, CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device)
            else:
                cv = torch.zeros(n_slots, KCONV - 1, CONV_DIM, dtype=conv_dtype, device=device)
            return cv, torch.zeros(n_slots, HV, DV, DK, dtype=ssm_dtype, device=device)

        def _run_real():
            cv, sm = _fresh_real_buffers()
            real.kv_cache = [cv, sm]
            out, _, _ = drive_real(real, Q, md_p, hs, conv_dtype, ssm_dtype, device)
            cv_ds = cv if dim_first else cv.transpose(-1, -2)
            return out, cv_ds[slot].contiguous(), sm[slot]

        def _run_mini():
            cv = torch.zeros(n_slots, CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device)
            sm = torch.zeros(n_slots, HV, DV, DK, dtype=ssm_dtype, device=device)
            # conv_metadata None vs md_p are both correct at slot>=1 (probe: rel 7e-3 vs
            # CPU-ref); default None is the production-simple path. Toggle for confirmation.
            cmeta = md_p if os.environ.get("GDN_CONVMETA") == "mdp" else None
            out = mini.forward_prefill(hs, cv, sm, qsl_p, state_idx, has_init, conv_metadata=cmeta)
            return out, cv[slot], sm[slot]

        # WARMUP REGIME (proven): a single warm forward is NOT sufficient to settle the
        # in-place autotuned causal_conv1d_fn here (warm-once -> measured was NaN/0). The
        # GEMM-free multi-call conv warmup (2 iters x both namespaces, throwaway buffers)
        # is what made the first measured forward correct in the full harness. Replicate it.
        try:
            ww = mini._conv_weights()
            qslw = torch.tensor([0, T_PREFILL], dtype=torch.int32, device=device)
            ciw = torch.tensor([n_slots - 1], dtype=torch.int32, device=device)
            hiw = torch.tensor([False], device=device)
            cs_ds = torch.zeros(n_slots, CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device)
            cs_sd = torch.zeros(n_slots, KCONV - 1, CONV_DIM, dtype=conv_dtype, device=device)
            for _ in range(2):
                xw = torch.randn(T_PREFILL, CONV_DIM, dtype=conv_dtype, device=device).transpose(0, 1)
                Q.causal_conv1d_fn(xw.clone(), ww, None, activation="silu",
                                   conv_states=cs_sd.transpose(-1, -2), has_initial_state=hiw,
                                   cache_indices=ciw, query_start_loc=qslw, metadata=None)
                ML.causal_conv1d_fn(xw.clone(), ww, None, activation="silu",
                                    conv_states=cs_ds, has_initial_state=hiw,
                                    cache_indices=ciw, query_start_loc=qslw, metadata=None)
            print("  GEMM-free conv warmup regime complete")
        except Exception as e:  # noqa: BLE001
            print(f"  conv warmup raised (non-fatal): {type(e).__name__}: {e}")

        if single == "probe":
            # Isolate mini's conv_out=0: namespace(Q vs ML) x conv_states layout(DS vs
            # SD-transposed) x slot(0 vs 1), all with IDENTICAL packed input + md_p. The
            # kernel SKIPS (returns unwritten) any program whose cache_indices[seq] ==
            # null_block_id (=0); slot 0 may collide. Q SD s0 == real's actual config.
            banner("PROBE: namespace x layout x slot (identical input + md_p)")
            with torch.no_grad():
                _qk = mini.in_proj_qkvz(hs)
                _mqv, _, _, _ = mini._split_qkvz_ba(_qk, mini.in_proj_ba(hs), hs.shape[0])
            xin = _mqv.contiguous().transpose(0, 1)
            # GROUND TRUTH: pure-torch depthwise causal conv1d + silu (no init state).
            # out[c,t] = silu(sum_k w[c,k] * x_pad[c, t-3+k]). A variant is CORRECT iff its
            # conv_out matches this; otherwise the kernel didn't write (empty_like aliasing).
            import torch.nn.functional as _F
            _xc = _mqv.float().transpose(0, 1)  # (C, T)
            _w = ww.float()  # (C, K=4)
            _xp = _F.pad(_xc, (KCONV - 1, 0))  # left-pad causal
            _ref = torch.zeros_like(_xc)
            for _k in range(KCONV):
                _ref = _ref + _w[:, _k:_k + 1] * _xp[:, _k:_k + T_PREFILL]
            _ref = _F.silu(_ref).transpose(0, 1)  # (T, C)
            print(f"  input mixed_qkv |max|={_mqv.float().abs().max().item():.3e} "
                  f"| CPU-ref conv |max|={_ref.abs().max().item():.3e}", flush=True)
            def _mk(packed):  # fresh input each row (gapped = split-view, token-stride=qkvz_dim)
                qk = mini.in_proj_qkvz(hs)
                mqv, _, _, _ = mini._split_qkvz_ba(qk, mini.in_proj_ba(hs), hs.shape[0])
                return (mqv.contiguous() if packed else mqv).transpose(0, 1)

            # (tag, ns, layout, slot, metadata, packed): slot 1 avoids the NULL_BLOCK_ID=0
            # skip. Tests namespace, conv_states layout (DS vs real's SD-transposed), metadata
            # None vs md_p, and PACKED vs GAPPED input — at slot 1 ALL are CORRECT vs the CPU
            # ref (so gapped split-view == contiguous: NO explicit .contiguous() is needed).
            matrix = [
                ("Q  DS s0 mdp PK", Q, "DS", 0, md_p, True),
                ("ML DS s1 mdp PK", ML, "DS", 1, md_p, True),
                ("ML DS s1 None PK", ML, "DS", 1, None, True),
                ("ML SD s1 mdp PK", ML, "SD", 1, md_p, True),
                ("Q  SD s1 mdp PK", Q, "SD", 1, md_p, True),
                ("ML DS s1 None GAP", ML, "DS", 1, None, False),
                ("ML DS s1 mdp GAP", ML, "DS", 1, md_p, False),
            ]
            for tag, ns, layout, sl, meta, packed in matrix:
                if layout == "DS":
                    cs = torch.zeros(n_slots, CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device)
                    cs_arg = cs
                else:
                    cs = torch.zeros(n_slots, KCONV - 1, CONV_DIM, dtype=conv_dtype, device=device)
                    cs_arg = cs.transpose(-1, -2)
                ci = torch.tensor([sl], dtype=torch.int32, device=device)
                hi = torch.tensor([False], device=device)
                o = ns.causal_conv1d_fn(_mk(packed), ww, None,
                                        activation="silu", conv_states=cs_arg, has_initial_state=hi,
                                        cache_indices=ci, query_start_loc=qsl_p,
                                        metadata=meta).transpose(0, 1)
                d = (o.float() - _ref).abs().max().item()
                rel = d / (_ref.abs().max().item() + 1e-12)
                verdict = "CORRECT" if rel < 5e-2 else "WRONG/unwritten"
                print(f"  {tag:14s} conv_out nan={torch.isnan(o).any().item()!s:5s} "
                      f"|max|={o.float().abs().max().item():.3e} vs CPU-ref max|Δ|={d:.3e} "
                      f"rel={rel:.3e}  {verdict}", flush=True)
            return

        # ---- per-stage capture: localize divergence by stage (q/k/v/g/beta + conv + core) ----
        # IN-PROCESS so real and mini share weights (the weight-copy + _SharedProj only
        # equalize within one process) AND share hs. conv matches CPU-ref on both sides and
        # chunk is bit-exact (3b-1); a residual divergence localizes to fused_post_conv_prep.
        def _capture(which):
            stages: dict = {}
            prep_ns = Q if which == "real" else ML
            chunk_name = ("fla_chunk_gated_delta_rule" if which == "real"
                          else "chunk_gated_delta_rule")
            op, oc = prep_ns.fused_post_conv_prep, getattr(prep_ns, chunk_name)

            def wp(*a, **k):
                r = op(*a, **k)
                for nm, t in zip(("q", "k", "v", "g", "beta"), r):
                    stages[nm] = t.detach().float().cpu().flatten()
                return r

            def wch(*a, **k):
                r = oc(*a, **k)
                stages["core"] = r[0].detach().float().cpu().flatten()
                return r

            prep_ns.fused_post_conv_prep = wp
            setattr(prep_ns, chunk_name, wch)
            out, conv_s, ssm_s = (_run_real if which == "real" else _run_mini)()
            prep_ns.fused_post_conv_prep = op
            setattr(prep_ns, chunk_name, oc)
            print(f"  [{which}] out|max|={out.float().abs().max().item():.3e} "
                  f"ssm|max|={ssm_s.float().abs().max().item():.3e} "
                  f"conv_state|max|={conv_s.float().abs().max().item():.3e} | stages: " +
                  " ".join(f"{k}={v.abs().max().item():.3e}" for k, v in stages.items()), flush=True)
            return {"out": out.float().cpu(), "conv_state": conv_s.float().cpu(),
                    "ssm_state": ssm_s.float().cpu(), "stages": stages}

        if single == "both":
            # Single-process real-vs-mini: shared weights + hs + slot. The load-bearing,
            # layout-anchored parity (advisor): core_attn_out and ssm_state element-wise.
            R, M = _capture("real"), _capture("mini")
            banner("PARITY (in-process, shared weights+hs, slot>=1)")
            ok = True
            for k, t in (("core", R["stages"]), ("q", R["stages"]), ("k", R["stages"]),
                         ("v", R["stages"]), ("g", R["stages"]), ("beta", R["stages"])):
                a, b = R["stages"][k], M["stages"][k]
                md = (a - b).abs().max().item()
                rel = md / (b.abs().max().item() + 1e-12)
                print(f"  stage {k:5s} max|Δ|={md:.3e} rel={rel:.3e} "
                      f"|real|={a.abs().max():.3e} |mini|={b.abs().max():.3e}")
            for k in ("out", "ssm_state", "conv_state"):
                a, b = R[k].flatten(), M[k].flatten()
                md = (a - b).abs().max().item()
                rel = md / (b.abs().max().item() + 1e-12)
                lb = k in ("ssm_state",)  # out is z-dominated; conv_state is SD/DS frame
                good = (rel < 5e-3) or (md < 2e-2)
                if lb and not good:
                    ok = False
                print(f"  {k:11s} max|Δ|={md:.3e} rel={rel:.3e} |real|={a.abs().max():.3e} "
                      f"|mini|={b.abs().max():.3e}{'  <-- load-bearing' if lb else ''}")
            print("\nRESULT:", "PASS — in-process prefill parity (ssm_state + core)"
                  if ok else "FAIL", flush=True)
            return

        # single-layer capture/print only (debug). A cross-PROCESS real-vs-mini compare is
        # INVALID — each process random-inits its own real layer (incl. A_log ~ N(-2,0.3)),
        # so weights differ; use GDN_SINGLE=both for in-process, shared-weight parity.
        _capture(single)
        return

    # ---- prefill stage capture: localize where prefill signal dies (conv->prep->chunk) ----
    import minisgl.gdn.layer as ML

    pcap: dict = {}

    def _pwrap(ns, name, key, pick=lambda r: r):
        orig = getattr(ns, name)

        def w(*a, **k):
            r = orig(*a, **k)
            t = pick(r)
            pcap[key] = (t.detach().float().abs().max().item()
                         if torch.is_tensor(t) else None)
            return r

        setattr(ns, name, w)

    _pwrap(Q, "causal_conv1d_fn", "real.conv")
    _pwrap(ML, "causal_conv1d_fn", "mini.conv")
    _pwrap(Q, "fused_post_conv_prep", "real.v", pick=lambda r: r[2])
    _pwrap(ML, "fused_post_conv_prep", "mini.v", pick=lambda r: r[2])
    _pwrap(Q, "fla_chunk_gated_delta_rule", "real.chunk", pick=lambda r: r[0])
    _pwrap(ML, "chunk_gated_delta_rule", "mini.chunk", pick=lambda r: r[0])

    # ---- WARM the in-place autotuned causal_conv1d_fn (prefill conv) ----
    # causal_conv1d_fn autotunes on its FIRST call by benchmarking candidate configs on the
    # LIVE input/output buffer, so the result of an unwarmed measured prefill is sensitive to
    # op sequence. A GEMM-free direct warmup (no in_proj -> no hipBLASLt OOM risk) of BOTH
    # namespaces' conv settles the autotune cache first so the measured prefill is stable.
    # (3c needs an equivalent warmup-prefill hook before the first real batch.)
    banner("warm causal_conv1d_fn autotune (both namespaces; GEMM-free)")
    try:
        scw = n_slots - 1
        ww = mini._conv_weights()
        qslw = torch.tensor([0, T_PREFILL], dtype=torch.int32, device=device)
        ciw = torch.tensor([scw], dtype=torch.int32, device=device)
        hiw = torch.tensor([False], device=device)
        cs_ds = torch.zeros(n_slots, CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device)
        cs_sd = torch.zeros(n_slots, KCONV - 1, CONV_DIM, dtype=conv_dtype, device=device)
        for _ in range(2):
            xw = torch.randn(T_PREFILL, CONV_DIM, dtype=conv_dtype, device=device).transpose(0, 1)
            Q.causal_conv1d_fn(xw.clone(), ww, None, activation="silu",
                               conv_states=cs_sd.transpose(-1, -2), has_initial_state=hiw,
                               cache_indices=ciw, query_start_loc=qslw, metadata=None)
            ML.causal_conv1d_fn(xw.clone(), ww, None, activation="silu",
                                conv_states=cs_ds, has_initial_state=hiw,
                                cache_indices=ciw, query_start_loc=qslw, metadata=None)
        print("  conv autotune warmed")
    except Exception as e:  # noqa: BLE001
        print(f"  conv warmup raised (non-fatal): {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ PREFILL
    banner("PREFILL parity (output + conv_state + ssm_state)")
    # slot MUST be >= 1: cache index 0 == NULL_BLOCK_ID, which causal_conv1d_fn treats as a
    # null/padding block and SKIPS (returns conv output unwritten -> 0/NaN/aliased garbage).
    # The earlier "CPU-origin hs -> all-zeros" note was a MISATTRIBUTION of this slot-0 bug
    # (GPU hs sometimes aliased non-zero garbage; CPU hs aliased zero). hs origin is benign;
    # the prefer-the-clean validation is GDN_SINGLE=both. (See PORT.md 3b-3.)
    slot = 1
    torch.manual_seed(SEED)
    hs = torch.randn(T_PREFILL, HIDDEN, dtype=torch.bfloat16, device=device)
    # Sanity: in_proj produces signal (not zeros). mini shares real's in_proj (_SharedProj),
    # so mixed_qkv is bit-identical between the two paths — that is why prefill parity comes
    # out bit-exact (it isolates the GDN ORCHESTRATION, not the GEMM backend).
    with torch.no_grad():
        _ip = mini.in_proj_qkvz(hs)
        _mq, _zc, _, _ = mini._split_qkvz_ba(_ip, mini.in_proj_ba(hs), hs.shape[0])
    print(f"  prefill in_proj |max|: qkvz={_ip.abs().max().item():.3e} "
          f"mixed_qkv={_mq.abs().max().item():.3e} z={_zc.abs().max().item():.3e} "
          f"(input hs |max|={hs.abs().max().item():.3e})")
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

    # In-process real-vs-mini prefill (shared weights + in_proj + hs at slot>=1). At slot 0
    # the conv would be skipped (NULL_BLOCK_ID) and read back garbage; at slot>=1 it writes
    # correctly. conv_metadata=None is correct here (== precomputed md_p; verified vs CPU ref).
    mini_conv = torch.zeros(n_slots, CONV_DIM, KCONV - 1, dtype=conv_dtype, device=device)
    mini_ssm = torch.zeros(n_slots, HV, DV, DK, dtype=ssm_dtype, device=device)
    mini_out = mini.forward_prefill(hs, mini_conv, mini_ssm, qsl_p, state_idx, has_init,
                                    conv_metadata=None)
    real_out, _, _ = drive_real(real, Q, md_p, hs, conv_dtype, ssm_dtype, device)

    real_conv_ds = real.kv_cache[0] if dim_first else real.kv_cache[0].transpose(-1, -2)
    print(f"  prefill stage |max|: conv real={pcap.get('real.conv')} mini={pcap.get('mini.conv')} | "
          f"v(post_prep) real={pcap.get('real.v')} mini={pcap.get('mini.v')} | "
          f"chunk_core real={pcap.get('real.chunk')} mini={pcap.get('mini.chunk')}")
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
    # Guard against a vacuous parity claim: the prefill-accumulated ssm_state that
    # continue-decode reads back MUST be substantial, else "prefill.ssm_state Δ=0" is 0==0.
    print(f"  prefill state magnitudes (must be >> 0): "
          f"|ssm[slot]|={mini_ssm[slot].abs().max().item():.3e} "
          f"|conv[slot]|={mini_conv[slot].abs().max().item():.3e} "
          f"|output|={mini_out.abs().max().item():.3e}")
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
