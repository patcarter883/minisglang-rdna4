"""vLLM general-plugin entry point for the native GDN HIP ops (gfx1201).

Wired via ``entry_points={"vllm.general_plugins": ["gdn_hip = gdn_vllm.register:register"]}``
(setup.py). vLLM's ``load_general_plugins()`` imports this module (try/excepted by the loader) and
then calls ``register()`` UNWRAPPED in every process (incl. each EngineCore worker) — a raise here
would crash boot, so every failure path below is caught and turns into a clean ``return False``
(stay on the fla-Triton GDN path). Registering the OOT PluggableLayer subclass is what routes Qwen
GDN through gdn_hip; ``_force_fp32_gdn_state()`` forces the fp32 recurrent state (see below).
"""
import os

_REGISTERED = False


def _force_fp32_gdn_state() -> None:
    """The gdn_hip kernels read/write the GDN conv+ssm state IN PLACE as fp32 (``data_ptr<float>``),
    so the state MUST be fp32. There is exactly ONE clean chokepoint for this:
    ``MambaStateDtypeCalculator.gated_delta_net_state_dtype`` — BOTH the KV-cache-spec path
    (``layer.get_state_dtype`` -> MambaSpec) AND the attention-block-size planner
    (``model_cls.get_mamba_state_dtype_from_config`` in ``platforms/interface.py``, run BEFORE any
    layer exists) funnel through it. A per-layer ``get_state_dtype`` override only covers the FIRST
    path, so the planner sizes the attention block for the stock (smaller) mamba page and the actual
    fp32 spec then trips ``MambaSpec.page_size_bytes: assert page_size_padded >= page_size`` at KV
    init (observed: block 528 vs the 544 fp32 needs). Forcing fp32 at the calculator keeps both paths
    consistent. Runtime attribute-patch (no vLLM source edit; same category as w4a8's register hooks),
    gated by the caller, idempotent. Global-for-gated-delta-net, exactly like the old source patch."""
    import torch
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    cur = MambaStateDtypeCalculator.gated_delta_net_state_dtype
    if getattr(getattr(cur, "__func__", cur), "_gdn_hip_fp32", False):
        return

    def _fp32_gdn(cls, *args, **kwargs):
        return (torch.float32, torch.bfloat16)  # conv fp32 (kernel float*), ssm bf16 (halve state traffic; kernel state_t-dispatched)

    _fp32_gdn._gdn_hip_fp32 = True
    MambaStateDtypeCalculator.gated_delta_net_state_dtype = classmethod(_fp32_gdn)


def _install_fp8_hybrid_failfast() -> None:
    """FAIL FAST on a genuinely un-unifiable fp8-KV + hybrid (mamba/GDN) config.

    vLLM sizes the attention block up so its page >= the mamba state page
    (``Platform._align_hybrid_block_size``); with fp8 attention KV (half-size page) it just picks a
    bigger block, which normally works (e.g. GDN Qwen3.5-4B -> block 1072, boots fine). But if the
    required block exceeds the allowed size the function hits a BARE ``assert attn_page_size >=
    mamba_page_size`` — an opaque failure that, without this, surfaces ~25 min into warmup at the
    downstream ``MambaSpec.page_size_bytes`` assert. We wrap the (early, config-time) alignment so that
    on that assert, for an fp8 KV cache, we raise a CLEAR, actionable error immediately. Working configs
    are untouched (the wrapper only intercepts the assert). Runtime attribute-patch, idempotent."""
    from vllm.platforms.interface import Platform

    orig = Platform.__dict__.get("_align_hybrid_block_size")
    if orig is None or getattr(getattr(orig, "__func__", orig), "_gdn_failfast", False):
        return
    orig_func = orig.__func__

    def _wrapped(cls, vllm_config, *args, **kwargs):
        try:
            return orig_func(cls, vllm_config, *args, **kwargs)
        except AssertionError as e:
            dt = str(getattr(vllm_config.cache_config, "cache_dtype", "auto"))
            if any(t in dt for t in ("fp8", "e4m3", "e5m2")):
                raise RuntimeError(
                    f"FP8 KV cache (kv_cache_dtype={dt}) is incompatible with this hybrid "
                    "(mamba/GDN) model's recurrent-state page size: the attention page cannot be "
                    "grown to match the fp32 mamba state page within the allowed block size. "
                    "Re-run with `--kv-cache-dtype auto` (bf16 attention KV), which unifies for "
                    "these architectures."
                ) from e
            raise

    _wrapped._gdn_failfast = True
    Platform._align_hybrid_block_size = classmethod(_wrapped)


def register(verbose: bool = True) -> bool:
    """Register QwenGdnHipAttention as the OOT replacement for QwenGatedDeltaNetAttention.
    Returns True if registered (or already), False if disabled / not gfx12x / load failed (all of
    which leave the stock fla-Triton GDN path in charge)."""
    global _REGISTERED
    if _REGISTERED:
        return True
    if os.environ.get("VLLM_GDN_HIP", "0") != "1":
        if verbose:
            print("[gdn_hip] disabled via VLLM_GDN_HIP=0 — fla-Triton GDN path.")
        return False

    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return False
    try:
        from vllm.platforms.rocm import on_gfx12x
    except Exception:
        def on_gfx12x() -> bool:
            return False
    if not on_gfx12x():
        if verbose:
            print("[gdn_hip] not gfx12x — native GDN ops disabled (stock path).")
        return False

    # Guard double-registration: register_oot assert-crashes on a duplicate name.
    from vllm.model_executor.custom_op import op_registry_oot

    if "QwenGatedDeltaNetAttention" in op_registry_oot:
        _REGISTERED = True
        return True

    try:
        _force_fp32_gdn_state()  # keep block-planner + KV-spec state dtype consistent (fp32)
        _install_fp8_hybrid_failfast()  # clear early error if fp8 KV can't unify with the mamba page
        from gdn_vllm import vllm_oot  # noqa: F401  runs the @PluggableLayer.register_oot decorator
    except Exception as e:  # build/load failure -> fall back to fla-Triton GDN
        print(f"[gdn_hip] VLLM_GDN_HIP=1 but gdn_hip load/registration failed: {e}; "
              "falling back to the fla-Triton GDN path.")
        return False

    _REGISTERED = True
    if verbose:
        print("[gdn_hip] native HIP GDN ops ENABLED — QwenGatedDeltaNetAttention -> "
              "QwenGdnHipAttention via PluggableLayer.register_oot (no Triton JIT on the GDN path).")
    return True
