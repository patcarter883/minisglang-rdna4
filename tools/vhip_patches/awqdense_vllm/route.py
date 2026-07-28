# SPDX-License-Identifier: Apache-2.0
"""Route `auto_awq` DENSE linears on gfx12x to our W4A8 FP8-WMMA HIP kernel.

WHY THIS EXISTS
    `w4a8_vllm` already installs RocmW4A8Fp8WmmaLinearKernel at _POSSIBLE_KERNELS[ROCM][0], but for
    an AWQ checkpoint that registry is never consulted: AutoAWQConfig.get_quant_method() only routes
    to AutoAWQMarlinLinearMethod (the class that calls choose_mp_linear_kernel) on CUDA and CPU. On
    ROCm it returns AutoAWQLinearMethod, whose apply() calls vLLM's pure-Triton `awq_gemm_triton`.
    So on this box every AWQ dense linear ran stock Triton and our kernel was dead weight.

    That Triton kernel is also outright broken for bf16 — it accumulates in the OUTPUT dtype
    (`tl.dot(..., out_dtype=bfloat16)`), which Triton rejects, so a bf16 AWQ serve dies at graph
    capture. Routing to our kernel fixes the performance gap and sidesteps that bug at its source
    rather than patching a kernel we should not be running.

    Despite the name, AutoAWQMarlinLinearMethod is NOT Marlin-specific — it is the generic
    MPLinearKernel framework path (its own docstring: "Uses choose_mp_linear_kernel to select the
    best available kernel"), and vLLM already reuses it on CPU for exactly this reason. Its
    process_weights_after_loading() converts AWQ packing to the standard GPTQ-like layout that our
    adapter's process_weights_after_loading() expects, so no repacking work is needed here.

    Belongs upstream in `w4a8_vllm` alongside the MoE oracle hook; kept as a separate mounted plugin
    so the baked image stays untouched.

Set VLLM_ROCM_W4A8_AWQ_DENSE=0 to disable and fall back to vLLM's Triton AWQ path.
"""

from __future__ import annotations

import os

import torch

_REGISTERED = False


def _on_gfx12x() -> bool:
    try:
        from w4a8_vllm.vllm_adapter import _on_gfx12x as impl

        return bool(impl())
    except Exception:
        return False


def register(verbose: bool = True) -> bool:
    global _REGISTERED
    if _REGISTERED:
        return True
    if os.environ.get("VLLM_ROCM_W4A8_AWQ_DENSE", "1") != "1":
        if verbose:
            print("[awq_dense_hip] disabled via VLLM_ROCM_W4A8_AWQ_DENSE=0")
        return False

    from vllm.platforms import current_platform

    if not current_platform.is_rocm() or not _on_gfx12x():
        return False

    from vllm.model_executor.layers.quantization import auto_awq as aa
    from vllm.model_executor.kernels.linear import MPLinearLayerConfig
    from vllm.scalar_type import scalar_types

    from w4a8_vllm.vllm_adapter import RocmW4A8Fp8WmmaLinearKernel

    class RocmAwqMPLinearMethod(aa.AutoAWQMarlinLinearMethod):
        """AutoAWQMarlinLinearMethod minus its Marlin availability assert.

        The parent __init__ calls verify_marlin_supported(), which fails on ROCm — but we are not
        using Marlin at all, only the MPLinearKernel dispatch it happens to live behind.
        """

        def __init__(self, quant_config) -> None:
            self.quant_config = quant_config
            self.quant_type = scalar_types.uint4
            self.input_dtype = None

    _orig_get_quant_method = aa.AutoAWQConfig.get_quant_method

    def get_quant_method(self, layer, prefix):
        method = _orig_get_quant_method(self, layer, prefix)
        if not isinstance(method, aa.AutoAWQLinearMethod):
            return method  # MoE / skipped / already on the MPLinearKernel path

        # Only claim the layer if OUR kernel can actually take it — otherwise
        # choose_mp_linear_kernel would pick (or fail to find) some CUDA-only kernel, turning a
        # working-if-slow Triton path into a hard boot error.
        # get_quant_method() runs from inside LinearBase.__init__, so the per-partition shapes and
        # params_dtype may not be set yet; fall back to the unsharded sizes and a dtype the kernel
        # accepts either way (can_implement only gates on fp16-or-bf16). The authoritative config is
        # rebuilt with the real partition shapes in create_weights().
        try:
            in_size = getattr(layer, "input_size_per_partition", None) or layer.input_size
            out_size = getattr(layer, "output_size_per_partition", None) or layer.output_size
            act_type = getattr(layer, "params_dtype", None) or torch.get_default_dtype()
            if act_type not in (torch.float16, torch.bfloat16):
                act_type = torch.bfloat16
            cfg = MPLinearLayerConfig(
                full_weight_shape=(layer.input_size, layer.output_size),
                partition_weight_shape=(in_size, out_size),
                weight_type=scalar_types.uint4,
                act_type=act_type,
                group_size=self.group_size,
                zero_points=self.zero_point,
                has_g_idx=False,
            )
            ok, reason = RocmW4A8Fp8WmmaLinearKernel.can_implement(cfg)
        except Exception as e:  # pragma: no cover - defensive
            ok, reason = False, f"probe failed: {e}"

        if not ok:
            if verbose:
                print(f"[awq_dense_hip] {prefix}: staying on stock AWQ ({reason})")
            return method
        return RocmAwqMPLinearMethod(self)

    aa.AutoAWQConfig.get_quant_method = get_quant_method
    _REGISTERED = True
    if verbose:
        print("[awq_dense_hip] AWQ dense linears routed to RocmW4A8Fp8WmmaLinearKernel "
              "(was vLLM's Triton awq_gemm_triton)")
    return True
