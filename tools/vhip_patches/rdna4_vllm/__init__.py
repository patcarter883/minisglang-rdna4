# SPDX-License-Identifier: Apache-2.0
"""RDNA4 (gfx1201) native-HIP kernel wiring for vLLM 0.24.

One plugin, registered through supported vLLM hooks — no monkeypatching, no Triton fallbacks.
See docs/RDNA4_VLLM_WIRING_SPEC.md for the op inventory and the no-fallback / no-copy policy.
"""
