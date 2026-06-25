"""GDN (Gated Delta Net) hybrid-attention support for minisgl-rdna4.

The `QwenGatedDeltaNet` layer (`layer.py`) runs entirely on native HIP kernels
(`torch.ops.gdn_hip.*`): depthwise causal conv, gated-delta-rule prefill/decode, and the
gated RMSNorm. There is no Triton dependency — importing the layer pulls in no JIT tree.
"""
