"""GDN (Gated Delta Net) hybrid-attention support for minisgl-rdna4.

`fla/` and `mamba/` hold Triton linear-attention kernels vendored verbatim from
vLLM 0.22.69 (flash-linear-attention origin); their `vllm.*` imports are rewritten
to `minisgl.gdn._compat`. These run on gfx1201 under HIP — integration, not a port.
The clean `QwenGatedDeltaNet` layer that wires them lands in Phase 3b.
"""
