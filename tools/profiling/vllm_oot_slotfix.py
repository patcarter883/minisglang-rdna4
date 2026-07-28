"""Out-of-tree ``PluggableLayer`` subclass that routes the Qwen Gated-Delta-Net linear-attention
compute (conv1d + gated-delta-rule + gated RMSNorm) through the native gdn_hip HIP ops on gfx1201
— with ZERO edits to vLLM source (qwen_gdn_linear_attn.py / mamba_utils.py).

Mechanism (verified): GDN compute is a MODEL layer (``QwenGatedDeltaNetAttention._forward_core``),
NOT an AttentionImpl (mamba backends are metadata-only). ``QwenGatedDeltaNetAttention`` is a
``PluggableLayer``; ``PluggableLayer.__new__`` swaps the in-tree class for a subclass registered
under its ``__name__``. So we subclass + ``@PluggableLayer.register_oot`` and override methods only.

IMPORTANT: ``__new__`` returns the subclass instance but Python runs the BASE ``__init__`` — this
subclass may ONLY override methods, NEVER add ``__init__`` state. The overrides read existing attrs
only, so this is safe.

Spec-decode (multi-query-per-seq) and warmup (no attn_metadata) keep the stock fla-Triton path;
gdn_hip covers the non-spec single-token decode + varlen-recurrent / WMMA-chunked prefill only.

``get_state_dtype`` forces the GDN conv+ssm state to fp32 (the gdn_hip kernels read/write the state
in place as fp32) — scoped to Qwen GDN only; this replaces the old mamba_utils.py global patch.
"""
from __future__ import annotations

import os

import torch

from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

import gdn_hip  # noqa: F401  canonical rdna4-hip-kernels pkg: loads gdn_hip_C + registers fakes

logger = init_logger(__name__)

# WMMA matrix-core chunked prefill is the default GDN prefill when gdn_hip is on (4.8-5.9x faster
# than the scalar recurrent prefill); set VLLM_GDN_HIP_RECURRENT_ONLY=1 to force the recurrent path.
VLLM_GDN_HIP_RECURRENT_ONLY = os.environ.get("VLLM_GDN_HIP_RECURRENT_ONLY", "0") == "1"

# Run the spec-decode (multi-query-per-seq) GDN path entirely on gdn_hip (verify ops), no Triton.
# WORKS: coherent under graph capture, and it FIXES the concurrency hang — vLLM's fla-Triton spec
# path reaches the only @triton.autotune'd GDN kernel (chunk_scaled_dot_kkt, via chunk_gated_delta_rule)
# on any step that mixes spec-decode with a prefill; a cold tune of a fresh shape outlives the
# executor RPC timeout, parking BOTH TP ranks in do_bench (measured 19+ min) so vLLM declares the
# workers dead. That reads as a TP deadlock but is an unbounded first-run compile.
#
# THE BUG THAT COST THE MOST (for anyone extending this): the prefill/verify kernels used to assume
# a CONTIGUOUS state. Passing vLLM's paged conv/ssm tensors raw produced one correct token then
# token-0 garbage (" Paris!!!!!!"), because `core` is computed in registers and stays right while
# the state written back landed in the wrong place. vLLM carves the mamba cache with
# torch.as_strided: the inner (HV,V,K) dims ARE contiguous, but the SLOT stride is the padded page
# size (conv and ssm share a page), so the packed `((slot*HV+hv)*V+r)*K` walked into a neighbouring
# slot. Worked around first with a whole-cache .contiguous() shadow (a 2.75 GB copy worth ~64% of
# decode, 39.1 tok/s), then with a compact gather/scatter of just the touched slots (49.6).
# FIXED PROPERLY 2026-07-28: gdn_prefill / gdn_prefill_verify / gdn_prefill_wmma /
# gdn_prefill_chunked now take ssm_state.stride() and index the paged cache in place, exactly like
# the decode kernels always did. Verified bit-exact paged-vs-contiguous for all four, fp32 and bf16
# state (tools/vhip_patches/test_stride_aware_state.py). Both workarounds are gone below.
#
# MEASURED 2026-07-28 (BEFORE this fix), Qwen3.6-35B-A3B TP=2, MTP K=2, graph capture ON, 60k ctx,
# max_num_seqs=4:
#     bs=1 decode      49.6 tok/s   (no-spec 75.1 | fla-Triton spec 98.0 | full-shadow 39.1)
#     concurrency-4   156 tok/s peak / 134 mean, 6/6 trials  (no-spec 222; Triton spec HANGS)
#     accept_len ~2.4/3
# This is the only spec config that survives concurrency. Remaining known cost on this path: the
# per-position state publish loop below, and gdn_prefill_verify being the scalar RECURRENT oracle
# (no WMMA). NEXT LEVER: a WMMA verify path and/or folding the per-position publish into the kernel.
VLLM_GDN_HIP_SPEC = os.environ.get("VLLM_GDN_HIP_SPEC", "1") == "1"

# One-shot dump of the spec state-slot contract (slot table, num_accepted, chosen load slot) for the
# first few real spec steps, to check it against what vLLM's own kernel would read.
# REQUIRES --enforce-eager. Every field is a `.tolist()` on a device tensor = a D2H copy, which under
# CUDA graph capture dies with "Cannot copy between CPU and CUDA tensors during CUDA graph capture".
# That is inherent to dumping device state, not a bug to fix — boot eager when you need this.
VLLM_GDN_HIP_SPEC_DEBUG = os.environ.get("VLLM_GDN_HIP_SPEC_DEBUG", "0") == "1"
_SPEC_DBG_N = 0
# BISECT: skip the per-position scatter, leaving only the kernels' own in-place final-state write.
_SPEC_NO_SCATTER = os.environ.get("VLLM_GDN_HIP_SPEC_NO_SCATTER", "0") == "1"
_SPEC_AB = os.environ.get("VLLM_GDN_HIP_SPEC_AB", "0") == "1"
_SPEC_AB_N = 0
# BISECT: restore the pre-stride-fix compact gather/scatter (copy the touched slots into a contiguous
# 1-based buffer, run, copy back) so its cost can be MEASURED against the raw-paged path rather than
# assumed. Only meaningful now that the kernels are stride-aware — the compact buffer is just another
# (contiguous) stride set to them. Correctness is identical either way; this is purely for attribution.
_SPEC_LEGACY_GATHER = os.environ.get("VLLM_GDN_HIP_SPEC_GATHER", "0") == "1"
# FUSED per-position publish (default ON): the verify kernels take the 2-D slot table + the
# acceptance count, resolve the load slot themselves and store each token's state straight into
# conv_state/ssm_state[slots[n,t]] — exactly what vLLM's fla kernel does (INPLACE_FINAL_STATE).
# Deletes the per-position Python scatter (MEASURED 141 us/layer, more than the recurrence kernel
# itself), its two scratch buffers, and the host-side load-slot gather. Set 0 for the old path.
_SPEC_FUSED_PUBLISH = os.environ.get("VLLM_GDN_HIP_SPEC_FUSED", "1") == "1"


def _gdn_select_prefill(layer):
    """Pick the GDN prefill op: WMMA matrix-core chunked prefill (4.8-5.9x faster) for 128-dim
    Qwen3.5/3.6 GDN heads, else the scalar recurrent prefill (also forced by
    VLLM_GDN_HIP_RECURRENT_ONLY=1). Both share the fp32 op boundary and the same state cache."""
    if (not VLLM_GDN_HIP_RECURRENT_ONLY
            and getattr(layer, "head_k_dim", 0) == 128
            and getattr(layer, "head_v_dim", 0) == 128):
        return gdn_hip.gdn_prefill_wmma
    return gdn_hip.gdn_prefill


@PluggableLayer.register_oot(name="QwenGatedDeltaNetAttention")
class QwenGdnHipAttention(QwenGatedDeltaNetAttention):
    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        # gdn_hip native path: route the non-spec prefill/decode GDN compute through the AOT HIP
        # ops (no Triton JIT). Spec-decode keeps the fla-Triton path — gdn_hip is single-token-per-seq
        # decode / varlen-recurrent prefill only.
        raw = get_forward_context().attn_metadata
        am = raw[self.prefix] if isinstance(raw, dict) else None
        if am is None:
            # WARMUP / memory-profiling pass (no attn_metadata): the output is discarded, so do NOT
            # run the fla-Triton GDN prefill just to feed the profiler — on ROCm the native GDN prefill
            # is ALWAYS Triton (flashinfer/cutedsl are CUDA-only), and its JIT/autotune is the exact
            # ~30-min cost gdn_hip exists to eliminate (worse: a fresh GDN shape misses the warm cache).
            # Return the pre-allocated core_attn_out buffer; the REAL forward runs gdn_hip, whose
            # workspace is smaller than Triton's, so the profiled peak (hence KV sizing) stays safe.
            return core_attn_out
        if am.spec_sequence_masks is None:
            return self._forward_core_gdn_hip(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=am,
            )
        # Spec-decode: run the WHOLE GDN path on gdn_hip. A hybrid (gdn_hip writes the recurrent
        # state, Triton's spec kernel reads it) makes two implementations share an INTERNAL state
        # layout; head_v_dim == head_k_dim == 128, so a mismatch is shape-invisible and surfaces only
        # as corrupted generation (one correct token, then token 0 forever). Owning both the read and
        # the write keeps the layout private — and removes the last autotuning Triton kernel.
        if VLLM_GDN_HIP_SPEC:
            return self._forward_core_gdn_hip_spec(
                mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=core_attn_out, am=am,
            )
        return super()._forward_core(mixed_qkv, b, a, core_attn_out)

    # NOTE: fp32 recurrent state is NOT forced here. A per-layer get_state_dtype override only
    # affects the KV-cache-spec path and NOT the attention-block-size planner
    # (get_mamba_state_dtype_from_config), which desyncs them and trips the MambaSpec page_size
    # assert. It is forced at the shared calculator in register.py::_force_fp32_gdn_state instead.

    def _forward_core_gdn_hip(
        self,
        *,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """Native-HIP GDN core (no Triton JIT). Mirrors the fla-Triton ``_forward_core`` non-spec
        path exactly — conv1d + gated-delta-rule + the gated RMSNorm output — but every op is an
        AOT-compiled ``torch.ops.gdn_hip.*`` kernel. fp32 at the op boundary: the conv/ssm state
        are fp32 (forced in get_state_dtype when gdn_hip is on) and the bf16 activations are cast
        ``.float()`` here; ``core_attn_out`` is written back in the layer dtype by the caller's
        output projection.

        Drops ``fused_post_conv_prep`` + ``l2norm_fwd``: the gdn_hip recurrence folds the q/k
        L2-norm and the g/beta (softplus/sigmoid) computation in, consuming raw ``a, b``.
        """
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        has_initial_state = attn_metadata.has_initial_state
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels (DS layout); SD needs a
        # transpose. gdn_hip's conv kernels take that [slots, C, W-1] layout directly.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # Every gdn_hip kernel on this path is stride-aware, so vLLM's paged mamba cache is consumed
        # in place via its real strides — no contiguous shadow, no gather/scatter, on prefill or
        # decode. (fla-Triton reads the strided view directly; this matches it.)

        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        ).float()
        state_indices = non_spec_state_indices_tensor[:num_actual_tokens].long()  # type: ignore[index]
        scale = self.head_k_dim**-0.5
        # A_log is fp32 by construction; dt_bias defaults fp32 but the loader may cast it to the
        # model dtype — float() both so the fp32 gdn_hip kernels never see a bf16 input.
        A_log = self.A_log.float()
        dt_bias = self.dt_bias.float()

        n_k = self.num_k_heads // self.tp_size
        n_v = self.num_v_heads // self.tp_size

        def _split_conv_qkv(conv_out: torch.Tensor):
            q, k, v = conv_out.split(
                [self.key_dim // self.tp_size,
                 self.key_dim // self.tp_size,
                 self.value_dim // self.tp_size],
                dim=-1,
            )
            q = q.reshape(-1, n_k, self.head_k_dim).contiguous()
            k = k.reshape(-1, n_k, self.head_k_dim).contiguous()
            v = v.reshape(-1, n_v, self.head_v_dim).contiguous()
            return q, k, v

        if attn_metadata.num_prefills > 0:
            # Varlen recurrent/WMMA path (covers any decode-len-1 seqs in the same batch via
            # cu_seqlens). The prefill kernels are stride-aware too, so the paged cache goes in RAW —
            # this used to .contiguous() the WHOLE cache and copy it back on every prefill step.
            assert has_initial_state is not None
            x = mixed_qkv.float().contiguous()  # prefill keeps fp32 acts (not the decode hot path)
            conv_out = gdn_hip.causal_conv1d_fwd(
                x,
                conv_weights,
                None,
                non_spec_query_start_loc,  # cu_seqlens int32
                state_indices,
                has_initial_state.to(torch.uint8),
                conv_state,
                1,  # SiLU
            )
            q, k, v = _split_conv_qkv(conv_out)
            core = _gdn_select_prefill(self)(
                q, k, v, a.float(), b.float(), A_log, dt_bias,
                non_spec_query_start_loc, state_indices,
                has_initial_state.to(torch.uint8), ssm_state, scale, 1,
            )
        else:
            # Pure single-token-per-seq decode: the STRIDE-AWARE gdn_hip kernels index the (possibly
            # non-contiguous) paged mamba cache IN PLACE via state_indices + the tensor's real strides —
            # no gather, no scatter, no contiguous shadow (exactly what fla-Triton does). state_indices
            # carries the real slot ids (slot 0 == NULL_BLOCK_ID, handled in-kernel).
            # FUSED cast-free decode: ONE launch (conv + l2norm + recurrence, stride-aware paged cache,
            # raw core out) replaces causal_conv1d_update + 3 splits + gdn_decode — the launch-storm
            # cut that closes the residual gap to fla. fp16 acts (scalar_t), state_t from ssm_state;
            # conv_weights stays fp32 (`const float*`). Returns the RAW core; caller applies self.norm.
            core = gdn_hip.ops.gdn_decode_conv(
                mixed_qkv.contiguous(), conv_weights, None, conv_state,
                a, b, A_log, dt_bias, ssm_state, state_indices, 1, scale, 1,
            )

        core_attn_out[:num_actual_tokens] = core.to(core_attn_out.dtype)
        return

    def _forward_core_gdn_hip_tokens(
        self,
        *,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        am: GDNAttentionMetadata,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> torch.Tensor:
        """Non-spec (prefill / plain-decode) gdn_hip core for an explicit token slice.

        Same ops/conventions as ``_forward_core_gdn_hip`` but RETURNS the core output instead of
        writing ``core_attn_out``, so the spec path can stitch both halves back into token order.
        """
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        ).float()
        state_indices = am.non_spec_state_indices_tensor.long()
        scale = self.head_k_dim ** -0.5
        A_log = self.A_log.float()
        dt_bias = self.dt_bias.float()
        n_k = self.num_k_heads // self.tp_size
        n_v = self.num_v_heads // self.tp_size

        if am.num_prefills > 0:
            has_init = am.has_initial_state.to(torch.uint8)
            conv_out = gdn_hip.causal_conv1d_fwd(
                mixed_qkv.float().contiguous(), conv_weights, None,
                am.non_spec_query_start_loc, state_indices, has_init, conv_state, 1,
            )
            q, k, v = conv_out.split(
                [self.key_dim // self.tp_size, self.key_dim // self.tp_size,
                 self.value_dim // self.tp_size], dim=-1,
            )
            return _gdn_select_prefill(self)(
                q.reshape(-1, n_k, self.head_k_dim).contiguous(),
                k.reshape(-1, n_k, self.head_k_dim).contiguous(),
                v.reshape(-1, n_v, self.head_v_dim).contiguous(),
                a.float().contiguous(), b.float().contiguous(), A_log, dt_bias,
                am.non_spec_query_start_loc, state_indices, has_init, ssm_state, scale, 1,
            )
        return gdn_hip.ops.gdn_decode_conv(
            mixed_qkv.contiguous(), conv_weights, None, conv_state,
            a, b, A_log, dt_bias, ssm_state, state_indices, 1, scale, 1,
        )

    def _forward_core_gdn_hip_spec(
        self,
        *,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        am: GDNAttentionMetadata,
    ):
        """Spec-decode GDN core, entirely on gdn_hip — no Triton on any sub-path.

        ROLLBACK CONTRACT: vLLM gives each spec sequence ``num_spec + 1`` mamba slots
        (``spec_state_indices_tensor[s, :]``) and reports the previous step's acceptance in
        ``num_accepted_tokens``. We honour that table but fill it ourselves:
          load  <- slot spec_state_indices[s, num_accepted[s] - 1]   (resume from what was accepted)
          store -> slot spec_state_indices[s, t] for every query position t, from the verify scratch
        so a later step can resume from whichever prefix ends up accepted. Because gdn_hip performs
        both the load and the store, the state layout stays private and self-consistent.
        """
        # ---- one-shot A/B vs the (known-correct) fla-Triton spec path on the SAME step ----------
        # Snapshot conv+ssm state, run upstream, capture its output+state, restore, then run ours and
        # diff. This compares real in-serve data instead of a hand-written reference (mine was wrong).
        global _SPEC_AB_N
        if _SPEC_AB and _SPEC_AB_N < 2:
            _SPEC_AB_N += 1
            _kv = self.kv_cache
            _cs = _kv[0] if is_conv_state_dim_first() else _kv[0].transpose(-1, -2)
            # Snapshot only the SLOTS this step touches (cloning the whole cache OOMs at serve
            # memory-utilisation; these rows are a few hundred KB).
            _sl = am.spec_state_indices_tensor[: am.num_spec_decodes].flatten().long().unique()
            _c0, _s0 = _cs[_sl].clone(), _kv[1][_sl].clone()
            _out_tri = core_attn_out.clone()
            super()._forward_core(mixed_qkv, b, a, _out_tri)
            _c_tri, _s_tri = _cs[_sl].clone(), _kv[1][_sl].clone()
            _cs[_sl] = _c0; _kv[1][_sl] = _s0          # restore: ours starts from the same state
            self._forward_core_gdn_hip_spec_impl(
                mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=core_attn_out, am=am)
            n = am.num_actual_tokens
            print(f"[gdn_ab#{_SPEC_AB_N}] {self.prefix} slots={_sl.tolist()} "
                  f"core_maxdiff={(core_attn_out[:n].float()-_out_tri[:n].float()).abs().max().item():.5f} "
                  f"ssm_maxdiff={(_kv[1][_sl].float()-_s_tri.float()).abs().max().item():.5f} "
                  f"conv_maxdiff={(_cs[_sl].float()-_c_tri.float()).abs().max().item():.5f} "
                  f"|core|={_out_tri[:n].abs().max().item():.4f} |ssm|={_s_tri.abs().max().item():.4f}",
                  flush=True)
            return
        return self._forward_core_gdn_hip_spec_impl(
            mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=core_attn_out, am=am)

    def _forward_core_gdn_hip_spec_impl(self, *, mixed_qkv, b, a, core_attn_out, am):
        spec_idx = am.spec_token_indx
        non_spec_idx = am.non_spec_token_indx
        spec_state_indices = am.spec_state_indices_tensor
        num_accepted = am.num_accepted_tokens
        n_spec = am.num_spec_decodes
        assert spec_state_indices is not None and num_accepted is not None

        kv = self.kv_cache
        conv_state = kv[0] if is_conv_state_dim_first() else kv[0].transpose(-1, -2)
        ssm_state = kv[1]

        n_tok = am.num_actual_tokens
        mixed_qkv, a, b = mixed_qkv[:n_tok], a[:n_tok], b[:n_tok]

        has_non_spec = non_spec_idx is not None and non_spec_idx.numel() > 0
        mqkv_spec = mixed_qkv.index_select(0, spec_idx) if has_non_spec else mixed_qkv
        a_spec = a.index_select(0, spec_idx) if has_non_spec else a
        b_spec = b.index_select(0, spec_idx) if has_non_spec else b

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        ).float()
        A_log, dt_bias = self.A_log.float(), self.dt_bias.float()
        scale = self.head_k_dim ** -0.5
        n_k = self.num_k_heads // self.tp_size
        n_v = self.num_v_heads // self.tp_size

        sqsl = am.spec_query_start_loc[: n_spec + 1]
        max_qlen = int(spec_state_indices.size(-1))
        slots = spec_state_indices[:n_spec]

        has_init = torch.ones(n_spec, device=mixed_qkv.device, dtype=torch.uint8)
        # FUSED path: hand `slots` (int32 [n_spec, max_qlen]) and `num_accepted` (int32) straight to
        # the kernels — they are templated on the index dtype, so no .long() cast per layer, and the
        # load slot is resolved in-kernel. Only the legacy bisect path gathers on the host.
        if _SPEC_LEGACY_GATHER or not _SPEC_FUSED_PUBLISH:
            acc_t = num_accepted[:n_spec].to(torch.long)
            load_slots = slots.gather(1, (acc_t.clamp(min=1) - 1)[:, None]).squeeze(1).long()
        else:
            acc_t, load_slots = num_accepted[:n_spec], None

        if VLLM_GDN_HIP_SPEC_DEBUG:
            global _SPEC_DBG_N
            if _SPEC_DBG_N < 6:
                _SPEC_DBG_N += 1
                print(f"[gdn_spec_dbg#{_SPEC_DBG_N}] prefix={self.prefix} n_spec={n_spec} "
                      f"max_qlen={max_qlen} qlens={(sqsl[1:]-sqsl[:-1]).tolist()} "
                      f"num_accepted={num_accepted[:n_spec].tolist()} "
                      f"slots={slots.tolist()} "
                      # load_slots is None on the fused path (the kernel resolves it), so derive the
                      # SAME selection here for the dump instead of dereferencing None.
                      f"my_load={(load_slots if load_slots is not None else slots.long().gather(1, (num_accepted[:n_spec].long().clamp(min=1) - 1)[:, None]).squeeze(1)).tolist()} "
                      f"resolved_by={'host' if load_slots is not None else 'kernel'} "
                      f"conv_state.shape={tuple(conv_state.shape)} conv_contig={conv_state.is_contiguous()} "
                      f"ssm_state.shape={tuple(ssm_state.shape)} W={conv_weights.size(1)} "
                      f"null_in_slots={(slots<=0).any(dim=1).tolist()}", flush=True)

        # Every path below hands the verify kernels the RAW paged cache — they index it through the
        # tensors' real strides. (vLLM carves the mamba cache with torch.as_strided: the inner dims
        # are contiguous but the SLOT stride is the padded page size, not HV*V*K, so assuming the
        # packed layout walked into the neighbouring slot — that is what corrupted generation.)
        # The paths differ only in HOW the per-position state is published; see each branch.
        if _SPEC_FUSED_PUBLISH and not _SPEC_LEGACY_GATHER:
            # ONE kernel each for conv and ssm: they select the load slot from
            # slots[n, num_accepted[n]-1] and publish every query position straight into
            # conv_state/ssm_state[slots[n, t]] in-kernel. No scratch, no Python scatter loop,
            # no host-side gather — this is what vLLM's fla kernel does (INPLACE_FINAL_STATE).
            conv_out, _ = gdn_hip.causal_conv1d_fwd_verify(
                mqkv_spec.float().contiguous(), conv_weights, None,
                sqsl, None, has_init, conv_state, max_qlen, 1, slots, acc_t,
            )
            q, k, v = conv_out.split(
                [self.key_dim // self.tp_size, self.key_dim // self.tp_size,
                 self.value_dim // self.tp_size], dim=-1,
            )
            core_spec, _ = gdn_hip.gdn_prefill_verify(
                q.reshape(-1, n_k, self.head_k_dim).contiguous(),
                k.reshape(-1, n_k, self.head_k_dim).contiguous(),
                v.reshape(-1, n_v, self.head_v_dim).contiguous(),
                a_spec.float().contiguous(), b_spec.float().contiguous(), A_log, dt_bias,
                sqsl, None, has_init, ssm_state, max_qlen, scale, 1, slots, acc_t,
            )
            return self._finish_spec(core_spec, core_attn_out, am, mixed_qkv, a, b,
                                     conv_state, ssm_state, n_tok, spec_idx, non_spec_idx,
                                     has_non_spec)

        if _SPEC_LEGACY_GATHER:
            n_ld = load_slots.numel()
            conv_k = conv_state.new_zeros((n_ld + 1, *conv_state.shape[1:]))
            ssm_k = ssm_state.new_zeros((n_ld + 1, *ssm_state.shape[1:]))
            conv_k[1:] = conv_state[load_slots]
            ssm_k[1:] = ssm_state[load_slots]
            cs_in, ss_in = conv_k, ssm_k
            idx_in = torch.arange(1, n_ld + 1, device=load_slots.device, dtype=torch.long)
        else:
            cs_in, ss_in, idx_in = conv_state, ssm_state, load_slots

        conv_out, conv_scratch = gdn_hip.causal_conv1d_fwd_verify(
            mqkv_spec.float().contiguous(), conv_weights, None,
            sqsl, idx_in, has_init, cs_in, max_qlen, 1,
        )
        q, k, v = conv_out.split(
            [self.key_dim // self.tp_size, self.key_dim // self.tp_size,
             self.value_dim // self.tp_size], dim=-1,
        )
        core_spec, ssm_scratch = gdn_hip.gdn_prefill_verify(
            q.reshape(-1, n_k, self.head_k_dim).contiguous(),
            k.reshape(-1, n_k, self.head_k_dim).contiguous(),
            v.reshape(-1, n_v, self.head_v_dim).contiguous(),
            a_spec.float().contiguous(), b_spec.float().contiguous(), A_log, dt_bias,
            sqsl, idx_in, has_init, ss_in, max_qlen, scale, 1,
        )
        if _SPEC_LEGACY_GATHER:
            conv_state[load_slots] = conv_k[1:]
            ssm_state[load_slots] = ssm_k[1:]

        # Publish each query position into its own slot so any accepted prefix can be resumed.
        # CUDAGRAPH-SAFE: no .nonzero()/.item() anywhere — a data-dependent select here is a
        # device->host sync and raises "operation not permitted when stream is capturing". Instead
        # the loop bound is a Python constant and every row is written unconditionally, with a
        # sequence shorter than max_qlen clamped to its own last position. Those surplus slots are
        # never read back (the next step loads slot num_accepted-1, and num_accepted <= qlen), so
        # re-writing the final state into them is harmless.
        conv_w = conv_weights.size(1) - 1
        qlens = (sqsl[1:] - sqsl[:-1]).to(torch.long)
        rows = torch.arange(n_spec, device=mixed_qkv.device)
        last = (qlens - 1).clamp(min=0)
        for t in range(max_qlen if not _SPEC_NO_SCATTER else 0):
            src = torch.minimum(torch.full_like(last, t), last)
            tgt = slots[:, t].long()
            ssm_state[tgt] = ssm_scratch[src, rows].to(ssm_state.dtype)
            # vLLM WIDENS the conv slot under spec to (W-1) + num_spec so its own kernel can roll
            # back by a read offset; our verify scratch is a plain (W-1) window per position. Since
            # gdn_hip owns both the read and the write we just use the leading W-1 columns
            # consistently and leave the rollback to the per-position slot table above.
            conv_state[tgt, :, :conv_w] = conv_scratch[src, rows].to(conv_state.dtype)

        return self._finish_spec(core_spec, core_attn_out, am, mixed_qkv, a, b,
                                 conv_state, ssm_state, n_tok, spec_idx, non_spec_idx,
                                 has_non_spec)

    def _finish_spec(self, core_spec, core_attn_out, am, mixed_qkv, a, b, conv_state, ssm_state,
                     n_tok, spec_idx, non_spec_idx, has_non_spec):
        """Stitch the spec-decode core back into token order, running the non-spec half if the step
        mixes both. Shared by the fused and scratch publish paths."""
        if not has_non_spec:
            core_attn_out[:n_tok] = core_spec.to(core_attn_out.dtype)
            return

        core_ns = self._forward_core_gdn_hip_tokens(
            mixed_qkv=mixed_qkv.index_select(0, non_spec_idx),
            a=a.index_select(0, non_spec_idx),
            b=b.index_select(0, non_spec_idx),
            am=am, conv_state=conv_state, ssm_state=ssm_state,
        )
        merged = torch.empty((n_tok, *core_spec.shape[1:]),
                             dtype=core_spec.dtype, device=core_spec.device)
        merged.index_copy_(0, spec_idx, core_spec)
        merged.index_copy_(0, non_spec_idx, core_ns.to(core_spec.dtype))
        core_attn_out[:n_tok] = merged.to(core_attn_out.dtype)
