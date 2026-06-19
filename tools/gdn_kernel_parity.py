"""Phase 3b-1 regression: vendored GDN kernels vs installed vLLM originals.

The kernel bodies under `minisgl.gdn.{fla,mamba}.ops` are byte-identical copies of
vLLM 0.22.69's; only their `vllm.*` imports were rewritten to `minisgl.gdn._compat`.
So the ONLY way vendoring could change numerics is if the compat shim steers a
different code path — specifically:
  * `current_platform.is_cuda_alike()` -> fla/ops/utils.py `device` -> shared-mem
    query -> `check_shared_mem()` -> autotune config in chunk_o (chunk_gated_delta_rule)
  * `num_compute_units()` -> fla/ops/layernorm_guard.py (RMSNormGated)

This harness feeds IDENTICAL inputs to the vendored and installed entry points and
asserts the outputs match bit-for-bit (or within fp tolerance). It does NOT check
that the GDN math is correct — that is 3b-3 (single-layer parity vs the real layer).

The AUTHORITATIVE faithfulness proof is the source byte-diff: all 18 vendored files
are identical to installed vLLM modulo the rewritten import lines. This numeric pass
corroborates that the compat shim doesn't perturb autotune/device selection.

`causal_conv1d_fn` is INFORMATIONAL only (not gated): invoked faithfully it needs a
full GDNAttentionMetadata object; with metadata=None it leaves a tail of its output
buffer unwritten, so it is nondeterministic against itself (M-vs-M) — a harness
invocation limitation, not a vendoring issue. It is covered by the byte-diff and is
exercised correctly through the real layer in 3b-3.

Run on GPU via the lease:
  gpu-lease.sh -n 1 -- docker run ... --entrypoint bash IMG -lc \
    'source /app/.venv/bin/activate && PYTHONPATH=/engine/python python /engine/tools/gdn_kernel_parity.py'
"""

from __future__ import annotations

import torch

import minisgl.gdn.fla.ops as M_fla
import minisgl.gdn.mamba.ops.causal_conv1d as M_conv
from vllm.model_executor.layers.fla import ops as V_fla
from vllm.model_executor.layers.mamba.ops import causal_conv1d as V_conv

DEV = "cuda"
results: list[tuple[str, bool, float, str]] = []


def _seeded(fn):
    """Build inputs under a fixed seed so both impls see identical tensors."""

    def make():
        torch.manual_seed(1234)
        return fn()

    return make


def _cmp(name: str, a, b, gated: bool = True) -> None:
    """Compare two (tuples of) tensors; record max abs diff + exact-match flag.

    gated=False marks the row informational (reported, excluded from PASS/FAIL).
    """
    if isinstance(a, torch.Tensor):
        a, b = (a,), (b,)
    a = [t for t in a if isinstance(t, torch.Tensor)]
    b = [t for t in b if isinstance(t, torch.Tensor)]
    md = 0.0
    exact = True
    for ta, tb in zip(a, b):
        d = (ta.float() - tb.float()).abs().max().item()
        md = max(md, d)
        exact = exact and torch.equal(ta, tb)
    note = "exact" if exact else "approx"
    if not gated:
        note += ", info-only"
    results.append((name, md < 1e-3, md, note, gated))


# ----- prefill core: chunk_gated_delta_rule (exercises is_cuda_alike path) -----
def test_chunk():
    Hk = Hv = 4
    Dk = Dv = 128
    T = 192  # > a couple of 64-chunks

    @_seeded
    def mk():
        return dict(
            q=torch.randn(1, T, Hk, Dk, device=DEV, dtype=torch.bfloat16),
            k=torch.randn(1, T, Hk, Dk, device=DEV, dtype=torch.bfloat16),
            v=torch.randn(1, T, Hv, Dv, device=DEV, dtype=torch.bfloat16),
            g=torch.randn(1, T, Hv, device=DEV, dtype=torch.float32),
            beta=torch.rand(1, T, Hv, device=DEV, dtype=torch.bfloat16),
            cu_seqlens=torch.tensor([0, T], device=DEV, dtype=torch.int32),
            output_final_state=True,
        )

    om, sm = M_fla.chunk_gated_delta_rule(**mk())
    ov, sv = V_fla.chunk_gated_delta_rule(**mk())
    _cmp("chunk_gated_delta_rule.o", om, ov)
    _cmp("chunk_gated_delta_rule.state", sm, sv)


# ----- RMSNormGated (exercises num_compute_units path) -----
def test_rmsnorm_gated():
    D = 128
    N = 256
    torch.manual_seed(7)
    mn = M_fla.RMSNormGated(D, eps=1e-5, norm_before_gate=True, activation="silu").to(DEV)
    vn = V_fla.RMSNormGated(D, eps=1e-5, norm_before_gate=True, activation="silu").to(DEV)
    vn.load_state_dict(mn.state_dict())
    torch.manual_seed(7)
    x = torch.randn(N, D, device=DEV, dtype=torch.bfloat16)
    z = torch.randn(N, D, device=DEV, dtype=torch.bfloat16)
    _cmp("RMSNormGated", mn(x.clone(), z.clone()), vn(x.clone(), z.clone()))


# ----- fused_post_conv_prep -----
def test_post_conv_prep():
    Hk = 4
    Hv = 4
    Dk = Dv = 128
    L = 192
    qkv_dim = Hk * Dk * 2 + Hv * Dv

    @_seeded
    def mk():
        return dict(
            conv_output=torch.randn(L, qkv_dim, device=DEV, dtype=torch.bfloat16),
            a=torch.randn(L, Hv, device=DEV, dtype=torch.float32),
            b=torch.randn(L, Hv, device=DEV, dtype=torch.float32),
            A_log=torch.randn(Hv, device=DEV, dtype=torch.float32),
            dt_bias=torch.randn(Hv, device=DEV, dtype=torch.float32),
            num_k_heads=Hk,
            head_k_dim=Dk,
            head_v_dim=Dv,
        )

    om = M_fla.fused_post_conv_prep(**mk())
    ov = V_fla.fused_post_conv_prep(**mk())
    _cmp("fused_post_conv_prep", om, ov)


# ----- causal_conv1d_fn (prefill conv) -----
def test_causal_conv1d_fn():
    conv_dim = 4 * 128 * 2 + 4 * 128
    kernel = 4
    T = 192

    @_seeded
    def mk():
        return dict(
            x=torch.randn(conv_dim, T, device=DEV, dtype=torch.bfloat16),
            weight=torch.randn(conv_dim, kernel, device=DEV, dtype=torch.bfloat16),
            bias=None,
            conv_states=torch.zeros(2, conv_dim, kernel - 1, device=DEV, dtype=torch.bfloat16),
            query_start_loc=torch.tensor([0, T], device=DEV, dtype=torch.int32),
            cache_indices=torch.tensor([0], device=DEV, dtype=torch.int32),
            has_initial_state=torch.tensor([False], device=DEV),
            activation="silu",
        )

    # causal_conv1d_fn is an in-place kernel under @triton.autotune: the first
    # (cold) call runs candidate configs on the live input buffer, mutating it.
    # Warm BOTH autotune caches on throwaway inputs so the measured calls each run
    # once-clean on their own freshly-seeded inputs.
    M_conv.causal_conv1d_fn(**mk())
    V_conv.causal_conv1d_fn(**mk())
    om = M_conv.causal_conv1d_fn(**mk())
    om2 = M_conv.causal_conv1d_fn(**mk())  # self-consistency: M vs M on identical input
    ov = V_conv.causal_conv1d_fn(**mk())
    _cmp("causal_conv1d_fn[M-vs-M]", om, om2, gated=False)
    _cmp("causal_conv1d_fn", om, ov, gated=False)


# ----- causal_conv1d_update (decode conv) -----
def test_causal_conv1d_update():
    conv_dim = 4 * 128 * 2 + 4 * 128
    kernel = 4
    B = 3

    @_seeded
    def mk():
        return dict(
            x=torch.randn(B, conv_dim, device=DEV, dtype=torch.bfloat16),
            conv_state=torch.randn(B, conv_dim, kernel - 1, device=DEV, dtype=torch.bfloat16),
            weight=torch.randn(conv_dim, kernel, device=DEV, dtype=torch.bfloat16),
            bias=None,
            activation="silu",
            conv_state_indices=torch.arange(B, device=DEV, dtype=torch.int32),
        )

    om = M_conv.causal_conv1d_update(**mk())
    ov = V_conv.causal_conv1d_update(**mk())
    _cmp("causal_conv1d_update", om, ov)


def main():
    print(f"torch {torch.__version__}  hip={torch.version.hip}  cuda_avail={torch.cuda.is_available()}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    for t in (
        test_chunk,
        test_rmsnorm_gated,
        test_post_conv_prep,
        test_causal_conv1d_fn,
        test_causal_conv1d_update,
    ):
        try:
            t()
        except Exception as e:
            import traceback

            traceback.print_exc()
            results.append((t.__name__, False, float("nan"), f"ERROR: {type(e).__name__}: {e}", True))

    print("\n=== vendored vs installed-vLLM GDN kernel parity ===")
    all_ok = True
    for name, ok, md, note, gated in results:
        if gated:
            all_ok = all_ok and ok
        tag = ("PASS" if ok else "FAIL") if gated else "INFO"
        print(f"  [{tag}] {name:34s} max|Δ|={md:.3e}  ({note})")
    print("\nRESULT:", "PASS — vendoring is numerically faithful" if all_ok else "FAIL — investigate")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
