"""Isolate the GDN spec-verify kernel cost at the REAL verify shape (N=1, qlen=K+1).

The serve profile showed the GDN mixer costs ~4.7ms/layer at decode but ~19ms in spec-verify (a 4x
fixed jump, ~independent of K). Verify uses `gdn_prefill_verify` (recurrent scan + per-token FULL
conv/ssm state capture into scratch). This bench attributes that jump: it times, at the verify
geometry, per single-layer call (x48 layers in the model):

  * gdn_decode           — the O(1) decode recurrence (baseline; what a 1-token step pays)
  * gdn_prefill_verify   — the verify kernel WITH per-token state capture
  * gdn_prefill          — the same recurrence WITHOUT capture (isolates the scratch-write cost)
  * gdn_prefill_chunked  — the chunked variant (if it were bit-stable enough for verify)

If prefill_verify >> prefill, the per-token state capture is the target. If they're close, the cost
is the recurrence/occupancy at N=1 and the fix is elsewhere. x48 layers → multiply the per-call ms.

Run under a 1-card lease:
  gpu-lease -n 1 -- bash -c 'docker run ... python /engine/tools/gdn_verify_microbench.py'
"""
from __future__ import annotations

import time
import torch
import gdn_hip as gdn  # module-level callables (ops namespace is prefixed), used exactly like layer.py

DEV = "cuda"
torch.manual_seed(0)
# Real 35B GDN geometry (matches gdn_hip_bench): H heads (q/k), HV value heads, K/V head dims.
H, HV, K, V = 16, 32, 128, 128
SCALE = K ** -0.5
N = 1            # single sequence (bs=1 decode/verify)
QLENS = [1, 5, 9]  # decode(=1), K=4 verify(=5), K=8 verify(=9)
NLAYERS = 48
ITERS = 50


def _mk(T: int):
    return dict(
        q=torch.randn(T, H, K, device=DEV), k=torch.randn(T, H, K, device=DEV),
        v=torch.randn(T, HV, V, device=DEV), a=torch.randn(T, HV, device=DEV),
        b=torch.randn(T, HV, device=DEV),
        A_log=torch.randn(HV, device=DEV) * 0.5 - 2.0, dt_bias=torch.randn(HV, device=DEV),
        cu=torch.tensor([0, T], dtype=torch.int32, device=DEV),
        idx=torch.tensor([1], dtype=torch.long, device=DEV),
        has_init=torch.tensor([1], dtype=torch.uint8, device=DEV),
    )


def _bench(fn, iters=ITERS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms/call


def main():
    assert torch.cuda.is_available()
    dev = torch.cuda.get_device_name()
    print(f"=== GDN verify-shape microbench (N={N}, {dev}) — ms per single-layer call, x{NLAYERS} layers ===")
    print(f"{'qlen':>5} | {'decode':>8} | {'verify':>8} | {'prefill':>8} | {'chunked':>8} | "
          f"{'vfy/dec':>8} | {'vfy/pfl':>8} | {'verify x48':>10}")
    print("-" * 92)
    for T in QLENS:
        g = _mk(T)
        st = lambda: torch.zeros(2, HV, V, K, device=DEV)  # noqa: E731

        # decode is a 1-token op; for qlen>1 emulate the recurrence as T sequential decode calls.
        def run_decode():
            s = st()
            for t in range(T):
                gdn.gdn_decode(g["q"][t:t+1], g["k"][t:t+1], g["v"][t:t+1], g["a"][t:t+1], g["b"][t:t+1],
                               g["A_log"], g["dt_bias"], s, g["idx"], SCALE, 1)

        def run_verify():
            gdn.gdn_prefill_verify(g["q"], g["k"], g["v"], g["a"], g["b"], g["A_log"], g["dt_bias"],
                                   g["cu"], g["idx"], g["has_init"], st(), int(T), SCALE, 1)

        def run_prefill():
            gdn.gdn_prefill(g["q"], g["k"], g["v"], g["a"], g["b"], g["A_log"], g["dt_bias"],
                            g["cu"], g["idx"], g["has_init"], st(), SCALE, 1)

        def run_chunked():
            gdn.gdn_prefill_chunked(g["q"], g["k"], g["v"], g["a"], g["b"], g["A_log"], g["dt_bias"],
                                    g["cu"], g["idx"], g["has_init"], st(), SCALE, 1)

        t_dec = _bench(run_decode)
        t_vfy = _bench(run_verify)
        t_pfl = _bench(run_prefill)
        try:
            t_chk = _bench(run_chunked)
        except Exception:
            t_chk = float("nan")
        print(f"{T:>5} | {t_dec:>8.3f} | {t_vfy:>8.3f} | {t_pfl:>8.3f} | {t_chk:>8.3f} | "
              f"{t_vfy/max(t_dec,1e-6):>7.2f}x | {t_vfy/max(t_pfl,1e-6):>7.2f}x | {t_vfy*NLAYERS:>9.2f}ms")
    print("\nvfy/pfl >> 1 => per-token state CAPTURE dominates (target it). ~1 => recurrence/occupancy is the cost.")


if __name__ == "__main__":
    main()
