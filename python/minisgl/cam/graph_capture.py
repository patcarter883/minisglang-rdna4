"""CAM decode-tap graph capture (Phase 2).

Mirrors GDNGraphCapture: a static per-row bank/conf buffer threaded through the captured decode graph so
the L24 residual tap runs INSIDE the graph (the ~20% TPOT win), with NO product-key read ever captured
(the read is hoisted to prefill; see scheduler._prepare_cam). Row i of the buffer carries request i's
tap bank; a zero row is a tap no-op (padding / non-memory / seed-once-placed).

- prepare_for_capture: stage the (zero) buffer so the tap's per-row ops are recorded into the graph.
- prepare_for_replay: refresh the buffer IN PLACE from the batch's reqs (bank + seed-once state), zero
  the padded tail, then g.replay() re-runs the captured ops over it.
"""
from __future__ import annotations

import torch


class CAMGraphCapture:
    def __init__(self, cam, model_inner, device: torch.device, max_bs: int) -> None:
        self.cam = cam
        self.inner = model_inner
        self.bank_buf = torch.zeros(max_bs, cam.k_slots, cam.mem_dim, device=device, dtype=torch.float32)
        self.conf_buf = torch.zeros(max_bs, device=device, dtype=torch.float32)
        # Graph correctness invariant: unlike eager (which SKIPS the tap for non-memory reqs), the
        # captured graph always runs the tap ops, so a zero bank row MUST be a byte-exact no-op. With
        # bias-free to_k/to_v/to_o a zero bank => zero value => zero update; conf_gate/norm_gate keep
        # that zero. The `twosided` variant adds a `-s*c*h` suppression term that fires even at zero
        # bank, which WOULD perturb non-memory rows — it needs a per-row active mask before capture.
        if getattr(cam.tap, "twosided", False):
            import logging
            logging.getLogger(__name__).warning(
                "CAMGraphCapture: tap is twosided — a zero-bank row is NOT a pure no-op under graph "
                "capture; non-memory requests in a captured decode may be perturbed. Use --graph 0 or "
                "add a per-row active mask.")

    def prepare_for_capture(self, batch) -> None:
        # Zero buffer + tap ON via the static buffer: records the per-row tap ops into the graph. The
        # values are irrelevant at capture (only the op sequence + tensor addresses are); replay refreshes.
        self.inner.stage_cam_buf(self.cam, self.bank_buf, self.conf_buf, use_buf=True)

    def after_capture(self) -> None:
        # Revert the Python hook to the eager single-bank path; the captured graph already baked in the
        # buffer-path ops, so replay does not need _cam_use_buf. Prevents an eager prefill from wrongly
        # taking the buffer branch.
        self.inner.stage_cam_buf(self.cam, self.bank_buf, self.conf_buf, use_buf=False)

    def prepare_for_replay(self, batch) -> None:
        self.bank_buf.zero_()
        self.conf_buf.zero_()
        for i, req in enumerate(batch.padded_reqs):        # decode: one token per req, row i
            bank = getattr(req, "mem_bank", None)
            if bank is None or getattr(req, "_mem_placed", False):
                continue                                    # seed-once placed / non-memory -> zero -> no-op
            self.bank_buf[i].copy_(bank[0])                 # [1,K,mem] -> [K,mem]
            conf = getattr(req, "mem_conf", None)
            if conf is not None:
                self.conf_buf[i].copy_(conf.reshape(-1)[0])
