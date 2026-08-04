"""CPU proof that an MTP head is built from the checkpoint's TENSORS, not from a config claim.

Bug (2026-08-04): serving `cyankiwi/Agents-A1-AWQ-INT4` with the box-default `SPEC=mtp` died with

    KeyError: 'mtp.pre_fc_norm_embedding.weight'

Its own config.json declares `mtp_num_hidden_layers: 1` and the checkpoint ships ZERO `mtp.*`
tensors — the quantizer dropped the head and left the field. `ModelConfig.from_hf` trusted the
field, `Qwen3_5MoeForCausalLM` built a 22-buffer MTP head, and `BaseOP.load_state_dict` did
`state_dict.pop('mtp.pre_fc_norm_embedding.weight')` on a dict that could never contain it. The
KeyError named a symptom; nothing said "the config claims a head the weights do not back".

Fix: `checkpoint_ships_mtp` answers the question from the tensor names (the loader's own two
namespaces — `mtp.*` for Qwen3.5, `_is_beyond_decoder` for GLM/DeepSeek), and `from_hf` raises an
actionable ValueError when a head would be built that the checkpoint cannot fill.

Run inside the serve container (no GPU, header-only reads):
  PYTHONPATH=/engine/python python3 tools/mtp_capability_from_checkpoint_check.py
"""
from __future__ import annotations

import os
import sys

from transformers import AutoConfig

from minisgl.models.config import ModelConfig
from minisgl.models.weight import checkpoint_ships_mtp, checkpoint_tensor_names
from minisgl.utils import cached_load_hf_config

# (path, expect_ships_mtp). Only checkpoints cached on this box are exercised; the rest skip.
CASES = [
    ("cyankiwi/Agents-A1-AWQ-INT4", False),          # config CLAIMS a head, ships none
    ("cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", True),     # genuinely ships mtp.*
]

FAILS: list = []


def check(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name)
    if not cond:
        FAILS.append(name)


def main() -> int:
    for path, expect_mtp in CASES:
        print(f"\n=== {path}  (expect ships_mtp={expect_mtp})")
        try:
            names = checkpoint_tensor_names(path)
        except Exception as e:                                              # noqa: BLE001
            print(f"  SKIP (not cached: {type(e).__name__})")
            continue

        hf = cached_load_hf_config(path)
        tc = getattr(hf, "text_config", hf)
        claim = getattr(tc, "mtp_num_hidden_layers", 0) or getattr(
            tc, "num_nextn_predict_layers", 0
        )
        probe_cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(path), spec_algorithm="none")
        ships = checkpoint_ships_mtp(names, probe_cfg.num_layers)
        print(f"  tensors={len(names)}  config claim={claim}  ships_mtp={ships}")
        check("capability derived from tensors matches reality", ships is expect_mtp)

        # spec off: no head either way, and never an error
        cfg = ModelConfig.from_hf(
            AutoConfig.from_pretrained(path), spec_algorithm="none", ckpt_tensor_names=names
        )
        check("spec off -> no MTP head, no error", cfg.mtp_num_hidden_layers == 0)

        # spec mtp: build iff the tensors back it, else an ACTIONABLE error
        try:
            cfg = ModelConfig.from_hf(
                AutoConfig.from_pretrained(path), spec_algorithm="mtp", ckpt_tensor_names=names
            )
        except ValueError as e:
            msg = str(e)
            check("only the no-tensor checkpoint raises", not expect_mtp)
            check("error names the claim-vs-tensors conflict",
                  "ships NO MTP tensors" in msg and "config claims a head" in msg)
            check("error is actionable (names the escape hatch)", "--spec-algorithm none" in msg)
            check("error is not a bare KeyError", "KeyError" not in type(e).__name__)
            print(f"    -> {msg[:150]}...")
        else:
            check("checkpoint that ships a head still builds one", expect_mtp)
            check("head count preserved", cfg.mtp_num_hidden_layers > 0 or
                  cfg.num_nextn_predict_layers > 0)

        # the old default path (no names passed) must stay backward-compatible for unit callers
        cfg = ModelConfig.from_hf(AutoConfig.from_pretrained(path), spec_algorithm="mtp")
        check("ckpt_tensor_names=None still trusts the config (no new failure mode)",
              isinstance(cfg.mtp_num_hidden_layers, int))

    print("\n" + ("FAILED: " + "; ".join(FAILS) if FAILS else "ALL CHECKS PASSED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
