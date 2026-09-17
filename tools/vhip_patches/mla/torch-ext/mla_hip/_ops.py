"""LOCAL shim for the no-Nix build (git-ignored). kernel-builder generates its own _ops.py with a
build-unique namespace; here the extension is plain mla_hip_C, so alias to that."""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "mla_hip_C*.so"))
if not _so:
    raise ImportError("mla_hip_C not built; run local/build_local.sh from the package root")
torch.ops.load_library(_so[0])

ops = torch.ops.mla_hip_C


def add_op_namespace_prefix(name: str) -> str:
    return f"mla_hip_C::{name}"
