#!/usr/bin/env bash
# Local (no-Nix) build of gdn_hip_C for greening. Run from the package root (gdn/), inside the ROCm
# torch image (torch 2.10 / ROCm 7.x / hipcc for gfx1201). Produces:
#   torch-ext/gdn_hip/gdn_hip_C*.so   — the compiled extension
#   torch-ext/gdn_hip/_ops.py         — a shim providing `ops` + `add_op_namespace_prefix`, so the
#                                        SAME __init__.py/autograd.py used for the Hub build import here
# The Hub build (kernel-builder) generates its own _ops.py; this shim is git-ignored.
set -euo pipefail

ARCH="${GPU_ARCHS:-gfx1201}"
PKG_DIR="torch-ext/gdn_hip"

echo ">> building gdn_hip_C for ${ARCH}"
GPU_ARCHS="${ARCH}" python local/setup.py build_ext --inplace

# CUDAExtension --inplace drops gdn_hip_C*.so at the cwd; move it next to the python package.
SO="$(ls gdn_hip_C*.so 2>/dev/null | head -1 || true)"
if [ -z "${SO}" ]; then echo "ERROR: no gdn_hip_C*.so produced" >&2; exit 1; fi
mv -f "${SO}" "${PKG_DIR}/"
echo ">> placed ${PKG_DIR}/${SO}"

# Local _ops shim: the extension registers under torch.ops.gdn_hip_C (TORCH_EXTENSION_NAME=gdn_hip_C).
cat > "${PKG_DIR}/_ops.py" <<'PYEOF'
"""LOCAL shim for the no-Nix build (git-ignored). kernel-builder generates its own _ops.py with a
build-unique namespace; here the extension is plain gdn_hip_C, so alias to that."""
import glob
import os

import torch

_so = glob.glob(os.path.join(os.path.dirname(__file__), "gdn_hip_C*.so"))
if not _so:
    raise ImportError("gdn_hip_C not built; run local/build_local.sh from the package root")
torch.ops.load_library(_so[0])

ops = torch.ops.gdn_hip_C


def add_op_namespace_prefix(name: str) -> str:
    return f"gdn_hip_C::{name}"
PYEOF
echo ">> wrote ${PKG_DIR}/_ops.py (local shim)"
echo ">> done"
