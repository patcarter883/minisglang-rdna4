#!/usr/bin/env bash
# Build fp8_wmma from the g2sk KERNELS worktree INSIDE the serve image (a .so built in another image
# will not load — see CLAUDE.md). CPU only: no GPU lease.
set -euo pipefail
IMG="${G2_IMG:-minisgl-rdna4:post-tile}"
KERN="${G2_KERN_SRC:-/home/pat/code/rdna4-hip-kernels-g2sk}"
PKG="${1:-fp8_wmma}"
# hipify leftovers from a previous build silently shadow the real sources.
find "$KERN/$PKG" \( -name "*_hip.hip" -o -name "*_hip.h" -o -name "*_hip.cpp" \) -delete
docker run --rm --name g2build-$$ \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v "$KERN":/kern \
  --entrypoint bash "$IMG" -lc \
  "set -e; source /opt/venv/bin/activate 2>/dev/null || source /app/.venv/bin/activate; \
   cd /kern/$PKG && GPU_ARCHS=gfx1201 bash local/build_local.sh"
echo ">> built: $(ls -la "$KERN/$PKG"/torch-ext/$PKG/${PKG}_C*.so)"
