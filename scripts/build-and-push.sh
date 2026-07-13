#!/usr/bin/env bash
# Build the prebuilt, self-contained minisglang-rdna4 serving image and (optionally) push it to a
# registry. The engine source and every custom HIP kernel are baked in, so the resulting image runs
# with a single `docker compose up` and no source/kernel mounts.
#
# The custom HIP kernels live in a SEPARATE repo injected as a named build context (`kernels`).
# Building requires that repo on disk; end users never do — they just pull the pushed image.
#
# Usage:
#   scripts/build-and-push.sh            # build + tag :latest and :<git-sha>
#   PUSH=1 scripts/build-and-push.sh     # also `docker push` both tags (needs `docker login`)
#
# Env overrides:
#   IMAGE        registry/name          (default ghcr.io/patcarter883/minisglang-rdna4)
#   KERNELS_DIR  path to rdna4-hip-kernels checkout (default /home/pat/code/rdna4-hip-kernels)
#   PUSH         1 to push after building (default 0)
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/patcarter883/minisglang-rdna4}"
KERNELS_DIR="${KERNELS_DIR:-/home/pat/code/rdna4-hip-kernels}"
PUSH="${PUSH:-0}"

cd "$(dirname "$0")/.."
SHA="$(git rev-parse --short HEAD)"

if [[ ! -d "$KERNELS_DIR" ]]; then
  echo "ERROR: kernels repo not found at $KERNELS_DIR (set KERNELS_DIR=...)" >&2
  exit 1
fi

echo "==> Building $IMAGE:latest  and  $IMAGE:$SHA"
echo "    kernels build-context: $KERNELS_DIR"

# CPU-only build (hipcc cross-compiles for gfx1201; no GPU/lease needed).
docker build \
  -f Dockerfile \
  --build-context "kernels=${KERNELS_DIR}" \
  -t "${IMAGE}:latest" \
  -t "${IMAGE}:${SHA}" \
  .

echo "==> Built:"
docker images "${IMAGE}" --format '  {{.Repository}}:{{.Tag}}  {{.Size}}'

if [[ "$PUSH" == "1" ]]; then
  echo "==> Pushing (ensure you have run: docker login ghcr.io)"
  docker push "${IMAGE}:latest"
  docker push "${IMAGE}:${SHA}"
  echo "==> Pushed ${IMAGE}:latest and ${IMAGE}:${SHA}"
else
  echo "==> Skipping push (set PUSH=1 to push). To push manually:"
  echo "      echo \$GITHUB_TOKEN | docker login ghcr.io -u <user> --password-stdin"
  echo "      docker push ${IMAGE}:latest && docker push ${IMAGE}:${SHA}"
fi
