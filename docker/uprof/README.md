# AMD uProf package drop

The image's CPU-profiler stage installs AMD uProf from a `.deb` that AMD serves only behind a
EULA click-through, so it cannot be fetched at build time. `docker-compose.yml` injects this
directory as the `uprof` build context, and the stage skips cleanly (printing a NOTE) when no
`amduprof_*.deb` is here — that is why this placeholder exists at all: without an existing
directory, `docker compose build` fails with a misleading "pull access denied" as docker tries to
resolve `uprof` as a registry image.

To get CPU profiling in the image, download `amduprof_<version>_amd64.deb` and drop it here, or
point the build at wherever you keep it:

    UPROF_PKG_DIR=~/Downloads docker compose build

`.deb` files in this directory are gitignored.
