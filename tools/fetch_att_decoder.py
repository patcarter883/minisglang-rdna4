"""Fetch librocprof-trace-decoder.so, the one file that makes ATT/SQTT profiling work here.

WHY THIS EXISTS AS A FILE: it runs from a Dockerfile RUN, and a heredoc inside a backslash-continued
RUN does not parse ("unterminated heredoc") — which cost one full image build to discover. A script
that is COPYed and executed is both parseable and reviewable.

WHY IT MATTERS: hardware counters (`rocprofv3 --pmc`) HANG on gfx1201 here, but `rocprofv3 --att`
(SQTT thread trace) is a different mechanism that works — verified rc=0 in ~1s with real per-wave
traces. It had been failing for one reason only: this .so was absent, and rocprofv3 reports that as
"rocprof-trace-decoder library path not found". linex publishes the URL it expects, so we take the URL
from linex itself rather than hardcoding a version that will rot.
"""
from __future__ import annotations

import inspect
import pathlib
import re
import sys
import urllib.request

DEST = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else "/opt/rocprof-decoder/librocprof-trace-decoder.so")


def decoder_url() -> str:
    """Read the decoder URL out of linex rather than pinning our own copy of it."""
    from linex.api import Linex

    url = getattr(Linex, "DEFAULT_DECODER_URL", None)
    if isinstance(url, str) and url.startswith("http"):
        return url
    # Older/newer layouts keep it as a class-body literal; recover it from the source.
    m = re.search(r'DEFAULT_DECODER_URL\s*=\s*["\']([^"\']+)["\']', inspect.getsource(Linex))
    if not m:
        raise SystemExit("could not find DEFAULT_DECODER_URL in linex — has linex changed layout?")
    return m.group(1)


def main() -> int:
    url = decoder_url()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"[att-decoder] {url}\n[att-decoder] -> {DEST}", flush=True)
    urllib.request.urlretrieve(url, DEST)
    size = DEST.stat().st_size
    # A truncated or HTML-error-page download would otherwise surface much later as a profiling
    # failure, so fail the BUILD here instead.
    if size < 50_000:
        raise SystemExit(f"decoder is only {size} bytes — looks like an error page, not a library")
    print(f"[att-decoder] ok, {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
