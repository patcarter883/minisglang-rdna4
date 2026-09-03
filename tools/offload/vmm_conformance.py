#!/usr/bin/env python3
"""Stable entry point for the HIP VMM remap conformance tripwire (probe P6).

The implementation lives in `p6_vmm_remap_conformance.py`; this name is the one
referenced by docs/WEIGHT_OFFLOAD_PLAN.md sec. 3 and is what CI should call:

    python3 tools/offload/vmm_conformance.py --expect broken

Exit 3 means the tripwire fired: hipMemUnmap+hipMemMap started behaving, and
dynamic page-level residency (plan tier T3) may now be reachable.

NEVER import this from engine code -- it is a bug repro, not a code path.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from p6_vmm_remap_conformance import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
