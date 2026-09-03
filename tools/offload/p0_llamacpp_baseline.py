#!/usr/bin/env python3
"""
PROBE P0 -- llama.cpp baseline on the target checkpoint.

WEIGHT_OFFLOAD_PLAN.md Section 3, row P0:

    "llama.cpp baseline. Same target checkpoint, same box, tok/s at bs=1 and at
     CONC=6 aggregate.  Sets the real bar.  ... State it now or the whole plan is
     measured against zero."

This probe is the *denominator* of the whole weight-offload project.  Gate K4
("M1 tok/s < 0.64 x llama.cpp -> hard kill") and gate A0.4 ("byte hit rate >=
max(50%, break-even vs P0's llama.cpp number)") both read the number this script
produces.  A wrong number here silently retunes two hard kills, so this script:

  * measures both legs itself -- it never reprints a figure it did not measure;
  * carries the previously-reported 4.58 tok/s bs=1 figure ONLY as a clearly
    labelled `cited_prior` block with `reproduced_by_this_run` computed against
    the fresh measurement;
  * records full box state (meminfo, vmstat swap/major-fault counters, ZFS ARC,
    nvme diskstats, per-card VRAM) BEFORE and AFTER every leg, because on this
    box the checkpoint (93.7 GB) is larger than RAM (91.8 GB, ~41 GB already in
    use) and the baseline is storage/ARC-bound, not DDR-bound.  A capacity
    number without the box state it was taken on is worthless;
  * discards a warm-up repetition, reports median + spread over >= 5 reps, AND
    checks that those reps are a steady state rather than a ramp (one 100-token
    warm-up cannot warm a 93.7 GB working set, so a trending series would make
    the median a point on a curve);
  * names WHICH physical cards the timing came from (card0 = RX 9070 XT,
    card1 = RX 9070 -- they are not identical parts), classifying the Ryzen iGPU
    out by NAME as well as by index;
  * ATTRIBUTES the run to the configuration it claims to measure: it parses the
    server log for the device enumeration and the per-device weight buffers, and
    independently checks that VRAM actually moved on the discrete cards across
    model load.  A plausible tok/s number cannot distinguish "2 cards, --fit
    auto-offload" from a CPU-only fallback or from a run where the iGPU's 47 GB
    GTT pool leaked into `--fit` sizing -- and either would silently retune K4
    and A0.4.  Assert on the operation, never on the flag having been passed;
  * compares concurrency against bs=1 on MATCHED bases only (wall-vs-wall and
    decode-vs-decode); the naive `CONC wall / bs=1 decode` ratio mixes a
    prefill-inclusive aggregate with a prefill-exclusive rate and is recorded
    under `MISMATCHED_do_not_cite`.

It also settles, from measured values only, whether the plan's *derived*
"33.8 tok/s CPU-expert ceiling" (Section 1, marked "derived -- must be
measured") is reachable in practice, and recomputes the break-even hit rate `h`
against the number actually measured instead of against 45 GB/s of DDR.

--------------------------------------------------------------------------- ---
USAGE

  # what the Run phase should execute (both legs, 5 reps each, ~25-45 min):
  python3 tools/offload/p0_llamacpp_baseline.py

  # cheap, GPU-free validation of argument handling + JSON shape:
  python3 tools/offload/p0_llamacpp_baseline.py --selftest

  # print the resolved llama-server argv and preconditions, run nothing:
  python3 tools/offload/p0_llamacpp_baseline.py --dry-run

Outputs (durable, in the worktree -- never /tmp):
  docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p0.json
  docs/measurements/WEIGHT_OFFLOAD_2026-09-02/P0_LLAMACPP_BASELINE.md
  docs/measurements/WEIGHT_OFFLOAD_2026-09-02/p0_server_<leg>.log

--------------------------------------------------------------------------- ---
EXIT CODES

  0  ok
  2  precondition failed (nothing launched, nothing written)
  3  llama-server died during load or missed --ready-timeout
  4  unhandled error during measurement (nothing written)
  5  a leg produced ZERO valid repetitions
  6  the assembled document failed schema validation.  The document is written
     to `p0.INVALID.json` and the real artifacts (`p0.json`, the .md) are NOT
     written or overwritten -- a non-conforming document must never be readable
     under the name a downstream gate globs for.
  7  degraded: fewer valid reps than requested, --reps < 5 forced, ran over a
     busy card, a leg failed the steady-state drift check, or GPU offload could
     not be positively confirmed from the server log AND the VRAM deltas.
  8  the run cannot be attributed to the configuration this probe claims to
     measure: the server positively ran CPU-only (zero GPU weight bytes in the
     log AND no VRAM movement on either compute card), or the iGPU leaked into
     device enumeration.  A CPU-only number quoted as "2 cards, --fit
     auto-offload" would silently retune K4 and A0.4.  Artifacts are written
     under `p0.INVALID.json` only.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Device fencing MUST happen before anything can enumerate ROCm devices.
# ROCm device 2 on this box is the Ryzen 7800X3D iGPU advertising ~47 GB of GTT.
# If it enters enumeration it poisons llama.cpp's --fit auto-sizing (it looks
# like the biggest free pool on the machine) and the run is meaningless.
# --------------------------------------------------------------------------- #
import os

os.environ["ROCR_VISIBLE_DEVICES"] = "0,1"
os.environ.pop("HIP_VISIBLE_DEVICES", None)

import argparse
import json
import re
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Fixed, box-specific paths.  All are asserted to exist before anything runs.
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO / "docs" / "measurements" / "WEIGHT_OFFLOAD_2026-09-02"

LLAMACPP_DIR = Path("/home/pat/.cache/lemonade/bin/llamacpp/rocm-nightly")
LLAMA_SERVER = LLAMACPP_DIR / "llama-server"
LLAMACPP_VERSION_FILE = LLAMACPP_DIR / "version.txt"

MODEL_DIR = Path(
    "/home/pat/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/"
    "snapshots/5d16c055a7c5cb276e721ee154f9c22420dde2a1/UD-IQ4_XS"
)
MODEL_SHARD0 = MODEL_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
MODEL_SHARDS = [
    MODEL_DIR / f"Qwen3.8-Flash-Next-UD-IQ4_XS-0000{i}-of-00003.gguf" for i in (1, 2, 3)
]

# The two discrete gfx1201 compute cards.  ROCm device 2 is the iGPU and is
# never a compute target -- see the ROCR fence at the top of this file.
PHYSICAL_CARDS = {
    0: "AMD Radeon RX 9070 XT (16 GB)",
    1: "AMD Radeon RX 9070 (16 GB)",
}

# --------------------------------------------------------------------------- #
# The figure this probe must either reproduce or contradict.  It is carried as
# a CITATION, never as a result: `cited_prior.value_tok_s` is copied into the
# JSON verbatim and `reproduced_by_this_run` is computed from the fresh number.
# --------------------------------------------------------------------------- #
CITED_PRIOR = {
    "metric": "decode tok/s at bs=1",
    "value_tok_s": 4.58,
    "prompt_tok_s": 11.1,
    "source": (
        "session note recorded in the weight-offload Phase-0 brief, 2026-09-02: "
        "'llama.cpp on the target checkpoint on THIS box is 4.58 tok/s at bs=1 "
        "(decode), prompt 11.1 t/s, measured this session via lemonade + "
        "llama.cpp rocm-nightly, 2 cards, auto-fit offload.'"
    ),
    "provenance_quality": (
        "NOT a durable fixture: no log, no JSON, no recorded box state, no rep "
        "count, no spread, no record of which cards or which llama-server argv. "
        "This probe exists to replace it with one."
    ),
    "measured_by_this_script": False,
}

# The plan's DERIVED (not measured) CPU-expert ceiling, Section 1.
PLAN_DERIVED = {
    "cpu_expert_ceiling_tok_s": 33.8,
    "basis": "host DDR5 aggregate read ~45 GB/s / 1.33 GB active expert bytes per token",
    "status_in_plan": "derived -- must be measured",
    "host_ddr_read_gbs": 45.0,
    "active_expert_bytes_per_token_gb": 1.33,
    "pcie_h2d_granule_gbs": 26.8,
}

# 4: adds top-level `gpu_engagement` (attribution of the run to the 2-card --fit
#    configuration) and `backend_from_server_log`; `summarize().values` is now
#    CHRONOLOGICAL with a `drift` block; `derived.conc_scaling` replaces the
#    mismatched `conc_scaling_factor`; io_attribution keys renamed to carry their
#    scope (`_boxwide` vs `server_`) and to stop dividing by the plan's
#    cross-packing 1.33 GB/token constant.
# 5: the same-packing expert-bytes denominator now comes from an EXACT walk of
#    the GGUF tensor table (`gguf_expert_geometry`), validated against the
#    on-disk shard sizes, instead of the server log.  Run 1 (2026-09-02) showed
#    this build emits no `print_info:` / `load_tensors:` lines at all, so the
#    log-derived denominator was null and the primary I/O-attribution ratio
#    could not be formed.  `expert_bytes_per_token` gains `gguf_geometry`,
#    `estimate_source` and `pro_rata_upper_bound_gb`.
SCHEMA_VERSION = 5

DEFAULT_PORT = 18099
MIN_REPS = 5


# --------------------------------------------------------------------------- #
# Loud failure
# --------------------------------------------------------------------------- #
class PreconditionError(RuntimeError):
    pass


def die(msg: str, code: int = 2) -> "None":
    print(f"\n*** P0 PRECONDITION FAILED ***\n{msg}\n", file=sys.stderr, flush=True)
    sys.exit(code)


def log(msg: str) -> None:
    print(f"[p0 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Box state -- captured before/after every leg.  The brief is explicit: "A
# capacity probe run on a busy box is a misleading probe -- record the state or
# the number is worthless."
# --------------------------------------------------------------------------- #
def _read_meminfo() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            parts = v.split()
            if parts:
                try:
                    out[k.strip()] = int(parts[0])  # kB
                except ValueError:
                    pass
    except OSError as e:
        return {"_error": str(e)}
    keep = (
        "MemTotal",
        "MemFree",
        "MemAvailable",
        "Buffers",
        "Cached",
        "SwapTotal",
        "SwapFree",
        "Dirty",
        "Writeback",
        "Mlocked",
        "Shmem",
    )
    kb = {k: out.get(k) for k in keep if k in out}
    gb = {k + "_gb": round(v / 1024.0 / 1024.0, 2) for k, v in kb.items() if v is not None}
    return {"kb": kb, "gb": gb}


def _read_vmstat() -> dict:
    keep = (
        "pswpin",
        "pswpout",
        "pgmajfault",
        "pgpgin",
        "pgpgout",
        "pgscan_direct",
        "pgsteal_direct",
        "nr_free_pages",
    )
    out = {}
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            k, _, v = line.partition(" ")
            if k in keep:
                out[k] = int(v)
    except OSError as e:
        return {"_error": str(e)}
    return out


def _read_arcstats() -> dict:
    """ZFS ARC.  /home is ZFS, so llama.cpp's mmap of a 93.7 GB GGUF reads
    through a 16 GiB-capped ARC.  This is the likely dominant term in the
    baseline and must be on the record."""
    p = Path("/proc/spl/kstat/zfs/arcstats")
    if not p.exists():
        return {"_absent": True}
    keep = (
        "size",
        "c",
        "c_max",
        "hits",
        "misses",
        "demand_data_hits",
        "demand_data_misses",
        "mfu_hits",
        "mru_hits",
    )
    out = {}
    try:
        for line in p.read_text().splitlines()[2:]:
            f = line.split()
            if len(f) >= 3 and f[0] in keep:
                out[f[0]] = int(f[2])
    except (OSError, ValueError) as e:
        return {"_error": str(e)}
    return out


def _read_diskstats() -> dict:
    """Per-nvme sectors read/written.  Sector = 512 B."""
    out = {}
    try:
        for line in Path("/proc/diskstats").read_text().splitlines():
            f = line.split()
            if len(f) < 10:
                continue
            name = f[2]
            if re.fullmatch(r"nvme\d+n\d+", name):
                out[name] = {
                    "reads_completed": int(f[3]),
                    "sectors_read": int(f[5]),
                    "writes_completed": int(f[7]),
                    "sectors_written": int(f[9]),
                }
    except (OSError, ValueError) as e:
        return {"_error": str(e)}
    return out


def _read_loadavg() -> dict:
    try:
        f = Path("/proc/loadavg").read_text().split()
        return {"1m": float(f[0]), "5m": float(f[1]), "15m": float(f[2]), "procs": f[3]}
    except (OSError, ValueError, IndexError) as e:
        return {"_error": str(e)}


# The iGPU is ROCm device 2 (`card2`) on this box and reports ~1.1 GB of its
# 47 GB GTT pool as "VRAM used" at idle.  It is NEVER a compute target.  Match it
# by NAME as well as by index: keying on the literal string "card2" alone means a
# renumbering silently promotes the iGPU to a compute card, which both defeats the
# busy-card gate (1.1 GB idle > the 1 GB threshold) and poisons `--fit`.
_IGPU_NAME_RE = re.compile(r"Ryzen|Raphael|Granite Ridge|\bAPU\b|Processor$", re.I)
_DISCRETE_NAME_RE = re.compile(r"Radeon\s+RX", re.I)


def _is_compute_card(dev: str, name: str) -> bool:
    if _IGPU_NAME_RE.search(name or ""):
        return False
    if dev == "card2":
        return False
    return bool(_DISCRETE_NAME_RE.search(name or ""))


def rocm_smi_cards() -> dict:
    """Per-card identity + VRAM used, straight from rocm-smi --csv.

    Deliberately NOT called in --selftest/--dry-run so those modes are provably
    GPU-free.

    The header row is located by CONTENT, not by position: rocm-smi on this box
    prints a "low-power state" warning, and any build that routes such a preamble
    to stdout would make a positional parse take the warning as the header, match
    zero data rows, and return an EMPTY dict.  That failure is silent and lethal:
    an empty dict makes the busy-card precondition vacuously pass and erases the
    per-card identity the repo rule requires on every timing.
    """
    exe = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
    if not Path(exe).exists():
        return {"_error": f"rocm-smi not found at {exe}"}
    try:
        r = subprocess.run(
            [exe, "--showid", "--showmeminfo", "vram", "--csv"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"_error": str(e)}
    if r.returncode != 0:
        return {"_error": f"rocm-smi rc={r.returncode}: {r.stderr[-400:]}"}
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    hdr_idx = next(
        (i for i, l in enumerate(lines) if l.split(",")[0].strip() == "device" and "," in l),
        None,
    )
    if hdr_idx is None:
        return {
            "_error": "rocm-smi produced no CSV header row starting with 'device'; "
            f"first lines: {lines[:3]}"
        }
    hdr = [h.strip() for h in lines[hdr_idx].split(",")]
    cards = {}
    for line in lines[hdr_idx + 1 :]:
        f = [x.strip() for x in line.split(",")]
        if len(f) != len(hdr):
            continue
        row = dict(zip(hdr, f))
        dev = row.get("device", "?")
        rec = {"raw": row}
        for key in hdr:
            if "Used" in key:
                try:
                    rec["vram_used_bytes"] = int(row[key])
                    rec["vram_used_gb"] = round(int(row[key]) / 1e9, 3)
                except ValueError:
                    pass
            if "Total Memory" in key and "Used" not in key:
                try:
                    rec["vram_total_gb"] = round(int(row[key]) / 1e9, 3)
                except ValueError:
                    pass
        rec["name"] = row.get("Device Name", "?")
        rec["is_compute_card"] = _is_compute_card(dev, rec["name"])
        cards[dev] = rec
    if not cards:
        return {"_error": f"rocm-smi CSV had a header but no parsable rows: {lines[:4]}"}
    if not any(v.get("is_compute_card") for v in cards.values()):
        return {
            "_error": "rocm-smi listed no discrete compute card (all rows classified "
            f"as iGPU/unknown): {[(k, v.get('name')) for k, v in cards.items()]}"
        }
    return cards


def compute_card_vram(state: dict) -> dict:
    """{card: vram_used_gb} for the DISCRETE cards only, out of a box_state."""
    out = {}
    for dev, rec in (state.get("rocm_smi") or {}).items():
        if not isinstance(rec, dict) or not rec.get("is_compute_card"):
            continue
        v = rec.get("vram_used_gb")
        if v is not None:
            out[dev] = v
    return out


def vram_delta_gb(before: dict, after: dict) -> dict:
    b, a = compute_card_vram(before), compute_card_vram(after)
    return {k: round(a[k] - b[k], 3) for k in a if k in b}


def box_state(with_gpu: bool) -> dict:
    st = {
        "t_wall": time.time(),
        "t_iso": datetime.now(timezone.utc).isoformat(),
        "meminfo": _read_meminfo(),
        "vmstat": _read_vmstat(),
        "zfs_arcstats": _read_arcstats(),
        "diskstats_nvme": _read_diskstats(),
        "loadavg": _read_loadavg(),
    }
    st["rocm_smi"] = rocm_smi_cards() if with_gpu else {"_skipped": "gpu-free mode"}
    return st


def read_proc_io(pid) -> dict:
    """Per-PROCESS storage counters for llama-server.

    /proc/diskstats is box-wide: any other job's reads land in the same delta and
    would be charged to this leg's tokens.  `read_bytes` from /proc/<pid>/io is
    the actual block-layer read attributable to this server, which is what the
    compute-bound-vs-storage-bound discriminator needs.
    """
    if pid is None:
        return {"_absent": "no server pid"}
    keep = ("rchar", "wchar", "syscr", "read_bytes", "write_bytes", "cancelled_write_bytes")
    out = {}
    try:
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            k, _, v = line.partition(":")
            if k.strip() in keep:
                out[k.strip()] = int(v)
    except (OSError, ValueError) as e:
        return {"_error": str(e)}
    return out


def proc_io_delta(before: dict, after: dict) -> dict:
    d = {}
    for k in ("rchar", "read_bytes", "syscr"):
        b, a = before.get(k), after.get(k)
        if isinstance(b, int) and isinstance(a, int):
            d[f"d_{k}"] = a - b
    if "d_read_bytes" in d:
        d["d_read_gb"] = round(d["d_read_bytes"] / 1e9, 3)
    return d


def annotate_leg_io(
    leg: dict,
    delta: dict,
    expert_bytes: dict | None = None,
    pio: dict | None = None,
) -> None:
    """Attribute the leg's storage traffic to the tokens it produced.

    This is the discriminator between the two competing explanations of a low
    llama.cpp number: CPU-compute bound (i-quant expert GEMV on 8 cores) vs
    storage bound (a 93.7 GB checkpoint that does not fit in 91.8 GB of RAM,
    mmap'd through a 16 GiB ARC).  It is MEASURED, not assumed: bytes actually
    read during the leg, divided by tokens actually generated.

    Two denominators are reported and both are labelled, because the plan's
    1.33 GB/token constant is arithmetic on the minisgl target shape and NOT on
    this IQ4_XS GGUF.  A single ratio against the plan constant would be a
    confident cross-packing comparison.
    """
    if leg.get("kind", "").startswith("conc"):
        toks = sum(
            s.get("predicted_n") or 0
            for b in leg.get("reps", [])
            for s in b.get("streams", [])
            if s.get("ok")
        )
        toks += sum(
            s.get("predicted_n") or 0
            for s in (leg.get("warmup_discarded") or {}).get("streams", [])
            if s.get("ok")
        )
    else:
        toks = sum(r.get("predicted_n") or 0 for r in leg.get("reps", []) if r.get("ok"))
        w = leg.get("warmup_discarded") or {}
        if w.get("ok"):
            toks += w.get("predicted_n") or 0
    io = {
        "tokens_generated_incl_warmup": toks,
        "_scope": (
            "nvme_* are BOX-WIDE /proc/diskstats deltas and include any other process's "
            "reads in the same window; server_* are process-attributed /proc/<pid>/io."
        ),
    }
    per_tok = None
    rd = delta.get("d_nvme_read_gb")
    if rd is not None and toks:
        per_tok = rd / toks
        io["nvme_read_gb_per_generated_token_boxwide"] = round(per_tok, 4)
    if pio and pio.get("d_read_gb") is not None and toks:
        io["server_read_gb_per_generated_token"] = round(pio["d_read_gb"] / toks, 4)
        io["server_read_gb_total"] = pio["d_read_gb"]
        per_tok = pio["d_read_gb"] / toks  # prefer the attributed figure
        io["_denominator_source"] = "process-attributed /proc/<pid>/io read_bytes"
    elif per_tok is not None:
        io["_denominator_source"] = "box-wide /proc/diskstats (NO process attribution)"
    if per_tok is not None:
        eb = expert_bytes or {}
        est = eb.get("estimate_gb")
        if est:
            io["fraction_of_this_checkpoints_expert_bytes_from_storage"] = round(per_tok / est, 6)
            io["this_checkpoint_expert_bytes_per_token_gb"] = est
            io["this_checkpoint_expert_bytes_basis"] = eb.get("estimate_basis")
        else:
            io["fraction_of_this_checkpoints_expert_bytes_from_storage"] = None
            io["this_checkpoint_expert_bytes_note"] = (
                "checkpoint geometry not parseable; no same-packing denominator available"
            )
        io["fraction_of_PLAN_constant_1p33GB_NOT_THIS_PACKING"] = round(
            per_tok / PLAN_DERIVED["active_expert_bytes_per_token_gb"], 4
        )
        io["plan_constant_caveat"] = eb.get(
            "plan_constant_basis",
            "1.33 GB/token is the minisgl target shape, not this GGUF; contrast only",
        )
    mf = delta.get("d_pgmajfault")
    if mf is not None and toks:
        io["major_faults_per_generated_token_boxwide"] = round(mf / toks, 2)
    leg["io_attribution"] = io


def box_state_delta(before: dict, after: dict) -> dict:
    """The interesting part: how much the run swapped / major-faulted / read."""
    d = {}
    for k in ("pswpin", "pswpout", "pgmajfault", "pgpgin", "pgpgout"):
        b, a = before.get("vmstat", {}).get(k), after.get("vmstat", {}).get(k)
        if isinstance(b, int) and isinstance(a, int):
            d[f"d_{k}"] = a - b
    for k in ("hits", "misses", "demand_data_hits", "demand_data_misses"):
        b = before.get("zfs_arcstats", {}).get(k)
        a = after.get("zfs_arcstats", {}).get(k)
        if isinstance(b, int) and isinstance(a, int):
            d[f"d_arc_{k}"] = a - b
    rd = 0
    have_disk = False
    for dev, arec in (after.get("diskstats_nvme") or {}).items():
        brec = (before.get("diskstats_nvme") or {}).get(dev)
        if isinstance(brec, dict) and isinstance(arec, dict):
            rd += arec["sectors_read"] - brec["sectors_read"]
            have_disk = True
    if have_disk:
        d["d_nvme_read_gb"] = round(rd * 512 / 1e9, 3)
    bm = before.get("meminfo", {}).get("gb", {})
    am = after.get("meminfo", {}).get("gb", {})
    for k in ("MemAvailable_gb", "MemFree_gb", "SwapFree_gb", "Cached_gb"):
        if k in bm and k in am:
            d[f"d_{k}"] = round(am[k] - bm[k], 2)
    dt = after.get("t_wall", 0) - before.get("t_wall", 0)
    d["window_s"] = round(dt, 2)
    if have_disk and dt > 0:
        d["nvme_read_gbs_avg"] = round(d["d_nvme_read_gb"] / dt, 3)
    return d


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def git_provenance() -> dict:
    def g(*args):
        try:
            r = subprocess.run(
                ["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=30
            )
            return r.stdout.strip() if r.returncode == 0 else f"<rc={r.returncode}>"
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"<{e}>"

    return {
        "worktree": str(REPO),
        "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
        "sha": g("rev-parse", "HEAD"),
        "dirty": bool(g("status", "--porcelain")),
    }


def model_provenance() -> dict:
    shards = []
    total = 0
    for p in MODEL_SHARDS:
        stt = p.stat()
        total += stt.st_size
        shards.append(
            {
                "path": str(p),
                "size_bytes": stt.st_size,
                "size_gb": round(stt.st_size / 1e9, 3),
                "mtime_iso": datetime.fromtimestamp(stt.st_mtime, timezone.utc).isoformat(),
            }
        )
    return {
        "repo": "unsloth/Qwen3.8-Flash-Next-GGUF",
        "revision": MODEL_DIR.parent.name,
        "quant": "UD-IQ4_XS",
        "shards": shards,
        "total_bytes": total,
        "total_gb": round(total / 1e9, 3),
        "fingerprint": "size+mtime (no sha256: 93.7 GB, hashing would evict ARC and perturb the probe)",
        "note": (
            "checkpoint is LARGER than installed RAM (91.8 GB) and the pool is ZFS, "
            "so the baseline is expected to be storage/ARC bound"
        ),
    }


def runtime_provenance() -> dict:
    ver = "<absent>"
    if LLAMACPP_VERSION_FILE.exists():
        ver = LLAMACPP_VERSION_FILE.read_text().strip()
    return {
        "engine": "llama.cpp (lemonade rocm-nightly build)",
        "server_binary": str(LLAMA_SERVER),
        "lemonade_channel_version": ver,
        "binary_mtime_iso": (
            datetime.fromtimestamp(LLAMA_SERVER.stat().st_mtime, timezone.utc).isoformat()
            if LLAMA_SERVER.exists()
            else None
        ),
    }


# --------------------------------------------------------------------------- #
# llama-server lifecycle
# --------------------------------------------------------------------------- #
def build_server_argv(a: argparse.Namespace, n_parallel: int, port: int) -> list:
    """The launch line.  Only knobs that change between legs are `-np` and the
    per-slot context; everything else is held fixed so the two legs are
    comparable, and the whole argv is recorded in the JSON."""
    argv = [
        str(LLAMA_SERVER),
        "-m",
        str(MODEL_SHARD0),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        str(n_parallel),
        "-c",
        str(a.ctx_total),
        "-fit",
        a.fit,
        "-fa",
        a.flash_attn,
        "-t",
        str(a.threads),
        "--no-webui",
        "--metrics",
        # Belt-and-braces against a served prefill.  The request body already
        # sets cache_prompt=false and every prompt carries a fresh 16-byte nonce
        # at token ~8, but `cache_prompt` is a per-request field upstream has
        # churned and `--cache-ram` (default 8192 MiB) plus --cache-idle-slots
        # are server-side caches the request body does not reach.  0 disables.
        "--cache-ram",
        str(a.cache_ram_mib),
    ]
    if a.load_mode:
        argv += ["-lm", a.load_mode]
    if a.lazy_mode:
        argv += ["-lzm", a.lazy_mode]
    if a.split_mode:
        argv += ["-sm", a.split_mode]
    if a.extra_server_args:
        argv += a.extra_server_args.split()
    return argv


_LOG_KEEP_PATTERNS = [
    re.compile(p)
    for p in (
        r"^load_tensors:",
        r"^print_info:\s+(n_layer|n_expert|n_expert_used|n_embd|n_ctx_train|model type|model params|model size|file type|arch)",
        r"^llama_context:",
        r"^llama_kv_cache",
        r"^init:",
        r"^ggml_(cuda|hip)_init",
        r"Device \d+:",
        r"^main: server is listening",
        r"^srv .*(loading model|starting)",
        r"^build:",
        r"^system_info:",
        r"fit:",
    )
]


# Lines that must NEVER be dropped by the excerpt cap -- they carry the device
# enumeration, the --fit decision and the offload split, i.e. the evidence that
# this run is the configuration the probe claims to have measured.
_LOG_CRITICAL_PATTERNS = [
    re.compile(p)
    for p in (
        r"^ggml_(cuda|hip)_init",
        r"^\s*Device \d+:",
        r"^load_tensors:.*(buffer size|offload)",
        r"fit:",
        r"^main: server is listening",
        r"^print_info:\s+(n_layer|n_expert|n_expert_used|model size|model params|arch|file type)",
        r"^llama_context:\s+n_ctx",
    )
]


def harvest_server_log(logpath: Path, limit: int = 400) -> list:
    """Excerpt the server log.

    The cap is applied to the NOISY class only.  Critical lines (device list,
    fit decision, per-device buffer sizes, listening banner) are always kept:
    llama.cpp emits hundreds of `load_tensors:` lines for a 3-shard 93.7 GB
    checkpoint, and a naive head-of-file cap silently truncates exactly the
    evidence that proves a GPU was engaged.
    """
    if not logpath.exists():
        return []
    critical, noisy = [], []
    try:
        for line in logpath.read_text(errors="replace").splitlines():
            s = line.strip()
            if any(p.search(s) for p in _LOG_CRITICAL_PATTERNS):
                critical.append(s)
            elif any(p.search(s) for p in _LOG_KEEP_PATTERNS):
                if len(noisy) < limit:
                    noisy.append(s)
    except OSError:
        pass
    if len(critical) > limit * 2:
        critical = critical[: limit * 2]
    return critical + noisy


# --------------------------------------------------------------------------- #
# Backend attribution -- did this run actually use the two discrete cards?
#
# The probe's headline claim is "llama.cpp on BOTH gfx1201 cards with --fit
# auto-offload".  Nothing about a plausible tok/s number distinguishes that from
# a run where the HIP backend failed to register, or where `--fit` decided zero
# layers fit, or where the iGPU leaked into enumeration and `--fit` sized against
# its 47 GB GTT pool.  All three produce a confident, wrong baseline that then
# retunes K4 and A0.4.  So: assert on the OPERATION (weights landed on device,
# VRAM moved), never on the flag having been passed.
# --------------------------------------------------------------------------- #
_RE_DEV_COUNT = re.compile(r"ggml_(?:cuda|hip)_init: found (\d+) (?:ROCm|HIP|CUDA) devices")
_RE_DEV_LINE = re.compile(r"^\s*Device (\d+):\s*(.+?),\s*gfx(\w+)")
_RE_DEV_LINE_LOOSE = re.compile(r"^\s*Device (\d+):\s*(.+)$")
_RE_BUF = re.compile(r"^load_tensors:\s+(\S+)\s+model buffer size\s*=\s*([\d.]+)\s*(\w+)")
_RE_OFFLOADED = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
_RE_PI_INT = re.compile(r"^print_info:\s+(n_layer|n_expert|n_expert_used|n_embd)\s*=\s*(\d+)")
_RE_PI_SIZE = re.compile(r"^print_info:\s+model size\s*=\s*([\d.]+)\s+(\w+)")

_UNIT_GB = {"B": 1e-9, "KiB": 1024 / 1e9, "MiB": 1024**2 / 1e9, "GiB": 1024**3 / 1e9,
            "KB": 1e-6, "MB": 1e-3, "GB": 1.0, "TiB": 1024**4 / 1e9}


def _to_gb(val: float, unit: str):
    f = _UNIT_GB.get(unit)
    return round(val * f, 4) if f else None


def parse_server_log_backend(logpath: Path) -> dict:
    """Extract device enumeration, per-buffer weight placement and model geometry.

    Everything here is best-effort parsing of a log format that upstream churns;
    a parse miss yields `unknown`, never a fabricated `True`.
    """
    out = {
        "devices": [],
        "device_count_reported": None,
        "buffers_gb": {},
        "gpu_weight_gb": None,
        "cpu_weight_gb": None,
        "layers_offloaded": None,
        "layers_total": None,
        "model_geometry": {},
        "igpu_in_enumeration": None,
        "parse_notes": [],
    }
    if not logpath.exists():
        out["parse_notes"].append("server log absent")
        return out
    try:
        text = logpath.read_text(errors="replace")
    except OSError as e:
        out["parse_notes"].append(f"unreadable: {e}")
        return out
    for line in text.splitlines():
        s = line.strip()
        m = _RE_DEV_COUNT.search(s)
        if m:
            out["device_count_reported"] = int(m.group(1))
        m = _RE_DEV_LINE.match(s) or _RE_DEV_LINE_LOOSE.match(s)
        if m and "model buffer" not in s:
            out["devices"].append(s)
        m = _RE_BUF.match(s)
        if m:
            gb = _to_gb(float(m.group(2)), m.group(3))
            if gb is not None:
                out["buffers_gb"][m.group(1)] = round(
                    out["buffers_gb"].get(m.group(1), 0.0) + gb, 4
                )
        m = _RE_OFFLOADED.search(s)
        if m:
            out["layers_offloaded"], out["layers_total"] = int(m.group(1)), int(m.group(2))
        m = _RE_PI_INT.match(s)
        if m:
            out["model_geometry"][m.group(1)] = int(m.group(2))
        m = _RE_PI_SIZE.match(s)
        if m:
            gb = _to_gb(float(m.group(1)), m.group(2))
            if gb is not None:
                out["model_geometry"]["model_size_gb"] = gb
    if out["buffers_gb"]:
        gpu = sum(
            v for k, v in out["buffers_gb"].items()
            if re.match(r"(ROCm|CUDA|HIP|Vulkan|SYCL)\d*$", k)
        )
        cpu = sum(v for k, v in out["buffers_gb"].items() if "CPU" in k or "Host" in k)
        out["gpu_weight_gb"] = round(gpu, 4)
        out["cpu_weight_gb"] = round(cpu, 4)
    if out["devices"]:
        archs = [m.group(1) for m in (re.search(r"gfx(\w+)", d) for d in out["devices"]) if m]
        out["device_archs"] = archs
        foreign_name = [d for d in out["devices"] if _IGPU_NAME_RE.search(d)]
        # The only legal compute targets on this box are gfx1201.  Any other arch
        # in llama.cpp's enumeration means the ROCR fence did not hold -- most
        # likely the Ryzen iGPU, whose ~47 GB GTT pool poisons `--fit` sizing.
        foreign_arch = [a_ for a_ in archs if not a_.lower().startswith("1201")]
        if foreign_name or foreign_arch:
            out["igpu_in_enumeration"] = True
            out["parse_notes"].append(
                f"foreign device(s) enumerated: names={foreign_name} archs={foreign_arch}"
            )
        elif archs:
            out["igpu_in_enumeration"] = False
        else:
            out["parse_notes"].append(
                "device lines found but no gfx arch parsed; cannot prove the fence held"
            )
    else:
        out["parse_notes"].append("no `Device N:` lines found; device list unknown")
    return out


def attribute_gpu_engagement(backend: dict, vram_deltas: dict, min_gb: float) -> dict:
    """Combine the log parse with the format-independent VRAM evidence.

    Verdicts:
      confirmed    -- weights demonstrably on device (log and/or VRAM movement)
      unconfirmed  -- no positive evidence either way -> degrade (exit 7)
      cpu_only     -- positive evidence of NO device weights -> invalid (exit 8)
      igpu_leak    -- the iGPU entered enumeration -> invalid (exit 8)
    """
    moved = {k: v for k, v in (vram_deltas or {}).items() if v is not None and v >= min_gb}
    log_gpu = backend.get("gpu_weight_gb")
    ev = {
        "cards_with_vram_growth_gb": moved,
        "vram_delta_gb_all_compute_cards": vram_deltas,
        "gpu_weight_gb_from_log": log_gpu,
        "cpu_weight_gb_from_log": backend.get("cpu_weight_gb"),
        "layers_offloaded": backend.get("layers_offloaded"),
        "layers_total": backend.get("layers_total"),
        "devices_enumerated": backend.get("devices"),
        "device_count_reported": backend.get("device_count_reported"),
        "min_gb_threshold": min_gb,
    }
    if backend.get("igpu_in_enumeration"):
        ev["verdict"] = "igpu_leak"
        ev["why"] = (
            "the Ryzen iGPU appears in llama.cpp's device enumeration despite "
            "ROCR_VISIBLE_DEVICES=0,1. Its ~47 GB GTT pool poisons `--fit` auto-sizing, "
            "so the offload split -- and therefore the tok/s -- is not the configuration "
            "this probe claims to measure."
        )
        return ev
    dc = backend.get("device_count_reported")
    if dc is not None and dc != 2:
        ev["verdict"] = "igpu_leak" if dc > 2 else "unconfirmed"
        ev["why"] = (
            f"llama.cpp enumerated {dc} ROCm device(s); this box must present exactly 2 "
            f"discrete gfx1201 cards under ROCR_VISIBLE_DEVICES=0,1."
        )
        return ev
    positive = bool(moved) or (log_gpu is not None and log_gpu >= min_gb)
    if positive:
        ev["verdict"] = "confirmed"
        ev["why"] = (
            f"weights on device: {log_gpu} GB in the log's ROCm buffers; "
            f"VRAM grew on {sorted(moved)} across model load."
        )
        return ev
    negative = (
        log_gpu is not None
        and log_gpu < min_gb
        and vram_deltas
        and all((v or 0.0) < min_gb for v in vram_deltas.values())
    )
    if negative:
        ev["verdict"] = "cpu_only"
        ev["why"] = (
            "the server placed no meaningful weight bytes on either compute card and no "
            "VRAM moved during load. This is a CPU-only baseline; quoting it as the "
            "2-card --fit auto-offload number would silently retune K4 and A0.4."
        )
        return ev
    ev["verdict"] = "unconfirmed"
    ev["why"] = (
        "neither the server log nor the VRAM deltas positively establish that weights "
        "landed on the discrete cards. The number may still be right, but it is not "
        "attributable to the claimed configuration."
    )
    return ev


# --------------------------------------------------------------------------- #
# GGUF geometry, read from the checkpoint itself.
#
# The server log is the WRONG place to get this.  On this llama.cpp build
# (lemonade rocm-nightly b1319) the `print_info:` / `load_tensors:` lines are
# not emitted at all, so the log-derived denominator came back null and the
# primary I/O-attribution ratio -- measured storage bytes per token as a
# fraction of the checkpoint's OWN active expert bytes -- could not be formed.
# The checkpoint's tensor table is authoritative, always present, and does not
# depend on a log format: walk it and sum the `*_exps` tensors EXACTLY, using
# each tensor's real ggml type size.  This replaces the pro-rata
# `total_bytes x k/E` upper bound (which charges attention/embedding/PLE bytes
# to the expert term: 1.83 GB vs the true 1.16 GB here, a 57% overstatement).
#
# Validation is on the operation, not the query: the walk's summed total is
# compared against the on-disk file sizes and REJECTED if it does not land
# within 1%, so a wrong type-size table cannot silently fabricate a
# denominator.
# --------------------------------------------------------------------------- #
_GGML_QK_K = 256
# ggml type id -> (block elements, bytes per block)
_GGML_TYPE_SIZE = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 24), 7: (32, 34),
    8: (32, 34), 9: (32, 36),
    10: (_GGML_QK_K, 84), 11: (_GGML_QK_K, 110), 12: (_GGML_QK_K, 144),
    13: (_GGML_QK_K, 176), 14: (_GGML_QK_K, 210), 15: (_GGML_QK_K, 292),
    16: (_GGML_QK_K, 66), 17: (_GGML_QK_K, 74), 18: (_GGML_QK_K, 98),
    19: (_GGML_QK_K, 50), 20: (32, 18), 21: (_GGML_QK_K, 110),
    22: (_GGML_QK_K, 82), 23: (_GGML_QK_K, 136),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (_GGML_QK_K, 56), 30: (1, 2),
}
_GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    18: "IQ3_XXS", 20: "IQ4_NL", 21: "IQ3_S", 23: "IQ4_XS", 30: "BF16",
}


def _gguf_read_shard(path: Path) -> tuple:
    """(kv_of_interest, [(tensor_name, ggml_type, n_bytes), ...]) for one shard."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path.name}: not a GGUF file")
        struct.unpack("<I", f.read(4))  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def skip_value(t: int):
            if t in (0, 1, 7):
                f.read(1)
            elif t in (2, 3):
                f.read(2)
            elif t in (4, 5, 6):
                f.read(4)
            elif t in (10, 11, 12):
                f.read(8)
            elif t == 8:
                f.read(struct.unpack("<Q", f.read(8))[0])
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                for _ in range(n):
                    skip_value(et)
            else:
                raise ValueError(f"unknown gguf value type {t}")

        def read_scalar(t: int):
            if t in (4, 10):
                return struct.unpack("<I" if t == 4 else "<Q", f.read(4 if t == 4 else 8))[0]
            if t in (5, 11):
                return struct.unpack("<i" if t == 5 else "<q", f.read(4 if t == 5 else 8))[0]
            skip_value(t)
            return None

        kv = {}
        for _ in range(n_kv):
            key = f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf8", "replace")
            t = struct.unpack("<I", f.read(4))[0]
            if key.endswith((".expert_count", ".expert_used_count", ".block_count")):
                kv[key] = read_scalar(t)
            else:
                skip_value(t)

        tensors = []
        for _ in range(n_tensors):
            name = f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf8", "replace")
            nd = struct.unpack("<I", f.read(4))[0]
            ne = 1
            for _ in range(nd):
                ne *= struct.unpack("<Q", f.read(8))[0]
            ttype = struct.unpack("<I", f.read(4))[0]
            struct.unpack("<Q", f.read(8))  # offset
            blk, tsz = _GGML_TYPE_SIZE[ttype]
            tensors.append((name, ttype, ne // blk * tsz))
    return kv, tensors


def gguf_expert_geometry(shards) -> dict:
    """Exact active-expert bytes per decoded token, from the checkpoint itself."""
    out = {"source": "gguf tensor table", "ok": False}
    try:
        total = 0
        exps = 0
        by_type: dict = {}
        layers = set()
        kv_all: dict = {}
        for p in shards:
            kv, tensors = _gguf_read_shard(Path(p))
            kv_all.update({k: v for k, v in kv.items() if v is not None})
            for name, ttype, nb in tensors:
                total += nb
                if "_exps" in name:
                    exps += nb
                    tn = _GGML_TYPE_NAME.get(ttype, str(ttype))
                    by_type[tn] = by_type.get(tn, 0) + nb
                    parts = name.split(".")
                    if len(parts) > 1 and parts[1].isdigit():
                        layers.add(int(parts[1]))
        n_expert = next((v for k, v in kv_all.items() if k.endswith(".expert_count")), None)
        k_used = next((v for k, v in kv_all.items() if k.endswith(".expert_used_count")), None)
        on_disk = sum(Path(p).stat().st_size for p in shards)
        out.update(
            {
                "tensor_bytes_total": total,
                "tensor_bytes_total_gb": round(total / 1e9, 4),
                "on_disk_bytes": on_disk,
                "on_disk_gb": round(on_disk / 1e9, 4),
                "walk_vs_on_disk_ratio": round(total / on_disk, 6) if on_disk else None,
                "expert_tensor_bytes_gb": round(exps / 1e9, 4),
                "expert_fraction_of_checkpoint": round(exps / total, 4) if total else None,
                "expert_bytes_by_ggml_type_gb": {k: round(v / 1e9, 4) for k, v in by_type.items()},
                "moe_layers_seen": len(layers),
                "n_expert": n_expert,
                "n_expert_used": k_used,
            }
        )
        # Assert on the OPERATION: the walk must reconstruct the on-disk size.
        if not (on_disk and 0.99 <= total / on_disk <= 1.01):
            out["reject_reason"] = (
                f"tensor-table walk summed {total/1e9:.3f} GB but the shards are "
                f"{on_disk/1e9:.3f} GB on disk (>1% apart): the ggml type-size table "
                f"does not cover this checkpoint, so the denominator is not trustworthy."
            )
            return out
        if not (n_expert and k_used and n_expert > 0):
            out["reject_reason"] = "expert_count / expert_used_count missing from GGUF KV"
            return out
        out["active_expert_bytes_per_token_gb"] = round(exps * k_used / n_expert / 1e9, 4)
        out["basis"] = (
            f"EXACT: sum of all `*_exps` tensors ({exps/1e9:.3f} GB over "
            f"{len(layers)} MoE layers, per-tensor ggml type sizes) x "
            f"{k_used}/{n_expert} routed. Walk reconstructs "
            f"{total/on_disk*100:.2f}% of the on-disk bytes."
        )
        out["ok"] = True
    except Exception as e:  # noqa: BLE001 - a bad parse must degrade, never crash the probe
        out["reject_reason"] = f"{type(e).__name__}: {e}"
    return out


_GGUF_GEOMETRY_CACHE: dict = {}


def gguf_geometry_cached() -> dict:
    if "v" not in _GGUF_GEOMETRY_CACHE:
        _GGUF_GEOMETRY_CACHE["v"] = gguf_expert_geometry(MODEL_SHARDS)
    return _GGUF_GEOMETRY_CACHE["v"]


def expert_bytes_per_token_gb(backend: dict) -> dict:
    """Per-token active expert bytes for THIS checkpoint, not for the plan's shape.

    The plan's 1.33 GB/token is arithmetic on the *minisgl* target shape
    (48 layers x 10 routed x 2.8 MB of w4a8/bf16 rows).  This leg runs a
    UD-IQ4_XS GGUF with different layer count, expert count and bytes/row, so
    dividing measured nvme bytes/token by 1.33 GB produces a ratio against a
    constant from a different model packing.  Estimate the real figure from the
    checkpoint's own geometry and carry the plan constant only for contrast.
    """
    g = backend.get("model_geometry") or {}
    e, k, size = g.get("n_expert"), g.get("n_expert_used"), g.get("model_size_gb")
    out = {
        "plan_constant_gb": PLAN_DERIVED["active_expert_bytes_per_token_gb"],
        "plan_constant_basis": (
            "48L x 10 routed x 2.8 MB for the minisgl target shape -- a DIFFERENT "
            "packing from this GGUF; not directly comparable"
        ),
        "geometry": {"n_expert": e, "n_expert_used": k, "model_size_gb": size},
    }
    # Preferred source: the checkpoint's own tensor table.  Independent of the
    # server log's format, exact rather than pro-rata, and self-validating
    # against the on-disk shard sizes.
    gg = backend.get("_gguf_geometry")
    if gg is None:
        gg = gguf_geometry_cached()
    out["gguf_geometry"] = gg
    if gg.get("ok"):
        out["estimate_gb"] = gg["active_expert_bytes_per_token_gb"]
        out["estimate_basis"] = gg["basis"]
        out["estimate_source"] = "gguf_tensor_table_exact"
        # The cruder `total x k/E` bound, for contrast.  Derive it from the GGUF
        # walk's own total so it is available even when the server log yields no
        # geometry at all (this llama.cpp build emits none).
        pr_total = gg.get("tensor_bytes_total_gb") if size is None else size
        pr_e = gg.get("n_expert") if not e else e
        pr_k = gg.get("n_expert_used") if not k else k
        if pr_total and pr_e and pr_k:
            out["pro_rata_upper_bound_gb"] = round(pr_total * pr_k / pr_e, 4)
            out["pro_rata_upper_bound_basis"] = (
                "total weight bytes x n_expert_used/n_expert -- charges "
                "attention/dense/embedding/PLE bytes to the expert term pro-rata, "
                "so it OVERSTATES per-token expert traffic. Contrast only."
            )
        return out
    if e and k and size and e > 0:
        out["estimate_gb"] = round(size * k / e, 4)
        out["estimate_source"] = "server_log_pro_rata_upper_bound"
        out["estimate_basis"] = (
            "total weight bytes x n_expert_used/n_expert. Dominant-term "
            "approximation: it charges attention/dense/embedding bytes to the expert "
            "term pro-rata, so it is an UPPER bound on per-token expert traffic. "
            f"(GGUF tensor-table walk unavailable: {gg.get('reject_reason')})"
        )
    else:
        out["estimate_gb"] = None
        out["estimate_source"] = None
        out["estimate_basis"] = (
            "neither the GGUF tensor table nor the server log yielded geometry "
            f"(gguf: {gg.get('reject_reason')})"
        )
    return out


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_port_free(port: int, timeout: float = 120.0) -> bool:
    """Both legs reuse one port.  The second server binds seconds after the first
    was SIGKILLed; if the old listener has not been reaped, leg 2 dies at load and
    the ~30 minutes leg 1 cost are thrown away with an exit 3 that looks like a
    model problem.  Wait for the port instead of racing it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_free(port):
            return True
        time.sleep(1.0)
    return port_is_free(port)


class LlamaServer:
    def __init__(self, argv: list, logpath: Path, ready_timeout: float, port: int):
        self.argv = argv
        self.logpath = logpath
        self.ready_timeout = ready_timeout
        self.port = port
        self.proc = None
        self.logfh = None
        self.load_seconds = None

    @property
    def pid(self):
        return self.proc.pid if self.proc is not None else None

    def start(self) -> None:
        if not wait_port_free(self.port):
            raise PreconditionError(
                f"port {self.port} did not become free within 120 s; a previous "
                f"llama-server has not released it. Refusing to race the bind."
            )
        self.logfh = open(self.logpath, "w", buffering=1)
        self.logfh.write("# argv: " + " ".join(self.argv) + "\n")
        self.logfh.write(f"# ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}\n")
        self.logfh.write(f"# HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES','<unset>')}\n")
        self.logfh.flush()
        log(f"launching llama-server -> {self.logpath.name}")
        log("  " + " ".join(self.argv))
        t0 = time.time()
        env = dict(os.environ)
        env["ROCR_VISIBLE_DEVICES"] = "0,1"
        env.pop("HIP_VISIBLE_DEVICES", None)
        self.proc = subprocess.Popen(
            self.argv,
            stdout=self.logfh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        self._wait_ready(t0)

    def _wait_ready(self, t0: float) -> None:
        url = f"http://127.0.0.1:{self.port}/health"
        deadline = t0 + self.ready_timeout
        last_note = 0.0
        while time.time() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                tail = self._log_tail()
                self.stop()
                raise PreconditionError(
                    f"llama-server exited with rc={rc} during load.\n"
                    f"argv: {' '.join(self.argv)}\n"
                    f"log tail ({self.logpath}):\n{tail}"
                )
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    if r.status == 200:
                        self.load_seconds = round(time.time() - t0, 2)
                        log(f"server ready after {self.load_seconds}s")
                        return
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
                pass
            now = time.time()
            if now - last_note > 30:
                last_note = now
                log(f"  ... loading, {now - t0:.0f}s elapsed (93.7 GB checkpoint on ZFS)")
            time.sleep(2.0)
        tail = self._log_tail()
        self.stop()
        raise PreconditionError(
            f"llama-server did not become healthy within {self.ready_timeout}s.\n"
            f"Raise --ready-timeout if the checkpoint is cold.\nlog tail:\n{tail}"
        )

    def _log_tail(self, n: int = 40) -> str:
        try:
            if self.logfh:
                self.logfh.flush()
            return "\n".join(self.logpath.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "<log unreadable>"

    def props(self) -> dict:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/props", timeout=30
            ) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001 - provenance only, never fatal
            return {"_error": repr(e)}

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            log("stopping llama-server")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except OSError:
                try:
                    self.proc.terminate()
                except OSError:
                    pass
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                log("SIGTERM ignored; SIGKILL")
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        self.proc = None
        if self.logfh:
            try:
                self.logfh.close()
            except OSError:
                pass
            self.logfh = None


# --------------------------------------------------------------------------- #
# One generation request
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATE = (
    "You are a systems engineer. Session nonce {nonce}. "
    "Explain, in careful and concrete detail, how a memory-mapped file backed by a "
    "copy-on-write filesystem behaves when the resident working set exceeds physical "
    "RAM, and what the resulting page-fault pattern implies for sustained throughput.\n\n"
    "Answer:"
)


def make_prompt() -> str:
    # Fresh nonce per request: no repetition may be served from a prompt cache,
    # otherwise the "decode" leg quietly measures cache lookups.
    return PROMPT_TEMPLATE.format(nonce=os.urandom(16).hex())


def one_request(port: int, n_predict: int, timeout: float, seed: int) -> dict:
    """Returns a rep record.  `ok` False means the rep is INVALID and must not
    enter any statistic."""
    body = {
        "prompt": make_prompt(),
        "n_predict": n_predict,
        # Sampled, never greedy (repo rule); irrelevant to speed but keeps the
        # generation on a realistic path and out of any degenerate short loop.
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        # Exactly n_predict tokens every rep, or the reps are not comparable.
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
        "seed": seed,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "error": repr(e),
            "wall_s": round(time.time() - t0, 3),
            "t_start": t0,
            "t_end": time.time(),
        }
    t1 = time.time()
    tim = payload.get("timings") or {}
    rec = {
        "ok": True,
        "t_start": t0,
        "t_end": t1,
        "wall_s": round(t1 - t0, 3),
        "predicted_n": tim.get("predicted_n"),
        "predicted_ms": tim.get("predicted_ms"),
        "predicted_per_second": tim.get("predicted_per_second"),
        "prompt_n": tim.get("prompt_n"),
        "prompt_ms": tim.get("prompt_ms"),
        "prompt_per_second": tim.get("prompt_per_second"),
        "stop_type": payload.get("stop_type"),
        "truncated": payload.get("truncated"),
    }
    if rec["predicted_n"] != n_predict or not rec["predicted_per_second"]:
        rec["ok"] = False
        rec["error"] = (
            f"server returned predicted_n={rec['predicted_n']} (wanted {n_predict}) / "
            f"predicted_per_second={rec['predicted_per_second']}"
        )
    return rec


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def summarize(vals: list) -> dict:
    """Median/spread PLUS the chronological series and a drift check.

    `values` is kept in REP ORDER, not sorted: sorting destroys the only evidence
    that distinguishes a steady state from a ramp.  One 100-token warm-up cannot
    warm a 93.7 GB working set on a box with ~50 GB available, so the reps can be
    monotonically climbing (ARC filling) or sagging (memory pressure) -- in which
    case the median is a point on a curve, not a rate, and every gate downstream
    inherits the error.
    """
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    out = {
        "n": len(s),
        "median": round(statistics.median(s), 4),
        "mean": round(statistics.fmean(s), 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "stdev": round(statistics.stdev(s), 4) if len(s) > 1 else 0.0,
        "values": [round(v, 4) for v in vals],  # CHRONOLOGICAL, not sorted
        "values_sorted": [round(v, 4) for v in s],
    }
    out["spread_pct_of_median"] = (
        round(100.0 * (out["max"] - out["min"]) / out["median"], 2) if out["median"] else None
    )
    out["drift"] = drift_stats(vals)
    return out


DRIFT_PCT_LIMIT = 25.0


def drift_stats(vals: list, limit_pct: float = DRIFT_PCT_LIMIT) -> dict:
    """Half-vs-half trend over the chronological reps."""
    n = len(vals)
    if n < 4:
        return {"n": n, "assessable": False, "note": "need >= 4 reps to assess drift"}
    h = n // 2
    first = statistics.fmean(vals[:h])
    second = statistics.fmean(vals[n - h :])
    pct = 100.0 * (second - first) / first if first else None
    monotone = all(b >= a for a, b in zip(vals, vals[1:])) or all(
        b <= a for a, b in zip(vals, vals[1:])
    )
    d = {
        "n": n,
        "assessable": True,
        "first_half_mean": round(first, 4),
        "second_half_mean": round(second, 4),
        "drift_pct_second_vs_first_half": round(pct, 2) if pct is not None else None,
        "strictly_monotone": bool(monotone),
        "limit_pct": limit_pct,
    }
    d["steady_state"] = bool(
        pct is not None and abs(pct) <= limit_pct and not (monotone and abs(pct) > 5.0)
    )
    if not d["steady_state"]:
        d["why"] = (
            "reps trend rather than fluctuate: the median is a point on a ramp, not a "
            "steady-state rate. Extend the warm-up or the rep count before citing it."
        )
    return d


# --------------------------------------------------------------------------- #
# Legs
# --------------------------------------------------------------------------- #
def run_leg_bs1(a: argparse.Namespace, port: int) -> dict:
    log(f"LEG bs=1: 1 warmup + {a.reps} reps x {a.n_predict} tokens")
    warm = one_request(port, a.n_predict, a.request_timeout, seed=1000)
    log(f"  warmup (DISCARDED): {warm.get('predicted_per_second')} tok/s ok={warm['ok']}")
    reps = []
    for i in range(a.reps):
        r = one_request(port, a.n_predict, a.request_timeout, seed=2000 + i)
        reps.append(r)
        log(
            f"  rep {i + 1}/{a.reps}: decode={r.get('predicted_per_second')} tok/s "
            f"prompt={r.get('prompt_per_second')} tok/s wall={r.get('wall_s')}s ok={r['ok']}"
        )
    good = [r for r in reps if r["ok"]]
    leg = {
        "kind": "bs1",
        "concurrency": 1,
        "n_predict": a.n_predict,
        "reps_requested": a.reps,
        "reps_valid": len(good),
        "warmup_discarded": warm,
        "reps": reps,
        "decode_tok_s": summarize([r["predicted_per_second"] for r in good]),
        "prompt_tok_s": summarize([r["prompt_per_second"] for r in good]),
        "wall_s": summarize([r["wall_s"] for r in good]),
    }
    # Wall-clock rate: tokens over the FULL request wall, prefill included. This
    # is the only figure comparable like-for-like with the CONC leg's aggregate
    # wall throughput; `decode_tok_s` is server-side decode-only.
    leg["wall_tok_s"] = summarize(
        [a.n_predict / r["wall_s"] for r in good if r.get("wall_s")]
    )
    leg["prompt_cache_check"] = prompt_cache_check(good)
    leg["steady_state"] = bool((leg["decode_tok_s"].get("drift") or {}).get("steady_state", True))
    return leg


def prompt_cache_check(reps: list) -> dict:
    """A served prefill would show up as a collapsing prompt_n or prompt_ms.

    Every prompt carries a fresh 16-byte nonce at token ~8 and cache_prompt is
    false, but the assertion is on the OBSERVED prefill cost, not on the flags
    having been passed.
    """
    ns = sorted({r.get("prompt_n") for r in reps if r.get("prompt_n") is not None})
    ms = [r.get("prompt_ms") for r in reps if r.get("prompt_ms") is not None]
    out = {"distinct_prompt_n": ns, "prompt_n_constant": len(ns) <= 1}
    if ms:
        out["prompt_ms_min"] = round(min(ms), 2)
        out["prompt_ms_max"] = round(max(ms), 2)
        out["prompt_ms_collapse_ratio"] = round(min(ms) / max(ms), 3) if max(ms) else None
        # A real cache hit drops prefill by orders of magnitude, not by jitter.
        out["suspected_prefill_cache_hit"] = bool(
            max(ms) and (min(ms) / max(ms)) < 0.1
        )
    return out


def run_leg_conc(a: argparse.Namespace, port: int, conc: int) -> dict:
    log(f"LEG CONC={conc}: 1 warmup + {a.reps} reps of {conc} concurrent x {a.n_predict} tokens")

    def burst(seed_base: int) -> dict:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [
                ex.submit(one_request, port, a.n_predict, a.request_timeout, seed_base + j)
                for j in range(conc)
            ]
            recs = [f.result() for f in futs]
        t1 = time.time()
        good = [r for r in recs if r["ok"]]
        tokens = sum(r["predicted_n"] for r in good)
        wall = t1 - t0
        return {
            "streams": recs,
            "streams_valid": len(good),
            "wall_s": round(wall, 3),
            "tokens_generated": tokens,
            # Primary: what a user actually observes -- total generated tokens
            # over the wall-clock of the burst, prefill included.
            "aggregate_tok_s_wall": round(tokens / wall, 4) if wall > 0 else None,
            # Secondary: server-side decode-only rates summed across slots.
            # Higher than the wall figure by exactly the prefill+queueing cost.
            "aggregate_decode_tok_s_sum_of_rates": round(
                sum(r["predicted_per_second"] for r in good), 4
            )
            if good
            else None,
            "per_stream_decode_tok_s": summarize(
                [r["predicted_per_second"] for r in good]
            ),
            "per_stream_prompt_ms": summarize([r["prompt_ms"] for r in good]),
        }

    warm = burst(3000)
    log(
        f"  warmup (DISCARDED): aggregate={warm['aggregate_tok_s_wall']} tok/s "
        f"valid={warm['streams_valid']}/{conc}"
    )
    reps = []
    for i in range(a.reps):
        b = burst(4000 + 100 * i)
        reps.append(b)
        log(
            f"  rep {i + 1}/{a.reps}: aggregate={b['aggregate_tok_s_wall']} tok/s "
            f"(sum-of-rates {b['aggregate_decode_tok_s_sum_of_rates']}) "
            f"per-stream median={b['per_stream_decode_tok_s'].get('median')} "
            f"valid={b['streams_valid']}/{conc}"
        )
    full = [b for b in reps if b["streams_valid"] == conc]
    leg = {
        "kind": f"conc{conc}",
        "concurrency": conc,
        "n_predict": a.n_predict,
        "reps_requested": a.reps,
        "reps_valid": len(full),
        "warmup_discarded": warm,
        "reps": reps,
        "aggregate_tok_s_wall": summarize([b["aggregate_tok_s_wall"] for b in full]),
        "aggregate_decode_tok_s_sum_of_rates": summarize(
            [b["aggregate_decode_tok_s_sum_of_rates"] for b in full]
        ),
        "per_stream_decode_tok_s": summarize(
            [
                s["predicted_per_second"]
                for b in full
                for s in b["streams"]
                if s["ok"]
            ]
        ),
        "burst_measurement_caveat": (
            "aggregate_tok_s_wall is a CLOSED burst: all `conc` requests start together "
            "and the wall runs until the LAST finishes, so the tail of each burst runs "
            "below full concurrency. It under-reports steady-state aggregate throughput "
            "and is the conservative of the two aggregates."
        ),
    }
    leg["prompt_cache_check"] = prompt_cache_check(
        [s for b in full for s in b["streams"] if s["ok"]]
    )
    leg["steady_state"] = bool(
        (leg["aggregate_tok_s_wall"].get("drift") or {}).get("steady_state", True)
    )
    return leg


# --------------------------------------------------------------------------- #
# Derived analysis -- arithmetic on MEASURED values only, each labelled.
# --------------------------------------------------------------------------- #
def derive(
    bs1_tok_s,
    conc_tok_s,
    conc: int,
    bs1_wall_tok_s=None,
    conc_sum_of_rates=None,
) -> dict:
    d = {
        "_note": (
            "Every field here is ARITHMETIC on the measured bs=1 median. Nothing "
            "here is itself a measurement. Constants come from WEIGHT_OFFLOAD_PLAN.md "
            "Section 1 and are labelled."
        ),
        "constants": dict(PLAN_DERIVED),
    }
    if bs1_tok_s is None:
        d["_unavailable"] = "bs=1 leg did not produce a valid median"
        return d

    ceil_ = PLAN_DERIVED["cpu_expert_ceiling_tok_s"]
    d["measured_bs1_tok_s"] = bs1_tok_s
    d["fraction_of_plan_derived_ceiling"] = round(bs1_tok_s / ceil_, 4)
    d["plan_derived_ceiling_reachable_in_practice"] = bool(bs1_tok_s >= 0.8 * ceil_)
    d["shortfall_factor_vs_plan_ceiling"] = round(ceil_ / bs1_tok_s, 2)

    # Gate K4: M1 must reach >= llama.cpp / 1.57, else hard kill.
    d["K4_hard_kill_threshold_tok_s"] = round(bs1_tok_s / 1.57, 3)
    d["K4_statement"] = (
        f"M1 (all-host weight arena) must measure >= {bs1_tok_s / 1.57:.2f} tok/s at bs=1 "
        f"or the charter's all-compute-on-GPU constraint costs more than it buys."
    )

    # Section 3, P0 kill-note: "If llama.cpp >= 25 tok/s ... the project's value
    # proposition is quality/context, not speed."
    d["P0_note_threshold_tok_s"] = 25.0
    d["llamacpp_above_25_tok_s"] = bool(bs1_tok_s >= 25.0)

    # Break-even hit rate h against the MEASURED baseline rather than against a
    # 45 GB/s DDR abstraction.  GPU-side streaming at hit rate h delivers
    #   tok/s = 1 / (t_compute + B*(1-h)/H)
    # with B = 1.33 GB active expert bytes/token and H = 26.8 GB/s PCIe at the
    # 2.8 MiB granule.  Solve tok/s >= measured baseline for h.
    B = PLAN_DERIVED["active_expert_bytes_per_token_gb"]
    H = PLAN_DERIVED["pcie_h2d_granule_gbs"]
    be = {}
    for t_compute_ms in (0.0, 5.0, 10.0):
        t_c = t_compute_ms / 1000.0
        budget_s = 1.0 / bs1_tok_s - t_c  # seconds of DMA allowed per token
        if budget_s <= 0:
            be[f"t_compute_{t_compute_ms:g}ms"] = {
                "required_h": None,
                "note": "compute floor alone already slower than the baseline",
            }
            continue
        h = 1.0 - (H * budget_s) / B
        be[f"t_compute_{t_compute_ms:g}ms"] = {
            "required_h": round(max(h, 0.0), 4),
            "required_h_pct": round(100.0 * max(h, 0.0), 2),
            "already_satisfied_at_h_0": bool(h <= 0.0),
        }
    d["break_even_hit_rate_vs_measured_baseline"] = be
    d["break_even_vs_plan_ddr_abstraction"] = {
        "required_h_pct": round(100.0 * (1.0 - H / PLAN_DERIVED["host_ddr_read_gbs"]), 2),
        "note": "the plan's Section 1(a) figure, 1 - 26.8/45; kept for comparison only",
    }

    if conc_tok_s is not None:
        d["measured_conc_aggregate_tok_s"] = conc_tok_s
        d["conc_concurrency"] = conc
        # MATCHED comparisons only.  `conc_tok_s` is burst WALL throughput and
        # therefore includes prefill and queueing; `bs1_tok_s` is the server's
        # decode-only rate and excludes prefill entirely.  Dividing one by the
        # other mixes bases and understates scaling by the prefill fraction --
        # roughly a fifth of the wall at ~4.6 tok/s with a ~5.7 s prefill.  So:
        #   wall aggregate     vs bs=1 WALL rate           (both include prefill)
        #   sum-of-decode-rates vs bs=1 DECODE rate        (both exclude prefill)
        scal = {
            "_why": (
                "bs=1 decode tok/s excludes prefill; the CONC wall aggregate includes it. "
                "Only like-for-like pairs are reported."
            )
        }
        if bs1_wall_tok_s:
            scal["wall_vs_wall"] = {
                "bs1_wall_tok_s": bs1_wall_tok_s,
                "conc_wall_tok_s": conc_tok_s,
                "scaling_factor": round(conc_tok_s / bs1_wall_tok_s, 3),
                "efficiency_vs_linear": round(conc_tok_s / (bs1_wall_tok_s * conc), 3),
                "basis": "both sides are tokens / full request wall clock, prefill included",
            }
        else:
            scal["wall_vs_wall"] = {
                "_unavailable": "no bs=1 wall-clock rate; cannot form a matched wall ratio"
            }
        if conc_sum_of_rates:
            scal["decode_vs_decode"] = {
                "bs1_decode_tok_s": bs1_tok_s,
                "conc_sum_of_decode_rates_tok_s": conc_sum_of_rates,
                "scaling_factor": round(conc_sum_of_rates / bs1_tok_s, 3),
                "efficiency_vs_linear": round(conc_sum_of_rates / (bs1_tok_s * conc), 3),
                "basis": "both sides are server-side decode-only rates, prefill excluded",
            }
        scal["MISMATCHED_do_not_cite"] = {
            "conc_wall_over_bs1_decode": round(conc_tok_s / bs1_tok_s, 3),
            "note": "retained only to show what the naive ratio would have been",
        }
        d["conc_scaling"] = scal
    return d


def verdict_text(doc: dict) -> str:
    d = doc.get("derived", {})
    bs1 = d.get("measured_bs1_tok_s")
    if bs1 is None:
        return "INCOMPLETE: no valid bs=1 median; P0 has not set a bar."
    parts = [
        f"P0 baseline: llama.cpp bs=1 decode = {bs1:.3f} tok/s on the target checkpoint."
    ]
    if not d.get("plan_derived_ceiling_reachable_in_practice", False):
        parts.append(
            f"The plan's DERIVED 33.8 tok/s CPU-expert ceiling is NOT reachable in practice: "
            f"measured is {d.get('shortfall_factor_vs_plan_ceiling')}x below it. Break-even "
            f"math must use the measured number, not 33.8."
        )
    else:
        parts.append("The plan's derived 33.8 tok/s ceiling IS approached in practice.")
    be0 = d.get("break_even_hit_rate_vs_measured_baseline", {}).get("t_compute_5ms", {})
    if be0.get("already_satisfied_at_h_0"):
        parts.append(
            "Against this baseline, a pure zero-cache host-streaming tier already wins: the "
            "required byte hit rate h is 0. The 40% break-even in Section 1(a) is an artifact "
            "of comparing to a 45 GB/s DDR abstraction llama.cpp does not achieve here."
        )
    elif be0.get("required_h_pct") is not None:
        parts.append(
            f"Required byte hit rate to beat this baseline (5 ms compute floor): "
            f"h >= {be0['required_h_pct']:.1f}%."
        )
    parts.append(f"K4 hard-kill threshold for M1: {d.get('K4_hard_kill_threshold_tok_s')} tok/s.")
    ca = d.get("measured_conc_aggregate_tok_s")
    if ca is not None:
        ww = (d.get("conc_scaling") or {}).get("wall_vs_wall") or {}
        if ww.get("scaling_factor") is not None:
            parts.append(
                f"CONC={d.get('conc_concurrency')} aggregate = {ca:.3f} tok/s wall "
                f"({ww['scaling_factor']}x the bs=1 WALL rate, "
                f"{100 * ww['efficiency_vs_linear']:.0f}% of linear; matched bases)."
            )
        else:
            parts.append(
                f"CONC={d.get('conc_concurrency')} aggregate = {ca:.3f} tok/s wall "
                f"(no matched bs=1 wall rate available, so no scaling ratio is quoted)."
            )
    ge = doc.get("gpu_engagement", {})
    v = ge.get("verdict")
    if v and v != "confirmed":
        parts.append(
            f"ATTRIBUTION {v.upper()}: {ge.get('why')} Treat every number above as "
            f"NOT attributable to the 2-card --fit configuration."
        )
    unsteady = [n for n, lg in (doc.get("legs") or {}).items() if lg.get("steady_state") is False]
    if unsteady:
        parts.append(
            f"NOT STEADY STATE in leg(s) {', '.join(unsteady)}: the reps trend rather than "
            f"fluctuate, so the median is a point on a ramp."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# JSON shape validation -- exercised by --selftest
# --------------------------------------------------------------------------- #
REQUIRED_SHAPE = {
    "probe": str,
    "probe_name": str,
    "schema_version": int,
    "status": str,
    "timestamp_utc": str,
    "hostname": str,
    "cards_used": list,
    "cited_prior": dict,
    "provenance": dict,
    "box_state": dict,
    "legs": dict,
    "derived": dict,
    "verdict": str,
    "gpu_engagement": dict,
}
REQUIRED_PROVENANCE = ("git", "model", "runtime", "env", "launch", "python")
REQUIRED_BOX_STATE = ("before", "after", "delta")


def validate_shape(doc: dict) -> list:
    errs = []
    for k, t in REQUIRED_SHAPE.items():
        if k not in doc:
            errs.append(f"missing top-level key: {k}")
        elif not isinstance(doc[k], t):
            errs.append(f"key {k}: expected {t.__name__}, got {type(doc[k]).__name__}")
    for k in REQUIRED_PROVENANCE:
        if k not in doc.get("provenance", {}):
            errs.append(f"missing provenance.{k}")
    for k in REQUIRED_BOX_STATE:
        if k not in doc.get("box_state", {}):
            errs.append(f"missing box_state.{k}")
    for name, leg in doc.get("legs", {}).items():
        if not isinstance(leg, dict):
            errs.append(f"leg {name} is not a dict")
            continue
        for k in ("kind", "concurrency", "n_predict", "reps_requested", "reps_valid", "reps"):
            if k not in leg:
                errs.append(f"leg {name}: missing {k}")
        # A short rep count is a DEGRADED result (exit 7), not a malformed
        # document (exit 6).  It is only a schema error if the document also
        # failed to admit it -- i.e. it claims `status: ok` on thin data.
        if (
            leg.get("reps_valid", 0) < MIN_REPS
            and not doc.get("reps_below_minimum")
            and doc.get("status") not in ("degraded",)
            and "SELFTEST" not in str(doc.get("status", ""))
        ):
            errs.append(
                f"leg {name}: reps_valid={leg.get('reps_valid')} < {MIN_REPS} while "
                f"status={doc.get('status')!r} and reps_below_minimum was not stamped"
            )
    if doc.get("cited_prior", {}).get("measured_by_this_script") is not False:
        errs.append("cited_prior.measured_by_this_script must be exactly False")
    ge = doc.get("gpu_engagement", {})
    if not isinstance(ge, dict) or "verdict" not in ge:
        errs.append("gpu_engagement.verdict is missing; GPU attribution is mandatory")
    elif ge["verdict"] != "confirmed" and doc.get("status") == "ok":
        errs.append(
            f"gpu_engagement.verdict={ge['verdict']!r} but status is 'ok'; a run that cannot "
            f"be attributed to the 2-card --fit configuration must never present as clean"
        )
    try:
        json.dumps(doc)
    except (TypeError, ValueError) as e:
        errs.append(f"document is not JSON-serialisable: {e}")
    return errs


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def write_markdown(doc: dict, path: Path) -> None:
    d = doc.get("derived", {})
    legs = doc.get("legs", {})
    bs1 = legs.get("bs1", {})
    concleg = next((v for k, v in legs.items() if k.startswith("conc")), {})
    bx = doc.get("box_state", {})
    L = []
    A = L.append
    A(f"# P0 — llama.cpp baseline on the target checkpoint")
    A("")
    A(f"**Probe:** P0 (WEIGHT_OFFLOAD_PLAN.md §3). **Status:** `{doc['status']}`. "
      f"**Run:** {doc['timestamp_utc']} on `{doc['hostname']}`.")
    A("")
    A("> This is the denominator of the entire weight-offload project. Gates **K4** "
      "(`M1 tok/s < 0.64 × llama.cpp` → hard kill) and **A0.4** (`byte hit rate ≥ "
      "max(50 %, break-even vs P0)`) both read the number below.")
    A("")
    A("## Verdict")
    A("")
    A(doc.get("verdict", "(none)"))
    A("")
    A("## Measured")
    A("")
    A("| Leg | Metric | Median | Min | Max | Spread % | Valid reps |")
    A("|---|---|---|---|---|---|---|")
    if bs1:
        s = bs1.get("decode_tok_s", {})
        A(f"| bs=1 | decode tok/s | {s.get('median')} | {s.get('min')} | {s.get('max')} | "
          f"{s.get('spread_pct_of_median')} | {bs1.get('reps_valid')}/{bs1.get('reps_requested')} |")
        p = bs1.get("prompt_tok_s", {})
        A(f"| bs=1 | prompt tok/s | {p.get('median')} | {p.get('min')} | {p.get('max')} | "
          f"{p.get('spread_pct_of_median')} | {bs1.get('reps_valid')}/{bs1.get('reps_requested')} |")
        w = bs1.get("wall_tok_s", {})
        A(f"| bs=1 | wall tok/s (prefill incl.) | {w.get('median')} | {w.get('min')} | "
          f"{w.get('max')} | {w.get('spread_pct_of_median')} | "
          f"{bs1.get('reps_valid')}/{bs1.get('reps_requested')} |")
    if concleg:
        s = concleg.get("aggregate_tok_s_wall", {})
        c = concleg.get("concurrency")
        A(f"| CONC={c} | aggregate tok/s (wall) | {s.get('median')} | {s.get('min')} | "
          f"{s.get('max')} | {s.get('spread_pct_of_median')} | "
          f"{concleg.get('reps_valid')}/{concleg.get('reps_requested')} |")
        s2 = concleg.get("aggregate_decode_tok_s_sum_of_rates", {})
        A(f"| CONC={c} | aggregate tok/s (Σ server decode rates) | {s2.get('median')} | "
          f"{s2.get('min')} | {s2.get('max')} | {s2.get('spread_pct_of_median')} | "
          f"{concleg.get('reps_valid')}/{concleg.get('reps_requested')} |")
        s3 = concleg.get("per_stream_decode_tok_s", {})
        A(f"| CONC={c} | per-stream decode tok/s | {s3.get('median')} | {s3.get('min')} | "
          f"{s3.get('max')} | {s3.get('spread_pct_of_median')} | {s3.get('n')} streams |")
    A("")
    A("One warm-up repetition was discarded in every leg. `ignore_eos` forces exactly "
      "`n_predict` tokens per rep so reps are comparable; a fresh 16-byte nonce at token ~8 of "
      "every prompt defeats prefix reuse, `cache_prompt:false` is set per request, and the "
      "server-side prompt cache is disabled with `--cache-ram "
      f"{doc['provenance']['launch'].get('sampling', {}).get('server_cache_ram_mib')}`. "
      "Whether the reps are actually steady state is checked, not assumed — see below.")
    A("")
    A("## Does the plan's derived 33.8 tok/s ceiling hold?")
    A("")
    A(f"- Plan §1 lists **33.8 tok/s** as the *llama.cpp CPU-expert ceiling*, explicitly "
      f"marked **\"derived — must be measured\"** (45 GB/s DDR ÷ 1.33 GB/token).")
    A(f"- Measured here: **{d.get('measured_bs1_tok_s')} tok/s**, i.e. "
      f"**{d.get('fraction_of_plan_derived_ceiling')}×** the derived ceiling "
      f"({d.get('shortfall_factor_vs_plan_ceiling')}× short).")
    A(f"- Reachable in practice: **{d.get('plan_derived_ceiling_reachable_in_practice')}**.")
    A("")
    A("The derived ceiling assumes the CPU runtime is DDR-bandwidth-bound at 45 GB/s. Whether "
      "it is, on this box, is settled by the measured I/O attribution below — not asserted "
      "here. What is certain either way: any break-even or K4 arithmetic that uses 33.8 is "
      "wrong by the shortfall factor above.")
    A("")
    A("### Measured I/O attribution — what actually bound the baseline")
    A("")
    A("| Leg | tokens generated | server read GB / token (per-PID) | nvme read GB / token (box-wide) | fraction of THIS checkpoint's expert bytes | major faults / token |")
    A("|---|---|---|---|---|---|")
    for name, leg in legs.items():
        io = leg.get("io_attribution") or {}
        A(f"| {name} | {io.get('tokens_generated_incl_warmup')} | "
          f"{io.get('server_read_gb_per_generated_token')} | "
          f"{io.get('nvme_read_gb_per_generated_token_boxwide')} | "
          f"{io.get('fraction_of_this_checkpoints_expert_bytes_from_storage')} | "
          f"{io.get('major_faults_per_generated_token_boxwide')} |")
    A("")
    A("Read this as the discriminator between the two candidate explanations of a low number: "
      "**CPU-compute bound** (i-quant expert GEMV on 8 cores) vs **storage bound** (a "
      f"{doc['provenance']['model']['total_gb']} GB checkpoint mmap'd on a box with less RAM "
      "than that, through a capped ZFS ARC). A near-zero read-per-token says compute; a "
      "fraction approaching 1 says the expert bytes are coming off disk every token.")
    A("")
    A("The denominator is this checkpoint's OWN active expert bytes per token, **not** the "
      "plan's 1.33 GB — that constant is arithmetic on the minisgl target shape "
      "(48L × 10 × 2.8 MB) and a different packing from UD-IQ4_XS, so a ratio against it would "
      "compare across quantizations. The plan-constant ratio is kept in the JSON as "
      "`fraction_of_PLAN_constant_1p33GB_NOT_THIS_PACKING` for contrast only. `server read` is "
      "process-attributed (`/proc/<pid>/io`); `nvme read` is box-wide and includes any other "
      "job's traffic in the same window.")
    A("")
    _gg = ((doc.get("legs") or {}).get("bs1") or {}).get("expert_bytes_per_token") or {}
    _g = _gg.get("gguf_geometry") or {}
    if _g.get("ok"):
        A(f"**Denominator provenance ({_gg.get('estimate_source')}).** This llama.cpp build "
          "emits no `print_info:`/`load_tensors:` lines, so the geometry is read from the "
          "checkpoint itself: an exact walk of the GGUF tensor table summing every `*_exps` "
          f"tensor at its own ggml type size — **{_g['expert_tensor_bytes_gb']} GB** of expert "
          f"weights ({_g['expert_fraction_of_checkpoint']*100:.1f}% of the checkpoint) over "
          f"{_g['moe_layers_seen']} MoE layers, {_g['n_expert']} experts, top-"
          f"{_g['n_expert_used']} routed → **{_g['active_expert_bytes_per_token_gb']} GB per "
          "decoded token**. The walk is validated on the operation, not the query: it "
          f"reconstructs {_g['walk_vs_on_disk_ratio']*100:.2f}% of the on-disk shard bytes "
          f"({_g['on_disk_gb']} GB) and is rejected outside ±1 %. Expert byte breakdown by "
          f"quant type (GB): {_g.get('expert_bytes_by_ggml_type_gb')}."
          + (
              f" The cruder pro-rata bound (`total × k/E`) would have said "
              f"{_gg['pro_rata_upper_bound_gb']} GB/token — "
              f"{_gg['pro_rata_upper_bound_gb'] / _g['active_expert_bytes_per_token_gb']:.2f}× "
              "too high, because it charges attention/embedding/PLE bytes to the expert term."
              if _gg.get("pro_rata_upper_bound_gb") else ""
          ))
    elif _g:
        A(f"**Denominator provenance.** GGUF tensor-table walk unavailable "
          f"(`{_g.get('reject_reason')}`); fell back to `{_gg.get('estimate_source')}`.")
    A("")
    A("## Attribution — was this actually the 2-card `--fit` configuration?")
    A("")
    ge = doc.get("gpu_engagement", {})
    A(f"**Verdict:** `{ge.get('verdict')}`. {ge.get('why')}")
    A("")
    A("| Leg | log ROCm weight GB | log CPU weight GB | layers offloaded | VRAM delta across load (GB) | devices enumerated |")
    A("|---|---|---|---|---|---|")
    for k, ev in (ge.get("per_leg") or {}).items():
        A(f"| {k} | {ev.get('gpu_weight_gb_from_log')} | {ev.get('cpu_weight_gb_from_log')} | "
          f"{ev.get('layers_offloaded')}/{ev.get('layers_total')} | "
          f"{ev.get('vram_delta_gb_all_compute_cards')} | "
          f"{ev.get('device_count_reported')} |")
    A("")
    A("A plausible tok/s number does not distinguish the claimed configuration from a run "
      "where the HIP backend failed to register, where `--fit` offloaded nothing, or where the "
      "Ryzen iGPU (47 GB of GTT) leaked into enumeration and poisoned `--fit` auto-sizing. "
      "This section asserts on the **operation** — weight bytes on device and VRAM that "
      "actually moved — not on the flags having been passed.")
    A("")
    A("## Steady state — is the median a rate or a point on a ramp?")
    A("")
    A("| Leg | steady state | 1st-half mean | 2nd-half mean | drift % | strictly monotone | reps in order |")
    A("|---|---|---|---|---|---|---|")
    for name, leg in legs.items():
        key = "decode_tok_s" if "decode_tok_s" in leg else "aggregate_tok_s_wall"
        st = leg.get(key) or {}
        dr = st.get("drift") or {}
        A(f"| {name} ({key}) | {leg.get('steady_state')} | {dr.get('first_half_mean')} | "
          f"{dr.get('second_half_mean')} | {dr.get('drift_pct_second_vs_first_half')} | "
          f"{dr.get('strictly_monotone')} | {st.get('values')} |")
    A("")
    A("One 100-token warm-up cannot warm a 93.7 GB working set on a box with ~50 GB available, "
      "so this table is load-bearing: a trending series means the median is a point on a curve "
      "and every gate that reads it inherits the error.")
    A("")
    A("## Break-even hit rate, recomputed against the measured baseline")
    A("")
    A("| Compute floor | Required byte hit rate `h` to beat llama.cpp |")
    A("|---|---|")
    for k, v in (d.get("break_even_hit_rate_vs_measured_baseline") or {}).items():
        if v.get("already_satisfied_at_h_0"):
            val = "**0 % — beaten with no cache at all**"
        elif v.get("required_h_pct") is not None:
            val = f"{v['required_h_pct']:.1f} %"
        else:
            val = v.get("note", "n/a")
        A(f"| {k.replace('t_compute_', '').replace('ms', ' ms')} | {val} |")
    A("")
    bp = d.get("break_even_vs_plan_ddr_abstraction", {})
    A(f"For contrast, the plan's §1(a) figure — computed against a 45 GB/s DDR abstraction "
      f"rather than a measured runtime — is **{bp.get('required_h_pct')} %**.")
    A("")
    A(f"**K4 hard-kill threshold for M1:** {d.get('K4_hard_kill_threshold_tok_s')} tok/s at bs=1.")
    A("")
    sc = d.get("conc_scaling") or {}
    if sc:
        A("## Concurrency scaling — matched bases only")
        A("")
        A("| Comparison | bs=1 | CONC | ×bs=1 | % of linear |")
        A("|---|---|---|---|---|")
        ww = sc.get("wall_vs_wall") or {}
        if "scaling_factor" in ww:
            A(f"| wall vs wall (prefill included) | {ww.get('bs1_wall_tok_s')} | "
              f"{ww.get('conc_wall_tok_s')} | {ww.get('scaling_factor')} | "
              f"{100 * ww['efficiency_vs_linear']:.0f} % |")
        dv = sc.get("decode_vs_decode") or {}
        if "scaling_factor" in dv:
            A(f"| decode vs Σ decode rates (prefill excluded) | {dv.get('bs1_decode_tok_s')} | "
              f"{dv.get('conc_sum_of_decode_rates_tok_s')} | {dv.get('scaling_factor')} | "
              f"{100 * dv['efficiency_vs_linear']:.0f} % |")
        mm = sc.get("MISMATCHED_do_not_cite") or {}
        A("")
        A(f"The naive `CONC wall ÷ bs=1 decode` ratio would have been "
          f"**{mm.get('conc_wall_over_bs1_decode')}** — it divides a prefill-inclusive "
          f"aggregate by a prefill-exclusive rate and is recorded only so it is not "
          f"mistaken for a result.")
        A("")
    A("## Box state — recorded with every leg (the box was NOT idle)")
    A("")
    A("| Window | Δ major faults | Δ pswpout | Δ pswpin | Δ nvme read (GB) | avg nvme read GB/s | Δ MemAvailable (GB) |")
    A("|---|---|---|---|---|---|---|")
    for name, dd in (bx.get("delta") or {}).items():
        A(f"| {name} | {dd.get('d_pgmajfault')} | {dd.get('d_pswpout')} | {dd.get('d_pswpin')} | "
          f"{dd.get('d_nvme_read_gb')} | {dd.get('nvme_read_gbs_avg')} | {dd.get('d_MemAvailable_gb')} |")
    A("")
    mb = (bx.get("before") or {}).get("meminfo", {}).get("gb", {})
    A(f"At probe start: MemTotal {mb.get('MemTotal_gb')} GB, MemAvailable "
      f"{mb.get('MemAvailable_gb')} GB, SwapFree {mb.get('SwapFree_gb')} GB. "
      f"Checkpoint on disk: {doc['provenance']['model']['total_gb']} GB across 3 shards "
      f"(**larger than installed RAM**), on a ZFS pool with a "
      f"{round(((bx.get('before') or {}).get('zfs_arcstats', {}).get('c_max') or 0) / 2**30, 1)} GiB ARC cap.")
    A("")
    A("## Cards")
    A("")
    for c in doc.get("cards_used", []):
        A(f"- {c}")
    A("")
    A("## Prior figure carried as a citation only")
    A("")
    cp = doc["cited_prior"]
    A(f"- Cited: **{cp['value_tok_s']} tok/s** decode, **{cp['prompt_tok_s']} tok/s** prompt.")
    A(f"- Source: {cp['source']}")
    A(f"- Provenance quality: {cp['provenance_quality']}")
    A(f"- Measured by this script: `{cp['measured_by_this_script']}`.")
    if "reproduced_by_this_run" in cp:
        A(f"- Reproduced by this run: **{cp['reproduced_by_this_run']}** "
          f"(fresh median {cp.get('fresh_median_tok_s')} tok/s, "
          f"{cp.get('delta_pct')} % from the cited value).")
    A("")
    A("## Reproduce")
    A("")
    A("```")
    A(doc["provenance"]["launch"].get("probe_command", ""))
    A("```")
    A("")
    A("llama-server launch lines (one per leg):")
    A("")
    A("```")
    for name, argv in (doc["provenance"]["launch"].get("server_argv") or {}).items():
        A(f"# {name}")
        A(" ".join(argv))
    A("```")
    A("")
    A(f"Raw data: `p0.json` (schema {doc['schema_version']}). "
      f"Server logs: `p0_server_*.log`.")
    A("")
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #
def check_preconditions(a: argparse.Namespace, gpu_checks: bool) -> dict:
    notes = {}
    if os.environ.get("ROCR_VISIBLE_DEVICES") != "0,1":
        raise PreconditionError(
            f"ROCR_VISIBLE_DEVICES is {os.environ.get('ROCR_VISIBLE_DEVICES')!r}, expected '0,1'. "
            "This script sets it at import; something overwrote it."
        )
    if "HIP_VISIBLE_DEVICES" in os.environ:
        raise PreconditionError(
            "HIP_VISIBLE_DEVICES is set. It must be UNSET so ROCR alone fences the iGPU."
        )
    if not LLAMA_SERVER.exists() or not os.access(LLAMA_SERVER, os.X_OK):
        raise PreconditionError(f"llama-server missing or not executable: {LLAMA_SERVER}")
    missing = [str(p) for p in MODEL_SHARDS if not p.exists()]
    if missing:
        raise PreconditionError("GGUF shard(s) missing:\n  " + "\n  ".join(missing))
    total = sum(p.stat().st_size for p in MODEL_SHARDS)
    if total < 80e9:
        raise PreconditionError(
            f"GGUF total is {total / 1e9:.1f} GB; expected ~93.7 GB. Wrong or truncated checkpoint."
        )
    notes["model_total_gb"] = round(total / 1e9, 3)
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        probe = RESULTS_DIR / ".p0_write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as e:
        raise PreconditionError(f"results dir not writable ({RESULTS_DIR}): {e}") from e
    if RESULTS_DIR.is_symlink() or str(RESULTS_DIR).startswith(("/tmp", "/dev/shm", "/run")):
        raise PreconditionError(
            f"results dir {RESULTS_DIR} is not durable worktree storage. Fixtures must be durable."
        )
    if a.reps < MIN_REPS and not a.allow_fewer_reps:
        raise PreconditionError(
            f"--reps {a.reps} < {MIN_REPS}. Median+spread needs >= {MIN_REPS} steady-state reps. "
            f"Pass --allow-fewer-reps to stamp the result as degraded."
        )
    if a.conc < 1:
        raise PreconditionError(f"--conc {a.conc} must be >= 1")
    if a.n_predict < 16:
        raise PreconditionError(f"--n-predict {a.n_predict} too small to time a decode rate")
    if a.ctx_total < a.conc * (a.n_predict + 512):
        raise PreconditionError(
            f"--ctx-total {a.ctx_total} cannot hold {a.conc} slots x "
            f"~{a.n_predict + 512} tokens; raise it or lower --conc."
        )
    if not port_is_free(a.port):
        raise PreconditionError(
            f"port {a.port} is already bound. Another llama-server is probably running; "
            f"stop it or pass --port."
        )

    if gpu_checks:
        cards = rocm_smi_cards()
        if "_error" in cards:
            raise PreconditionError(f"rocm-smi unusable: {cards['_error']}")
        notes["cards_raw"] = cards
        compute = {d: r for d, r in cards.items() if r.get("is_compute_card")}
        if len(compute) != 2:
            raise PreconditionError(
                f"expected exactly 2 discrete gfx1201 compute cards, rocm-smi classified "
                f"{len(compute)}: {[(d, r.get('name')) for d, r in cards.items()]}. "
                f"Without both cards identified, the busy-card gate is vacuous and the "
                f"'which physical card' record the repo rule requires cannot be written."
            )
        notes["compute_cards"] = {d: r.get("name") for d, r in compute.items()}
        busy = []
        for dev, rec in compute.items():
            used = rec.get("vram_used_gb")
            if used is not None and used > a.vram_busy_gb:
                busy.append(f"{dev} ({rec.get('name')}) has {used} GB VRAM in use")
        if busy and not a.force:
            raise PreconditionError(
                "GPU(s) appear busy — another job is holding VRAM:\n  "
                + "\n  ".join(busy)
                + f"\nA baseline measured against a contended card is worthless. "
                f"Serialise the run, or pass --force with the contention on the record."
            )
        if busy:
            notes["forced_over_busy_gpu"] = busy
    return notes


# --------------------------------------------------------------------------- #
# Selftest -- fully GPU-free
# --------------------------------------------------------------------------- #
def synth_rep(rate: float, n: int, seed: int, prompt_ms: float = 5700.0) -> dict:
    """A synthetic rep whose WALL includes prefill, exactly as a real one does.

    This is what makes the matched-basis check in the selftest meaningful: if the
    wall and the decode-only rate were identical, the mismatched ratio the script
    now refuses to headline would look correct.
    """
    t0 = 1.0e9 + seed
    wall = n / rate + prompt_ms / 1000.0
    return {
        "ok": True,
        "t_start": t0,
        "t_end": t0 + wall,
        "wall_s": round(wall, 3),
        "predicted_n": n,
        "predicted_ms": 1000.0 * n / rate,
        "predicted_per_second": rate,
        "prompt_n": 64,
        "prompt_ms": prompt_ms,
        "prompt_per_second": round(64.0 / (prompt_ms / 1000.0), 3),
        "stop_type": "limit",
        "truncated": False,
    }


def run_selftest(a: argparse.Namespace) -> int:
    print("=== P0 SELFTEST (no GPU, no server, no model read) ===", flush=True)
    failures = []

    # 1. Device fencing is applied by import.
    if os.environ.get("ROCR_VISIBLE_DEVICES") != "0,1":
        failures.append("ROCR_VISIBLE_DEVICES not fenced to 0,1")
    if "HIP_VISIBLE_DEVICES" in os.environ:
        failures.append("HIP_VISIBLE_DEVICES not unset")
    print(f"[1] device fence: ROCR={os.environ.get('ROCR_VISIBLE_DEVICES')} "
          f"HIP={os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")

    # 2. Argument validation must REJECT bad input -- and reject it for the RIGHT
    #    REASON.  A bare "did it raise?" test goes green when an unrelated
    #    precondition (a bound port, a missing shard) fires on every case, which
    #    is exactly when the argument checks most need to be trusted.
    def expect_reject(mutate, why, expect_substr):
        ns = argparse.Namespace(**vars(a))
        mutate(ns)
        try:
            check_preconditions(ns, gpu_checks=False)
        except PreconditionError as e:
            if expect_substr.lower() not in str(e).lower():
                failures.append(
                    f"precondition rejected {why!r} for the WRONG reason "
                    f"(expected {expect_substr!r}, got: {e})"
                )
                return False
            return True
        failures.append(f"precondition check accepted bad input: {why}")
        return False

    rejected = [
        expect_reject(lambda n: setattr(n, "reps", 2),
                      "--reps 2 (< 5) without --allow-fewer-reps", "--reps 2"),
        expect_reject(lambda n: setattr(n, "conc", 0), "--conc 0", "--conc 0"),
        expect_reject(lambda n: setattr(n, "n_predict", 4), "--n-predict 4", "n-predict"),
        expect_reject(lambda n: setattr(n, "ctx_total", 128),
                      "--ctx-total too small for the slots", "ctx-total"),
    ]
    print(f"[2] argument rejection: {sum(rejected)}/{len(rejected)} bad inputs rejected "
          f"for the right reason")

    # 2b. rocm-smi parsing must survive a stdout preamble and must classify the
    #     iGPU out, and must NOT return an empty dict that silently disarms the
    #     busy-card gate.
    if _is_compute_card("card2", "AMD Ryzen 7 7800X3D 8-Core Processor"):
        failures.append("_is_compute_card classified the Ryzen iGPU as a compute card")
    if not _is_compute_card("card0", "AMD Radeon RX 9070 XT"):
        failures.append("_is_compute_card rejected a discrete RX 9070 XT")
    if _is_compute_card("card1", "AMD Radeon Graphics"):
        failures.append("_is_compute_card accepted an ambiguous iGPU-style name")
    print("[2b] card classification: iGPU excluded, discrete cards accepted")

    # 2c. GPU attribution must not call a CPU-only run 'confirmed'.
    cpu_only = attribute_gpu_engagement(
        {"gpu_weight_gb": 0.0, "cpu_weight_gb": 93.0, "devices": ["Device 0: AMD Radeon RX 9070 XT, gfx1201"],
         "device_count_reported": 2, "igpu_in_enumeration": False, "model_geometry": {}},
        {"card0": 0.0, "card1": 0.0},
        1.0,
    )
    if cpu_only["verdict"] != "cpu_only":
        failures.append(f"attribution called a CPU-only run {cpu_only['verdict']!r}")
    leaked = attribute_gpu_engagement(
        {"gpu_weight_gb": 40.0, "cpu_weight_gb": 50.0,
         "devices": ["Device 0: AMD Radeon RX 9070 XT, gfx1201",
                     "Device 1: AMD Radeon RX 9070, gfx1201",
                     "Device 2: AMD Radeon Graphics, gfx1103"],
         "device_count_reported": 3, "igpu_in_enumeration": True, "model_geometry": {}},
        {"card0": 10.0, "card1": 10.0},
        1.0,
    )
    if leaked["verdict"] != "igpu_leak":
        failures.append(f"attribution missed an iGPU leak: {leaked['verdict']!r}")
    good = attribute_gpu_engagement(
        {"gpu_weight_gb": 26.0, "cpu_weight_gb": 67.0,
         "devices": ["Device 0: AMD Radeon RX 9070 XT, gfx1201",
                     "Device 1: AMD Radeon RX 9070, gfx1201"],
         "device_count_reported": 2, "igpu_in_enumeration": False, "model_geometry": {}},
        {"card0": 13.1, "card1": 12.9},
        1.0,
    )
    if good["verdict"] != "confirmed":
        failures.append(f"attribution failed to confirm a good run: {good['verdict']!r}")
    unknown = attribute_gpu_engagement(
        {"gpu_weight_gb": None, "cpu_weight_gb": None, "devices": [],
         "device_count_reported": None, "igpu_in_enumeration": None, "model_geometry": {}},
        {},
        1.0,
    )
    if unknown["verdict"] != "unconfirmed":
        failures.append(f"attribution fabricated a verdict from no evidence: {unknown['verdict']!r}")
    print("[2c] gpu attribution: cpu_only / igpu_leak / confirmed / unconfirmed all correct")

    # 2d. Drift detection must catch a ramp and pass a genuine steady state.
    ramp = drift_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    if ramp.get("steady_state") is not False or not ramp.get("strictly_monotone"):
        failures.append(f"drift_stats accepted a monotone ramp as steady state: {ramp}")
    flat = drift_stats([4.58, 4.51, 4.62, 4.55, 4.60])
    if flat.get("steady_state") is not True:
        failures.append(f"drift_stats rejected a genuine steady state: {flat}")
    if summarize([3.0, 1.0, 2.0])["values"] != [3.0, 1.0, 2.0]:
        failures.append("summarize() sorted `values`; rep order must be preserved for drift")
    print(f"[2d] drift: ramp steady={ramp.get('steady_state')} "
          f"flat steady={flat.get('steady_state')}; rep order preserved")

    # 3. Good input must be ACCEPTED (non-GPU checks only).
    try:
        check_preconditions(a, gpu_checks=False)
        print("[3] good arguments accepted; model shards + binary + results dir all present")
    except PreconditionError as e:
        failures.append(f"precondition check rejected valid setup: {e}")
        print(f"[3] FAILED: {e}")

    # 4. Statistics: median/spread over synthetic reps.
    s = summarize([4.5, 4.6, 4.58, 4.62, 4.55])
    if s["n"] != 5 or abs(s["median"] - 4.58) > 1e-9:
        failures.append(f"summarize() wrong: {s}")
    print(f"[4] summarize(): median={s['median']} spread={s['spread_pct_of_median']}% n={s['n']}")

    # 5. Full document assembly + shape validation on synthetic legs.
    n = a.n_predict
    bs1_reps = [synth_rep(r, n, i) for i, r in enumerate((4.51, 4.60, 4.58, 4.62, 4.55))]
    bs1 = {
        "kind": "bs1",
        "concurrency": 1,
        "n_predict": n,
        "reps_requested": 5,
        "reps_valid": 5,
        "warmup_discarded": synth_rep(3.9, n, 99),
        "reps": bs1_reps,
        "decode_tok_s": summarize([r["predicted_per_second"] for r in bs1_reps]),
        "prompt_tok_s": summarize([r["prompt_per_second"] for r in bs1_reps]),
        "wall_s": summarize([r["wall_s"] for r in bs1_reps]),
        "wall_tok_s": summarize([n / r["wall_s"] for r in bs1_reps]),
        "prompt_cache_check": prompt_cache_check(bs1_reps),
    }
    bs1["steady_state"] = bool((bs1["decode_tok_s"].get("drift") or {}).get("steady_state", True))
    cbursts = []
    for i in range(5):
        streams = [synth_rep(1.7 + 0.02 * j, n, 500 + 10 * i + j) for j in range(a.conc)]
        wall = n * a.conc / (9.8 + 0.05 * i)
        cbursts.append(
            {
                "streams": streams,
                "streams_valid": a.conc,
                "wall_s": round(wall, 3),
                "tokens_generated": n * a.conc,
                "aggregate_tok_s_wall": round(n * a.conc / wall, 4),
                "aggregate_decode_tok_s_sum_of_rates": round(
                    sum(s["predicted_per_second"] for s in streams), 4
                ),
                "per_stream_decode_tok_s": summarize(
                    [s["predicted_per_second"] for s in streams]
                ),
                "per_stream_prompt_ms": summarize([s["prompt_ms"] for s in streams]),
            }
        )
    conc = {
        "kind": f"conc{a.conc}",
        "concurrency": a.conc,
        "n_predict": n,
        "reps_requested": 5,
        "reps_valid": 5,
        "warmup_discarded": cbursts[0],
        "reps": cbursts,
        "aggregate_tok_s_wall": summarize([b["aggregate_tok_s_wall"] for b in cbursts]),
        "aggregate_decode_tok_s_sum_of_rates": summarize(
            [b["aggregate_decode_tok_s_sum_of_rates"] for b in cbursts]
        ),
        "per_stream_decode_tok_s": summarize(
            [s["predicted_per_second"] for b in cbursts for s in b["streams"]]
        ),
        "prompt_cache_check": prompt_cache_check(
            [s for b in cbursts for s in b["streams"]]
        ),
    }
    conc["steady_state"] = bool(
        (conc["aggregate_tok_s_wall"].get("drift") or {}).get("steady_state", True)
    )
    before = box_state(with_gpu=False)
    after = box_state(with_gpu=False)
    # Exercise the I/O-attribution path with a synthetic, obviously-fake delta.
    synth_delta = {"d_nvme_read_gb": 123.4, "d_pgmajfault": 456789, "window_s": 110.0}
    synth_pio = {"d_read_bytes": 88_000_000_000, "d_read_gb": 88.0, "d_rchar": 99}
    # (a) fallback path: GGUF walk unavailable -> server-log pro-rata upper bound.
    synth_eb_fallback = expert_bytes_per_token_gb(
        {
            "model_geometry": {"n_expert": 512, "n_expert_used": 10, "model_size_gb": 93.68},
            "_gguf_geometry": {"ok": False, "reject_reason": "<selftest: walk suppressed>"},
        }
    )
    if synth_eb_fallback.get("estimate_source") != "server_log_pro_rata_upper_bound":
        failures.append("expert_bytes_per_token_gb did not fall back to the log pro-rata bound")
    # (b) preferred path: the REAL checkpoint's tensor table.  This reads only
    #     the GGUF headers/tensor tables (a few MB), never the weights.
    gg = gguf_geometry_cached()
    if not gg.get("ok"):
        failures.append(f"gguf_expert_geometry rejected the checkpoint: {gg.get('reject_reason')}")
    else:
        ratio = gg.get("walk_vs_on_disk_ratio")
        if not (0.99 <= (ratio or 0) <= 1.01):
            failures.append(f"gguf walk did not reconstruct the on-disk size (ratio {ratio})")
        if gg["active_expert_bytes_per_token_gb"] >= synth_eb_fallback["estimate_gb"]:
            failures.append(
                "exact expert bytes should be BELOW the pro-rata upper bound "
                f"({gg['active_expert_bytes_per_token_gb']} vs "
                f"{synth_eb_fallback['estimate_gb']})"
            )
        print(
            f"[4a] gguf geometry: {gg['n_expert']} experts, top-{gg['n_expert_used']}, "
            f"{gg['moe_layers_seen']} MoE layers, exps={gg['expert_tensor_bytes_gb']} GB "
            f"({gg['expert_fraction_of_checkpoint']*100:.1f}% of ckpt), walk reconstructs "
            f"{ratio*100:.2f}% of on-disk -> {gg['active_expert_bytes_per_token_gb']} GB/token "
            f"(pro-rata upper bound was {synth_eb_fallback['estimate_gb']})"
        )
    synth_eb = expert_bytes_per_token_gb(
        {"model_geometry": {"n_expert": 512, "n_expert_used": 10, "model_size_gb": 93.68}}
    )
    if synth_eb.get("estimate_gb") is None:
        failures.append("expert_bytes_per_token_gb failed on complete geometry")
    annotate_leg_io(bs1, synth_delta, synth_eb, synth_pio)
    annotate_leg_io(conc, synth_delta, synth_eb, synth_pio)
    if "server_read_gb_per_generated_token" not in bs1.get("io_attribution", {}):
        failures.append("annotate_leg_io did not attribute per-process reads for the bs1 leg")
    if bs1["io_attribution"].get("this_checkpoint_expert_bytes_per_token_gb") is None:
        failures.append("annotate_leg_io used no same-packing expert-bytes denominator")
    if conc["io_attribution"]["tokens_generated_incl_warmup"] != n * a.conc * 6:
        failures.append(
            f"annotate_leg_io token count wrong for conc leg: "
            f"{conc['io_attribution']['tokens_generated_incl_warmup']}"
        )
    print(f"[4b] io attribution: bs1 tokens="
          f"{bs1['io_attribution']['tokens_generated_incl_warmup']} "
          f"conc tokens={conc['io_attribution']['tokens_generated_incl_warmup']} "
          f"expert_bytes/token={synth_eb.get('estimate_gb')} GB (this packing)")
    doc = assemble_doc(
        a,
        status="selftest-SYNTHETIC-DATA-NOT-A-MEASUREMENT",
        legs={"bs1": bs1, f"conc{a.conc}": conc},
        box_before=before,
        box_after=after,
        deltas={"selftest": box_state_delta(before, after)},
        cards_used=["<selftest: rocm-smi deliberately not invoked>"],
        server_argv={
            "bs1": build_server_argv(a, 1, a.port),
            f"conc{a.conc}": build_server_argv(a, a.conc, a.port),
        },
        server_log_excerpts={},
        server_props={},
        load_seconds={},
        gpu_engagement={
            "verdict": "confirmed",
            "why": "<selftest: synthetic attribution, no GPU was touched>",
            "per_leg": {"bs1": good, f"conc{a.conc}": good},
        },
        backend={"bs1": {"model_geometry": synth_eb["geometry"]}},
    )
    errs = validate_shape(doc)
    if errs:
        failures.extend(f"schema: {e}" for e in errs)
        print(f"[5] JSON shape: {len(errs)} ERRORS")
        for e in errs:
            print(f"      - {e}")
    else:
        print("[5] JSON shape: valid")

    # 6. Derived block sanity on the synthetic numbers.
    dd = doc["derived"]
    for k in (
        "measured_bs1_tok_s",
        "fraction_of_plan_derived_ceiling",
        "K4_hard_kill_threshold_tok_s",
        "break_even_hit_rate_vs_measured_baseline",
        "conc_scaling",
    ):
        if k not in dd:
            failures.append(f"derived block missing {k}")
    if abs(dd.get("K4_hard_kill_threshold_tok_s", 0) - dd["measured_bs1_tok_s"] / 1.57) > 1e-3:
        failures.append("K4 threshold is not measured_bs1 / 1.57 (plan Section 7, gate K4)")
    sc = dd.get("conc_scaling") or {}
    if "scaling_factor" not in (sc.get("wall_vs_wall") or {}):
        failures.append("conc_scaling is missing the matched wall-vs-wall comparison")
    if "scaling_factor" not in (sc.get("decode_vs_decode") or {}):
        failures.append("conc_scaling is missing the matched decode-vs-decode comparison")
    ww = sc.get("wall_vs_wall", {})
    if ww.get("scaling_factor") is not None and (
        abs(ww["scaling_factor"] - ww["conc_wall_tok_s"] / ww["bs1_wall_tok_s"]) > 5e-3
    ):
        failures.append("wall_vs_wall scaling factor does not match its own operands")
    naive = (sc.get("MISMATCHED_do_not_cite") or {}).get("conc_wall_over_bs1_decode")
    if naive is not None and ww.get("scaling_factor") is not None and (
        abs(naive - ww["scaling_factor"]) < 1e-6
    ):
        failures.append(
            "matched and mismatched conc ratios are identical on synthetic data, so the "
            "matched-basis check is vacuous: synth_rep must put prefill in the wall"
        )
    print(f"[6] derived: bs1={dd.get('measured_bs1_tok_s')} "
          f"frac_of_ceiling={dd.get('fraction_of_plan_derived_ceiling')} "
          f"K4={dd.get('K4_hard_kill_threshold_tok_s')} "
          f"conc wall/wall={ww.get('scaling_factor')} "
          f"(naive mismatched would be "
          f"{(sc.get('MISMATCHED_do_not_cite') or {}).get('conc_wall_over_bs1_decode')})")

    # 6b. The schema must REFUSE a clean status on an unattributable run.
    bad = json.loads(json.dumps(doc))
    bad["status"] = "ok"
    bad["gpu_engagement"] = {"verdict": "cpu_only", "why": "synthetic"}
    if not validate_shape(bad):
        failures.append("validate_shape accepted status=ok with a cpu_only attribution")
    bad2 = json.loads(json.dumps(doc))
    del bad2["gpu_engagement"]
    if not validate_shape(bad2):
        failures.append("validate_shape accepted a document with no gpu_engagement block")
    print("[6b] schema refuses a clean status on an unattributable run")

    # 7. Write to the SELFTEST paths only -- never over p0.json.
    selftest_dir = RESULTS_DIR / "_selftest"
    selftest_dir.mkdir(parents=True, exist_ok=True)
    out_json = selftest_dir / "p0.SELFTEST-SYNTHETIC-DO-NOT-CITE.json"
    out_md = selftest_dir / "p0.SELFTEST-SYNTHETIC-DO-NOT-CITE.md"
    if out_json.name == "p0.json" or out_md.name.startswith("P0_"):
        failures.append("selftest is about to clobber a real artifact")
    if out_json.parent == RESULTS_DIR:
        failures.append("selftest output must not sit beside the real result files")
    out_json.write_text(json.dumps(doc, indent=2, sort_keys=False))
    write_markdown(doc, out_md)
    roundtrip = json.loads(out_json.read_text())
    if validate_shape(roundtrip):
        failures.append("round-tripped JSON failed shape validation")
    print(f"[7] wrote + re-validated {out_json}")
    print(f"    wrote {out_md}")
    if (RESULTS_DIR / "p0.json").exists():
        print("    (existing p0.json left untouched)")

    # 8. The cited prior must never be presentable as a measurement.
    if doc["cited_prior"]["measured_by_this_script"] is not False:
        failures.append("cited_prior claims to be measured")
    if "reproduced_by_this_run" in doc["cited_prior"] and "SYNTHETIC" not in doc["status"]:
        failures.append("cited_prior comparison leaked out of a synthetic run")
    print("[8] cited prior is flagged as a citation, not a measurement")

    print()
    if failures:
        print(f"=== SELFTEST FAILED: {len(failures)} problem(s) ===", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("=== SELFTEST PASSED ===")
    return 0


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #
def assemble_doc(
    a: argparse.Namespace,
    status: str,
    legs: dict,
    box_before: dict,
    box_after: dict,
    deltas: dict,
    cards_used: list,
    server_argv: dict,
    server_log_excerpts: dict,
    server_props: dict,
    load_seconds: dict,
    gpu_engagement: dict | None = None,
    backend: dict | None = None,
) -> dict:
    bs1 = legs.get("bs1", {})
    bs1_med = (bs1.get("decode_tok_s") or {}).get("median")
    bs1_wall_med = (bs1.get("wall_tok_s") or {}).get("median")
    concleg = next((v for k, v in legs.items() if k.startswith("conc")), {})
    conc_med = (concleg.get("aggregate_tok_s_wall") or {}).get("median")
    conc_sum_med = (concleg.get("aggregate_decode_tok_s_sum_of_rates") or {}).get("median")
    conc_n = concleg.get("concurrency", a.conc)

    cited = dict(CITED_PRIOR)
    if bs1_med is not None:
        cited["fresh_median_tok_s"] = bs1_med
        cited["delta_pct"] = round(
            100.0 * (bs1_med - cited["value_tok_s"]) / cited["value_tok_s"], 2
        )
        cited["reproduced_by_this_run"] = bool(abs(cited["delta_pct"]) <= 15.0)

    doc = {
        "probe": "P0",
        "probe_name": "llamacpp_baseline",
        "plan_section": "WEIGHT_OFFLOAD_PLAN.md §3 (Phase 0), row P0",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "cards_used": cards_used,
        "physical_card_map": {str(k): v for k, v in PHYSICAL_CARDS.items()},
        "reps_below_minimum": bool(a.reps < MIN_REPS),
        "cited_prior": cited,
        "provenance": {
            "python": sys.version,
            "script": str(Path(__file__).resolve()),
            "git": git_provenance(),
            "model": model_provenance(),
            "runtime": runtime_provenance(),
            "env": {
                "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
                "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "<unset>"),
                "gpu_lease": "WAIVED for this development run by explicit user instruction",
            },
            "launch": {
                "probe_command": "python3 "
                + str(Path(__file__).resolve().relative_to(REPO))
                + " "
                + " ".join(sys.argv[1:]),
                "server_argv": server_argv,
                "server_load_seconds": load_seconds,
                "sampling": {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 50,
                    "ignore_eos": True,
                    "cache_prompt": False,
                    "server_cache_ram_mib": a.cache_ram_mib,
                    "note": "sampled, never greedy; ignore_eos pins every rep to exactly n_predict "
                    "tokens; a fresh os.urandom(16) nonce at token ~8 of every prompt defeats "
                    "prefix reuse, and --cache-ram disables the server-side prompt cache "
                    "(which the per-request cache_prompt field does not reach)",
                },
            },
            "server_props": server_props,
            "server_log_excerpts": server_log_excerpts,
        },
        "box_state": {"before": box_before, "after": box_after, "delta": deltas},
        "legs": legs,
        "gpu_engagement": gpu_engagement
        or {
            "verdict": "unconfirmed",
            "why": "no attribution evidence was supplied to assemble_doc()",
        },
        "backend_from_server_log": backend or {},
        "derived": derive(
            bs1_med,
            conc_med,
            conc_n,
            bs1_wall_tok_s=bs1_wall_med,
            conc_sum_of_rates=conc_sum_med,
        ),
    }
    doc["verdict"] = verdict_text(doc)
    return doc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P0 -- llama.cpp baseline on the target checkpoint (weight-offload Phase 0).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=("both", "bs1", "conc"), default="both",
                   help="which legs to run")
    p.add_argument("--reps", type=int, default=5,
                   help="steady-state repetitions per leg (a warm-up rep is always discarded on top)")
    p.add_argument("--allow-fewer-reps", action="store_true",
                   help="permit --reps < 5; stamps the result 'degraded'")
    p.add_argument("--n-predict", type=int, default=100, help="tokens generated per request")
    p.add_argument("--conc", type=int, default=6, help="concurrency for the aggregate leg")
    p.add_argument("--ctx-total", type=int, default=8192,
                   help="llama-server total context (-c), shared across slots")
    p.add_argument("--threads", type=int, default=8,
                   help="CPU threads (-t); the 7800X3D has 8 physical cores")
    p.add_argument("--fit", choices=("on", "off"), default="on",
                   help="llama.cpp auto-fit offload (-fit), matches the cited prior run")
    p.add_argument("--flash-attn", choices=("on", "off", "auto"), default="auto")
    p.add_argument("--split-mode", choices=("none", "layer", "row", "tensor"), default="layer")
    p.add_argument("--load-mode", default="",
                   help="llama.cpp -lm (auto|none|mmap|mlock|mmap+mlock|dio); empty = server default")
    p.add_argument("--lazy-mode", default="",
                   help="llama.cpp -lzm (on|auto|off); empty = server default (auto)")
    p.add_argument("--extra-server-args", default="",
                   help="verbatim extra llama-server args (recorded in provenance)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--ready-timeout", type=float, default=2400.0,
                   help="seconds to wait for /health; the 93.7 GB checkpoint loads cold from ZFS")
    p.add_argument("--request-timeout", type=float, default=1200.0,
                   help="seconds per generation request")
    p.add_argument("--cache-ram-mib", type=int, default=0,
                   help="llama-server --cache-ram (MiB); 0 disables the server-side prompt "
                        "cache entirely so no rep can be served a cached prefill")
    p.add_argument("--min-gpu-weight-gb", type=float, default=1.0,
                   help="minimum device-resident weight bytes (log ROCm buffers, or VRAM "
                        "growth across model load) required to call GPU offload confirmed")
    p.add_argument("--vram-busy-gb", type=float, default=1.0,
                   help="fail if either compute card already holds more VRAM than this")
    p.add_argument("--force", action="store_true",
                   help="run even if a compute card looks busy (contention recorded in the JSON)")
    p.add_argument("--shared-server", action="store_true",
                   help="run both legs on ONE -np <conc> server instead of relaunching with -np 1 "
                        "for the bs=1 leg (saves one ~2 min load; bs=1 then sits on a multi-slot "
                        "server, which is recorded per leg)")
    p.add_argument("--out", default=str(RESULTS_DIR / "p0.json"), help="JSON output path")
    p.add_argument("--md-out", default=str(RESULTS_DIR / "P0_LLAMACPP_BASELINE.md"),
                   help="Markdown output path")
    p.add_argument("--selftest", action="store_true",
                   help="GPU-free validation of argument handling, card classification, GPU "
                        "attribution, drift detection and JSON shape; writes only into "
                        "_selftest/p0.SELFTEST-SYNTHETIC-DO-NOT-CITE.{json,md}")
    p.add_argument("--dry-run", action="store_true",
                   help="check preconditions and print the resolved launch lines; run nothing")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)

    if a.selftest:
        return run_selftest(a)

    if a.dry_run:
        print("=== P0 DRY RUN (no GPU touched, nothing launched) ===")
        try:
            notes = check_preconditions(a, gpu_checks=False)
        except PreconditionError as e:
            die(str(e))
        print(f"preconditions OK (non-GPU): {json.dumps(notes, indent=2)}")
        print(f"results dir : {RESULTS_DIR}")
        print(f"json out    : {a.out}")
        print(f"md out      : {a.md_out}")
        print(f"ROCR        : {os.environ.get('ROCR_VISIBLE_DEVICES')}")
        print(f"HIP         : {os.environ.get('HIP_VISIBLE_DEVICES', '<unset>')}")
        print("\nllama-server launch lines:")
        if a.mode in ("both", "bs1"):
            print("  [bs1] " + " ".join(build_server_argv(a, 1 if not a.shared_server else a.conc, a.port)))
        if a.mode in ("both", "conc"):
            print(f"  [conc{a.conc}] " + " ".join(build_server_argv(a, a.conc, a.port)))
        print("\nNOTE: rocm-smi was deliberately NOT invoked in dry-run mode.")
        return 0

    try:
        notes = check_preconditions(a, gpu_checks=True)
    except PreconditionError as e:
        die(str(e))

    cards = notes.get("cards_raw", {})
    cards_used = [
        f"{dev}: {rec.get('name')} (vram_total {rec.get('vram_total_gb')} GB, "
        f"{rec.get('vram_used_gb')} GB in use at probe start)"
        for dev, rec in cards.items()
        if rec.get("is_compute_card")
    ]
    cards_used.append(
        "ROCR_VISIBLE_DEVICES=0,1 -- card2 (Ryzen 7800X3D iGPU) fenced out of enumeration"
    )
    log("cards: " + "; ".join(cards_used))

    box_before = box_state(with_gpu=True)
    legs, deltas, argvs, excerpts, props, loads = {}, {}, {}, {}, {}, {}
    backends, engagements = {}, {}
    server = None
    status = "ok"

    def _serve(leg_key: str, n_parallel: int, logname: str):
        """Start a server, returning (server, pre_server_box_state, logpath)."""
        nonlocal server
        argv = build_server_argv(a, n_parallel, a.port)
        argvs[leg_key] = argv
        logpath = RESULTS_DIR / logname
        # VRAM BEFORE the server exists: the only format-independent evidence
        # that model weights actually landed on the discrete cards.
        pre = box_state(with_gpu=True)
        server = LlamaServer(argv, logpath, a.ready_timeout, a.port)
        server.start()
        loads[leg_key] = server.load_seconds
        props[leg_key] = server.props()
        return server, pre, logpath

    def _attribute(key: str, logpath: Path, pre: dict, post_load: dict) -> dict:
        backend = parse_server_log_backend(logpath)
        backends[key] = backend
        ev = attribute_gpu_engagement(
            backend, vram_delta_gb(pre, post_load), a.min_gpu_weight_gb
        )
        ev["leg"] = key
        engagements[key] = ev
        log(
            f"GPU attribution [{key}]: {ev['verdict']} "
            f"(log ROCm buffers {backend.get('gpu_weight_gb')} GB, "
            f"CPU buffers {backend.get('cpu_weight_gb')} GB, "
            f"vram delta {ev['cards_with_vram_growth_gb']})"
        )
        return ev

    def run_one(leg_name: str, n_parallel: int, runner) -> None:
        nonlocal server
        srv, pre, logpath = _serve(leg_name, n_parallel, f"p0_server_{leg_name}.log")
        b0 = box_state(with_gpu=True)
        pio0 = read_proc_io(srv.pid)
        ev = _attribute(leg_name, logpath, pre, b0)
        eb = expert_bytes_per_token_gb(backends[leg_name])
        try:
            legs[leg_name] = runner()
        finally:
            pio1 = read_proc_io(srv.pid)
            b1 = box_state(with_gpu=True)
            deltas[leg_name] = box_state_delta(b0, b1)
            lg = legs.setdefault(leg_name, {})
            lg["box_state_before_leg"] = b0
            lg["box_state_after_leg"] = b1
            lg["vram_used_gb_pre_server"] = compute_card_vram(pre)
            lg["vram_used_gb_after_load"] = compute_card_vram(b0)
            lg["server_n_parallel"] = n_parallel
            lg["server_load_seconds"] = srv.load_seconds
            lg["gpu_engagement"] = ev
            lg["expert_bytes_per_token"] = eb
            lg["comparability_note"] = (
                f"this leg ran on its OWN llama-server started with -np {n_parallel}. "
                f"`-fit on` sizes the offload against the KV pool, so the bs=1 and CONC "
                f"legs may not have offloaded the same number of layers: compare "
                f"backend_from_server_log[*].layers_offloaded before reading the "
                f"CONC/bs=1 ratio as pure batching effect."
            )
            annotate_leg_io(lg, deltas[leg_name], eb, proc_io_delta(pio0, pio1))
            excerpts[leg_name] = harvest_server_log(logpath)
            srv.stop()
            server = None

    try:
        if a.shared_server:
            np_ = max(a.conc, 1)
            srv, pre, logpath = _serve("shared", np_, "p0_server_shared.log")
            b_load = box_state(with_gpu=True)
            ev = _attribute("shared", logpath, pre, b_load)
            eb = expert_bytes_per_token_gb(backends["shared"])
            try:
                for name, runner in (
                    ("bs1", lambda: run_leg_bs1(a, a.port)),
                    (f"conc{a.conc}", lambda: run_leg_conc(a, a.port, a.conc)),
                ):
                    if name == "bs1" and a.mode not in ("both", "bs1"):
                        continue
                    if name != "bs1" and a.mode not in ("both", "conc"):
                        continue
                    b0 = box_state(with_gpu=True)
                    pio0 = read_proc_io(srv.pid)
                    legs[name] = runner()
                    pio1 = read_proc_io(srv.pid)
                    b1 = box_state(with_gpu=True)
                    deltas[name] = box_state_delta(b0, b1)
                    legs[name]["server_n_parallel"] = np_
                    legs[name]["shared_server"] = True
                    legs[name]["box_state_before_leg"] = b0
                    legs[name]["box_state_after_leg"] = b1
                    legs[name]["vram_used_gb_pre_server"] = compute_card_vram(pre)
                    legs[name]["vram_used_gb_after_load"] = compute_card_vram(b_load)
                    legs[name]["gpu_engagement"] = ev
                    legs[name]["expert_bytes_per_token"] = eb
                    legs[name]["comparability_note"] = (
                        f"--shared-server: this leg ran on ONE server started with "
                        f"-np {np_}, so the bs=1 leg sits on a multi-slot server with "
                        f"1/{np_} of the context per slot."
                    )
                    annotate_leg_io(legs[name], deltas[name], eb, proc_io_delta(pio0, pio1))
            finally:
                excerpts["shared"] = harvest_server_log(logpath)
                srv.stop()
                server = None
        else:
            if a.mode in ("both", "bs1"):
                run_one("bs1", 1, lambda: run_leg_bs1(a, a.port))
            if a.mode in ("both", "conc"):
                run_one(f"conc{a.conc}", a.conc, lambda: run_leg_conc(a, a.port, a.conc))
    except PreconditionError as e:
        if server:
            server.stop()
        die(str(e), code=3)
    except KeyboardInterrupt:
        if server:
            server.stop()
        print("\ninterrupted; partial results are NOT written", file=sys.stderr)
        return 130
    except Exception:  # noqa: BLE001
        if server:
            server.stop()
        traceback.print_exc()
        die("unhandled error during measurement; no numbers written", code=4)

    box_after = box_state(with_gpu=True)

    # Validity gate: never emit a headline number derived from too few reps.
    for name, leg in legs.items():
        want = leg.get("reps_requested", a.reps)
        got = leg.get("reps_valid", 0)
        if got < want:
            status = "degraded"
            log(f"WARNING: leg {name} produced {got}/{want} valid reps")
        if got == 0:
            die(
                f"leg {name} produced ZERO valid repetitions. Refusing to write a result. "
                f"Inspect {RESULTS_DIR}/p0_server_*.log",
                code=5,
            )
    if a.reps < MIN_REPS:
        status = "degraded"
    if notes.get("forced_over_busy_gpu"):
        status = "degraded"

    # Steady state: a median taken off a ramp is a number, not a rate.
    for name, leg in legs.items():
        if leg.get("steady_state") is False:
            status = "degraded"
            log(f"WARNING: leg {name} is NOT in steady state (see decode_tok_s.drift)")
        pc = leg.get("prompt_cache_check") or {}
        if pc.get("suspected_prefill_cache_hit"):
            status = "degraded"
            log(f"WARNING: leg {name} shows a collapsing prompt_ms — a prefill may be cached")

    # GPU attribution: the run must be the configuration this probe claims.
    worst = "confirmed"
    order = {"confirmed": 0, "unconfirmed": 1, "cpu_only": 2, "igpu_leak": 2}
    for ev in engagements.values():
        if order.get(ev.get("verdict"), 1) > order.get(worst, 0):
            worst = ev["verdict"]
    overall_engagement = {
        "verdict": worst if engagements else "unconfirmed",
        "why": (
            "; ".join(f"[{k}] {v.get('why', '')}" for k, v in engagements.items())
            or "no server was started, so nothing could be attributed"
        ),
        "per_leg": engagements,
        "min_gpu_weight_gb": a.min_gpu_weight_gb,
    }
    if overall_engagement["verdict"] == "unconfirmed":
        status = "degraded"
        log("WARNING: GPU offload could not be positively confirmed for every leg")
    invalid = overall_engagement["verdict"] in ("cpu_only", "igpu_leak")
    if invalid:
        status = f"INVALID-{overall_engagement['verdict']}"
        log(f"FATAL ATTRIBUTION: {overall_engagement['why']}")

    doc = assemble_doc(
        a,
        status=status,
        legs=legs,
        box_before=box_before,
        box_after=box_after,
        deltas=deltas,
        cards_used=cards_used,
        server_argv=argvs,
        server_log_excerpts=excerpts,
        server_props=props,
        load_seconds=loads,
        gpu_engagement=overall_engagement,
        backend=backends,
    )
    if notes.get("forced_over_busy_gpu"):
        doc["contention_at_start"] = notes["forced_over_busy_gpu"]

    invalid_json = Path(a.out).with_name(Path(a.out).stem + ".INVALID.json")

    if invalid:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        invalid_json.write_text(json.dumps(doc, indent=2))
        print(
            "\n*** P0 RESULT IS NOT ATTRIBUTABLE TO THE CLAIMED CONFIGURATION ***\n"
            f"{overall_engagement['why']}\n"
            f"Raw data preserved at {invalid_json}. NOTHING was written to {a.out}: a "
            f"CPU-only or iGPU-poisoned number must never be readable as the P0 bar.\n",
            file=sys.stderr,
        )
        return 8

    errs = validate_shape(doc)
    if errs:
        print("\nJSON shape validation FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        # Preserve the data -- it cost ~30 min -- but NEVER under the name a
        # downstream gate globs for.  The contract is "writes nothing" for p0.json.
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        invalid_json.write_text(json.dumps(doc, indent=2))
        print(
            f"(raw document preserved at {invalid_json}; {a.out} was NOT written)",
            file=sys.stderr,
        )
        return 6

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=2))
    write_markdown(doc, Path(a.md_out))

    print()
    print(json.dumps({k: doc[k] for k in ("probe", "status", "verdict")}, indent=2))
    print()
    print("--- machine-readable summary ---")
    print(
        json.dumps(
            {
                "probe": "P0",
                "status": doc["status"],
                "gpu_engagement": overall_engagement["verdict"],
                "steady_state": {n: lg.get("steady_state") for n, lg in legs.items()},
                "bs1_decode_tok_s": (legs.get("bs1", {}).get("decode_tok_s") or {}).get("median"),
                "bs1_wall_tok_s": (legs.get("bs1", {}).get("wall_tok_s") or {}).get("median"),
                "bs1_prompt_tok_s": (legs.get("bs1", {}).get("prompt_tok_s") or {}).get("median"),
                "conc_aggregate_tok_s_wall": (
                    (legs.get(f"conc{a.conc}", {}).get("aggregate_tok_s_wall") or {}).get("median")
                ),
                "conc_aggregate_tok_s_sum_of_rates": (
                    (legs.get(f"conc{a.conc}", {}).get("aggregate_decode_tok_s_sum_of_rates")
                     or {}).get("median")
                ),
                "conc": a.conc,
                "plan_derived_ceiling_tok_s": PLAN_DERIVED["cpu_expert_ceiling_tok_s"],
                "fraction_of_plan_derived_ceiling": doc["derived"].get(
                    "fraction_of_plan_derived_ceiling"
                ),
                "K4_hard_kill_threshold_tok_s": doc["derived"].get("K4_hard_kill_threshold_tok_s"),
                "cited_prior_tok_s": CITED_PRIOR["value_tok_s"],
                "cited_prior_reproduced": doc["cited_prior"].get("reproduced_by_this_run"),
                "cards": cards_used,
            },
            indent=2,
        )
    )
    print()
    print(f"wrote {a.out}")
    print(f"wrote {a.md_out}")
    return 0 if status == "ok" else 7


if __name__ == "__main__":
    sys.exit(main())
