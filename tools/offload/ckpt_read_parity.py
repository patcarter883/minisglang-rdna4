#!/usr/bin/env python3
"""MERGE GATE for `weights/ckpt_read.py`: prove the read() reader is byte-identical to `safe_open`.

"A faster boot that loads different weights is worthless." The arena is layout-sensitive, so this
checks three things per shard and fails on any of them:

  1. `keys()` LIST EQUALITY, order included. The carve order IS the arena layout — a reordering
     changes `carve_digest()` and every region address, and on a quantized checkpoint that means
     dequantizing one expert against another's scale: plausible text, no crash, nothing downstream
     can detect it. The shard headers are NOT in sorted order, so this is a live hazard, not a
     theoretical one.
  2. dtype and shape per tensor.
  3. the RAW BYTES, compared as uint8 so that NaN payloads, fp8 encodings and denormals are all
     compared bit-for-bit rather than by float equality.

It also times both readers on the same cold shards, which is the number the fix is justified by.

CPU only. No GPU, no lease. Run it inside the serve image (torch is not importable on the host).
"""
import hashlib
import os
import sys
import time

sys.path.insert(0, "/engine/python")

import safetensors  # noqa: E402
import torch  # noqa: E402

from minisgl.weights import ckpt_read  # noqa: E402
from minisgl.weights.ckpt_read import WHOLE_FILE_CAP, ReadSafeOpen  # noqa: E402

MODEL = os.environ.get("PARITY_MODEL", "/model")
N_FILES = int(os.environ.get("PARITY_FILES", "4"))
GIB = 1 << 30
MIB = 1 << 20


def raw(t: torch.Tensor) -> torch.Tensor:
    """uint8 view of the storage, so fp8/bf16/NaN payloads compare bit-for-bit.

    `reshape(-1)` FIRST: the checkpoint carries 0-dim scalars (`input_scale`), and `view(uint8)` on
    a 0-dim tensor raises rather than reinterpreting.
    """
    f = t.contiguous().reshape(-1)
    return f.view(torch.uint8) if f.numel() else f


files = sorted(f for f in os.listdir(MODEL) if f.endswith(".safetensors"))
# A mix on purpose: the expert shards (unsorted header, F8_E4M3 + U8 + F32) and the dense bf16
# shards (sorted header, BF16) exercise different branches of the key-order and dtype handling.
pick = files[:N_FILES // 2] + files[-(N_FILES - N_FILES // 2):]
print(f"model={MODEL}  {len(files)} shards, checking {len(pick)}", flush=True)

tot_bytes = 0
t_mmap = t_read = 0.0
failures = []
checked = 0

for fn in pick:
    path = os.path.join(MODEL, fn)
    size = os.path.getsize(path)

    t0 = time.perf_counter()
    ref = safetensors.safe_open(path, framework="pt", device="cpu")
    ref_keys = list(ref.keys())
    ref_t = {k: ref.get_tensor(k) for k in ref_keys}
    # mmap defers the fault to the first READ, so a fair timing must actually touch the bytes --
    # that deferral is the whole defect being fixed and timing get_tensor alone would report 10 GiB/s
    # for a reader that has not read anything.
    ref_h = [raw(v).sum(dtype=torch.int64) for v in ref_t.values()]
    t1 = time.perf_counter()

    # Through the POLICY, not the class: `safe_open` is what the loader calls, and for a shard
    # over WHOLE_FILE_CAP the correct answer is 'safetensors' mmap, unchanged'. Asserting that
    # here is half the gate -- a cap regression that silently pushed 10 GiB body shards onto
    # the O_DIRECT reader is precisely the defect that cost the 2026-09-06 after-leg.
    got = ckpt_read.safe_open(path)
    got_keys = got.keys()
    got_t = {k: got.get_tensor(k) for k in got_keys}
    got_h = [raw(v).sum(dtype=torch.int64) for v in got_t.values()]
    t2 = time.perf_counter()

    t_mmap += t1 - t0
    t_read += t2 - t1
    tot_bytes += size

    if ref_keys != got_keys:
        d = [i for i, (a, b) in enumerate(zip(ref_keys, got_keys)) if a != b][:3]
        failures.append(f"{fn}: keys() ORDER/CONTENT differs (n={len(ref_keys)}/{len(got_keys)}); "
                        f"first divergences at {d}")
        continue
    for k in ref_keys:
        a, b = ref_t[k], got_t[k]
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            failures.append(f"{fn}:{k}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}")
            continue
        ra, rb = raw(a), raw(b)
        if ra.numel() != rb.numel() or not torch.equal(ra, rb):
            failures.append(f"{fn}:{k}: BYTES differ ({ra.numel()} vs {rb.numel()} B)")
            continue
        checked += 1
    ha = hashlib.sha256()
    hb = hashlib.sha256()
    for k in ref_keys:
        ha.update(raw(ref_t[k]).numpy().tobytes())
        hb.update(raw(got_t[k]).numpy().tobytes())
    if ha.hexdigest() != hb.hexdigest():
        failures.append(f"{fn}: whole-shard sha256 differs {ha.hexdigest()[:16]} vs "
                        f"{hb.hexdigest()[:16]}")
    # WHICH MODE RAN matters as much as the bytes: `ReadSafeOpen` serves shards <= WHOLE_FILE_CAP
    # from one buffer and larger ones with a per-tensor aligned pread, and those are two different
    # offset computations (the per-tensor one indexes into a block-aligned read-around window).
    # `pick` deliberately spans both -- expert shards are 337 MiB, the `model-bf16-*` body shards
    # are 3.4-10.0 GiB -- so a green run that exercised only one would be half a gate. `direct`
    # records whether O_DIRECT was actually granted on this mount.
    mode = "O_DIRECT" if size <= WHOLE_FILE_CAP else "mmap(policy)"
    if (size <= WHOLE_FILE_CAP) != isinstance(got, ReadSafeOpen):
        failures.append(f"{fn}: safe_open routed a {size} B shard to the wrong reader "
                        f"({type(got).__name__})")
    print(f"  {fn:<46} {size / MIB:8.1f} MiB  keys={len(ref_keys):>5}  {mode:<10} "
          f"direct={str(getattr(got, 'direct', 'n/a')):<5} "
          f"mmap {size / MIB / (t1 - t0):7.1f} MiB/s   O_DIRECT {size / MIB / (t2 - t1):7.1f} MiB/s  "
          f"sha {ha.hexdigest()[:16]}", flush=True)
    del ref_t, got_t, ref, got, ref_h, got_h

print(f"\n{checked} tensors compared byte-for-byte over {tot_bytes / GIB:.2f} GiB", flush=True)
print(f"mmap  (safe_open + touch): {t_mmap:7.2f} s = {tot_bytes / MIB / t_mmap:8.1f} MiB/s")
print(f"O_DIRECT (ReadSafeOpen)  : {t_read:7.2f} s = {tot_bytes / MIB / t_read:8.1f} MiB/s")
print(f"speedup {t_mmap / t_read:.2f}x")
if failures:
    print(f"\nFAIL: {len(failures)} problem(s)")
    for f in failures[:20]:
        print("  " + f)
    sys.exit(1)
print("\nPARITY PASS: identical keys(), order, dtypes, shapes and bytes")
