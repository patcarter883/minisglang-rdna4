"""fp8-KV sidecar store: content-keyed lookup and the calibrator's --if-missing no-op (CPU).

    PYTHONPATH=python python tests/kv_scales_store_test.py

  * the store key is the same for two paths holding the same checkpoint, and differs when the
    config or the weight files differ (a different quantization of the same model);
  * resolve_kv_fp8_scales finds a sidecar placed in the store, ahead of the snapshot's own;
  * default_sidecar_path points into the store when MINISGL_KV_SCALES_DIR is set;
  * `kv_fp8_calibrate.py --if-missing` exits 0 without touching a GPU when scales exist.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import save_file

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def make_ckpt(root, name, quant):
    d = os.path.join(root, name)
    os.makedirs(d)
    json.dump({"model_type": "toy", "quantization_config": {"format": quant}},
              open(os.path.join(d, "config.json"), "w"))
    save_file({"model.layers.0.mlp.w": torch.zeros(4, 4)}, os.path.join(d, "model.safetensors"))
    return d


def sidecar(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({"model.layers.0.self_attn.k_scale": torch.full((2,), value),
               "model.layers.0.self_attn.v_scale": torch.full((2,), value)}, path,
              metadata={"format": "minisgl-kv-fp8-e4m3", "fp8_max": "448.0"})


def main() -> int:
    from minisgl.kvcache import fp8_scales as S

    root = tempfile.mkdtemp()
    try:
        a = make_ckpt(root, "mxfp4", "mxfp4-pack-quantized")
        a2 = os.path.join(root, "same-ckpt-other-mount")
        shutil.copytree(a, a2)
        b = make_ckpt(root, "nvfp4", "nvfp4-pack-quantized")
        ka, ka2, kb = S.store_key(a), S.store_key(a2), S.store_key(b)
        check("same checkpoint, different path -> same key", ka == ka2, ka)
        check("different quantization -> different key", ka != kb, f"{ka} vs {kb}")

        store = os.path.join(root, "store")
        os.environ[S.STORE_ENV] = store
        check("no scales anywhere -> None", S.resolve_kv_fp8_scales(a) is None)
        sidecar(os.path.join(a, S.SIDECAR_NAME), 1.0)            # snapshot sidecar
        sidecar(S.default_sidecar_path(a), 2.0)                   # store sidecar
        check("default path is in the store", S.default_sidecar_path(a).startswith(store))
        found = S.resolve_kv_fp8_scales(a2)
        k = found.scales[0][0] if found else None
        check("store sidecar found from another mount path, ahead of the snapshot",
              found is not None and store in found.source and float(k.flatten()[0]) == 2.0,
              found.source if found else "none")
        check("other quantization still has none", S.resolve_kv_fp8_scales(b) is None)

        env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "python") + ":" + os.environ.get("PYTHONPATH", ""),
                   HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
        r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "kv_fp8_calibrate.py"),
                            "--if-missing", "--model", a2, "--text", "/nonexistent"],
                           capture_output=True, text=True, env=env, timeout=300)
        check("--if-missing is a no-op when scales exist",
              r.returncode == 0 and "nothing to calibrate" in r.stdout, (r.stdout + r.stderr)[-200:])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("ALL CHECKS PASS" if not FAILED else "FAILED: " + "; ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
