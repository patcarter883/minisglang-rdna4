# `qwen4_exp` (Qwen3.8-Flash-Next) test fixtures

Durable, checked-in copies of the two things the CPU-only bring-up tests need. Both were taken from
the real checkpoint, not written by hand — the point is that the tests keep working when nothing is
downloaded and keep testing the *shipping* names, not a paraphrase of them.

| file | what it is |
|---|---|
| `config.json` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` @ `7b71922`, verbatim. This is the bring-up target: `quant_method: "modelopt"`, `quant_algo: "NVFP4"`, group_size 16, and an ignore list that leaves everything except the routed experts in bf16. |
| `config_ct_variant.json` | A SECOND, different `qwen4_exp` checkpoint's config (compressed-tensors `mxfp4-pack-quantized`, with fp8-block `self_attn`/`linear_attn` groups and an extra `mtp` expert group). Same architecture, different quantization. Kept so the config parser is tested against two independent spellings rather than one; it is **not** the checkpoint the loader targets. |
| `ckpt_keys.txt.gz` | All **296,475** tensor names from the NVFP4 checkpoint's `model.safetensors.index.json` (206 files), sorted, one per line. This is the ground truth `qwen4exp_remap_test.py` proves the name mapper covers. |

Regenerating `ckpt_keys.txt.gz`:

```python
import json, gzip
wm = json.load(open("<snapshot>/model.safetensors.index.json"))["weight_map"]
with gzip.open("ckpt_keys.txt.gz", "wt", compresslevel=9) as f:
    f.write("\n".join(sorted(wm)) + "\n")
```

Not checked in: the 135 GB of weights, and any tensor SHAPE/DTYPE table. Shape parity needs the body
downloaded and is a later bring-up step (plan T1.4); the tests here are name-level only and say so.
