"""Guard: a model must not hand-roll a computation that has a FUSED kernel behind a helper.

    python tests/no_unreachable_fused_paths_test.py

WHY THIS FILE EXISTS. Three separate optimisations were found unreachable in a single day
(2026-09-17), all with the same shape: the fast path existed, was correct, was tested -- and nothing
called it.

  * `Linear.forward_swiglu` routes a merged gate_up through a FUSED gemm+silu kernel and otherwise
    falls back to exactly `silu_and_mul(self.forward(x))`. Two MoE models called it. THREE other
    MLPs -- the shared `GatedMLP` (qwen3, qwen3_5, laguna, mistral, llama, qwen2, muse_glimmer),
    `Qwen3_5MoE`'s dense MLP, and Laguna's shared expert -- spelled the fallback out by hand, so the
    fusion was unreachable on most of the fleet no matter how the quant path was configured.
  * `Int4A16GemvLoader` never got the per-16-K-half scale fold its fp8 sibling had, so NVFP4 had no
    unquantized decode arm.
  * `supports_producer_actquant` was corrected on two of three linear methods.

A grep cannot catch the general case. What it CAN catch is the specific, recurring one: writing the
fallback expression by hand instead of calling the helper that chooses between fallback and fused.
That is cheap to detect and it is exactly the mistake that keeps happening.

This test is deliberately NARROW. It does not ban `silu_and_mul` -- the activation is legitimately
used by unfused paths, by MoE experts, and inside `forward_swiglu` itself. It bans exactly one
shape: applying it to the output of a gate_up projection's plain `.forward(...)`.
"""
import ast
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(_HERE, "..", "python", "minisgl", "models")
LAYERS = os.path.join(_HERE, "..", "python", "minisgl", "layers")
FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILS.append(name)


def hand_rolled_swiglu(path):
    """Find `silu_and_mul(<anything>.forward(...))` where the inner receiver names a gate_up.

    AST, not regex: the expression spans lines in some models, and a regex that tries to match
    balanced parens across a line break is how a guard like this ends up silently matching nothing.
    """
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return []
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "silu_and_mul" and node.args):
            continue
        inner = node.args[0]
        if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "forward"):
            continue
        # the receiver of .forward(...) -- e.g. self.gate_up_proj
        recv = inner.func.value
        name = recv.attr if isinstance(recv, ast.Attribute) else getattr(recv, "id", "")
        if "gate_up" in name or "gateup" in name:
            hits.append((node.lineno, name))
    return hits


print("== no model may hand-roll silu_and_mul(gate_up.forward(x)) ==")
print("   (call Linear.forward_swiglu instead: identical bits when unfused, fused when it applies)")
offenders = []
for fn in sorted(os.listdir(MODELS)):
    if not fn.endswith(".py"):
        continue
    for lineno, recv in hand_rolled_swiglu(os.path.join(MODELS, fn)):
        offenders.append(f"{fn}:{lineno} (self.{recv})")
report("no hand-rolled gate_up+silu in models/", not offenders,
       "; ".join(offenders) if offenders else "all merged gate_up MLPs go through forward_swiglu")

# The helper must still BE the thing that chooses, or routing everyone to it achieves nothing.
lin = open(os.path.join(LAYERS, "linear.py")).read()
report("Linear.forward_swiglu still prefers the fused kernel",
       "apply_swiglu" in lin and "def forward_swiglu" in lin,
       "it must consult the quant method's apply_swiglu")
report("...and still falls back to the identical unfused expression",
       re.search(r"return\s+silu_and_mul\(self\.forward\(x\)\)", lin) is not None,
       "the fallback IS the expression this test bans elsewhere -- that is the point")

# Every quant method that can serve a merged gate_up should expose apply_swiglu, or the helper
# silently degrades for that scheme. This is the sibling-drift check.
method = open(os.path.join(LAYERS, "..", "quant", "method.py")).read()
classes = re.findall(r"^class (\w*LinearMethod)\b", method, re.M)
with_swiglu = set()
for m in re.finditer(r"^class (\w*LinearMethod)\b(.*?)(?=^class |\Z)", method, re.M | re.S):
    if "def apply_swiglu" in m.group(2):
        with_swiglu.add(m.group(1))
# Only the 4-bit weight methods have a fused gemm+silu kernel; fp8/unquantized legitimately do not.
FOUR_BIT = [c for c in classes if c in ("W4A8LinearMethod", "MxFp4LinearMethod", "NvFp4LinearMethod")]
missing = [c for c in FOUR_BIT if c not in with_swiglu]
report("every 4-bit linear method exposes apply_swiglu", not missing,
       f"missing: {missing}" if missing else f"{sorted(with_swiglu & set(FOUR_BIT))}")

print(f"\n{'FAILED: ' + '; '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
