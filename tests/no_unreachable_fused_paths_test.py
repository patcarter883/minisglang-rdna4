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


# ---------------------------------------------------------------------------------------------
# THE SAME DISEASE, ONE LAYER OUT: a knob the engine reads that a compose serve cannot set.
#
# `docker-compose.yml` enumerates the MINISGL_* variables it forwards, BY HAND. The engine reads
# 220 of them; compose forwards 129. The other 91 are unreachable on a compose serve -- setting one
# on the command line changes nothing, silently. That is not a theoretical problem: it invalidated
# an A/B of the fused gate_up+silu path twice in one session, because MINISGL_DENSE_FUSED_SILU=0
# never reached the container and BOTH legs ran the fused arm.
#
# This gate covers the knobs that TOGGLE A FAST PATH, because those are the ones whose
# unreachability silently corrupts a measurement rather than merely being inconvenient. The rest are
# listed, not failed -- a 91-item blocker would just get skipped.
#
# NOTE for whoever fixes the rest: do NOT mechanically add `MINISGL_X: "${MINISGL_X:-}"`. That sets
# the variable to the EMPTY STRING rather than leaving it unset, and 34 of the missing knobs are
# int()/float()-parsed, so it turns an unreachable knob into a boot crash. Use a passthrough that
# omits unset variables (an env_file, or the list form `- MINISGL_X`).
print()
print("== every fast-path toggle the engine reads must be reachable from a compose serve ==")
ENGINE = os.path.join(_HERE, "..", "python", "minisgl")
COMPOSE = os.path.join(_HERE, "..", "docker-compose.yml")
read = set()
for root, _dirs, files in os.walk(ENGINE):
    for fn in files:
        if not fn.endswith(".py"):
            continue
        t = open(os.path.join(root, fn)).read()
        read |= set(re.findall(r"""environ(?:\.get)?\(\s*["'](MINISGL_[A-Z0-9_]+)["']""", t))
        read |= set(re.findall(r"""getenv\(\s*["'](MINISGL_[A-Z0-9_]+)["']""", t))
forwarded = set(re.findall(r"^\s+(MINISGL_[A-Z0-9_]+):", open(COMPOSE).read(), re.M))
missing = read - forwarded
# A toggle is anything that switches an implementation on or off. These are the ones where an
# unreachable knob turns an A/B into new-vs-itself.
TOGGLE = re.compile(r"FUSED|REGDIRECT|_GEMV$|GEMV_|_FLAG$|SILU|W4A16|NVFP4_|BF16_|_ALIGN$")
# PRE-EXISTING DEBT, recorded 2026-09-17. Each of these gates a fast path and cannot be set on a
# compose serve, so any A/B of it silently measures the default on BOTH legs. Each needs one of two
# things, and DELETING is the preferred one: a merged fast path that has proved itself should not
# have a revert knob at all (the worktree is the isolation). Forward it only if it is a genuine
# operating lever someone still tunes. This list must only ever SHRINK -- a new entry means the
# mistake was repeated.
KNOWN_UNREACHABLE_TOGGLES = {
    "MINISGL_CCA_DECODE_FUSED",     # CCA decode fusion
    "MINISGL_GDN_FUSED_CONV",       # GDN fused conv1d      (tools/_gdn_ab.sh sets it via docker run)
    "MINISGL_GDN_FUSED_NORM",       # GDN fused norm        (tools/car_gdn_fused_smoke.sh)
    "MINISGL_MOE_ALIGN",
    "MINISGL_MOE_BF16_GEMV_MAX",
    "MINISGL_MOE_FLAG",             # tools/gemma4_swa_radix_validate.sh
    "MINISGL_MOE_MXFP4_REGDIRECT",
    "MINISGL_MOE_W8A8_REGDIRECT",
    "MINISGL_NVFP4_GEMV",
    "MINISGL_ZAYA_FUSED_MERGE",
}
bad = sorted(k for k in missing if TOGGLE.search(k) and k not in KNOWN_UNREACHABLE_TOGGLES)
stale = sorted(k for k in KNOWN_UNREACHABLE_TOGGLES if k not in missing)
report("no NEW fast-path toggle is unreachable from compose", not bad,
       ", ".join(bad) if bad else
       f"{len(read)} knobs read, {len(forwarded)} forwarded, "
       f"{len(KNOWN_UNREACHABLE_TOGGLES)} known-unreachable (delete them, do not add)")
report("the known-unreachable list has not gone stale", not stale,
       f"these are now reachable or deleted -- remove them from the list: {stale}" if stale
       else "every entry still describes a real gap")
if missing:
    print(f"   FYI {len(missing)} non-toggle knobs are also unreachable "
          f"({len([k for k in missing if k not in bad])} listed as informational only)")


# ---------------------------------------------------------------------------------------------
# THE FUSED ARM MUST REFUSE SCALES IT CANNOT TAKE.
#
# The two fused gate_up+silu kernels do NOT accept the same scale formats:
#   W4A16 `mmq_regdirect_w4a16_gemv_silu` carries all three WSP policies (fp16 group scale, MXFP4
#     E8M0 byte, NVFP4 e4m3 block + f32 global).
#   W4A8  `mmq_fp8_gemm_silu` is fp16-ONLY -- `TORCH_CHECK(scales.scalar_type() == at::kHalf)`.
# So an MXFP4/NVFP4 layer routed to the W4A8 arm does not fall back, it CRASHES in the op and takes
# the serve down at the first decode step. That is not hypothetical: it killed a spec-decode run on
# qwen38-27b (NVFP4) the first time the dense MLPs were routed through forward_swiglu, because the
# gate checked shape and never the scale dtype.
print()
print("== _fused_swiglu_ok must not be stricter than the kernels it gates ==")
try:
    import torch
    sys.path.insert(0, os.path.join(_HERE, "..", "python"))
    from minisgl.quant.method import _fused_swiglu_ok

    w = torch.empty(512, 64, dtype=torch.int32)   # N=512 (even), K/8
    # BOTH fused arms carry all three WSP policies. This briefly gated W4A8 to fp16 because an
    # MXFP4/NVFP4 layer crashed there -- but the crash was two STALE BINDING GUARDS (fp16-only
    # scales, and group_size % 32) refusing what launch_mmq_fp8_gemm_silu_gfx1201 computes. Both
    # now match their launcher; a scale-format gate here would make the exclusion permanent.
    for group, what in ((128, "AWQ int4 fp16 group scale"),
                        (32, "MXFP4 E8M0 byte scale"),
                        (16, "NVFP4 e4m3 block + f32 global")):
        report(f"group {group:<3} ({what}) reaches the fused arm",
               _fused_swiglu_ok(torch.empty(4, 512), w, group),
               "the kernel takes it -- the engine must not refuse it")
    report("shape gate still bites (M>16)",
           not _fused_swiglu_ok(torch.empty(17, 512), w, 128), "decode-only")
    report("shape gate still bites (odd N)",
           not _fused_swiglu_ok(torch.empty(4, 512), torch.empty(511, 64, dtype=torch.int32), 128),
           "N must be 2*inter")
except OSError as e:
    # minisgl imports the distributed runtime, which only exists inside the serve image. This is
    # NOT a skip: a skip that prints green is how a gate rots. Exit 2 -- distinct from pass (0) and
    # from fail (1) -- so a host run can never be mistaken for a clean one.
    print(f"  REQUIRES CONTAINER  cannot import minisgl here ({type(e).__name__}: {str(e)[:60]})")
    print("  run: docker run --rm -v <worktree>:/engine --entrypoint bash <image> -lc \\")
    print("         'source /opt/venv/bin/activate && python /engine/" + os.path.basename(__file__) + "'")
    print(f"\n{'FAILED: ' + '; '.join(FAILS) if FAILS else 'INCOMPLETE — the scale-format gate did not run'}")
    sys.exit(2)

print(f"\n{'FAILED: ' + '; '.join(FAILS) if FAILS else 'ALL PASS'}")
sys.exit(1 if FAILS else 0)
