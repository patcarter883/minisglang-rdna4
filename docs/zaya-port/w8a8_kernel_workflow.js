export const meta = {
  name: 'w8a8-fp8-moe-kernel',
  description: 'Build a native W8A8-fp8 grouped-MoE HIP kernel (full parity with W4A8), wire it into ZAYA, GPU-validate',
  phases: [
    { title: 'Scaffold', detail: 'create w8a8_fp8_wmma package: copy W4A8 sources, setup.py, op.py, .gitignore' },
    { title: 'Kernel', detail: 'edit .hip: fp8 weight load + per-N scale + fused silu/gemv/scatter; drop int4/zeros/e2m1' },
    { title: 'Bindings', detail: 'bindings.cpp TORCH_LIBRARY w8a8 surface + register_fake + python wrappers' },
    { title: 'Build', detail: 'AOT compile for gfx1201 (CPU, no lease); bounded compile-fix loop' },
    { title: 'Parity', detail: 'GPU: w8a8_moe vs fp8-dequant + fp32 reference; prefill+decode; bounded debug loop' },
    { title: 'Integrate', detail: 'kernels.w8a8_moe + _GroupedFP8Experts.post_load + MoELayer.forward swap' },
    { title: 'Validate', detail: 'GPU: ZAYA eager+graph coherence + decode TPOT A/B vs dequant path' },
  ],
}

const REPO = '/home/pat/code/minisgl-rdna4'
const SPEC = `${REPO}/docs/zaya-port/W8A8_KERNEL_SPEC.md`
const W4A8 = '/home/pat/code/vllm-gfx1201/w4a8_fp8_wmma'
const PKG = `${REPO}/w8a8_fp8_wmma`

const COMMON = `
You are building a native **W8A8-fp8 grouped-MoE HIP kernel** for gfx1201 (RDNA4) in the minisglang
repo ${REPO} (branch rdna4), to FULL PARITY with the existing W4A8 kernel, then wiring it into the ZAYA
model's fp8-expert MoE path (replacing the current fp8->bf16-dequant->Triton path).

READ FIRST (the authoritative implementation spec — it has the exact line refs + replacement code):
- ${SPEC}
The source-of-truth W4A8 kernel to adapt: ${W4A8}/ (moe_kernel.hip, moe_gemm_tiled.h, tile_config.h,
kernel_names.h, bindings.cpp, __init__.py, setup.py). Vendoring template: ${REPO}/rxf_hip/ and
${REPO}/cca_hip/ (flat pure-TORCH_LIBRARY package: setup.py flat _C name, bindings.cpp TORCH_LIBRARY,
op.py torch.ops.load_library + register_fake, __init__.py). Integration targets:
${REPO}/python/minisgl/quant/kernels.py (w4a8_moe = the parity template), ${REPO}/python/minisgl/layers/moe.py
(_GroupedFP8Experts + MoELayer.forward fp8 branch), ${REPO}/python/minisgl/models/zaya.py (ZayaMoEBlock).

W8A8 = fp8 (e4m3) weights with a PER-OUTPUT-CHANNEL fp32 scale + fp8 activations. It is a STRICT
SIMPLIFICATION of W4A8: identical WMMA core, identical activation quant, identical scatter/gather/reduce/
gemv plumbing. The ONLY changes (full table in the spec): (1) weight load = contiguous fp8 byte copy
instead of int4 nibble-unpack; (2) weight scale moves from per-K-group (in the K-loop) to per-N-channel
(once in the epilogue); (3) drop zeros / group-scale / weight_is_e2m1.

Rules: GPU work MUST go through /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh (absolute path) + the
container conventions in ${REPO}/CLAUDE.md / README.md (image vllm22-w4a8:combined, full device passthrough,
forward the lease HIP/ROCR pair, PYTHONPATH=/engine/python:/engine, reuse the warm triton cache, mount
/home/pat/models ro for the model). The AOT kernel BUILD is CPU-only — do NOT lease a GPU for it. Use
bounded timeouts so nothing holds a lease. msgpack is missing from the image venv — pip install msgpack
before any minisgl import. Match surrounding code style.
`.trim()

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    passed: { type: 'boolean' },
    summary: { type: 'string' },
    errorClass: { type: 'string', description: 'short failure tag, or "none"' },
    logTail: { type: 'string', description: 'last ~40 lines of build/test output or traceback' },
    suggestedFix: { type: 'string', description: 'concrete next patch if failed, else "none"' },
  },
  required: ['passed', 'summary', 'errorClass', 'logTail', 'suggestedFix'],
}

// ===========================================================================
phase('Scaffold')
log('Phase 1: scaffolding the w8a8_fp8_wmma package.')
await agent(`${COMMON}

TASK: Create the vendored package directory ${PKG}/ mirroring ${REPO}/rxf_hip/ structure. Steps:
- Copy the W4A8 MoE kernel sources you will adapt into ${PKG}/: moe_kernel.hip and the headers it
  #includes for the MoE path (moe_gemm_tiled.h, tile_config.h, kernel_names.h — check the actual
  #includes in moe_kernel.hip and copy exactly what the MoE launchers need; do NOT copy the dense-only
  gemm_tiled.h / w4a8_fp8_wmma_kernel.hip or the SWMMAC int4 headers unless an include requires them).
  Rename the kernel translation unit to w8a8_moe_kernel.hip.
- Write ${PKG}/setup.py (flat CUDAExtension name="w8a8_fp8_wmma_C", sources=[bindings.cpp,
  w8a8_moe_kernel.hip], GPU_ARCHS env default gfx1201, include_dirs=["/opt/rocm-7.2.1/include"] if the
  kernel uses rocwmma headers — check), mirroring rxf_hip/setup.py.
- Write ${PKG}/.gitignore (build/  *.so  *_hip.cpp  *_hip.hip  __pycache__/).
- Write ${PKG}/op.py (torch.ops.load_library on glob of w8a8_fp8_wmma_C*.so, with register_fake stubs
  for the ops — you can fill exact fakes in the Bindings phase; for now a correct skeleton) and
  __init__.py (from .op import ...). Mirror rxf_hip/op.py + __init__.py.
Do NOT build yet. Just create the files (verbatim copies + the package scaffolding). Return the file list
and the exact set of W4A8 source files copied (with their #include graph).`,
  { label: 'scaffold', phase: 'Scaffold', agentType: 'claude' })

// ===========================================================================
phase('Kernel')
log('Phase 2: editing the HIP kernel for fp8 weights.')
await agent(`${COMMON}

TASK: Edit ${PKG}/w8a8_moe_kernel.hip (the copied W4A8 MoE kernel) to compute W8A8-fp8 per the spec's
delta table. Apply ALL of these (line refs are in the W4A8 originals; find the analogous code in the copy):
- Weight HBM operand: int32 packed (E,N,K/8) + group scales + optional zeros  ->  fp8 (E,N,K) uint8
  (e4m3) + per-N fp32 scale (E,N). Update launcher signatures + pointer math (wq_e = w_fp8 + e*N*K).
- B stage: replace the nibble-unpack loop (moe_gemm_tiled.h ~104-122 and ~261-283; helper
  decode_w4_to_e4m3 in tile_config.h) with a contiguous fp8 byte copy into B_tile (vectorize as uint;
  K%16==0). Delete ppr/PACK_FACTOR/wz_e/zeros/decode_w4_to_e4m3/weight_is_e2m1.
- Scale: DELETE the per-K-group weight-scale fold in the K-loop (~144-151); accumulate raw acc. ADD the
  per-N weight scale ONCE in the epilogue alongside the existing per-token act scale (out =
  running*asc*w_scale_e[abs_n]); same for the SCATTER atomicAdd epilogue.
- Fused gemm1+silu (moe_gemm1_silu_v6_kernel): keep the gate|up two-slab structure; drop per-group fold;
  epilogue moe_silu_and_mul_h(run_g*asc*wsc_g_n, run_u*asc*wsc_u_n) with wsc_g_n=w_scale_e[abs_n],
  wsc_u_n=w_scale_e[inter+abs_n].
- GEMV decode (moe_gemv_v7_kernel): read fp8 weights directly from HBM (no decode_w4_to_f32/zp); per-N
  scale once in the epilogue.
- gather_reduce + the activation fp8 quant kernel (moe_compute_act_fp8_kernel) + the WMMA intrinsic +
  the A-shuffle/K-loop/accumulator structure + tiling/LDS budget: KEEP VERBATIM.
- Rename all launchers/symbols from the w4a8 names to w8a8 equivalents (launch_mmq_w8a8_moe_gemm,
  _gemm1_silu, _gemm_scatter, _gather_reduce). Keep the served WMMA (kernel id 6) + GEMV (7) paths.
Keep the 2:4-SWMMAC path OUT. Return a summary of every edit + any ambiguity you resolved.`,
  { label: 'kernel-edit', phase: 'Kernel', agentType: 'claude' })

// ===========================================================================
phase('Bindings')
log('Phase 3: bindings + python surface.')
await agent(`${COMMON}

TASK: Write ${PKG}/bindings.cpp and finalize ${PKG}/op.py for the W8A8 op surface (mirror W4A8
bindings.cpp + __init__.py, dropping w_zeros / weight_is_e2m1; scale per-channel). Register under
TORCH_LIBRARY(w8a8_fp8_wmma) + TORCH_LIBRARY_IMPL(..., CUDA) dispatching to the launch_* in
w8a8_moe_kernel.hip. NO PYBIND11_MODULE. Ops (every m.def carries {at::Tag::pt2_compliant_tag}):
  mmq_w8a8_moe_gemm(Tensor x, Tensor w_fp8, Tensor scales, Tensor sorted_token_ids, Tensor expert_ids,
     Tensor num_tokens_post_padded, int top_k, int block_m, int kernel) -> Tensor
  mmq_w8a8_moe_gemm1_silu(...same...) -> Tensor   # returns (P, inter)
  mmq_w8a8_moe_gemm_scatter(..., Tensor topk_weights, Tensor(a!) output, int top_k, int block_m, int kernel) -> ()
  mmq_w8a8_moe_gather_reduce(Tensor out2, Tensor sorted_token_ids, Tensor topk_weights, Tensor num_tokens_post_padded, int top_k) -> Tensor
Add the matching @torch.library.register_fake in op.py (correct output shapes: moe_gemm ->
(sorted_token_ids.shape[0], w_fp8.shape[1]) f16; gemm1_silu -> (..., w_fp8.shape[1]//2) f16; scatter ->
None; gather_reduce -> (topk_weights.shape[0]//top_k, out2.shape[1]) f32) and thin python wrappers
(resolve kernel name->int: wmma=6, gemv=7). TORCH_CHECK the dtype/shape contracts from the spec.
Return the final op schemas + wrapper signatures.`,
  { label: 'bindings', phase: 'Bindings', agentType: 'claude' })

// ===========================================================================
phase('Build')
log('Phase 4: AOT build (CPU, no lease) with bounded compile-fix loop.')
let buildV = null
for (let attempt = 1; attempt <= 4; attempt++) {
  buildV = await agent(`${COMMON}

TASK (build attempt ${attempt}/4): AOT-compile the package for gfx1201, CPU-only (NO gpu-lease):
  docker run --rm -v ${REPO}:/engine --entrypoint bash vllm22-w4a8:combined -lc \\
   'source /app/.venv/bin/activate && cd /engine/w8a8_fp8_wmma && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace 2>&1 | tail -60'
Then verify the .so loads: in the same image, python -c "import torch; torch.ops.load_library(glob...);
print([o for o in dir(torch.ops.w8a8_fp8_wmma)])". passed=true only if it compiles AND the ops register.
Put the last ~40 lines of compiler output (or the first error) in logTail.`,
    { label: `build-try${attempt}`, phase: 'Build', schema: VERDICT_SCHEMA, agentType: 'claude' })
  log(`Build attempt ${attempt}: ${buildV?.passed ? 'PASS' : 'FAIL'} — ${buildV?.summary || ''}`)
  if (buildV?.passed) break
  if (attempt < 4 && buildV) {
    await agent(`${COMMON}

The W8A8 kernel build attempt ${attempt} FAILED. Fix the HIP/C++ in ${PKG} so it compiles for gfx1201.
Diagnose from this output; keep the W8A8 semantics from the spec. Make the smallest correct change.
errorClass: ${buildV.errorClass}
suggestedFix: ${buildV.suggestedFix}
logTail:
${buildV.logTail}
Return what you changed.`,
      { label: `build-fix-${attempt}`, phase: 'Build', agentType: 'claude' })
  }
}

// ===========================================================================
phase('Parity')
log('Phase 5: GPU parity test (bounded debug loop).')
let parityV = null
for (let attempt = 1; attempt <= 3; attempt++) {
  parityV = await agent(`${COMMON}

TASK (parity attempt ${attempt}/3): Write ${PKG}/test_w8a8_parity.py and run it ON GPU (lease 1 card,
container recipe per CLAUDE.md). It must validate the new w8a8 MoE kernel against references for BOTH
prefill (M large -> wmma, kernel=6) and decode (M<=2 -> gemv gemm1 + scatter gemm2):
  - Build random per-expert fp8 weights w13 (E, 2*inter, K) e4m3 + per-N f32 scale, w2 (E, K, inter) +
    scale; random fp16 x; top_k=1 routing (sorted_token_ids/expert_ids/num_tokens_post_padded via the
    same moe_align the engine uses). E.g. E=16, K=2048, inter=4096 (ZAYA dims).
  - Reference A: the CURRENT path — dequant fp8->bf16 (w.float()*scale) then minisgl
    fused_experts_impl. Reference B: an fp32 torch reference of silu_and_mul(x@w13^T-ish with per-token
    act-fp8 quant + per-N scale) @ w2. Compare the full gemm1->silu->gemm2 pipeline driven through
    kernels.w8a8_moe (or the raw ops) vs the references. Assert rel-err within fp8 tolerance (qk-style
    ~1e-2..1e-1 abs depending on magnitude; compare to ref A which is itself bf16, and sanity-check
    direction vs ref B). Print max/rel err per stage + PASS/FAIL and sys.exit nonzero on FAIL.
passed=true only if parity holds for prefill AND decode. logTail = the err lines + PASS/FAIL.`,
    { label: `parity-try${attempt}`, phase: 'Parity', schema: VERDICT_SCHEMA, agentType: 'claude' })
  log(`Parity attempt ${attempt}: ${parityV?.passed ? 'PASS' : 'FAIL'} — ${parityV?.summary || ''}`)
  if (parityV?.passed) break
  if (attempt < 3 && parityV) {
    await agent(`${COMMON}

W8A8 parity attempt ${attempt} FAILED. Fix the kernel (${PKG}) OR the test if the test is wrong — decide
which from the evidence. Most likely culprits: the per-N scale indexing in the epilogue, the fp8 byte
copy stride/LDS layout, gate|up half ordering in fused silu, or the gemv decode weight read. Verify
against ${W4A8} source. Re-build if you change the kernel (CPU, no lease). Return what you changed.
errorClass: ${parityV.errorClass}
suggestedFix: ${parityV.suggestedFix}
logTail:
${parityV.logTail}`,
      { label: `parity-fix-${attempt}`, phase: 'Parity', agentType: 'claude' })
  }
}

// ===========================================================================
phase('Integrate')
log('Phase 6: wiring into minisgl ZAYA MoE.')
await agent(`${COMMON}

TASK: Wire the validated kernel into the ZAYA fp8-expert path (spec "minisgl integration"):
1. ${REPO}/python/minisgl/quant/kernels.py: add w8a8_moe(...) mirroring w4a8_moe's body verbatim
   (per-GEMM kernel pick gemm1 gemv-if-M<=2 else wmma / gemm2 wmma; precomputed-route branch; moe_align;
   gemm1 -> tail_hip.silu_and_mul -> decode scatter (M<=2, MINISGL_MOE_SCATTER) OR prefill gather_reduce;
   reuse env gates). Drop the zeros/group args; scales are per-N.
2. ${REPO}/python/minisgl/layers/moe.py: add _GroupedFP8Experts.post_load building self._w_op (E,N,K
   e4m3 op-layout) + self._scales_op (E,N f32 = weight_scale.squeeze(-1).float()), then del the
   checkpoint buffers. Swap the if self.fp8_experts forward branch (currently dequant()+fused_experts_impl)
   to call kernels.w8a8_moe(hidden_states, w13._w_op, w13._scales_op, w2._w_op, w2._scales_op, None,
   self.top_k, self.renormalize, topk_weights=topk_weights, topk_ids=topk_ids).
Keep import-clean. post_load dispatch already works via BaseOP recursion (no engine/model change). If the
kernel needs a transposed op-layout (not natural (E,N,K)), do the transpose per-expert in post_load.
Return the diff summary.`,
  { label: 'integrate', phase: 'Integrate', agentType: 'claude' })

// ===========================================================================
phase('Validate')
log('Phase 7: GPU coherence (eager+graph) + decode TPOT A/B.')
let valV = null
for (let attempt = 1; attempt <= 3; attempt++) {
  valV = await agent(`${COMMON}

TASK (validate attempt ${attempt}/3): GPU-validate ZAYA1-8B-fp8 on the new W8A8 kernel path (lease 1 card,
container recipe; weights /home/pat/models/ZAYA1-8B-fp8 ro; pip install msgpack; MINISGL_MOE_SCATTER=0).
  - Coherence EAGER: run tools/zaya_coherence_smoke.py -> must stay coherent ("...Paris", "...4").
  - Coherence GRAPH: run tools/zaya_graph_smoke.py (cuda_graph_max_bs=8) -> captures + coherent. NOTE the
    decode scatter atomicAdd is NOT graph-capturable; for the graph run ensure the prefill gather_reduce
    path (or a capturable gemm2) is used at decode under capture (set MINISGL_MOE_SCATTER=0 and confirm
    w8a8_moe's M<=2 branch respects it — if scatter is gated off it must fall back to gather_reduce).
  - Perf A/B (decode TPOT, M=1): measure the new w8a8 path vs the OLD dequant->Triton path (toggle via a
    temporary env/flag or git stash of the moe.py swap). Report both TPOT numbers + speedup.
passed=true only if BOTH coherence runs pass AND you have A/B TPOT numbers. logTail = outputs + numbers.`,
    { label: `validate-try${attempt}`, phase: 'Validate', schema: VERDICT_SCHEMA, agentType: 'claude' })
  log(`Validate attempt ${attempt}: ${valV?.passed ? 'PASS' : 'FAIL'} — ${valV?.summary || ''}`)
  if (valV?.passed) break
  if (attempt < 3 && valV) {
    await agent(`${COMMON}

ZAYA W8A8 validation attempt ${attempt} FAILED. Fix the kernel or integration. Verify against ${W4A8}
+ the spec. Re-build the kernel if changed (CPU, no lease). Return what you changed.
errorClass: ${valV.errorClass}
suggestedFix: ${valV.suggestedFix}
logTail:
${valV.logTail}`,
      { label: `validate-fix-${attempt}`, phase: 'Validate', agentType: 'claude' })
  }
}

// ===========================================================================
return {
  build: buildV?.passed ? 'PASS' : 'FAIL',
  parity: parityV?.passed ? 'PASS' : 'FAIL',
  paritySummary: parityV?.summary,
  validation: valV?.passed ? 'PASS' : 'FAIL',
  validationSummary: valV?.summary,
  notes: 'Package: w8a8_fp8_wmma/; spec: docs/zaya-port/W8A8_KERNEL_SPEC.md; integration in quant/kernels.py + layers/moe.py',
}
