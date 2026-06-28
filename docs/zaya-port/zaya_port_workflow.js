export const meta = {
  name: 'zaya-port-into-minisgl',
  description: 'Port ZAYA1-8B (CCA+MoE hybrid) into minisglang, validate on GPU, then add a Markovian RSA shim',
  phases: [
    { title: 'Map framework', detail: 'map minisglang model/cache framework (GDN analog)' },
    { title: 'Design', detail: 'design the port: modules, state-cache wiring, MoE, weight loading' },
    { title: 'Implement', detail: 'write zaya.py + registry + CCA + MoE + weight loading (sequential)' },
    { title: 'Review', detail: 'adversarial review across correctness dimensions vs the reference' },
    { title: 'Fix', detail: 'apply confirmed review findings' },
    { title: 'Validate kernel', detail: 'GPU: run vendored cca_hip parity tests via gpu-lease' },
    { title: 'Validate model', detail: 'GPU: load ZAYA1-8B-fp8 in minisgl, coherence prompt, bounded debug loop' },
    { title: 'RSA shim', detail: 'implement the Markovian RSA shim proxy in minisglang' },
  ],
}

// ---------------------------------------------------------------------------
// Shared context every agent needs. Paths are absolute; agents run with cwd =
// the minisgl repo (/home/pat/code/minisgl-rdna4).
// ---------------------------------------------------------------------------
const REPO = '/home/pat/code/minisgl-rdna4'
const REF_DOC = `${REPO}/docs/zaya-port/ZAYA_REFERENCE.md`
const FRAMEWORK_DOC = `${REPO}/docs/zaya-port/MINISGL_FRAMEWORK.md`
const PLAN_DOC = `${REPO}/docs/zaya-port/PORT_PLAN.md`
const MODEL_FILE = `${REPO}/python/minisgl/models/zaya.py`
const VLLM = '/home/pat/code/vllm-gfx1201'

const COMMON = `
You are working on porting the ZAYA1-8B model into the minisglang inference engine at ${REPO}
(branch rdna4). minisglang is a native-HIP inference engine for RDNA4 (gfx1201) — NOT vLLM.

Read these first (they are the shared knowledge for this port):
- ${REF_DOC}  — AI-synthesized map of the ZAYA reference architecture (VERIFY claims against source).
- ${FRAMEWORK_DOC}  — map of minisglang's model/cache framework (the GDN hybrid path is the analog). [created in phase 1]
- ${PLAN_DOC}  — the agreed port plan. [created in phase 2]
- ${REPO}/CLAUDE.md  — repo rules. GPU work MUST go through the shared arbiter
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh (absolute path) and the container conventions in README.md.

The vendored CCA kernel already exists in this repo at ${REPO}/cca_hip/ and registers
torch.ops.zaya_cca.{conv_state_decode,cca_decode_qk,cca_prefill_qk}. The register_fake signatures in
${REPO}/cca_hip/cca_op.py are the AUTHORITATIVE current kernel arg order — trust that file over the
reference doc when they disagree. The reference vLLM implementation (source of truth for model math)
is under ${VLLM}/zaya/ — read it when the docs are ambiguous.

Match the surrounding minisglang code style. The closest existing model to study/copy is the GDN
hybrid path: python/minisgl/models/qwen3_5.py and python/minisgl/gdn/ (conv + recurrent state cache).
`.trim()

const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          file: { type: 'string' },
          title: { type: 'string' },
          detail: { type: 'string' },
          suggestedFix: { type: 'string' },
        },
        required: ['severity', 'file', 'title', 'detail', 'suggestedFix'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    passed: { type: 'boolean' },
    summary: { type: 'string' },
    errorClass: { type: 'string', description: 'short tag for the failure, or "none"' },
    logTail: { type: 'string', description: 'last ~40 lines of the run log / traceback' },
    suggestedFix: { type: 'string', description: 'concrete next patch if failed, else "none"' },
  },
  required: ['passed', 'summary', 'errorClass', 'logTail', 'suggestedFix'],
}

// ===========================================================================
phase('Map framework')
log('Phase 1: mapping the minisglang model/cache framework (GDN analog).')
const mapRes = await agent(`${COMMON}

TASK: Produce a precise, code-quoting map of minisglang's model framework and WRITE it to
${FRAMEWORK_DOC} (create it). This doc is what every later phase relies on, so be exhaustive and
concrete (file:line references + verbatim snippets). Cover:

1. The model base contract: python/minisgl/models/base.py — the class every model implements, its
   __init__ signature, the forward() signature and EXACTLY what objects/metadata it receives.
2. Registration: python/minisgl/models/register.py — the _MODEL_REGISTRY, how an HF architectures[]
   string maps to (module, class), and the precise edit to add "ZayaForCausalLM".
3. ModelConfig: python/minisgl/models/config.py — how HF config fields are surfaced to the model.
4. Weight loading: python/minisgl/models/weight.py + utils.py — how safetensors names map to params,
   how quantized (fp8/w4a8) weights load, the helper(s) a model uses.
5. THE GDN HYBRID PATH (study closely — the analog for CCA's conv/recurrent state):
   python/minisgl/models/qwen3_5.py, qwen3_5_moe.py, python/minisgl/gdn/layer.py + metadata.py.
   How layers are built + scheduled, how the conv/recurrent STATE CACHE is declared, sized, allocated,
   indexed per-sequence (slot/block ids), and updated; how prefill vs decode is dispatched; how native
   HIP ops are called.
6. Engine forward/metadata: search python/minisgl for forward_batch/ForwardMetadata/ForwardMode and
   describe what a layer can learn each step: which requests, seq positions, slot/page mapping, and
   prefill-vs-decode signalling. How the normal attention KV cache is allocated/addressed (CCA also
   needs a paged attention KV cache after the conv, per the reference doc — confirm the mechanism).
7. MoE building blocks: how an existing MoE model (qwen3_moe.py / glm4_moe_lite.py) implements fused
   experts + top-k routing + shared expert + fp8 experts — the exact ops/classes to reuse.
8. Serve/run entry + how --attn hip and graph capture interact with a model; any per-model hook for
   declaring state-cache shapes.

After writing the doc, return a 10-line summary of the single most important wiring decisions for the
Zaya port (especially: does a CCA layer use BOTH a conv state cache AND a paged attention KV cache,
and how is each addressed).`,
  { label: 'map-framework', phase: 'Map framework', agentType: 'claude' })

// ===========================================================================
phase('Design')
log('Phase 2: designing the port.')
const designRes = await agent(`${COMMON}

The framework map now exists at ${FRAMEWORK_DOC}. Reference map at ${REF_DOC}.

TASK: Design the ZAYA port and WRITE it to ${PLAN_DOC}. Be concrete and decision-complete so the
implementers do not have to make architecture calls. Include:
- The module breakdown for python/minisgl/models/zaya.py (classes: model, decoder layer(s), CCA
  module, MoE/router module) mapped onto minisglang's base contract.
- The layer schedule (even=CCA, odd=MoE — VERIFY from ${VLLM}/zaya source) and how to express it.
- EXACT state-cache plan: the CCA conv_states [NB,1280,2] + prev_hs [NB,2048] float32 caches AND the
  post-conv paged attention KV cache — how each is declared/sized/allocated/indexed in minisglang
  (mirror the GDN path for the conv/recurrent state; mirror an existing attention model for the KV).
- The prefill and decode forward paths, naming the exact torch.ops.zaya_cca.* calls and their args
  (cross-check cca_op.py), how slot/seg_pos/req_id/is_last/is_pad are built from minisglang metadata,
  RoPE (partial 0.5), and the attention call.
- MoE plan: router (down_proj/RMSNorm/router_mlp/EDA/MOD/balancing_biases), top-1, fp8 fused experts —
  which minisglang ops to reuse, and how EDA state threads across layers in forward().
- Weight-loading plan: name->module mapping incl. the local_experts fc1 split (gate w1 / up w3) and
  fp8 scales; tied embeddings.
- A risk list: where the reference doc is uncertain and must be confirmed against source during impl.
- An ordered implementation checklist split into the 3 implementer steps below.

Return a short confirmation + the top 5 risks.`,
  { label: 'design', phase: 'Design', agentType: 'claude' })

// ===========================================================================
phase('Implement')
log('Phase 3: implementing (sequential — shared working tree, no parallel file writes).')

const impl1 = await agent(`${COMMON}

The plan is at ${PLAN_DOC}. IMPLEMENT STEP 1 of 3 — scaffold + registration + model shell.

Write python/minisgl/models/zaya.py with: the config plumbing, the top-level model class implementing
minisglang's base contract (embeddings, the 80-layer tower with the CCA/MoE schedule as stubs that you
will fill in steps 2-3, final RMSNorm, tied LM head), and the forward() loop threading residual (fp32)
and the EDA router-hidden state across layers. Register "ZayaForCausalLM" in
python/minisgl/models/register.py. Add any config handling needed in python/minisgl/models/config.py.
Make the file import cleanly (python -c "import ..." level) even with the CCA/MoE bodies stubbed
(raise NotImplementedError in the stubs is fine). Do NOT break any existing model.
Return what you wrote + the exact remaining stubs for steps 2-3.`,
  { label: 'impl-1-scaffold', phase: 'Implement', agentType: 'claude' })

const impl2 = await agent(`${COMMON}

The plan is at ${PLAN_DOC}; step-1 scaffold is in ${MODEL_FILE}. IMPLEMENT STEP 2 of 3 — the CCA layer.

Fill in the CCA module: linear_q/linear_k/val_proj1/val_proj2/o_proj, conv weight params (w0/b0/w1/b1
laid out as the kernel expects — see cca_op.py + cca_kernel.hip), per-k-head temp. Wire BOTH state
caches per the plan: the conv_states+prev_hs recurrent caches (mirror the GDN state-cache pattern) and
the post-conv paged attention KV cache. Implement prefill and decode forward paths calling
torch.ops.zaya_cca.cca_prefill_qk / cca_decode_qk (build slot/seg_pos/req_id/is_last/is_pad from
minisglang's forward metadata), then partial RoPE (factor 0.5) and the attention call, then o_proj.
Keep an eager fallback path only if it's cheap; the kernel path is primary. Make it import-clean.
Return the CCA forward summary + any metadata fields you needed that the scaffold/framework must provide.`,
  { label: 'impl-2-cca', phase: 'Implement', agentType: 'claude' })

const impl3 = await agent(`${COMMON}

The plan is at ${PLAN_DOC}; scaffold+CCA are in ${MODEL_FILE}. IMPLEMENT STEP 3 of 3 — MoE + weights.

Fill in the MoE/router module (down_proj 2048->256, RMSNorm(256), router_mlp ->16, EDA via threaded
router-hidden + router_states_scale, MOD skip-expert, balancing_biases, top-1) and the fp8 fused
experts (reuse minisglang's existing fused-MoE op; confirm the exact one from glm4_moe_lite.py /
qwen3_moe.py). Implement load_weights: the full name->module mapping incl. local_experts.{e}.linear_fc1
split into gate(w1)/up(w3) and linear_fc2->w2, fp8 weight_scales, tied embeddings, conv_qk.{0,1}
weight/bias -> the kernel's w0/b0/w1/b1 layout (apply the transpose to w1 the kernel expects).
Make the whole module import-clean. Return a summary + any loose ends.`,
  { label: 'impl-3-moe-weights', phase: 'Implement', agentType: 'claude' })

// ===========================================================================
phase('Review')
log('Phase 4: adversarial review across correctness dimensions.')
const DIMENSIONS = [
  { key: 'cca-math', prompt: 'CCA forward correctness: conv state layout/roll, w1 transpose, grouped-mean injection, per-head RMS-norm, temp, partial RoPE, the post-conv attention call, and that the torch.ops.zaya_cca.* args EXACTLY match cca_op.py.' },
  { key: 'state-cache', prompt: 'State-cache wiring: are conv_states/prev_hs AND the paged attention KV cache declared/sized/allocated/indexed correctly per minisglang metadata? slot/seg_pos/req_id/is_last/is_pad built correctly for both prefill and decode? padding handled?' },
  { key: 'moe-router', prompt: 'MoE/router correctness: down_proj/RMSNorm/router_mlp dims, EDA threading across layers, MOD skip-expert, balancing_biases, top-1, and the fused fp8 expert op usage.' },
  { key: 'weights', prompt: 'Weight loading: every ZAYA weight name mapped, local_experts fc1 gate/up split, fc2->w2, fp8 scales, tied embeddings, conv weights -> kernel w0/b0/w1/b1 layout incl. transpose. Any unmapped or mis-shaped param?' },
  { key: 'integration', prompt: 'Registration + forward loop + base-contract conformance + import-cleanliness; does it break any existing model? residual fp32 threading; final norm + tied lm head.' },
]
const reviews = await parallel(DIMENSIONS.map(d => () =>
  agent(`${COMMON}

The implementation is in ${MODEL_FILE} (plus edits to register.py/config.py). REVIEW this dimension and
report concrete, real defects only (verify against ${VLLM}/zaya source and ${REPO}/cca_hip/cca_op.py):

${d.prompt}

For each finding give severity, file, a precise title, the detail (with line refs), and a concrete
suggestedFix. Be adversarial but do not invent issues.`,
    { label: `review:${d.key}`, phase: 'Review', schema: FINDINGS_SCHEMA, agentType: 'Explore' })))
const allFindings = reviews.filter(Boolean).flatMap(r => r.findings || [])
const actionable = allFindings.filter(f => f.severity === 'high' || f.severity === 'medium')
log(`Review: ${allFindings.length} findings, ${actionable.length} actionable (high/medium).`)

// ===========================================================================
phase('Fix')
if (actionable.length) {
  log(`Phase 5: applying ${actionable.length} actionable findings.`)
  await agent(`${COMMON}

Apply the following review findings to the code (edit the files in the tree). Only apply fixes you can
confirm are correct against the reference source + cca_op.py; if a finding is wrong, skip it and say so.
Keep the module import-clean after your edits.

FINDINGS (JSON):
${JSON.stringify(actionable, null, 2)}

Return what you changed and what you deliberately skipped (with reasons).`,
    { label: 'fix', phase: 'Fix', agentType: 'claude' })
} else {
  log('Phase 5: no actionable findings — skipping fix.')
}

// ===========================================================================
phase('Validate kernel')
log('Phase 6: GPU — validating the vendored CCA kernel (parity tests) via gpu-lease.')
const kernelVerdict = await agent(`${COMMON}

TASK: Validate that the vendored CCA kernel in ${REPO}/cca_hip builds/loads and passes its standalone
numeric parity tests ON GPU. Follow the repo CLAUDE.md container conventions:
- Lease 1 GPU via the ABSOLUTE-PATH arbiter: /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 -- <cmd>
- Run inside the shared ROCm image vllm22-w4a8:combined with the mandatory device passthrough, mounting
  this repo at /engine, activating /app/.venv, and forwarding the lease's HIP/ROCR_VISIBLE_DEVICES pair
  (see README.md "Running" + CLAUDE.md; do NOT set both to LEASE_ROCR_DEVICES).
- The .so may need an AOT rebuild for this checkout: cd /engine/cca_hip && GPU_ARCHS=gfx1201 python
  setup.py build_ext --inplace, then run the parity tests (test_cca_kernel.py, test_cca_decode_qk.py,
  test_cca_prefill_qk.py / run_parity.sh). Use a bounded timeout so a hang can't hold the lease.
Report passed=true only if the parity tests actually pass. Put the last ~40 lines of output in logTail.`,
  { label: 'validate-kernel', phase: 'Validate kernel', schema: VERDICT_SCHEMA, agentType: 'claude' })
log(`Kernel validation: ${kernelVerdict?.passed ? 'PASS' : 'FAIL'} — ${kernelVerdict?.summary || 'no result'}`)

// ===========================================================================
phase('Validate model')
log('Phase 7: GPU — loading ZAYA1-8B-fp8 in minisglang (bounded debug loop).')
let modelVerdict = null
const MAX_ATTEMPTS = 3
for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
  modelVerdict = await agent(`${COMMON}

TASK (attempt ${attempt}/${MAX_ATTEMPTS}): Load ZAYA1-8B-fp8 in minisglang and run a short coherence
generation, ON GPU, single-card (CCA has no TP>1). Weights are at /home/pat/models/ZAYA1-8B-fp8
(fp8, ~9.4GB; mount read-only). Follow CLAUDE.md: lease 1 GPU via the absolute-path arbiter, run in
vllm22-w4a8:combined with full device passthrough, mount this repo at /engine, set
PYTHONPATH=/engine/python:/engine, forward the lease HIP/ROCR pair, reuse the warm triton cache.
Use minisglang's own serve/offline entry (find it — likely a tools/ or python/minisgl entrypoint;
match the production serve config in the repo: --attn hip etc. as applicable to a non-graph first try).
Prompt e.g. "The capital of France is". CCA state numerics need float32 (mamba-cache-dtype float32 in
vLLM — find the minisglang equivalent). Use bounded timeouts so nothing holds the lease.

Report passed=true ONLY if it loads and emits coherent text. If it fails, capture the FULL traceback in
logTail, classify it in errorClass, and give the single most likely concrete fix in suggestedFix.`,
    { label: `validate-model-try${attempt}`, phase: 'Validate model', schema: VERDICT_SCHEMA, agentType: 'claude' })
  log(`Model load attempt ${attempt}: ${modelVerdict?.passed ? 'PASS' : 'FAIL'} — ${modelVerdict?.summary || ''}`)
  if (modelVerdict?.passed) break
  if (attempt < MAX_ATTEMPTS && modelVerdict) {
    log(`Patching based on: ${modelVerdict.errorClass} -> ${modelVerdict.suggestedFix?.slice(0, 120)}`)
    await agent(`${COMMON}

The ZAYA model load attempt ${attempt} FAILED. Fix the code in the tree so the next load attempt can
progress. Diagnose from this traceback and proposed fix, but verify against the reference source +
cca_op.py before editing. Make the smallest correct change; keep the module import-clean.

errorClass: ${modelVerdict.errorClass}
suggestedFix: ${modelVerdict.suggestedFix}
logTail:
${modelVerdict.logTail}

Return what you changed.`,
      { label: `debug-fix-${attempt}`, phase: 'Validate model', agentType: 'claude' })
  }
}

// ===========================================================================
phase('RSA shim')
log('Phase 8: implementing the Markovian RSA shim proxy inside minisglang.')
const rsaRes = await agent(`${COMMON}

TASK: Implement a **Markovian RSA shim proxy** inside minisglang's framework. RSA = Recursive
Self-Aggregation test-time compute. The vLLM reference exists as a standalone OpenAI-compatible proxy:
study ${VLLM}/Dockerfile.rsa and ${VLLM}/rsa/ (config N/K/T/tail-tokens; it sits in front of a backend
and aggregates multiple samples). Read it to understand the RSA loop, then implement the **Markovian**
variant — where each aggregation round conditions only on the previous round's aggregated state (a
Markov chain over rounds) rather than the full history — as a shim that fits minisglang's serving
framework (find how minisglang exposes its OpenAI-compatible server / engine API and integrate there
rather than copying vLLM's standalone proxy verbatim).

Match minisglang's code style and server abstractions. Make it import-clean and add a brief usage note
to docs/zaya-port/. If the precise "Markovian" semantics are ambiguous, implement the most faithful
interpretation (previous-round-only conditioning), document the assumption clearly, and make the round
count / sample count configurable. Return a summary of what you built, where it plugs in, and how to
invoke it.`,
  { label: 'rsa-shim', phase: 'RSA shim', agentType: 'claude' })

// ===========================================================================
return {
  kernelValidation: kernelVerdict?.passed ? 'PASS' : 'FAIL',
  kernelSummary: kernelVerdict?.summary,
  modelValidation: modelVerdict?.passed ? 'PASS' : 'FAIL',
  modelSummary: modelVerdict?.summary,
  modelErrorClass: modelVerdict?.passed ? 'none' : modelVerdict?.errorClass,
  reviewFindings: allFindings.length,
  actionableFindings: actionable.length,
  rsaShim: rsaRes ? 'implemented' : 'failed',
  notes: 'See docs/zaya-port/{ZAYA_REFERENCE,MINISGL_FRAMEWORK,PORT_PLAN}.md and python/minisgl/models/zaya.py',
}
