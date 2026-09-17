export const meta = {
  name: 'dp-ep-serving',
  description: 'Add a DP launcher (single endpoint, replicas+routing) + expert parallelism to minisglang; GPU-validate ZAYA with and without EP',
  phases: [
    { title: 'DP launcher', detail: 'dp_size launcher: spawn replicas, per-replica request routing, DP info plumbing' },
    { title: 'DP validate', detail: 'GPU 2-card UNDER GRAPH: dp=2 coherent + ~2x throughput (timeout-bounded)' },
    { title: 'EP impl', detail: 'expert shard + MoELayer all_gather/all_reduce IN-GRAPH + common-bs lockstep (capturable)' },
    { title: 'EP validate', detail: 'GPU 2-card UNDER GRAPH: dp=2 --enable-ep coherent + greedy-parity vs DP-only' },
    { title: 'AB report', detail: 'with vs without EP: KV pool size, throughput, top-1 skew' },
  ],
}

const REPO = '/home/pat/code/minisgl-rdna4'
const SPEC = `${REPO}/docs/zaya-port/DP_EP_SPEC.md`
const LEASE = '/home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh'

const COMMON = `
You are adding native **data-parallel (DP) + expert-parallel (EP)** serving to the minisglang engine at
${REPO} (branch rdna4), for the ZAYA1-8B-fp8 model on 2x gfx1201. CCA cannot tensor-parallelize, so the
attention/CCA backbone is always REPLICATED (DP); EP only shards the MoE experts.

READ FIRST: ${SPEC} (the design spec with exact file seams + the all_gather+all_reduce EP approach).
Key files: server/{launch.py,args.py,api_server.py}, scheduler/io.py, distributed/{impl.py,info.py},
engine/engine.py, layers/moe.py, models/weight.py, quant/kernels.py (w8a8_moe), models/zaya.py.

DP launcher = ONE OpenAI endpoint that internally routes each request to ONE of dp_size full-model
replicas (today rank 0 PUB-broadcasts every request to ALL ranks — that is the TP-lockstep fan-out DP
must replace with per-replica routing). EP (toggle --enable-ep) shards experts across the DP ranks and
combines via all_gather + masked local-expert compute + all_reduce (reuses existing collectives; NO new
all_to_all/C++). EP needs per-step lockstep: an idle replica must run a dummy forward or the all_reduce
deadlocks.

Rules: GPU work via ${LEASE} (absolute path) + the container conventions in ${REPO}/CLAUDE.md
(image vllm22-w4a8:combined, full device passthrough, forward the lease HIP/ROCR pair,
PYTHONPATH=/engine/python:/engine, mount /home/pat/models ro, warm triton cache). pip install msgpack
before any minisgl import. GPU tests need TWO cards: lease with \`-n 2\`. EVERY GPU command MUST be
wrapped in a hard timeout (a deadlocked lockstep test must TIME OUT and report FAIL, never hang).
Match surrounding code style; keep all existing models working (DP/EP must be inert when dp_size=1).
`.trim()

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    passed: { type: 'boolean' },
    summary: { type: 'string' },
    errorClass: { type: 'string' },
    logTail: { type: 'string' },
    suggestedFix: { type: 'string' },
    metrics: { type: 'string', description: 'measured numbers (throughput/KV/etc.) or "none"' },
  },
  required: ['passed', 'summary', 'errorClass', 'logTail', 'suggestedFix', 'metrics'],
}

// ===========================================================================
phase('DP launcher')
log('Phase 1: implementing the DP launcher (replicas + per-replica request routing).')
await agent(`${COMMON}

IMPLEMENT the DP launcher (EP OFF for now), per ${SPEC} "DP launcher" section. Concretely:
- args.py: add --data-parallel-size/--dp-size (default 1). Validate dp_size*tp_size processes.
- distributed/info.py: add DP coordinates — a DpInfo(dp_rank, dp_size) + global _DP_INFO with
  get_dp_info(), mirroring _TP_INFO. Plumb (dp_rank, tp_rank) to each spawned scheduler.
- server/launch.py: spawn dp_size*tp_size schedulers, each tagged with its (dp_rank, tp_rank); each
  builds its own full-model engine replica. Assign each replica its own device (device index = dp_rank
  when launched on a multi-card host; respect HIP/ROCR_VISIBLE_DEVICES from the lease).
- scheduler/io.py: replace the rank-0 PUB-broadcast-to-ALL-ranks with PER-REPLICA routing — each
  UserMsg is delivered to exactly ONE replica (round-robin or least-pending), then within a replica
  keep the existing tp broadcast. Replies unchanged (front-end demuxes by globally-unique uid); track
  which replica owns each uid for aborts.
- engine/engine.py: create a DP process group (torch.distributed.new_group over dp ranks) for later EP
  use; scope the >2GB free-memory imbalance guard to WITHIN a replica (relax across replicas). With EP
  off, replicas are INDEPENDENT — no per-step cross-replica collective; each schedules/forwards its own
  queue.
- Keep dp_size=1 a complete no-op (every existing model/serve path unchanged).
Make it import-clean. Return the files changed + how routing + reply-demux work + how a replica gets its
device.`,
  { label: 'dp-launcher', phase: 'DP launcher', agentType: 'claude' })

// ===========================================================================
phase('DP validate')
log('Phase 2: GPU 2-card validation of DP-only serving.')
let dpV = null
for (let attempt = 1; attempt <= 3; attempt++) {
  dpV = await agent(`${COMMON}

VALIDATE (attempt ${attempt}/3) the DP launcher on 2 cards, EP OFF, **UNDER CUDA GRAPH CAPTURE** (graph
is the gate — see ${SPEC} "Graph capture"). Lease 2 cards (${LEASE} -n 2) and in the container bring up
the ZAYA serve with --data-parallel-size 2 (one HTTP endpoint), model /home/pat/models/ZAYA1-8B-fp8,
--attn hip, **graph capture ENABLED** (the serve graph flag / cuda_graph_max_bs>0, e.g. --graph 16),
MINISGL_MOE_SCATTER=0. Wrap EVERYTHING in hard timeouts (timeout 600 ...; a hang => FAIL).
- Confirm each replica actually CAPTURED + REPLAYS graphs (grep the serve log for "capturing CUDA
  graphs" + that decode replays, not a silent eager fallback).
- Coherence: send a prompt -> coherent text; several concurrent requests -> all coherent (under graph).
- Throughput: drive N concurrent requests (e.g. 16) and measure aggregate tok/s; compare to a
  single-replica (dp_size=1) graph baseline on one card. Expect ~2x.
Report passed=true only if graphs captured AND coherent AND ~2x aggregate throughput, all with graph on.
metrics = tok/s (dp=2 vs dp=1, graph). logTail = serve log (incl. the capture lines) + client output.
If it HANGS, kill on timeout, errorClass=hang with where it stalled.`,
    { label: `dp-validate-try${attempt}`, phase: 'DP validate', schema: VERDICT_SCHEMA, agentType: 'claude' })
  log(`DP validate ${attempt}: ${dpV?.passed ? 'PASS' : 'FAIL'} — ${dpV?.summary || ''}`)
  if (dpV?.passed) break
  if (attempt < 3 && dpV) {
    await agent(`${COMMON}

DP-only validation attempt ${attempt} FAILED. Fix the DP launcher code. Diagnose from below; verify vs
${SPEC} and the existing TP launch path. Keep dp_size=1 a no-op. Return what you changed.
errorClass: ${dpV.errorClass}\nsuggestedFix: ${dpV.suggestedFix}\nlogTail:\n${dpV.logTail}`,
      { label: `dp-fix-${attempt}`, phase: 'DP validate', agentType: 'claude' })
  }
}

// ===========================================================================
phase('EP impl')
log('Phase 3: implementing expert parallelism (all_gather + all_reduce, lockstep).')
await agent(`${COMMON}

IMPLEMENT EP (toggle --enable-ep), per ${SPEC} "EP" section. The DP launcher from phase 1 exists.
- args.py: add --enable-ep (default off). When on, the dp ranks form the EP group (the DP group from
  engine.py) and run in LOCKSTEP.
- distributed/impl.py: EP uses ONLY existing all_gather + all_reduce on the DP/EP group — do NOT add
  all_to_all. (If a group arg is needed, thread it through DistributedCommunicator.)
- models/weight.py (_store_expert) + layers/moe.py (_GroupedFP8Experts, MoELayer): when EP on, each rank
  loads + sizes ONLY its expert shard [dp_rank*E/dp : (dp_rank+1)*E/dp] (E=16, dp=2 -> 8/rank).
- layers/moe.py MoELayer.forward fp8 branch: EP dispatch/combine — all_gather token rows + top-1
  ids/weights so every rank sees all tokens; remap global expert id -> local (gid - dp_rank*E/dp) and
  mask non-local tokens; run kernels.w8a8_moe over LOCAL experts with remapped ids; all_reduce(SUM) the
  partial outputs (each token's top-1 expert is on exactly one rank, so the sum is exact); each rank
  slices its own tokens. This REPLACES the tp all_reduce when EP is on. MOD skip stays home-side.
- LOCKSTEP must be GRAPH-CAPTURABLE (see ${SPEC} "Graph capture" — this is a hard requirement; a
  non-capturable EP is NOT done). Do NOT use an in-hot-path host "does anyone have work" handshake or a
  conditional dummy forward (that is host control flow and breaks capture). Instead: the EP all_gather +
  masked-local w8a8_moe + all_reduce run INSIDE the captured decode graph at FIXED shapes; before replay
  all EP replicas agree a COMMON bs (smallest captured graph bs >= max real batch across replicas, via
  ONE all_reduce(MAX) of per-replica batch size on the gloo CPU group — OUTSIDE the graph, it only
  selects which graph to replay); every replica replays that same-bs graph, an under-full replica pads
  with all-padding rows (the existing is_pad/slot-0 CCA capture mechanism) so it participates in the
  in-graph collectives, padding outputs discarded. Lockstep is then implicit (same-bs graph everywhere)
  → no deadlock, no in-graph host sync, fully capturable. Keep gather_reduce MoE epilogue (capture-safe),
  MINISGL_MOE_SCATTER=0. Gate the EP path on --enable-ep (EP off = phase-1 independent replicas).
Make it import-clean; EP off must behave exactly like phase 1. Return the dispatch/combine + the
GRAPH-CAPTURABLE common-bs lockstep as implemented, confirm the EP collectives are inside the captured
graph, and any correctness caveats (esp. top-1 skew handling).`,
  { label: 'ep-impl', phase: 'EP impl', agentType: 'claude' })

// ===========================================================================
phase('EP validate')
log('Phase 4: GPU 2-card validation of DP+EP (timeout-bounded for deadlocks).')
let epV = null
for (let attempt = 1; attempt <= 3; attempt++) {
  epV = await agent(`${COMMON}

VALIDATE (attempt ${attempt}/3) DP+EP on 2 cards **UNDER CUDA GRAPH CAPTURE** (graph is the gate — EP is
NOT done until it serves under capture; see ${SPEC}). Lease 2 cards (${LEASE} -n 2); bring up ZAYA serve
--data-parallel-size 2 --enable-ep, graph capture ENABLED (--graph 16), MINISGL_MOE_SCATTER=0. HARD
TIMEOUT everything (timeout 600 ...). A deadlock MUST surface as timeout->FAIL, not a hang.
- GRAPH: confirm graphs CAPTURED (incl. the in-graph EP all_gather/all_reduce) + REPLAYED — grep the
  serve log for the capture lines; if EP silently fell back to eager, that is a FAIL.
- Coherence: prompt -> coherent text; several concurrent requests -> all coherent (under graph).
- Parity: greedy (temp 0) output with EP (graph) should MATCH the DP-only greedy output (EP is
  mathematically equivalent — same routing, sum-combine). Confirm they match.
- Sharding: confirm each rank loaded only 8 experts (smaller weight footprint / larger KV pool than
  DP-only) — grep the serve log.
Report passed=true ONLY if graphs captured (with EP collectives in-graph) AND coherent AND greedy-parity
AND experts sharded — all under graph capture. metrics = per-rank KV pool (EP graph) vs DP-only,
expert count/rank, decode tok/s under graph. errorClass=hang if it times out (say which collective/step
stalled); errorClass=eager-fallback if graph capture didn't engage for the EP path.`,
    { label: `ep-validate-try${attempt}`, phase: 'EP validate', schema: VERDICT_SCHEMA, agentType: 'claude' })
  log(`EP validate ${attempt}: ${epV?.passed ? 'PASS' : 'FAIL'} — ${epV?.summary || ''}`)
  if (epV?.passed) break
  if (attempt < 3 && epV) {
    await agent(`${COMMON}

DP+EP validation attempt ${attempt} FAILED (${epV.errorClass}). Fix the EP code. The usual culprits:
lockstep/dummy-forward (deadlock), global->local expert id remap, the all_reduce combine (wrong sum /
double-count), expert shard loading, or the EP group. If it HANGS it's almost always a rank skipping the
collective — ensure idle ranks run the dummy forward. Verify vs ${SPEC}. Return what you changed.
suggestedFix: ${epV.suggestedFix}\nlogTail:\n${epV.logTail}`,
      { label: `ep-fix-${attempt}`, phase: 'EP validate', agentType: 'claude' })
  }
}

// ===========================================================================
phase('AB report')
log('Phase 5: with vs without EP A/B.')
const ab = await agent(`${COMMON}

Produce the with-vs-without-EP A/B for ZAYA on 2 cards and WRITE it to ${REPO}/docs/zaya-port/DP_EP_RESULTS.md.
Lease 2 cards; measure (hard timeouts):
- KV pool tokens per card: DP-only vs DP+EP (EP frees ~4GB of expert weights/card -> bigger KV). This is
  the headline (more KV -> more RSA concurrency).
- Decode throughput (aggregate, UNDER GRAPH CAPTURE) DP-only vs DP+EP at a representative load (reuse
  tools/zaya_serving_matrix.py per replica with graph on, or drive the endpoint with --graph).
- top-1 expert-skew note: 16 experts / 2 EP ranks, top-1 routing -> load-imbalance observations.
Include the DP-only throughput vs 1-replica baseline (the ~2x). Be honest about anything that didn't
work (deadlocks, skew, partial). Return a concise summary + the headline numbers.`,
  { label: 'ab-report', phase: 'AB report', schema: VERDICT_SCHEMA, agentType: 'claude' })

// ===========================================================================
return {
  dpLauncher: dpV?.passed ? 'PASS' : 'FAIL',
  dpMetrics: dpV?.metrics,
  ep: epV?.passed ? 'PASS' : 'FAIL',
  epMetrics: epV?.metrics,
  abReport: ab?.passed ? 'done' : 'partial',
  abSummary: ab?.summary,
  notes: 'Spec docs/zaya-port/DP_EP_SPEC.md; results docs/zaya-port/DP_EP_RESULTS.md. EP off must be a no-op for dp=1.',
}
