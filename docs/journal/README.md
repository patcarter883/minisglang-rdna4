# Engineering journal

Dated working records from the RDNA4 port, kept as evidence of how decisions were made — what was
measured, what was falsified, and why a thing is built the way it is. They span 2026-06-17 to
2026-09-16. **They are not maintained.** Each was written against the tree as it stood on its date,
so a file here may name a flag, a kernel, an env var or a file path that has since been renamed or
removed, and a "next step" in one of them may have been done, dropped or reversed long ago. Nothing
here is a guide: read it for the reasoning and the numbers, not for instructions — commands quoted
in them come from the author's own box, absolute paths and local-only tooling included, and will
not run as written anywhere else. The maintained documentation is [`../SERVING.md`](../SERVING.md)
(how to run a model) and [`../IMAGE.md`](../IMAGE.md) (what is inside the container and how to
build it).

Three filename conventions recur. `CONTINUANCE_*` / `CONTINUE_*` are handover notes — the state of
an in-flight task, written so the next session could pick it up, which is why they read as
instructions to someone who no longer exists. `*_PLAN` / `*_SPEC` are up-front designs, written
before the thing was built; the result usually lives in a separate measurement file rather than
being folded back in, though a few carry their own verdict inline (`GDN_RADIX_SPEC.md` is headed
"DOES NOT WORK YET"). `*_SCORECARD` is the opposite — a measured sweep, headed with the date,
the engine and kernel commits, and the card it ran on.

## Index by theme

125 markdown files plus the raw artefacts they cite — benchmark JSON, kernel traces, serve logs,
and the probe scripts that produced them. Grouped, not enumerated:

| Theme | Where | What is in it |
|---|---|---|
| Speculative decoding | [`SPEC_DECODE.md`](SPEC_DECODE.md), [`SPEC_MODES_REVIEW_2026-08-25.md`](SPEC_MODES_REVIEW_2026-08-25.md), `SPEC_*`, `SAMPLED_SPEC_VERIFY.md`, `PHASE_C_FUSED_FORWARD.md`, `V2_CCA_VERIFY_CAPTURE.md`, `CONTINUE_dflash_*`, `CONTINUE_FUSED_TIDAR.md` | The design and bring-up of n-gram, MTP, EAGLE3, DFlash and TiDAR: the bit-exact verify contract, adaptive verify width, propose-graph capture, acceptance measurement methodology, and the acceptance-vs-cost arguments that decided which modes ship. |
| Quantization and kernels | [`W4A8_MOE_HANDOFF.md`](W4A8_MOE_HANDOFF.md), [`KERNEL_PERF_BACKLOG.md`](KERNEL_PERF_BACKLOG.md), `*_SCORECARD.md`, `FUSION_JOURNAL.md`, `MOE_G2_SPLIT.md`, `CONTINUANCE_gemm_efficiency.md`, `CONTINUE_moe_scale_layout.md`, [`measurements/NVFP4_E4M3_SCALE_POLICY.md`](measurements/NVFP4_E4M3_SCALE_POLICY.md) | W4A8 / W8A8 / NVFP4 / MXFP4 weight-format work on the shared GEMM and MoE cores: scale layouts, split-K crossovers, roofline scoring, hardware counter sweeps, and the fusion attempts that were tried and rejected. |
| Weight offload and the CPU MoE tier | [`WEIGHT_OFFLOAD_PLAN.md`](WEIGHT_OFFLOAD_PLAN.md), [`CPU_MOE_OFFLOAD.md`](CPU_MOE_OFFLOAD.md), [`measurements/WEIGHT_OFFLOAD_2026-09-02/`](measurements/WEIGHT_OFFLOAD_2026-09-02/) | The probe series (P0–P6) behind running a model larger than VRAM: the llama.cpp baseline it is scored against, host read bandwidth, mixed-media MoE GEMM, the expert cache, and the stage/stream tiers. Probe files named `*.selftest.md` are harness self-checks, not measurements. |
| Model bring-ups | [`PORT.md`](PORT.md), `QWEN35B_BRINGUP_SCOPE.md`, `QWEN4EXP_*`, `GLM_CONTINUANCE.md`, `MUSE_GLIMMER_PORT.md`, `NEMOTRON35_LIGHTNING_PLAN.md`, `DIFFUSIONGEMMA_BLOCK_DIFFUSION.md`, [`laguna-port/`](laguna-port/) | Per-architecture ports: what the reference implementation does, which convention the checkpoint uses, what the layer-parity oracle had to prove, and the coherence bugs found on the way. `PORT.md` is the original GDN/Qwen3.5 port and the oldest file here. |
| ZAYA, CAM and RSA | [`zaya-port/`](zaya-port/), `ZAYA_SERVING_NORTH_STAR.md`, `CAM_ANN_SCOPING.md`, `RSA_KNOBS.md`, `CONTINUE_ZAYA_SERVING.md` | The ZAYA port and the CAM cartridge-memory work built on it: the serve contract, memory-layer and composer specs, DP/EP scaling, the RSA shim, and the Titans/MIRAS research triage. |
| Measurements | [`measurements/`](measurements/) | Run reports with their raw artefacts alongside: boot-time regressions, the QSA indexer, Qwen4Exp graph capture, DFlash ring-gate retakes. Each states its checkpoint, card and commit, because results here are not comparable across cards. |
| Box, toolchain and process | `ROCR_IDLE_SPIN.md`, `ROCM10_UPGRADE.md`, `HOST_PUBLISH_LOOP_ANTIPATTERN.md`, `WORKTREE_TRIAGE.md`, `CONTINUE_hsa_fault_conc1_ctx.md`, `RDNA4_VLLM_WIRING_SPEC.md` | Problems that were not in the engine: an HSA runtime spinning a core per rank, a ROCm upgrade, host-side publish loops that starved the scheduler, and the vLLM-side wiring the kernels were also consumed through. |
