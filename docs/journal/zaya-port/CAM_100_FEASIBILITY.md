# #100 sequential latent delivery — FEASIBILITY (2026-07-07)

memory-organ#100 = deliver genuine multi-token objects by emitting a FRESH value per decode step (the
whole phrase from memory), vs Option B's shipped seed-and-fluency (inject the first token, let the base
continue). It is the right answer for NOVEL/PRIVATE phrases the base can't continue on its own.

## What actually exists (softsteer worktree; NOT in main memory-organ)
- perpos & decoder readouts prototyped in the **Stage-1 episodic** path only (`pk_store_adapter.py`
  `_write_episode`/`memory_bank`/`_decoder_logits`). `--readout {linear,decoder,perpos}` default linear.
- **Store-side** (no base): disjoint-perpos **0.964** exact-match both tokens (`recall_boltA.py:eval_direct`).
- **Teacher-forced same-base**: 0.883 (`eval_generative_mag`, gold Kc-1 tokens injected — NOT free-run).

## The gaps (what #100 needs, none built)
1. **Free-running multi-token delivery through the base — NEVER demonstrated** (§3.25 of
   key-encoding-for-editing-memory.md calls it "the required path, not yet built end-to-end"). This is
   the GATING research risk: teacher-forced 0.883 ≠ free-running (errors compound per step).
2. **Persistent per-position write** — the persistent path (`_persistent_write_val`) writes ONE pooled
   value/subject; perpos write only exists episodically. Need a per-position persistent write.
3. **decoder `.generate()`** — `_decoder_logits` is teacher-forced only; a free-running AR loop is unbuilt.
4. **Export** strips perpos/decoder tensors (`store.stores.*`, `pos_tag/pos_gate/pos_proj`, `dec_*`);
   meta.json has no `readout/mt_value/perpos_key/mt_positions`. Loader can't tell it's a perpos store.
5. **Write-policy prereq** (#99): the base-known gate (`recall_mag.py:1946` `if not base_knows: write`)
   binds only base-unknowable facts → ~1 multi-token fact of 137. Multi-token needs facts bound OUTSIDE
   that filter (drop the guard; gate at DELIVERY on provenance/store-conf `g_pres = conf>c0`).

## Sequenced plan (de-risk before serve plumbing)
- **Phase A (memory-organ, GATING) — prove free-gen multi-token OFFLINE.** Drop the base-known filter to
  bind multi-token facts; add persistent per-position write; write a FREE-RUNNING (not teacher-forced)
  generation eval that emits value_t per step via perpos slots (or the AR decoder) and measure delivery
  on multi-token objects the base does NOT know. If this fails, #100 is blocked on readout research — stop.
- **Phase B (memory-organ) — export.** Add the per-position/decoder tensors + meta fields to
  export_serving.py so a serve loader can reconstruct a perpos store.
- **Phase C (minisgl serve) — decode-loop integration.** Extend CAMMemory to read a per-position value
  SEQUENCE; turn seed-ONCE into seed-a-SEQUENCE (deliver value_t at step t, advance until the object is
  complete, then hand off); router gates the sequence. KV-cache interaction. Reuse the Option B tap +
  per-token bank machinery (already shipped).

## The product question
Option B's seed-and-fluency ALREADY delivers multi-token objects the base can continue ("New"→"South
Wales"). #100 only adds objects the base CANNOT continue (novel/private phrases). Worth the multi-week,
research-risky build now? Depends on whether the private-facts / provenance regime (#99) is the priority.
</content>
</invoke>
