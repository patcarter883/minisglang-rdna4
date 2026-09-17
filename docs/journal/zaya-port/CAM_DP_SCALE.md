# CAM at DP scale (#11) — design + current state

## The problem
The CAM store (`engine.cam`) is per-scheduler-replica. Under data-parallel serving (`--data-parallel-size
> 1`) a naive round-robin splits one user's writes and reads across independent stores.

## What is in place (foundation)
1. **Namespacing (#6):** every store op is scoped to a namespace (`X-CAM-Namespace`) — isolation is
   orthogonal to replication.
2. **Replica pinning:** the tokenizer router pins CAM ops (`mem_subject`/`mem_remember`/`mem_op`) to one
   replica (`MINISGL_CAM_DP_RANK`, default 0). Correct + consistent, but that replica is a **throughput
   bottleneck and single point of failure** for CAM traffic.
3. **Persistence (#7):** the store loads on boot and autosaves to `MINISGL_CAM_STORE_PATH`.
4. **`reload()` / `POST /cam/reload`:** re-reads the store from the backing file.

## Pragmatic increment (eventual consistency, available now)
Point every replica's `MINISGL_CAM_STORE_PATH` at **shared storage** (same file). Writes autosave; other
replicas `POST /cam/reload` (a periodic reload is not yet wired) to pull them → **active-active reads**
with eventual consistency:
- Reads served by ANY replica once each has reloaded → lifts the read bottleneck.
- Writes should still be pinned/serialized to one replica: snapshotting is whole-store, so concurrent
  writers can clobber each other's file. Fine for read-heavy or frozen (ingested) knowledge bases.

## Full solution (remaining scope — keep #11 open)
Strong consistency + concurrent writes need an **external KV backend** (redis / shared object store)
holding per-namespace editable state (banks + side index + facts), with write-through on
`_write`/`forget`/`evict` and read-through/invalidation on delivery. Then the pin disappears and any
replica serves any op. A real subsystem, not a knob.

## Interactions
- **#7** is the substrate the shared-file increment rides on.
- **#9** capacity must run consistently per-backend (or move into the KV backend) so replicas agree on
  evictions.
- **#12** audit would centralize in the KV backend for a global log.
