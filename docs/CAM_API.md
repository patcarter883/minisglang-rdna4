# CAM HTTP API

The CAM (Canonical Associative Memory) API lets a client store `subject → object` facts and recall
them however they're phrased — the serve addresses them with a paraphrase-robust semantic key and
delivers the exact stored object token-for-token, then the base model continues. Served under `/cam/*`
by a running minisgl CAM serve (default `http://<host>:1919`).

A zero-dependency Python client is at [`minisgl.cam.client.CAMClient`](../python/minisgl/cam/client.py).

## Auth & namespaces (headers on every call)

| Header | When | Meaning |
|---|---|---|
| `Authorization: Bearer <token>` | if any token is configured | else all `/cam/*` return 401 |
| `X-CAM-Namespace: <name>` | optional | per-tenant/session store; omitted → `default` |

**Token config (choose one or both):**
- `MINISGL_CAM_API_TOKEN=<token>` — a single **admin** token, authorized for every namespace.
- `MINISGL_CAM_TOKENS='{"<tokA>":"acme","<tokB>":["b1","b2"],"<admin>":"*"}'` — **per-tenant** tokens.
  A token scoped to a namespace (or list) may ONLY touch those; `"*"` = admin. Neither set → open (dev).

**Enforcement:** a request's token must be authorized for the requested namespace, else **403**. Cross-
tenant access via `X-CAM-Namespace` is blocked, `GET /cam/namespaces` is filtered to the token's
namespaces, and `DELETE /cam/namespaces/{ns}` re-checks the path namespace. `MINISGL_CAM_MAX_NAMESPACES`
caps the number of tenant namespaces (`default` excluded) — creating one past the cap returns **429**.

## Endpoints

### Write
- **`POST /cam/remember`** — store a fact.
  Body: `{subject, object, prompt, relation?}`. `relation` (optional) stores this fact of `subject`
  under a relation so one entity can hold many facts (`"Mozart"` + `"birthplace"`). `prompt` is the
  relation prompt for the base-uncertainty write gate (not needed for delivery).
  → `{stored: bool, base_p: float, mode_served: "pointer", gate_reason: "novel"|"base-known"|"forced"|...}`
- **`DELETE /cam/facts/{subject}`** — tombstone-delete a subject. → `{deleted: bool}`

### Read
- **`POST /cam/ask`** — answer a prompt with a stored fact.
  Body: `{prompt, subject, relation?, max_tokens?}`. The serve forces the exact stored object tokens,
  then the base continues. → `{text, delivered: bool, object: str, mode_served: "pointer"}`
  (`object` = the exact delivered object, empty if nothing matched ≥ deliver_tau).
- **`GET /cam/lookup?subject=&relation=`** — dry-run of `ask` (no generation, no mutation).
  → `{delivered, subject, object}`
- **`GET /cam/lookup?text=`** — transparent-read: which stored facts does `text` mention (auto-RAG)?
  → `{matches: [{subject, object}, ...]}`
- **`GET /cam/facts`** — list this namespace's facts. → `[{subject, object}, ...]`
- **`GET /cam/stats`** — store health (fact count, `index_nn_cos_*` crowding, `recovered_from_backup`,
  `last_save_age_s`, `dirty`, `evicted`, ...).
- **`GET /cam/audit`** — recent write/forget/merge/evict events. → `[{ts, op, subject, object, ns}, ...]`

### Store ops
- **`POST /cam/save`** — force a durable atomic snapshot now. → `{saved: <#edits>}`
- **`POST /cam/reload`** — re-read the store from disk (pick up another replica's writes). → `{edits}`
- **`POST /cam/undo`** — undo the last write in this namespace.
- **`POST /cam/freeze?frozen=true|false`** — freeze/unfreeze: ambient auto-write refused; explicit
  remember still writes. → `{frozen: bool}`

### Namespaces
- **`GET /cam/namespaces`** — every namespace + fact count + freeze state.
- **`DELETE /cam/namespaces/{ns}`** — delete a namespace's entire store.

## Behavior notes

- **Delivery is exact + paraphrase-robust.** The subject is addressed by a whitened-GTE semantic key,
  so `"the composer Mozart"` recalls what you stored under `"Wolfgang Amadeus Mozart"`. A match below
  `deliver_tau` (default 0.70) delivers nothing and the base answers normally — a miss, never a wrong
  fact.
- **Corrections merge.** Re-`remember`ing the same (or a paraphrased) subject updates the object; near-
  duplicate subjects merge rather than accumulating duplicates.
- **Multi-fact.** Pass `relation` on both `remember` and `ask`/`lookup` to keep many facts per entity
  distinct; the semantic key bridges paraphrased relations (`"birthplace"` ↔ `"where was he born"`).

## curl examples

```bash
S=http://127.0.0.1:1919; NS=acme; A='Authorization: Bearer secret'
curl -s $S/cam/remember -H "$A" -H "X-CAM-Namespace: $NS" -H 'content-type: application/json' \
  -d '{"subject":"Wolfgang Amadeus Mozart","object":"Salzburg","prompt":"birthplace","relation":"birthplace"}'
curl -s -G $S/cam/lookup -H "$A" -H "X-CAM-Namespace: $NS" \
  --data-urlencode 'subject=the composer Mozart' --data-urlencode 'relation=where born'
```
