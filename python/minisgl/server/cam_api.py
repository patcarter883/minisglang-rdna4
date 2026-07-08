"""CAM edit-plane API — the ``/cam/*`` endpoints (WS-C).

This module folds the memory-organ **CAM** editable-memory serving path into the minisgl-rdna4
FastAPI server (Qwen3.5-4B, port 1919). It exposes a small, honest edit plane:

    POST   /cam/remember   {subject, prompt, object}  -> {stored, base_p}
    POST   /cam/ask        {prompt, subject}          -> {text}
    GET    /cam/facts                                 -> [{subject, object}, ...]
    DELETE /cam/facts/{subject}                       -> {deleted}

It is **additive and gated off by default**: it is only mounted when ``MINISGL_CAM=1`` (see the
guarded wire in ``api_server.py``), and every endpoint 503s cleanly if the CAM runtime is not
loaded, so a server without CAM still starts and ``/generate`` + ``/v1/*`` are never disturbed.

------------------------------------------------------------------------------------------------
Runtime seam (what WS-A must provide)
------------------------------------------------------------------------------------------------
The write gate and the router-gated decode loop need to run **base-model forwards** and call the
``CAMMemory`` store. In the standard multi-process minisgl deployment the base model lives in the
**backend scheduler process**, not in this FastAPI frontend, and the minisgl model is *not*
HF-callable (no ``inputs_embeds=`` / ``.logits`` — it takes a ``ForwardBatch``). CAM serving is
therefore **eager / single-process** (consistent with the contract's "eager-first, graph capture
is a TODO"), and WS-A co-locates a **CAM runtime handle** in the API-server process.

This module resolves that handle lazily via ``minisgl.cam`` and assumes the following seam. WS-A:
please match these names/signatures (they are the exact calls this module makes):

    minisgl.cam.get_cam_runtime() -> CAMRuntime | None
        Returns the loaded runtime, or None/raises if CAM is not loaded.

    CAMRuntime:
        .tokenizer                                   # HF tokenizer (same as the base)
        .memory                                      # CAMMemory (minisgl.cam.memory)
        .base_logits(token_ids: list[int]) -> Tensor # run ONE base forward (no CAM staged),
                                                     #   return last-position logits [vocab], float.
                                                     #   (Wraps the minisgl model forward; hides the
                                                     #    ForwardBatch / multi-process detail.)

    CAMMemory (the store; contract: python/minisgl/cam/memory.py):
        .read(subject_ids: list[int]) -> (bank, conf)
        .router_delta(base_last_logits: Tensor, bank, conf) -> Tensor   # per-token logit injection
        .remember(subject_ids, object_ids, prompt_last_logits) -> bool  # base-uncertainty WRITE GATE
            # (or, if WS-A prefers the stateful form, set_pending_object(object_ids) then
            #  remember(subject_ids, prompt_last_logits) — this module supports BOTH, see _remember)
        .list_facts() -> list[dict]                  # side-index enumeration: [{subject, object}, ...]
        .delete(subject_ids: list[int]) -> bool      # tombstone forget (alias: .forget)
        .seed_token(bank, conf) -> int               # OPTIONAL: the store's preferred first token
                                                     #   (serve_gen's ``store_tok``). If absent this
                                                     #   module derives it from router_delta.

If ``minisgl.cam`` is absent (WS-A not landed) or ``get_cam_runtime()`` yields nothing, every
endpoint returns HTTP 503 ``"CAM not loaded"``.

------------------------------------------------------------------------------------------------
Example curl calls
------------------------------------------------------------------------------------------------
Start the server with CAM enabled (``MINISGL_CAM=1 python -m minisgl.server ...``), then:

    # 1. Teach a novel fact. Stored only if the base can't already recall the object
    #    (base_p = p_base(object first token) < remember_tau).
    curl -s localhost:1919/cam/remember -H 'content-type: application/json' \
      -d '{"subject":"Elowen Marsh","prompt":"Elowen Marsh was born in the city of","object":"Reykjavik"}'
    # -> {"stored": true, "base_p": 0.0007}

    # 2. Ask — router-gated seed-once decode delivers the remembered object.
    curl -s localhost:1919/cam/ask -H 'content-type: application/json' \
      -d '{"prompt":"Elowen Marsh was born in the city of","subject":"Elowen Marsh"}'
    # -> {"text": " Reykjavik."}

    # 3. List what's stored (from the side index).
    curl -s localhost:1919/cam/facts
    # -> [{"subject":"Elowen Marsh","object":"Reykjavik"}]

    # 4. Forget (tombstone — subject stops being delivered).
    curl -s -X DELETE localhost:1919/cam/facts/Elowen%20Marsh
    # -> {"deleted": true}
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("minisgl.cam_api")

# Max tokens the /cam/ask decode loop will emit if it never hits EOS.
_DEFAULT_ASK_MAX_TOKENS = 32


# --------------------------------------------------------------------------------------------- #
# Request / response models (mirror api_server.py's pydantic BaseModel style)
# --------------------------------------------------------------------------------------------- #
class RememberRequest(BaseModel):
    subject: str
    prompt: str
    object: str


class RememberResponse(BaseModel):
    stored: bool
    base_p: float


class AskRequest(BaseModel):
    prompt: str
    subject: str
    # Optional decode cap; router injection stops once the object's first token lands (seed-once),
    # the base then continues fluently up to this many tokens (or EOS).
    max_tokens: int = _DEFAULT_ASK_MAX_TOKENS


class AskResponse(BaseModel):
    text: str


class FactItem(BaseModel):
    subject: str
    object: str


class DeleteResponse(BaseModel):
    deleted: bool


# --------------------------------------------------------------------------------------------- #
# CAM runtime resolution (guarded — this module imports even before WS-A lands)
# --------------------------------------------------------------------------------------------- #
def _get_runtime():
    """Resolve the co-located CAM runtime handle, or raise 503 if CAM is not loaded.

    Lazily imports ``minisgl.cam`` so this module is importable (CPU smoke test) even before
    WS-A ships ``minisgl/cam/``.
    """
    try:
        from minisgl import cam as cam_pkg  # WS-A package
    except Exception as e:  # noqa: BLE001 - absent package or import-time failure -> not loaded
        logger.debug("CAM package unavailable: %s", e)
        raise HTTPException(status_code=503, detail="CAM not loaded") from e

    getter = getattr(cam_pkg, "get_cam_runtime", None)
    if getter is None:
        raise HTTPException(status_code=503, detail="CAM not loaded")
    try:
        runtime = getter()
    except Exception as e:  # noqa: BLE001 - runtime not initialized
        logger.debug("CAM runtime not initialized: %s", e)
        raise HTTPException(status_code=503, detail="CAM not loaded") from e
    if runtime is None:
        raise HTTPException(status_code=503, detail="CAM not loaded")
    return runtime


def _bos_ids(tok) -> List[int]:
    bid = getattr(tok, "bos_token_id", None)
    return [bid] if bid is not None else []


def _encode(tok, text: str) -> List[int]:
    return list(tok(text, add_special_tokens=False).input_ids)


def _encode_sp(tok, text: str) -> List[int]:
    """Space-prefixed encode for SUBJECTS and OBJECTS — the CAM store/tap/router were trained on the
    mid-sentence ' <s>' encoding (memory-organ `_sp_tokens`). Prompts stay plain (`_encode`)."""
    return list(tok(" " + text, add_special_tokens=False).input_ids)


def _remember(memory, subject_ids, object_ids, prompt_last_logits, ns=None) -> bool:
    """Call the WRITE GATE, adapting to whichever object-passing form WS-A picked. `ns` scopes the write
    to a namespace (#6)."""
    if hasattr(memory, "set_pending_object"):
        memory.set_pending_object(object_ids)
        return bool(memory.remember(subject_ids, prompt_last_logits, ns=ns))
    return bool(memory.remember(subject_ids, object_ids, prompt_last_logits, ns=ns))


# --------------------------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------------------------- #
def _require_cam_auth(authorization: str = Header(None)) -> None:
    """#8 auth: when MINISGL_CAM_API_TOKEN is set, require `Authorization: Bearer <token>` on EVERY /cam/*
    route (the edit-plane mutates shared memory). Unset -> open (localhost/dev default). Applied as a
    router-level dependency so read and write routes are covered uniformly."""
    import os
    token = os.environ.get("MINISGL_CAM_API_TOKEN")
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid or missing CAM API token")


cam_router = APIRouter(prefix="/cam", tags=["cam"], dependencies=[Depends(_require_cam_auth)])


@cam_router.post("/remember", response_model=RememberResponse)
async def remember(req: RememberRequest, x_cam_namespace: str = Header(None)) -> RememberResponse:
    """Base-uncertainty WRITE GATE: store subject->object iff the base can't already recall it.

    base_p = softmax(base last logits over the prompt)[object first token]. The store writes only
    when base_p < remember_tau (decided inside CAMMemory.remember); we report base_p either way so
    the caller sees why a fact was or wasn't kept.
    """
    import torch

    runtime = _get_runtime()
    # MULTI-PROCESS model-share: the store lives in the backend engine.cam; route the write through it
    # (rides a generate with mem_remember). No local model/store on the frontend.
    if getattr(runtime, "is_frontend_share", False):
        if not _encode_sp(runtime.tokenizer, req.subject):
            raise HTTPException(status_code=422, detail="subject tokenized to empty")
        stored = await runtime.remember(req.subject, req.object, req.prompt, namespace=x_cam_namespace)
        return RememberResponse(stored=stored, base_p=0.0)
    tok = runtime.tokenizer
    memory = runtime.memory

    subject_ids = _encode_sp(tok, req.subject)
    object_ids = _encode_sp(tok, req.object)
    if not subject_ids:
        raise HTTPException(status_code=422, detail="subject tokenized to empty")
    if not object_ids:
        raise HTTPException(status_code=422, detail="object tokenized to empty")

    prompt_ids = _bos_ids(tok) + _encode(tok, req.prompt)

    with torch.no_grad():
        prompt_last_logits = runtime.base_logits(prompt_ids).float()
        val_tid = object_ids[0]
        base_p = float(torch.softmax(prompt_last_logits, dim=-1)[val_tid])
        stored = _remember(memory, subject_ids, object_ids, prompt_last_logits, ns=x_cam_namespace)

    return RememberResponse(stored=stored, base_p=base_p)


@cam_router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, x_cam_namespace: str = Header(None)) -> AskResponse:
    """Router-gated seed-once generation (the eval_serve ``serve_gen`` loop).

    read(subject) -> bank; each step take the base last logits and add CAMMemory.router_delta;
    stop injecting once the object's first token ("seed" token) has been placed, then let the base
    continue fluently. No KV cache (base_logits recomputes the prefix each step, exactly as the
    offline reference does) — fine for the short answers this delivers.
    """
    import torch

    runtime = _get_runtime()
    # MULTI-PROCESS model-share: deliver via the backend engine.cam — a normal generate carrying
    # mem_subject; the scheduler forces the exact stored object tokens (pointer), then the base continues.
    if getattr(runtime, "is_frontend_share", False):
        text = await runtime.ask(req.prompt, req.subject, max(1, int(req.max_tokens)),
                                 namespace=x_cam_namespace)
        return AskResponse(text=text.replace("\n", " ").strip())
    tok = runtime.tokenizer
    memory = runtime.memory

    # Backend-shared runtime: deliver via the residual TAP + seed-once (the validated path — a normal
    # generate carrying mem_subject through the served model). The co-located HF runtime has no served
    # decode loop, so it falls through to the logit-only router path below.
    ask_tap = getattr(runtime, "ask_tap", None)
    if ask_tap is not None:
        text = ask_tap(req.prompt, req.subject, max(1, int(req.max_tokens)))
        return AskResponse(text=text.replace("\n", " ").strip())

    subject_ids = _encode_sp(tok, req.subject)
    if not subject_ids:
        raise HTTPException(status_code=422, detail="subject tokenized to empty")

    cur = _bos_ids(tok) + _encode(tok, req.prompt)
    eos_id = getattr(tok, "eos_token_id", None)
    max_tokens = max(1, int(req.max_tokens))
    out: List[int] = []

    # POINTER delivery (#100): the exact stored object token sequence, retrieved via the store's
    # ADDRESSING (not a lossy value reconstruction — that floored genuine multi-token delivery at
    # ~0.5/token). Emit the object tokens straight from memory, then release to the base to continue the
    # sentence. Validated 4/4 span-exact offline. Falls back to the router-gated seed-once path when the
    # subject addresses no stored object (nothing to deliver).
    deliver = getattr(memory, "deliver_object_ids", None)
    obj_ids = deliver(subject_ids, x_cam_namespace) if deliver is not None else []

    with torch.no_grad():
        if obj_ids:
            for i in range(max_tokens):
                if i < len(obj_ids):
                    nxt = int(obj_ids[i])                        # exact object token from memory (pointer)
                else:
                    nxt = int(runtime.base_logits(cur).float().reshape(-1).argmax().item())  # base continues
                if eos_id is not None and nxt == eos_id:
                    break
                out.append(nxt)
                cur = cur + [nxt]
        else:
            # --- fallback: router-gated seed-once decode (unstored subject / no pointer) ---
            bank, conf = memory.read(subject_ids, ns=x_cam_namespace)
            seed_fn = getattr(memory, "seed_token", None)
            seed_tok = None
            placed = False
            for _ in range(max_tokens):
                logits = runtime.base_logits(cur).float()
                if not placed:
                    delta = memory.router_delta(logits, bank, conf)
                    if seed_tok is None:
                        seed_tok = (int(seed_fn(bank, conf)) if seed_fn is not None
                                    else int(torch.as_tensor(delta).reshape(-1).argmax().item()))
                    logits = logits + torch.as_tensor(delta).to(logits.device)
                nxt = int(logits.reshape(-1).argmax().item())
                if seed_tok is not None and nxt == seed_tok:
                    placed = True                                # seed-once: first token landed, stop injecting
                if eos_id is not None and nxt == eos_id:
                    break
                out.append(nxt)
                cur = cur + [nxt]

    text = tok.decode(out).replace("\n", " ").strip()
    return AskResponse(text=text)


@cam_router.get("/facts", response_model=List[FactItem])
async def list_facts(x_cam_namespace: str = Header(None)) -> List[FactItem]:
    """List stored edits from the CAMMemory side index (the bank tensor cannot be enumerated).

    CAMMemory keeps the side index as raw token-ids (it is deliberately tokenizer-free), so we
    decode ``subject_ids``/``object_ids`` back to text here with the runtime tokenizer. Subjects and
    objects were stored space-prefixed (``_encode_sp``); ``.decode`` yields a leading space we strip.
    Shapes with ready-made ``subject``/``object`` strings are passed through unchanged.
    """
    runtime = _get_runtime()
    if getattr(runtime, "is_frontend_share", False):     # multi-process: facts come from the backend engine.cam
        return [FactItem(subject=f.get("subject", ""), object=f.get("object", ""))
                for f in await runtime.facts(x_cam_namespace)]
    memory = runtime.memory
    tok = runtime.tokenizer
    lister = getattr(memory, "list_facts", None)
    if lister is None:
        raise HTTPException(status_code=503, detail="CAM side index unavailable")

    def _text(f, str_key, ids_key):
        if str_key in f and f[str_key] is not None:
            return str(f[str_key])
        ids = f.get(ids_key)
        return tok.decode(list(ids)).strip() if ids else ""

    return [FactItem(subject=_text(f, "subject", "subject_ids"),
                     object=_text(f, "object", "object_ids")) for f in lister(x_cam_namespace)]


@cam_router.get("/stats")
async def stats(x_cam_namespace: str = Header(None)) -> dict:
    """Per-bank occupancy + crowding health for the product-key VALUE banks (the router/tap fallback
    path). NOTE: the PRIMARY /cam/ask delivery is the cosine-NN subject index (exact retrieval, no bank
    collision), so crowding no longer degrades pointer delivery — this monitors only the value-bank
    fallback. Served from the side index."""
    runtime = _get_runtime()
    if getattr(runtime, "is_frontend_share", False):     # multi-process: stats from the backend engine.cam
        return await runtime.stats(x_cam_namespace)
    statter = getattr(runtime.memory, "stats", None)
    if statter is None:
        raise HTTPException(status_code=503, detail="CAM stats unavailable")
    return statter(x_cam_namespace)


@cam_router.post("/freeze")
async def freeze(frozen: bool = True, x_cam_namespace: str = Header(None)) -> dict:
    """Freeze (or with ?frozen=false, unfreeze) the namespace's store: while frozen, ambient transparent
    auto-write is refused so a curated/ingested store is not overwritten by conversation — explicit
    /cam/remember still curates. The natural switch after ingesting a knowledge base: POST /cam/freeze."""
    runtime = _get_runtime()
    if getattr(runtime, "is_frontend_share", False):        # multi-process: toggle the backend engine.cam
        return {"frozen": (await runtime.freeze(x_cam_namespace)) if frozen
                else (await runtime.unfreeze(x_cam_namespace))}
    mem = runtime.memory
    fn = getattr(mem, "freeze" if frozen else "unfreeze", None)
    if fn is None:
        raise HTTPException(status_code=503, detail="CAM freeze unavailable")
    return {"frozen": bool(fn(x_cam_namespace))}


@cam_router.post("/save")
async def save() -> dict:
    """#7 persistence: force a snapshot to MINISGL_CAM_STORE_PATH now (all namespaces). Returns #edits
    saved, or -1 when no store path is configured. (Autosave also runs debounced after writes.)"""
    runtime = _get_runtime()
    if getattr(runtime, "is_frontend_share", False):
        return await runtime.save()
    saver = getattr(runtime.memory, "save", None)
    if saver is None:
        raise HTTPException(status_code=503, detail="CAM persistence unavailable")
    return {"saved": saver()}


@cam_router.post("/undo")
async def undo(x_cam_namespace: str = Header(None)) -> dict:
    """#12: undo the most-recent write in a namespace (forget that subject). Returns the undone fact or {}."""
    runtime = _get_runtime()
    if getattr(runtime, "is_frontend_share", False):
        return await runtime.undo(x_cam_namespace)
    u = runtime.memory.undo(x_cam_namespace)
    tok = runtime.tokenizer
    return {"subject": tok.decode(list(u["subject_ids"])).strip(),
            "object": tok.decode(list(u["object_ids"])).strip()} if u else {}


@cam_router.post("/rebuild")
async def rebuild(x_cam_namespace: str = Header(None)) -> dict:
    """#12 TRUE ERASE / compaction: re-init a namespace's banks and replay only the surviving facts,
    discarding delta residue from forgotten/overwritten edits."""
    runtime = _get_runtime()
    if getattr(runtime, "is_frontend_share", False):
        return await runtime.rebuild(x_cam_namespace)
    return {"rebuilt": runtime.memory.rebuild(x_cam_namespace)}


@cam_router.get("/audit")
async def audit(x_cam_namespace: str = Header(None)) -> List[dict]:
    """#12: recent write/forget/evict events for a namespace (most-recent last)."""
    runtime = _get_runtime()
    if getattr(runtime, "is_frontend_share", False):
        return await runtime.audit(x_cam_namespace)
    tok = runtime.tokenizer
    out = []
    for r in runtime.memory.audit_log(x_cam_namespace):
        out.append({"op": r["op"], "ts": r["ts"],
                    "subject": tok.decode(list(r["subject_ids"])).strip(),
                    "object": tok.decode(list(r["object_ids"])).strip() if r["object_ids"] else ""})
    return out


@cam_router.delete("/facts/{subject}", response_model=DeleteResponse)
async def delete_fact(subject: str, x_cam_namespace: str = Header(None)) -> DeleteResponse:
    """Tombstone-forget a subject in a namespace: it stops being delivered from the serve path. Exact
    erase (rebuild) is a full-surface item, not MVP."""
    runtime = _get_runtime()
    if getattr(runtime, "is_frontend_share", False):     # multi-process: forget in the backend engine.cam
        if not _encode_sp(runtime.tokenizer, subject):
            raise HTTPException(status_code=422, detail="subject tokenized to empty")
        return DeleteResponse(deleted=await runtime.forget(subject, namespace=x_cam_namespace))
    tok = runtime.tokenizer
    memory = runtime.memory

    subject_ids = _encode_sp(tok, subject)
    if not subject_ids:
        raise HTTPException(status_code=422, detail="subject tokenized to empty")

    deleter = getattr(memory, "delete", None) or getattr(memory, "forget", None)
    if deleter is None:
        raise HTTPException(status_code=503, detail="CAM delete unavailable")
    deleted = bool(deleter(subject_ids, ns=x_cam_namespace))
    return DeleteResponse(deleted=deleted)
