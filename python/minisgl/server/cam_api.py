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

from fastapi import APIRouter, HTTPException
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


def _remember(memory, subject_ids, object_ids, prompt_last_logits) -> bool:
    """Call the WRITE GATE, adapting to whichever object-passing form WS-A picked.

    Contract left the object-passing convention to WS-A ("set_pending_object or a param"); this
    supports both so it interlocks regardless.
    """
    if hasattr(memory, "set_pending_object"):
        memory.set_pending_object(object_ids)
        return bool(memory.remember(subject_ids, prompt_last_logits))
    return bool(memory.remember(subject_ids, object_ids, prompt_last_logits))


# --------------------------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------------------------- #
cam_router = APIRouter(prefix="/cam", tags=["cam"])


@cam_router.post("/remember", response_model=RememberResponse)
async def remember(req: RememberRequest) -> RememberResponse:
    """Base-uncertainty WRITE GATE: store subject->object iff the base can't already recall it.

    base_p = softmax(base last logits over the prompt)[object first token]. The store writes only
    when base_p < remember_tau (decided inside CAMMemory.remember); we report base_p either way so
    the caller sees why a fact was or wasn't kept.
    """
    import torch

    runtime = _get_runtime()
    tok = runtime.tokenizer
    memory = runtime.memory

    subject_ids = _encode(tok, req.subject)
    object_ids = _encode(tok, req.object)
    if not subject_ids:
        raise HTTPException(status_code=422, detail="subject tokenized to empty")
    if not object_ids:
        raise HTTPException(status_code=422, detail="object tokenized to empty")

    prompt_ids = _bos_ids(tok) + _encode(tok, req.prompt)

    with torch.no_grad():
        prompt_last_logits = runtime.base_logits(prompt_ids).float()
        val_tid = object_ids[0]
        base_p = float(torch.softmax(prompt_last_logits, dim=-1)[val_tid])
        stored = _remember(memory, subject_ids, object_ids, prompt_last_logits)

    return RememberResponse(stored=stored, base_p=base_p)


@cam_router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """Router-gated seed-once generation (the eval_serve ``serve_gen`` loop).

    read(subject) -> bank; each step take the base last logits and add CAMMemory.router_delta;
    stop injecting once the object's first token ("seed" token) has been placed, then let the base
    continue fluently. No KV cache (base_logits recomputes the prefix each step, exactly as the
    offline reference does) — fine for the short answers this delivers.
    """
    import torch

    runtime = _get_runtime()
    tok = runtime.tokenizer
    memory = runtime.memory

    subject_ids = _encode(tok, req.subject)
    if not subject_ids:
        raise HTTPException(status_code=422, detail="subject tokenized to empty")

    bank, conf = memory.read(subject_ids)
    cur = _bos_ids(tok) + _encode(tok, req.prompt)
    eos_id = getattr(tok, "eos_token_id", None)
    max_tokens = max(1, int(req.max_tokens))

    seed_fn = getattr(memory, "seed_token", None)
    seed_tok = None
    out: List[int] = []
    placed = False

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = runtime.base_logits(cur).float()
            if not placed:
                delta = memory.router_delta(logits, bank, conf)
                if seed_tok is None:
                    # The store's preferred first token: prefer an explicit CAMMemory.seed_token;
                    # else derive it from the injection delta (its argmax is the store's top token,
                    # matching serve_gen's ``store_tok``).
                    if seed_fn is not None:
                        seed_tok = int(seed_fn(bank, conf))
                    else:
                        seed_tok = int(torch.as_tensor(delta).reshape(-1).argmax().item())
                logits = logits + torch.as_tensor(delta).to(logits.device)
            nxt = int(logits.reshape(-1).argmax().item())
            if seed_tok is not None and nxt == seed_tok:
                placed = True  # seed-once: object's first token landed, stop injecting
            if eos_id is not None and nxt == eos_id:
                break
            out.append(nxt)
            cur = cur + [nxt]

    text = tok.decode(out).replace("\n", " ").strip()
    return AskResponse(text=text)


@cam_router.get("/facts", response_model=List[FactItem])
async def list_facts() -> List[FactItem]:
    """List stored edits from the CAMMemory side index (the bank tensor cannot be enumerated)."""
    runtime = _get_runtime()
    memory = runtime.memory
    lister = getattr(memory, "list_facts", None)
    if lister is None:
        raise HTTPException(status_code=503, detail="CAM side index unavailable")
    facts = lister()
    return [FactItem(subject=str(f["subject"]), object=str(f["object"])) for f in facts]


@cam_router.delete("/facts/{subject}", response_model=DeleteResponse)
async def delete_fact(subject: str) -> DeleteResponse:
    """Tombstone-forget a subject: it stops being delivered from the serve path (bank residue
    stays; the serve path never reads a tombstoned subject). Exact erase (rebuild) is a full-surface
    item, not MVP."""
    runtime = _get_runtime()
    tok = runtime.tokenizer
    memory = runtime.memory

    subject_ids = _encode(tok, subject)
    if not subject_ids:
        raise HTTPException(status_code=422, detail="subject tokenized to empty")

    deleter = getattr(memory, "delete", None) or getattr(memory, "forget", None)
    if deleter is None:
        raise HTTPException(status_code=503, detail="CAM delete unavailable")
    deleted = bool(deleter(subject_ids))
    return DeleteResponse(deleted=deleted)
