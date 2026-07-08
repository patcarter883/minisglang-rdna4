"""CAMRuntime — the base-forward bridge for the CAM serving path (the seam WS-C's /cam/* API needs).

WS-C's endpoints need `base_logits(token_ids) -> Tensor` (one frozen-base forward, no CAM staged) for the
write gate's prompt probe and the router-gated decode loop; `CAMMemory` deliberately does NOT run the base
(it operates on logits passed in). minisgl's *served* model lives in the backend scheduler process and is
not directly callable from the FastAPI frontend, so for the MVP the CAM runtime loads its OWN frozen base
(HF Qwen3.5-4B) co-located in the API process and serves /cam/* via the LOGIT-ONLY router path (no residual
tap needed — matches memory-organ's serve_gen). This decouples CAM from the backend.

TWO caveats this MVP carries (documented, not hidden):
  1. RDNA4: HF Qwen3.5-4B's gated-delta-net (GDN) layers use the `fla` path, which HANGS on gfx1201. The
     runtime patches them to the native `gdn_hip` kernels (minisgl ships them) when CAM_NATIVE_GDN=1 — the
     same patch memory-organ applies. Without a GDN-capable base the forward will hang, so we require it.
  2. Resources: a co-located base is a SECOND ~8 GB model alongside minisgl's backend model — fine on a
     2-card box or when CAM runs standalone, tight on one 16 GB card. Sharing the exact backend model
     (ZMQ control-plane into the scheduler) is the future first-class integration.

Enable by setting MINISGL_CAM=1 and MINISGL_CAM_CHECKPOINT=<dir> (a memory-organ export).
"""
import os
import json
import logging

import torch

logger = logging.getLogger(__name__)
_RUNTIME = None


def _patch_native_gdn(model):
    """Patch HF Qwen3.5 GDN layers to native gdn_hip (RDNA4-safe). No-op if unavailable or not requested."""
    if os.environ.get("CAM_NATIVE_GDN", "1") != "1":
        return model
    try:
        # minisgl ships the native Triton-free gdn_hip patcher (fla HANGS on gfx1201 as the tap opens).
        from minisgl.gdn.hf_patch import patch_qwen3_5_gdn
        n = patch_qwen3_5_gdn(model)
        logger.info("CAM: patched %d Qwen3.5 GDN layer(s) to native gdn_hip.", n)
        return model
    except Exception as e:  # noqa
        logger.warning("CAM: native-GDN patch unavailable (%s) — HF GDN will run its default fla path "
                       "(HANGS on RDNA4). Run on CUDA/CPU or fix the gdn_hip import.", e)
        return model


class CAMRuntime:
    """Co-located frozen base + tokenizer + CAMMemory. Provides the base_logits seam for /cam/*."""

    def __init__(self, checkpoint_dir: str, base_model_id: str = None, device: str = None):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
        from .memory import CAMMemory

        meta = json.load(open(os.path.join(checkpoint_dir, "meta.json")))
        base_model_id = base_model_id or meta.get("base_model") or "Qwen/Qwen3.5-4B"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base, last = None, None
        for loader in (AutoModelForCausalLM, AutoModelForImageTextToText):   # mirror memory-organ load_frozen_base
            try:
                base = loader.from_pretrained(base_model_id, dtype=torch.bfloat16,
                                              low_cpu_mem_usage=True).to(self.device).eval()
                break
            except Exception as e:  # noqa
                last = e
        if base is None:
            raise last
        for p in base.parameters():
            p.requires_grad_(False)
        base.config.use_cache = False
        base = _patch_native_gdn(base)
        self.base = base
        self.base_embed = base.get_input_embeddings()
        self.memory = CAMMemory(checkpoint_dir, self.base_embed, base.get_output_embeddings().weight)
        logger.info("CAMRuntime ready: base=%s device=%s cam_enabled=%s tap_layer=%s",
                    base_model_id, self.device, self.memory.enabled, getattr(self.memory, "tap_layer", None))

    @torch.no_grad()
    def base_logits(self, token_ids) -> torch.Tensor:
        """One frozen-base forward on token_ids (no CAM staged) -> last-position logits [vocab].

        transformers >=5.13's Qwen3.5 leaks fp32 activations (RMSNorm/rotary) into bf16 linears; autocast
        casts each matmul's inputs to the base dtype uniformly, so we don't chase per-layer dtype (this is
        why minisgl ships its own model — the HF path is dtype-fragile). CPU: autocast is a no-op guard."""
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        use_amp = self.device != "cpu"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            return self.base(inputs_embeds=self.base_embed(ids)).logits[0, -1]

    def encode(self, text: str):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids))


class BackendCAMRuntime:
    """Backend-shared CAM runtime: presents the same seam as ``CAMRuntime`` (``.tokenizer``,
    ``.memory``, ``.base_logits``) but backed by the ACTUAL served minisgl model (an in-process
    ``LLM``), so there is **no co-located HF copy** — the whole point of Task-2 model-share.

    The store+tap+router (``CAMMemory``) is the one the Engine builds in-process from the served
    weights (``engine.cam``); ``base_logits`` captures the served model's own logits (``LLM.base_logits``);
    and ``ask_tap`` delivers via the validated residual-tap seed-once path (a normal generate carrying
    ``sampling_params.mem_subject``). Enabled by ``MINISGL_CAM_BACKEND=1``.
    """

    def __init__(self, model_path: str = None):
        import torch
        from minisgl.llm import LLM

        meta_ckpt = os.environ.get("MINISGL_CAM_CHECKPOINT")
        base_model = model_path or os.environ.get("MINISGL_CAM_MODEL")
        if base_model is None and meta_ckpt:
            base_model = json.load(open(os.path.join(meta_ckpt, "meta.json"))).get("base_model")
        base_model = base_model or "Qwen/Qwen3.5-4B"
        mrr = int(os.environ.get("MINISGL_CAM_MAX_RUNNING_REQ", "4"))
        mratio = float(os.environ.get("MINISGL_CAM_MEMORY_RATIO", "0.85"))
        self.llm = LLM(model_path=base_model, dtype=torch.bfloat16, cuda_graph_max_bs=0, page_size=16,
                       memory_ratio=mratio, attention_backend="hip", max_running_req=mrr)
        self.tokenizer = self.llm.tokenizer
        self.memory = self.llm.engine.cam
        self.device = self.llm.engine.device
        if self.memory is None or not self.memory.enabled:
            raise RuntimeError("engine.cam not built — set MINISGL_CAM=1 + MINISGL_CAM_CHECKPOINT")
        logger.info("BackendCAMRuntime ready: served model=%s (model-share, no HF copy), tap_layer=%s",
                    base_model, getattr(self.memory, "tap_layer", None))

    def base_logits(self, token_ids):
        return self.llm.base_logits(list(token_ids))

    def ask_tap(self, prompt: str, subject: str, max_tokens: int = 32) -> str:
        """Deliver via the residual tap (seed-once) — a normal generate carrying mem_subject."""
        from minisgl.core import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, mem_subject=subject)
        return self.llm.generate([prompt], sp)[0]["text"]


class FrontendCAMRuntime:
    """MULTI-PROCESS model-share (#100): the CAM store (`engine.cam`) lives in the BACKEND scheduler
    process; this frontend runtime holds NO model and NO local store — only a tokenizer — and routes the
    two CAM ops through the backend by riding a normal generate request (the same seam `mem_subject`
    already uses for the tap):

      * remember(subject, object)  -> generate(mem_subject=subject, mem_remember=object_ids, max_tokens=1);
        the scheduler writes subject->object into engine.cam at prefill (write-only stub generation).
      * ask(prompt, subject)       -> generate(prompt, mem_subject=subject); the scheduler FORCES the exact
        stored object tokens (engine.cam.deliver_object_ids), then the served base continues the sentence.

    No second model copy, no new message types. Enabled by MINISGL_CAM_FRONTEND=1 (inside the full
    api_server). facts/forget/stats need a small control-plane message (follow-up) — they return data that
    does not fit a generate. `_state` (the FrontendManager) is resolved lazily at first request."""

    is_frontend_share = True

    def __init__(self, model_path: str = None):
        from transformers import AutoTokenizer

        meta_ckpt = os.environ.get("MINISGL_CAM_CHECKPOINT")
        if model_path is None and meta_ckpt and os.path.isfile(os.path.join(meta_ckpt, "meta.json")):
            model_path = json.load(open(os.path.join(meta_ckpt, "meta.json"))).get("base_model")
        self.model_path = model_path or os.environ.get("MINISGL_CAM_MODEL") or "Qwen/Qwen3.5-4B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._state = None
        logger.info("FrontendCAMRuntime ready: routes /cam/* to the backend engine.cam (no model copy).")

    def _get_state(self):
        if self._state is None:
            from minisgl.server.api_server import get_global_state
            self._state = get_global_state()
        return self._state

    def _sp(self, s: str):
        return list(self.tokenizer(" " + s, add_special_tokens=False).input_ids)

    def _render_nothink(self, messages) -> str:
        """Render chat messages to a raw prompt string with THINKING DISABLED, so a thinking model
        (Qwen3.x) answers directly instead of emitting a `<think>…</think>` monologue that swamps the
        JSON. `/no_think` in the prompt is unreliable — the template kwarg is the real switch. Falls back
        to the plain template (thinking on; the regex fallback in extract_facts then covers it), then to a
        joined string, if the tokenizer's template doesn't accept `enable_thinking`. We render here (not in
        the backend TokenizeManager, which never passes the kwarg) and send the resulting string."""
        tok = self.tokenizer
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:  # noqa: BLE001
                return "\n".join(str(m.get("content") or "") for m in messages)

    async def _generate(self, prompt: str, max_tokens: int, *, mem_subject: str = None,
                        mem_remember=None, mem_op: str = None) -> str:
        """One raw-prompt generation through the backend, carrying the CAM sampling params (mirrors
        InProcessBackendClient but raw-prompt + mem_subject/mem_remember/mem_op)."""
        from minisgl.core import SamplingParams
        from minisgl.message import TokenizeMsg

        state = self._get_state()
        uid = state.new_user()
        try:
            await state.send_one(TokenizeMsg(
                uid=uid, text=prompt,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=max(1, max_tokens),
                                               mem_subject=mem_subject, mem_remember=mem_remember,
                                               mem_op=mem_op)))
            text = ""
            async for ack in state.wait_for_ack(uid):
                text += ack.incremental_output
            return text
        except Exception:
            state.ack_map.pop(uid, None)
            state.event_map.pop(uid, None)
            raise

    async def _ctrl(self, op: str, subject: str = None, max_tokens: int = 2048):
        """A control op (facts/forget/stats): the backend force-emits the JSON result as the reply text."""
        import json
        txt = await self._generate(".", max_tokens=max_tokens, mem_subject=subject, mem_op=op)
        try:
            return json.loads(txt.strip())
        except (json.JSONDecodeError, ValueError):
            logger.warning("CAM %s: could not parse backend reply as JSON: %r", op, txt[:200])
            return None

    async def extract_facts(self, text: str, max_tokens: int = 256) -> list:
        """TRANSPARENT write: model-assisted extraction of durable (subject, object) facts stated in
        `text` (a plain generation — no CAM params). Returns [(subject, object), ...]; [] on none/parse
        failure. Deliberately conservative so chit-chat doesn't pollute the store."""
        import json
        import re
        instr = ('Extract only DURABLE factual statements the text asserts, as a JSON array of '
                 '{"subject","object"} objects (e.g. a person\'s language, a place\'s country). Ignore '
                 'questions, opinions, and chit-chat. Return [] if none. Return ONLY the JSON array, no '
                 f'prose.\n\nText: {text}')
        # Render with thinking OFF (a raw string carrying the chat template) so the model emits the JSON
        # directly; strip any residual <think> block before parsing (defence in depth).
        prompt = self._render_nothink([{"role": "user", "content": instr}])
        out = await self._generate(prompt, max_tokens=max_tokens)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
        m = re.search(r"\[.*\]", out, re.DOTALL)
        facts = []
        if m:
            try:
                arr = json.loads(m.group(0))
                facts = [(str(f["subject"]).strip(), str(f["object"]).strip()) for f in arr
                         if isinstance(f, dict) and f.get("subject") and f.get("object")]
            except (json.JSONDecodeError, ValueError):
                facts = []
        if facts:
            return facts
        # Deterministic fallback (small thinking models extract JSON unreliably): explicit fact statements
        # "the <attr> of SUBJECT is OBJECT" / "SUBJECT's <attr> is OBJECT" / "SUBJECT is OBJECT", where both
        # SUBJECT and OBJECT are capitalised (proper-noun-shaped) — filters out "X is nice" style non-facts.
        pat = re.compile(r"(?:the\s+[\w ]+?\s+of\s+)?"
                         r"([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)*)(?:'s\s+[\w ]+?)?"
                         r"\s+(?:is|was|are|were)\s+([A-Z][A-Za-z.'-]+)")
        seen = set()
        for mm in pat.finditer(text):
            s = mm.group(1).strip().strip(".,;:!?'\"")
            o = mm.group(2).strip().strip(".,;:!?'\"")
            if s and o and (s, o) not in seen:
                seen.add((s, o)); facts.append((s, o))
        return facts

    async def retrieve(self, prompt: str, max_tokens: int = 512) -> list:
        """TRANSPARENT read: ask the backend which stored subjects this prompt mentions (cosine-matched,
        tau-gated) -> [{subject, object}]. The prompt itself is the query (backend extracts spans)."""
        import json
        txt = await self._generate(prompt, max_tokens=max_tokens, mem_op="retrieve")
        try:
            return json.loads(txt.strip()) or []
        except (json.JSONDecodeError, ValueError):
            return []

    async def facts(self) -> list:
        return (await self._ctrl("facts")) or []

    async def forget(self, subject: str) -> bool:
        return bool(await self._ctrl("forget", subject=subject))

    async def stats(self) -> dict:
        return (await self._ctrl("stats")) or {}

    async def remember(self, subject: str, object_str: str, prompt: str = None) -> bool:
        """Write subject->object into the backend engine.cam. Returns True if stored, False if skipped.

        Base-uncertainty gate (opt-in, MINISGL_CAM_WRITE_GATE=1): before writing, probe the served base
        with the relation prompt (one short no-CAM generation). If the base ALREADY produces `object_str`,
        the fact is base-known — storing it wastes a slot and the pointer would only re-deliver what the
        base says anyway — so skip. Off by default (the delivery contract is harmless on base-known facts).
        The probe rides the normal generate path (no base_logits seam needed in the multi-process frontend);
        it is a "does the base emit this object" test, which is exactly the delivery-relevant signal."""
        probe = prompt or f"The mother tongue of {subject} is"
        if os.environ.get("MINISGL_CAM_WRITE_GATE") == "1":
            n = len(self._sp(object_str))
            cont = await self._generate(probe, max_tokens=max(4, n + 2))
            if object_str.strip().lower() in cont.strip().lower():
                logger.debug("CAM write-gate: base already emits %r for %r; skipping store.",
                             object_str, subject)
                return False
        await self._generate(probe, max_tokens=1,
                             mem_subject=subject, mem_remember=self._sp(object_str))
        return True

    async def ask(self, prompt: str, subject: str, max_tokens: int = 32) -> str:
        """Retrieve: the backend forces the stored object tokens (pointer), then the base continues."""
        return await self._generate(prompt, max_tokens=max_tokens, mem_subject=subject)


def get_cam_runtime():
    """Lazy singleton. Returns None (so /cam/* replies 503) when CAM is not configured or fails to load.

    Three modes: MINISGL_CAM_FRONTEND=1 -> FrontendCAMRuntime (multi-process; routes to the backend
    engine.cam, no model copy); MINISGL_CAM_BACKEND=1 -> BackendCAMRuntime (single-process in-engine share,
    no HF copy); otherwise the co-located CAMRuntime (its own frozen HF base — the standalone MVP)."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    if os.environ.get("MINISGL_CAM_FRONTEND") == "1":
        try:
            _RUNTIME = FrontendCAMRuntime()               # store is in the backend; no checkpoint dir needed
            return _RUNTIME
        except Exception as e:  # noqa
            logger.error("CAM: frontend runtime load failed — /cam/* disabled. (%s)", e)
            return None
    ckpt = os.environ.get("MINISGL_CAM_CHECKPOINT")
    if not ckpt or not os.path.isdir(ckpt):
        logger.warning("CAM: MINISGL_CAM_CHECKPOINT unset/missing (%s) — /cam/* disabled.", ckpt)
        return None
    try:
        if os.environ.get("MINISGL_CAM_BACKEND") == "1":
            _RUNTIME = BackendCAMRuntime()
        else:
            _RUNTIME = CAMRuntime(ckpt)
    except Exception as e:  # noqa
        logger.error("CAM: runtime load failed — /cam/* disabled. (%s)", e)
        return None
    return _RUNTIME
