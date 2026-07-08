"""De-risk driver: does THIS checkpoint's residual TAP (not the router) deliver in-serve?

Task 2 was scoped to Option B (first-class residual-tap integration). Only the LOGIT router path
has delivered live (3/3, Task 1); the residual TAP is unvalidated in-serve. Before building weeks of
tap machinery in the minisgl engine, confirm the mechanism on the proven co-located HF base by
hooking `CAMMemory.apply_tap` into the base's decoder layer `tap_layer` and greedily decoding with
NO router_delta. A tap-OFF baseline is printed for contrast (base alone should NOT know the edited
object — the write gate only stored base-unknowable facts).

Run in the lean image on a GPU lease (see scratchpad/run_tap.sh):
    python /engine/python/minisgl/cam/tap_check.py
"""
import os, torch
os.environ.setdefault("MINISGL_CAM_CHECKPOINT", "/ckpt")
os.environ.setdefault("CAM_NATIVE_GDN", "1")
from minisgl.cam import get_cam_runtime

rt = get_cam_runtime()
assert rt is not None and rt.memory.enabled, "runtime/memory failed to load"
tok, mem = rt.tokenizer, rt.memory
bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
enc = lambda s: list(tok(s, add_special_tokens=False).input_ids)
encsp = lambda s: list(tok(" " + s, add_special_tokens=False).input_ids)


def _decoder_layers(base):
    """Locate the ModuleList of decoder layers on the HF base (Qwen3_5ForCausalLM.model.layers)."""
    m = getattr(base, "model", base)
    layers = getattr(m, "layers", None)
    if layers is None and hasattr(m, "model"):
        layers = getattr(m.model, "layers", None)
    assert layers is not None, f"could not find decoder layers on {type(base).__name__}"
    return layers


LAYERS = _decoder_layers(rt.base)
TAP_L = mem.tap_layer
print(f"base={type(rt.base).__name__}  n_layers={len(LAYERS)}  tap_layer={TAP_L}", flush=True)

_staged = {"bank": None, "conf": None}


def _tap_hook(module, inputs, output):
    if _staged["bank"] is None:
        return output
    is_tuple = isinstance(output, tuple)
    h = output[0] if is_tuple else output          # [1, T, H]
    h2 = mem.apply_tap(h[0], _staged["bank"], _staged["conf"]).unsqueeze(0).to(h.dtype)
    return ((h2,) + tuple(output[1:])) if is_tuple else h2


LAYERS[TAP_L].register_forward_hook(_tap_hook)


def remember(subject, prompt, obj):
    sids, oids = encsp(subject), encsp(obj)
    pl = rt.base_logits(bos + enc(prompt)).float()
    base_p = float(torch.softmax(pl, -1)[oids[0]])
    mem.set_pending_object(oids)
    return bool(mem.remember(sids, pl)), base_p


def gen(prompt, subject, mode, gen_len=14):
    """Greedy decode. mode: 'off' (no tap), 'always' (tap every step), 'seed' (tap until the object's
    first token lands, then clear — the intended Phase-1 discipline, mirrors clear_cam()). NO router."""
    seed = None
    if mode in ("always", "seed"):
        bank, conf = mem.read(encsp(subject))
        _staged["bank"], _staged["conf"] = bank, conf
        seed = int(mem.seed_token(bank, conf)) if mode == "seed" else None
    else:
        _staged["bank"], _staged["conf"] = None, None
    cur, out = bos + enc(prompt), []
    for _ in range(gen_len):
        lg = rt.base_logits(cur).float()            # tap active iff a bank is staged
        nxt = int(lg.argmax(-1).item())
        if mode == "seed" and nxt == seed:          # seed-once: object landed -> stop injecting
            _staged["bank"], _staged["conf"] = None, None
        out.append(nxt); cur = cur + [nxt]
    _staged["bank"], _staged["conf"] = None, None
    return tok.decode(out).replace("\n", " ").strip()


facts = [("Lionel Jospin", "The mother tongue of Lionel Jospin is", "Dutch"),
         ("Oleg Kotov",    "The mother tongue of Oleg Kotov is",    "English"),
         ("Marie NDiaye",  "The mother tongue of Marie NDiaye is",  "Russian")]

print("=== write gate ===", flush=True)
for s, p, o in facts:
    st, bp = remember(s, p, o); print(f"  remember {s!r} -> {o!r}: stored={st} base_p={bp:.3f}", flush=True)

print("=== TAP delivery (residual inject at L24, NO router) — OFF / always / seed-once ===", flush=True)
n_always = n_seed = 0
for s, p, o in facts:
    off = gen(p, s, "off")
    alw = gen(p, s, "always")
    sd = gen(p, s, "seed")
    n_always += o.lower() in alw.lower()
    n_seed += o.lower() in sd.lower()
    print(f"  {p!r}", flush=True)
    print(f"     OFF   -> {off!r}", flush=True)
    print(f"     TAP   -> {alw!r}   [delivered '{o}': {o.lower() in alw.lower()}]", flush=True)
    print(f"     SEED  -> {sd!r}   [delivered '{o}': {o.lower() in sd.lower()}]", flush=True)
print(f"CAM-TAP DELIVERY always={n_always}/{len(facts)} seed-once={n_seed}/{len(facts)}", flush=True)
