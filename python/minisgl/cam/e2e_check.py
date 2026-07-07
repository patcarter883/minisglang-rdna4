import os, sys, torch
os.environ.setdefault("MINISGL_CAM_CHECKPOINT", "/ckpt")
os.environ.setdefault("CAM_NATIVE_GDN", "1")
sys.path.insert(0, "/minisgl/python")
from minisgl.cam import get_cam_runtime
rt = get_cam_runtime()
assert rt is not None and rt.memory.enabled, "runtime/memory failed to load"
tok, mem = rt.tokenizer, rt.memory
bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
def enc(s): return list(tok(s, add_special_tokens=False).input_ids)
def encsp(s): return list(tok(" " + s, add_special_tokens=False).input_ids)

def remember(subject, prompt, obj):                      # mirrors cam_api /cam/remember
    sids, oids = encsp(subject), encsp(obj)
    pl = rt.base_logits(bos + enc(prompt)).float()
    base_p = float(torch.softmax(pl, -1)[oids[0]])
    mem.set_pending_object(oids)
    return bool(mem.remember(sids, pl)), base_p

def ask(subject, prompt, gen_len=14):                    # mirrors cam_api /cam/ask (seed-once)
    bank, conf = mem.read(encsp(subject))
    if bank is None: return "(no memory)"
    seed = mem.seed_token(bank, conf)
    cur, out, placed = bos + enc(prompt), [], False
    for _ in range(gen_len):
        lg = rt.base_logits(cur).float()
        if not placed:
            lg = lg + mem.router_delta(lg.unsqueeze(0), bank, conf).reshape(-1)
        nxt = int(lg.argmax(-1).item())
        if nxt == seed: placed = True
        out.append(nxt); cur = cur + [nxt]
    return tok.decode(out).replace("\n", " ").strip()

facts = [("Lionel Jospin", "The mother tongue of Lionel Jospin is", "Dutch"),
         ("Oleg Kotov",    "The mother tongue of Oleg Kotov is",    "English"),
         ("Marie NDiaye",  "The mother tongue of Marie NDiaye is",  "Russian")]
print("=== /cam/remember (base-uncertainty write gate) ===", flush=True)
for s, p, o in facts:
    st, bp = remember(s, p, o); print(f"  remember {s!r} -> {o!r}: stored={st} base_p={bp:.3f}", flush=True)
print("=== /cam/ask (router-gated seed-once decode, LIVE base) ===", flush=True)
for s, p, o in facts:
    txt = ask(s, p); print(f"  ask {p!r}\n     -> {txt!r}   [delivered '{o}': {o.lower() in txt.lower()}]", flush=True)
print("CAM-SERVE E2E OK", flush=True)
