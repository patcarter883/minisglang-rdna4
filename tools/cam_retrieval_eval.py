#!/usr/bin/env python3
"""#10 CAM retrieval-quality eval — measures the subject-key delivery precision/recall on the REAL base
embeddings (the same pooled-embedding key the store uses), over paraphrases (should match) and distractors
(should NOT), at a given deliver_tau, with and without subject canonicalization (MINISGL_CAM_CANON).

CPU-only; no GPU / no server. Run in the lean image so torch + the model's embed matrix are available:

  docker run --rm --entrypoint bash -v $HF:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    -v <repo>:/engine minisgl-rdna4:lean -lc \
    'source /opt/venv/bin/activate; python /engine/tools/cam_retrieval_eval.py'

This measures the RETRIEVAL (key) half of #10. Extraction (write) precision/recall needs a live-server
eval (the LLM fact extractor) and is a separate GPU harness. A trained semantic key is memory-organ
CAM_GTE_KEYS; this tool quantifies how far the cheap pooled key + canonicalization gets first.
"""
import glob, json, os, re, sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

MODEL = os.environ.get("CAM_EVAL_MODEL", "Qwen/Qwen3.5-4B")
TAU = float(os.environ.get("MINISGL_CAM_DELIVER_TAU", "0.7"))


def canon(text, on):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip() if on else text


def load_embed():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/*"))[0]
    key = fpath = None
    idx = glob.glob(base + "/*.index.json")
    if idx:
        for k, f in json.load(open(idx[0]))["weight_map"].items():
            if k.endswith("embed_tokens.weight"):
                key, fpath = k, os.path.join(base, f); break
    from safetensors import safe_open
    if key is None:
        for f in glob.glob(base + "/*.safetensors"):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    if k.endswith("embed_tokens.weight"):
                        key, fpath = k, f; break
            if key: break
    with safe_open(fpath, framework="pt") as sf:
        return tok, sf.get_tensor(key).float()


def subj_key(tok, W, text, on):
    ids = tok(" " + canon(text, on), add_special_tokens=False).input_ids
    return F.normalize(W[torch.tensor(ids)].mean(0), dim=-1)


# Labelled set: each stored subject + a list of paraphrase queries that MUST retrieve it.
STORED = {
    "Zephyrina Quillsworth": ["Zephyrina Quillsworth", "zephyrina quillsworth", "Quillsworth Zephyrina",
                               "Ms. Zephyrina Quillsworth", "Zephyrina Quillsworth."],
    "Bartholomew Fizzwick": ["Bartholomew Fizzwick", "bartholomew fizzwick", "Mr. Bartholomew Fizzwick"],
    "Cornelius Blackwood": ["Cornelius Blackwood", "cornelius blackwood", "Blackwood Cornelius"],
    "Delphine Ravenscroft": ["Delphine Ravenscroft", "delphine ravenscroft"],
}
# Distractors: DIFFERENT subjects that must NOT match any stored one (>=tau = false delivery).
DISTRACTORS = ["Aldous Quillsworth", "Zephyrina Blackwood", "Ophelia Nightingale",
               "Thaddeus Grimwald", "Montgomery Ashcroft"]


def evaluate(tok, W, on, tau):
    keys = {s: subj_key(tok, W, s, on) for s in STORED}
    names = list(keys); mat = torch.stack([keys[n] for n in names])

    def nn(text):
        q = subj_key(tok, W, text, on)
        sims = mat @ q
        j = int(sims.argmax()); return (names[j], float(sims[j])) if float(sims[j]) >= tau else (None, float(sims[j]))

    tp = fn = 0
    for s, paras in STORED.items():
        for p in paras:
            pred, _ = nn(p)
            if pred == s: tp += 1
            else: fn += 1
    false_deliv = sum(1 for d in DISTRACTORS if nn(d)[0] is not None)
    recall = tp / (tp + fn)
    fdr = false_deliv / len(DISTRACTORS)
    return recall, fdr, tp, fn, false_deliv


def main():
    tok, W = load_embed()
    print(f"model={MODEL} tau={TAU}\n")
    print(f"{'canon':<8}{'recall':>10}{'false-deliv':>14}{'detail':>22}")
    for on in (False, True):
        r, fdr, tp, fn, fd = evaluate(tok, W, on, TAU)
        print(f"{str(on):<8}{r:>10.3f}{fdr:>14.3f}   {f'tp={tp} fn={fn} falsedeliv={fd}':>22}")
    print("\nInterpretation: recall = paraphrase queries that retrieved the right subject; false-deliv ="
          " distractors that wrongly matched a stored subject. Canonicalization should lift recall"
          " (case/punct paraphrases) without raising false-delivery.")


if __name__ == "__main__":
    sys.exit(main())
