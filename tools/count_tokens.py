import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("QuantTrio/GLM-4.7-Flash-AWQ")
raw = json.load(open("/engine/tools/glm_loadtest_raw.json"))
for o in raw:
    toks = [len(tok.encode(r["text"])) for r in o["results"]]
    total = sum(toks)
    print(f"### {o['label']}: wall={o['wall']:.2f}s  total_completion_tokens={total}  AGG={total/o['wall']:.1f} tok/s")
    for i, (r, nt) in enumerate(zip(o["results"], toks)):
        print(f"   req{i}: {nt} tok in {r['elapsed']:.2f}s = {nt/r['elapsed']:.1f} tok/s")
