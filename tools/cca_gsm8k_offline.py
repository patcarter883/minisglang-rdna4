"""Offline GSM8K-style answer-corruption repro: shared few-shot + real questions, long CoT, radix
ON vs OFF, batched. Reports how many FINAL ANSWERS differ (the real quality signal, not tail ULP)."""
import argparse, re, torch
def build(model, radix, graph, me=8192):
    from minisgl.llm import LLM
    return LLM(model_path=model, dtype=torch.bfloat16, cuda_graph_max_bs=graph, page_size=16,
               memory_ratio=0.85, attention_backend="hip", max_running_req=8, gdn_radix=radix, cache_type="radix", max_extend_tokens=me)
FEW=("Question: Natalia sold clips to 48 friends in April, then half as many in May. How many total?\n"
 "Answer: April 48, May 48/2=24, total 48+24=72.\n#### 72\n\n"
 "Question: Weng earns $12/hour babysitting. She did 50 minutes. How much?\n"
 "Answer: 12/60=$0.2 per min, 50*0.2=$10.\n#### 10\n\n"
 "Question: Betty needs $100, has half. Parents give $15, grandparents twice that. How much more?\n"
 "Answer: half is 50, parents 15, grandparents 30, total 95, needs 100-95=5.\n#### 5\n\n"
 "Question: Julie reads a 120-page book. Read 12 yesterday, twice that today. Half of remaining tomorrow?\n"
 "Answer: today 24, so far 36, remaining 84, half 42.\n#### 42\n\n")
QS=["A robe takes 2 bolts blue and half that white. How many bolts total?",
    "James writes a 3-page letter to 2 friends twice a week. How many pages a year?",
    "A shop has 20 apples. Sells 8 morning, 5 afternoon. How many left?",
    "Tom has 4 boxes of 6 pencils. He gives away 7. How many left?",
    "A train goes 60 mph for 3 hours then 40 mph for 2 hours. Total miles?",
    "Sarah has $50, buys 3 books at $8 each. How much left?",
    "A farmer has 15 cows and 3 times as many chickens. Total animals?",
    "Lisa runs 5 km daily for a week except Sunday. Total km?"]
def ans(txt):
    m=re.findall(r"####\s*\$?(-?[0-9][0-9,]*\.?[0-9]*)",txt)
    if m: return m[-1].replace(",","")
    n=re.findall(r"-?[0-9][0-9,]*\.?[0-9]*",txt); return (n[-1].replace(",","") if n else "")
def run(a):
    from minisgl.core import SamplingParams
    llm=build(a.model, a.radix=="on", a.graph, a.max_extend); print(f"rec_radix={getattr(llm,'_rec_radix',False)}",flush=True)
    tok=llm.tokenizer
    prompts=[tok.encode(FEW+f"Question: {q}\nAnswer:",add_special_tokens=True) for q in QS]
    sp=SamplingParams(temperature=0.0,max_tokens=200,ignore_eos=True)
    if a.sequential:
        out=[llm.generate([list(p)], sp)[0] for p in prompts]
    else:
        out=llm.generate([list(p) for p in prompts], sp)
    res=[{"q":QS[i],"txt":out[i]["text"],"ans":ans(out[i]["text"])} for i in range(len(QS))]
    torch.save({"radix":a.radix,"res":res}, a.out); print("saved",a.out,flush=True)
def cmp(p1,p2):
    on,off=torch.load(p1),torch.load(p2); nd=0
    for i,(o,f) in enumerate(zip(on["res"],off["res"])):
        same=o["ans"]==f["ans"]; nd+=(not same)
        print(f"[{i}] {'OK ' if same else 'DIFF'} ON_ans={o['ans']!r:8} OFF_ans={f['ans']!r:8}  {o['q'][:40]}")
    print(f"\n{nd}/{len(on['res'])} FINAL ANSWERS differ ON vs OFF")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--radix",choices=["on","off"]); ap.add_argument("--model",default="/root/.cache/huggingface/ZAYA1-8B-RXF-h32")
    ap.add_argument("--graph",type=int,default=8); ap.add_argument("--sequential",action="store_true"); ap.add_argument("--max-extend",type=int,default=8192); ap.add_argument("--out",default="/engine/tools/_g.pt"); ap.add_argument("--compare",nargs=2)
    a=ap.parse_args()
    if a.compare: cmp(*a.compare)
    else: run(a)
main()
