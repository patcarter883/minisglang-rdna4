import concurrent.futures as cf, json, os, sys, urllib.request
N=int(os.environ.get("NREQ","3")); TO=int(os.environ.get("TOUT","50"))
URL="http://localhost:1919/v1/chat/completions"
def one(i):
    b=json.dumps({"model":"x","messages":[{"role":"user","content":f"Say the number {i} in words."}],"temperature":0,"max_tokens":24,"seed":0}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(r,timeout=TO) as x: d=json.loads(x.read())
        return i,repr(d["choices"][0]["message"]["content"])[:60]
    except Exception as e: return i,f"ERR:{type(e).__name__}"
with cf.ThreadPoolExecutor(max_workers=N) as ex:
    res=dict(ex.map(one,range(N)))
for i in sorted(res): print(i,res[i])
