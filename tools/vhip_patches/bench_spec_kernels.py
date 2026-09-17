"""Why is the gdn_hip spec path slower? Time OUR two verify ops vs vLLM's fused Triton spec kernels
at the real in-serve shapes. Standalone, no serve."""
import torch, time, gdn_hip
DEV="cuda"; NK,NV,HK,HV=8,16,128,128
C=HK*NK*2+HV*NV; W,NUM_SPEC=4,2; WM1=W-1; SLOTS=8; N=1; T=3   # bs=1, K=2 -> 3 tokens
cu=torch.tensor([0,T],dtype=torch.int32,device=DEV)
x=torch.randn(T,C,device=DEV).float(); wt=torch.randn(C,W,device=DEV).float()*0.1
a=torch.randn(T,NV,device=DEV).float(); b=torch.randn(T,NV,device=DEV).float()
A_log=torch.randn(NV,device=DEV).float(); dt=torch.randn(NV,device=DEV).float()
idx=torch.tensor([1],dtype=torch.long,device=DEV); hi=torch.ones(N,dtype=torch.uint8,device=DEV)
conv=torch.zeros(SLOTS,C,WM1+NUM_SPEC,device=DEV).float()
ssm=torch.zeros(SLOTS,NV,HV,HK,device=DEV,dtype=torch.bfloat16)
scale=HK**-0.5

def timeit(fn,n=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t)/n*1000

def ours():
    co,_=gdn_hip.causal_conv1d_fwd_verify(x,wt,None,cu,idx,hi,conv,T,1)
    q,k,v=co.split([HK*NK,HK*NK,HV*NV],dim=-1)
    gdn_hip.gdn_prefill_verify(q.reshape(-1,NK,HK).contiguous(),k.reshape(-1,NK,HK).contiguous(),
        v.reshape(-1,NV,HV).contiguous(),a,b,A_log,dt,cu,idx,hi,ssm,T,scale,1)

def gather_scatter():
    ck=conv.new_zeros((2,*conv.shape[1:])); sk=ssm.new_zeros((2,*ssm.shape[1:]))
    ck[1:]=conv[idx]; sk[1:]=ssm[idx]
    conv[idx]=ck[1:]; ssm[idx]=sk[1:]

print(f"ours: conv_verify+ssm_verify        {timeit(ours):7.3f} ms/layer")
print(f"      slot gather+scatter           {timeit(gather_scatter):7.3f} ms/layer")
try:
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update as fz
    ssi=torch.tensor([[1,2,3]],dtype=torch.int32,device=DEV); na=torch.ones(N,dtype=torch.int32,device=DEV)
    co,_=gdn_hip.causal_conv1d_fwd_verify(x,wt,None,cu,idx,hi,conv,T,1)
    q,k,v=(t.reshape(-1,NK if i<2 else NV,HK if i<2 else HV).contiguous().bfloat16()
           for i,t in enumerate(co.split([HK*NK,HK*NK,HV*NV],dim=-1)))
    def theirs():
        fz(A_log=A_log,a=a.bfloat16(),b=b.bfloat16(),dt_bias=dt,q=q,k=k,v=v,initial_state=ssm,
           inplace_final_state=True,cu_seqlens=cu,ssm_state_indices=ssi,num_accepted_tokens=na,
           use_qk_l2norm_in_kernel=True)
    print(f"vLLM: fused_sigmoid_gating (ssm only) {timeit(theirs):7.3f} ms/layer")
except Exception as e:
    print(f"vLLM fused kernel bench skipped: {type(e).__name__}: {str(e)[:120]}")
