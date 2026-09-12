"""w8a8_moe is a SHIPPED path; adopting the fused gemm1+silu must not change what it computes.

WHY THIS EXISTS. `w8a8_moe` ran gemm1 -> (P, 2*inter) -> a separate silu_and_mul launch, and its own
comment recorded why: "the gemm1-epilogue fused silu is wmma-only -> unusable at decode where gemm1
must be gemv". That was true — the fp8-WEIGHT decode GEMV loader had never been instantiated with
SILU=true (rdna4-hip-kernels/FORMAT_MATRIX.md G8) while its int4 twin had, and the host launcher
hard-rejected any kernel id but 'wmma'. With G8 landed both bands fuse, and this pins that adopting
it did not move the numbers.

NOT bit-identical, and not asserted to be: the fused epilogue applies the per-output-channel weight
scale to gate and up in fp32 and then computes silu, while the unfused path rounds both halves
through the output dtype first. MINISGL_MOE_FUSED_SILU=0 selects the old arm, which is what makes
the comparison possible at all.

M spans the gemm1 GEMV/GEMM crossover (_MOE_GEMM1_GEMV_MAX = 32) so both fused kernels are covered.

Run under a lease inside the serve image:
    PYTHONPATH=/engine/python:/opt/kernels python3 /engine/tests/w8a8_moe_fused_silu_parity_test.py
"""
import importlib, os, sys, torch
sys.path.insert(0,"/engine/python"); sys.path.append("/opt/kernels")
from minisgl.distributed import set_tp_info
set_tp_info(0,1)
dev=torch.device("cuda")
E,INTER,K,TOPK,M = 8,256,512,2,None
FAILED=[]
def check(n,ok,d=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {n}{('  — '+d) if d else ''}")
    if not ok: FAILED.append(n)
torch.manual_seed(7)
def to_fp8(w):
    s=(w.abs().amax(dim=2).clamp_min(1e-6).float()/448.0)
    return (w.float()/s.unsqueeze(2)).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).contiguous(), s.contiguous()
w13=torch.randn(E,2*INTER,K,device=dev)*0.05
w2=torch.randn(E,K,INTER,device=dev)*0.05
w13q,w13s=to_fp8(w13); w2q,w2s=to_fp8(w2)   # kByte views: check_w8a8_weights wants uint8, not float8
def run(M,fused):
    os.environ["MINISGL_MOE_FUSED_SILU"]="1" if fused else "0"
    import minisgl.quant.kernels as kk; kk=importlib.reload(kk)
    torch.manual_seed(100+M)
    x=torch.randn(M,K,dtype=torch.bfloat16,device=dev)*0.1
    g=torch.randn(M,E,dtype=torch.float32,device=dev)
    return kk.w8a8_moe(x,w13q,w13s,w2q,w2s,g,TOPK,True)
for M in (1,2,8,64):
    f=run(M,True).float(); u=run(M,False).float()
    rel=(f-u).abs().max()/u.abs().max().clamp_min(1e-6)
    check(f"M={M:<3} fused gemm1+silu matches unfused", rel<3e-2, f"rel={rel:.3e}")
    check(f"M={M:<3} the fused path actually RAN", not torch.equal(f,u))
    check(f"M={M:<3} finite and non-degenerate", torch.isfinite(f).all() and f.abs().max()>1e-5)
print("")
print(f"FAILED ({len(FAILED)}): "+"; ".join(FAILED) if FAILED else "ALL CHECKS PASS")
sys.exit(1 if FAILED else 0)
