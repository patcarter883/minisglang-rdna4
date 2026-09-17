"""Does gdn_prefill_wmma write the final recurrent state back into the ssm_state tensor in place?

The vhip spec+prefill shim assumes it does: it hands the kernel the dense
`ssm_state[prefill_state_indices]` gather and returns that same tensor as `last_recurrent_state`
for vLLM to write back. If the kernel does NOT mutate it, the write-back stores the *initial*
state, every later decode step reads a stale state, and generation emits one correct token then
degenerates to token 0 ("!!!!") — exactly the observed failure.
"""

import torch
import gdn_hip

torch.manual_seed(0)
DEV = "cuda"
NV, NK, HK, HV = 4, 4, 128, 128       # heads / head dims (128 => WMMA path)
SEQS = [7, 5]
T = sum(SEQS)

cu = torch.tensor([0, *torch.cumsum(torch.tensor(SEQS), 0).tolist()], dtype=torch.int32, device=DEV)
q = torch.randn(T, NK, HK, device=DEV).float()
k = torch.randn(T, NK, HK, device=DEV).float()
v = torch.randn(T, NV, HV, device=DEV).float()
a = torch.randn(T, NV, device=DEV).float()
b = torch.randn(T, NV, device=DEV).float()
A_log = torch.randn(NV, device=DEV).float()
dt_bias = torch.randn(NV, device=DEV).float()

state = torch.zeros(len(SEQS), NV, HV, HK, device=DEV).float()
state_indices = torch.arange(len(SEQS), device=DEV, dtype=torch.long)
has_init = torch.ones(len(SEQS), device=DEV, dtype=torch.uint8)

before = state.clone()
core = gdn_hip.gdn_prefill_wmma(
    q, k, v, a, b, A_log, dt_bias, cu, state_indices, has_init, state, HK ** -0.5, 1
)
after = state

changed = not torch.equal(before, after)
print(f"core out shape : {tuple(core.shape)}  dtype={core.dtype}")
print(f"state mutated in place : {changed}")
print(f"state |max| before={before.abs().max().item():.6f} after={after.abs().max().item():.6f}")
if not changed:
    print("\n=> KERNEL DOES NOT WRITE STATE IN PLACE. The shim's `return ..., state` hands back the")
    print("   INITIAL state, so vLLM caches a stale state -> one good token then degeneration.")
else:
    print("\n=> in-place write confirmed; the shim's state contract is correct, look elsewhere.")
