# P2prime — mixed-media grouped MoE GEMM on explicit two-stack pointers

*Probe spec:* the Phase 0 gate report §6 unknown #1 (`docs/measurements/WEIGHT_OFFLOAD_2026-09-02/PHASE0_REPORT.md`) — **is per-layer time LINEAR or CLIFFED in miss count?**
*Run:* 2026-09-02T22:51:29.549+00:00 → 2026-09-02T22:52:09.350+00:00 · *status:* **LINEAR**
*Worktree:* `/home/pat/code/minisgl-rdna4-offload` @ `8bcc7035d10937f0720cb3148f2acb473ab24fa2` (`feat/weight-offload`)
*Kernels:* `/home/pat/code/rdna4-hip-kernels/fp8_wmma/fp8_wmma_rocm` @ `169be577f36faed0842e9158f145c2605789075d` (dirty=False)

## Verdict

**LINEAR** — both cards agree: LINEAR. Per-expert placement is worth building only if it beats LAYER-GRANULAR placement at the same byte budget; the measured gain is in `best_gain_over_layer_granular` per card.

| card | BDF | status | cliff_index | shape | gain over layer-granular |
|---|---|---|---|---|---|
| 0 | `0000:03:00.0` | OK | 0.090 | LINEAR | 1.063x |
| 1 | `0000:07:00.0` | OK | 0.094 | LINEAR | 1.050x |

`cliff_index = (t(1) − t(0)) / (t(n) − t(0))`: **1/n ⇒ LINEAR** (misses cost additively, per-expert placement is worth building), **≈1.0 ⇒ CLIFFED** (one host expert costs what all of them do, per-expert placement buys ≈ zero and M2 is layer-granular).

## Mechanism

Two stacks, because a mixed-media single VA is **not constructible on this box** (P1/P2/P3: `hipMemCreate(location=Host)` silently returns VRAM):

* device stack — `hipMalloc`;
* host stack — `hipHostMalloc(Mapped|Portable)` + `hipHostGetDevicePointer`;
* a per-expert `__constant__` pointer table (weights **and** scales **and** zero-points, always together — a granule is one expert's slice of every tensor).

The kernel is the **shipped** `gemv_decode::gemv_decode_core`; the two-stack layout is a WLoad policy (`TwoStackInt4A16GemvLoader`) that inherits the shipped `Int4A16GemvLoader<__half>` and replaces only the three per-expert base-pointer hooks. Per `KERNEL_CORE_POLICY.md` a placement scheme is a loader policy, never a kernel fork.

## ROCm device 0 — AMD Radeon RX 9070 XT `0000:03:00.0`

*status:* **OK**

### Preconditions (placement is MEASURED, never declared)

| check | ok | detail |
|---|---|---|
| `device_stack_consumes_vram` | ✅ | classification=device_resident via amdgpu sysfs (mem_info_vram_used / mem_info_gtt_used); sysfs vram delta frac=1.169 |
| `host_stack_not_in_vram` | ✅ | sysfs vram delta frac=0.002 (must be < 0.25); THIS is the check P2 never made |
| `host_stack_is_anonymous_host_mapping` | ✅ | host maps_entry='7f0bf7400000-7f0c42200000 rw-p 00000000 00:00 0 ' (must be anonymous private rw-p, NOT /dev/dri/*); the device stack for contrast maps '7f0c42400000-7f0c8d200000 rw-s 101d5a000 00:07 560                       /dev/dri/renderD128' |
| `host_stack_consumes_host_ram` | ✅ | MemAvailable delta frac=0.997 (corroborating target >= 0.5). ADVISORY ONLY on a shared box -- it does not gate. Host residency is established by host_stack_not_in_vram, host_stack_is_cpu_accessible, host_stack_is_anonymous_host_mapping and media_separation_ratio, all of which are card-local or direct. |
| `host_stack_is_cpu_accessible` | ✅ | cpu read at host ptr: {'readable': True, 'read_errno': None, 'read_strerror': None}; P1 found the fake-host VMM pages were NOT CPU-accessible (---s on renderD128, os.write -> EFAULT) |
| `stacks_do_not_alias` | ✅ | device VA [0x7f0c42400000,0x7f0c8d200000) vs host device-visible VA [0x7f0bf7400000,0x7f0c42200000); overlap=0 B (must be 0). host_ptr=0x7f0bf7400000 device_ptr=0x7f0bf7400000 (equal is NORMAL under ROCm's unified VA -- see P1); the two stacks are separate ALLOCATIONS on different media and cannot be interleaved, which is what makes this a two-stack design |
| `read_checksums_valid` | ✅ | device ok=True host ok=True; a wrong checksum means the read did not land on the pages we think it did |
| `media_separation_ratio` | ✅ | device 389.9 / host 28.9 = 13.48x (need >= 4.0x). P2 measured 1.00x here and that is what proved the 'host' pages were VRAM |
| `device_read_is_hbm_speed` | ✅ | 389.91448368191783 GB/s (need >= 100.0) |
| `host_read_is_pcie_bounded` | ✅ | host 28.91633977106866 GB/s vs copy engine 28.675520095447137 GB/s x 1.25; bytes that cross PCIe cannot materially outrun the copy engine on the same card |

device read **389.9 GB/s** · host read **28.9 GB/s** · copy engine **28.7 GB/s** · separation **13.48×** (P2 measured 1.00× here, which is how it discovered the pages were VRAM)

### Correctness — **PASS**

* `gemm1_silu`: references differ = True; 1 misses → bit-exact=True; 5 misses → bit-exact=True; 9 misses → bit-exact=True
* `gemm2`: references differ = True; 1 misses → bit-exact=True; 5 misses → bit-exact=True; 9 misses → bit-exact=True

### Pointer-table overhead (A/B against the STOCK loader — old CODE, not emulated)

| arm | stock ms | table ms | overhead | bit-identical |
|---|---|---|---|---|
| `gemm1_silu` | 0.0415 | 0.0419 | 0.87% | True |
| `gemm2` | 0.0346 | 0.0349 | 0.75% | True |

### Curve — `gemm1_silu` (1.56 MiB/expert, non-scatter, bit-verified=True)

| misses | cold ms (median) | p10–p90 | hot ms | implied host GB/s |
|---|---|---|---|---|
| 0 | 0.0421 | 0.0411–0.0435 | 0.0397 | — |
| 1 | 0.0916 | 0.0812–0.0935 | 0.0984 | 17.8 |
| 2 | 0.1355 | 0.1263–0.1427 | 0.1404 | 24.1 |
| 3 | 0.1885 | 0.1831–0.1979 | 0.1964 | 26.0 |
| 4 | 0.2402 | 0.2394–0.2532 | 0.2533 | 27.2 |
| 5 | 0.2959 | 0.2955–0.3014 | 0.2935 | 27.6 |
| 6 | 0.3525 | 0.3520–0.3575 | 0.3542 | 27.8 |
| 7 | 0.4092 | 0.4088–0.4155 | 0.4119 | 28.0 |
| 8 | 0.4656 | 0.4650–0.4704 | 0.4678 | 28.1 |
| 9 | 0.5218 | 0.5203–0.5263 | 0.5237 | 28.2 |
| 10 | 0.5780 | 0.5776–0.5783 | 0.5804 | 28.3 |

**cliff_index 0.092** (pure-linear would be 0.100) → **LINEAR**. separation 13.72×. effective miss concurrency W = 0.72.

### Curve — `gemm2` (0.78 MiB/expert, scatter, bit-verified=True)

| misses | cold ms (median) | p10–p90 | hot ms | implied host GB/s |
|---|---|---|---|---|
| 0 | 0.0314 | 0.0309–0.0328 | 0.0331 | — |
| 1 | 0.0545 | 0.0485–0.0634 | 0.0520 | 15.0 |
| 2 | 0.0805 | 0.0683–0.0850 | 0.0792 | 20.3 |
| 3 | 0.1064 | 0.0988–0.1123 | 0.1000 | 23.0 |
| 4 | 0.1368 | 0.1267–0.1458 | 0.1280 | 23.9 |
| 5 | 0.1625 | 0.1609–0.1636 | 0.1562 | 25.1 |
| 6 | 0.1903 | 0.1866–0.1911 | 0.1839 | 25.8 |
| 7 | 0.2187 | 0.2102–0.2197 | 0.2125 | 26.2 |
| 8 | 0.2471 | 0.2453–0.2476 | 0.2412 | 26.5 |
| 9 | 0.2752 | 0.2668–0.2798 | 0.2691 | 26.7 |
| 10 | 0.3031 | 0.2945–0.3054 | 0.2971 | 27.0 |

**cliff_index 0.085** (pure-linear would be 0.100) → **LINEAR**. separation 9.65×. effective miss concurrency W = 0.77.

### Layer total (gemm1 + gemm2) — the decision table

| hit rate h | per-expert tier ms | layer-granular ms | per-expert gain | P(all resident) |
|---|---|---|---|---|
| 0.25 | 0.6702 | 0.6792 | 1.013x | 9.54e-07 |
| 0.50 | 0.4605 | 0.4773 | 1.036x | 9.77e-04 |
| 0.75 | 0.2590 | 0.2754 | 1.063x | 5.63e-02 |
| 0.90 | 0.1460 | 0.1543 | 1.057x | 3.49e-01 |
| 0.95 | 0.1097 | 0.1139 | 1.039x | 5.99e-01 |
| 0.99 | 0.0808 | 0.0816 | 1.010x | 9.04e-01 |

**LINEAR** — cliff_index 0.090 <= 0.25: misses cost additively, so a per-expert device tier converts byte hit rate into time almost 1:1 (best measured gain over layer-granular at equal byte budget: 1.063x).

## ROCm device 1 — AMD Radeon RX 9070 `0000:07:00.0`

*status:* **OK**

### Preconditions (placement is MEASURED, never declared)

| check | ok | detail |
|---|---|---|
| `device_stack_consumes_vram` | ✅ | classification=device_resident via amdgpu sysfs (mem_info_vram_used / mem_info_gtt_used); sysfs vram delta frac=1.163 |
| `host_stack_not_in_vram` | ✅ | sysfs vram delta frac=0.002 (must be < 0.25); THIS is the check P2 never made |
| `host_stack_is_anonymous_host_mapping` | ✅ | host maps_entry='7f0bf3400000-7f0c3e200000 rw-p 00000000 00:00 0 ' (must be anonymous private rw-p, NOT /dev/dri/*); the device stack for contrast maps '7f0c42400000-7f0c8d200000 rw-s 101d5a000 00:07 591                       /dev/dri/renderD129' |
| `host_stack_consumes_host_ram` | ✅ | MemAvailable delta frac=1.025 (corroborating target >= 0.5). ADVISORY ONLY on a shared box -- it does not gate. Host residency is established by host_stack_not_in_vram, host_stack_is_cpu_accessible, host_stack_is_anonymous_host_mapping and media_separation_ratio, all of which are card-local or direct. |
| `host_stack_is_cpu_accessible` | ✅ | cpu read at host ptr: {'readable': True, 'read_errno': None, 'read_strerror': None}; P1 found the fake-host VMM pages were NOT CPU-accessible (---s on renderD128, os.write -> EFAULT) |
| `stacks_do_not_alias` | ✅ | device VA [0x7f0c42400000,0x7f0c8d200000) vs host device-visible VA [0x7f0bf3400000,0x7f0c3e200000); overlap=0 B (must be 0). host_ptr=0x7f0bf3400000 device_ptr=0x7f0bf3400000 (equal is NORMAL under ROCm's unified VA -- see P1); the two stacks are separate ALLOCATIONS on different media and cannot be interleaved, which is what makes this a two-stack design |
| `read_checksums_valid` | ✅ | device ok=True host ok=True; a wrong checksum means the read did not land on the pages we think it did |
| `media_separation_ratio` | ✅ | device 402.5 / host 14.5 = 27.82x (need >= 4.0x). P2 measured 1.00x here and that is what proved the 'host' pages were VRAM |
| `device_read_is_hbm_speed` | ✅ | 402.5473238386492 GB/s (need >= 100.0) |
| `host_read_is_pcie_bounded` | ✅ | host 14.471088755023867 GB/s vs copy engine 14.343287901783107 GB/s x 1.25; bytes that cross PCIe cannot materially outrun the copy engine on the same card |

device read **402.5 GB/s** · host read **14.5 GB/s** · copy engine **14.3 GB/s** · separation **27.82×** (P2 measured 1.00× here, which is how it discovered the pages were VRAM)

### Correctness — **PASS**

* `gemm1_silu`: references differ = True; 1 misses → bit-exact=True; 5 misses → bit-exact=True; 9 misses → bit-exact=True
* `gemm2`: references differ = True; 1 misses → bit-exact=True; 5 misses → bit-exact=True; 9 misses → bit-exact=True

### Pointer-table overhead (A/B against the STOCK loader — old CODE, not emulated)

| arm | stock ms | table ms | overhead | bit-identical |
|---|---|---|---|---|
| `gemm1_silu` | 0.0430 | 0.0420 | -2.37% | True |
| `gemm2` | 0.0340 | 0.0339 | -0.47% | True |

### Curve — `gemm1_silu` (1.56 MiB/expert, non-scatter, bit-verified=True)

| misses | cold ms (median) | p10–p90 | hot ms | implied host GB/s |
|---|---|---|---|---|
| 0 | 0.0455 | 0.0444–0.0468 | 0.0402 | — |
| 1 | 0.1494 | 0.1398–0.1597 | 0.1526 | 10.9 |
| 2 | 0.2462 | 0.2400–0.2567 | 0.2496 | 13.3 |
| 3 | 0.3545 | 0.3528–0.3670 | 0.3638 | 13.8 |
| 4 | 0.4656 | 0.4653–0.4785 | 0.4677 | 14.0 |
| 5 | 0.5782 | 0.5782–0.5795 | 0.5809 | 14.1 |
| 6 | 0.6912 | 0.6910–0.6933 | 0.6936 | 14.2 |
| 7 | 0.8042 | 0.8038–0.8047 | 0.8062 | 14.2 |
| 8 | 0.9169 | 0.9167–0.9176 | 0.9196 | 14.3 |
| 9 | 1.0298 | 1.0296–1.0304 | 1.0322 | 14.3 |
| 10 | 1.1425 | 1.1424–1.1427 | 1.1450 | 14.3 |

**cliff_index 0.095** (pure-linear would be 0.100) → **LINEAR**. separation 25.12×. effective miss concurrency W = 0.82.

### Curve — `gemm2` (0.78 MiB/expert, scatter, bit-verified=True)

| misses | cold ms (median) | p10–p90 | hot ms | implied host GB/s |
|---|---|---|---|---|
| 0 | 0.0350 | 0.0346–0.0353 | 0.0330 | — |
| 1 | 0.0854 | 0.0776–0.0914 | 0.0788 | 9.6 |
| 2 | 0.1340 | 0.1252–0.1405 | 0.1340 | 12.2 |
| 3 | 0.1905 | 0.1830–0.1946 | 0.1816 | 12.9 |
| 4 | 0.2469 | 0.2382–0.2503 | 0.2374 | 13.2 |
| 5 | 0.3032 | 0.3015–0.3056 | 0.2969 | 13.5 |
| 6 | 0.3593 | 0.3582–0.3600 | 0.3532 | 13.6 |
| 7 | 0.4155 | 0.4072–0.4164 | 0.4095 | 13.8 |
| 8 | 0.4726 | 0.4719–0.4733 | 0.4632 | 13.8 |
| 9 | 0.5282 | 0.5200–0.5315 | 0.5218 | 13.9 |
| 10 | 0.5848 | 0.5792–0.5876 | 0.5784 | 14.0 |

**cliff_index 0.092** (pure-linear would be 0.100) → **LINEAR**. separation 16.70×. effective miss concurrency W = 0.81.

### Layer total (gemm1 + gemm2) — the decision table

| hit rate h | per-expert tier ms | layer-granular ms | per-expert gain | P(all resident) |
|---|---|---|---|---|
| 0.25 | 1.3045 | 1.3156 | 1.009x | 9.54e-07 |
| 0.50 | 0.8823 | 0.9039 | 1.025x | 9.77e-04 |
| 0.75 | 0.4690 | 0.4922 | 1.050x | 5.63e-02 |
| 0.90 | 0.2334 | 0.2452 | 1.050x | 3.49e-01 |
| 0.95 | 0.1570 | 0.1628 | 1.037x | 5.99e-01 |
| 0.99 | 0.0959 | 0.0970 | 1.011x | 9.04e-01 |

**LINEAR** — cliff_index 0.094 <= 0.25: misses cost additively, so a per-expert device tier converts byte hit rate into time almost 1:1 (best measured gain over layer-granular at equal byte budget: 1.050x).

## Provenance

Stack: 7.2.4 · amdgpu None · kernel 7.0.10-1-cachyos-custom · hipcc HIP version: 7.2.53211-3d9ef42

GPU lease **waived by explicit user instruction** for this workstream; cards were used serially and box state is recorded around each. `ROCR_VISIBLE_DEVICES=0,1` with `HIP_VISIBLE_DEVICES` unset throughout, so the Ryzen iGPU never entered enumeration.

