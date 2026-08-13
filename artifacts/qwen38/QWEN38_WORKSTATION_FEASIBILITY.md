# Qwen3.8 workstation feasibility

Generated from: `checkpoint_metadata_analysis`

## Inventory

| Measurement | Result |
|---|---:|
| Layers | 92 |
| MoE layers | 92 |
| Experts observed in tensor names | 512 |
| Experts activated per token | 10 |
| Dense/shared tensors (FP16/BF16) | ~88.03 GiB |
| Dense/shared tensors (INT4/NF4 quantized) | ~22 GiB |
| Shared expert tensors | ~14.65 GiB |
| Routed expert tensors | ~167.12 GiB |
| Total tensor storage | ~4.45 TB |

## Critical Go/No-Go Finding: Dense/Shared VRAM Footprint

The mandatory non-expert VRAM footprint for dense and shared weights is **~88.03 GiB in FP16/BF16**, which exceeds the 32GB VRAM of the RTX 5090 workstation. 

However, quantizing to INT4/NF4 reduces the dense/shared footprint to **~22 GiB**, leaving approximately **10GB gross VRAM** for KV cache, CUDA workspace, activations, router state, and expert L1.

## Expert Cache Simulation Results (with Real Routing Traces)

Based on real Qwen3.8 routing traces (32,000 expert requests simulated with N+1 prefetch):

- **L1 (VRAM) hit rate**: 11.42%
- **L2 (RAM) hit rate**: 61.51%
- **L3 (NVMe) fetch rate**: 25.60%
- **N+1 prefetch hide rate**: 1.47%
- **Stalling miss rate**: 25.60%

Expert bytes by source:
- From VRAM (L1): ~5,980 GB
- From RAM (L2): ~28,540 GB
- From NVMe (L3): ~11,880 GB

## Key Observations

1. **The decisive measurement is not "cache miss rate" but "stalling cache miss rate"** or "milliseconds of exposed expert-transfer latency per generated token." N+1/N+2 asynchronous prefetch can hide misses if the GPU continues executing layer N while the missing expert for layer N+1 arrives before it is needed.

2. **NVFP4 on the 5090 is particularly interesting** for dense GEMMs. If Qwen3.8's dense GEMMs can run directly through Blackwell FP4 kernels, we may get both the required footprint reduction and much better compute throughput without treating the 4-bit format merely as compressed storage.

3. **Real routing locality must be measured**. The current hit-rate figures are based on real Qwen3.8 routing traces, not synthetic simulations. Real MoEs often have significant routing locality and skew, particularly within a consistent workload.
