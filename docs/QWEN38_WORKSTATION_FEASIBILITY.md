# Qwen3.8 Workstation Feasibility Report

## Executive Summary

This document assesses the feasibility of running the ~495GB Qwen3.8 MoE checkpoint (~95B active parameters) on a workstation-class system equipped with an RTX 5090 (32GB VRAM), 96GB DDR5 RAM, and NVMe storage. The assessment is based on Phase 0 exact physical inventory and expert cache simulations.

## Critical Go/No-Go Finding: Dense/Shared VRAM Footprint

Phase 0 inventory revealed a critical constraint: the mandatory non-expert VRAM footprint for dense and shared weights is approximately **88.03 GiB in FP16/BF16 precision**. This exceeds the **32GB VRAM** capacity of the RTX 5090 workstation by a significant margin.

### Implications

- Dense and shared weights are typically fully resident on the GPU during inference.
- The ~88.03 GiB footprint cannot coexist with KV cache, CUDA workspace, and an expert L1 cache on a 32GB GPU.
- Expert tiering (managed by GreenBoost or vLLM-Moet's `ExpertWeightProvider`) only manages *routed expert weights*, not mandatory dense/shared weights.

## Viable Paths Forward

### Path 1: Lower-Bit Dense Quantization (INT4/NF4)

Quantizing dense/shared weights to INT4 or NF4 format reduces the footprint from ~88 GiB to approximately **~22 GiB**. This leaves sufficient VRAM budget for:
- KV cache (at reduced context lengths)
- CUDA workspace
- Expert L1 cache (hot expert set)

**Recommendation:** Proceed with INT4/NF4 quantization strategy as the primary path forward.

### Path 2: Context Length Reduction

Reducing maximum context length reduces KV cache VRAM consumption, freeing space for dense weights or expert cache. However, this does not solve the fundamental ~88 GiB dense weight footprint issue without quantization.

### Path 3: Dense-Layer Offload to CPU RAM

Offloading dense/shared weights to CPU RAM is theoretically possible but represents a severe performance bottleneck due to PCIe bandwidth limitations. This should be considered only if INT4/NF4 quantization is not viable.

## Expert Cache Simulation Results

GPU L1 Cache (VRAM) Simulations:
- 8GB: fits 5 experts, hit_rate=0.013, zero_miss_prob=0.0
- 12GB: fits 8 experts, hit_rate=0.0208, zero_miss_prob=0.0
- 16GB: fits 11 experts, hit_rate=0.0286, zero_miss_prob=0.0
- 20GB: fits 13 experts, hit_rate=0.0339, zero_miss_prob=0.0
- 24GB: fits 16 experts, hit_rate=0.0417, zero_miss_prob=0.0

RAM L2 Arena Simulations:
- 48GB: fits 33 experts, hit_rate=0.0027, zero_miss_prob=0.0021
- 64GB: fits 44 experts, hit_rate=0.0036, zero_miss_prob=0.0029
- 72GB: fits 50 experts, hit_rate=0.0041, zero_miss_prob=0.0033
- 96GB: fits 67 experts, hit_rate=0.0055, zero_miss_prob=0.0044
- 128GB: fits 89 experts, hit_rate=0.0072, zero_miss_prob=0.0058

## Key Observations

1. **Zero-Miss-Step Probability is Critically Low:** Even with generous GPU L1 cache sizes (24GB), the zero-miss-step probability remains at 0.0. This indicates that nearly every generation step will contain at least one expert miss, driving decode performance bottlenecks.

2. **Dense Quantization is Mandatory:** Without INT4/NF4 quantization of dense/shared weights, the architecture cannot proceed on a 32GB VRAM workstation.

3. **Expert Cache Hit Rates are Low:** The simulated hit rates and zero-miss probabilities suggest that expert locality and prefetching strategies will be critical to achieving acceptable performance.

## Next Steps: Phase 0.5 Go/No-Go Gate

Before proceeding to Phase 1 (vLLM-Moet expert-provider/cache design adaptation), the following must be verified:

1. **Dense Weight Quantization Compatibility:** Confirm that vLLM/vLLM-Moet can successfully load and execute Qwen3.8 with INT4/NF4 quantized dense/shared weights.
2. **SM120 Kernel Compatibility:** Verify that Qwen3.8 expert tensor shapes, activation functions, and MoE layout align with existing SM120 cubins proven for DeepSeek/GLM models.
3. **Context Length Constraints:** Determine maximum viable context length given the ~22 GiB dense footprint + KV cache + CUDA workspace + expert L1 cache budget.

## Conclusion

The Qwen3.8 workload is **feasible on a 32GB VRAM workstation only if dense/shared weights are quantized to INT4/NF4 format**. Without this quantization, the ~88 GiB dense footprint makes execution impossible on the target hardware. The expert tiering architecture (vLLM-Moet + GreenBoost) remains valid and necessary for managing routed expert weights, but dense quantization is the primary go/no-go gate for project viability.
