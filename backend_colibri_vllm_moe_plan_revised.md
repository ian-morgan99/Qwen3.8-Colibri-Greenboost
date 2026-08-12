# Backend Plan: Colibri/vLLM-MoE alongside Greenboost/TurboQuant, MOE/Expert Tensor Fit (Revised)

## 1. Executive Summary

This document outlines a detailed plan for implementing the backend inference runtime for the Qwen3.8-2.4T-A95B model (~95B active parameters, 512 experts) on a workstation-class system (RTX 5090 32GB VRAM, 96GB DDR5 RAM, NVMe storage). The plan integrates:

- **vLLM-Moet**: Base vLLM MoE inference engine as the execution foundation.
- **Colibrì-derived work**: Custom expert routing, heatmap tracking, lookahead, expert packing, and learned residency.
- **Greenboost/TurboQuant/SpectralQuant**: Pinned memory/transfer/allocation/telemetry tuning and optional compact representations after baseline correctness.
- **MOE/Expert Tensor Fit**: Strategies for fitting active experts across VRAM (L1), RAM (L2), and NVMe (L3) via an `ExpertWeightProvider` abstraction.

GGUF/llama.cpp is treated as a parallel investigation or optional ingestion/export path, not the primary runtime architecture.

## 2. Architecture Overview

### 2.1 Hardware Constraints
- **VRAM (L1)**: 32GB (RTX 5090) - Hot/current experts and permanently resident non-expert tensors
- **RAM (L2)**: 96GB DDR5 - Warm experts (RAM arena)
- **NVMe (L3)**: ~817GB available - Cold experts (pack store)

### 2.2 Model Characteristics (Qwen3.8-2.4T-A95B)
- Total parameters: ~2.4T
- Active parameters per token: ~95B
- Experts: 512 total, 10 routed + 1 shared activated per token
- Layers: 92
- Hidden dimension: 8192

### 2.3 Memory Budget Analysis (Phase 1A: Metadata Inventory Complete; Physical Tensor Accounting Pending)
- **Original FP16/BF16 Weights**: Dense/shared weights require ~293 GiB (exceeds 32 GiB VRAM). After dense weights, 0 GPU experts fit in VRAM. 54 experts can fit in the 96 GiB RAM. Cold active-expert NVMe traffic per token estimated at 1.036 seconds (useful as a warning, not a forecast).
- **Quantization Considerations**: The dense/shared portion quantization estimates must be validated with actual tensor inventory. Quant formats are usually tensor-type dependent, and routed experts, attention matrices, embeddings, router weights and shared experts may not all receive the same effective bits/weight. 
- **Dense Stack Residency Solution**: If the dense stack truly remains over 32GB, the solution is likely selective CPU/GPU residency, a better native low-bit representation, or deliberately keeping only part of the dense stack GPU-resident—not "further quantization to Q2/Q3" (which would mean *more* bits, not less).

## 3. Component Integration Plan

### 3.1 vLLM-Moet Execution Foundation
1. **ExpertWeightProvider Abstraction**: Expert weights should sit behind an `ExpertWeightProvider`/expert-residency abstraction. Do not extend PagedAttention for expert tiering, as PagedAttention is fundamentally the KV-cache side of vLLM. Mixing expert storage management into PagedAttention couples two unrelated memory lifecycles and will make long-context tuning much harder.
2. **Exact Routing Baseline**: The router should choose the model-correct experts; the cache manager should decide where those experts live. Memory-aware expert substitution should be a later experimental mode because it changes model semantics.
   Baseline: `router → exact expert IDs → ExpertWeightProvider → VRAM/RAM/NVMe`
3. **Persistent Pack Representation**: Ingest an upstream checkpoint, transform routed experts into a compact runtime representation, persist those expert packs, and execute them with the SM120 kernels.

### 3.2 Colibrì-Derived Work
1. **Heatmap/Lookahead**: Maintain real-time statistics on expert activation frequency and implement router lookahead.
2. **Expert Packing/Learned Residency**: Group frequently co-activated experts to minimize NVMe I/O and implement learned residency policies.

### 3.3 GreenBoost/TurboQuant/SpectralQuant Integration
1. **GreenBoost**: Pinned memory / transfer / allocation / telemetry tuning.
2. **TurboQuant / SpectralQuant**: Optional compact representations after baseline correctness.

## 4. Expert Tensor Fit Strategy (Architecture)

```
                 Qwen3.8 router
                       │
                 exact expert IDs
                       │
                       ▼
             ExpertWeightProvider
              /        |        \
             /         |         \
       GPU L1      RAM L2      NVMe L3
       hot set     arena       pack store
             \         |         /
              \        |        /
                 SM120 MoE kernel
```

## 5. Critical Telemetry Requirements (Mandatory)

1. **L1 hit rate**
2. **L2 hit rate**
3. **L3 fetch rate**
4. **zero-miss-step percentage** (especially important: a cache with 95% expert hits can still perform badly if nearly every generation step contains at least one miss)
5. **replay percentage**
6. **unique expert misses/step**
7. **bytes NVMe→RAM/token**
8. **bytes RAM→GPU/token**
9. **useful-prefetch ratio**
10. **wasted-prefetch bytes**
11. **cache churn**
12. **expert-set Jaccard similarity between adjacent tokens/layers**
13. **p50/p95 expert-fetch latency**
14. **GPU idle time waiting for experts**

## 6. Proposed Implementation Order (Phases 0-9)

### Phase 0 — Exact Physical Inventory
Prove the tensor numbers:
- Parse every Qwen3.8 tensor from the released checkpoint;
- Classify each tensor as dense/shared/router/routed-expert/KV-related;
- Record stored dtype and byte size;
- Calculate expert size per layer;
- Calculate the exact permanently resident non-expert footprint;
- Calculate the active expert bytes per layer/token;
- Generate 8/12/16GB GPU-cache simulations;
- Generate 48/64/72GB RAM-arena simulations;
- Run routing traces on representative coding prompts if the model can be executed anywhere;
- Estimate zero-miss-step probability, not just average expert hit rate.

### Phase 1 — Reuse the Existing vLLM-Moet Expert-Provider/Cache Design
Adapt model loader and expert indexing for Qwen3.8. No GGUF execution layer.

### Phase 2 — Exact L1 GPU + L2 RAM Execution
Prove bit/quality equivalence against an untiered reference for a small subset or representative layers.

### Phase 3 — L3 Persistent NVMe Expert Packs
Use contiguous expert records, metadata/version checks and persistent quantization cache.

### Phase 4 — CUDA-Graph-Safe Miss Recovery and Batched Fetch
Reuse the proven vLLM-Moet replay architecture where possible.

### Phase 5 — Router Lookahead and Asynchronous Prefetch

### Phase 6 — Workload Heat/Persistent Pinning

### Phase 7 — GreenBoost Transfer and Memory Optimisation

### Phase 8 — Alternate Quant Formats / TurboQuant / SpectralQuant

### Phase 9 — Experimental Cache-Aware Routing
Off by default.

## 7. GGUF/llama.cpp Note
GGUF/llama.cpp should be a **parallel investigation**, not Phase 2 of the primary runtime. It remains an optional ingestion/export path, not runtime architecture. Do not splice llama.cpp/GGUF into the hot path unless later evidence shows vLLM cannot support the released weights efficiently.