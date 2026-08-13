# ⚠️ SUPERSEDED — DO NOT IMPLEMENT

**This document has been superseded by [`backend_colibri_vllm_moe_detailed_plan.md`](./backend_colibri_vllm_moe_detailed_plan.md).**

The architecture described in this document conflates vLLM, GGUF/llama.cpp, and Colibri-style tiering in ways that are no longer recommended. Specifically:
- Do not extend PagedAttention for expert tiering
- Do not make routing memory-aware in the first implementation
- Do not make GGUF support a prerequisite for the vLLM path
- Do not treat Q2_K/Q3_K as further quantization

**Please refer to `backend_colibri_vllm_moe_detailed_plan.md` for the current, approved architecture.**

---

# Backend Plan: Colibri/vLLM-MoE alongside Greenboost/TurboQuant, MOE/Expert Tensor Fit

## 1. Executive Summary

This document outlines a detailed plan for implementing the backend inference runtime for the Qwen3.8-2.4T-A95B model (~95B active parameters, 512 experts) on a workstation-class system (RTX 5090 32GB VRAM, 96GB DDR5 RAM, NVMe storage). The plan integrates:

- **Colibri**: Custom expert routing and tensor fit management
- **vLLM-MoE**: Base vLLM MoE inference engine with modifications for tiered memory
- **Greenboost/TurboQuant**: Quantization and acceleration techniques for expert weights
- **MOE/Expert Tensor Fit**: Strategies for fitting active experts across VRAM (L1), RAM (L2), and NVMe (L3)

## 2. Architecture Overview

### 2.1 Hardware Constraints
- **VRAM (L1)**: 32GB (RTX 5090) - Hot/current experts and dense/shared weights
- **RAM (L2)**: 96GB DDR5 - Warm experts
- **NVMe (L3)**: ~817GB available - Cold experts and base model weights

### 2.2 Model Characteristics (Qwen3.8-2.4T-A95B)
- Total parameters: ~2.4T
- Active parameters per token: ~95B
- Experts: 512 total, 10 routed + 1 shared activated per token
- Layers: 92
- Hidden dimension: 8192

### 2.3 Memory Budget Analysis (Original vs GGUF Q1_0 Quantized)
**Original FP16/BF16 Weights:**
- Dense/shared weights require ~293 GiB (exceeds 32 GiB VRAM)
- After dense weights, 0 GPU experts fit in VRAM
- 54 experts can fit in the 96 GiB RAM
- Cold active-expert NVMe traffic per token estimated at 1.036 seconds

**GGUF Q1_0 Quantized Weights (unsloth/Qwen3.8-2.4T-A95B-GGUF UD-Q1_0):**
- Total GGUF size: ~397 GB for 10 file parts
- Quantization reduces memory footprint by approximately 8x compared to FP16
- Estimated dense/shared weights after Q1_0 quantization: ~36-40 GiB
- This still exceeds 32GB VRAM but is much closer to feasible with optimization

## 3. Component Integration Plan

### 3.1 vLLM-MoE Base Engine Modifications
1. **Tiered Memory Manager**: Extend vLLM's existing PagedAttention with tiered storage support
2. **Expert Router**: Modify MoE routing to consider memory tiers when selecting experts
3. **Asynchronous Prefetching**: Implement background loading of warm/cold experts from RAM/NVMe

### 3.2 Colibri Expert Routing and Tensor Fit
1. **Expert Heatmap Tracking**: Maintain real-time statistics on expert activation frequency
2. **Dynamic Tier Assignment**: Move experts between VRAM/RAM/NVMe based on usage patterns
3. **Tensor Fit Optimization**: Group frequently co-activated experts to minimize NVMe I/O

### 3.3 Greenboost/TurboQuant Integration (GGUF Q1_0 Approach)
1. **Quantization Strategy**: Apply Q1_0 quantization as seen in the unsloth GGUF release
2. **GGUF Format Compatibility**: Leverage existing GGUF parsing capabilities instead of custom dequantization
3. **Mixed Precision Support**: Keep critical tensors in higher precision if needed, quantize less critical ones
4. **GGML/GGUF Execution Engine**: Consider using llama.cpp's GGML engine as a foundation for GGUF execution

## 4. Expert Tensor Fit Strategy

### 4.1 VRAM (L1) - Hot Experts
- **Capacity**: ~32GB total, but dense/shared weights consume most of it
- **Strategy with Q1_0 Quantization**: 
  - Dense/shared weights after Q1_0: ~36-40 GiB (still exceeds VRAM)
  - Need additional optimization: partial loading or further quantization to Q2_K or Q3_K
  - Only store the currently activated experts (up to 10 per token) in VRAM
- **Implementation**: Lazy loading of experts into VRAM as they are routed

### 4.2 RAM (L2) - Warm Experts
- **Capacity**: 96GB DDR5
- **Strategy**: Store frequently activated experts that don't fit in VRAM
- **Implementation**: Pre-fetch experts based on access patterns and heatmap data

### 4.3 NVMe (L3) - Cold Experts
- **Capacity**: ~817GB available
- **Strategy**: Store infrequently activated experts and base model weights
- **Implementation**: Asynchronous loading with overlap computation and I/O

## 5. Critical Telemetry Requirements

1. **Expert Activation Frequency**: Track which experts are activated most often
2. **Memory Tier Utilization**: Monitor VRAM/RAM/NVMe usage patterns
3. **I/O Latency**: Measure NVMe read times for cold expert loading
4. **Routing Efficiency**: Evaluate how well the router selects experts that fit in higher tiers

## 6. Implementation Phases

### Phase 1: Architecture Inventory (Completed)
- Parse checkpoint and produce feasibility reports and JSON plans
- Generate simulated artifacts based on Hugging Face model metadata

### Phase 2: GGUF/Q1_0 Compatibility Layer
- Implement GGUF parsing and loading capabilities
- Integrate with llama.cpp's GGML execution engine or similar
- Validate Q1_0 quantization quality and performance

### Phase 3: Exact Three-Tier Runtime with Colibri/vLLM-MoE
- Implement tiered memory manager in vLLM-MoE or custom engine
- Integrate Colibri expert routing with memory awareness
- Add dynamic expert loading based on GGUF tensor fit

### Phase 4: Asynchronous Pipeline
- Implement background prefetching for warm/cold experts
- Optimize I/O overlap with computation

### Phase 5: Cache Policy
- Implement dynamic expert tier assignment based on usage patterns
- Refine routing based on telemetry data

## 7. Critical Evaluation and Risks

### 7.1 Memory Budget Challenges with Q1_0 Quantization
- **Issue**: Even with Q1_0 quantization, dense/shared weights (~36-40 GiB) still exceed VRAM capacity (32 GiB)
- **Mitigation Options**:
  1. Use further quantization (Q2_K, Q3_K) for non-critical tensors
  2. Implement partial loading of dense weights (load only what's needed for current token)
  3. Optimize memory layout to reduce VRAM fragmentation
- **Risk**: Aggressive quantization may degrade model quality and routing accuracy

### 7.2 GGUF Format Integration Challenges
- **Issue**: vLLM is optimized for safetensors/PyTorch formats, not GGUF
- **Mitigation**: 
  1. Use llama.cpp's GGML engine as the execution backend
  2. Implement a compatibility layer to translate vLLM's MoE routing to GGML's expert execution
  3. Consider building a custom inference engine based on GGUF/GGML rather than modifying vLLM
- **Risk**: Significant engineering effort to integrate GGUF with MoE routing

### 7.3 NVMe I/O Bottleneck
- **Issue**: Cold active-expert NVMe traffic per token estimated at 1.036 seconds
- **Mitigation**: Asynchronous prefetching and expert grouping to minimize random I/O
- **Risk**: Even with prefetching, NVMe latency may limit throughput

### 7.4 Expert Routing Complexity
- **Issue**: Dynamic tier assignment adds complexity to routing logic
- **Mitigation**: Simplify routing decisions based on heatmap thresholds
- **Risk**: Suboptimal routing may lead to increased NVMe access

## 8. Refinement and Next Steps

1. **GGUF/Q1_0 Validation**: Test the downloaded GGUF Q1_0 weights with llama.cpp or similar GGML engine
2. **Quantization Impact Assessment**: Evaluate quality degradation at Q1_0 vs Q2_K/Q3_K for dense weights
3. **Engine Selection Decision**: Choose between:
   - Modifying vLLM-MoE to support GGUF loading
   - Building a custom engine based on llama.cpp's GGML with MoE support
   - Using existing MoE GGUF implementations if available
4. **Implement Prototype Tiered Manager**: Create a minimal expert loading system with tiered storage
5. **Develop Telemetry Infrastructure**: Set up monitoring for expert activation and memory usage
6. **Benchmark NVMe I/O**: Measure actual NVMe read times for expert weight loading

## 9. Conclusion

The backend plan must be significantly revised to account for the GGUF Q1_0 quantized weights from unsloth. While Q1_0 quantization reduces the ~293 GiB dense/shared weights to ~36-40 GiB, this still exceeds the 32GB VRAM capacity. The integration of GGUF format with vLLM-MoE presents significant challenges, as vLLM is optimized for safetensors/PyTorch formats rather than GGUF. 

The most viable path forward appears to be:
1. Using llama.cpp's GGML engine as the execution backend for GGUF weights
2. Implementing a custom Colibri expert routing layer that works with GGML's MoE execution
3. Considering further quantization (Q2_K/Q3_K) for dense weights to fit within VRAM
4. Implementing tiered memory management for expert weights across VRAM/RAM/NVMe

Critical evaluation identifies key risks that must be addressed through prototyping and benchmarking, particularly around GGUF integration and memory budget constraints.