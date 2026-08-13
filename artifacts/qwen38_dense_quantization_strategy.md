# Qwen3.8 Dense Weight Quantization Strategy: INT4/NF4 Path

## Critical Go/No-Go Finding Recap
Phase 0 physical inventory revealed that the mandatory non-expert VRAM footprint is **~88.03 GiB in FP16/BF16**, which exceeds the **32GB VRAM** of the RTX 5090 workstation. This means the dense/shared portion alone cannot fit alongside KV cache, CUDA workspace, and an expert L1 cache on a 32GB GPU.

**Conclusion**: Expert tiering (vLLM-Moet + GreenBoost + Colibrì concepts) manages *routed expert weights*, but does not solve the physical constraint of the ~88 GiB dense footprint. The architecture **requires dense-layer offload or lower-bit dense kernels** before expert tiering can proceed.

---

## Viable Path: Lower-Bit Dense Quantization (INT4 / NF4)

### Memory Footprint Reduction
| Format | Bytes/Weight | Dense Footprint (88 GiB FP16) |
|--------|--------------|-------------------------------|
| FP16/BF16 | 2.0 bytes | ~88.03 GiB |
| INT8/FP8 | 1.0 byte | ~44.02 GiB (still too large) |
| **INT4/NF4** | **0.5 bytes** | **~22.01 GiB** |

### VRAM Budget with INT4/NF4 Dense Weights
With dense weights quantized to INT4/NF4 (~22 GiB), the 32GB VRAM budget becomes feasible:

| Component | Estimated VRAM Usage |
|-----------|---------------------|
| **Dense/Shared Weights (INT4/NF4)** | ~22.0 GiB |
| **KV Cache (32K context, 4 KV heads, FP16)** | ~5.0–6.0 GiB |
| **CUDA Workspace & Runtime Overhead** | ~1.0–2.0 GiB |
| **Expert L1 Cache (Hot Set)** | ~2.0–3.0 GiB |
| **Total Estimated VRAM Usage** | **~30.0–33.0 GiB** |

*Note: If the total exceeds 32GB at 32K context, reduce target context length to 8K–16K during initial phases to free up 2–4 GiB for the expert L1 cache.*

---

## Implementation Prerequisites for Phase 1

Before proceeding with Phase 1 (vLLM-Moet expert-provider/cache design), the following must be confirmed:

### 1. Dense Quantization Support in Loader/Runtime
- Confirm that the vLLM-Moet model loader supports loading INT4/NF4 quantized dense/shared weights.
- Validate that the Qwen3.8 dense tensor shapes (embed_tokens, lm_head, linear_attn, mlp.gate, layer norms, shared_expert) can be quantized using AWQ, GPTQ, or NF4 techniques without degrading model quality beyond acceptable thresholds.

### 2. SM120 Kernel Compatibility for Lower-Bit Dense Layers
- Validate that the SM120 MoE kernels (or equivalent vLLM-Moet kernels) can execute lower-bit dense layers (INT4/NF4) natively.
- If existing kernels do not support INT4/NF4 dense execution, a kernel extension or integration with a quantized dense kernel library (e.g., AWQ, GPTQ, or NF4 kernels) is required before cache work matters.

### 3. Quality Equivalence Regression Thresholds
- The quantized representation must separately pass **quality equivalence/regression thresholds** against the original FP16/BF16 model.
- Tiered execution must be **bit-identical to the same quantized representation running fully resident**.

---

## Alternative Path: Dense-Layer Offload to CPU RAM

If INT4/NF4 quantization is not viable or accurate enough, the alternative is dense-layer offload:

### Approach
- Partially or fully offload the dense/shared weights to the 96GB DDR5 RAM.
- Fetch dense weights across PCIe as needed during the forward pass.

### Performance Impact
- **Severe performance bottleneck**: Dense layers are accessed for *every single token* during the forward pass, unlike routed experts which are sparsely activated (10 out of 512 per token).
- Decode speeds would be extremely slow (likely only a few tokens per second or worse, depending on PCIe Gen4/Gen5 x16 bandwidth).
- This should only be considered a **fallback** if lower-bit dense kernels are not viable.

---

## Next Steps: Phase 0.5 — Dense Quantization Go/No-Go Gate

Before touching runtime code for expert tiering, Phase 0.5 must confirm:

1. **Dense quantization format selection**: Confirm INT4 or NF4 as the target quantization format for dense/shared weights.
2. **Loader/runtime support**: Validate that the vLLM-Moet loader can ingest INT4/NF4 quantized dense weights.
3. **Kernel compatibility**: Confirm SM120 or equivalent kernels support lower-bit dense layer execution.
4. **VRAM budget finalization**: Calculate exact VRAM footprint using INT4/NF4 dense weights + KV cache at target context length (8K/16K/32K) + CUDA workspace + expert L1 cache.
5. **Quality equivalence validation**: Establish regression thresholds for INT4/NF4 quantized model vs. original FP16/BF16 reference.

**If INT4/NF4 quantization is confirmed viable**: The project is absolutely still workable, and Phase 1 can proceed with expert tiering on the quantized model.

**If INT4/NF4 quantization is not viable**: Dense-layer offload must be implemented as a fallback, with acknowledged severe performance implications.
