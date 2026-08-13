# Phase 0.5: Runtime/Loader Support for INT4/NF4 Quantization and SM120 Kernel Compatibility

## 1. vLLM/vLLM-Moet Support for INT4/NF4 Dense Quantization

vLLM natively supports several 4-bit quantization formats for dense/shared weights:

### Supported Quantization Formats for Dense Weights:
1. **AWQ (Activation-Aware Weight Quantization)**
   - Format: 4-bit AWQ
   - vLLM flag: `--quantization awq`
   - Compatibility: Widely available for Qwen models (e.g., `Qwen/Qwen2.5-72B-Instruct-AWQ`)

2. **GPTQ (Generative Pre-trained Transformer Quantization)**
   - Format: 4-bit GPTQ
   - vLLM flag: `--quantization gptq`
   - Compatibility: Widely available for Qwen models (e.g., `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`)

3. **Marlin 4-bit**
   - Format: 4-bit Marlin
   - vLLM flag: `--quantization marlin`
   - Compatibility: Requires specific weight layouts; highly optimized for SM80+ and SM90 (Ada/Lovelace/Blackwell) GPUs

4. **Bitsandbytes NF4/INT4**
   - Format: NF4 (Normal Float 4-bit) or INT4
   - vLLM flag: `--quantization bitsandbytes` with `--load-format bitsandbytes` and `--dtype auto`
   - Compatibility: Supported via `bitsandbytes` library; commonly used in QLoRA/QLoRA-style quantization

### Recommendation for Qwen3.8:
- **Primary Target**: AWQ or GPTQ-Int4 for dense weights, as these are natively supported by vLLM with optimized kernels (Marlin or AWQ/GPTQ kernels).
- **Alternative Target**: NF4 via `bitsandbytes` if AWQ/GPTQ versions are not available for Qwen3.8.

## 2. SM120 Kernel Compatibility for Expert Weights

The vLLM-Moet project uses **SM120 MoE kernels** for expert pack execution. These kernels are designed for:
- MoE routing and expert selection
- Expert weight packing and unpacking
- Sparse expert matrix multiplication

**Important Distinction**: 
- SM120 kernels execute **routed expert weights** (the MoE layers).
- **Dense/shared weights** (embeddings, attention matrices, MLP gate/proj, layer norms) are executed by vLLM's standard dense kernels, which support AWQ/GPTQ/Marlin/FP8/bitsandbytes quantization.

Therefore, SM120 kernel compatibility is **not affected** by the dense weight quantization format (INT4/NF4 vs FP16). The SM120 kernels operate on the routed expert weights, which will be quantized separately (e.g., Q1_0 or 2-bit expert packs) and managed by the `ExpertWeightProvider`.

## 3. Phase 0.5 Go/No-Go Gate Checklist

Before proceeding to Phase 1 (vLLM-Moet expert-provider/cache design), confirm:

- [ ] **Dense quantization format selection**: Confirm AWQ, GPTQ-Int4, or NF4 (bitsandbytes) as the target quantization format for dense/shared weights.
- [ ] **Loader/runtime support**: Validate that vLLM/vLLM-Moet loader can ingest the selected quantized dense weights (AWQ/GPTQ/NF4).
- [ ] **Kernel compatibility**: Confirm vLLM's dense kernels (AWQ/GPTQ/Marlin/FP8/bitsandbytes) support the selected quantization format on SM90 (RTX 5090) hardware.
- [ ] **VRAM budget finalization**: Calculate exact VRAM footprint using quantized dense weights + KV cache at target context length (8K/16K/32K) + CUDA workspace + expert L1 cache.
- [ ] **Quality equivalence validation**: Establish regression thresholds for quantized model vs. original FP16/BF16 reference.

## 4. Next Steps

1. Search Hugging Face for Qwen3.8 or similar Qwen2.5/3.x models with AWQ, GPTQ-Int4, or NF4 quantization to confirm availability.
2. Validate vLLM's support for the selected quantization format on SM90 (RTX 5090) hardware.
3. Proceed to Phase 1 (vLLM-Moet expert-provider/cache design) once the Phase 0.5 go/no-go gate is passed.
