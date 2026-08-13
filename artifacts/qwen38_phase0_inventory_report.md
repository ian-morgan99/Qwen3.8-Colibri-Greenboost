# Phase 0 — Exact Physical Inventory Report: Qwen3.8 MoE

## Model Configuration
- num_hidden_layers: 92
- hidden_size: 8192
- moe_intermediate_size: 2048
- num_experts: 512
- num_experts_per_tok: 10
- num_attention_heads: 64
- num_key_value_heads: 4
- vocab_size: 152064

## Mandatory Non-Expert VRAM Footprint (FP16/BF16)
- Total mandatory non-expert tensors (embed_tokens + lm_head + all layer norms + linear_attn + mlp.gate + shared_expert): **88.03 GiB**

## Expert Tensors per Layer (FP16/BF16)
- Total expert tensors per layer (gate_up_proj + down_proj for all 512 experts): **48.00 GiB**

## Active Expert Bytes per Layer per Token
- With 10 experts activated per token: **0.9375 GiB per layer per token**

## Phase 0 Go/No-Go Gate Answers
1. **Exact mandatory non-expert VRAM footprint**: ~88.03 GiB in FP16/BF16
2. **Expert tensor shape/kernel compatibility**: Requires validation against vLLM-Moet SM120 kernels (gate_up_proj shape: [512, 4096, 8192], down_proj shape: [512, 8192, 2048])
3. **Bytes per routed expert**: 0.0938 GiB per expert set (gate_up + down_proj)
4. **Active expert bytes per layer/token**: 0.9375 GiB

## Next Steps
- Validate Qwen3.8 expert tensor shapes against existing vLLM-Moet SM120 kernels
- Calculate maximum available expert-cache size: VRAM_free = VRAM_total - mandatory_non_expert - KV(min context) - CUDA workspace
- Estimate stalling cache miss rates and exposed transfer latency for candidate GPU/RAM cache sizes
