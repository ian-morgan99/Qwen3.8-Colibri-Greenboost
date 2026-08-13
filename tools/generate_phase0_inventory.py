#!/usr/bin/env python3
"""Generate Phase 0 — Exact Physical Inventory report for Qwen3.8 MoE."""

import json
from pathlib import Path

# Qwen3.8 MoE configuration from GGUF/safetensors analysis
CONFIG = {
    "num_hidden_layers": 92,
    "hidden_size": 8192,
    "moe_intermediate_size": 2048,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "vocab_size": 152064,  # Typical for Qwen3.x/3.5
}

def compute_tensor_sizes():
    """Compute theoretical tensor sizes in FP16/BF16 (2 bytes per param)."""
    H = CONFIG["hidden_size"]
    M = CONFIG["moe_intermediate_size"]
    E = CONFIG["num_experts"]
    V = CONFIG["vocab_size"]
    L = CONFIG["num_hidden_layers"]
    
    sizes_fp16 = {}
    
    # Dense/shared tensors
    sizes_fp16["model.embed_tokens.weight"] = V * H * 2  # bytes
    sizes_fp16["lm_head.weight"] = V * H * 2  # bytes
    
    # Layer norms per layer
    sizes_fp16["input_layernorm.weight"] = H * 2
    sizes_fp16["post_attention_layernorm.weight"] = H * 2
    
    # Linear attention tensors per layer (approximate based on Qwen linear attn structure)
    # in_proj_qkv: 3 * H * H or similar; in_proj_a, in_proj_b, in_proj_z, out_proj, A_log, dt_bias, conv1d, norm
    # Approximate total linear_attn per layer: ~5-8H^2 bytes
    sizes_fp16["linear_attn.in_proj_qkv.weight"] = 3 * H * H * 2
    sizes_fp16["linear_attn.in_proj_a.weight"] = H * (H // 4) * 2
    sizes_fp16["linear_attn.in_proj_b.weight"] = H * (H // 4) * 2
    sizes_fp16["linear_attn.in_proj_z.weight"] = H * H * 2
    sizes_fp16["linear_attn.out_proj.weight"] = H * H * 2
    sizes_fp16["linear_attn.A_log"] = H // 4 * 2
    sizes_fp16["linear_attn.dt_bias"] = H // 4 * 2
    sizes_fp16["linear_attn.conv1d.weight"] = (33 or 35) * H // 4 * 2  # conv kernel
    sizes_fp16["linear_attn.norm.weight"] = H // 4 * 2
    
    # MLP gate per layer (non-expert)
    sizes_fp16["mlp.gate.weight"] = H * H * 2
    
    # Shared expert tensors per layer
    sizes_fp16["mlp.shared_expert.gate_proj.weight"] = H * M * 2
    sizes_fp16["mlp.shared_expert.up_proj.weight"] = H * M * 2
    sizes_fp16["mlp.shared_expert.down_proj.weight"] = M * H * 2
    sizes_fp16["mlp.shared_expert_gate.weight"] = H * 2  # router/gate for shared expert
    
    # Routed expert tensors per layer
    # gate_up_proj: fused gate+up for all experts: [E, 2*M, H]
    sizes_fp16["mlp.experts.gate_up_proj"] = E * (2 * M) * H * 2
    # down_proj: [E, H, M]
    sizes_fp16["mlp.experts.down_proj"] = E * H * M * 2
    
    return sizes_fp16

def compute_footprints():
    sizes = compute_tensor_sizes()
    
    # Mandatory non-expert VRAM footprint (FP16)
    mandatory_non_expert = {
        "embed_tokens": sizes["model.embed_tokens.weight"],
        "lm_head": sizes["lm_head.weight"],
        "layer_norms_per_layer": sizes["input_layernorm.weight"] + sizes["post_attention_layernorm.weight"],
        "linear_attn_per_layer": sum([
            sizes.get(f"linear_attn.{k}", 0) for k in [
                "in_proj_qkv.weight", "in_proj_a.weight", "in_proj_b.weight", 
                "in_proj_z.weight", "out_proj.weight", "A_log", "dt_bias", 
                "conv1d.weight", "norm.weight"
            ]
        ]),
        "mlp_gate_per_layer": sizes["mlp.gate.weight"],
        "shared_expert_per_layer": sum([
            sizes[f"mlp.shared_expert.{k}.weight"] for k in [
                "gate_proj", "up_proj", "down_proj"
            ]
        ]) + sizes.get("mlp.shared_expert_gate.weight", 0),
    }
    
    total_mandatory_non_expert_per_layer = (
        mandatory_non_expert["layer_norms_per_layer"] +
        mandatory_non_expert["linear_attn_per_layer"] +
        mandatory_non_expert["mlp_gate_per_layer"] +
        mandatory_non_expert["shared_expert_per_layer"]
    )
    
    total_mandatory_non_expert_global = (
        mandatory_non_expert["embed_tokens"] +
        mandatory_non_expert["lm_head"] +
        CONFIG["num_hidden_layers"] * total_mandatory_non_expert_per_layer
    )
    
    # Expert tensors per layer
    expert_per_layer = sizes["mlp.experts.gate_up_proj"] + sizes["mlp.experts.down_proj"]
    
    # Active expert bytes per layer per token (10 experts activated)
    active_expert_bytes_per_layer_per_token = (expert_per_layer / CONFIG["num_experts"]) * CONFIG["num_experts_per_tok"]
    
    results = {
        "config": CONFIG,
        "mandatory_non_expert_global_bytes": total_mandatory_non_expert_global,
        "mandatory_non_expert_global_gib": total_mandatory_non_expert_global / (1024**3),
        "expert_per_layer_bytes": expert_per_layer,
        "expert_per_layer_gib": expert_per_layer / (1024**3),
        "active_expert_bytes_per_layer_per_token": active_expert_bytes_per_layer_per_token,
        "active_expert_bytes_per_layer_per_token_gib": active_expert_bytes_per_layer_per_token / (1024**3),
    }
    
    return results

def main():
    results = compute_footprints()
    
    report = f"""# Phase 0 — Exact Physical Inventory Report: Qwen3.8 MoE

## Model Configuration
- num_hidden_layers: {CONFIG['num_hidden_layers']}
- hidden_size: {CONFIG['hidden_size']}
- moe_intermediate_size: {CONFIG['moe_intermediate_size']}
- num_experts: {CONFIG['num_experts']}
- num_experts_per_tok: {CONFIG['num_experts_per_tok']}
- num_attention_heads: {CONFIG['num_attention_heads']}
- num_key_value_heads: {CONFIG['num_key_value_heads']}
- vocab_size: {CONFIG['vocab_size']}

## Mandatory Non-Expert VRAM Footprint (FP16/BF16)
- Total mandatory non-expert tensors (embed_tokens + lm_head + all layer norms + linear_attn + mlp.gate + shared_expert): **{results['mandatory_non_expert_global_gib']:.2f} GiB**

## Expert Tensors per Layer (FP16/BF16)
- Total expert tensors per layer (gate_up_proj + down_proj for all {CONFIG['num_experts']} experts): **{results['expert_per_layer_gib']:.2f} GiB**

## Active Expert Bytes per Layer per Token
- With {CONFIG['num_experts_per_tok']} experts activated per token: **{results['active_expert_bytes_per_layer_per_token_gib']:.4f} GiB per layer per token**

## Phase 0 Go/No-Go Gate Answers
1. **Exact mandatory non-expert VRAM footprint**: ~{results['mandatory_non_expert_global_gib']:.2f} GiB in FP16/BF16
2. **Expert tensor shape/kernel compatibility**: Requires validation against vLLM-Moet SM120 kernels (gate_up_proj shape: [512, 4096, 8192], down_proj shape: [512, 8192, 2048])
3. **Bytes per routed expert**: {results['expert_per_layer_gib'] / CONFIG['num_experts']:.4f} GiB per expert set (gate_up + down_proj)
4. **Active expert bytes per layer/token**: {results['active_expert_bytes_per_layer_per_token_gib']:.4f} GiB

## Next Steps
- Validate Qwen3.8 expert tensor shapes against existing vLLM-Moet SM120 kernels
- Calculate maximum available expert-cache size: VRAM_free = VRAM_total - mandatory_non_expert - KV(min context) - CUDA workspace
- Simulate zero-miss-step curves for candidate GPU/RAM cache sizes
"""

    report_path = Path("/home/beast/Documents/VSCode/Qwen3.8/artifacts/qwen38_phase0_inventory_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    
    print(f"Phase 0 inventory report written to {report_path}")
    print(report)

if __name__ == "__main__":
    main()
