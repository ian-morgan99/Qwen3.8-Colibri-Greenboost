#!/usr/bin/env python3
"""Parse safetensors index to extract tensor metadata for Qwen3.8 physical inventory."""

import json
from pathlib import Path

def classify_tensor(name: str):
    """Classify tensor as dense_shared, shared_expert, or routed_expert."""
    if "experts.down_proj" in name or "experts.gate_up_proj" in name:
        return "routed_expert"
    if "shared_expert" in name:
        return "shared_expert"
    if "lm_head.weight" in name or "embed_tokens.weight" in name:
        return "dense_shared"
    if "input_layernorm.weight" in name or "post_attention_layernorm.weight" in name:
        return "dense_shared"
    if "linear_attn." in name:
        return "dense_shared"
    if "mlp.gate.weight" in name:
        return "dense_shared"
    return "dense_shared"

def main():
    index_path = Path("/home/beast/Documents/VSCode/Qwen3.8/checkpoints/Qwen3.8-2.4T-A95B/model.safetensors.index.json")
    
    with open(index_path) as f:
        data = json.load(f)
    
    total_size = data["metadata"]["total_size"]
    weight_map = data["weight_map"]
    
    # Classify tensors
    categories = {
        "dense_shared": {"count": 0, "size_bytes": 0},
        "shared_expert": {"count": 0, "size_bytes": 0},
        "routed_expert": {"count": 0, "size_bytes": 0}
    }
    
    # We need to get the actual tensor sizes from safetensors metadata
    # For now, let's count the entries by category
    
    for tensor_name, file_name in weight_map.items():
        cat = classify_tensor(tensor_name)
        categories[cat]["count"] += 1
    
    print(f"Total model size: {total_size / (1024**4):.2f} TB")
    print(f"Total weight map entries: {len(weight_map)}")
    print(f"Dense/shared tensors: {categories['dense_shared']['count']}")
    print(f"Shared expert tensors: {categories['shared_expert']['count']}")
    print(f"Routed expert tensors: {categories['routed_expert']['count']}")

if __name__ == "__main__":
    main()