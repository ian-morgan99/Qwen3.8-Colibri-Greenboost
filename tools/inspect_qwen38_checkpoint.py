#!/usr/bin/env python3
"""
Phase 0 Tool: inspect_qwen38_checkpoint.py

Parses the Qwen3.8 safetensors checkpoint index to extract tensor metadata,
classifying tensors by category and computing physical footprints.
"""

import json
import os
from typing import Dict, List, Any

CHECKPOINT_INDEX_PATH = "checkpoints/Qwen3.8-2.4T-A95B/model.safetensors.index.json"

def load_checkpoint_index() -> Dict[str, Any]:
    with open(CHECKPOINT_INDEX_PATH, 'r') as f:
        return json.load(f)

def classify_tensor(tensor_name: str) -> str:
    """Classify a tensor into its category based on naming patterns."""
    if tensor_name.startswith('model.layers.') and tensor_name.endswith('.q_proj.weight'):
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and tensor_name.endswith('.k_proj.weight'):
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and tensor_name.endswith('.v_proj.weight'):
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and tensor_name.endswith('.o_proj.weight'):
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and tensor_name.endswith('.gate_proj.weight'):
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and tensor_name.endswith('.up_proj.weight'):
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and tensor_name.endswith('.down_proj.weight'):
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and ('wqkv' in tensor_name or 'qkv_proj' in tensor_name):
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and '.mlp.gate' in tensor_name and '.experts.' not in tensor_name:
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and '.mlp.up' in tensor_name and '.experts.' not in tensor_name:
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and '.mlp.down' in tensor_name and '.experts.' not in tensor_name:
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and '.attention.' in tensor_name and '.wqkv' not in tensor_name:
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and '.input_layernorm' in tensor_name:
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and '.post_attention_layernorm' in tensor_name:
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and '.resid_dropout' in tensor_name:
        return 'dense_shared'
    elif tensor_name.startswith('model.layers.') and '.mlp.experts.' in tensor_name:
        return 'routed_expert'
    elif tensor_name.startswith('model.shared_experts.') or 'shared_mlp' in tensor_name:
        return 'shared_expert'
    elif tensor_name.startswith('model.embed_tokens') or 'embeddings' in tensor_name:
        return 'dense_shared'
    elif tensor_name.startswith('model.norm') or 'lm_head' in tensor_name or 'output' in tensor_name:
        return 'dense_shared'
    elif tensor_name.startswith('model.router') or 'router' in tensor_name and 'expert' not in tensor_name.lower():
        return 'router'
    else:
        # Default classification based on common patterns
        if 'experts' in tensor_name and ('gate' in tensor_name or 'w1' in tensor_name or 'w2' in tensor_name or 'w3' in tensor_name):
            return 'routed_expert'
        elif any(kw in tensor_name for kw in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head', 'embed_tokens', 'norm']):
            return 'dense_shared'
        else:
            return 'unknown'

def compute_tensor_size(bytes_per_element: int, shape: List[int]) -> int:
    """Compute total bytes for a tensor given element size and shape."""
    size = 1
    for dim in shape:
        size *= dim
    return size * bytes_per_element

def main():
    index_data = load_checkpoint_index()
    weight_map = index_data.get('weight_map', {})
    
    # Classify tensors
    classified_tensors: Dict[str, List[Dict[str, Any]]] = {
        'dense_shared': [],
        'shared_expert': [],
        'routed_expert': [],
        'router': [],
        'unknown': []
    }
    
    for tensor_name, file_name in weight_map.items():
        category = classify_tensor(tensor_name)
        classified_tensors[category].append({
            'name': tensor_name,
            'file': file_name
        })
    
    # Output the classification map
    layout_data = {
        'tensor_classification': {
            'dense_shared': [t['name'] for t in classified_tensors['dense_shared']],
            'shared_expert': [t['name'] for t in classified_tensors['shared_expert']],
            'routed_expert': [t['name'] for t in classified_tensors['routed_expert']],
            'router': [t['name'] for t in classified_tensors['router']],
            'unknown': [t['name'] for t in classified_tensors['unknown']]
        },
        'total_tensors': len(weight_map)
    }
    
    # Save layout JSON
    with open('artifacts/qwen38-layout.json', 'w') as f:
        json.dump(layout_data, f, indent=2)
        
    print(f"Classified {len(weight_map)} tensors into categories:")
    for cat, tensors in classified_tensors.items():
        print(f"  {cat}: {len(tensors)} tensors")

if __name__ == '__main__':
    main()
