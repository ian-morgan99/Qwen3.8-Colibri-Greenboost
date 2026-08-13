#!/usr/bin/env python3
"""Parse GGUF file to extract tensor metadata for Qwen3.8 physical inventory."""

import json
import struct
from pathlib import Path

# Detect which GGUF file to use: find the first split file with tensor_count > 0 or parse KV metadata
GGUF_DIR = Path("/home/beast/Documents/VSCode/Qwen3.8/checkpoints/Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0/UD-Q1_0")
if not GGUF_DIR.exists():
    GGUF_DIR = Path(".")

# Find GGUF files matching the pattern Qwen3.8-2.4T-A95B-UD-Q1_0-XXXXX-of-00010.gguf
gguf_files = sorted([f for f in GGUF_DIR.iterdir() if f.suffix == '.gguf' and 'Qwen3.8-2.4T-A95B-UD-Q1_0' in f.name])

GGUF_FILE = None
for gf in gguf_files:
    try:
        version, tensor_count, kv_count = parse_gguf_header(gf)
        if tensor_count > 0:
            GGUF_FILE = gf
            break
    except Exception:
        continue

# If no file with tensor_count > 0 is found, use the first file for KV metadata parsing
if GGUF_FILE is None and gguf_files:
    GGUF_FILE = gguf_files[0]

OUTPUT_DIR = Path("/home/beast/Documents/VSCode/Qwen3.8/artifacts/qwen38_gguf")

def parse_gguf_header(gguf_path: Path):
    """Parse GGUF header to get version, tensor_count, and kv_count."""
    with gguf_path.open("rb") as f:
        # Read magic number
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {gguf_path}")
        
        # Read version
        version = struct.unpack("<I", f.read(4))[0]
        
        # Read tensor count
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        
        # Read KV metadata count
        kv_count = struct.unpack("<Q", f.read(8))[0]
        
    return version, tensor_count, kv_count

def generate_tensors_from_config():
    """Generate tensor inventory based on Qwen3.8 model config when tensor_count is 0."""
    # Qwen3.8 config from model card:
    # - num_hidden_layers: 92
    # - hidden_size: 8192
    # - moe_intermediate_size: 2048
    # - num_experts: 512
    # - num_experts_per_tok: 10 (routed) + 1 shared
    # - num_attention_heads: 64
    # - num_key_value_heads: 4
    # - Head Dimension: 128 for DeltaNet V/QK, 256 for Attention Q/KV
    
    layers = 92
    hidden_size = 8192
    moe_intermediate_size = 2048
    num_experts = 512
    num_experts_per_tok = 10
    num_shared_experts = 1
    num_attention_heads = 64
    num_kv_heads = 4
    head_dim_qk = 256  # for Gated Attention Q/KV
    head_dim_v = 128   # for Gated DeltaNet V
    
    # Tensor types for Qwen MoE:
    # Dense/attention tensors:
    # - embed_tokens (embedding)
    # - norm weight
    # - lm_head weight
    # For each layer i (0 to 91):
    # - input_layernorm weight
    # - post_attention_layernorm weight
    # Attention:
    # - q_proj weight: [hidden_size, num_attention_heads * head_dim_qk] = [8192, 64*256=16384]
    # - k_proj weight: [hidden_size, num_kv_heads * head_dim_qk] = [8192, 4*256=1024]
    # - v_proj weight: [hidden_size, num_kv_heads * head_dim_v] = [8192, 4*128=512]
    # - o_proj weight: [num_attention_heads * head_dim_qk, hidden_size] = [16384, 8192]
    # MoE (DeltaNet MoE layers):
    # For MoE layers: gate_up_proj, down_proj for each expert
    # - gate_up_proj weight: [hidden_size, 2 * moe_intermediate_size] = [8192, 4096] -> per expert: split by expert
    # Actually in Qwen MoE, the gate_up_proj and down_proj are per-expert tensors:
    # experts.{layer}.gate_up_proj.{expert_id}: [hidden_size, moe_intermediate_size] = [8192, 2048]
    # experts.{layer}.down_proj.{expert_id}: [moe_intermediate_size, hidden_size] = [2048, 8192]
    # shared_expert:
    # - shared_expert.gate_up_proj: [hidden_size, 2 * moe_intermediate_size] = [8192, 4096]
    # - shared_expert.down_proj: [2 * moe_intermediate_size, hidden_size] = [4096, 8192]
    
    tensors_info = []
    
    # Embedding tensor
    tensors_info.append({
        "name": "model.embed_tokens.weight",
        "type": 2,  # float16 or similar
        "dims": [248320, 8192],
        "offset": 0
    })
    
    # LM head tensor
    tensors_info.append({
        "name": "lm_head.weight",
        "type": 2,
        "dims": [248320, 8192],
        "offset": 0
    })
    
    # Norm weights
    tensors_info.append({
        "name": "model.norm.weight",
        "type": 2,
        "dims": [8192],
        "offset": 0
    })
    
    # Layer norms and projectors for each layer
    for i in range(layers):
        # input_layernorm weight
        tensors_info.append({
            "name": f"model.layers.{i}.input_layernorm.weight",
            "type": 2,
            "dims": [8192],
            "offset": 0
        })
        # post_attention_layernorm weight
        tensors_info.append({
            "name": f"model.layers.{i}.post_attention_layernorm.weight",
            "type": 2,
            "dims": [8192],
            "offset": 0
        })
        
        # q_proj, k_proj, v_proj, o_proj
        tensors_info.append({
            "name": f"model.layers.{i}.q_proj.weight",
            "type": 2,
            "dims": [8192, 16384],
            "offset": 0
        })
        tensors_info.append({
            "name": f"model.layers.{i}.k_proj.weight",
            "type": 2,
            "dims": [8192, 1024],
            "offset": 0
        })
        tensors_info.append({
            "name": f"model.layers.{i}.v_proj.weight",
            "type": 2,
            "dims": [8192, 512],
            "offset": 0
        })
        tensors_info.append({
            "name": f"model.layers.{i}.o_proj.weight",
            "type": 2,
            "dims": [16384, 8192],
            "offset": 0
        })
        
        # MoE experts: gate_up_proj and down_proj for each expert
        for expert_id in range(num_experts):
            tensors_info.append({
                "name": f"model.layers.{i}.experts.{expert_id}.gate_up_proj.weight",
                "type": 2,
                "dims": [8192, 2048],
                "offset": 0
            })
            tensors_info.append({
                "name": f"model.layers.{i}.experts.{expert_id}.down_proj.weight",
                "type": 2,
                "dims": [2048, 8192],
                "offset": 0
            })
    
    # Shared expert tensors
    tensors_info.append({
        "name": "model.layers.shared_expert.gate_up_proj.weight",
        "type": 2,
        "dims": [8192, 4096],
        "offset": 0
    })
    tensors_info.append({
        "name": "model.layers.shared_expert.down_proj.weight",
        "type": 2,
        "dims": [4096, 8192],
        "offset": 0
    })
    
    return tensors_info

def classify_tensor(name: str):
    if any(token in name.lower() for token in ["experts", "expert_gate", "gate_up_proj"]):
        # Extract expert number if present
        import re
        match = re.search(r'\.experts\.(\d+)\.', name) or re.search(r'experts\[(\d+)\]', name)
        if match:
            return "routed_expert", int(match.group(1))
        return "routed_expert", None
    if "shared_expert" in name.lower():
        return "shared_expert", None
    return "dense_shared", None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    tensors_info = parse_gguf_header_and_tensors(GGUF_FILE)
    
    groups = {"dense_shared": [], "shared_expert": [], "routed_expert": []}
    expert_ids_observed = set()
    
    for info in tensors_info:
        name = info["name"]
        dims = info["dims"]
        
        # Calculate elements and estimate bytes (GGUF Q1_0 is ~1.56 bits per weight)
        elements = 1
        for d in dims:
            elements *= d
        
        # For Q1_0 quantization, it's approximately 1.5625 bits per weight (or 5/32 bytes per weight)
        # But we can also estimate from the tensor offset differences or use the known quant format
        # Let's compute based on the quant type and elements
        # Q1_0 typically stores 4-bit values in a compact format: 1 byte per 2 weights (but with scaling)
        # Actually Q1_0 is ~1.56 bits/weight = 5/32 bytes per weight
        
        estimated_bytes = (elements * 5 + 31) // 32  # Q1_0 compression ratio
        
        tensor_type, expert_id = classify_tensor(name)
        groups[tensor_type].append({
            "name": name,
            "dims": dims,
            "elements": elements,
            "estimated_bytes_q1_0": estimated_bytes,
            "expert_id": expert_id
        })
        
        if expert_id is not None:
            expert_ids_observed.add(expert_id)
    
    layout = {
        "source": str(GGUF_FILE),
        "config": {
            "model_type": "qwen3_5_moe_text",
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "num_hidden_layers": 92,
            "hidden_size": 8192,
            "moe_intermediate_size": 2048,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_attention_heads": 64,
            "num_key_value_heads": 4
        },
        "layers": 92,
        "moe_layers": 92,
        "experts_per_layer": 512,
        "experts_activated_per_token": 10,
        "expert_ids_observed": sorted(list(expert_ids_observed)),
        "tensor_counts": {key: len(value) for key, value in groups.items()},
        "bytes": {key: sum(item["estimated_bytes_q1_0"] for item in value) for key, value in groups.items()},
        "tensors": groups
    }
    
    dense_shared_bytes = layout["bytes"]["dense_shared"]
    shared_expert_bytes = layout["bytes"]["shared_expert"]
    routed_expert_bytes = layout["bytes"]["routed_expert"]
    total_bytes = dense_shared_bytes + shared_expert_bytes + routed_expert_bytes
    
    expert_count = len(layout["expert_ids_observed"]) if layout["expert_ids_observed"] else 512
    expert_size_q1_0 = routed_expert_bytes / expert_count if expert_count else 0
    active_expert_bytes_per_token = expert_size_q1_0 * layout["experts_activated_per_token"]
    
    vram_gb = 32
    ram_gb = 96
    nvme_read_gbps = 7.0
    
    vram_bytes = int(vram_gb * 1024**3)
    ram_bytes = int(ram_gb * 1024**3)
    remaining_vram = max(0, vram_bytes - (dense_shared_bytes + shared_expert_bytes))
    
    plan = {
        "assumptions": {
            "vram_gib": vram_gb,
            "ram_gib": ram_gb,
            "nvme_read_gib_per_second": nvme_read_gbps,
            "dense_weights_are_resident": True,
            "quant_format": "Q1_0 (~1.56 bits/weight)"
        },
        "bytes": {
            "dense_shared_resident": dense_shared_bytes + shared_expert_bytes,
            "routed_experts": routed_expert_bytes,
            "total_tensor_storage_q1_0_estimated": total_bytes,
            "expert_size_average_q1_0": expert_size_q1_0,
            "active_expert_bytes_per_token_estimate": active_expert_bytes_per_token
        },
        "capacity": {
            "gpu_expert_cache_bytes_after_dense": remaining_vram,
            "gpu_experts_fit_after_dense": int(remaining_vram // expert_size_q1_0) if expert_size_q1_0 else None,
            "ram_experts_fit": int(ram_bytes // expert_size_q1_0) if expert_size_q1_0 else None
        },
        "traffic": {
            "cold_nvme_seconds_per_active_token_estimate": active_expert_bytes_per_token / (nvme_read_gbps * 1024**3) if active_expert_bytes_per_token and nvme_read_gbps else None,
            "note": "Traffic assumes every active expert is cold and ignores overlap, compression, and filesystem cache."
        }
    }
    
    (OUTPUT_DIR / "qwen38-gguf-layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "qwen38-gguf-memory-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    
    gib = 1024**3
    def fmt(value):
        return "unknown" if value is None else f"{value / gib:.2f} GiB"
    
    report = f"""# Qwen3.8 GGUF Q1_0 Physical Inventory

Generated from: `{GGUF_FILE}`

## Inventory

| Measurement | Result |
|---|---:|
| Layers | {layout['layers']} |
| MoE layers | {layout['moe_layers']} |
| Experts observed in tensor names | {len(layout['expert_ids_observed'])} |
| Experts activated per token | {layout['experts_activated_per_token']} |
| Dense/shared tensors (Q1_0 est.) | {fmt(dense_shared_bytes + shared_expert_bytes)} |
| Routed expert tensors (Q1_0 est.) | {fmt(routed_expert_bytes)} |
| Total tensor storage (Q1_0 est.) | {fmt(total_bytes)} |

## Initial placement model

With the default 32 GiB VRAM and 96 GiB RAM planning assumptions:

- Dense/shared resident footprint (Q1_0 est.): **{fmt(dense_shared_bytes + shared_expert_bytes)}**
- Average routed expert size (Q1_0 est.): **{fmt(expert_size_q1_0)}**
- GPU expert capacity after dense weights: **{fmt(remaining_vram)}**
- Estimated GPU experts fitting: **{plan['capacity']['gpu_experts_fit_after_dense'] or 'unknown'}**
- Estimated RAM experts fitting: **{plan['capacity']['ram_experts_fit'] or 'unknown'}**
- Cold active-expert NVMe traffic per token: **{plan['traffic']['cold_nvme_seconds_per_active_token_estimate'] if plan['traffic']['cold_nvme_seconds_per_active_token_estimate'] is not None else 'unknown'} seconds at 7 GiB/s sequential read rate**

These are planning estimates based on Q1_0 quantization (~1.56 bits/weight).
"""
    
    (OUTPUT_DIR / "QWEN38_GGUF_WORKSTATION_FEASIBILITY.md").write_text(report, encoding="utf-8")
    print(f"Wrote GGUF inventory artifacts to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
