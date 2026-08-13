#!/usr/bin/env python3
"""Inventory a local Hugging Face checkpoint for the Qwen3.8 workstation plan."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path
from typing import Any


DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "float32": 4,
    "int32": 4,
    "int64": 8,
    "float64": 8,
}
EXPERT_RE = re.compile(r"(?:\.experts?\.|\.experts\[?)(\d+)")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size > 100 * 1024 * 1024:
            raise ValueError(f"safetensors header is unexpectedly large: {path}")
        return json.loads(handle.read(header_size))


def tensor_bytes(tensor: dict[str, Any]) -> int:
    if "data_offsets" in tensor:
        start, end = tensor["data_offsets"]
        return end - start
    elements = math.prod(tensor.get("shape", []))
    dtype = tensor.get("dtype", "")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"cannot calculate size for dtype {dtype!r}")
    return elements * DTYPE_BYTES[dtype]


def collect_tensors(checkpoint: Path) -> tuple[dict[str, dict[str, Any]], str]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        tensors = {}
        for name, shard in index["weight_map"].items():
            tensors[name] = {"shard": shard}
        # Index files normally omit shape and dtype, so enrich from shard headers.
        headers: dict[str, dict[str, Any]] = {}
        for shard in sorted(set(index["weight_map"].values())):
            headers.update({name: value for name, value in safetensors_header(checkpoint / shard).items() if name != "__metadata__"})
        for name, value in tensors.items():
            value.update(headers.get(name, {}))
        return tensors, "safetensors index and shard headers"

    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("expected model.safetensors.index.json or *.safetensors")
    tensors = {}
    for shard in files:
        tensors.update({name: {**value, "shard": shard.name} for name, value in safetensors_header(shard).items() if name != "__metadata__"})
    return tensors, "safetensors shard headers"


def expert_number(name: str) -> int | None:
    match = EXPERT_RE.search(name)
    return int(match.group(1)) if match else None


def classify(name: str, config: dict[str, Any]) -> str:
    if expert_number(name) is not None or any(token in name.lower() for token in ("experts", "expert_gate", "gate_up_proj")):
        return "routed_expert"
    if "shared_expert" in name.lower() or "shared_experts" in name.lower():
        return "shared_expert"
    return "dense_shared"


def build_layout(config: dict[str, Any], tensors: dict[str, dict[str, Any]], source: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"dense_shared": [], "shared_expert": [], "routed_expert": []}
    for name, tensor in tensors.items():
        entry = {"name": name, "bytes": tensor_bytes(tensor), "dtype": tensor.get("dtype"), "shape": tensor.get("shape"), "shard": tensor.get("shard")}
        groups[classify(name, config)].append(entry)
    expert_names = [item["name"] for item in groups["routed_expert"]]
    expert_ids = [expert_number(name) for name in expert_names if expert_number(name) is not None]
    layers = config.get("num_hidden_layers", config.get("num_layers"))
    experts_per_token = config.get("num_experts_per_tok", config.get("num_selected_experts"))
    total_experts = config.get("num_local_experts", config.get("num_experts"))
    return {
        "source": source,
        "config": {key: config.get(key) for key in ("model_type", "architectures", "torch_dtype", "num_hidden_layers", "num_layers", "hidden_size", "intermediate_size", "num_local_experts", "num_experts", "num_experts_per_tok", "num_selected_experts", "num_attention_heads", "num_key_value_heads") if key in config},
        "layers": layers,
        "moe_layers": None,
        "experts_per_layer": total_experts,
        "experts_activated_per_token": experts_per_token,
        "expert_ids_observed": sorted(set(expert_ids)),
        "tensor_counts": {key: len(value) for key, value in groups.items()},
        "bytes": {key: sum(item["bytes"] for item in value) for key, value in groups.items()},
        "tensors": groups,
        "limitations": ["moe_layers is unknown unless represented explicitly in config or tensor names", "experts_activated_per_token is copied from config and is not inferred from generated tokens"] if layers is None or experts_per_token is None else [],
    }


def build_memory_plan(layout: dict[str, Any], vram_gb: float, ram_gb: float, nvme_read_gbps: float) -> dict[str, Any]:
    dense = layout["bytes"]["dense_shared"] + layout["bytes"]["shared_expert"]
    routed = layout["bytes"]["routed_expert"]
    expert_count = len(layout["expert_ids_observed"])
    expert_size = routed / expert_count if expert_count else None
    vram_bytes = int(vram_gb * 1024**3)
    ram_bytes = int(ram_gb * 1024**3)
    remaining_vram = max(0, vram_bytes - dense)
    active = expert_size * (layout["experts_activated_per_token"] or 0) if expert_size is not None else None
    return {
        "assumptions": {"vram_gib": vram_gb, "ram_gib": ram_gb, "nvme_read_gib_per_second": nvme_read_gbps, "dense_weights_are_resident": True},
        "bytes": {"dense_shared_resident": dense, "routed_experts": routed, "total_tensor_storage": dense + routed, "expert_size_average": expert_size, "active_expert_bytes_per_token_estimate": active},
        "capacity": {"gpu_expert_cache_bytes_after_dense": remaining_vram, "gpu_experts_fit_after_dense": int(remaining_vram // expert_size) if expert_size else None, "ram_experts_fit": int(ram_bytes // expert_size) if expert_size else None},
        "traffic": {"cold_nvme_seconds_per_active_token_estimate": active / (nvme_read_gbps * 1024**3) if active and nvme_read_gbps else None, "note": "Traffic assumes every active expert is cold and ignores overlap, compression, and filesystem cache."},
    }


def report(layout: dict[str, Any], plan: dict[str, Any]) -> str:
    gib = 1024**3
    def fmt(value: Any) -> str:
        return "unknown" if value is None else f"{value / gib:.2f} GiB"
    return f"""# Qwen3.8 workstation feasibility

Generated from: `{layout['source']}`

## Inventory

| Measurement | Result |
|---|---:|
| Layers | {layout['layers'] or 'unknown'} |
| MoE layers | {layout['moe_layers'] or 'unknown'} |
| Experts observed in tensor names | {len(layout['expert_ids_observed'])} |
| Experts activated per token | {layout['experts_activated_per_token'] or 'unknown'} |
| Dense/shared tensors | {fmt(layout['bytes']['dense_shared'])} |
| Shared expert tensors | {fmt(layout['bytes']['shared_expert'])} |
| Routed expert tensors | {fmt(layout['bytes']['routed_expert'])} |
| Total tensor storage | {fmt(sum(layout['bytes'].values()))} |

## Initial placement model

With the default 32 GiB VRAM and 70 GiB RAM planning assumptions:

- Dense/shared resident footprint: **{fmt(plan['bytes']['dense_shared_resident'])}**
- Average routed expert size: **{fmt(plan['bytes']['expert_size_average'])}**
- GPU expert capacity after dense weights: **{fmt(plan['capacity']['gpu_expert_cache_bytes_after_dense'])}**
- Estimated GPU experts fitting: **{plan['capacity']['gpu_experts_fit_after_dense'] or 'unknown'}**
- Estimated RAM experts fitting: **{plan['capacity']['ram_experts_fit'] or 'unknown'}**
- Cold active-expert NVMe traffic per token: **{plan['traffic']['cold_nvme_seconds_per_active_token_estimate'] if plan['traffic']['cold_nvme_seconds_per_active_token_estimate'] is not None else 'unknown'} seconds at the configured sequential read rate**

These are planning estimates. The report deliberately leaves MoE-layer count and
runtime hit rates unknown when the checkpoint does not expose enough metadata.

## Limitations and gate

{chr(10).join(f'- {item}' for item in layout['limitations']) or '- No metadata limitations detected.'}

This inventory does not establish inference correctness or RTX 5090 kernel support.
Those remain Phase 1 gate items before implementing the tiered runtime.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen38"))
    parser.add_argument("--vram-gib", type=float, default=32)
    parser.add_argument("--ram-gib", type=float, default=70)
    parser.add_argument("--nvme-read-gibps", type=float, default=7)
    args = parser.parse_args()
    config = read_json(args.checkpoint / "config.json")
    tensors, source = collect_tensors(args.checkpoint)
    layout = build_layout(config, tensors, source)
    plan = build_memory_plan(layout, args.vram_gib, args.ram_gib, args.nvme_read_gibps)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "qwen38-layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    (args.output / "qwen38-memory-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (args.output / "QWEN38_WORKSTATION_FEASIBILITY.md").write_text(report(layout, plan), encoding="utf-8")
    print(f"Wrote inventory artifacts to {args.output}")


if __name__ == "__main__":
    main()