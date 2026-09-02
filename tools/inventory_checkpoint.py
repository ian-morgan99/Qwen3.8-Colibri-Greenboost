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
    """Build the v2 memory plan from the SoT and the routing-trace artifact.

    The previous v1 schema (assumptions/bytes/capacity/traffic) is now produced
    by ``simulate_expert_cache.build_memory_plan`` and stamped with
    ``schema_version = qwen38.memory_plan.v2``. We delegate to that builder so
    that ``inventory_checkpoint.py`` and ``simulate_expert_cache.py`` cannot
    drift apart — the per-expert byte size, GGUF quantisation capacity and
    config_sha256 all come from ``tools.qwen38_config`` rather than from the
    in-memory tensor scan, which is more robust against partial inventory
    (see GitHub Issue #2 AC1, AC2, AC6).

    The v1 ``traffic`` block is intentionally dropped: cold-NVMe estimates are
    superseded by the stalling-miss-rate measured in
    ``artifacts/qwen38_routing_trace_metrics.json``, which is the
    authoritative behaviour metric.
    """
    from simulate_expert_cache import build_memory_plan as _v2_builder

    # ``vram_gb`` / ``ram_gb`` arguments are accepted but only the cache-size
    # tables carry through to the v2 schema. The caller selects the sweep.
    return _v2_builder(
        gpu_cache_sizes_gb=[8, 12, 16, 20, 24],
        ram_arena_sizes_gb=[48, 64, 72, 96, 128],
        routing_trace_metrics_path="artifacts/qwen38_routing_trace_metrics.json",
    )


def report(layout: dict[str, Any], plan: dict[str, Any]) -> str:
    gib = 1024**3
    def fmt(value: Any) -> str:
        return "unknown" if value is None else f"{value / gib:.2f} GiB"
    bytes_per_expert = plan.get("bytes_per_expert_per_quant", {})
    quant_table = "\n".join(
        f"| `{name}` | {size:,} |"
        for name, size in sorted(bytes_per_expert.items())
    ) or "| _(none)_ |  |"

    def gpu_row(sim: dict[str, Any]) -> str:
        return (
            f"| {sim['cache_gb']} GiB | {sim['experts_fit_bf16']} | "
            f"{sim['experts_fit_per_quant'].get('q1_0', '?')} | "
            f"{sim['experts_fit_per_quant'].get('q4_k_m', '?')} | "
            f"{sim['experts_fit_per_quant'].get('q8_0', '?')} |"
        )

    def ram_row(sim: dict[str, Any]) -> str:
        return (
            f"| {sim['arena_gb']} GiB | {sim['experts_fit_bf16']} | "
            f"{sim['experts_fit_per_quant'].get('q1_0', '?')} | "
            f"{sim['experts_fit_per_quant'].get('q4_k_m', '?')} | "
            f"{sim['experts_fit_per_quant'].get('q8_0', '?')} |"
        )

    gpu_rows = "\n".join(gpu_row(s) for s in plan.get("gpu_cache_simulations", []))
    ram_rows = "\n".join(ram_row(s) for s in plan.get("ram_arena_simulations", []))
    hit = plan.get("hit_rates_from_routing_trace", {})
    prov = plan.get("provenance", {})
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

## Per-expert footprint (GGUF quantisations)

| Quant | Bytes per expert |
|---|---:|
{quant_table}

## GPU expert cache sweep (after dense/shared resident footprint)

| Cache size | BF16 experts | Q1_0 experts | Q4_K_M experts | Q8_0 experts |
|---|---:|---:|---:|---:|
{gpu_rows or '| _(none)_ |  |  |  |  |'}

## RAM expert arena sweep

| Arena size | BF16 experts | Q1_0 experts | Q4_K_M experts | Q8_0 experts |
|---|---:|---:|---:|---:|
{ram_rows or '| _(none)_ |  |  |  |  |'}

## Hit/miss behaviour (from `qwen38_routing_trace_metrics.json`)

- L1 hit rate: **{hit.get('l1_hit_rate_pct', 'unknown')}%**
- L2 hit rate: **{hit.get('l2_hit_rate_pct', 'unknown')}%**
- Stalling miss rate: **{hit.get('stalling_miss_rate_pct', 'unknown')}%**

## Provenance

- config_sha256: `{prov.get('config_sha256', 'unknown')}`
- routing trace artifact: `{prov.get('routing_trace_metrics_path', 'unknown')}`
- simulator_commit: `{prov.get('simulator_commit', 'unknown')}`
- data classification: `{prov.get('data_classification', 'unknown')}`

These numbers are derived from the checkpoint architecture (not the tensor
scan) and from the synthetic LFRU trace in
`artifacts/qwen38_routing_trace_metrics.json`. Cold-NVMe traffic estimates
from the v1 schema are intentionally omitted; the stalling-miss rate above
is the authoritative behaviour metric. See
`docs/architecture/QWEN38_CHECKPOINT_DERIVED.md` for derivation details.

## Limitations and gate

{chr(10).join(f'- {item}' for item in layout['limitations']) or '- No metadata limitations detected.'}

This inventory does not establish inference correctness or RTX 5090 kernel support.
Those remain Phase 1 gate items before implementing the tiered runtime.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    # Canonical namespace: artifacts/ (root), not artifacts/qwen38/.
    # See tools/test_no_duplicate_artifacts.py for the regression test that
    # guards this invariant.
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--feasibility-output", type=Path, default=Path("docs"),
                        help="Directory for the narrative QWEN38_WORKSTATION_FEASIBILITY.md. "
                             "Canonical location is docs/; override only if you know what you are doing.")
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
    args.feasibility_output.mkdir(parents=True, exist_ok=True)
    (args.feasibility_output / "QWEN38_WORKSTATION_FEASIBILITY.md").write_text(report(layout, plan), encoding="utf-8")
    print(f"Wrote inventory artifacts to {args.output}")
    print(f"Wrote feasibility report to {args.feasibility_output / 'QWEN38_WORKSTATION_FEASIBILITY.md'}")


if __name__ == "__main__":
    main()