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

# Use the typed error taxonomy introduced for the Colibrì v1.6.2
# hardening (Issue #6). Anything that fails a byte-range check here
# raises ``TensorOutOfFileBounds`` so the caller can branch on a
# well-known code rather than catching ValueError.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader_errors import (  # noqa: E402
    InvalidTensorShape,
    MalformedShardHeader,
    TensorOutOfFileBounds,
)


# safetensors dtype codes and their byte widths. The on-disk spec uses
# the short uppercase forms (``F32``, ``BF16``, ``I64`` …); some
# internal tooling also emits the long lowercase form (``float32`` …).
# Both forms are listed so the validator works against either.
DTYPE_BYTES = {
    "bool": 1, "BOOL": 1,
    "int8": 1, "I8": 1,
    "uint8": 1, "U8": 1,
    "float8_e4m3fn": 1, "F8_E4M3": 1,
    "float8_e4m3fnuz": 1, "F8_E4M3FNUZ": 1,
    "float16": 2, "F16": 2,
    "bfloat16": 2, "BF16": 2,
    "int16": 2, "I16": 2,
    "float32": 4, "F32": 4,
    "int32": 4, "I32": 4,
    "int64": 8, "I64": 8,
    "float64": 8, "F64": 8,
}
EXPERT_RE = re.compile(r"(?:\.experts?\.|\.experts\[?)(\d+)")


# Header size sanity cap: 100 MiB is far above any real safetensors header
# we've seen, and a malformed header that declares itself to be 8 GiB
# should fail fast rather than OOM the inventory run.
MAX_HEADER_SIZE = 100 * 1024 * 1024


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safetensors_header(path: Path) -> dict[str, Any]:
    """Read a safetensors file's JSON header.

    Raises :class:`MalformedShardHeader` if the file is too small to
    even contain the 8-byte header-length prefix, or if the declared
    header size is non-positive or above :data:`MAX_HEADER_SIZE`.
    """
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise MalformedShardHeader(
                    message=f"shard is shorter than 8-byte header prefix: {path}",
                    details={"path": str(path), "got_bytes": len(prefix)},
                )
            (header_size,) = struct.unpack("<Q", prefix)
            if header_size == 0 or header_size > MAX_HEADER_SIZE:
                raise MalformedShardHeader(
                    message=f"shard declares implausible header size: {path}",
                    details={
                        "path": str(path),
                        "header_size": header_size,
                        "max_allowed": MAX_HEADER_SIZE,
                    },
                )
            blob = handle.read(header_size)
            if len(blob) != header_size:
                raise MalformedShardHeader(
                    message="short read on safetensors header body",
                    details={
                        "path": str(path),
                        "declared": header_size,
                        "read": len(blob),
                    },
                )
            return json.loads(blob)
    except OSError as exc:
        raise MalformedShardHeader(
            message=f"could not read safetensors shard: {path}",
            details={"path": str(path), "os_error": str(exc)},
        ) from exc


def validate_tensor_offsets(
    shard: Path, tensor: dict[str, Any], tensor_name: str
) -> dict[str, Any]:
    """Return a copy of ``tensor`` augmented with range validation.

    The safetensors header for each tensor declares
    ``data_offsets = [start, end]`` (absolute byte offsets within the
    file, *after* the 8-byte header-size prefix and the JSON header
    body). If those offsets are inconsistent with the actual file size,
    or if the byte count disagrees with the declared shape × dtype, the
    function raises :class:`TensorOutOfFileBounds` or
    :class:`InvalidTensorShape` respectively.

    On success the returned dict is augmented with::

        bytes_validated: True
        source_offset:   start
        source_end:      end
        file_size:       <on-disk size at the time of the check>

    These fields are stamped onto the layout so downstream tooling
    can see that the check ran (rather than silently trusting the
    header).
    """
    if "data_offsets" not in tensor:
        # If the header didn't include data_offsets (some experimental
        # formats), we cannot validate; surface that as a typed error
        # rather than silently skipping.
        raise TensorOutOfFileBounds(
            message=f"tensor has no data_offsets: {tensor_name}",
            details={"tensor": tensor_name, "shard": str(shard)},
        )
    start, end = tensor["data_offsets"]
    if not (isinstance(start, int) and isinstance(end, int)):
        raise TensorOutOfFileBounds(
            message=f"data_offsets must be ints, got ({type(start).__name__}, "
            f"{type(end).__name__})",
            details={"tensor": tensor_name, "shard": str(shard)},
        )
    if start < 0 or end < start:
        raise TensorOutOfFileBounds(
            message="data_offsets must be non-negative and end >= start",
            details={
                "tensor": tensor_name,
                "shard": str(shard),
                "start": start,
                "end": end,
            },
        )
    declared = end - start
    try:
        file_size = shard.stat().st_size
    except OSError as exc:
        # A missing or unreadable shard is a *header*-level problem from
        # the caller's perspective: the shard cannot be inspected at all.
        # Surface it as a typed MalformedShardHeader so the caller can
        # branch on a well-known code rather than catching OSError.
        raise MalformedShardHeader(
            message=f"could not stat safetensors shard: {shard}",
            details={"path": str(shard), "os_error": str(exc)},
        ) from exc
    if end > file_size:
        raise TensorOutOfFileBounds(
            message="tensor data_offsets end is beyond end-of-file",
            details={
                "tensor": tensor_name,
                "shard": str(shard),
                "end_offset": end,
                "file_size": file_size,
            },
        )
    # Cross-check: shape × dtype must equal the declared byte count.
    # A mismatch here means the header is lying about its own contents.
    elements = math.prod(tensor.get("shape", []))
    dtype = tensor.get("dtype", "")
    if dtype not in DTYPE_BYTES:
        raise InvalidTensorShape(
            message=f"unknown dtype for shape/byte cross-check: {dtype!r}",
            details={"tensor": tensor_name, "shard": str(shard), "dtype": dtype},
        )
    expected = elements * DTYPE_BYTES[dtype]
    if expected != declared:
        raise InvalidTensorShape(
            message="declared byte count does not match shape × dtype",
            details={
                "tensor": tensor_name,
                "shard": str(shard),
                "expected_bytes": expected,
                "declared_bytes": declared,
                "shape": tensor.get("shape"),
                "dtype": dtype,
            },
        )
    return {
        **tensor,
        "bytes_validated": True,
        "source_offset": start,
        "source_end": end,
        "file_size": file_size,
    }


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
    """Walk the checkpoint's index and shard files.

    Every tensor that has its ``data_offsets`` declared in the shard
    header is also range-validated against the on-disk file
    (:func:`validate_tensor_offsets`). The returned ``source`` string
    describes where the metadata came from.
    """
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = read_json(index_path)
        tensors: dict[str, dict[str, Any]] = {}
        for name, shard in index["weight_map"].items():
            tensors[name] = {"shard": shard}
        # Index files normally omit shape and dtype, so enrich from shard
        # headers and *also* range-validate the byte ranges while we're
        # there. This is the cheap version of "open every shard" because
        # the header read is a single small fseek; the actual data block
        # is never touched.
        for shard in sorted(set(index["weight_map"].values())):
            shard_path = checkpoint / shard
            for name, header in safetensors_header(shard_path).items():
                if name == "__metadata__":
                    continue
                if name not in tensors:
                    continue  # referenced from a different shard
                tensors[name].update(header)
                if "data_offsets" in header:
                    tensors[name] = validate_tensor_offsets(
                        shard_path, tensors[name], name
                    )
        return tensors, "safetensors index and validated shard headers"

    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError("expected model.safetensors.index.json or *.safetensors")
    for shard in files:
        for name, value in safetensors_header(shard).items():
            if name == "__metadata__":
                continue
            tensors[name] = {**value, "shard": shard.name}
            if "data_offsets" in value:
                tensors[name] = validate_tensor_offsets(
                    shard, tensors[name], name
                )
    return tensors, "validated safetensors shard headers"


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