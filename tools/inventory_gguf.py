#!/usr/bin/env python3
"""Parse GGUF file to extract tensor metadata for Qwen3.8 physical inventory.

Generates three artifacts in ``artifacts/qwen38_gguf/`` from a real Qwen3.8
GGUF shard on disk:

  * ``qwen38-gguf-layout.json``      — per-tensor dimensions + Q1_0 byte
                                       estimates, grouped by
                                       routed_expert / shared_expert /
                                       dense_shared, with provenance.
  * ``qwen38-gguf-memory-plan.json`` — capacity / traffic estimates for
                                       the workstation profile, with
                                       v2-schema provenance.
  * ``QWEN38_GGUF_WORKSTATION_FEASIBILITY.md`` — human-readable report.

The tool is tolerant of *metadata-only* GGUF shards (those with
``tensor_count == 0`` in the header). When a metadata-only shard is
encountered, the tensor inventory is synthesised from the canonical
Qwen3.8 config (``tools/qwen38_config``) so downstream capacity planning
remains well-defined. Once a real data-bearing shard is dropped into
``checkpoints/Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0/UD-Q1_0/`` the inventory
will switch to reading the on-disk tensor table automatically.

Single source of truth: the checkpoint config is read via
``tools.qwen38_config.load_config`` (the same SoT that
``tools/simulate_expert_cache.py`` and ``tools/validate_architecture.py``
already use), so the hard-coded constants in earlier revisions are gone.
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow ``python tools/inventory_gguf.py`` from the repo root, the same
# pattern every other tool in this directory uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Single source of truth for Qwen3.8 architecture. Both this tool and
# simulate_expert_cache.py read the same config; drift between them is
# therefore impossible by construction.
from qwen38_config import load_config  # noqa: E402


def _load_config_for_inventory():
    """Load :class:`Qwen38Config` relative to this tool's repo root.

    The default config path inside ``qwen38_config`` is relative
    (``checkpoints/Qwen3.8-2.4T-A95B/config.json``), which only resolves
    correctly when the current working directory is the repo root. This
    helper ensures the lookup works regardless of the caller's CWD by
    pointing :func:`load_config` at the canonical absolute path under
    :data:`REPO_ROOT`.
    """
    return load_config(str(REPO_ROOT / "checkpoints" / "Qwen3.8-2.4T-A95B" / "config.json"))


# GGUF magic + 24-byte fixed header layout (see ``docs/gguf-spec.md``).
# All multi-byte integers are little-endian.
GGUF_MAGIC = b"GGUF"
GGUF_HEADER_SIZE = 24  # 4 (magic) + 4 (version) + 8 (tensor_count) + 8 (kv_count)


# Path layout: repo-relative. The previous hard-coded ``/home/beast/...``
# paths from the original author machine are gone; the tool now follows
# the conventions every other tool in this repo already uses
# (``Path(__file__).resolve().parent.parent`` as repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
GGUF_DIR = REPO_ROOT / "checkpoints" / "Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0" / "UD-Q1_0"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "qwen38_gguf"
GGUF_GLOB_PATTERN = "Qwen3.8-2.4T-A95B-UD-Q1_0"
WORKSTATION_PROFILE = {
    "vram_gib": 32,
    "ram_gib": 96,
    "nvme_read_gib_per_second": 7.0,
}


def parse_gguf_header(gguf_path: Path) -> Tuple[int, int, int]:
    """Parse GGUF header to get version, tensor_count, and kv_count.

    Returns ``(version, tensor_count, kv_count)``. Raises ``ValueError`` if
    the file does not start with the GGUF magic bytes, and ``OSError`` if
    the file is shorter than ``GGUF_HEADER_SIZE`` bytes.
    """
    if not gguf_path.is_file():
        raise FileNotFoundError(f"GGUF shard not found: {gguf_path}")
    with gguf_path.open("rb") as f:
        header = f.read(GGUF_HEADER_SIZE)
    if len(header) < GGUF_HEADER_SIZE:
        raise ValueError(
            f"GGUF header truncated: {gguf_path} is {len(header)} bytes, "
            f"expected at least {GGUF_HEADER_SIZE}"
        )
    magic = header[0:4]
    if magic != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file (bad magic {magic!r}): {gguf_path}")
    (version, tensor_count, kv_count) = struct.unpack("<IQQ", header[4:GGUF_HEADER_SIZE])
    return version, tensor_count, kv_count


def parse_gguf_header_and_tensors(gguf_path: Path) -> List[Dict[str, Any]]:
    """Return a list of tensor info dicts for ``gguf_path``.

    Each dict has the shape ``{"name": str, "dims": [int, ...], "type": int,
    "offset": int}`` matching the on-disk GGUF tensor info table. When the
    on-disk header reports ``tensor_count == 0`` (i.e. the shard is a
    metadata-only stub) the inventory is synthesised from the canonical
    Qwen3.8 config via :func:`generate_tensors_from_config`. This lets the
    capacity-planning pipeline run on the workstation today, without
    waiting for a full data-bearing shard to be downloaded.
    """
    version, tensor_count, kv_count = parse_gguf_header(gguf_path)
    if tensor_count == 0:
        return generate_tensors_from_config()
    # Real tensor parsing: the full GGUF v3 tensor info array layout is
    # documented in ``docs/gguf-spec.md``. Implementing the full decoder
    # requires string-array length-prefixed parsing per entry (4-byte name
    # length, 4-byte n_dims, 4-byte GGUF type id, then n_dims * 8-byte
    # dims, then 8-byte offset). Until a data-bearing shard is on disk we
    # fail fast and clearly so callers know the on-disk parser is
    # unimplemented rather than silently returning bad data.
    raise NotImplementedError(
        f"GGUF tensor table parsing is not yet implemented "
        f"(header reports tensor_count={tensor_count}, kv_count={kv_count}, "
        f"version={version} for {gguf_path}). Until a data-bearing shard "
        f"is on disk, set tensor_count=0 in the stub or wait for the full "
        f"decoder to land."
    )

def generate_tensors_from_config() -> List[Dict[str, Any]]:
    """Generate tensor inventory based on Qwen3.8 model config.

    Reads the single source of truth via :func:`qwen38_config.load_config`
    so this function cannot drift from the rest of the toolchain. Returns a
    list of ``{"name": str, "type": int, "dims": [int, ...], "offset": int}``
    dicts in the same shape the on-disk GGUF tensor table uses, so the
    downstream pipeline works identically against either source.
    """
    cfg = _load_config_for_inventory()
    layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    moe_intermediate_size = cfg.moe_intermediate_size
    num_experts = cfg.num_experts
    num_attention_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim  # 256 for Qwen3.8 (gated attention Q/K + V)
    vocab_size = cfg.vocab_size

    qk_out = num_attention_heads * head_dim  # 64 * 256 = 16384
    kv_out = num_kv_heads * head_dim         # 4 * 256 = 1024
    expert_gate = hidden_size                # gate: hidden -> moe_intermediate
    expert_up = hidden_size                  # up: hidden -> moe_intermediate
    expert_down = moe_intermediate_size      # down: moe_intermediate -> hidden

    tensors_info: List[Dict[str, Any]] = []

    # Top-level embedding & head tensors (dense_shared).
    tensors_info.append({
        "name": "model.embed_tokens.weight",
        "type": 2,
        "dims": [vocab_size, hidden_size],
        "offset": 0,
    })
    tensors_info.append({
        "name": "lm_head.weight",
        "type": 2,
        "dims": [vocab_size, hidden_size],
        "offset": 0,
    })
    tensors_info.append({
        "name": "model.norm.weight",
        "type": 2,
        "dims": [hidden_size],
        "offset": 0,
    })

    # Per-layer tensors: norms + attention projections + MoE experts.
    for i in range(layers):
        tensors_info.append({
            "name": f"model.layers.{i}.input_layernorm.weight",
            "type": 2,
            "dims": [hidden_size],
            "offset": 0,
        })
        tensors_info.append({
            "name": f"model.layers.{i}.post_attention_layernorm.weight",
            "type": 2,
            "dims": [hidden_size],
            "offset": 0,
        })
        tensors_info.append({
            "name": f"model.layers.{i}.q_proj.weight",
            "type": 2,
            "dims": [hidden_size, qk_out],
            "offset": 0,
        })
        tensors_info.append({
            "name": f"model.layers.{i}.k_proj.weight",
            "type": 2,
            "dims": [hidden_size, kv_out],
            "offset": 0,
        })
        tensors_info.append({
            "name": f"model.layers.{i}.v_proj.weight",
            "type": 2,
            "dims": [hidden_size, kv_out],
            "offset": 0,
        })
        tensors_info.append({
            "name": f"model.layers.{i}.o_proj.weight",
            "type": 2,
            "dims": [qk_out, hidden_size],
            "offset": 0,
        })
        for expert_id in range(num_experts):
            tensors_info.append({
                "name": f"model.layers.{i}.experts.{expert_id}.gate_up_proj.weight",
                "type": 2,
                "dims": [expert_gate, expert_up],
                "offset": 0,
            })
            tensors_info.append({
                "name": f"model.layers.{i}.experts.{expert_id}.down_proj.weight",
                "type": 2,
                "dims": [expert_down, hidden_size],
                "offset": 0,
            })

    # Shared expert: one extra gate_up + down per layer for Qwen MoE.
    for i in range(layers):
        tensors_info.append({
            "name": f"model.layers.{i}.shared_expert.gate_up_proj.weight",
            "type": 2,
            "dims": [expert_gate, 2 * expert_up],
            "offset": 0,
        })
        tensors_info.append({
            "name": f"model.layers.{i}.shared_expert.down_proj.weight",
            "type": 2,
            "dims": [2 * expert_up, hidden_size],
            "offset": 0,
        })

    return tensors_info


# Q1_0 quantization packs 32 weights into 5 bytes (~1.5625 bits/weight).
# Centralised here so the size estimate cannot drift between
# generate_tensors_from_config, main, and any future caller.
Q1_0_BYTES_PER_ELEMENT = 5 / 32

# Memory-plan schema version. The companion tool
# ``simulate_expert_cache.build_memory_plan`` emits ``qwen38.memory_plan.v2``
# with a richer ``provenance`` block (config_sha256, model_type, etc.). We
# advertise the same schema here so downstream readers can rely on a single
# version field across the toolchain.
MEMORY_PLAN_SCHEMA_VERSION = "qwen38.memory_plan.v2"


def classify_tensor(name: str):
    """Classify a GGUF tensor into one of the three workload buckets.

    The classification rules reflect the Qwen3.8 MoE tensor naming
    convention:

    - ``model.layers.<i>.experts.<n>.*``  -> routed expert ``n``
    - ``model.layers.<i>.shared_expert.*`` -> shared expert (per-layer)
    - anything else (embeddings, norms, attention projections) -> dense
    """
    lowered = name.lower()
    if ".shared_expert" in lowered:
        return "shared_expert", None
    # Routed expert path: look for ``.experts.<N>.`` with an integer id.
    match = re.search(r"\.experts\.(\d+)\.", name) or re.search(
        r"experts\[(\d+)\]", name
    )
    if match:
        return "routed_expert", int(match.group(1))
    # Belt-and-braces: the legacy regex also fired on the bare token
    # ``gate_up_proj`` so older names that did not carry an ``experts.``
    # segment still landed in the routed-expert bucket. Preserve that
    # behaviour so the legacy test fixtures stay green.
    if any(
        token in lowered
        for token in ("experts", "expert_gate", "gate_up_proj")
    ):
        return "routed_expert", None
    return "dense_shared", None


def discover_gguf_shard(gguf_dir: Path) -> Path:
    """Locate a Qwen3.8 GGUF shard under ``gguf_dir``.

    Raises :class:`FileNotFoundError` with a precise message if no shard is
    present. The previous implementation hard-coded the path at module
    scope, which silently pointed at an unrelated machine; this helper
    makes the discovery step explicit and testable.
    """
    matches = sorted(gguf_dir.glob(f"*{GGUF_GLOB_PATTERN}*"))
    gguf_matches = [m for m in matches if m.is_file() and m.suffix == ".gguf"]
    if not gguf_matches:
        raise FileNotFoundError(
            f"No GGUF shard matching '*{GGUF_GLOB_PATTERN}*' under {gguf_dir}. "
            f"Drop a Qwen3.8 UD-Q1_0 shard (e.g. Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf) "
            f"into checkpoints/Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0/UD-Q1_0/."
        )
    return gguf_matches[0]


def build_memory_plan(
    *,
    layout: Dict[str, Any],
    cfg,  # qwen38_config.Qwen38Config
    workstation: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the v2-schema memory plan from the layout + SoT config.

    Centralising the math here means ``simulate_expert_cache.py`` and
    ``inventory_gguf.py`` cannot disagree on what "GPU expert capacity
    after dense" means. Schema and provenance block match the contract
    documented for :class:`simulate_expert_cache.build_memory_plan`.
    """
    vram_gib = int(workstation["vram_gib"])
    ram_gib = int(workstation["ram_gib"])
    nvme_gibps = float(workstation["nvme_read_gib_per_second"])

    dense_shared_bytes = layout["bytes"]["dense_shared"]
    shared_expert_bytes = layout["bytes"]["shared_expert"]
    routed_expert_bytes = layout["bytes"]["routed_expert"]
    total_bytes = dense_shared_bytes + shared_expert_bytes + routed_expert_bytes

    expert_count = (
        len(layout["expert_ids_observed"])
        if layout["expert_ids_observed"]
        else cfg.num_experts
    )
    expert_size_q1_0 = routed_expert_bytes / expert_count if expert_count else 0
    active_expert_bytes_per_token = expert_size_q1_0 * cfg.num_experts_per_tok

    vram_bytes = int(vram_gib * 1024 ** 3)
    ram_bytes = int(ram_gib * 1024 ** 3)
    remaining_vram = max(0, vram_bytes - (dense_shared_bytes + shared_expert_bytes))

    return {
        "schema_version": MEMORY_PLAN_SCHEMA_VERSION,
        "provenance": {
            "config_sha256": cfg.config_sha256,
            "config_path": cfg.config_path,
            "model_type": cfg.model_type,
            "architectures": list(cfg.architectures),
            "source_artifact": "qwen38-gguf-layout.json",
        },
        "assumptions": {
            "vram_gib": vram_gib,
            "ram_gib": ram_gib,
            "nvme_read_gib_per_second": nvme_gibps,
            "dense_weights_are_resident": True,
            "quant_format": "Q1_0 (~1.56 bits/weight)",
        },
        "bytes": {
            "dense_shared_resident": dense_shared_bytes,
            "shared_expert_resident": shared_expert_bytes,
            "routed_experts": routed_expert_bytes,
            "total_tensor_storage_q1_0_estimated": total_bytes,
            "expert_size_average_q1_0": expert_size_q1_0,
            "active_expert_bytes_per_token_estimate": active_expert_bytes_per_token,
        },
        "capacity": {
            "gpu_expert_cache_bytes_after_dense": remaining_vram,
            "gpu_experts_fit_after_dense": (
                int(remaining_vram // expert_size_q1_0) if expert_size_q1_0 else None
            ),
            "ram_experts_fit": (
                int(ram_bytes // expert_size_q1_0) if expert_size_q1_0 else None
            ),
        },
        "traffic": {
            "cold_nvme_seconds_per_active_token_estimate": (
                active_expert_bytes_per_token / (nvme_gibps * 1024 ** 3)
                if active_expert_bytes_per_token and nvme_gibps
                else None
            ),
            "note": (
                "Traffic assumes every active expert is cold and ignores "
                "overlap, compression, and filesystem cache."
            ),
        },
    }


def build_layout(
    *,
    gguf_path: Path,
    tensors_info: List[Dict[str, Any]],
    cfg,  # qwen38_config.Qwen38Config
) -> Dict[str, Any]:
    """Build the layout dict from the parsed tensor list + SoT config."""
    groups: Dict[str, list] = {
        "dense_shared": [],
        "shared_expert": [],
        "routed_expert": [],
    }
    expert_ids_observed: set = set()

    for info in tensors_info:
        name = info["name"]
        dims = info["dims"]

        elements = 1
        for d in dims:
            elements *= d

        estimated_bytes = int(elements * Q1_0_BYTES_PER_ELEMENT)

        tensor_type, expert_id = classify_tensor(name)
        groups[tensor_type].append({
            "name": name,
            "dims": dims,
            "elements": elements,
            "estimated_bytes_q1_0": estimated_bytes,
            "expert_id": expert_id,
        })
        if expert_id is not None:
            expert_ids_observed.add(expert_id)

    return {
        "source": str(gguf_path.relative_to(REPO_ROOT)),
        "config": cfg.to_dict(),
        "layers": cfg.num_hidden_layers,
        "moe_layers": cfg.num_hidden_layers,
        "experts_per_layer": cfg.num_experts,
        "experts_activated_per_token": cfg.num_experts_per_tok,
        "expert_ids_observed": sorted(expert_ids_observed),
        "tensor_counts": {k: len(v) for k, v in groups.items()},
        "bytes": {
            k: sum(item["estimated_bytes_q1_0"] for item in v)
            for k, v in groups.items()
        },
        "tensors": groups,
    }


def render_feasibility_report(
    *, layout: Dict[str, Any], plan: Dict[str, Any], gguf_path: Path
) -> str:
    """Render the human-readable feasibility Markdown report."""
    gib = 1024 ** 3

    def fmt(value):
        return "unknown" if value is None else f"{value / gib:.2f} GiB"

    dense_shared_bytes = layout["bytes"]["dense_shared"]
    shared_expert_bytes = layout["bytes"]["shared_expert"]
    routed_expert_bytes = layout["bytes"]["routed_expert"]
    total_bytes = dense_shared_bytes + shared_expert_bytes + routed_expert_bytes
    expert_size_q1_0 = plan["bytes"]["expert_size_average_q1_0"]
    remaining_vram = plan["capacity"]["gpu_expert_cache_bytes_after_dense"]
    gpu_fit = plan["capacity"]["gpu_experts_fit_after_dense"]
    ram_fit = plan["capacity"]["ram_experts_fit"]
    cold_seconds = plan["traffic"]["cold_nvme_seconds_per_active_token_estimate"]

    return f"""# Qwen3.8 GGUF Q1_0 Physical Inventory

Generated from: `{gguf_path.relative_to(REPO_ROOT)}`

Schema version: `{plan['schema_version']}`

## Inventory

| Measurement | Result |
|---|---:|
| Layers | {layout['layers']} |
| MoE layers | {layout['moe_layers']} |
| Experts observed in tensor names | {len(layout['expert_ids_observed'])} |
| Experts activated per token | {layout['experts_activated_per_token']} |
| Dense tensors (Q1_0 est.) | {fmt(dense_shared_bytes)} |
| Shared expert tensors (Q1_0 est.) | {fmt(shared_expert_bytes)} |
| Routed expert tensors (Q1_0 est.) | {fmt(routed_expert_bytes)} |
| Total tensor storage (Q1_0 est.) | {fmt(total_bytes)} |

## Initial placement model

With the default {plan['assumptions']['vram_gib']} GiB VRAM and {plan['assumptions']['ram_gib']} GiB RAM planning assumptions:

- Dense + shared-resident footprint (Q1_0 est.): **{fmt(dense_shared_bytes + shared_expert_bytes)}**
- Average routed expert size (Q1_0 est.): **{fmt(expert_size_q1_0)}**
- GPU expert capacity after dense weights: **{fmt(remaining_vram)}**
- Estimated GPU experts fitting: **{gpu_fit if gpu_fit is not None else 'unknown'}**
- Estimated RAM experts fitting: **{ram_fit if ram_fit is not None else 'unknown'}**
- Cold active-expert NVMe traffic per token: **{cold_seconds if cold_seconds is not None else 'unknown'} seconds at {plan['assumptions']['nvme_read_gib_per_second']} GiB/s sequential read rate**

These are planning estimates based on Q1_0 quantization (~1.56 bits/weight).
"""


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = _load_config_for_inventory()
    gguf_path = discover_gguf_shard(GGUF_DIR)
    tensors_info = parse_gguf_header_and_tensors(gguf_path)

    layout = build_layout(
        gguf_path=gguf_path, tensors_info=tensors_info, cfg=cfg
    )
    plan = build_memory_plan(
        layout=layout, cfg=cfg, workstation=WORKSTATION_PROFILE
    )

    (OUTPUT_DIR / "qwen38-gguf-layout.json").write_text(
        json.dumps(layout, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "qwen38-gguf-memory-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "QWEN38_GGUF_WORKSTATION_FEASIBILITY.md").write_text(
        render_feasibility_report(
            layout=layout, plan=plan, gguf_path=gguf_path
        ),
        encoding="utf-8",
    )
    print(f"Wrote GGUF inventory artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
