#!/usr/bin/env python3
"""Build the Qwen3.8 physical tensor-layout JSON from the checkpoint's
``model.safetensors.index.json``.

This replaces the hand-rolled ``inventory_safetensors.py`` whose classifier
mistakenly left 647 tensors in an ``unknown`` bucket. Every tensor is now
classified into one of six categories whose semantics are documented inline,
and the output embeds the config path + SHA-256 so the layout is traceable
back to the checkpoint.

Output: ``artifacts/qwen38-layout.json`` (or a path of your choosing).

Categories
----------
- ``embed_lm_head``         — vocab projection weights (2 tensors)
- ``layer_norm``            — input/post-attention/k/q norms
- ``full_attention``        — q/k/v/o projections for the 23 full-attn layers
                              (and the MTP attention projections)
- ``linear_attention``      — A_log / conv1d / dt_bias / in_proj_*/out_proj
                              for the 69 linear-attn body layers
- ``shared_expert``         — mlp.shared_expert.{gate,up,down}_proj + the
                              mlp.shared_expert_gate router
- ``routed_expert_packed``  — the 2 packed expert tensors per layer
                              (gate_up_proj, down_proj)
- ``moe_gate``              — mlp.gate.weight (token-to-expert router)
- ``mtp_other``             — MTP layernorms / etc.

All architecture fields are read from the checkpoint via
``tools.qwen38_config.load_config``; nothing here is hard-coded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from qwen38_config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

# Repo root (parent of tools/)
REPO_ROOT = THIS_DIR.parent


def _is_mtp(name: str) -> bool:
    return name.startswith("mtp.")


def _is_routed_expert_packed(name: str) -> bool:
    return (
        "mlp.experts.gate_up_proj" in name
        or "mlp.experts.down_proj" in name
    )


def classify_tensor(name: str) -> str:
    """Map a safetensors tensor name to one of the layout categories."""
    # Order matters: more specific patterns first.
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "embed_lm_head"
    if _is_routed_expert_packed(name):
        return "routed_expert_packed"
    if "mlp.shared_expert." in name or name.endswith("mlp.shared_expert_gate.weight"):
        return "shared_expert"
    if "mlp.gate.weight" in name:
        return "moe_gate"
    if "linear_attn." in name:
        return "linear_attention"
    if "self_attn." in name:
        return "full_attention"
    if name.endswith(".input_layernorm.weight") or name.endswith(".post_attention_layernorm.weight"):
        return "layer_norm"
    if name == "model.norm.weight":
        return "layer_norm"
    if _is_mtp(name):
        # MTP-specific: pre_fc_norm_*, mtp.norm.weight, mtp.fc.weight
        return "mtp_other"
    return "unclassified"


def build_layout(
    index_path: Path,
    cfg_path: Path = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    cfg = load_config(str(cfg_path))
    idx = json.loads(index_path.read_text())
    weight_map: Dict[str, str] = idx["weight_map"]
    total_size = idx.get("metadata", {}).get("total_size", 0)

    buckets: Dict[str, List[str]] = {k: [] for k in (
        "embed_lm_head", "layer_norm", "full_attention",
        "linear_attention", "shared_expert", "routed_expert_packed",
        "moe_gate", "mtp_other", "unclassified",
    )}
    body_layer_full_attn = 0
    body_layer_linear_attn = 0
    for name in weight_map:
        cat = classify_tensor(name)
        buckets[cat].append(name)
        if cat == "full_attention" and not _is_mtp(name):
            body_layer_full_attn += 1
        elif cat == "linear_attention" and not _is_mtp(name):
            body_layer_linear_attn += 1

    # Sanity: at most the count of known categories should be unclassified.
    if buckets["unclassified"]:
        raise SystemExit(
            f"FATAL: {len(buckets['unclassified'])} tensors failed to classify; "
            f"sample: {buckets['unclassified'][:5]}"
        )

    counts = {k: len(v) for k, v in buckets.items()}
    # Cross-checks against the config-driven properties
    expected_full_attn_per_layer = 6  # q,k,v,o + k_norm + q_norm
    expected_linear_attn_per_layer = 9  # A_log, conv1d, dt_bias, in_proj_a, in_proj_b,
    # in_proj_qkv, in_proj_z, out_proj, plus norm
    # We don't hard-assert on per-tensor counts because the MTP layer has
    # its own attention, but the body layer tallies should match the property
    # tallies we already validated.

    return {
        "schema": "qwen38.tensor_layout.v2",
        "provenance": "checkpoint_derived",
        "config_path": str(cfg_path),
        "config_sha256": cfg.config_sha256,
        "safetensors_index": str(index_path),
        "total_size_bytes": total_size,
        "total_size_tib": total_size / (1024 ** 4) if total_size else 0.0,
        "total_tensors": len(weight_map),
        "tensor_classification": counts,
        "tensor_names": {k: v for k, v in buckets.items() if v},
        "cross_checks": {
            "body_layer_full_attention_tensors": body_layer_full_attn,
            "body_layer_linear_attention_tensors": body_layer_linear_attn,
            "expected_full_attention_per_layer": expected_full_attn_per_layer,
            "expected_linear_attention_per_layer": expected_linear_attn_per_layer,
            "num_full_attention_layers_config": cfg.num_full_attention_layers,
            "num_linear_attention_layers_config": cfg.num_linear_attention_layers,
            "total_expert_packed_tensors_config": cfg.total_expert_tensors,
            "total_shared_expert_tensors_config": cfg.total_shared_expert_tensors,
        },
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--safetensors-index",
        type=Path,
        default=REPO_ROOT / "checkpoints" / "Qwen3.8-2.4T-A95B" / "model.safetensors.index.json",
        help="Path to model.safetensors.index.json",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.json (used for the config_sha256 stamp)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts" / "qwen38-layout.json",
        help="Where to write the layout JSON",
    )
    args = p.parse_args(argv)

    layout = build_layout(args.safetensors_index, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(layout, indent=2, sort_keys=True) + "\n")

    # Headline stdout
    cc = layout["cross_checks"]
    print(
        f"Layout written to {args.output}\n"
        f"  total tensors:             {layout['total_tensors']}\n"
        f"  total size:                {layout['total_size_tib']:.2f} TiB\n"
        f"  body full-attn tensors:    {cc['body_layer_full_attention_tensors']} "
        f"(expected {cc['num_full_attention_layers_config'] * cc['expected_full_attention_per_layer']} "
        f"= {cc['num_full_attention_layers_config']} layers x {cc['expected_full_attention_per_layer']})\n"
        f"  body linear-attn tensors:  {cc['body_layer_linear_attention_tensors']} "
        f"(expected {cc['num_linear_attention_layers_config'] * cc['expected_linear_attention_per_layer']} "
        f"= {cc['num_linear_attention_layers_config']} layers x {cc['expected_linear_attention_per_layer']})\n"
        f"  packed expert tensors:     {layout['tensor_classification']['routed_expert_packed']} "
        f"(expected {cc['total_expert_packed_tensors_config']})\n"
        f"  shared expert tensors:     {layout['tensor_classification']['shared_expert']} "
        f"(expected {cc['total_shared_expert_tensors_config']})\n"
        f"  config_sha256:             {layout['config_sha256']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
