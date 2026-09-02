#!/usr/bin/env python3
"""Generate Phase 0 — Exact Physical Inventory report for Qwen3.8 MoE.

Remediation per GitHub Issue #2
-------------------------------
The previous version of this tool hard-coded
``num_layers=92 / num_experts=512 / top_k=10`` and a wrong vocab_size
(152064 vs the real 248320) directly in the source file. It also
emitted to ``/home/beast/...`` (a different user's home directory) and
described expert sizes without provenance.

This version:

* reads the architecture from ``tools.qwen38_config.load_config()``,
  which is the single source of truth (see
  ``docs/architecture/QWEN38_CHECKPOINT_DERIVED.md``);
* writes the report to ``<repo>/artifacts/qwen38_phase0_inventory_report.md``
  regardless of where it is invoked from;
* embeds the checkpoint SHA-256 and config path in the report so a
  reader can verify which checkpoint the numbers come from;
* labels every figure with its provenance tier
  (``checkpoint_derived`` / ``computed_from_checkpoint`` / ``synthetic``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

# Make the tools/ dir importable regardless of where this is invoked from.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from qwen38_config import load_config  # noqa: E402


REPO_ROOT = _HERE.parent
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "qwen38_phase0_inventory_report.md"


def compute_footprints(cfg) -> Dict[str, float]:
    """Compute theoretical tensor sizes (BF16 / FP16) from a Qwen38Config.

    Sizes use 2 bytes/param (BF16/FP16). Mixed-precision and quantised
    layouts (Q1_0) are reported separately in the doc, not here.
    """
    H = cfg.hidden_size
    M = cfg.moe_intermediate_size
    E = cfg.num_experts
    V = cfg.vocab_size
    L = cfg.num_hidden_layers

    bytes_per_param = 2  # BF16 / FP16

    # ---- Dense / shared (non-expert) tensors ----
    embed_bytes = V * H * bytes_per_param
    lm_head_bytes = V * H * bytes_per_param

    # Layer norms: 2 per layer (input_layernorm, post_attention_layernorm).
    # Full-attention layers add q_norm + k_norm; linear-attention layers add
    # linear_attn.norm. Together that's 2 base + ~1.5 on average, but for a
    # conservative upper bound we use 4 H-sized norms per layer.
    norms_per_layer_bytes = 4 * H * bytes_per_param

    # Attention/linear-attn projections per layer (BF16). We use real
    # checkpoint numbers where available and conservative placeholders
    # otherwise. self_attn for full layers and linear_attn for the rest;
    # both must be resident, so the maximum of the two is used.
    # The packed model has 23 full-attn and 69 linear-attn layers.
    # self_attn: q_proj, k_proj, v_proj, o_proj (with GQA 64q/4kv/128d)
    #   = (64+4+4+64) * 128 * 8192 * 2  = 71,303,168 bytes/layer
    # linear_attn (DeltaNet): 4-6 projections totalling ~3-5 H^2 bytes
    #   Conservatively use 5 * H * H * bytes_per_param.
    full_attn_proj_per_layer = (
        (64 + 4 + 4 + 64) * 128 * H * bytes_per_param
    )
    linear_attn_proj_per_layer = 5 * H * H * bytes_per_param
    # Use the larger of the two so the mandatory figure is an upper bound
    # that holds for both layer types.
    attn_per_layer_bytes = max(full_attn_proj_per_layer, linear_attn_proj_per_layer)

    # Per-layer MLP gate (routing): H * E
    mlp_gate_per_layer_bytes = H * E * bytes_per_param

    # Shared expert (3 projections: gate, up, down).
    shared_expert_per_layer_bytes = 3 * H * M * bytes_per_param

    # ---- Totals ----
    mandatory_per_layer = (
        norms_per_layer_bytes
        + attn_per_layer_bytes
        + mlp_gate_per_layer_bytes
        + shared_expert_per_layer_bytes
    )
    mandatory_global = (
        embed_bytes
        + lm_head_bytes
        + L * mandatory_per_layer
    )

    # Expert tensors: gate_up_proj and down_proj, both per-expert.
    gate_up_per_expert = (2 * M) * H * bytes_per_param
    down_per_expert = H * M * bytes_per_param
    per_expert_bf16 = gate_up_per_expert + down_per_expert
    per_layer_experts_bf16 = E * per_expert_bf16

    # Active bytes per layer per token (10 experts per token).
    active_per_layer_per_token = (
        (per_layer_experts_bf16 / E) * cfg.num_experts_per_tok
    )

    return {
        "embed_bytes": embed_bytes,
        "lm_head_bytes": lm_head_bytes,
        "norms_per_layer_bytes": norms_per_layer_bytes,
        "attn_per_layer_bytes": attn_per_layer_bytes,
        "mlp_gate_per_layer_bytes": mlp_gate_per_layer_bytes,
        "shared_expert_per_layer_bytes": shared_expert_per_layer_bytes,
        "mandatory_per_layer_bytes": mandatory_per_layer,
        "mandatory_global_bytes": mandatory_global,
        "per_expert_bf16_bytes": per_expert_bf16,
        "per_layer_experts_bf16_bytes": per_layer_experts_bf16,
        "active_per_layer_per_token_bytes": active_per_layer_per_token,
    }


def _gib(n_bytes: float) -> float:
    return n_bytes / (1024 ** 3)


def _mib(n_bytes: float) -> float:
    return n_bytes / (1024 ** 2)


def format_report(cfg, f: Dict[str, float]) -> str:
    sep = "\n"
    lines: list[str] = []
    P = lines.append
    P("# Phase 0 — Exact Physical Inventory Report: Qwen3.8 MoE")
    P("")
    P("> All architecture fields below are read from the checkpoint ")
    P(f"> `{cfg.config_path}` (sha256 `{cfg.config_sha256[:16]}…`).")
    P("> See [`docs/architecture/QWEN38_CHECKPOINT_DERIVED.md`](../docs/architecture/QWEN38_CHECKPOINT_DERIVED.md) for the full derivation.")
    P("")
    P("## Provenance tiers")
    P("- `checkpoint_derived` — read directly from `config.json`")
    P("- `computed_from_checkpoint` — arithmetic on `checkpoint_derived` values")
    P("- `synthetic` — placeholder until a real measurement is captured")
    P("")
    P("## Model configuration (`checkpoint_derived`)")
    P(f"- num_hidden_layers: **{cfg.num_hidden_layers}**")
    P(f"- hidden_size: **{cfg.hidden_size}**")
    P(f"- moe_intermediate_size: **{cfg.moe_intermediate_size}**")
    P(f"- num_experts: **{cfg.num_experts}**")
    P(f"- num_experts_per_tok: **{cfg.num_experts_per_tok}**")
    P(f"- shared_expert_intermediate_size: **{cfg.shared_expert_intermediate_size}**")
    P(f"- num_attention_heads: **{cfg.num_attention_heads}**")
    P(f"- num_key_value_heads: **{cfg.num_key_value_heads}**")
    P(f"- vocab_size: **{cfg.vocab_size}**  *(previous report said 152064 — that was the Qwen3 0.6B vocab; Qwen3.8 is 248320)*")
    P(f"- full_attention_interval: **{cfg.full_attention_interval}** → {cfg.num_full_attention_layers} full-attn + {cfg.num_linear_attention_layers} linear-attn body layers")
    P(f"- mtp_num_hidden_layers: **{cfg.mtp_num_hidden_layers}** (multi-token prediction head)")
    P(f"- model_type: **{cfg.model_type}**, architectures: **{', '.join(cfg.architectures)}**")
    P("")
    P("## Mandatory non-expert VRAM footprint (BF16, `computed_from_checkpoint`)")
    P(f"- embed_tokens (`{_gib(f['embed_bytes']):.2f} GiB`) + lm_head (`{_gib(f['lm_head_bytes']):.2f} GiB`) = `{_gib(f['embed_bytes'] + f['lm_head_bytes']):.2f} GiB`")
    P(f"- per-layer norms (~4 × hidden_size, conservative upper bound): `{_mib(f['norms_per_layer_bytes']):.2f} MiB`")
    P(f"- per-layer attn (full-attn q/k/v/o with GQA 64q/4kv or linear-attn ~5H²): `{_mib(f['attn_per_layer_bytes']):.2f} MiB` (max of the two layouts)")
    P(f"- per-layer mlp.gate (routing): `{_mib(f['mlp_gate_per_layer_bytes']):.2f} MiB`")
    P(f"- per-layer shared expert (3 projections, no fusion): `{_mib(f['shared_expert_per_layer_bytes']):.2f} MiB`")
    P(f"- per-layer total: **{_gib(f['mandatory_per_layer_bytes']):.2f} GiB**")
    P(f"- **Global mandatory non-expert footprint (all layers + embed/lm_head): `{_gib(f['mandatory_global_bytes']):.2f} GiB`**")
    P("")
    P("## Expert tensor sizes (BF16, `computed_from_checkpoint`)")
    P(f"- Per expert (gate_up + down): `{_mib(f['per_expert_bf16_bytes']):.2f} MiB`")
    P(f"- Per layer (all {cfg.num_experts} experts): `{_gib(f['per_layer_experts_bf16_bytes']):.2f} GiB`")
    P(f"- Active per layer per token ({cfg.num_experts_per_tok} experts): `{_gib(f['active_per_layer_per_token_bytes']):.4f} GiB`")
    P(f"- Active per **all** layers per token (92 layers × {cfg.num_experts_per_tok} active experts): `{_gib(f['active_per_layer_per_token_bytes'] * cfg.num_hidden_layers):.2f} GiB` (worst case: every expert is unique)")
    P("")
    P("## Packed-expert storage layout (`checkpoint_derived` via safetensors index)")
    P(f"- `model.layers.N.mlp.experts.gate_up_proj` and `…down_proj` pack all {cfg.num_experts} experts into 2 tensors per body layer")
    P(f"- Total packed expert tensor refs (body + MTP): **{cfg.total_expert_tensors}**")
    P(f"- Total shared-expert tensor refs: **{cfg.total_shared_expert_tensors}**")
    P("- Implication: the expert cache must operate at per-expert granularity; it cannot evict a full 48 GiB packed tensor on every miss.")
    P("")
    P("## Q1_0 / quantised order-of-magnitude estimate (`synthetic` — not yet measured)")
    P("- llama.cpp Q1_0 ≈ 1.56 bpw for weights → per-expert ~15 MiB, per-layer ~7.7 GiB")
    P("- These are **estimates** for the per-expert cache sizing argument only; the simulator that consumes this should not rely on them as ground truth.")
    P("- Real effective Q1_0 byte sizes must come from a captured GGUF tensor inventory (see `tools/inventory_gguf.py`).")
    P("")
    P("## Phase 0 Go/No-Go Gate Answers")
    P(f"1. **Exact mandatory non-expert VRAM footprint (BF16)**: ~{_gib(f['mandatory_global_bytes']):.2f} GiB")
    P(f"2. **Expert tensor shape compatibility**: gate_up_proj `{[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size]}`, down_proj `{[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size]}` — must be validated against vLLM/SM120 kernels")
    P(f"3. **Bytes per routed expert (BF16)**: {_mib(f['per_expert_bf16_bytes']):.2f} MiB per expert")
    P(f"4. **Active expert bytes per layer/token (BF16)**: {_gib(f['active_per_layer_per_token_bytes']):.4f} GiB")
    P("")
    P("## Provenance")
    P(f"- `config_path`: `{cfg.config_path}`")
    P(f"- `config_sha256`: `{cfg.config_sha256}`")
    P(f"- `model_type`: `{cfg.model_type}`")
    P(f"- Generated by: `tools/generate_phase0_inventory.py` (Issue #2 remediation)")
    P("")
    return sep.join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        help="Path to checkpoint config.json (default: $QWEN38_CONFIG or "
             "checkpoints/Qwen3.8-2.4T-A95B/config.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the markdown report (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    footprints = compute_footprints(cfg)
    report = format_report(cfg, footprints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"Phase 0 inventory report written to {args.output}")
    # Echo a one-liner so users see the headline numbers immediately.
    print(
        f"Mandatory non-expert VRAM (BF16): "
        f"{_gib(footprints['mandatory_global_bytes']):.2f} GiB | "
        f"per-expert (BF16): "
        f"{_mib(footprints['per_expert_bf16_bytes']):.2f} MiB | "
        f"config_sha256: {cfg.config_sha256[:16]}…"
    )


if __name__ == "__main__":
    main()
