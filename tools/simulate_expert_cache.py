#!/usr/bin/env python3
"""
Phase 0 Tool: simulate_expert_cache.py

Regenerates ``artifacts/qwen38-memory-plan.json`` from the authoritative
checkpoint config (``tools.qwen38_config``) — see Issue #2 acceptance
criteria AC1, AC2, AC6 and AC10.

This tool is intentionally simple: it does not run a routing simulation.
The actual cache-miss / hit-rate numbers come from
``tools/trace_qwen38_routing.py`` (see
``artifacts/qwen38_routing_trace_metrics.json``) — that artifact is the
authoritative source for hit rates. This file only records the *capacity*
math (how many experts fit in a given budget) and points downstream
consumers at the routing-trace artifact for the *behaviour* numbers.

The Q1_0 / Q4_K_M / Q8_0 byte-per-expert values are computed from the
checkpoint weight count per expert (``3 * hidden * moe_int``) at the
conventional average bits-per-weight for those GGUF quantizations. The
old hard-coded ``BYTES_PER_ROUTED_EXPERT_Q1_0 = 1_536_000_000`` constant
in this file was roughly 150x too large and is the bug tracked in
Issue #2.
"""
import json
import os
from typing import Any, Dict, List

# Single source of truth for the Qwen3.8 architecture.
from qwen38_config import load_config  # noqa: E402

# GGUF average bits-per-weight (Q1_0, Q4_K_M, Q8_0). These are the
# typical llama.cpp / GGUF averages for the fused MoE expert tensors; they
# are NOT a substitute for measuring the real GGUF file.  They are used
# here only to convert a known weight count into an approximate byte size
# for capacity planning. The authoritative byte size of an actual Q1_0
# file is reported by ``tools/inventory_gguf.py`` against the real GGUF.
GGUF_AVG_BITS_PER_WEIGHT = {
    "Q1_0": 1.75,
    "Q2_K": 3.35,
    "Q3_K_M": 4.10,
    "Q4_K_M": 4.85,
    "Q5_K_M": 5.69,
    "Q6_K": 6.56,
    "Q8_0": 8.50,
}


def bytes_per_expert(weights_per_expert: int, avg_bits_per_weight: float) -> int:
    """Approximate bytes per expert for the fused MoE expert tensors.

    ``weights_per_expert = 3 * hidden * moe_int`` for the standard MoE
    layout (gate_up_proj fused + down_proj).  Average bits per weight
    is the GGUF quant-specific figure from
    :data:`GGUF_AVG_AVG_BITS_PER_WEIGHT`.
    """
    return int(round(weights_per_expert * avg_bits_per_weight / 8.0))


def build_memory_plan(
    gpu_cache_sizes_gb: List[int],
    ram_arena_sizes_gb: List[int],
    routing_trace_metrics_path: str = "artifacts/qwen38_routing_trace_metrics.json",
) -> Dict[str, Any]:
    """Build the memory-plan JSON from the SoT and the routing-trace artifact."""
    config = load_config()
    weights_per_expert = 3 * config.hidden_size * config.moe_intermediate_size

    # Capacity math per quantization. The BF16 number is the ground truth
    # (it is exact); the GGUF numbers are capacity-planning approximations.
    bytes_per_expert_bf16 = config.bytes_per_expert_bf16
    per_quant_bytes_per_expert = {
        "BF16": bytes_per_expert_bf16,
    }
    for q, bits in GGUF_AVG_BITS_PER_WEIGHT.items():
        per_quant_bytes_per_expert[q] = bytes_per_expert(weights_per_expert, bits)

    # Read the routing-trace artifact for the *behaviour* numbers.
    hit_rates: Dict[str, float] = {}
    trace_meta: Dict[str, Any] = {}
    if os.path.exists(routing_trace_metrics_path):
        with open(routing_trace_metrics_path, "r", encoding="utf-8") as f:
            rtm = json.load(f)
        m = rtm.get("metrics", {})
        hit_rates["l1_vram_hit_rate"] = m.get("l1_hit_rate")
        hit_rates["l2_ram_hit_rate"] = m.get("l2_hit_rate")
        hit_rates["nvme_fetch_rate"] = m.get("nvme_fetch_rate")
        hit_rates["stalling_miss_rate"] = m.get("stalling_miss_rate")
        hit_rates["prefetch_hide_rate"] = m.get("prefetch_hide_rate")
        trace_meta = {
            "config_sha256": rtm.get("provenance", {}).get("config_sha256"),
            "trace_source": rtm.get("provenance", {}).get("trace_source"),
            "data_classification": rtm.get("provenance", {}).get(
                "data_classification"
            ),
            "policy": rtm.get("provenance", {}).get("policy"),
            "simulator_tool": rtm.get("provenance", {}).get("simulator_tool"),
        }

    def capacity_rows(sizes_gb: List[int]) -> Dict[str, Dict[str, Any]]:
        rows: Dict[str, Dict[str, Any]] = {}
        for size_gb in sizes_gb:
            budget = size_gb * 1024 ** 3
            row: Dict[str, Any] = {
                "cache_size_gb": size_gb,
                "experts_fit_per_quant": {
                    q: int(budget // bpe)
                    for q, bpe in per_quant_bytes_per_expert.items()
                },
                "experts_fit_bf16": int(budget // bytes_per_expert_bf16),
            }
            rows[f"{size_gb}GB"] = row
        return rows

    plan: Dict[str, Any] = {
        "schema_version": "qwen38.memory_plan.v2",
        "provenance": {
            "config_path": config.config_path,
            "config_sha256": config.config_sha256,
            "model_type": config.model_type,
            "architectures": list(config.architectures),
            "num_hidden_layers": config.num_hidden_layers,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "hidden_size": config.hidden_size,
            "moe_intermediate_size": config.moe_intermediate_size,
            "weights_per_expert": weights_per_expert,
            "bytes_per_expert_bf16": bytes_per_expert_bf16,
            "bytes_per_layer_experts_bf16": config.bytes_per_layer_experts_bf16,
            "gguf_avg_bits_per_weight": GGUF_AVG_BITS_PER_WEIGHT,
            "bytes_per_expert_per_quant": per_quant_bytes_per_expert,
            "routing_trace_artifact": (
                routing_trace_metrics_path if os.path.exists(routing_trace_metrics_path) else None
            ),
            "routing_trace_meta": trace_meta,
            "regenerator": "tools/simulate_expert_cache.py",
        },
        "configuration": {
            "num_hidden_layers": config.num_hidden_layers,
            "num_experts": config.num_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "bytes_per_expert_bf16": bytes_per_expert_bf16,
            "bytes_per_layer_experts_bf16": config.bytes_per_layer_experts_bf16,
        },
        "gpu_cache_simulations": capacity_rows(gpu_cache_sizes_gb),
        "ram_arena_simulations": capacity_rows(ram_arena_sizes_gb),
        "hit_rates_from_routing_trace": hit_rates,
        "notes": (
            "Per-expert byte sizes are checkpoint-derived (weights_per_expert = "
            "3 * hidden * moe_int). BF16 is exact; GGUF values are average-bits "
            "approximations for capacity planning. The actual GGUF file size is "
            "reported by tools/inventory_gguf.py. Cache hit / miss / stall rates "
            "are NOT estimated here — they come from "
            "artifacts/qwen38_routing_trace_metrics.json, which is the "
            "authoritative source for cache behaviour. The historical "
            "1.5GB-per-expert Q1_0 constant in this file was a 150x error and "
            "has been removed; see GitHub Issue #2."
        ),
    }
    return plan


def main() -> None:
    gpu_cache_sizes = [8, 12, 16, 20, 24]
    ram_arena_sizes = [48, 64, 72, 96, 128]
    plan = build_memory_plan(gpu_cache_sizes, ram_arena_sizes)
    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/qwen38-memory-plan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
        f.write("\n")

    print(f"Memory plan regenerated -> {out_path}")
    print(f"  config_sha256: {plan['provenance']['config_sha256'][:16]}...")
    print(
        f"  bytes_per_expert_bf16: {plan['provenance']['bytes_per_expert_bf16']:,}"
    )
    print(
        f"  bytes_per_expert_q1_0 (approx): "
        f"{plan['provenance']['bytes_per_expert_per_quant']['Q1_0']:,}"
    )
    print(
        f"  bytes_per_expert_q4_k_m (approx): "
        f"{plan['provenance']['bytes_per_expert_per_quant']['Q4_K_M']:,}"
    )
    print(
        f"  bytes_per_expert_q8_0 (approx): "
        f"{plan['provenance']['bytes_per_expert_per_quant']['Q8_0']:,}"
    )
    if plan["hit_rates_from_routing_trace"]:
        print("  hit_rates_from_routing_trace:")
        for k, v in plan["hit_rates_from_routing_trace"].items():
            print(f"    {k}: {v}")
    else:
        print("  (no routing-trace metrics found; run tools/trace_qwen38_routing.py)")


if __name__ == "__main__":
    main()
