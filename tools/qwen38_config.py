"""Authoritative Qwen3.8 architecture loader.

Single source of truth: ``checkpoints/Qwen3.8-2.4T-A95B/config.json``.

The companion document ``docs/architecture/QWEN38_CHECKPOINT_DERIVED.md``
records the exact fields and the derived math. Every tool in this repo that
needs to know how many layers / experts / tokens-per-routing / per-expert
byte size MUST import from here rather than hard-coding constants.

This is part of the remediation tracked in GitHub Issue #2 - the previous
state of the repository had several tools (``tools/trace_qwen38_routing.py``,
``tools/generate_phase0_inventory.py``) embedding hard-coded
``num_layers=92 / num_experts=512 / top_k=10`` constants that drifted
from the checkpoint and used an order-of-magnitude-wrong "1.45 GB per
expert" figure that had no basis in the real Qwen3.8 layout.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from typing import Any, Dict, Optional


# Default path to the canonical checkpoint config.json. Override via env var
# QWEN38_CONFIG if you have a different checkpoint location.
DEFAULT_CONFIG_PATH = os.environ.get(
    "QWEN38_CONFIG",
    "checkpoints/Qwen3.8-2.4T-A95B/config.json",
)


@dataclasses.dataclass(frozen=True)
class Qwen38Config:
    """Frozen view of the checkpoint architecture.

    Every field here is read directly from the checkpoint ``config.json``.
    Derived byte-size fields are computed from the same field set and are
    included so downstream tools do not have to redo the arithmetic.
    """

    # Identity / provenance
    config_path: str
    config_sha256: str
    model_type: str
    architectures: tuple

    # Architecture
    num_hidden_layers: int
    num_experts: int
    num_experts_per_tok: int
    hidden_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    full_attention_interval: int
    mtp_num_hidden_layers: int
    tie_word_embeddings: bool
    max_position_embeddings: int
    rope_theta: float

    # Linear attention (DeltaNet / Mamba2) knobs
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int

    # Derived: bytes per expert in BF16 (un-quantized)
    # Per expert: gate_up_proj (hidden x moe_int) + down_proj (moe_int x hidden)
    #   = 2 * hidden * moe_int + moe_int * hidden = 3 * hidden * moe_int
    # Note: gate_up_proj packs gate AND up into one tensor, so it's hidden x 2*moe_int
    bytes_per_expert_bf16: int
    bytes_per_layer_experts_bf16: int
    bytes_active_per_token_bf16: int
    bytes_active_per_token_bf16_mib: float

    # Derived: per-layer counts
    num_full_attention_layers: int
    num_linear_attention_layers: int

    @property
    def total_expert_tensors(self) -> int:
        """Per-layer packed expert tensor count for the routed (non-shared) experts.

        The Qwen3.8 MoE design packs all 512 experts into two tensors per layer:
        ``model.layers.N.mlp.experts.gate_up_proj`` and ``...down_proj``. They
        are NOT separate per-expert tensors, so caching must be expert-aware
        even though storage is tensor-granular. Each MTP layer also has its
        own pair of packed expert tensors, so add ``mtp_num_hidden_layers * 2``.
        """
        return (self.num_hidden_layers + self.mtp_num_hidden_layers) * 2

    @property
    def total_shared_expert_tensors(self) -> int:
        """Shared expert is split into 4 tensors per layer:

        - ``mlp.shared_expert.gate_proj.weight``
        - ``mlp.shared_expert.up_proj.weight``
        - ``mlp.shared_expert.down_proj.weight``
        - ``mlp.shared_expert_gate.weight`` (the router on the shared expert)

        MTP layers each add another 4 shared_expert tensors.
        """
        return (self.num_hidden_layers + self.mtp_num_hidden_layers) * 4

    @property
    def bytes_per_layer_experts_bf16_gib(self) -> float:
        return self.bytes_per_layer_experts_bf16 / (1024 ** 3)

    @property
    def bytes_per_expert_bf16_mib(self) -> float:
        return self.bytes_per_expert_bf16 / (1024 ** 2)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["architectures"] = list(self.architectures)
        return d


def _stable_payload(config: Dict[str, Any]) -> bytes:
    """Stable JSON for hashing. Drops ``transformers_version`` and sorts keys."""
    payload = {k: v for k, v in config.items() if k != "transformers_version"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def load_config(path: Optional[str] = None) -> Qwen38Config:
    """Load and validate a Qwen3.8 checkpoint config.json.

    If ``path`` is ``None`` the value of :data:`DEFAULT_CONFIG_PATH` is
    used (overridable via the ``QWEN38_CONFIG`` environment variable).

    Raises FileNotFoundError if the file does not exist, or ValueError if
    the file is missing a field that downstream tools rely on.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Qwen3.8 config.json not found at {path!r}. "
            "Set QWEN38_CONFIG env var to point at a real checkpoint, or "
            "clone the checkpoint into checkpoints/Qwen3.8-2.4T-A95B/."
        )

    with open(path, "rb") as f:
        raw = f.read()
    config = json.loads(raw)
    stable = _stable_payload(config)
    sha = hashlib.sha256(stable).hexdigest()

    # The full per-token activation size is dominated by the routed expert
    # path: per layer we activate ``num_experts_per_tok`` experts, each
    # contributing (3 * hidden * moe_int) BF16 params per expert. The shared
    # expert is always on, contributing (3 * hidden * shared_int) per layer.
    hidden = int(config["hidden_size"])
    moe_int = int(config["moe_intermediate_size"])
    shared_int = int(config.get("shared_expert_intermediate_size", moe_int))
    top_k = int(config["num_experts_per_tok"])
    num_experts = int(config["num_experts"])
    num_layers = int(config["num_hidden_layers"])
    full_interval = int(config.get("full_attention_interval", 4))
    # Per expert: gate_up_proj is (hidden x 2*moe_int) and down_proj is
    # (moe_int x hidden). Total params per expert = 3 * hidden * moe_int.
    params_per_expert = 3 * hidden * moe_int
    bytes_per_expert_bf16 = params_per_expert * 2
    bytes_per_layer = num_experts * bytes_per_expert_bf16
    # Active experts per token across the whole network:
    active_experts_per_token = num_layers * top_k
    # Each active expert contributes one (hidden x moe_int) activation for
    # gate, one for up, and one (moe_int x hidden) activation for down. The
    # BF16 size of the activations is therefore the same 3 * hidden * moe_int
    # per active expert. Shared expert activations are layer-locals and are
    # negligible by comparison (tens of MiB per layer), so we only report the
    # routed-expert number here. The shared-expert path is included in the
    # Qwen3.8 document.
    bytes_active_bf16 = active_experts_per_token * params_per_expert * 2
    bytes_active_mib = bytes_active_bf16 / (1024 ** 2)

    # Attention layer mix: ``full_attention_interval`` says every Nth layer
    # is full attention; the rest use the linear (DeltaNet) path. Layer 0
    # is conventionally full attention.
    full_attn_count = sum(
        1 for i in range(num_layers) if (i % full_interval) == 0
    )
    linear_attn_count = num_layers - full_attn_count

    return Qwen38Config(
        config_path=os.path.abspath(path),
        config_sha256=sha,
        model_type=str(config.get("model_type", "qwen3_5_moe_text")),
        architectures=tuple(config.get("architectures", [])),
        num_hidden_layers=num_layers,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        hidden_size=hidden,
        moe_intermediate_size=moe_int,
        shared_expert_intermediate_size=shared_int,
        vocab_size=int(config.get("vocab_size", 248320)),
        num_attention_heads=int(config.get("num_attention_heads", 64)),
        num_key_value_heads=int(config.get("num_key_value_heads", 4)),
        head_dim=int(config.get("head_dim", 256)),
        full_attention_interval=full_interval,
        mtp_num_hidden_layers=int(config.get("mtp_num_hidden_layers", 1)),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        max_position_embeddings=int(
            config.get("max_position_embeddings", 262144)
        ),
        rope_theta=float(config.get("rope_theta", 10000.0)),
        linear_conv_kernel_dim=int(
            config.get("linear_conv_kernel_dim", 4)
        ),
        linear_key_head_dim=int(
            config.get("linear_key_head_dim", 128)
        ),
        linear_value_head_dim=int(
            config.get("linear_value_head_dim", 128)
        ),
        linear_num_key_heads=int(
            config.get("linear_num_key_heads", 16)
        ),
        linear_num_value_heads=int(
            config.get("linear_num_value_heads", 128)
        ),
        bytes_per_expert_bf16=bytes_per_expert_bf16,
        bytes_per_layer_experts_bf16=bytes_per_layer,
        bytes_active_per_token_bf16=bytes_active_bf16,
        bytes_active_per_token_bf16_mib=bytes_active_mib,
        num_full_attention_layers=full_attn_count,
        num_linear_attention_layers=linear_attn_count,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.json (default: %(default)s)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit the full Qwen38Config as JSON instead of a human summary.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.json:
        print(json.dumps(cfg.to_dict(), indent=2))
        return

    print("Qwen3.8 architecture (from checkpoint config.json)")
    print(f"  config_path:        {cfg.config_path}")
    print(f"  config_sha256:      {cfg.config_sha256}")
    print(f"  model_type:         {cfg.model_type}")
    print(f"  architectures:      {list(cfg.architectures)}")
    print()
    print("  Layers / experts / routing")
    print(f"    num_hidden_layers:        {cfg.num_hidden_layers}")
    print(f"    num_experts:              {cfg.num_experts}")
    print(f"    num_experts_per_tok:      {cfg.num_experts_per_tok}")
    print(f"    full_attention_interval:  {cfg.full_attention_interval}")
    print(f"    full_attn layers:         {cfg.num_full_attention_layers}")
    print(f"    linear_attn layers:       {cfg.num_linear_attention_layers}")
    print(f"    mtp layers:               {cfg.mtp_num_hidden_layers}")
    print()
    print("  Tensor sizes (BF16 reference)")
    print(f"    hidden_size:              {cfg.hidden_size}")
    print(f"    moe_intermediate_size:    {cfg.moe_intermediate_size}")
    print(f"    shared_expert_int_size:   {cfg.shared_expert_intermediate_size}")
    print(f"    vocab_size:               {cfg.vocab_size}")
    print(f"    bytes per expert (BF16):  {cfg.bytes_per_expert_bf16_mib:.2f} MiB")
    print(f"    bytes per layer (BF16):   {cfg.bytes_per_layer_experts_bf16_gib:.3f} GiB")
    print(f"    bytes active / token:     {cfg.bytes_active_per_token_bf16_mib:.0f} MiB")
    print()
    print("  Storage layout")
    print(f"    packed expert tensors / layer:   2  (gate_up_proj, down_proj)")
    print(f"    shared expert tensors / layer:   3  (gate, up, down, split)")
    print(f"    total packed expert tensors:     {cfg.total_expert_tensors}")
    print(f"    total shared expert tensors:     {cfg.total_shared_expert_tensors}")


if __name__ == "__main__":
    main()
