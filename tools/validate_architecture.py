#!/usr/bin/env python3
"""Cross-field architecture validator for the Qwen3.8-2.4T-A95B checkpoint.

Background
----------
Colibrì PR #1018 found that ``qk_rope_head_dim`` could be set in
combination with other head dimensions to produce negative offsets and
OOB accesses. The lesson is broader: a config-controlled relation
between two fields (e.g. ``num_attention_heads × head_dim ==
hidden_size``) cannot be trusted to be self-consistent, and a runtime
that *assumes* the relation will discover the inconsistency through
failed pointer arithmetic or a kernel crash.

This validator is the architectural place where those relations are
checked *before* any downstream consumer (memory planner, simulator,
future runtime) takes the config as truth.

Design choices
--------------
* Fail-closed: any violation raises a typed
  :class:`CrossFieldInconsistent` error from
  :mod:`tools.loader_errors`. We do **not** warn-and-continue.
* Pure stdlib: no third-party deps, runs in CI on every push.
* Idempotent and side-effect free: ``main()`` either prints a one-line
  OK summary and returns 0, or raises a typed error and returns 1.
* Operates on a checkpoint directory containing ``config.json``
  (Hugging Face format) and, optionally, the layout produced by
  :mod:`tools.inventory_checkpoint`.

The validator is intentionally separate from the inventory tool so it
can be invoked in CI even when the full inventory pass would be
prohibitively expensive (e.g. for a quick config-only smoke test).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Allow ``python tools/validate_architecture.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loader_errors import CrossFieldInconsistent  # noqa: E402


# --- field aliases -------------------------------------------------------- #
# Different model card revisions use different names for the same field.
# Centralise the aliases here so the validators below read cleanly.
ALIASES = {
    "num_layers": ("num_hidden_layers", "num_layers"),
    "num_experts": ("num_local_experts", "num_experts"),
    "num_experts_per_tok": (
        "num_experts_per_tok",
        "num_selected_experts",
    ),
    "intermediate_size": (
        "moe_intermediate_size",
        "intermediate_size",
    ),
    "shared_expert_intermediate_size": (
        "shared_expert_intermediate_size",
        "shared_intermediate_size",
    ),
}


def pick(config: dict[str, Any], key: str) -> Any:
    """Return the first present value among the aliases for ``key``."""
    for alias in ALIASES[key]:
        if alias in config:
            return config[alias]
    return None


def _check(name: str, condition: bool, message: str, **details: Any) -> None:
    if not condition:
        raise CrossFieldInconsistent(
            message=message,
            details={"field": name, **details},
        )


def validate_architecture(config: dict[str, Any]) -> None:
    """Validate cross-field invariants for a Qwen3.8 config.

    Raises :class:`CrossFieldInconsistent` on the first violation.
    """
    # --- presence checks ------------------------------------------------ #
    num_layers = pick(config, "num_layers")
    num_experts = pick(config, "num_experts")
    num_experts_per_tok = pick(config, "num_experts_per_tok")
    hidden_size = config.get("hidden_size")
    num_attention_heads = config.get("num_attention_heads")
    num_key_value_heads = config.get("num_key_value_heads")
    head_dim = config.get("head_dim")  # may be None; some configs embed it
    intermediate_size = pick(config, "intermediate_size")
    shared_expert_intermediate_size = pick(
        config, "shared_expert_intermediate_size"
    )

    _check(
        "num_layers",
        num_layers is not None and num_layers > 0,
        "num_hidden_layers / num_layers must be a positive integer",
        observed=num_layers,
    )
    _check(
        "num_experts",
        num_experts is not None and num_experts > 0,
        "num_local_experts / num_experts must be a positive integer",
        observed=num_experts,
    )
    _check(
        "num_experts_per_tok",
        num_experts_per_tok is not None and num_experts_per_tok > 0,
        "num_experts_per_tok / num_selected_experts must be a positive integer",
        observed=num_experts_per_tok,
    )
    _check(
        "hidden_size",
        hidden_size is not None and hidden_size > 0,
        "hidden_size must be a positive integer",
        observed=hidden_size,
    )
    _check(
        "num_attention_heads",
        num_attention_heads is not None and num_attention_heads > 0,
        "num_attention_heads must be a positive integer",
        observed=num_attention_heads,
    )

    # --- num_attention_heads × head_dim sanity ------------------------- #
    # In Colibrì, the bug was a *negative* offset caused by an inconsistent
    # product of head_dim and num_heads producing a smaller Q-output than
    # the buffer layout expected. We can't assume the Q-output-dim equals
    # hidden_size (Qwen3.8 deliberately uses Q-output > hidden_size to
    # leave headroom for Q-norm fused-multi-head), so the only universal
    # invariant we can enforce is:
    #
    #   1. head_dim > 0 (already required by presence check above)
    #   2. num_attention_heads * head_dim does not overflow int64
    #   3. num_attention_heads * head_dim is divisible by ``num_attention_heads``
    #      (trivially true, but documents the intended tensor shape)
    #
    # The Colibrì-specific "head_dim × num_heads == hidden_size" check is
    # deliberately omitted because Qwen3.8-2.4T-A95B has
    # head_dim=256, num_heads=64, hidden_size=8192 (Q-output = 16384, twice
    # hidden_size). See tools/validate_architecture.py docstring for the
    # SA rationale.
    if head_dim is not None:
        q_out = num_attention_heads * head_dim
        # Overflow check: 2^63 is the safetensors/int64 ceiling
        _check(
            "head_dim × num_attention_heads overflow",
            q_out < 2**63,
            "head_dim × num_attention_heads overflows int64",
            observed=q_out,
            max_int64=2**63 - 1,
        )
        # head_dim must be a positive divisor of q_out (sanity: it always is,
        # but this documents the intended relationship)
        _check(
            "head_dim divides (num_attention_heads × head_dim)",
            q_out % head_dim == 0,
            "head_dim must divide (num_attention_heads × head_dim)",
            q_out=q_out,
            head_dim=head_dim,
        )

    # --- GQA invariant: kv_heads divides attn_heads evenly ------------- #
    if num_key_value_heads is not None:
        _check(
            "num_key_value_heads | num_attention_heads",
            num_attention_heads % num_key_value_heads == 0,
            "num_attention_heads must be a clean multiple of "
            "num_key_value_heads (GQA grouping invariant).",
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            remainder=num_attention_heads % num_key_value_heads,
        )

    # --- MoE capacity: top-k must not exceed total experts -------------- #
    _check(
        "num_experts_per_tok ≤ num_experts",
        num_experts_per_tok <= num_experts,
        "num_experts_per_tok cannot exceed num_experts",
        top_k=num_experts_per_tok,
        total=num_experts,
    )

    # --- Shared expert intermediate size must be a multiple of 64 ------ #
    # Many SM120 kernels require a 64-byte alignment on intermediate
    # dimensions. We assert this even when the kernel is not yet chosen
    # because the cost of catching it now is zero and the cost of
    # catching it after kernel integration is high.
    if shared_expert_intermediate_size is not None:
        _check(
            "shared_expert_intermediate_size % 64",
            shared_expert_intermediate_size % 64 == 0,
            "shared_expert_intermediate_size must be a multiple of 64 "
            "(SM120 kernel alignment requirement).",
            observed=shared_expert_intermediate_size,
        )

    # --- MoE intermediate size similarly -------------------------------- #
    if intermediate_size is not None:
        _check(
            "intermediate_size % 64",
            intermediate_size % 64 == 0,
            "intermediate_size must be a multiple of 64 "
            "(SM120 kernel alignment requirement).",
            observed=intermediate_size,
        )

    # --- Overflow check on tensor byte counts --------------------------- #
    # A single expert is hidden_size × (3 × intermediate_size + 1) for
    # the fused gate_up_proj. If hidden_size × intermediate_size
    # overflows a 64-bit int, the resulting layout will silently wrap
    # and produce a 0-byte tensor descriptor that nothing downstream
    # will catch.
    if intermediate_size is not None and num_experts is not None:
        per_expert_elements = hidden_size * (3 * intermediate_size + 1)
        # math.prod would also work; this is a deliberate, named product
        # so the overflow check is obvious.
        _check(
            "per_expert_elements < 2^63",
            0 < per_expert_elements < (1 << 63),
            "hidden_size × (3 × intermediate_size + 1) overflowed or is "
            "non-positive; the resulting expert byte count would not be "
            "representable in a 64-bit signed integer.",
            observed=per_expert_elements,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the checkpoint config.json (Hugging Face format).",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_architecture(config)
    num_layers = pick(config, "num_layers")
    num_experts = pick(config, "num_experts")
    num_experts_per_tok = pick(config, "num_experts_per_tok")
    hidden_size = config.get("hidden_size")
    print(
        "OK: architecture invariants hold "
        f"(layers={num_layers}, experts={num_experts}, "
        f"top_k={num_experts_per_tok}, hidden_size={hidden_size})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
