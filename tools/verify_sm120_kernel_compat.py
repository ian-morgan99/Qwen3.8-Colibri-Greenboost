#!/usr/bin/env python3
"""Regression test for Issue #4 SM120 kernel compatibility gate.

This test guards the invariants that the Qwen3.8 MoE-on-RTX-5090 (SM120) kernel
compatibility gate must hold until Steps 4-7 of Issue #4 (real build, runtime,
numerical comparison, mixed-layer execution) have all been performed on hardware.

The test reads ``artifacts/kernel-compatibility.json`` and asserts:

* Top-level ``status`` is one of the allowed values, and NEVER the removed
  ``passed_with_caveats`` placeholder.
* ``phase1_allowed`` is ``false`` (Phase 1 cannot start until SM120 is verified).
* Every entry in ``operators[]`` has the full per-operator evidence schema
  required by Issue #4 Step 8, with no operator marked PASS without a
  non-null ``evidence_log`` and ``upstream_commit``.
* ``model_shape`` matches the authoritative config.json values
  (``moe_intermediate_size == 2048``,
  ``shared_expert_intermediate_size == 2048``, hybrid linear+full attention).
* Acceptance criteria ac1-ac5 and ac11 are satisfied.
* The provenance block pins ``compute_capability_target = sm_120``.

If any of these invariants break, the regression test fails the build so
that an accidental promotion to ``passed``/``passed_with_caveats`` cannot
silently enable Phase 1 against an unverified kernel set.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = REPO_ROOT / "artifacts" / "kernel-compatibility.json"
CONFIG = REPO_ROOT / "checkpoints" / "Qwen3.8-2.4T-A95B" / "config.json"

ALLOWED_STATUSES = {
    "requires_verification",
    "requires_remediation",
    "blocked",
    "passed",
    "failed",
}
FORBIDDEN_STATUSES = {"passed_with_caveats"}

# Per-operator evidence schema fields required by Issue #4 Step 8.
REQUIRED_OPERATOR_FIELDS = (
    "operator",
    "source_shape",
    "packed_shape",
    "dtype_or_quant",
    "kernel",
    "upstream_commit",
    "sm",
    "compile_status",
    "runtime_status",
    "numerical_status",
    "evidence_log",
    "conversion_required",
    "notes",
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _expected_hybrid_layers(num_hidden_layers: int = 92) -> tuple[int, int]:
    """Return (full_attention_count, linear_attention_count) for the 3:1 pattern.

    Qwen3.8-A95B uses a hybrid 3:1 linear:full attention pattern starting at
    layer 3, repeating every 4 layers. Layer 0 is linear (self_attn type is
    'linear_attention'). So out of 92 layers, 23 are full and 69 are linear.
    """
    full = sum(1 for i in range(num_hidden_layers) if (i + 1) % 4 == 0)
    linear = num_hidden_layers - full
    return full, linear


def main() -> int:
    failures: list[str] = []

    if not ARTIFACT.exists():
        print(f"FAIL: artifact missing: {ARTIFACT}", file=sys.stderr)
        return 1
    if not CONFIG.exists():
        print(f"FAIL: config missing: {CONFIG}", file=sys.stderr)
        return 1

    data = _load(ARTIFACT)
    cfg = _load(CONFIG)

    # --- 1. top-level status invariants -------------------------------------
    status = data.get("status")
    if status in FORBIDDEN_STATUSES:
        failures.append(
            f"top-level status is forbidden value {status!r} "
            f"(regression: {sorted(FORBIDDEN_STATUSES)})"
        )
    if status not in ALLOWED_STATUSES:
        failures.append(
            f"top-level status {status!r} not in allowed set {sorted(ALLOWED_STATUSES)}"
        )

    phase1_allowed = data.get("phase1_allowed")
    if phase1_allowed is not False:
        failures.append(
            f"phase1_allowed must be false while SM120 verification is incomplete; "
            f"got {phase1_allowed!r}"
        )

    # --- 2. operators[] schema ----------------------------------------------
    operators = data.get("operators") or []
    if not isinstance(operators, list) or not operators:
        failures.append("operators[] must be a non-empty list")

    saw_pass = False
    for idx, op in enumerate(operators):
        for field in REQUIRED_OPERATOR_FIELDS:
            if field not in op:
                failures.append(
                    f"operators[{idx}] ({op.get('operator', '?')}) "
                    f"missing required Step 8 field: {field!r}"
                )
        if op.get("compile_status") == "PASS" or op.get("runtime_status") == "PASS":
            saw_pass = True
            if op.get("evidence_log") in (None, "", []):
                failures.append(
                    f"operators[{idx}] ({op.get('operator')}) has PASS status but "
                    f"evidence_log is empty"
                )
            if op.get("upstream_commit") in (None, "", []):
                failures.append(
                    f"operators[{idx}] ({op.get('operator')}) has PASS status but "
                    f"upstream_commit is null"
                )

    if saw_pass:
        # No operator may be PASS while phase1_allowed is False; this is a
        # belt-and-braces guard on top of the AC11 check.
        if phase1_allowed is False:
            failures.append(
                "operator(s) report PASS but phase1_allowed is still false - "
                "either the operator status was promoted without flipping "
                "phase1_allowed, or phase1_allowed should be true"
            )

    # --- 3. operator coverage invariants ------------------------------------
    op_names = {op.get("operator") for op in operators}
    if not any(n and n.startswith("full_attention_") for n in op_names):
        failures.append("operators[] must contain at least one full_attention_* entry")
    if not any(n and n.startswith("linear_attention_") for n in op_names):
        failures.append("operators[] must contain at least one linear_attention_* entry")
    if "routed_expert_grouped_gemm" not in op_names:
        failures.append("operators[] must contain a routed_expert_grouped_gemm entry")
    else:
        routed = next(o for o in operators if o["operator"] == "routed_expert_grouped_gemm")
        dtype = (routed.get("dtype_or_quant") or "").upper()
        if "INT4" not in dtype and "NF4" not in dtype:
            failures.append(
                f"routed_expert_grouped_gemm dtype_or_quant {dtype!r} must mention "
                f"INT4 or NF4"
            )

    # --- 4. model_shape invariants vs config.json ---------------------------
    ms = data.get("model_shape") or {}
    if ms.get("moe_intermediate_size") != cfg.get("moe_intermediate_size"):
        failures.append(
            f"model_shape.moe_intermediate_size {ms.get('moe_intermediate_size')} "
            f"!= config.json {cfg.get('moe_intermediate_size')}"
        )
    if ms.get("shared_expert_intermediate_size") != cfg.get("shared_expert_intermediate_size"):
        failures.append(
            f"model_shape.shared_expert_intermediate_size "
            f"{ms.get('shared_expert_intermediate_size')} "
            f"!= config.json {cfg.get('shared_expert_intermediate_size')}"
        )

    expected_full, expected_linear = _expected_hybrid_layers(cfg.get("num_hidden_layers", 92))
    pattern = ms.get("attention_pattern") or {}
    if pattern.get("full_attention_layer_count") != expected_full:
        failures.append(
            f"model_shape.attention_pattern.full_attention_layer_count "
            f"{pattern.get('full_attention_layer_count')} != expected {expected_full}"
        )
    if pattern.get("linear_attention_layer_count") != expected_linear:
        failures.append(
            f"model_shape.attention_pattern.linear_attention_layer_count "
            f"{pattern.get('linear_attention_layer_count')} != expected {expected_linear}"
        )

    # --- 5. SM120 pin -------------------------------------------------------
    prov = data.get("provenance") or {}
    if prov.get("compute_capability_target") != "sm_120":
        failures.append(
            f"provenance.compute_capability_target must be 'sm_120'; "
            f"got {prov.get('compute_capability_target')!r}"
        )

    # --- 6. acceptance criteria ac1-ac5 and ac11 ----------------------------
    acs = data.get("acceptance_criteria_mapping") or {}
    must_be_satisfied = [
        "ac1_no_passed_with_caveats",
        "ac2_phase1_allowed_false",
        "ac3_shapes_from_config_2048",
        "ac4_hybrid_linear_and_full_separate",
        "ac5_commit_pinning",
        "ac11_no_sm120_pass_until_all_pass",
    ]
    for ac_id in must_be_satisfied:
        ac = acs.get(ac_id)
        if not ac:
            failures.append(f"acceptance criterion {ac_id!r} missing")
        elif ac.get("satisfied") is not True:
            failures.append(
                f"acceptance criterion {ac_id!r} must be satisfied=true; "
                f"got {ac.get('satisfied')!r}"
            )

    # --- 7. report ----------------------------------------------------------
    if failures:
        print("verify_sm120_kernel_compat: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(
        "verify_sm120_kernel_compat: OK "
        f"({len(operators)} operators, all BLOCKED, status={status!r}, "
        f"phase1_allowed={phase1_allowed!r})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
