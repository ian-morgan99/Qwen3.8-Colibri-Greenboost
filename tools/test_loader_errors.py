"""Adversarial tests for the typed loader error taxonomy.

These tests construct malformed config dicts and assert that the validator
raises the *right* typed error with the *right* code and structured
``details``. They are the regression net for the Colibrì v1.6.2 hardening
work (see Issue #6).

Run:
    python3 tools/test_loader_errors.py
    python3 tools/test_loader_errors.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Allow ``python tools/test_loader_errors.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader_errors import (  # noqa: E402
    ALL_CODES,
    CrossFieldInconsistent,
    ExpertPackRangeError,
    IntegerOverflow,
    InvalidModelConfig,
    InvalidTensorShape,
    LoaderError,
    MalformedShardHeader,
    RequestContextOverflow,
    TensorOutOfFileBounds,
    UnsupportedRaggedExpert,
)
from validate_architecture import validate_architecture  # noqa: E402


# --- a minimal but architecturally valid base config --------------------- #
BASE_CONFIG: dict = {
    "num_hidden_layers": 4,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "moe_intermediate_size": 64,
}


def _config_with(**overrides: object) -> dict:
    """Return a copy of BASE_CONFIG with ``overrides`` patched in."""
    cfg = dict(BASE_CONFIG)
    cfg.update(overrides)
    return cfg


# --- tests for the error taxonomy itself --------------------------------- #
class TestErrorTaxonomy(unittest.TestCase):
    """The taxonomy must be exhaustive, fail-loud on unknown codes, and
    every subclass must serialise as JSON."""

    def test_all_codes_is_frozenset(self) -> None:
        self.assertIsInstance(ALL_CODES, frozenset)
        self.assertGreater(len(ALL_CODES), 0)

    def test_every_subclass_registers_a_known_code(self) -> None:
        # Every dataclass subclass should have a default ``code`` that is
        # in ALL_CODES. This catches drift between the canonical code set
        # and the per-error classes.
        for cls in (
            InvalidModelConfig,
            InvalidTensorShape,
            TensorOutOfFileBounds,
            ExpertPackRangeError,
            UnsupportedRaggedExpert,
            IntegerOverflow,
            RequestContextOverflow,
            CrossFieldInconsistent,
            MalformedShardHeader,
        ):
            self.assertIn(
                cls().code,
                ALL_CODES,
                f"{cls.__name__}.code not in ALL_CODES",
            )

    def test_unknown_code_raises_at_construction(self) -> None:
        # Subclassing with a typo'd code must blow up at import-time, not
        # in production when the error is raised.
        with self.assertRaises(ValueError):
            InvalidModelConfig(code="NOT_A_REAL_CODE", message="x")

    def test_round_trips_through_json(self) -> None:
        # Codes are strings (not enum ints) so they survive json.dumps.
        err = CrossFieldInconsistent(
            message="head_dim mismatch",
            details={"field": "head_dim", "observed": 256, "expected": 128},
        )
        blob = json.loads(json.dumps({"code": err.code, "details": err.details}))
        self.assertEqual(blob["code"], "CROSS_FIELD_INCONSISTENT")
        self.assertEqual(blob["details"]["observed"], 256)


# --- tests for cross-field architecture invariants ----------------------- #
class TestArchitectureValidator(unittest.TestCase):
    """The cross-field validator must catch every documented footgun."""

    def test_valid_config_passes(self) -> None:
        # Should not raise.
        validate_architecture(BASE_CONFIG)

    def test_missing_num_layers_fails(self) -> None:
        cfg = _config_with()
        del cfg["num_hidden_layers"]
        with self.assertRaises(CrossFieldInconsistent) as cm:
            validate_architecture(cfg)
        self.assertEqual(cm.exception.code, "CROSS_FIELD_INCONSISTENT")
        self.assertEqual(cm.exception.details["field"], "num_layers")

    def test_top_k_exceeds_total_experts(self) -> None:
        cfg = _config_with(num_experts_per_tok=999)
        with self.assertRaises(CrossFieldInconsistent) as cm:
            validate_architecture(cfg)
        self.assertEqual(
            cm.exception.details["field"],
            "num_experts_per_tok ≤ num_experts",
        )

    def test_kv_heads_must_divide_attn_heads(self) -> None:
        # Colibrì-style GQA footgun: 5 kv_heads does not divide 4 attn_heads.
        cfg = _config_with(num_attention_heads=4, num_key_value_heads=5)
        with self.assertRaises(CrossFieldInconsistent) as cm:
            validate_architecture(cfg)
        self.assertIn("num_key_value_heads", cm.exception.details["field"])

    def test_intermediate_size_must_be_64_aligned(self) -> None:
        # SM120 kernel alignment: 65 fails.
        cfg = _config_with(moe_intermediate_size=65)
        with self.assertRaises(CrossFieldInconsistent) as cm:
            validate_architecture(cfg)
        self.assertEqual(
            cm.exception.details["field"],
            "intermediate_size % 64",
        )

    def test_per_expert_overflow(self) -> None:
        # Construct a config that overflows int64 in
        # hidden_size × (3 × intermediate_size + 1).
        cfg = _config_with(hidden_size=1 << 40, moe_intermediate_size=1 << 24)
        with self.assertRaises(CrossFieldInconsistent) as cm:
            validate_architecture(cfg)
        self.assertEqual(
            cm.exception.details["field"], "per_expert_elements < 2^63"
        )

    def test_aliases_are_respected(self) -> None:
        # num_local_experts should be accepted as a num_experts alias.
        cfg = _config_with()
        del cfg["num_experts"]
        cfg["num_local_experts"] = 8
        # Should not raise.
        validate_architecture(cfg)


# --- tests for the file-level deserialisation of the validator ------------ #
class TestValidatorCLI(unittest.TestCase):
    """The CLI must work end-to-end on a real on-disk config."""

    def test_writes_valid_config_to_tempfile_and_invokes_validator(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fp:
            json.dump(BASE_CONFIG, fp)
            path = Path(fp.name)
        try:
            with open(path) as fp:
                cfg = json.load(fp)
            validate_architecture(cfg)  # must not raise
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    # Run as a script (not via pytest) so CI can use plain ``python3``.
    verbosity = 2 if "-v" in sys.argv else 1
    unittest.main(verbosity=verbosity)
