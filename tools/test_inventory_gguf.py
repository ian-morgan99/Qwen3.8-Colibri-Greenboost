#!/usr/bin/env python3
"""Regression test: ``tools/inventory_gguf.py`` is correct end-to-end.

This test guards against the structural bugs the inventory tool had
when it was first authored: hard-coded machine paths, a call to an
undefined ``parse_gguf_header_and_tensors`` helper, and module-level
code that ran before any ``def`` was bound. After the rewrite, the
tool must:

  1. Locate a Qwen3.8 GGUF shard under ``checkpoints/`` without
     crashing on a missing directory.
  2. Parse the GGUF header and report the magic, version, tensor
     count, and KV count from a real on-disk shard.
  3. When the shard is metadata-only (tensor_count == 0), derive the
     tensor table from the SoT Qwen38Config so downstream sizing is
     deterministic.
  4. Classify tensors using the Qwen3.8 MoE naming convention
     (``.shared_expert.`` and ``.experts.<N>.`` patterns).
  5. Generate three artifacts in ``artifacts/qwen38_gguf/`` whose
     ``source`` path is repo-relative, whose memory plan advertises
     the v2 schema, and whose byte counts are non-zero.

Run as: ``python3 tools/test_inventory_gguf.py`` (exit 0 on pass).
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
GGUF_DIR = REPO_ROOT / "checkpoints" / "Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0" / "UD-Q1_0"


def _import_inventory_gguf():
    """Import the inventory module fresh, with the tools/ on sys.path."""
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    if "inventory_gguf" in sys.modules:
        return importlib.reload(sys.modules["inventory_gguf"])
    return importlib.import_module("inventory_gguf")


class TestClassifyTensor(unittest.TestCase):
    """Tensor classification must respect Qwen3.8 MoE naming."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _import_inventory_gguf()

    def test_routed_expert_with_index(self) -> None:
        bucket, expert_id = self.mod.classify_tensor(
            "model.layers.5.experts.42.gate_up_proj.weight"
        )
        self.assertEqual(bucket, "routed_expert")
        self.assertEqual(expert_id, 42)

    def test_routed_expert_explicit_id(self) -> None:
        # Per Qwen3.8 naming the segment is ``.experts.<N>.``; the
        # index is recovered from the integer segment.
        bucket, expert_id = self.mod.classify_tensor(
            "model.layers.0.experts.511.down_proj.weight"
        )
        self.assertEqual(bucket, "routed_expert")
        self.assertEqual(expert_id, 511)

    def test_shared_expert_per_layer(self) -> None:
        # Per-layer shared expert: appears under model.layers.<i>.
        bucket, expert_id = self.mod.classify_tensor(
            "model.layers.0.shared_expert.gate_proj.weight"
        )
        self.assertEqual(bucket, "shared_expert")
        self.assertIsNone(expert_id)

    def test_shared_expert_fused(self) -> None:
        # Fused gate_up_proj variant of the shared expert.
        bucket, expert_id = self.mod.classify_tensor(
            "model.layers.91.shared_expert.gate_up_proj.weight"
        )
        self.assertEqual(bucket, "shared_expert")
        self.assertIsNone(expert_id)

    def test_dense_components(self) -> None:
        # Embedding, norms, attention projections all bucket as dense.
        for name in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.q_proj.weight",
            "model.layers.0.k_proj.weight",
            "model.layers.0.v_proj.weight",
            "model.layers.0.o_proj.weight",
            "lm_head.weight",
        ):
            with self.subTest(name=name):
                bucket, expert_id = self.mod.classify_tensor(name)
                self.assertEqual(bucket, "dense_shared")
                self.assertIsNone(expert_id)


class TestSchemaConstants(unittest.TestCase):
    """The script must advertise a v2 memory-plan schema."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _import_inventory_gguf()

    def test_memory_plan_schema_v2(self) -> None:
        # Must match the schema ``simulate_expert_cache.build_memory_plan``
        # emits, otherwise downstream readers cannot rely on a single
        # version field across the toolchain.
        self.assertEqual(
            self.mod.MEMORY_PLAN_SCHEMA_VERSION, "qwen38.memory_plan.v2"
        )

    def test_q1_0_bytes_per_element(self) -> None:
        # Q1_0 packs 32 weights into 5 bytes ~= 1.5625 bits/weight.
        self.assertAlmostEqual(self.mod.Q1_0_BYTES_PER_ELEMENT, 5 / 32)


class TestParseGgufHeader(unittest.TestCase):
    """Header parsing on the real on-disk shard."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _import_inventory_gguf()

    def setUp(self) -> None:
        if not GGUF_DIR.is_dir():
            self.skipTest(f"GGUF dir not present on this machine: {GGUF_DIR}")

    def test_parse_real_shard_header(self) -> None:
        shard = self.mod.discover_gguf_shard(GGUF_DIR)
        header = self.mod.parse_gguf_header(shard)
        # Header tuple is (version, tensor_count, kv_count).
        version, tensor_count, kv_count = header
        self.assertEqual(version, 3)
        # The on-disk shard is the metadata-only stub (10MB, no tensors).
        self.assertEqual(tensor_count, 0)
        # 58 KV pairs match the published Qwen3.8 stub metadata.
        self.assertEqual(kv_count, 58)

    def test_tensor_count_zero_uses_config_fallback(self) -> None:
        # When the shard has zero tensors, the parser must fall back to
        # the SoT config so we still get a deterministic tensor table.
        shard = self.mod.discover_gguf_shard(GGUF_DIR)
        tensors = self.mod.parse_gguf_header_and_tensors(shard)
        self.assertGreater(len(tensors), 0)
        # 92 layers * (2 norms + 4 attn + 2 shared + 2*512 routed) = 94944
        # Plus embeddings/output/mtp = 555 dense, 184 shared, 94208 routed.
        bucket_counts: dict[str, int] = {}
        for t in tensors:
            bucket, _ = self.mod.classify_tensor(t["name"])
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        self.assertEqual(bucket_counts.get("dense_shared"), 555)
        self.assertEqual(bucket_counts.get("shared_expert"), 184)
        self.assertEqual(bucket_counts.get("routed_expert"), 94208)


class TestDiscoverGgufShard(unittest.TestCase):
    """Shard discovery must be precise and testable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _import_inventory_gguf()

    def test_missing_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "no-such-dir"
            with self.assertRaises(FileNotFoundError):
                self.mod.discover_gguf_shard(missing)

    def test_present_directory_returns_shard(self) -> None:
        if not GGUF_DIR.is_dir():
            self.skipTest(f"GGUF dir not present: {GGUF_DIR}")
        shard = self.mod.discover_gguf_shard(GGUF_DIR)
        self.assertTrue(shard.is_file())
        self.assertEqual(shard.suffix, ".gguf")
        # Magic bytes must start with GGUF.
        with open(shard, "rb") as f:
            self.assertEqual(f.read(4), b"GGUF")


class TestRegeneratedArtifacts(unittest.TestCase):
    """The three ``artifacts/qwen38_gguf/`` outputs must be non-trivial."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _import_inventory_gguf()
        cls.layout_path = (
            REPO_ROOT / "artifacts" / "qwen38_gguf" / "qwen38-gguf-layout.json"
        )
        cls.plan_path = (
            REPO_ROOT / "artifacts" / "qwen38_gguf" / "qwen38-gguf-memory-plan.json"
        )
        cls.feasibility_path = (
            REPO_ROOT
            / "artifacts"
            / "qwen38_gguf"
            / "QWEN38_GGUF_WORKSTATION_FEASIBILITY.md"
        )

    def test_layout_source_is_repo_relative(self) -> None:
        layout = json.loads(self.layout_path.read_text())
        # Must NOT be the old /home/beast/... hard-coded machine path.
        self.assertFalse(
            layout["source"].startswith("/home/beast/"),
            f"layout.source must be repo-relative, got {layout['source']!r}",
        )
        self.assertTrue(layout["source"].startswith("checkpoints/"))

    def test_layout_bytes_are_nonzero(self) -> None:
        layout = json.loads(self.layout_path.read_text())
        b = layout["bytes"]
        self.assertGreater(b["dense_shared"], 0)
        self.assertGreater(b["shared_expert"], 0)
        self.assertGreater(b["routed_expert"], 0)
        # 575 GiB routed + 4 GiB dense + 3.5 GiB shared ~= 583 GiB
        total_gib = sum(b.values()) / 2**30
        self.assertGreater(total_gib, 100.0)

    def test_plan_schema_v2(self) -> None:
        plan = json.loads(self.plan_path.read_text())
        self.assertEqual(plan["schema_version"], "qwen38.memory_plan.v2")
        # Provenance block is required by the v2 contract.
        self.assertIn("provenance", plan)
        self.assertIn("config_sha256", plan["provenance"])
        # Workstation profile fields must be propagated.
        self.assertIn("assumptions", plan)
        self.assertEqual(plan["assumptions"]["vram_gib"], 32)
        self.assertEqual(plan["assumptions"]["ram_gib"], 96)

    def test_feasibility_report_renders(self) -> None:
        text = self.feasibility_path.read_text()
        self.assertIn("qwen38.memory_plan.v2", text)
        # Capacity numbers must appear (not the old zeroed output).
        self.assertIn("21", text)  # gpu_experts_fit_after_dense
        self.assertIn("85", text)  # ram_experts_fit


if __name__ == "__main__":
    # Ensure the test class setUpClass can find tools/inventory_gguf.py
    # even when the runner is invoked from a different CWD.
    sys.path.insert(0, str(TOOLS_DIR))
    unittest.main(verbosity=2)
