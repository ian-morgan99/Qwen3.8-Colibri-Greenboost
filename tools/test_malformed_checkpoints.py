"""Adversarial tests for the byte-range validator in ``inventory_checkpoint.py``.

The previous Colibrì v1.6.2 bug (Issue #6) was a ``qk_rope_head_dim``
footgun that surfaced as a negative ``data_offsets`` end — i.e. a
byte-range problem, not a config-level one. The ``loader_errors.py``
taxonomy catches the config-level shape up front (``validate_architecture``);
this file catches the byte-range half.

Each test crafts a malformed safetensors shard on disk, calls
``validate_tensor_offsets`` (or ``safetensors_header``) against it, and
asserts the *right* typed error is raised. This is the regression net for
the Colibrì hardening work.

Run:
    python3 tools/test_malformed_checkpoints.py
    python3 tools/test_malformed_checkpoints.py -v
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inventory_checkpoint import safetensors_header, validate_tensor_offsets
from loader_errors import (  # noqa: E402
    InvalidTensorShape,
    MalformedShardHeader,
    TensorOutOfFileBounds,
)


def _write_shard(header_dict: dict, payload: bytes) -> Path:
    """Write a synthetic safetensors shard and return its path."""
    hb = json.dumps(header_dict).encode()
    pad = (8 - len(hb) % 8) % 8
    hb += b" " * pad
    tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors").name)
    tmp.write_bytes(struct.pack("<Q", len(hb)) + hb + payload)
    return tmp


class TestShardHeader(unittest.TestCase):
    """``safetensors_header`` must reject every malformed header shape."""

    def test_short_prefix(self) -> None:
        # Fewer than 8 bytes -> can't even read the size prefix.
        tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors").name)
        tmp.write_bytes(b"abc")
        with self.assertRaises(MalformedShardHeader) as ctx:
            safetensors_header(tmp)
        self.assertEqual(ctx.exception.details["got_bytes"], 3)
        tmp.unlink()

    def test_implausible_header_size(self) -> None:
        # Declares 8 GiB header. MAX_HEADER_SIZE is 100 MiB.
        tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors").name)
        tmp.write_bytes(struct.pack("<Q", 8 * 1024**3))
        with self.assertRaises(MalformedShardHeader) as ctx:
            safetensors_header(tmp)
        self.assertEqual(ctx.exception.details["header_size"], 8 * 1024**3)
        tmp.unlink()

    def test_zero_header_size(self) -> None:
        tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors").name)
        tmp.write_bytes(struct.pack("<Q", 0))
        with self.assertRaises(MalformedShardHeader):
            safetensors_header(tmp)
        tmp.unlink()

    def test_short_header_body(self) -> None:
        # Header says 1000 bytes; only 5 follow.
        tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors").name)
        tmp.write_bytes(struct.pack("<Q", 1000) + b"hello")
        with self.assertRaises(MalformedShardHeader) as ctx:
            safetensors_header(tmp)
        self.assertEqual(ctx.exception.details["declared"], 1000)
        self.assertEqual(ctx.exception.details["read"], 5)
        tmp.unlink()

    def test_missing_shard(self) -> None:
        # Path that does not exist -> OSError wrapped to typed.
        with self.assertRaises(MalformedShardHeader) as ctx:
            safetensors_header(Path("/nonexistent/path/shard.safetensors"))
        self.assertIn("os_error", ctx.exception.details)


class TestTensorOffsets(unittest.TestCase):
    """``validate_tensor_offsets`` must catch every footgun the
    Colibrì v1.6.2 bug family can produce."""

    def _good_tensor(self) -> dict:
        # I32, shape [1] = 1 element * 4 bytes = 4.  Fits inside the
        # 4-byte payload below.
        return {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}

    def _shard(self) -> Path:
        return _write_shard(
            {"a": self._good_tensor()},
            b"\x01\x00\x00\x00",
        )

    def test_happy_path(self) -> None:
        sp = self._shard()
        result = validate_tensor_offsets(sp, self._good_tensor(), "a")
        self.assertTrue(result["bytes_validated"])
        self.assertEqual(result["source_offset"], 0)
        self.assertEqual(result["source_end"], 4)
        self.assertEqual(result["file_size"], sp.stat().st_size)
        sp.unlink()

    def test_end_beyond_file(self) -> None:
        # The Colibrì v1.6.2 bug class: end > file_size.
        sp = self._shard()
        bad = {"dtype": "I32", "shape": [1], "data_offsets": [0, 9999]}
        with self.assertRaises(TensorOutOfFileBounds) as ctx:
            validate_tensor_offsets(sp, bad, "a")
        self.assertEqual(ctx.exception.details["end_offset"], 9999)
        self.assertEqual(ctx.exception.details["file_size"], sp.stat().st_size)
        sp.unlink()

    def test_negative_start(self) -> None:
        sp = self._shard()
        bad = {"dtype": "I32", "shape": [1], "data_offsets": [-1, 4]}
        with self.assertRaises(TensorOutOfFileBounds) as ctx:
            validate_tensor_offsets(sp, bad, "a")
        self.assertEqual(ctx.exception.details["start"], -1)
        sp.unlink()

    def test_end_before_start(self) -> None:
        sp = self._shard()
        bad = {"dtype": "I32", "shape": [1], "data_offsets": [5, 4]}
        with self.assertRaises(TensorOutOfFileBounds):
            validate_tensor_offsets(sp, bad, "a")
        sp.unlink()

    def test_missing_offsets(self) -> None:
        sp = self._shard()
        bad = {"dtype": "I32", "shape": [1]}
        with self.assertRaises(TensorOutOfFileBounds):
            validate_tensor_offsets(sp, bad, "a")
        sp.unlink()

    def test_non_int_offsets(self) -> None:
        sp = self._shard()
        bad = {"dtype": "I32", "shape": [1], "data_offsets": [0, "four"]}
        with self.assertRaises(TensorOutOfFileBounds):
            validate_tensor_offsets(sp, bad, "a")
        sp.unlink()

    def test_shape_dtype_mismatch(self) -> None:
        # I32, shape [2] = 8 bytes; data_offsets claims 16. The header
        # is lying about its own contents.
        sp = self._shard()
        bad = {"dtype": "I32", "shape": [2], "data_offsets": [0, 16]}
        with self.assertRaises(InvalidTensorShape) as ctx:
            validate_tensor_offsets(sp, bad, "a")
        self.assertEqual(ctx.exception.details["expected_bytes"], 8)
        self.assertEqual(ctx.exception.details["declared_bytes"], 16)
        sp.unlink()

    def test_unknown_dtype(self) -> None:
        sp = self._shard()
        bad = {"dtype": "F128", "shape": [1], "data_offsets": [0, 16]}
        with self.assertRaises(InvalidTensorShape) as ctx:
            validate_tensor_offsets(sp, bad, "a")
        self.assertEqual(ctx.exception.details["dtype"], "F128")
        sp.unlink()

    def test_shard_missing(self) -> None:
        # stat() fails on a missing shard -> wrapped to typed error.
        with self.assertRaises(MalformedShardHeader):
            validate_tensor_offsets(
                Path("/nonexistent/shard.safetensors"),
                self._good_tensor(),
                "a",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
