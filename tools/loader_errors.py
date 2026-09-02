#!/usr/bin/env python3
"""Typed error taxonomy for the Qwen3.8 checkpoint/config loader.

Background: Colibrì PR #1018 (v1.6.2) found six memory-safety bugs in the
loader that all stemmed from un-typed, catch-all ``ValueError`` /
``RuntimeError`` failures. The fix is not just the boundary check, it is
*also* making the failure machine-classifiable so a junior coding agent or
a CI bot can tell ``INVALID_TENSOR_SHAPE`` from ``INTEGER_OVERFLOW`` from
``REQUEST_CONTEXT_OVERFLOW`` and react appropriately (e.g. block a
deployment vs. surface a 4xx to the user).

This module is the single source of truth for loader error codes. Adding
a new code here is the only acceptable way to introduce a new typed
failure.

Conventions
-----------
* Codes are ``SCREAMING_SNAKE_CASE`` strings, not enum integers, so they
  survive ``json.dumps`` round-trips without an extra ``int->str`` step.
* Every error has a ``code`` (machine), ``message`` (human), and an
  optional ``details`` dict (variable: shape, byte range, expected vs.
  observed, etc.). The ``details`` dict must not contain arbitrary
  model-file contents — it is a structured summary, not a log dump.
* Errors inherit from ``LoaderError`` so callers can catch the family
  with a single ``except LoaderError`` without missing any of the typed
  subclasses.
* The class name and the ``code`` attribute are intentionally redundant
  so that ``repr(error)`` is human-readable in a traceback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Canonical code set. Kept as a frozenset so accidental string typos
# fail loudly at import time, not when the error is raised in production.
ALL_CODES: frozenset[str] = frozenset({
    "INVALID_MODEL_CONFIG",
    "INVALID_TENSOR_SHAPE",
    "TENSOR_OUT_OF_FILE_BOUNDS",
    "EXPERT_PACK_RANGE_ERROR",
    "UNSUPPORTED_RAGGED_EXPERT",
    "INTEGER_OVERFLOW",
    "REQUEST_CONTEXT_OVERFLOW",
    "CROSS_FIELD_INCONSISTENT",
    "MALFORMED_SHARD_HEADER",
})


@dataclass
class LoaderError(Exception):
    """Base class for all typed loader failures."""

    code: str = "INVALID_MODEL_CONFIG"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ALL_CODES:
            # Failing loud here means a developer cannot accidentally
            # invent a code without registering it in ALL_CODES.
            raise ValueError(
                f"unknown loader error code {self.code!r}; "
                f"add it to tools/loader_errors.ALL_CODES"
            )
        super().__init__(self._format())

    def _format(self) -> str:
        if self.details:
            return f"[{self.code}] {self.message} ({self.details})"
        return f"[{self.code}] {self.message}"


# --- Concrete typed errors ------------------------------------------------ #
# Each subclass fixes a single ``code`` so the only thing the caller
# customises is the message and the details dict. This is the same
# pattern Colibrì adopted in the upstream PR.


@dataclass
class InvalidModelConfig(LoaderError):
    code: str = "INVALID_MODEL_CONFIG"
    message: str = ""


@dataclass
class InvalidTensorShape(LoaderError):
    code: str = "INVALID_TENSOR_SHAPE"
    message: str = ""


@dataclass
class TensorOutOfFileBounds(LoaderError):
    code: str = "TENSOR_OUT_OF_FILE_BOUNDS"
    message: str = ""


@dataclass
class ExpertPackRangeError(LoaderError):
    code: str = "EXPERT_PACK_RANGE_ERROR"
    message: str = ""


@dataclass
class UnsupportedRaggedExpert(LoaderError):
    code: str = "UNSUPPORTED_RAGGED_EXPERT"
    message: str = ""


@dataclass
class IntegerOverflow(LoaderError):
    code: str = "INTEGER_OVERFLOW"
    message: str = ""


@dataclass
class RequestContextOverflow(LoaderError):
    code: str = "REQUEST_CONTEXT_OVERFLOW"
    message: str = ""


@dataclass
class CrossFieldInconsistent(LoaderError):
    code: str = "CROSS_FIELD_INCONSISTENT"
    message: str = ""


@dataclass
class MalformedShardHeader(LoaderError):
    code: str = "MALFORMED_SHARD_HEADER"
    message: str = ""
