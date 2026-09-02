#!/usr/bin/env python3
"""
Regression test: ensure canonical Phase 0 artifact basenames appear at most
once under ``artifacts/`` (the canonical namespace) and never alongside
``docs/``-canonical feasibility reports.

Run as: ``python tools/test_no_duplicate_artifacts.py``.
Exit code 0 on pass, 1 on duplicate found.

This test exists to prevent the duplicate-namespace regression that produced
the Issue #3 follow-up comment: ``artifacts/qwen38-layout.json`` and
``artifacts/qwen38/qwen38-layout.json`` both existed at the same time,
``artifacts/qwen38/qwen38-memory-plan.json`` and
``artifacts/qwen38/kernel-compatibility.json`` likewise, and
``artifacts/qwen38/QWEN38_WORKSTATION_FEASIBILITY.md`` shadowed
``docs/QWEN38_WORKSTATION_FEASIBILITY.md``.

Canonical namespaces (do not change without re-validating every consumer):

  artifacts/qwen38-layout.json         (Phase 0 — generated)
  artifacts/qwen38-memory-plan.json    (Phase 0 — generated)
  artifacts/kernel-compatibility.json  (Phase 0 — generated)
  artifacts/qwen38-phase-status.json   (Phase 0 — gate status)
  artifacts/qwen38_phase0_inventory_report.md (Phase 0 — narrative)
  docs/QWEN38_WORKSTATION_FEASIBILITY.md      (Phase 0 — feasibility)

Any duplicate or shadow copy must trigger a failure so the regression cannot
recur silently.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each entry: (canonical_relpath, set_of_acceptable_additional_paths)
# Acceptable extras should be empty; this lets the test be tightened later
# (e.g. to allow an exception in a known-bad branch) without touching logic.
CANONICAL_ARTIFACTS = {
    "artifacts/qwen38-layout.json": set(),
    "artifacts/qwen38-memory-plan.json": set(),
    "artifacts/kernel-compatibility.json": set(),
    "artifacts/qwen38-phase-status.json": set(),
    "artifacts/qwen38_phase0_inventory_report.md": set(),
    "docs/QWEN38_WORKSTATION_FEASIBILITY.md": set(),
}


def find_all(repo: Path, basename: str) -> list[Path]:
    """Find every path under ``repo`` whose final component equals ``basename``."""
    hits: list[Path] = []
    for root, _dirs, files in os.walk(repo):
        # Skip vendored trees and hidden directories
        rel_root = Path(root).relative_to(repo)
        parts = rel_root.parts
        if parts and parts[0] in (".git", "llama.cpp-iq1narrow", "freetoken-env", "dist"):
            continue
        if basename in files:
            hits.append(Path(root) / basename)
    return hits


def main() -> int:
    failures: list[str] = []
    for canonical, acceptable in CANONICAL_ARTIFACTS.items():
        basename = Path(canonical).name
        all_hits = find_all(REPO_ROOT, basename)
        rel_hits = {p.relative_to(REPO_ROOT).as_posix() for p in all_hits}
        # Required: canonical path must exist
        if canonical not in rel_hits:
            failures.append(
                f"MISSING canonical artifact: {canonical} (not generated yet, "
                f"or generator was redirected away from canonical path)"
            )
            continue
        # Disallowed: any extra copy not in acceptable set
        extras = rel_hits - {canonical} - acceptable
        if extras:
            failures.append(
                f"DUPLICATE {basename}: canonical at {canonical}, "
                f"also found at {sorted(extras)}"
            )
    if failures:
        print("FAIL: duplicate or missing canonical artifacts detected:")
        for f in failures:
            print(f"  - {f}")
        print()
        print("Fix: rerun the canonical generator (e.g. `python tools/build_qwen38_layout.py`),")
        print("then delete the stale copy, or update CANONICAL_ARTIFACTS in this test if the")
        print("canonical path is genuinely changing.")
        return 1
    print(f"PASS: all {len(CANONICAL_ARTIFACTS)} canonical artifacts exist exactly once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
