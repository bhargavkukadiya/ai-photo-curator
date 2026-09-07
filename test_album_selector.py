"""Backward-compatibility test runner for photo_curator.

Re-exports the test suites from tests/ so that legacy invocations of
`pytest test_album_selector.py` continue to run all unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on sys.path
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tests.test_cli import TestCLIValidation, test_preview_inside_fresh_output
from tests.test_scoring import (
    TestAestheticScore,
    TestCalculateTotalScore,
    TestEmotionScore,
    TestIsImageFile,
    TestLoadImage,
    TestTechnicalScore,
)
from tests.test_selection import TestFilterNearDuplicates, TestScoredImage
from tests.test_storage import (
    TestCopyTopImages,
    TestExportScoresCSV,
    test_commit_cancellation_restores_album_and_releases_lock,
    test_competing_process_is_rejected_before_manifest_access,
    test_interrupted_rollback_preserves_recovery_files,
)

__all__ = [
    "TestIsImageFile",
    "TestTechnicalScore",
    "TestAestheticScore",
    "TestEmotionScore",
    "TestCalculateTotalScore",
    "TestLoadImage",
    "TestScoredImage",
    "TestFilterNearDuplicates",
    "TestExportScoresCSV",
    "TestCopyTopImages",
    "TestCLIValidation",
    "test_preview_inside_fresh_output",
    "test_commit_cancellation_restores_album_and_releases_lock",
    "test_interrupted_rollback_preserves_recovery_files",
    "test_competing_process_is_rejected_before_manifest_access",
]
