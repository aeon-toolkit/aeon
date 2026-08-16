"""Tests for API reference documentation completeness."""

from pathlib import Path

import pytest

from aeon.utils.discovery import all_estimators


def test_all_estimators_in_api_reference():
    """Test that all public estimators are listed in the API reference docs."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    docs_dir = repo_root / "docs" / "api_reference"

    if not docs_dir.exists():
        pytest.skip(f"API reference directory not found at {docs_dir}.")

    doc_files = list(docs_dir.glob("*.rst")) + list(docs_dir.glob("*.md"))
    doc_contents = " ".join([f.read_text(encoding="utf-8") for f in doc_files])

    estimators = all_estimators(include_sklearn=False)
    missing = []

    for name, klass in estimators:
        if name not in doc_contents:
            missing.append(f"{name} ({klass.__module__})")

    assert not missing, (
        f"The following {len(missing)} estimator(s) are missing from the API reference "
        f"in {docs_dir}:\n" + "\n".join(f"- {m}" for m in missing)
    )
