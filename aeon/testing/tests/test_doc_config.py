"""Tests for documentation configuration callbacks."""

import runpy
import sys
from pathlib import Path
from shutil import copyfile

import pytest

from aeon.classification import DummyClassifier
from aeon.utils import discovery


@pytest.mark.parametrize("working_directory", ["root", "docs"])
@pytest.mark.parametrize("existing_table", [False, True])
def test_estimator_overview_output_directory(
    tmp_path, monkeypatch, working_directory, existing_table
):
    """Generate the included table beside conf.py regardless of launch directory."""
    pytest.importorskip("tabulate")
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    source = repo_root / "docs" / "conf.py"
    if not source.exists():
        pytest.skip("Tests are not being run from a repository checkout.")

    docs = tmp_path / "docs"
    docs.mkdir()
    copyfile(source, docs / "conf.py")
    table = docs / "estimator_overview_table.md"
    if existing_table:
        table.write_text("stale table", encoding="utf-8")

    monkeypatch.setattr(sys, "path", sys.path.copy())
    config = runpy.run_path(str(docs / "conf.py"))
    monkeypatch.setattr(
        discovery,
        "all_estimators",
        lambda **kwargs: [("DummyClassifier", DummyClassifier)],
    )
    monkeypatch.chdir(docs if working_directory == "docs" else tmp_path)

    config["_make_estimator_overview"](None)

    assert table.exists()
    assert "DummyClassifier" in table.read_text(encoding="utf-8")
    assert not (tmp_path / "estimator_overview_table.md").exists()
