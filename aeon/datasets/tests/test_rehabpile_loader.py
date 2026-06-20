"""Test functions for RehabPile dataset loaders."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aeon.datasets import (
    load_rehab_pile_classification_datasets,
    load_rehab_pile_dataset,
    load_rehab_pile_regression_datasets,
)
from aeon.datasets._data_loaders import CONNECTION_ERRORS
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_rehabpile_listing_functions():
    """Test functions that list available RehabPile datasets."""
    class_datasets = load_rehab_pile_classification_datasets()
    assert isinstance(class_datasets, list)
    assert len(class_datasets) > 0
    assert all(isinstance(name, str) for name in class_datasets)

    reg_datasets = load_rehab_pile_regression_datasets()
    assert isinstance(reg_datasets, list)
    assert len(reg_datasets) > 0
    assert all(isinstance(name, str) for name in reg_datasets)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_rehab_pile_dataset_from_archive():
    """
    Test loading a dataset from the RehabPile archive.

    This test downloads a sample classification and regression dataset to a
    temporary directory, verifies the structure of the returned data, and
    validates the data dimensions against the returned metadata.
    """
    with tempfile.TemporaryDirectory() as tmp:
        extract_path = Path(tmp)

        # Test loading a classification problem with metadata
        classification_names = load_rehab_pile_classification_datasets()
        if classification_names:
            clf_name = classification_names[0]
            X_train, y_train, meta = load_rehab_pile_dataset(
                name=clf_name,
                split="train",
                extract_path=extract_path,
                return_meta=True,
            )

            # Check types
            assert isinstance(X_train, np.ndarray)
            assert isinstance(y_train, np.ndarray)
            assert isinstance(meta, dict)

            # Check dimensions and consistency against metadata
            assert X_train.ndim == 3
            assert y_train.ndim == 1
            assert len(X_train) == len(y_train)
            # keys from info.json
            assert X_train.shape[2] == meta["length_TS"]
            assert X_train.shape[1] == meta["n_joints"] * meta["n_dim"]

        # Test loading a regression problem
        regression_names = load_rehab_pile_regression_datasets()
        if regression_names:
            reg_name = regression_names[0]
            X_reg, y_reg = load_rehab_pile_dataset(
                name=reg_name, split="test", extract_path=extract_path
            )

            assert isinstance(X_reg, np.ndarray)
            assert isinstance(y_reg, np.ndarray)
            assert X_reg.ndim == 3
            assert y_reg.ndim == 1
            assert len(X_reg) == len(y_reg)
            assert np.issubdtype(y_reg.dtype, np.number)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_rehab_pile_dataset_return_meta():
    """Test the return_meta parameter."""
    classification_names = load_rehab_pile_classification_datasets()
    if classification_names:
        clf_name = classification_names[0]
        # Test return_meta=False
        result_false = load_rehab_pile_dataset(name=clf_name, return_meta=False)
        assert isinstance(result_false, tuple)
        assert len(result_false) == 2

        # Test return_meta=True
        result_true = load_rehab_pile_dataset(name=clf_name, return_meta=True)
        assert isinstance(result_true, tuple)
        assert len(result_true) == 3
        assert isinstance(result_true[2], dict)


def test_load_rehab_pile_dataset_invalid_fold(mocker):
    """Test that load_rehab_pile_dataset raises errors for invalid fold inputs."""
    # Mock the discovery functions to avoid network calls
    mocker.patch(
        "aeon.datasets.rehabpile_loader.load_rehab_pile_classification_datasets",
        return_value=["A_valid_clf_dataset"],
    )
    mocker.patch(
        "aeon.datasets.rehabpile_loader.load_rehab_pile_regression_datasets",
        return_value=[],
    )
    mocker.patch(
        "aeon.datasets.rehabpile_loader.REHABPILE_FOLDS",
        {"classification": {"A_valid_clf_dataset": 5}},
    )

    # Test with a fold number that is too high
    with pytest.raises(ValueError, match="Invalid fold"):
        load_rehab_pile_dataset(name="A_valid_clf_dataset", fold=99)
    # Test with a negative fold number
    with pytest.raises(ValueError, match="Invalid fold"):
        load_rehab_pile_dataset(name="A_valid_clf_dataset", fold=-1)


def test_load_rehab_pile_dataset_wrong_name_and_split(mocker):
    """Test that load_rehab_pile_dataset raises errors for invalid inputs."""
    # Mock the discovery functions to avoid network calls
    mocker.patch(
        "aeon.datasets.rehabpile_loader.load_rehab_pile_classification_datasets",
        return_value=["A_valid_clf_dataset"],
    )
    mocker.patch(
        "aeon.datasets.rehabpile_loader.load_rehab_pile_regression_datasets",
        return_value=[],
    )

    # Test invalid dataset name (now runs offline)
    with pytest.raises(
        ValueError, match="Dataset FOO_BAR not found in the RehabPile collection."
    ):
        load_rehab_pile_dataset("FOO_BAR")

    # Test invalid split parameter (now runs offline)
    with pytest.raises(
        ValueError, match="Split must be 'train' or 'test', but found 'validation'."
    ):
        load_rehab_pile_dataset(name="A_valid_clf_dataset", split="validation")
