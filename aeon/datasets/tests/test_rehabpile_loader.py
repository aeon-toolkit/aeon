"""Test functions for RehabPile dataset loaders."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aeon.datasets._data_loaders import CONNECTION_ERRORS
from aeon.datasets.rehabpile_loader import (
    load_rehabpile,
    rehabpile_classification_datasets,
    rehabpile_regression_datasets,
)
from aeon.testing.testing_config import PR_TESTING


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_rehabpile_listing_functions():
    """Test functions that list available RehabPile datasets.

    This test requires an internet connection to scrape the website.
    It checks that the returned objects are lists and contain strings.
    """
    class_datasets = rehabpile_classification_datasets()
    assert isinstance(class_datasets, list)
    assert len(class_datasets) > 0
    assert all(isinstance(name, str) for name in class_datasets)

    reg_datasets = rehabpile_regression_datasets()
    assert isinstance(reg_datasets, list)
    assert len(reg_datasets) > 0
    assert all(isinstance(name, str) for name in reg_datasets)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Only run on overnights because of read from internet.",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_rehabpile_from_archive():
    """
    Test loading a dataset from the RehabPile archive.

    This test downloads a sample classification and regression dataset to a
    temporary directory and verifies the structure of the returned data.
    """
    with tempfile.TemporaryDirectory() as tmp:
        extract_path = Path(tmp)

        # Test loading a classification problem
        classification_names = rehabpile_classification_datasets()
        if classification_names:
            clf_name = classification_names[0]
            X_train, y_train = load_rehabpile(
                name=clf_name, split="train", extract_path=extract_path
            )
            X_test, y_test = load_rehabpile(
                name=clf_name, split="test", extract_path=extract_path
            )

            # Check types
            assert isinstance(X_train, np.ndarray)
            assert isinstance(y_train, np.ndarray)
            assert isinstance(X_test, np.ndarray)
            assert isinstance(y_test, np.ndarray)

            # Check dimensions and consistency
            assert X_train.ndim == 3, "X_train should be 3D numpy array"
            assert y_train.ndim == 1, "y_train should be 1D numpy array"
            assert X_test.ndim == 3, "X_test should be 3D numpy array"
            assert y_test.ndim == 1, "y_test should be 1D numpy array"
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)

        # Test loading a regression problem
        regression_names = rehabpile_regression_datasets()
        if regression_names:
            reg_name = regression_names[0]
            X_reg, y_reg = load_rehabpile(
                name=reg_name, split="test", extract_path=extract_path
            )

            assert isinstance(X_reg, np.ndarray)
            assert isinstance(y_reg, np.ndarray)
            assert X_reg.ndim == 3
            assert y_reg.ndim == 1
            assert len(X_reg) == len(y_reg)
            assert np.issubdtype(y_reg.dtype, np.number)


def test_load_rehabpile_wrong_name_and_split():
    """Test that load_rehabpile raises errors for invalid inputs."""
    # Test invalid dataset name
    with pytest.raises(
        ValueError, match="Dataset FOO_BAR not found in the RehabPile collection."
    ):
        load_rehabpile("FOO_BAR")

    # Test invalid split parameter
    classification_names = rehabpile_classification_datasets()
    if classification_names:
        valid_name = classification_names[0]
        with pytest.raises(
            ValueError, match="Split must be 'train' or 'test', but found 'validation'."
        ):
            load_rehabpile(name=valid_name, split="validation")
