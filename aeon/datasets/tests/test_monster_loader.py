"""Test functions for monster dataset loader."""

import numpy as np
import pytest

from aeon.datasets._data_loaders import CONNECTION_ERRORS
from aeon.datasets.monster_loader import (
    load_monster_dataset,
    load_monster_dataset_names,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("huggingface-hub", severity="none"),
    reason="required soft dependency huggingface-hub not available",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_monster_dataset_names():
    """Test loading the list of Monster dataset names."""
    dataset_names = load_monster_dataset_names()
    assert isinstance(dataset_names, list)
    assert len(dataset_names) > 0
    assert all(isinstance(name, str) for name in dataset_names)


@pytest.mark.skipif(
    not _check_soft_dependencies("huggingface-hub", severity="none"),
    reason="required soft dependency huggingface-hub not available",
)
@pytest.mark.xfail(raises=CONNECTION_ERRORS)
def test_load_monster_dataset():
    """Test loading a Monster dataset and verify its structure."""
    dataset_name = "FOO"
    with pytest.raises(
        ValueError, match=f"Dataset {dataset_name} not found in the Monster collection."
    ):
        load_monster_dataset(dataset_name)
    dataset_name = "Pedestrian"

    X_train, y_train, X_test, y_test = load_monster_dataset(
        dataset_name=dataset_name, fold=0, normalize=True
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    assert X_train.ndim == 3
    assert X_test.ndim == 3
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    # Check normalization
    mean = np.mean(X_train, axis=(0, 2))
    std = np.std(X_train, axis=(0, 2))
    np.testing.assert_array_almost_equal(mean, np.zeros_like(mean), decimal=3)
    np.testing.assert_array_almost_equal(std, np.ones_like(std), decimal=3)
