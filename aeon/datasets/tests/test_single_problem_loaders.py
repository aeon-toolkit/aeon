# -*- coding: utf-8 -*-
"""Test single problem loaders with varying return types."""
import numpy as np
import pandas as pd
import pytest

from aeon.datasets import (  # Univariate; Unequal length; Multivariate
    load_acsf1,
    load_arrow_head,
    load_basic_motions,
    load_italy_power_demand,
    load_japanese_vowels,
    load_osuleaf,
    load_plaid,
    load_unit_test,
)

UNIVARIATE_PROBLEMS = [
    load_acsf1,
    load_arrow_head,
    load_italy_power_demand,
    load_osuleaf,
    load_unit_test,
]
MULTIVARIATE_PROBLEMS = [
    load_basic_motions,
]
UNEQUAL_LENGTH_PROBLEMS = [
    load_plaid,
    load_japanese_vowels,
]


@pytest.mark.parametrize("loader", UNEQUAL_LENGTH_PROBLEMS)
def test_load_dataframe(loader):
    """Test unequal length baked in TSC problems load into List of numpy."""
    # should work for all
    X, y = loader()
    assert isinstance(X, list)
    assert isinstance(X[0], np.ndarray)
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    X = loader(return_X_y=False)
    assert isinstance(X, pd.DataFrame)


@pytest.mark.parametrize("loader", UNIVARIATE_PROBLEMS + MULTIVARIATE_PROBLEMS)
def test_load_numpy3d(loader):
    """Test equal length TSC problems load into numpy3d."""
    X, y = loader()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 3
    assert y.ndim == 1
