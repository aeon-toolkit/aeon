"""Test single problem loaders with varying return types."""

import os

import numpy as np
import pandas as pd
import pytest

import aeon
from aeon.datasets import (  # Univariate; Unequal length; Multivariate
    load_acsf1,
    load_airline,
    load_arrow_head,
    load_basic_motions,
    load_covid_3month,
    load_from_tsf_file,
    load_italy_power_demand,
    load_japanese_vowels,
    load_longley,
    load_lynx,
    load_macroeconomic,
    load_osuleaf,
    load_plaid,
    load_shampoo_sales,
    load_solar,
    load_unit_test,
    load_unit_test_tsf,
    load_uschange,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies

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
def test_load_unequal_length(loader):
    """Test unequal length baked in TSC problems load into List of numpy."""
    # should work for all
    X, y = loader()
    assert isinstance(X, list)
    assert isinstance(X[0], np.ndarray)
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1


@pytest.mark.parametrize("loader", UNIVARIATE_PROBLEMS + MULTIVARIATE_PROBLEMS)
def test_load_numpy3d(loader):
    """Test equal length TSC problems load into numpy3d."""
    X, y = loader()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 3
    assert y.ndim == 1


@pytest.mark.parametrize("loader", UNIVARIATE_PROBLEMS)
def test_load_numpy2d_uni(loader):
    """Test equal length univariate TSC problems can be loaded into numpy2d."""
    X, y = loader(return_type="numpy2d")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 2
    assert y.ndim == 1


def test_load_unit_test_tsf():
    """Test load unit test tsf."""
    tuple = load_unit_test_tsf()
    assert isinstance(tuple[0], pd.DataFrame)
    assert tuple[0].shape == (3, 3)
    assert tuple[1] == "yearly"
    assert tuple[2] == 4
    assert not tuple[3]
    assert not tuple[4]
    tuple = load_unit_test_tsf(return_type="pd_multiindex_hier")
    assert tuple[0].shape == (15, 1)


def test_basic_load_tsf_to_dataframe():
    """Simple loader test."""
    full_path = os.path.join(
        os.path.dirname(aeon.__file__),
        "datasets/data/UnitTest/UnitTest_Tsf_Loader.tsf",
    )
    df, metadata = load_from_tsf_file(full_path)
    assert isinstance(df, pd.DataFrame)
    assert metadata["frequency"] == "yearly"
    assert metadata["forecast_horizon"] == 4
    assert metadata["contain_missing_values"] is False
    assert metadata["contain_equal_length"] is False


def test_load_solar():
    """Test function to load solar data."""
    solar = load_solar()
    assert type(solar) is pd.Series
    assert solar.shape == (289,)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency statsmodels not available",
)
def test_load_macroeconomic():
    """Test load macroeconomic."""
    y = load_macroeconomic()
    assert isinstance(y, pd.DataFrame)
    assert y.shape == (203, 12)


def test_load_covid_3month():
    """Test load covid 3 month."""
    X, y = load_covid_3month()
    assert isinstance(X, np.ndarray)
    assert len(X) == len(y)
    assert X.shape == (201, 1, 84)
    assert isinstance(y, np.ndarray)


def test_shampoo_sales():
    """Test baked in loader of shampoo sales."""
    y = load_shampoo_sales()
    assert isinstance(y, np.ndarray)
    y2 = load_shampoo_sales(return_array=False)
    assert isinstance(y2, pd.Series)
    assert y2.shape == (36,)
    assert y.shape == y2.shape


def test_lynx():
    """Test baked in loader of lynx data."""
    y = load_lynx()
    assert isinstance(y, np.ndarray)
    y2 = load_lynx(return_array=False)
    assert isinstance(y2, pd.Series)
    assert y2.shape == (114,)
    assert y.shape == y2.shape


def test_airline():
    """Test baked in loader of lynx data."""
    y = load_airline()
    assert isinstance(y, np.ndarray)
    y2 = load_airline(return_array=False)
    assert isinstance(y2, pd.Series)
    assert y2.shape == (144,)
    assert y.shape == y2.shape


def test_solar():
    """Test function to load solar data."""
    y = load_solar()
    assert isinstance(y, np.ndarray)
    y2 = load_solar(return_array=False)
    assert isinstance(y2, pd.Series)
    assert y2.shape == (289,)
    assert y.shape == y2.shape


def test_uschange():
    """Test if uschange dataset is loaded correctly."""
    y, X = load_uschange()
    assert isinstance(y, pd.Series)
    assert len(y) == 187
    assert y.dtype == "float64"
    assert isinstance(X, pd.DataFrame)
    col_names = ["Income", "Production", "Savings", "Unemployment"]

    assert X.columns.values.tolist() == col_names
    for col in col_names:
        assert X[col].dtype == "float64"
    assert len(X) == len(y)


def test_longley():
    """Test if longley dataset is loaded correctly."""
    y, X = load_longley()
    assert isinstance(y, pd.Series)
    assert len(y) == 16
    assert y.dtype == "float64"
    assert isinstance(X, pd.DataFrame)
    col_names = ["GNPDEFL", "GNP", "UNEMP", "ARMED", "POP"]
    assert X.columns.values.tolist() == col_names
    for col in col_names:
        assert X[col].dtype == "float64"
    assert len(X) == len(y)
