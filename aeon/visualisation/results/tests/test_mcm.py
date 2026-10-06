"""Test the multi-comparison-matrix visualisation."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation.results._mcm import create_multi_comparison_matrix


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_mcm():
    """Test the multi-comparison-matrix visualisation."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        np.random.rand(10, 3),  # 10 rows, 3 columns of random numbers
        columns=["Classifier1", "Classifier2", "Classifier3"],
    )
    fig = create_multi_comparison_matrix(df)
    assert isinstance(fig, plt.Figure)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_mcm_original_pvalue():
    """Test the multi-comparison-matrix visualisation."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        np.random.rand(10, 3),  # 10 rows, 3 columns of random numbers
        columns=["Classifier1", "Classifier2", "Classifier3"],
    )
    fig = create_multi_comparison_matrix(
        df, pvalue_test_params={"zero_method": "pratt", "alternative": "two-sided"}
    )
    assert isinstance(fig, plt.Figure)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_mcm_file_save():
    """Test file save  in different formats."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        np.random.rand(10, 3),  # 10 rows, 3 columns of random numbers
        columns=["Classifier1", "Classifier2", "Classifier3"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "test")
        fig = create_multi_comparison_matrix(
            df,
            save_path=prefix,
            formats=("pdf", "png", "csv", "json", "tex"),
            pvalue_correction="Holm",
        )
        assert isinstance(fig, plt.Figure)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "order_stats",
    ["average-statistic", "average-rank", "max-wins", "amean-amean", "pvalue"],
)
def test_mcm_order_stats(order_stats):
    """Test every order_stats option."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        np.random.rand(10, 3),  # 10 rows, 3 columns of random numbers
        columns=["Classifier1", "Classifier2", "Classifier3"],
    )
    fig = create_multi_comparison_matrix(df, order_stats=order_stats)
    assert isinstance(fig, plt.Figure)
