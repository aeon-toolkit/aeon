"""Test the multi-comparison-matrix visualisation."""

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
    fig = create_multi_comparison_matrix(df, save_as_json=False)
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
        fig = create_multi_comparison_matrix(
            df,
            output_dir=tmp,
            pdf_savename="test1",
            png_savename="test",
            tex_savename="test",
            save_as_json=True,
            pvalue_correction="Holm",
        )
        assert isinstance(fig, plt.Figure)


#        assert os.path.isfile(os.path.join(tmp, "test1.pdf"))
