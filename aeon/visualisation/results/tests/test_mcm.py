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
    """Test the multi-comparison-matrix visualisation returns a Matplotlib figure."""
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
    """Test that the MCM outputs files in the configured formats."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(
        np.random.rand(10, 3),
        columns=["Classifier1", "Classifier2", "Classifier3"],
    )
    with tempfile.TemporaryDirectory() as tmp:
        fig = create_multi_comparison_matrix(
            df,
            output_config={
                "save_path": tmp,
                "base_name": "test",
                "formats": ["pdf", "png", "tex"],
            },
            save_as_json=True,
            pvalue_correction="Holm",
        )
        assert isinstance(fig, plt.Figure)

        pdf_path = os.path.join(tmp, "test.pdf")
        png_path = os.path.join(tmp, "test.png")
        tex_path = os.path.join(tmp, "test.tex")

        assert os.path.isfile(pdf_path), f"PDF file not found at {pdf_path}"
        assert os.path.isfile(png_path), f"PNG file not found at {png_path}"
        assert os.path.isfile(tex_path), f"TeX file not found at {tex_path}"
