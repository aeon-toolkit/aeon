"""Test ClaSP plotting."""

import numpy as np
import pytest

from aeon.segmentation import ClaSPSegmenter
from aeon.testing.data_generation import make_example_pandas_series
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_series_with_profiles


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series_with_profiles():
    """Test whether plot_series_with_profiles runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    series = make_example_pandas_series(n_timepoints=50)
    clasp = ClaSPSegmenter()
    clasp.fit_predict(series)

    fig, ax = plot_series_with_profiles(
        series, clasp.profiles, true_cps=[25], found_cps=clasp.found_cps
    )
    plt.gcf().canvas.draw_idle()

    assert (
        isinstance(fig, plt.Figure)
        and isinstance(ax, np.ndarray)
        and all([isinstance(ax_, plt.Axes) for ax_ in ax])
    )

    plt.close()
