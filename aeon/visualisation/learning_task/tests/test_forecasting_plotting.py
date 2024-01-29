import pytest

from aeon.forecasting.model_selection import SlidingWindowSplitter
from aeon.testing.utils.series import _make_series
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_series_windows


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series_windows():
    """Test whether plot_series_windows runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    series = _make_series()
    cv = SlidingWindowSplitter(fh=5, window_length=10, step_length=5)

    fig, ax = plot_series_windows(series, cv)
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    plt.close()
