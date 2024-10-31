"""Test functionality of time series plotting functions."""

import re

import numpy as np
import pandas as pd
import pytest

from aeon.datasets import load_airline
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.utils.validation.series import VALID_DATA_TYPES
from aeon.visualisation import (
    plot_correlations,
    plot_lags,
    plot_series,
    plot_spectrogram,
)

y_airline = load_airline(return_array=False)
y_airline_true = y_airline.iloc[y_airline.index < "1960-01"]
y_airline_test = y_airline.iloc[y_airline.index >= "1960-01"]
series_to_test = [y_airline, (y_airline_true, y_airline_test)]

# can be used with pytest.mark.parametrize to check plots that accept
# univariate series
univariate_plots = [plot_correlations, plot_lags]


@pytest.fixture
def valid_data_types():
    """Filter valid data types for those that work with plotting functions."""
    valid_data_types = tuple(filter(lambda x: x is not np.ndarray, VALID_DATA_TYPES))
    return valid_data_types


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series():
    """Test whether plot_series runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    series = np.random.random((50,))
    fig, ax = plot_series(series)
    # plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    series = np.array([1, 2, 3])
    fig, ax = plot_series(series)
    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    series = pd.Series(series)
    fig, ax = plot_series(series)
    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    # # Test with labels specified
    fig, ax = plot_series(series, labels=["Series 1"])
    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    # # Test with markers
    series = np.array([1, 2, 3, 4, 5, 6])
    fig, ax = plot_series(series, markers=["x"])
    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    # Test with multivariate series
    series = [np.random.random((1, 50)) for _ in range(3)]
    fig, ax = plot_series(series, title="FOOBAR")
    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    series = np.random.random((4, 50))
    fig, ax = plot_series(series, title="FOOBAR", x_label="FOO", y_label="BAR")
    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    plt.close()


invalid_input_types = [
    pd.DataFrame({"y1": y_airline, "y2": y_airline}),
    "this_is_a_string",
]


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series_invalid_input_type_raises_error():
    """Tests whether plot_series raises error for invalid input types."""
    series_to_plot = "This is a string"
    with pytest.raises((TypeError), match="found type: <class 'str'>"):
        plot_series(series_to_plot)
    series_to_plot = (pd.DataFrame({"y1": y_airline, "y2": y_airline}),)
    with pytest.raises(ValueError, match="input must be univariate"):
        plot_series(series_to_plot)


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize(
    "series_to_plot", [(y_airline_true, y_airline_test.reset_index(drop=True))]
)
def test_plot_series_with_unequal_index_type_raises_error(
    series_to_plot, valid_data_types
):
    """Tests whether plot_series raises error for series with unequal index."""
    match = "Found series with inconsistent index types"
    with pytest.raises(TypeError, match=match):
        plot_series(series_to_plot)


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("series_to_plot", series_to_test)
def test_plot_series_invalid_marker_kwarg_len_raises_error(series_to_plot):
    """Tests whether plot_series raises error for inconsistent series/markers."""
    match = """There must be one marker for each time series,
                but found inconsistent numbers of series and
                markers."""
    with pytest.raises(ValueError, match=match):
        # Generate error by creating list of markers with length that does
        # not match input number of input series
        if isinstance(series_to_plot, pd.Series):
            markers = ["o", "o"]
        elif isinstance(series_to_plot, tuple):
            markers = ["o" for _ in range(len(series_to_plot) - 1)]

        plot_series(series_to_plot, markers=markers)


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("series_to_plot", series_to_test)
def test_plot_series_invalid_label_kwarg_len_raises_error(series_to_plot):
    """Tests whether plot_series raises error for inconsistent series/labels."""
    match = """There must be one label for each time series,
                but found inconsistent numbers of series and
                labels."""
    with pytest.raises(ValueError, match=match):
        # Generate error by creating list of labels with length that does
        # not match input number of input series
        if isinstance(series_to_plot, pd.Series):
            labels = ["Series 1", "Series 2"]
        elif isinstance(series_to_plot, tuple):
            labels = [f"Series {i}" for i in range(len(series_to_plot) - 1)]

        plot_series(series_to_plot, labels=labels)


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("series_to_plot", series_to_test)
def test_plot_series_existing_axes(series_to_plot):
    """Tests whether plot_series works with existing axes as input."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    # Test output case where an existing plt.Axes object is passed to kwarg ax
    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    ax = plot_series(series_to_plot, ax=ax)

    assert isinstance(ax, plt.Axes)


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series_uniform_treatment_of_int64_range_index_types():
    """Verify that plot_series treats Int64 and Range indices equally."""
    # We test that int64 and range indices are treated uniformly and do not raise an
    # error of inconsistent index types
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    y1 = pd.Series(np.arange(10))
    y2 = pd.Series(np.random.normal(size=10))
    y1.index = pd.Index(y1.index, dtype=int)
    y2.index = pd.RangeIndex(y2.index)

    plot_series([y1, y2])
    plt.gcf().canvas.draw_idle()
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("series_to_plot", [y_airline])
@pytest.mark.parametrize("plot_func", univariate_plots)
def test_univariate_plots_run_without_error(series_to_plot, plot_func):
    """Tests whether plots that accept univariate series run without error.

    Generically test whether plots only accepting univariate input and outputting
    an array of axes run. Currently only plot_lags and plot_correlations are tested.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    fig, ax = plot_func(series_to_plot)
    plt.gcf().canvas.draw_idle()

    assert (
        isinstance(fig, plt.Figure)
        and isinstance(ax, np.ndarray)
        and all([isinstance(ax_, plt.Axes) for ax_ in ax])
    )

    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("series_to_plot", invalid_input_types)
@pytest.mark.parametrize("plot_func", univariate_plots)
def test_univariate_plots_invalid_input_type_raises_error(
    series_to_plot, plot_func, valid_data_types
):
    """Tests whether plots that accept univariate series run without error.

    Generically test whether plots only accepting univariate input raise an error when
    invalid input type is found. Currently only plot_lags and plot_correlations are
    tested.
    """
    if not isinstance(series_to_plot, (pd.Series, pd.DataFrame)):
        series_type = type(series_to_plot)
        match = (
            rf"input must be a one of {valid_data_types}, but found type: {series_type}"
        )
        with pytest.raises(TypeError, match=re.escape(match)):
            plot_func(series_to_plot)
    else:
        match = "input must be univariate, but found 2 variables."
        with pytest.raises(ValueError, match=match):
            plot_func(series_to_plot)


# For plots that only accept univariate input, from here onwards are
# tests specific to a given plot. E.g. to test specific arguments or functionality.
@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("series_to_plot", [y_airline])
@pytest.mark.parametrize("lags", [2, (1, 2, 3)])
def test_plot_lags_arguments(series_to_plot, lags):
    """Tests whether plot_lags run with different input arguments."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    plot_lags(series_to_plot, lags=lags, suptitle="Lag Plot")
    plt.gcf().canvas.draw_idle()
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("series_to_plot", [y_airline])
@pytest.mark.parametrize("lags", [6, 24])
def test_plot_correlations_arguments(series_to_plot, lags):
    """Tests whether plot_correlations run with different input arguments."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    plot_correlations(
        series_to_plot,
        lags=lags,
        suptitle="Correlation Plot",
        series_title="Time Series",
    )
    plt.gcf().canvas.draw_idle()
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_spectrogram():
    """Tests whether plot_spectrogram run with the given inputs or not."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    fig, ax = plot_spectrogram(y_airline, fs=1)
    plt.gcf().canvas.draw_idle()
    plt.close()

    assert fig is not None
    assert ax is not None

    fig, ax = plot_spectrogram(y_airline, fs=1, return_onesided=False)
    plt.gcf().canvas.draw_idle()
    plt.close()
    assert fig is not None
    assert ax is not None
