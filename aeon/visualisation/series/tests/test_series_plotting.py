"""Test functionality of time series plotting functions."""

import numpy as np
import pandas as pd
import pytest

from aeon.datasets import load_airline
from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_pandas_series,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import (
    plot_correlations,
    plot_lags,
    plot_series,
    plot_spectrogram,
)

y_np_array = make_example_1d_numpy()
y_pd_series = make_example_pandas_series()
y_airline = load_airline(return_array=False)
y_airline_train = y_airline.iloc[y_airline.index < "1960-01"]
y_airline_test = y_airline.iloc[y_airline.index >= "1960-01"]
y_dataframe = pd.DataFrame(
    {y_airline_train.name: y_airline_train, y_airline_test.name: y_airline_test}
)
series_to_test = [y_np_array, y_pd_series, y_dataframe]


# can be used with pytest.mark.parametrize to check plots that accept
# univariate series
univariate_plots = [plot_correlations, plot_lags]

longer_series = [np.random.random(100), pd.Series(np.random.random(100))]
bad_series = [
    "This is a string",
    (pd.DataFrame({"y1": y_airline, "y2": y_airline}),),
    np.random.random((4, 1, 50)),
]


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("series", series_to_test)
def test_plot_series(series):
    """Test whether plot_series runs without error."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    fig, ax = plot_series(series)
    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series_with_arguments():
    """Test whether plot_series runs with additional arguments."""
    import matplotlib.pyplot as plt

    series = make_example_1d_numpy()
    fig, ax = plot_series(
        series,
        labels=["Series 1"],
        markers=["x"],
        title="FOOBAR",
        x_label="FOO",
        y_label="BAR",
    )
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("bad_series", bad_series)
def test_plot_series_invalid_input_type_raises_error(bad_series):
    """Tests whether plot_series raises error for invalid input types."""
    with pytest.raises((ValueError), match="series must be a single time series"):
        plot_series(bad_series)


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series_invalid_marker():
    """Tests whether plot_series raises error for inconsistent series/markers."""
    match = """There must be one marker for each time series,
                but found inconsistent numbers of series and
                markers."""
    series = make_example_1d_numpy()
    with pytest.raises(ValueError, match=match):
        # Generate error by creating list of markers with length that does
        # not match input number of input series
        markers = ["o", "o"]
        plot_series(series, markers=markers)
    series = make_example_2d_numpy_series(n_channels=2)
    with pytest.raises(ValueError, match=match):
        # Generate error by creating list of markers with length that does
        # not match input number of input series
        markers = ["o"]
        plot_series(series, markers=markers)


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series_invalid_label():
    """Tests whether plot_series raises error for inconsistent series/labels."""
    match = """There must be one label for each time series,
                but found inconsistent numbers of series and
                labels."""
    with pytest.raises(ValueError, match=match):
        series = make_example_1d_numpy()
        labels = ["Series 1", "Series 2"]
        plot_series(series, labels=labels)

    with pytest.raises(ValueError, match=match):
        series = make_example_2d_numpy_series(n_channels=2)
        labels = ["Series 1"]
        plot_series(series, labels=labels)


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_series_existing_axes():
    """Tests whether plot_series works with existing axes as input."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    # Test output case where an existing plt.Axes object is passed to kwarg ax
    series = make_example_1d_numpy()
    fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
    ax = plot_series(series, ax=ax)
    assert isinstance(ax, plt.Axes)
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("series_to_plot", longer_series)
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
@pytest.mark.parametrize("series_to_plot", bad_series + [y_dataframe])
@pytest.mark.parametrize("plot_func", univariate_plots)
def test_univariate_plots_invalid_input_type_raises_error(
    series_to_plot,
    plot_func,
):
    """Tests whether plots that accept univariate series run without error.

    Generically test whether plots only accepting univariate input raise an error when
    invalid input type is found. Currently only plot_lags and plot_correlations are
    tested.
    """

    def test_plot_series_invalid_input_type_raises_error(bad_series):
        """Tests whether plot_series raises error for invalid input types."""
        with pytest.raises((ValueError), match="series must be a single time series"):
            plot_func(bad_series)


# For plots that only accept univariate input, from here onwards are
# tests specific to a given plot. E.g. to test specific arguments or functionality.
@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("lags", [2, (1, 2, 3)])
def test_plot_lags_arguments(lags):
    """Tests whether plot_lags run with different input arguments."""
    import matplotlib
    import matplotlib.pyplot as plt

    series_to_plot = make_example_1d_numpy(n_timepoints=100)
    matplotlib.use("Agg")

    plot_lags(series_to_plot, lags=lags, suptitle="Lag Plot")
    plt.gcf().canvas.draw_idle()
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("lags", [6, 24])
def test_plot_correlations_arguments(lags):
    """Tests whether plot_correlations run with different input arguments."""
    import matplotlib
    import matplotlib.pyplot as plt

    series_to_plot = make_example_1d_numpy(n_timepoints=100)

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

    series_to_plot = make_example_1d_numpy(n_timepoints=100)

    matplotlib.use("Agg")
    fig, ax = plot_spectrogram(series_to_plot, fs=1)
    plt.gcf().canvas.draw_idle()
    plt.close()

    assert fig is not None
    assert ax is not None

    fig, ax = plot_spectrogram(y_airline, fs=1, return_onesided=False)
    plt.gcf().canvas.draw_idle()
    plt.close()
    assert fig is not None
    assert ax is not None
