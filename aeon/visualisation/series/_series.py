"""Common timeseries plotting functionality."""

__all__ = ["plot_series", "plot_lags", "plot_correlations", "plot_spectrogram"]
__maintainer__ = []

import math
from warnings import warn

import numpy as np
import pandas as pd
from scipy.fft import fftshift
from scipy.signal import spectrogram

from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.utils.validation.series import is_single_series, is_univariate_series


def plot_series(
    series,
    labels=None,
    markers=None,
    colors=None,
    title=None,
    x_label=None,
    y_label=None,
    ax=None,
    pred_interval=None,
):
    """Plot one or more time series.

    Parameters
    ----------
    series : np.ndarray, pd.Series or pd.DataFrame
        One or more time series stored in `(n_channels, n_timepoints)` format.
    labels : list, default = None
        Names of series, will be displayed in figure legend.
    markers : list, default = None
        Markers of data points, if None the marker "o" is used by default.
        The length of the list has to match with the number of series.
    colors : list, default = None
        The colors to use for plotting each series. Must contain one color per series
    title : str, default = None
        The text to use as the figure's suptitle.
    x_label : str or None, default = None
       String label to put on the x-axis.
    y_label : str or None, default = None
       String label to put on the -axis.
    ax : plt.Axis or None
        Axis to plot on. If None, a new figure is created.
    pred_interval : pd.DataFrame, default = None
        Contains columns for lower and upper boundaries of confidence interval.

    Returns
    -------
    fig : plt.Figure
    ax : plt.Axis

    Examples
    --------
    >>> from aeon.visualisation import plot_series
    >>> from aeon.datasets import load_airline
    >>> y = load_airline(return_array=False)
    >>> fig, ax = plot_series(y)  # doctest: +SKIP
    """
    if not is_single_series(series):
        raise ValueError(
            "series must be a single time series, either univariate (1D "
            "np.ndarray or pd.Series) or multivariate (2D np.ndarray or pd.DataFrame)"
            "of shape (n_channels, n_timepoints)"
        )

    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cbook import flatten
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    # Convert single multivariate series in wide format to list of series
    if isinstance(series, pd.DataFrame):
        series = [row for _, row in series.iterrows()]
    elif isinstance(series, pd.Series):
        series = [series]
    elif isinstance(series, np.ndarray):
        series = series.squeeze()
        if series.ndim > 1:
            series = [pd.Series(row) for row in series]
        else:
            series = [pd.Series(series)]
    n_series = len(series)
    # labels
    if labels is not None:
        if n_series != len(labels):
            raise ValueError(
                """There must be one label for each time series,
                but found inconsistent numbers of series and
                labels."""
            )
        legend = True
    else:
        labels = ["" for _ in range(n_series)]
        legend = False

    # markers
    if markers is not None:
        if n_series != len(markers):
            raise ValueError(
                """There must be one marker for each time series,
                but found inconsistent numbers of series and
                markers."""
            )
    else:
        markers = ["o" for _ in range(n_series)]

    # create combined index
    index = series[0].index

    # generate integer x-values
    xs = [np.argwhere(index.isin(y.index)).ravel() for y in series]

    # create figure if no ax provided for plotting
    local_ax = ax
    if ax is None:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))

    # colors
    if colors is None or not _check_colors(colors, n_series):
        colors = sns.color_palette("colorblind", n_colors=n_series)

    # plot series
    for x, y, color, label, marker in zip(xs, series, colors, labels, markers):
        # scatter if little data is available or index is not complete
        if len(x) <= 3 or not np.array_equal(np.arange(x[0], x[-1] + 1), x):
            plot_func = sns.scatterplot
        else:
            plot_func = sns.lineplot

        plot_func(x=x, y=y, ax=ax, marker=marker, label=label, color=color)

    # combine data points for all series
    xs_flat = list(flatten(xs))

    # set x label of data point to the matching index
    def format_fn(tick_val, tick_pos):
        if int(tick_val) in xs_flat:
            return index[int(tick_val)]
        else:
            return ""

    # dynamically set x label ticks and spacing from index labels
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set the figure's title
    if title is not None:
        fig.suptitle(title, size="xx-large")

    # Label the x and y axes
    if x_label is not None:
        ax.set_xlabel(x_label)

    _y_label = y_label if y_label is not None else series[0].name
    ax.set_ylabel(_y_label)

    if legend:
        ax.legend()
    if pred_interval is not None:
        ax = _plot_interval(ax, pred_interval)

    if local_ax is None:
        return fig, ax
    else:
        return ax


def _plot_interval(ax, interval_df):
    cov = interval_df.columns.levels[1][0]
    ax.fill_between(
        ax.get_lines()[-1].get_xdata(),
        interval_df["Coverage"][cov]["lower"].astype("float64"),
        interval_df["Coverage"][cov]["upper"].astype("float64"),
        alpha=0.2,
        color=ax.get_lines()[-1].get_c(),
        label=f"{int(cov * 100)}% prediction interval",
    )
    ax.legend()
    return ax


def _check_colors(colors, n_series):
    """Verify color list is correct length and contains only colors."""
    from matplotlib.colors import is_color_like

    if n_series == len(colors) and all([is_color_like(c) for c in colors]):
        return True
    warn(
        "Color list must be same length as `series` and contain only matplotlib colors"
    )
    return False


def plot_lags(series, lags=1, suptitle=None):
    """Plot one or more lagged versions of a time series.

    A lag plot is a scatter plot of a time series against a lag of itself.
    It is normally used to check for autocorrelation.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Single univariate  ime series for plotting lags.
    lags : int or array-like, default=1
        The lag or lags to plot.

        - int plots the specified lag
        - array-like  plots specified lags in the array/list
    suptitle : str, default=None
        The text to use as the Figure's suptitle. If None, then the title
        will be "Plot of series against lags {lags}"

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of the figure's Axe objects

    Examples
    --------
    >>> from aeon.visualisation import plot_lags
    >>> from aeon.datasets import load_airline
    >>> y = load_airline(return_array=False)
    >>> fig, ax = plot_lags(y, lags=2) # plot of y(t) with y(t-2)  # doctest: +SKIP
    >>> fig, ax = plot_lags(y, lags=[1,2,3]) # y(t) & y(t-1), y(t-2).. # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    if not (is_single_series(series) and is_univariate_series(series)):
        raise ValueError("series must be a single univariate time series")

    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    if isinstance(lags, int):
        single_lag = True
        lags = [lags]
    elif isinstance(lags, (tuple, list, np.ndarray)):
        single_lag = False
    else:
        raise ValueError("`lags should be an integer, tuple, list, or np.ndarray.")

    length = len(lags)
    n_cols = min(3, length)
    n_rows = math.ceil(length / n_cols)
    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(8, 6 * n_rows),
        sharex=True,
        sharey=True,
    )
    if single_lag:
        axes = ax
        pd.plotting.lag_plot(series, lag=lags[0], ax=axes)
    else:
        axes = ax.ravel()
        for i, val in enumerate(lags):
            pd.plotting.lag_plot(series, lag=val, ax=axes[i])

    if suptitle is None:
        fig.suptitle(
            f"Plot of series against lags {', '.join([str(lag) for lag in lags])}",
            size="xx-large",
        )
    else:
        fig.suptitle(suptitle, size="xx-large")

    return fig, np.array(fig.get_axes())


def plot_correlations(
    series,
    lags=24,
    alpha=0.05,
    zero_lag=True,
    acf_fft=False,
    acf_adjusted=True,
    pacf_method="ywadjusted",
    suptitle=None,
    series_title=None,
    acf_title="Autocorrelation",
    pacf_title="Partial Autocorrelation",
):
    """Plot series and its ACF and PACF values.

    Parameters
    ----------
    series : pd.Series
        A time series.
    lags : int, default = 24
        Number of lags to include in ACF and PACF plots
    alpha : int, default = 0.05
        Alpha value used to set confidence intervals. Alpha = 0.05 results in
        95% confidence interval with standard deviation calculated via
        Bartlett's formula.
    zero_lag : bool, default = True
        If True, start ACF and PACF plots at 0th lag
    acf_fft : bool,  = False
        Whether to compute ACF via FFT.
    acf_adjusted : bool, default = True
        If True, denonimator of ACF calculations uses n-k instead of n, where
        n is number of observations and k is the lag.
    pacf_method : str, default = 'ywadjusted'
        Method to use in calculation of PACF.
    suptitle : str, default = None
        The text to use as the Figure's suptitle.
    series_title : str, default = None
        Used to set the title of the series plot if provided. Otherwise, series
        plot has no title.
    acf_title : str, default = 'Autocorrelation'
        Used to set title of ACF plot.
    pacf_title : str, default = 'Partial Autocorrelation'
        Used to set title of PACF plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        Array of the figure's Axe objects

    Examples
    --------
    >>> from aeon.visualisation import plot_correlations
    >>> from aeon.datasets import load_airline
    >>> y = load_airline(return_array=False)
    >>> fig, ax = plot_correlations(y)  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib", "statsmodels")
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    if not (is_single_series(series) and is_univariate_series(series)):
        raise ValueError("series must be a single univariate time series")

    # Setup figure for plotting
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    f_ax1 = fig.add_subplot(gs[0, :])
    if series_title is not None:
        f_ax1.set_title(series_title)
    f_ax2 = fig.add_subplot(gs[1, 0])
    f_ax3 = fig.add_subplot(gs[1, 1])

    # Create expected plots on their respective Axes
    plot_series(series, ax=f_ax1)
    plot_acf(
        series,
        ax=f_ax2,
        lags=lags,
        zero=zero_lag,
        alpha=alpha,
        title=acf_title,
        adjusted=acf_adjusted,
        fft=acf_fft,
    )
    plot_pacf(
        series,
        ax=f_ax3,
        lags=lags,
        zero=zero_lag,
        alpha=alpha,
        title=pacf_title,
        method=pacf_method,
    )
    if suptitle is not None:
        fig.suptitle(suptitle, size="xx-large")

    return fig, np.array(fig.get_axes())


def plot_spectrogram(series, fs=1, return_onesided=True):
    """
    Plot the spectrogram of a given time series.

    Parameters
    ----------
    series : array_like
        Input time series.
    fs : float, Default is 1.
        Sampling frequency of the input series (in Hz).
    return_onesided : bool, Default is True.
        Whether to return one-sided spectrum.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    ax : matplotlib.axes._axes.Axes
        The axes of the plot.

    Examples
    --------
    >>> from aeon.visualisation import plot_spectrogram
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> fig, ax = plot_spectrogram(y)  # doctest: +SKIP
    """
    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    if not (is_single_series(series) and is_univariate_series(series)):
        raise ValueError("series must be a single univariate time series")
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    fig, ax = plt.subplots()

    _, _, _spectrogram = spectrogram(series, fs=fs, return_onesided=return_onesided)
    if not return_onesided:
        ax.pcolormesh(fftshift(_spectrogram, axes=0))
    else:
        ax.pcolormesh(_spectrogram)

    ax.set_ylabel("Frequency [Hz]")
    return fig, ax
