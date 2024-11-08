__all__ = [
    "plot_series_with_profiles",
]

__maintainer__ = []

import numpy as np

from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.utils.validation.series import check_series


def plot_series_with_profiles(
    ts,
    profiles,
    true_cps=None,
    found_cps=None,
    score_name="ClaSP Score",
    title=None,
    font_size=16,
):
    """Plot the TS with the known and found change points and profiles.

    Parameters
    ----------
    ts: array-like, shape=[n]
        the univariate time series of length n to be annotated.
        the time series is plotted as the first subplot.
    profiles: array-like, shape=[n-m+1, n_cpts], dtype=float
        the n_cpts profiles computed by the method used
        the profiles are plotted as subsequent subplots to the time series.
    true_cps: array-like, dtype=int, default=None
        the known change points.
        these are highlighted in the time series subplot as vertical lines
    found_cps: array-like, shape=[n_cpts], dtype=int, default=None
        the found change points
        these are highlighted in the profiles subplot as vertical lines
    score_name: str, default="ClaSP Score
        name of the scoring method used, i.e. 'ClaSP'
    title: str, default=None
        the name of the time series (dataset) to be annotated
    font_size: int, default=16
        for plotting

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with 1 + len(profiles) subplots, one for the time series
        and others for each profile
    axes : np.ndarray
        Array of the figure's Axe objects
    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt

    ts = check_series(ts)

    fig, ax = plt.subplots(
        len(profiles) + 1,
        1,
        sharex=True,
        gridspec_kw={"hspace": 0.05},
        figsize=(20, 5 * len(profiles)),
    )
    ax = ax.reshape(-1)

    if true_cps is not None:
        segments = [0] + list(true_cps) + [ts.shape[0]]
        for idx in np.arange(0, len(segments) - 1):
            ax[0].plot(
                np.arange(segments[idx], segments[idx + 1]),
                ts[segments[idx] : segments[idx + 1]],
            )
    else:
        ax[0].plot(np.arange(ts.shape[0]), ts)

    for i, profile in enumerate(profiles):
        ax[i + 1].plot(np.arange(len(profile)), profile, color="b")
        ax[i + 1].set_ylabel(f"{score_name} {i}. Split", fontsize=font_size)

    ax[-1].set_xlabel("split point $s$", fontsize=font_size)

    # Set the figure's title
    if title is not None:
        ax[0].set_title(title, fontsize=font_size)

    for a in ax:
        for tick in a.xaxis.get_major_ticks():
            if hasattr(tick, "label"):
                tick.label.set_fontsize(font_size)
            if hasattr(tick, "label1"):
                tick.label1.set_fontsize(font_size)
            if hasattr(tick, "label2"):
                tick.label2.set_fontsize(font_size)

        for tick in a.yaxis.get_major_ticks():
            if hasattr(tick, "label"):
                tick.label.set_fontsize(font_size)
            if hasattr(tick, "label1"):
                tick.label1.set_fontsize(font_size)
            if hasattr(tick, "label2"):
                tick.label2.set_fontsize(font_size)

    if true_cps is not None:
        for idx, true_cp in enumerate(true_cps):
            ax[0].axvline(
                x=true_cp,
                linewidth=2,
                color="r",
                label="True Change Point" if idx == 0 else None,
            )

    if found_cps is not None:
        for idx, found_cp in enumerate(found_cps):
            ax[0].axvline(
                x=found_cp,
                linewidth=2,
                color="g",
                label="Predicted Change Point" if idx == 0 else None,
            )
            ax[idx + 1].axvline(
                x=found_cp,
                linewidth=2,
                color="g",
                label="Predicted Change Point" if idx == 0 else None,
            )

    ax[0].legend(prop={"size": font_size})

    return fig, ax
