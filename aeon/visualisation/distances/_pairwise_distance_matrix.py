__all__ = [
    "plot_pairwise_distance_matrix",
]

__maintainer__ = []

import numpy as np

from aeon.utils.validation._dependencies import _check_soft_dependencies


def plot_pairwise_distance_matrix(
    distance_matrix,
    a,
    b,
    path,
):
    """Plot a pairwise distance matrix between two time series.

    Parameters
    ----------
    distance_matrix : np.ndarray
        The pairwise distance matrix to plot.
    a : np.ndarray
        The first time series.
    b : np.ndarray
        The second time series.
    path : list of tuple
        The path of the minimum distances.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object with the plot.
    """
    # Checks availability of plotting libraries
    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(1, figsize=(10, 10))
    if len(a.shape) == 1:
        a_size = ts_length = len(a)
    else:
        a_size, ts_length = a.shape

    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02
    width_gram = width if a_size < 16 else width + 0.15

    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width_gram, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    # Plot the distance matrix with values.
    ax = sns.heatmap(
        distance_matrix,
        annot=True if a_size < 16 else False,
        fmt=".2f",
        cmap="viridis",
        ax=ax_gram,
        cbar=False if a_size < 16 else True,
        mask=np.isinf(distance_matrix),
    )

    for i in range(len(path)):
        ax.add_patch(
            plt.Rectangle(
                path[i][::-1],
                1,
                1,
                fill=False if a_size < 16 else True,
                edgecolor="red",
                facecolor="red",
                lw=2,
            )
        )

    ax_gram.axis("off")

    if b.ndim == 1:
        b = b[np.newaxis, :]  # reshape (1, len(b))

    # Plot the orange time series b.
    for i in range(b.shape[0]):

        ax_s_x.plot(
            np.linspace(0.5, ts_length - 1.5, ts_length),
            b[i],
            linewidth=2.0,
            color="orange",
            alpha=0.8,
        )

    ax_s_x.set_xlim((0, a_size - 1))
    ax_s_x.set_title(r"$\mathbf{b}$", fontsize=15)
    ax_s_x.axis("off")

    if a.ndim == 1:
        a = a[np.newaxis, :]

    # Plot the blue time series a.
    for i in range(a.shape[0]):
        ax_s_y.plot(
            -a[i][::-1],
            np.linspace(0.5, ts_length - 1.5, ts_length),
            "-",
            linewidth=3.0,
            color="blue",
        )
    ax_s_y.set_ylim((0, a_size - 1))
    ax_s_y.set_ylabel(r"$\mathbf{a}$", fontsize=15)

    ax_s_y.spines[:].set_visible(False)
    ax_s_y.set_xticks([])
    ax_s_y.set_yticks([])

    return ax


# %%
