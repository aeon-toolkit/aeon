"""Alignment path plotting utilities."""

import numpy as np

from aeon.distances import cost_matrix as compute_cost_matrix
from aeon.distances._distance import alignment_path, pairwise_distance
from aeon.utils.validation._dependencies import _check_soft_dependencies


def _path_mask(cost_matrix, path, ax, theme=None):  # pragma: no cover
    _check_soft_dependencies("matplotlib")

    import matplotlib.colors as colorplt

    if theme is None:
        theme = colorplt.LinearSegmentedColormap.from_list("", ["#c9cacb", "white"])

    plot_matrix = np.zeros_like(cost_matrix)
    max_size = max(cost_matrix.shape)
    for i in range(max_size):
        for j in range(max_size):
            if (i, j) in path:
                plot_matrix[i, j] = 1.0
            elif cost_matrix[i, j] == np.inf:
                plot_matrix[i, j] = 0.0
            else:
                plot_matrix[i, j] = 0.25

    for i in range(max_size):
        for j in range(max_size):
            c = cost_matrix[j, i]
            ax.text(i, j, str(round(c, 2)), va="center", ha="center", size=10)
            ax.text(i, j, str(round(c, 2)), va="center", ha="center", size=10)

    ax.matshow(plot_matrix, cmap=theme)


def _pairwise_path(x, y, method):  # pragma: no cover
    pw_matrix = pairwise_distance(x, y, method=method)
    path = []
    for i in range(pw_matrix.shape[0]):
        for j in range(pw_matrix.shape[1]):
            if i == j:
                path.append((i, j))
    return path, pw_matrix.trace(), pw_matrix


def _plot_path(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    dist_kwargs: dict | None = None,
    title: str = "",
    plot_over_pw: bool = False,
):  # pragma: no cover
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    if dist_kwargs is None:
        dist_kwargs = {}
    try:
        path, dist = alignment_path(x, y, method=method, **dist_kwargs)
        cost_matrix = compute_cost_matrix(x, y, method=method, **dist_kwargs)

        if method == "lcss":
            _path = []
            for tup in path:
                _path.append(tuple(x + 1 for x in tup))
            path = _path

        if plot_over_pw is True:
            if method == "lcss":
                pw = pairwise_distance(x, y, method="euclidean")
                cost_matrix = np.zeros_like(cost_matrix)
                cost_matrix[1:, 1:] = pw
            else:
                pw = pairwise_distance(x, y, method="squared")
                cost_matrix = pw
    except NotImplementedError:
        path, dist, cost_matrix = _pairwise_path(x, y, method)

    plt.figure(1, figsize=(8, 8))
    x_size = x.shape[0]

    # definitions for the axes
    left, bottom = 0.01, 0.1
    w_ts = h_ts = 0.2
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02

    rect_s_y = (left, bottom, w_ts, height)
    rect_gram = (left_h, bottom, width, height)
    rect_s_x = (left_h, bottom_h, width, h_ts)

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    _path_mask(cost_matrix, path, ax_gram)
    ax_gram.axis("off")
    ax_gram.autoscale(False)
    # ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "w-",
    #              linewidth=3.)

    ax_s_x.plot(np.arange(x_size), y, "b-", linewidth=3.0, color="#818587")
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, x_size - 1))

    ax_s_y.plot(-x, np.arange(x_size), "b-", linewidth=3.0, color="#818587")
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, x_size - 1))

    ax_s_x.set_title(title, size=10)

    return plt


def _plot_alignment(
    x, y, method, dist_kwargs: dict | None = None, title: str = ""
):  # pragma: no cover
    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    if dist_kwargs is None:
        dist_kwargs = {}
    try:
        path, dist = alignment_path(x, y, method=method, **dist_kwargs)
    except NotImplementedError:
        path, dist, cost_matrix = _pairwise_path(x, y, method)

    plt.figure(1, figsize=(8, 8))

    plt.plot(x, "b-", color="black")
    plt.plot(y, "g-", color="black")

    for positions in path:
        try:
            plt.plot(
                [positions[0], positions[1]],
                [x[positions[0]], y[positions[1]]],
                "--",
                color="#818587",
            )
        except Exception:
            continue
    plt.title(title)

    plt.tight_layout()
    return plt
