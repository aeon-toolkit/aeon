__all__ = [
    "plot_sax_representation",
]

import numpy as np


def plot_sax_representation(
    X,
    X_sax,
    X_inverse,
    sax,
    series_index=0,
    channel_index=0,
    window_index=None,
):
    """Plot a standard SAX word or one selected word from windowed SAX."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.stats import norm

    n_timepoints = X.shape[-1]
    timepoints = np.arange(n_timepoints)

    original_series = X[series_index, channel_index]
    inverse_series = X_inverse[series_index, channel_index]
    is_windowed = X_sax.ndim == 4

    if is_windowed:
        if window_index is None:
            window_index = 0

        n_windows = X_sax.shape[2]
        if not 0 <= window_index < n_windows:
            raise IndexError(f"window_index must be between 0 and {n_windows - 1}")

        window_size = sax.window_size
        stride = window_size if sax.stride is None else sax.stride
        window_start = window_index * stride
        window_end = min(window_start + window_size, n_timepoints)

        segment_symbols = X_sax[
            series_index,
            channel_index,
            window_index,
        ]

        local_boundaries = np.linspace(
            0,
            window_size,
            sax.n_segments + 1,
            dtype=int,
        )
        segment_boundaries = window_start + local_boundaries
        segment_boundaries[-1] = window_end
    else:
        n_windows = None
        window_start = 0
        window_end = n_timepoints
        segment_symbols = X_sax[series_index, channel_index]
        segment_boundaries = np.linspace(
            0,
            n_timepoints,
            sax.n_segments + 1,
            dtype=int,
        )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        timepoints,
        original_series,
        label="Original series",
        color="#86BBD8",
        linewidth=3.5,
        zorder=2,
    )
    ax.plot(
        timepoints,
        inverse_series,
        label="Inverse SAX approximation",
        color="#D95F59",
        linestyle=":",
        linewidth=4,
        alpha=0.75,
        zorder=3,
    )

    if is_windowed:
        ax.axvspan(
            window_start,
            window_end - 1,
            color="#F4D6A0",
            alpha=0.18,
            zorder=0,
            label=f"SAX window {window_index + 1}",
        )
        ax.axvline(
            window_start,
            color="black",
            linewidth=1.4,
            alpha=0.5,
            zorder=1,
        )
        ax.axvline(
            window_end - 1,
            color="black",
            linewidth=1.4,
            alpha=0.5,
            zorder=1,
        )

    for segment_index, symbol in enumerate(segment_symbols):
        start = segment_boundaries[segment_index]
        end = segment_boundaries[segment_index + 1]

        if end <= start:
            continue

        if segment_index > 0:
            ax.axvline(
                start,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.4,
                zorder=1,
            )

        center = (start + end - 1) / 2
        midpoint = min(int(round(center)), n_timepoints - 1)
        segment_value = inverse_series[midpoint]

        ax.annotate(
            str(symbol),
            xy=(center, segment_value),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color="black",
            zorder=20,
            annotation_clip=False,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )

    for breakpoint_index, breakpoint in enumerate(sax.breakpoints):
        ax.axhline(
            breakpoint,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.55,
            zorder=1,
        )
        ax.text(
            int(0.05 * n_timepoints),
            breakpoint,
            rf"$\beta_{{{breakpoint_index + 1}}} = {breakpoint:.2f}$",
            ha="right",
            va="bottom",
            fontsize=9,
            color="dimgray",
            zorder=8,
        )

    title = "Original series and SAX approximation"
    if is_windowed:
        title += (
            f" — window {window_index + 1}/{n_windows} "
            f"[{window_start}:{window_end}]"
        )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Value")
    ax.legend(loc="lower center")
    ax.grid(alpha=0.15)

    divider = make_axes_locatable(ax)
    ax_gaussian = divider.append_axes(
        "left",
        size="18%",
        pad=0.12,
        sharey=ax,
    )

    y_min, y_max = ax.get_ylim()
    y_values = np.linspace(y_min, y_max, 1000)
    scale = sax.distribution_params_.get("scale", 1.0)
    gaussian_density = norm.pdf(y_values, loc=0.0, scale=scale)

    ax_gaussian.plot(
        gaussian_density,
        y_values,
        color="gray",
        linewidth=2,
    )
    ax_gaussian.fill_betweenx(
        y_values,
        0,
        gaussian_density,
        color="lightgray",
        alpha=0.45,
    )

    for breakpoint in sax.breakpoints:
        ax_gaussian.axhline(
            breakpoint,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.55,
        )

    ax_gaussian.invert_xaxis()
    ax_gaussian.set_xlabel("Density", fontsize=9)
    ax_gaussian.set_ylabel("Gaussian distribution")
    ax_gaussian.tick_params(axis="x", labelsize=8)
    ax_gaussian.tick_params(axis="y", labelleft=True)
    ax_gaussian.spines["top"].set_visible(False)
    ax_gaussian.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, ax
