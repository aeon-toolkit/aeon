import matplotlib.pyplot as plt
import numpy as np

from aeon.distances import alignment_path, cost_matrix
from aeon.distances.elastic.soft._soft_dtw import soft_dtw_gradient


def plot_ts(series, color="#4A90E2", name=None):
    # Set figure with specific pixel size (e.g., 600x300 pixels)
    dpi = 100  # dots per inch
    width_pixels = 600
    height_pixels = 300
    width_inches = width_pixels / dpi
    height_inches = height_pixels / dpi

    plt.figure(figsize=(width_inches, height_inches), dpi=dpi, facecolor="white")

    # Create axis with white background
    ax = plt.gca()
    ax.set_facecolor("white")

    # Plot the series
    plt.plot(series, color=color, linewidth=2)

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Use tight layout but maintain figure size
    plt.tight_layout()

    if name is not None:
        plt.savefig(f"{name}.png", bbox_inches="tight", dpi=dpi)
    plt.show()


def plot_two_ts(series1, series2, color1="#4A90E2", color2="#E24A4A", name=None):
    # Set figure with specific pixel size (e.g., 600x300 pixels)
    dpi = 100  # dots per inch
    width_pixels = 600
    height_pixels = 300
    width_inches = width_pixels / dpi
    height_inches = height_pixels / dpi

    plt.figure(figsize=(width_inches, height_inches), dpi=dpi, facecolor="white")

    # Create axis with white background
    ax = plt.gca()
    ax.set_facecolor("white")

    # Plot both series without labels
    plt.plot(series1, color=color1, linewidth=2)
    plt.plot(series2, color=color2, linewidth=2)

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Use tight layout but maintain figure size
    plt.tight_layout()

    if name is not None:
        plt.savefig(f"{name}.png", bbox_inches="tight", dpi=dpi)
    plt.show()


if __name__ == "__main__":
    series1 = np.array([0.33, 0.45, 0.35, 0.32, 0.45])
    series2 = np.array([0.32, 0.15, 0.48, 0.35, 0.38])

    series1 = series1 * 10
    series2 = series2 * 10

    plot_ts(series1)
    plot_ts(series2, color="#E24A4A")
    plot_two_ts(series1, series2, name="series1_series2")

    cost_matrix_val = cost_matrix(series1, series2, method="soft_dtw", gamma=1.0)
    alignment = alignment_path(series1, series2, method="soft_dtw", gamma=1.0)
    grad, dist = soft_dtw_gradient(series1, series2, gamma=1.0)
    cost_matrix = cost_matrix(series1, series2, method="dtw")
    print("Distance:", dist)  # noqa: T201
    stop = ""
