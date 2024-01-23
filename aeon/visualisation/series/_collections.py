from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.utils.validation.collection import convert_collection


def plot_series_collection(X):
    """Plot series collection."""
    _check_soft_dependencies("matplotlib")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    X = convert_collection(X, "numpy3D")
    plt.figure(figsize=(5, 10))
    plt.rcParams["figure.dpi"] = 100

    fig, axes = plt.subplots(nrows=len(X), ncols=1)
    for i in range(len(X)):
        curr = X[i][0]
        curr_axes = axes[i]
        curr_axes.plot(curr, color="b")

    blue_patch = mpatches.Patch(color="blue", label="Series that belong to the cluster")
    plt.legend(
        handles=[blue_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.40),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    plt.tight_layout()
    plt.show()
