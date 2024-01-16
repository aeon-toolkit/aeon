import numpy as np

from aeon.utils.validation._dependencies import _check_soft_dependencies

__all__ = [
    "plot_windows",
]


def plot_windows(cv, y, title=""):
    """Plot training and test windows.

    Parameters
    ----------
    y : pd.Series
        Time series to split
    cv : temporal cross-validation iterator object
        Temporal cross-validation iterator
    title : str
        Plot title
    """
    _check_soft_dependencies("matplotlib", "seaborn")

    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    train_windows = []
    test_windows = []
    for train, test in cv.split(y):
        train_windows.append(train)
        test_windows.append(test)

    train_color, test_color = sns.color_palette("colorblind")[:2]

    fig, ax = plt.subplots(figsize=plt.figaspect(0.3))

    for i in range(len(train_windows)):
        train = train_windows[i]
        test = test_windows[i]

        ax.plot(np.arange(len(y)), np.ones(len(y)) * i, marker="o", c="lightgray")
        ax.plot(
            train,
            np.ones(len(train)) * i,
            marker="o",
            c=train_color,
            label="Window",
        )
        ax.plot(
            test,
            np.ones(len(test)) * i,
            marker="o",
            c=test_color,
            label="Forecasting horizon",
        )
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(
        title=title,
        ylabel="Window number",
        xlabel="Time",
        xticklabels=y.index,
    )
    # remove duplicate labels/handles
    handles, labels = [(leg[:2]) for leg in ax.get_legend_handles_labels()]
    ax.legend(handles, labels)
