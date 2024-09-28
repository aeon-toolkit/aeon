# %%NBQA-CELL-SEPfc780c
import numpy as np

classifiers = ["Classifier 1", "Classifier 2", "Classifier 3", "Classifier 4"]
classifier_accuracies = [
    [0.8, 0.7, 0.6, 0.5],
    [0.7, 0.9, 0.4, 0.0],
    [0.8, 0.7, 0.6, 0.5],
    [0.7, 0.9, 0.4, 0.0],
    [0.7, 0.6, 0.5, 0.4],
]
regressor_preds = [0.8, 0.7, 0.6, 0.5]
regressor_targets = [0.9, 0.7, 0.4, 0.0]


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_critical_difference


# %%NBQA-CELL-SEPfc780c
_ = plot_critical_difference(classifier_accuracies, classifiers)


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import (
    plot_pairwise_scatter,
    plot_scatter_predictions,
    plot_score_vs_time_scatter,
)


# %%NBQA-CELL-SEPfc780c
_ = plot_pairwise_scatter(
    classifier_accuracies[0], classifier_accuracies[1], classifiers[0], classifiers[1]
)


# %%NBQA-CELL-SEPfc780c
_ = plot_scatter_predictions(regressor_targets, regressor_preds, title="Regressor 1")


# %%NBQA-CELL-SEPfc780c
_ = plot_score_vs_time_scatter(
    np.mean(classifier_accuracies, axis=0),
    [9000, 4000, 1500, 100],
    names=classifiers,
    title="Score vs Time",
    log_time=True,
)


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_boxplot


# %%NBQA-CELL-SEPfc780c
_ = plot_boxplot(
    classifier_accuracies,
    classifiers,
    relative=True,
    plot_type="boxplot",
    title="Boxplot",
)
