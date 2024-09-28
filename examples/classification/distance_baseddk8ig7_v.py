# %%NBQA-CELL-SEPfc780c
from sklearn import metrics

from aeon.datasets import load_italy_power_demand
from aeon.registry import all_estimators

all_estimators("classifier", filter_tags={"algorithm_type": "distance"})


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# %%NBQA-CELL-SEPfc780c
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(8, 6))
axs[1, 1].axis("off")
axs[1, 0].axis("off")
ax_combined = fig.add_subplot(2, 1, (2, 3))
axs[0, 0].set_title("All days class 1")
axs[0, 1].set_title("All days class 2")
ax_combined.set_title("Both classes on top of each other")
for i in np.where(y_test == "1")[0]:
    axs[0, 0].plot(X_test[i][0], alpha=0.1, color="cornflowerblue", linestyle="solid")
    ax_combined.plot(X_test[i][0], alpha=0.1, color="cornflowerblue", linestyle="--")
for i in np.where(y_test == "2")[0]:
    axs[0, 1].plot(X_test[i][0], alpha=0.1, color="orange", linestyle="solid")
    ax_combined.plot(X_test[i][0], alpha=0.1, color="orange", linestyle=":")


# %%NBQA-CELL-SEPfc780c
from aeon.classification.distance_based import (
    ElasticEnsemble,
    KNeighborsTimeSeriesClassifier,
)


# %%NBQA-CELL-SEPfc780c
knn = KNeighborsTimeSeriesClassifier(distance="msm", n_neighbors=3, weights="distance")
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)
metrics.accuracy_score(y_test, knn_preds)


# %%NBQA-CELL-SEPfc780c
ee = ElasticEnsemble(
    distance_measures=["dtw", "msm"],
    proportion_of_param_options=0.1,
    proportion_train_in_param_finding=0.3,
    proportion_train_for_test=0.5,
)
ee.fit(X_train, y_train)
ee_preds = ee.predict(X_test)
metrics.accuracy_score(y_test, ee_preds)


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

est = ["ElasticEnsemble", "KNeighborsTimeSeriesClassifier"]
for c in est:
    print(c)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate

names = [t.replace("Classifier", "") for t in est]
names.append(
    "PF"
)  # Results from Java implementation, as are the ElasticEnsemble results

results, present_names = get_estimator_results_as_array(
    names, univariate, include_missing=False
)
results.shape


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_boxplot, plot_critical_difference

plot_critical_difference(results, names)


# %%NBQA-CELL-SEPfc780c
plot_boxplot(results, names, relative=True)
