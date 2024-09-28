# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking import get_available_estimators

cls = get_available_estimators(task="classification")
print(len(cls), " classifier results available\n", cls)


# %%NBQA-CELL-SEPfc780c
reg = get_available_estimators(task="regression")
print(len(reg), " regressor results available\n", reg)


# %%NBQA-CELL-SEPfc780c
clst = get_available_estimators(task="clustering", return_dataframe=False)
print(len(clst), " clustering results available\n", clst)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking.results_loaders import (
    get_estimator_results,
    get_estimator_results_as_array,
)
from aeon.visualisation import (
    plot_boxplot,
    plot_critical_difference,
    plot_pairwise_scatter,
)

classifiers = [
    "FreshPRINCEClassifier",
    "HIVECOTEV2",
    "InceptionTimeClassifier",
    "WEASEL-Dilation",
]
datasets = ["ACSF1", "ArrowHead", "GunPoint", "ItalyPowerDemand"]
# get results. To read locally, set the path variable.
# If you do not set path, results are loaded from
# https://timeseriesclassification.com/results/ReferenceResults.
# You can download the files directly from there
default_split_all, data_names = get_estimator_results_as_array(estimators=classifiers)
print(
    " Returns an array with each column an estimator, shape (data_names, classifiers)"
)
print(
    f"By default recovers the default test split results for {len(data_names)} "
    f"equal length UCR datasets."
)
default_split_some, names = get_estimator_results_as_array(
    estimators=classifiers, datasets=datasets
)
print(
    f"Or specify datasets for result recovery. For example, {len(names)} datasets. "
    f"HIVECOTEV2 accuracy {names[3]} = {default_split_some[3][1]}"
)


# %%NBQA-CELL-SEPfc780c
hash_table = get_estimator_results(estimators=classifiers)
print("Keys = ", hash_table.keys())
print(
    "Accuracy of HIVECOTEV2 on ItalyPowerDemand = ",
    hash_table["HIVECOTEV2"]["ItalyPowerDemand"],
)


# %%NBQA-CELL-SEPfc780c
resamples_all, data_names = get_estimator_results_as_array(
    estimators=classifiers, default_only=False
)
print("Results are averaged over 30 stratified resamples.")
print(
    f" HIVECOTEV2 default train test partition of {data_names[3]} = "
    f"{default_split_all[3][1]} and averaged over 30 resamples = "
    f"{resamples_all[3][1]}"
)


# %%NBQA-CELL-SEPfc780c
plot = plot_critical_difference(
    default_split_all, classifiers, test="wilcoxon", correction="holm"
)


# %%NBQA-CELL-SEPfc780c
plot = plot_critical_difference(
    resamples_all, classifiers, test="wilcoxon", correction="holm"
)


# %%NBQA-CELL-SEPfc780c
plot = plot_critical_difference(
    resamples_all,
    classifiers,
    test="wilcoxon",
    correction="holm",
    highlight={"HIVECOTEV2": "#8a9bf8"},
)


# %%NBQA-CELL-SEPfc780c
plot = plot_boxplot(
    resamples_all,
    classifiers,
    relative=True,
    plot_type="boxplot",
    outliers=True,
)


# %%NBQA-CELL-SEPfc780c
plot = plot_boxplot(
    resamples_all,
    classifiers,
    relative=True,
    plot_type="boxplot",
    outliers=True,
    y_min=0.4,
    y_max=0.6,
)


# %%NBQA-CELL-SEPfc780c
plot = plot_boxplot(
    resamples_all,
    classifiers,
    relative=True,
    plot_type="violin",
    title="Violin plot",
)


# %%NBQA-CELL-SEPfc780c
methods = ["InceptionTimeClassifier", "WEASEL-Dilation"]

results, datasets = get_estimator_results_as_array(estimators=methods)
results = results.T

fig, ax = plot_pairwise_scatter(
    results[0],
    results[1],
    methods[0],
    methods[1],
    title="Comparison of IT and WEASEL2",
)
fig.show()
