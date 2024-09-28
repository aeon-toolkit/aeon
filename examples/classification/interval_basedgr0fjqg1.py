# %%NBQA-CELL-SEPfc780c
import warnings

from sklearn import metrics

from aeon.classification.interval_based import (
    CanonicalIntervalForestClassifier,
    DrCIFClassifier,
    RandomIntervalSpectralEnsembleClassifier,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)
from aeon.datasets import load_basic_motions, load_italy_power_demand
from aeon.registry import all_estimators

warnings.filterwarnings("ignore")
all_estimators("classifier", filter_tags={"algorithm_type": "interval"})


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_italy_power_demand(split="train", return_X_y=True)
X_test, y_test = load_italy_power_demand(split="test", return_X_y=True)
X_test = X_test[:50]
y_test = y_test[:50]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train_mv, y_train_mv = load_basic_motions(split="train", return_X_y=True)
X_test_mv, y_test_mv = load_basic_motions(split="test", return_X_y=True)

X_train_mv = X_train_mv[:50]
y_train_mv = y_train_mv[:50]
X_test_mv = X_test_mv[:50]
y_test_mv = y_test_mv[:50]

print(X_train_mv.shape, y_train_mv.shape, X_test_mv.shape, y_test_mv.shape)


# %%NBQA-CELL-SEPfc780c
tsf = TimeSeriesForestClassifier(n_estimators=50, random_state=47)
tsf.fit(X_train, y_train)

tsf_preds = tsf.predict(X_test)
print("TSF Accuracy: " + str(metrics.accuracy_score(y_test, tsf_preds)))


# %%NBQA-CELL-SEPfc780c
rise = RandomIntervalSpectralEnsembleClassifier(n_estimators=50, random_state=47)
rise.fit(X_train, y_train)

rise_preds = rise.predict(X_test)
print("RISE Accuracy: " + str(metrics.accuracy_score(y_test, rise_preds)))


# %%NBQA-CELL-SEPfc780c
stsf = SupervisedTimeSeriesForest(n_estimators=50, random_state=47)
stsf.fit(X_train, y_train)

stsf_preds = stsf.predict(X_test)
print("STSF Accuracy: " + str(metrics.accuracy_score(y_test, stsf_preds)))


# %%NBQA-CELL-SEPfc780c
cif = CanonicalIntervalForestClassifier(
    n_estimators=50, att_subsample_size=8, random_state=47
)
cif.fit(X_train, y_train)

cif_preds = cif.predict(X_test)
print("CIF Accuracy: " + str(metrics.accuracy_score(y_test, cif_preds)))


# %%NBQA-CELL-SEPfc780c
cif_m = CanonicalIntervalForestClassifier(
    n_estimators=50, att_subsample_size=8, random_state=47
)
cif_m.fit(X_train_mv, y_train_mv)

cif_m_preds = cif_m.predict(X_test_mv)
print("CIF Accuracy: " + str(metrics.accuracy_score(y_test_mv, cif_m_preds)))


# %%NBQA-CELL-SEPfc780c
drcif = DrCIFClassifier(n_estimators=5, att_subsample_size=10, random_state=47)
drcif.fit(X_train, y_train)

drcif_preds = drcif.predict(X_test)
print("DrCIF Accuracy: " + str(metrics.accuracy_score(y_test, drcif_preds)))


# %%NBQA-CELL-SEPfc780c
drcif_m = DrCIFClassifier(n_estimators=5, att_subsample_size=10, random_state=47)
drcif_m.fit(X_train_mv, y_train_mv)

drcif_m_preds = drcif_m.predict(X_test_mv)
print("DrCIF Accuracy: " + str(metrics.accuracy_score(y_test_mv, drcif_m_preds)))


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

est = all_estimators("classifier", filter_tags={"algorithm_type": "interval"})
for c in est:
    print(c)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate

names = [t[0].replace("Classifier", "") for t in est]
names.remove("IntervalForest")  # Base class
names.remove("RandomInterval")  # Pipeline
names.remove("SupervisedInterval")  # Pipeline
results, present_names = get_estimator_results_as_array(
    names, univariate, include_missing=False
)
results.shape


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_boxplot, plot_critical_difference

plot_critical_difference(results, names)


# %%NBQA-CELL-SEPfc780c
plot_boxplot(results, names, relative=True)
