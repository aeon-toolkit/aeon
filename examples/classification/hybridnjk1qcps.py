# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_basic_motions, load_italy_power_demand

X_train, y_train = load_italy_power_demand(split="train")
X_test, y_test = load_italy_power_demand(split="test")
X_test = X_test[:50]
y_test = y_test[:50]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train_mv, y_train_mv = load_basic_motions(split="train")
X_test_mv, y_test_mv = load_basic_motions(split="test")

print(X_train_mv.shape, y_train_mv.shape, X_test_mv.shape, y_test_mv.shape)


# %%NBQA-CELL-SEPfc780c
from sklearn.metrics import accuracy_score

from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2

hc1 = HIVECOTEV1()
hc2 = HIVECOTEV2(time_limit_in_minutes=0.2)
hc2.fit(X_train, y_train)
y_pred = hc2.predict(X_test)
accuracy_score(y_test, y_pred)


# %%NBQA-CELL-SEPfc780c
hc2.fit(X_train_mv, y_train_mv)
y_pred = hc2.predict(X_test_mv)

accuracy_score(y_test_mv, y_pred)


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

est = all_estimators("classifier", filter_tags={"algorithm_type": "hybrid"})
for c in est:
    print(c)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate

names = [t[0] for t in est]
names.append("TS-CHIEF")  # TS-Chief is not available in aeon, results from a
# Java implementation

results, present_names = get_estimator_results_as_array(
    names, univariate, include_missing=False
)
results.shape


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_boxplot, plot_critical_difference

plot_critical_difference(results, names)


# %%NBQA-CELL-SEPfc780c
plot_boxplot(results, names, relative=True)
