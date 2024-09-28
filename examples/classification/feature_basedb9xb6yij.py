# %%NBQA-CELL-SEPfc780c
import warnings

from sklearn import metrics

from aeon.classification.feature_based import Catch22Classifier, FreshPRINCEClassifier
from aeon.datasets import load_basic_motions, load_italy_power_demand
from aeon.registry import all_estimators
from aeon.transformations.collection.feature_based import Catch22

warnings.filterwarnings("ignore")

X_train, y_train = load_italy_power_demand(split="train")
X_test, y_test = load_italy_power_demand(split="test")
X_test = X_test[:50]
y_test = y_test[:50]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train_mv, y_train_mv = load_basic_motions(split="train")
X_test_mv, y_test_mv = load_basic_motions(split="test")

X_train_mv = X_train_mv[:20]
y_train_mv = y_train_mv[:20]
X_test_mv = X_test_mv[:20]
y_test_mv = y_test_mv[:20]

print(X_train_mv.shape, y_train_mv.shape, X_test_mv.shape, y_test_mv.shape)
all_estimators("classifier", filter_tags={"algorithm_type": "feature"})


# %%NBQA-CELL-SEPfc780c
c22 = Catch22()
x_trans = c22.fit_transform(X_train)
x_trans.shape


# %%NBQA-CELL-SEPfc780c
c22cls = Catch22Classifier()
c22cls.fit(X_train, y_train)
c22_preds = c22cls.predict(X_test)
metrics.accuracy_score(y_test, c22_preds)


# %%NBQA-CELL-SEPfc780c
fp = FreshPRINCEClassifier()
fp.fit(X_train, y_train)
fp_preds = c22cls.predict(X_test)
metrics.accuracy_score(y_test, fp_preds)


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

est = all_estimators("classifier", filter_tags={"algorithm_type": "feature"})
for c in est:
    print(c)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate

names = [t[0].replace("Classifier", "") for t in est]
names.remove("Summary")  # Need to evaluate this
results, present_names = get_estimator_results_as_array(
    names, univariate, include_missing=False
)
results.shape


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_boxplot, plot_critical_difference

plot_critical_difference(results, names)


# %%NBQA-CELL-SEPfc780c
plot_boxplot(results, names, relative=True)
