# %%NBQA-CELL-SEPfc780c
import warnings

from aeon.registry import all_estimators

warnings.filterwarnings("ignore")
all_estimators(
    "classifier", filter_tags={"algorithm_type": "convolution"}, as_dataframe=True
)


# %%NBQA-CELL-SEPfc780c
from sklearn.metrics import accuracy_score

from aeon.classification.convolution_based import Arsenal, RocketClassifier
from aeon.datasets import load_basic_motions  # multivariate dataset
from aeon.datasets import load_italy_power_demand  # univariate dataset

italy, italy_labels = load_italy_power_demand(split="train")
italy_test, italy_test_labels = load_italy_power_demand(split="test")
motions, motions_labels = load_basic_motions(split="train")
motions_test, motions_test_labels = load_basic_motions(split="test")
italy.shape


# %%NBQA-CELL-SEPfc780c
rocket = RocketClassifier()
rocket.fit(italy, italy_labels)
y_pred = rocket.predict(italy_test)
accuracy_score(italy_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
afc = Arsenal()
afc.fit(italy, italy_labels)
y_pred = afc.predict(italy_test)
accuracy_score(italy_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
multi_r = Arsenal(rocket_transform="multirocket")
multi_r.fit(italy, italy_labels)
y_pred = multi_r.predict(italy_test)
accuracy_score(italy_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
mini_r = RocketClassifier(rocket_transform="minirocket")
mini_r.fit(motions, motions_labels)
y_pred = mini_r.predict(motions_test)
accuracy_score(motions_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

est = all_estimators("classifier", filter_tags={"algorithm_type": "convolution"})
for c in est:
    print(c)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate

names = [t[0].replace("Classifier", "") for t in est]
names.append("MiniROCKET")  # Alternatve configuration of the RocketClassifier
results, present_names = get_estimator_results_as_array(
    names, univariate, include_missing=False
)
results.shape


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_boxplot, plot_critical_difference

plot_critical_difference(results, names)


# %%NBQA-CELL-SEPfc780c
plot_boxplot(results, names, relative=True)
