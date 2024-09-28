# %%NBQA-CELL-SEPfc780c
# Plotting and data loading imports used in this notebook
import warnings

import matplotlib.pyplot as plt

from aeon.datasets import load_arrow_head, load_basic_motions

warnings.filterwarnings("ignore")

arrow, arrow_labels = load_arrow_head(split="train")
motions, motions_labels = load_basic_motions(split="train")
print(f"ArrowHead series of type {type(arrow)} and shape {arrow.shape}")
print(f"Motions type {type(motions)} of shape {motions_labels.shape}")


# %%NBQA-CELL-SEPfc780c
plt.title(
    f"First and second dimensions of the first instance in BasicMotions data, "
    f"(student {motions_labels[0]})"
)
plt.plot(motions[0][0])
plt.plot(motions[0][1])


# %%NBQA-CELL-SEPfc780c
plt.title(f"First instance in ArrowHead data (class {arrow_labels[0]})")
plt.plot(arrow[0, 0])


# %%NBQA-CELL-SEPfc780c
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rand_forest = RandomForestClassifier(n_estimators=100)
arrow2d = arrow.squeeze()
arrow_test, arrow_test_labels = load_arrow_head(split="test", return_type="numpy2d")
rand_forest.fit(arrow2d, arrow_labels)
y_pred = rand_forest.predict(arrow_test)
accuracy_score(arrow_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
from aeon.classification.convolution_based import RocketClassifier

rocket = RocketClassifier(num_kernels=2000)
rocket.fit(arrow, arrow_labels)
y_pred = rocket.predict(arrow_test)

accuracy_score(arrow_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
from aeon.classification.hybrid import HIVECOTEV2

hc2 = HIVECOTEV2(time_limit_in_minutes=0.2)
hc2.fit(arrow, arrow_labels)
y_pred = hc2.predict(arrow_test)

accuracy_score(arrow_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
motions_test, motions_test_labels = load_basic_motions(split="test")
motions2d = motions.reshape(motions.shape[0], motions.shape[1] * motions.shape[2])
motions2d_test = motions_test.reshape(
    motions_test.shape[0], motions_test.shape[1] * motions_test.shape[2]
)
rand_forest.fit(motions2d, motions_labels)
y_pred = rand_forest.predict(motions2d_test)
accuracy_score(motions_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
rocket.fit(motions, motions_labels)
y_pred = rocket.predict(motions_test)
accuracy_score(motions_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

all_estimators(
    filter_tags={"capability:multivariate": True},
    estimator_types="classifier",
    as_dataframe=True,
)


# %%NBQA-CELL-SEPfc780c
from aeon.classification.compose import ChannelEnsembleClassifier
from aeon.classification.interval_based import DrCIFClassifier

cls = ChannelEnsembleClassifier(
    estimators=[
        ("DrCIF0", DrCIFClassifier(n_estimators=5, n_intervals=2), [0]),
        ("ROCKET3", RocketClassifier(num_kernels=1000), [3, 4]),
    ]
)

cls.fit(motions, motions_labels)
y_pred = cls.predict(motions_test)

accuracy_score(motions_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
from sklearn.model_selection import KFold, cross_val_score

cross_val_score(rocket, arrow, y=arrow_labels, cv=KFold(n_splits=4))


# %%NBQA-CELL-SEPfc780c
from sklearn.model_selection import GridSearchCV

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

knn = KNeighborsTimeSeriesClassifier()
param_grid = {"n_neighbors": [1, 5], "distance": ["euclidean", "dtw"]}
parameter_tuning_method = GridSearchCV(knn, param_grid, cv=KFold(n_splits=4))

parameter_tuning_method.fit(arrow, arrow_labels)
y_pred = parameter_tuning_method.predict(arrow_test)

accuracy_score(arrow_test_labels, y_pred)


# %%NBQA-CELL-SEPfc780c
from sklearn.calibration import CalibratedClassifierCV

from aeon.classification.interval_based import DrCIFClassifier

calibrated_drcif = CalibratedClassifierCV(
    estimator=DrCIFClassifier(n_estimators=10, n_intervals=5), cv=4
)

calibrated_drcif.fit(arrow, arrow_labels)
y_pred = calibrated_drcif.predict(arrow_test)

accuracy_score(arrow_test_labels, y_pred)
