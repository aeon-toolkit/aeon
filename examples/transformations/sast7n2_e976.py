# %%NBQA-CELL-SEPfc780c
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from aeon.classification.shapelet_based import SASTClassifier
from aeon.datasets import load_classification
from aeon.transformations.collection.shapelet_based import SAST


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_classification("UnitTest", split="train")


# %%NBQA-CELL-SEPfc780c
sast = SAST()
sast.fit(X_train, y_train)
X_train_transform = sast.transform(X_train)


# %%NBQA-CELL-SEPfc780c
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transform, y_train)


# %%NBQA-CELL-SEPfc780c
X_test, y_test = load_classification("UnitTest", split="test")
X_test_transform = sast.transform(X_test)


# %%NBQA-CELL-SEPfc780c
classifier.score(X_test_transform, y_test)


# %%NBQA-CELL-SEPfc780c
sast_pipeline = make_pipeline(SAST(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_classification("UnitTest", split="train")

# it is necessary to pass y_train to the pipeline
# y_train is not used for the transform, but it is used by the classifier
sast_pipeline.fit(X_train, y_train)


# %%NBQA-CELL-SEPfc780c
X_test, y_test = load_classification("UnitTest", split="test")

sast_pipeline.score(X_test, y_test)


# %%NBQA-CELL-SEPfc780c
clf = SASTClassifier(seed=42)
clf


# %%NBQA-CELL-SEPfc780c
clf.fit(X_train, y_train)


# %%NBQA-CELL-SEPfc780c
clf.score(X_test, y_test)


# %%NBQA-CELL-SEPfc780c
fig = clf.plot_most_important_feature_on_ts(
    X_test[y_test == "1"][0, 0], clf._classifier.coef_[0]
)


# %%NBQA-CELL-SEPfc780c
fig = clf.plot_most_important_feature_on_ts(
    X_test[y_test == "2"][0, 0], clf._classifier.coef_[0]
)
