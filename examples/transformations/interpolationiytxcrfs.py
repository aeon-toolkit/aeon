# %%NBQA-CELL-SEPfc780c
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_basic_motions


# %%NBQA-CELL-SEPfc780c
X, y = load_basic_motions()
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = RocketClassifier(num_kernels=200)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_plaid

X, y = load_plaid()
X_train, X_test, y_train, y_test = train_test_split(X, y)

try:
    clf = RocketClassifier(num_kernels=200)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
except ValueError as e:
    print(f"ValueError: {e}")


# %%NBQA-CELL-SEPfc780c
from aeon.transformations.collection.interpolate import TSInterpolator

steps = [
    ("transform", TSInterpolator(50)),
    ("classify", RocketClassifier()),
]
clf = Pipeline(steps)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
