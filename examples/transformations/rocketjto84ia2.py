# %%NBQA-CELL-SEPfc780c
# !pip install --upgrade numba


# %%NBQA-CELL-SEPfc780c
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from aeon.datasets import load_arrow_head  # univariate dataset
from aeon.datasets import load_basic_motions  # multivariate dataset
from aeon.transformations.collection.convolution_based import Rocket


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_arrow_head(split="train")


# %%NBQA-CELL-SEPfc780c
rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
rocket.fit(X_train)
X_train_transform = rocket.transform(X_train)


# %%NBQA-CELL-SEPfc780c
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transform, y_train)


# %%NBQA-CELL-SEPfc780c
X_test, y_test = load_arrow_head(split="test")
X_test_transform = rocket.transform(X_test)


# %%NBQA-CELL-SEPfc780c
classifier.score(X_test_transform, y_test)


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_basic_motions(split="train")


# %%NBQA-CELL-SEPfc780c
rocket = Rocket()
rocket.fit(X_train)
X_train_transform = rocket.transform(X_train)


# %%NBQA-CELL-SEPfc780c
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transform, y_train)


# %%NBQA-CELL-SEPfc780c
X_test, y_test = load_basic_motions(split="test")
X_test_transform = rocket.transform(X_test)


# %%NBQA-CELL-SEPfc780c
classifier.score(X_test_transform, y_test)


# %%NBQA-CELL-SEPfc780c
rocket_pipeline = make_pipeline(
    Rocket(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
)


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_arrow_head(split="train")

# it is necessary to pass y_train to the pipeline
# y_train is not used for the transform, but it is used by the classifier
rocket_pipeline.fit(X_train, y_train)


# %%NBQA-CELL-SEPfc780c
X_test, y_test = load_arrow_head(split="test", return_X_y=True)

rocket_pipeline.score(X_test, y_test)
