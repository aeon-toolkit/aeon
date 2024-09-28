# %%NBQA-CELL-SEPfc780c
# !pip install --upgrade numba


# %%NBQA-CELL-SEPfc780c
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aeon.datasets import load_arrow_head  # univariate dataset
from aeon.datasets import load_basic_motions  # multivariate dataset
from aeon.datasets import (
    load_japanese_vowels,  # multivariate dataset with unequal length
)
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariateVariable,
)


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_arrow_head(split="train")
minirocket = MiniRocket()  # by default, MiniRocket uses ~10_000 kernels
minirocket.fit(X_train)
X_train_transform = minirocket.transform(X_train)
# test shape of transformed training data -> (n_cases, 9_996)
X_train_transform.shape


# %%NBQA-CELL-SEPfc780c
scaler = StandardScaler(with_mean=False)
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

X_train_scaled_transform = scaler.fit_transform(X_train_transform)
classifier.fit(X_train_scaled_transform, y_train)


# %%NBQA-CELL-SEPfc780c
X_test, y_test = load_arrow_head(split="test")
X_test_transform = minirocket.transform(X_test)


# %%NBQA-CELL-SEPfc780c
X_test_scaled_transform = scaler.transform(X_test_transform)
classifier.score(X_test_scaled_transform, y_test)


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_basic_motions(split="train")


# %%NBQA-CELL-SEPfc780c
mr = MiniRocket()
mr.fit(X_train)
X_train_transform = mr.transform(X_train)


# %%NBQA-CELL-SEPfc780c
scaler = StandardScaler(with_mean=False)
X_train_scaled_transform = scaler.fit_transform(X_train_transform)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_scaled_transform, y_train)


# %%NBQA-CELL-SEPfc780c
X_test, y_test = load_basic_motions(split="test")
X_test_transform = mr.transform(X_test)


# %%NBQA-CELL-SEPfc780c
X_test_scaled_transform = scaler.transform(X_test_transform)
classifier.score(X_test_scaled_transform, y_test)


# %%NBQA-CELL-SEPfc780c
minirocket_pipeline = make_pipeline(
    MiniRocket(),
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
)


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_arrow_head(split="train")

# it is necessary to pass y_train to the pipeline
# y_train is not used for the transform, but it is used by the classifier
minirocket_pipeline.fit(X_train, y_train)


# %%NBQA-CELL-SEPfc780c
X_test, y_test = load_arrow_head(split="test")

minirocket_pipeline.score(X_test, y_test)


# %%NBQA-CELL-SEPfc780c
X_train_jv, y_train_jv = load_japanese_vowels(split="train")
# lets visualize the first three voice recordings with dimension 0-11

print("number of samples training: ", len(X_train_jv))
print("series length of recoding 0, dimension 5: ", X_train_jv[0][5].shape)
print("series length of recoding 1, dimension 0: ", X_train_jv[1][0].shape)


# %%NBQA-CELL-SEPfc780c
minirocket_mv_var_pipeline = make_pipeline(
    MiniRocketMultivariateVariable(
        pad_value_short_series=-10.0, random_state=42, max_dilations_per_kernel=16
    ),
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
)
print(minirocket_mv_var_pipeline)

minirocket_mv_var_pipeline.fit(X_train_jv, y_train_jv)


# %%NBQA-CELL-SEPfc780c
X_test_jv, y_test_jv = load_japanese_vowels(split="test")

minirocket_mv_var_pipeline.score(X_test_jv, y_test_jv)
