# %%NBQA-CELL-SEPfc780c
import numpy as np

from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.datasets import load_japanese_vowels, load_plaid
from aeon.registry import all_estimators
from aeon.utils.validation import has_missing, is_equal_length, is_univariate


# %%NBQA-CELL-SEPfc780c
X = np.random.random(size=(10, 2, 200))
has_missing(X)


# %%NBQA-CELL-SEPfc780c
X[5][0][55] = np.NAN
has_missing(X)


# %%NBQA-CELL-SEPfc780c
all_estimators(
    estimator_types=["classifier", "regressor", "clusterer", "forecaster"],
    filter_tags={"capability:missing_values": True},
    as_dataframe=True,
)


# %%NBQA-CELL-SEPfc780c
plaid_X, plaid_y = load_plaid(split="train")
print(
    f"PLAID is univariate = {is_univariate(plaid_X)} has missing ="
    f"{has_missing(plaid_X)} is equal length = {is_equal_length(plaid_X)}"
)
vowels_X, vowels_y = load_japanese_vowels(split="train")
print(
    f"JapaneseVowels is univariate = {is_univariate(vowels_X)} "
    f"has missing = {has_missing(vowels_X)} is "
    f"equal length = {is_equal_length(vowels_X)}"
)


# %%NBQA-CELL-SEPfc780c
all_estimators(
    estimator_types=["classifier", "regressor", "clusterer"],
    filter_tags={"capability:unequal_length": True},
    as_dataframe=True,
)


# %%NBQA-CELL-SEPfc780c
knn = KNeighborsTimeSeriesClassifier()
knn.fit(plaid_X, plaid_y)


# %%NBQA-CELL-SEPfc780c
from aeon.transformations.collection import PaddingTransformer

pt = PaddingTransformer()
plaid_equal = pt.fit_transform(plaid_X)
plaid_equal.shape


# %%NBQA-CELL-SEPfc780c
import matplotlib.pyplot as plt

plt.title("Before and after padding: PLAID first case (shifted up for unpadded)")
plt.plot(plaid_X[0][0] + 10)
plt.plot(plaid_equal[0][0])
