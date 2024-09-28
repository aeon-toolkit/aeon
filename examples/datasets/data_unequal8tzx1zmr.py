# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

all_estimators(
    estimator_types="classifier", filter_tags={"capability:unequal_length": True}
)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_japanese_vowels, load_plaid

j_vowels, j_labels = load_japanese_vowels()
p_vowels, p_labels = load_plaid()
print(type(j_vowels[0].shape), "  ", type(p_vowels[0].shape))
print("shape first =", j_vowels[0].shape, "shape 11th =", j_vowels[10].shape)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets.tsc_datasets import (
    multivariate_unequal_length,
    univariate_variable_length,
)

print(univariate_variable_length)
print(multivariate_unequal_length)


# %%NBQA-CELL-SEPfc780c
from aeon.transformations.collection import PaddingTransformer, TruncationTransformer

padder = PaddingTransformer()
truncator = TruncationTransformer()
padded_j_vowels = padder.fit_transform(j_vowels)
truncated_j_vowels = truncator.fit_transform(j_vowels)
print(padded_j_vowels.shape, truncated_j_vowels.shape)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_classification

j_equal, _ = load_classification("JapaneseVowels")
j_unequal, _ = load_classification("JapaneseVowels", load_equal_length=False)
print(type(j_equal))
print(j_equal.shape)
print(type(j_unequal))
