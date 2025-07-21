"""Test for SPARTAN transformer."""

from scipy.sparse import csr_matrix

from aeon.datasets import load_unit_test
from aeon.transformations.collection.dictionary_based import SPARTAN, SFAFast


def test_spartan_dict():
    X_train, y_train = load_unit_test(split="train")

    # Fit, then transform
    spartan = SPARTAN(
        dilation=2,
        window_size=10,
        return_sparse=True
    )
    spartan.fit(X_train)

    x = spartan.fit_transform(X_train)
    x2 = spartan.transform(X_train)

    assert isinstance(x, csr_matrix)
    assert isinstance(x2, csr_matrix)


def test_sfa_dict():
    X_train, y_train = load_unit_test(split="train")

    # Fit, then transform
    sfa = SFAFast()
    sfa.fit(X_train)

    x = sfa.fit_transform(X_train)

    print(x)
